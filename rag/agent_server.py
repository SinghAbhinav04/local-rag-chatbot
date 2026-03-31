"""
Agent Web Server — serves the PicoClaw-inspired web UI and REST API.
Runs in a background thread so the CLI keeps working alongside.

Start:   /agent start   (from the chat loop)
Stop:    /agent stop
"""

import os
import re
import json
import threading
import webbrowser
import datetime
import uuid
import tempfile
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import ollama

from rag.config import (
    get_available_models, EMBED_MODEL, TOP_K, CHUNK_SIZE, CHUNK_OVERLAP,
    DOC_FOLDER, SCRAPED_DOCS_FOLDER, REDDIT_JSON_FOLDER,
    SCRAPED_DATA_FOLDER, NEWS_DATA_FOLDER, WIKI_DATA_FOLDER,
    MIN_MODEL_SIZE_MB,
)
from rag.console import console
from rag.vectordb import retrieve_raw, build_vector_db, add_doc_to_db, get_embedding
from rag.loaders import load_file
from rag.chunking import chunk_text
from rag import agent_memory, agent_skills
from rag.agent_tools import create_default_registry
from rag import cloud_providers

# ── Global state shared with CLI ──────────────────────────────────────────────
_tool_registry = create_default_registry()
_state = {
    "collection": None,
    "model": "",
    "messages": [],
    "doc_chunk_counts": {},
    "selected_paths": [],
    "chunk_offset": 0,
    "session_start": None,
    "voice_enabled": False,
    "tool_calling": True,
}
_server: HTTPServer | None = None
_thread: threading.Thread | None = None

PORT = 18800
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
UPLOAD_DIR = os.path.expanduser("~/.rag-agent/uploads")


def _json_response(handler, data, status=200):
    """Send a JSON response."""
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def _read_body(handler) -> dict:
    """Read JSON body from request."""
    length = int(handler.headers.get("Content-Length", 0))
    if length == 0:
        return {}
    raw = handler.rfile.read(length)
    return json.loads(raw)


class AgentHandler(SimpleHTTPRequestHandler):
    """Handles both static files and API endpoints."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=STATIC_DIR, **kwargs)

    def log_message(self, format, *args):
        """Suppress noisy access logs."""
        pass

    # ── Routing ───────────────────────────────────────────────────────────────
    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/status":
            self._handle_status()
        elif path == "/api/models":
            self._handle_models()
        elif path == "/api/docs":
            self._handle_docs()
        elif path == "/api/history":
            self._handle_history()
        elif path == "/api/config":
            self._handle_config()
        elif path == "/api/tools":
            self._handle_tools_list()
        elif path == "/api/memory":
            self._handle_memory_read()
        elif path == "/api/memory/today":
            self._handle_memory_today_read()
        elif path == "/api/skills":
            self._handle_skills_list()
        elif path == "/api/cron":
            self._handle_cron_list()
        elif path == "/api/cloud-config":
            self._handle_cloud_config_get()
        elif path == "/api/cloud-providers":
            self._handle_cloud_providers_list()
        else:
            # Serve static files; for SPA, serve index.html for unknown paths
            if path != "/" and not path.startswith("/api") and not os.path.exists(os.path.join(STATIC_DIR, path.lstrip("/"))):
                self.path = "/"
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/chat":
            self._handle_chat()
        elif path == "/api/models/select":
            self._handle_model_select()
        elif path == "/api/docs/add-url":
            self._handle_add_url()
        elif path == "/api/docs/remove":
            self._handle_remove_doc()
        elif path == "/api/clear":
            self._handle_clear()
        elif path == "/api/wiki":
            self._handle_wiki()
        elif path == "/api/weather":
            self._handle_weather()
        elif path == "/api/news":
            self._handle_news()
        elif path == "/api/tools/execute":
            self._handle_tool_execute()
        elif path == "/api/memory":
            self._handle_memory_write()
        elif path == "/api/memory/today":
            self._handle_memory_today_write()
        elif path == "/api/tools/toggle":
            self._handle_tool_toggle()
        elif path == "/api/cron":
            self._handle_cron_action()
        elif path == "/api/cloud-config":
            self._handle_cloud_config_save()
        elif path == "/api/cloud-models/add":
            self._handle_cloud_model_add()
        elif path == "/api/cloud-models/remove":
            self._handle_cloud_model_remove()
        elif path == "/api/cloud-models/activate":
            self._handle_cloud_model_activate()
        elif path == "/api/cloud-config/defaults":
            self._handle_cloud_defaults_update()
        elif path == "/api/tool-defaults":
            self._handle_tool_defaults_update()
        elif path == "/api/upload":
            self._handle_file_upload()
        else:
            _json_response(self, {"error": "Not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    # ── API Handlers ──────────────────────────────────────────────────────────
    def _handle_status(self):
        s = _state
        elapsed = datetime.datetime.now() - (s["session_start"] or datetime.datetime.now())
        hours, rem = divmod(int(elapsed.total_seconds()), 3600)
        mins, secs = divmod(rem, 60)
        _json_response(self, {
            "model": s["model"],
            "docs": list(s["doc_chunk_counts"].keys()),
            "total_chunks": sum(s["doc_chunk_counts"].values()),
            "top_k": TOP_K,
            "turns": len([m for m in s["messages"] if m["role"] == "user"]),
            "voice": s["voice_enabled"],
            "uptime": f"{hours:02d}:{mins:02d}:{secs:02d}",
        })

    def _handle_models(self):
        models = get_available_models()
        _json_response(self, {
            "models": models,
            "current": _state["model"],
        })

    def _handle_model_select(self):
        body = _read_body(self)
        name = body.get("name", "")
        if not name:
            _json_response(self, {"error": "Missing model name"}, 400)
            return
        _state["model"] = name
        _state["messages"].clear()
        _json_response(self, {"ok": True, "model": name})

    def _handle_docs(self):
        _json_response(self, {
            "docs": [
                {"name": name, "chunks": count}
                for name, count in _state["doc_chunk_counts"].items()
            ],
        })

    def _handle_history(self):
        _json_response(self, {"messages": _state["messages"]})

    def _handle_clear(self):
        _state["messages"].clear()
        _json_response(self, {"ok": True})

    def _handle_config(self):
        _json_response(self, {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "top_k": TOP_K,
            "embed_model": EMBED_MODEL,
            "min_model_size_mb": MIN_MODEL_SIZE_MB,
        })

    def _handle_chat(self):
        """RAG response using same ollama.chat() as CLI, with optional tool-calling loop."""
        body = _read_body(self)
        question = body.get("question", "").strip()
        if not question:
            _json_response(self, {"error": "Empty question"}, 400)
            return

        collection = _state["collection"]
        model = _state["model"]
        messages = _state["messages"]

        # Retrieve RAG context
        raw_chunks = []
        if collection is None or collection.count() == 0:
            context = "No documents have been loaded into the system yet."
        else:
            raw_chunks = retrieve_raw(collection, question)
            if not raw_chunks:
                context = "No relevant context found for this specific query, though documents are loaded."
            else:
                context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in raw_chunks)

        # Build system prompt with memory context
        memory_ctx = agent_memory.get_memory_context()
        system_parts = [
            "You are a helpful AI assistant with tools and persistent memory. ",
            "Use the CONTEXT below to answer the user's question when relevant. ",
            "You can use tools to search the web, run code, read/write files, and manage memory.\n\n",
            f"CONTEXT:\n{context}\n\n",
        ]
        if memory_ctx:
            system_parts.append(f"MEMORY:\n{memory_ctx}\n\n")
        system_parts.append("Answer the question thoroughly. Mention sources when possible.")
        system_prompt = "".join(system_parts)

        turn_messages = (
            [{"role": "system", "content": system_prompt}]
            + messages[-16:]  # Keep recent context manageable
            + [{"role": "user", "content": question}]
        )

        # Attach files if provided (for Gemini multimodal)
        attachments = body.get("attachments", [])
        if attachments and turn_messages:
            turn_messages[-1]["attachments"] = attachments

        try:
            tool_calls_log = []  # Track tool usage for the response

            # Check if using cloud provider
            cloud_cfg = cloud_providers.get_config()
            active = cloud_cfg.get("active_provider", "ollama")
            cloud_model = cloud_providers.get_cloud_model(active) if active != "ollama" else None

            if cloud_model:
                # ── Cloud LLM path ────────────────────────────────────────
                tools = _tool_registry.get_ollama_tools() if _state.get("tool_calling") else None
                agent_defaults = cloud_cfg.get("agent_defaults", {})
                max_iters = agent_defaults.get("max_tool_iterations", 5)

                # Merge agent defaults into cloud model config
                chat_cfg = {**cloud_model}
                if "max_tokens" not in chat_cfg:
                    chat_cfg["max_tokens"] = agent_defaults.get("max_tokens", 4096)
                if "temperature" not in chat_cfg:
                    chat_cfg["temperature"] = agent_defaults.get("temperature", 0.7)

                for _ in range(max_iters):
                    resp = cloud_providers.chat_cloud(chat_cfg, turn_messages, tools)
                    if "error" in resp:
                        _json_response(self, {"error": resp["error"]}, 500)
                        return

                    if resp.get("tool_calls") and _state.get("tool_calling"):
                        turn_messages.append({"role": "assistant", "content": resp.get("content", ""), "tool_calls": resp["tool_calls"]})
                        for tc in resp["tool_calls"]:
                            fn = tc.get("function", {})
                            fn_name = fn.get("name", "")
                            fn_args = fn.get("arguments", {})
                            if isinstance(fn_args, str):
                                try: fn_args = json.loads(fn_args)
                                except: fn_args = {}
                            result = _tool_registry.execute(fn_name, fn_args)
                            tool_calls_log.append({"tool": fn_name, "args": fn_args, "result": result})
                            turn_messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                        continue
                    else:
                        answer = resp.get("content", "")
                        break
                else:
                    answer = resp.get("content", "(Tool loop limit reached)")

            else:
                # ── Ollama path (existing) ────────────────────────────────
                if _state.get("tool_calling"):
                    tools = _tool_registry.get_ollama_tools()
                    for _ in range(5):
                        response = ollama.chat(model=model, messages=turn_messages, tools=tools)
                        msg = response["message"]

                        if msg.get("tool_calls"):
                            turn_messages.append(msg)
                            for tc in msg["tool_calls"]:
                                fn_name = tc["function"]["name"]
                                fn_args = tc["function"].get("arguments", {})
                                result = _tool_registry.execute(fn_name, fn_args)
                                tool_calls_log.append({"tool": fn_name, "args": fn_args, "result": result})
                                turn_messages.append({
                                    "role": "tool",
                                    "content": json.dumps(result, ensure_ascii=False),
                                })
                            continue
                        else:
                            answer = msg.get("content", "")
                            break
                    else:
                        answer = msg.get("content", "(Tool loop limit reached)")
                else:
                    stream = ollama.chat(model=model, messages=turn_messages, stream=True)
                    answer = ""
                    for chunk in stream:
                        answer += chunk["message"]["content"]

            # Save to conversation history
            messages.append({"role": "user", "content": question})
            messages.append({"role": "assistant", "content": answer})
            if len(messages) > 20:
                _state["messages"] = messages[-20:]

            _json_response(self, {
                "answer": answer,
                "sources": [{"source": c["source"], "text": c["text"][:200]} for c in raw_chunks],
                "tool_calls": tool_calls_log,
                "provider": cloud_model["model_name"] if cloud_model else "ollama",
            })
        except Exception as e:
            _json_response(self, {"error": str(e)}, 500)

    # ── Tools API ─────────────────────────────────────────────────────────────
    def _handle_tools_list(self):
        _json_response(self, {
            "tools": _tool_registry.list_tools(),
            "tool_calling": _state.get("tool_calling", True),
        })

    def _handle_tool_execute(self):
        body = _read_body(self)
        name = body.get("name", "")
        args = body.get("args", {})
        if not name:
            _json_response(self, {"error": "Missing tool name"}, 400)
            return
        result = _tool_registry.execute(name, args)
        _json_response(self, {"result": result})

    def _handle_tool_toggle(self):
        body = _read_body(self)
        _state["tool_calling"] = body.get("enabled", True)
        _json_response(self, {"tool_calling": _state["tool_calling"]})

    # ── Cron API ──────────────────────────────────────────────────────────────
    def _handle_cron_list(self):
        from rag.agent_tools import CronTool
        jobs = []
        for jid, info in CronTool._jobs.items():
            jobs.append({"id": jid, "message": info["message"], "type": info["type"], "created": info["created"]})
        _json_response(self, {"jobs": jobs})

    def _handle_cron_action(self):
        body = _read_body(self)
        result = _tool_registry.execute("cron", body)
        _json_response(self, result)

    # ── Cloud Provider API ────────────────────────────────────────────────────
    def _handle_cloud_config_get(self):
        cfg = cloud_providers.get_config()
        # Mask API keys for security
        safe_models = []
        for m in cfg.get("cloud_models", []):
            safe = dict(m)
            if safe.get("api_key"):
                safe["api_key"] = safe["api_key"][:4] + "..." + safe["api_key"][-4:] if len(safe["api_key"]) > 8 else "***"
            safe_models.append(safe)
        _json_response(self, {
            "cloud_models": safe_models,
            "active_provider": cfg.get("active_provider", "ollama"),
            "agent_defaults": cfg.get("agent_defaults", {}),
        })

    def _handle_cloud_config_save(self):
        body = _read_body(self)
        cloud_providers.save_config(body)
        _json_response(self, {"ok": True})

    def _handle_cloud_providers_list(self):
        _json_response(self, {"providers": cloud_providers.KNOWN_PROVIDERS})

    def _handle_cloud_model_add(self):
        body = _read_body(self)
        result = cloud_providers.add_cloud_model(body)
        if "error" in result:
            _json_response(self, result, 400)
        else:
            _json_response(self, {"ok": True, "config": result})

    def _handle_cloud_model_remove(self):
        body = _read_body(self)
        model_name = body.get("model_name", "")
        if not model_name:
            _json_response(self, {"error": "Missing model_name"}, 400)
            return
        result = cloud_providers.remove_cloud_model(model_name)
        _json_response(self, {"ok": True, "config": result})

    def _handle_cloud_model_activate(self):
        body = _read_body(self)
        model_name = body.get("model_name", "ollama")
        result = cloud_providers.set_active_provider(model_name)
        _json_response(self, {"ok": True, "active_provider": model_name})

    def _handle_cloud_defaults_update(self):
        body = _read_body(self)
        result = cloud_providers.update_agent_defaults(body)
        _json_response(self, {"ok": True, "agent_defaults": result.get("agent_defaults", {})})

    def _handle_tool_defaults_update(self):
        body = _read_body(self)
        result = cloud_providers.update_tool_defaults(body)
        _json_response(self, {"ok": True, "tool_defaults": result.get("tool_defaults", {})})

    def _handle_file_upload(self):
        """Handle file uploads for Gemini multimodal chat (images, PDFs)."""
        content_type = self.headers.get("Content-Type", "")
        length = int(self.headers.get("Content-Length", 0))

        if length == 0:
            _json_response(self, {"error": "No file data"}, 400)
            return

        os.makedirs(UPLOAD_DIR, exist_ok=True)

        # Handle multipart form data
        if "multipart/form-data" in content_type:
            import cgi
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": content_type}
            )
            uploaded_files = []
            items = form["file"] if "file" in form else []
            if not isinstance(items, list):
                items = [items]
            for item in items:
                if item.file:
                    ext = os.path.splitext(item.filename or "")[1] or ".bin"
                    fname = f"{uuid.uuid4().hex}{ext}"
                    fpath = os.path.join(UPLOAD_DIR, fname)
                    with open(fpath, "wb") as out:
                        out.write(item.file.read())
                    # Detect MIME type
                    MIME_MAP = {
                        ".pdf": "application/pdf", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                        ".png": "image/png", ".webp": "image/webp", ".heic": "image/heic",
                        ".gif": "image/gif", ".txt": "text/plain", ".md": "text/markdown",
                    }
                    mime = MIME_MAP.get(ext.lower(), "application/octet-stream")
                    uploaded_files.append({
                        "file_path": fpath,
                        "filename": item.filename,
                        "mime_type": mime,
                        "size": os.path.getsize(fpath),
                    })
            _json_response(self, {"files": uploaded_files})
        else:
            # Raw binary upload — read Content-Type header for mime
            raw = self.rfile.read(length)
            ext = ".bin"
            if "pdf" in content_type: ext = ".pdf"
            elif "jpeg" in content_type or "jpg" in content_type: ext = ".jpg"
            elif "png" in content_type: ext = ".png"
            elif "webp" in content_type: ext = ".webp"
            fname = f"{uuid.uuid4().hex}{ext}"
            fpath = os.path.join(UPLOAD_DIR, fname)
            with open(fpath, "wb") as out:
                out.write(raw)
            _json_response(self, {"files": [{
                "file_path": fpath,
                "filename": fname,
                "mime_type": content_type.split(";")[0].strip(),
                "size": len(raw),
            }]})

    # ── Memory API ────────────────────────────────────────────────────────────
    def _handle_memory_read(self):
        _json_response(self, {
            "long_term": agent_memory.read_long_term(),
            "recent_notes": agent_memory.get_recent_notes(3),
        })

    def _handle_memory_write(self):
        body = _read_body(self)
        content = body.get("content", "")
        mode = body.get("mode", "overwrite")
        if not content:
            _json_response(self, {"error": "Missing content"}, 400)
            return
        if mode == "append":
            agent_memory.append_long_term(content)
        else:
            agent_memory.write_long_term(content)
        _json_response(self, {"ok": True})

    def _handle_memory_today_read(self):
        _json_response(self, {
            "today": agent_memory.read_today(),
        })

    def _handle_memory_today_write(self):
        body = _read_body(self)
        content = body.get("content", "")
        if not content:
            _json_response(self, {"error": "Missing content"}, 400)
            return
        agent_memory.append_today(content)
        _json_response(self, {"ok": True})

    # ── Skills API ────────────────────────────────────────────────────────────
    def _handle_skills_list(self):
        _json_response(self, {
            "skills": agent_skills.list_skills(),
        })

    def _handle_add_url(self):
        body = _read_body(self)
        url = body.get("url", "").strip()
        if not url:
            _json_response(self, {"error": "Missing URL"}, 400)
            return
        try:
            from rag.scraper import add_url_to_db
            _state["chunk_offset"] = add_url_to_db(
                _state["collection"], url,
                _state["doc_chunk_counts"], _state["chunk_offset"]
            )
            _json_response(self, {"ok": True, "docs": list(_state["doc_chunk_counts"].keys())})
        except Exception as e:
            _json_response(self, {"error": str(e)}, 500)

    def _handle_remove_doc(self):
        body = _read_body(self)
        name = body.get("name", "")
        if not name or name not in _state["doc_chunk_counts"]:
            _json_response(self, {"error": "Invalid document name"}, 400)
            return
        try:
            _state["collection"].delete(where={"source": name})
            del _state["doc_chunk_counts"][name]
            _state["selected_paths"] = [p for p in _state["selected_paths"] if os.path.basename(p) != name]
            _json_response(self, {"ok": True})
        except Exception as e:
            _json_response(self, {"error": str(e)}, 500)

    def _handle_wiki(self):
        body = _read_body(self)
        topic = body.get("topic", "").strip()
        if not topic:
            _json_response(self, {"error": "Missing topic"}, 400)
            return
        try:
            from rag.wiki import fetch_wiki, wiki_to_rag
            folder, text, title = fetch_wiki(topic)
            label = os.path.basename(folder)
            _state["chunk_offset"] = wiki_to_rag(
                _state["collection"], text, label,
                _state["doc_chunk_counts"], _state["chunk_offset"]
            )
            _json_response(self, {"ok": True, "title": title, "label": label})
        except Exception as e:
            _json_response(self, {"error": str(e)}, 500)

    def _handle_weather(self):
        body = _read_body(self)
        city = body.get("city", "").strip()
        if not city:
            _json_response(self, {"error": "Missing city"}, 400)
            return
        try:
            from rag.weather import fetch_weather, weather_to_rag
            official_name, overview = fetch_weather(city)
            _state["chunk_offset"] = weather_to_rag(
                _state["collection"], overview, official_name,
                _state["doc_chunk_counts"], _state["chunk_offset"]
            )
            _json_response(self, {"ok": True, "city": official_name})
        except Exception as e:
            _json_response(self, {"error": str(e)}, 500)

    def _handle_news(self):
        body = _read_body(self)
        query = body.get("query", "").strip()
        endpoint = body.get("endpoint", "headlines")
        if not query and endpoint != "sources":
            _json_response(self, {"error": "Missing query"}, 400)
            return
        try:
            from rag.news import fetch_top_headlines, fetch_everything, fetch_sources, news_to_rag
            if endpoint == "headlines":
                folder, text, count = fetch_top_headlines(query)
            elif endpoint == "everything":
                folder, text, count = fetch_everything(query)
            else:
                folder, text, count = fetch_sources()

            if count > 0:
                label = os.path.basename(folder)
                _state["chunk_offset"] = news_to_rag(
                    _state["collection"], text, label,
                    _state["doc_chunk_counts"], _state["chunk_offset"]
                )
            _json_response(self, {"ok": True, "count": count})
        except Exception as e:
            _json_response(self, {"error": str(e)}, 500)


# ── Server lifecycle ──────────────────────────────────────────────────────────

def start_server(collection, model, messages, doc_chunk_counts, selected_paths,
                 chunk_offset, session_start, voice_enabled):
    """Start the web server in a background thread, sharing CLI state."""
    global _server, _thread

    if _server is not None:
        console.print(f"  [system]Agent UI already running →[/] [cmd]http://localhost:{PORT}[/]")
        return

    # Share state references
    _state["collection"] = collection
    _state["model"] = model
    _state["messages"] = messages
    _state["doc_chunk_counts"] = doc_chunk_counts
    _state["selected_paths"] = selected_paths
    _state["chunk_offset"] = chunk_offset
    _state["session_start"] = session_start
    _state["voice_enabled"] = voice_enabled

    # Ensure static dir exists
    os.makedirs(STATIC_DIR, exist_ok=True)

    _server = HTTPServer(("127.0.0.1", PORT), AgentHandler)

    def _run():
        _server.serve_forever()

    _thread = threading.Thread(target=_run, daemon=True)
    _thread.start()

    url = f"http://localhost:{PORT}"
    console.print(f"\n  [system]✓ Agent Web UI started →[/] [cmd]{url}[/]")
    console.print(f"  [info]Your CLI session continues to work alongside the browser.[/]")

    try:
        webbrowser.open(url)
    except Exception:
        pass


def stop_server():
    """Stop the web server."""
    global _server, _thread
    if _server is None:
        console.print("  [info]Agent UI is not running.[/]")
        return
    _server.shutdown()
    _server = None
    _thread = None
    console.print("  [system]✓ Agent Web UI stopped.[/]")


def is_running() -> bool:
    return _server is not None
