"""
Cloud LLM Providers — configurable cloud model support (OpenAI, Anthropic, Groq, etc.)
Inspired by PicoClaw's ModelConfig and provider factory.

Config stored as JSON at ~/.rag-agent/cloud_config.json.
All OpenAI-compatible providers (OpenAI, Groq, DeepSeek, Mistral, Together, OpenRouter, etc.)
are supported via the same adapter. Anthropic uses its own messages API.
"""

import os
import json
import urllib.request
import urllib.error

CONFIG_PATH = os.path.expanduser("~/.rag-agent/cloud_config.json")

# ── Known Providers (pre-filled api_base) ─────────────────────────────────────
KNOWN_PROVIDERS = {
    "openai":       {"name": "OpenAI",        "api_base": "https://api.openai.com/v1",          "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1-mini", "o1-preview"]},
    "anthropic":    {"name": "Anthropic",      "api_base": "https://api.anthropic.com",          "models": ["claude-sonnet-4-20250514", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"]},
    "gemini":       {"name": "Google Gemini",  "api_base": "generativelanguage.googleapis.com",  "models": ["gemini-3-flash-preview", "gemini-3-pro-preview", "gemini-2.5-flash-preview-05-20", "gemini-2.0-flash", "gemini-2.0-flash-lite"]},
    "groq":         {"name": "Groq",           "api_base": "https://api.groq.com/openai/v1",    "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"]},
    "deepseek":     {"name": "DeepSeek",       "api_base": "https://api.deepseek.com/v1",       "models": ["deepseek-chat", "deepseek-coder", "deepseek-reasoner"]},
    "mistral":      {"name": "Mistral",        "api_base": "https://api.mistral.ai/v1",         "models": ["mistral-large-latest", "mistral-small-latest", "codestral-latest"]},
    "together":     {"name": "Together AI",    "api_base": "https://api.together.xyz/v1",       "models": ["meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "mistralai/Mixtral-8x7B-Instruct-v0.1"]},
    "openrouter":   {"name": "OpenRouter",     "api_base": "https://openrouter.ai/api/v1",      "models": ["openai/gpt-4o", "anthropic/claude-sonnet-4-20250514", "google/gemini-2.0-flash-001"]},
    "fireworks":    {"name": "Fireworks AI",   "api_base": "https://api.fireworks.ai/inference/v1", "models": ["accounts/fireworks/models/llama-v3p1-70b-instruct"]},
    "custom":       {"name": "Custom (OpenAI-compatible)", "api_base": "", "models": []},
}

# ── Default config ────────────────────────────────────────────────────────────

_DEFAULT_CONFIG = {
    "cloud_models": [],  # List of ModelConfig dicts
    "active_provider": "ollama",  # "ollama" or cloud model_name
    "agent_defaults": {
        "max_tokens": 4096,
        "temperature": 0.7,
        "max_tool_iterations": 5,
        "context_window": 16384,
    },
    "tool_defaults": {
        "wiki": "",
        "weather": "",
        "news": "",
        "scrape": ""
    }
}


def _load_config() -> dict:
    """Load cloud config from disk or return defaults."""
    if os.path.isfile(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
            # Merge with defaults for missing keys
            for k, v in _DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception:
            pass
    return dict(_DEFAULT_CONFIG)


def _save_config(cfg: dict):
    """Save cloud config to disk."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)


def get_config() -> dict:
    return _load_config()


def save_config(cfg: dict):
    _save_config(cfg)


def add_cloud_model(model_cfg: dict) -> dict:
    """Add or update a cloud model. Returns the updated config."""
    cfg = _load_config()
    # Required fields
    if not model_cfg.get("model_name") or not model_cfg.get("provider"):
        return {"error": "model_name and provider are required"}

    # Auto-fill api_base from known providers
    provider = model_cfg["provider"]
    if provider in KNOWN_PROVIDERS and not model_cfg.get("api_base"):
        model_cfg["api_base"] = KNOWN_PROVIDERS[provider]["api_base"]

    # Update or insert
    existing = next((i for i, m in enumerate(cfg["cloud_models"]) if m["model_name"] == model_cfg["model_name"]), None)
    if existing is not None:
        cfg["cloud_models"][existing] = model_cfg
    else:
        cfg["cloud_models"].append(model_cfg)

    _save_config(cfg)
    return cfg


def remove_cloud_model(model_name: str) -> dict:
    """Remove a cloud model. Returns updated config."""
    cfg = _load_config()
    cfg["cloud_models"] = [m for m in cfg["cloud_models"] if m["model_name"] != model_name]
    if cfg["active_provider"] == model_name:
        cfg["active_provider"] = "ollama"
    _save_config(cfg)
    return cfg


def set_active_provider(model_name: str) -> dict:
    """Set the active provider to 'ollama' or a cloud model_name."""
    cfg = _load_config()
    cfg["active_provider"] = model_name
    _save_config(cfg)
    return cfg


def update_agent_defaults(updates: dict) -> dict:
    """Update agent defaults (temperature, max_tokens, etc.)."""
    cfg = _load_config()
    for k in ("max_tokens", "temperature", "max_tool_iterations", "context_window"):
        if k in updates:
            cfg["agent_defaults"][k] = updates[k]
    _save_config(cfg)
    return cfg


def update_tool_defaults(updates: dict) -> dict:
    """Update UI tool input defaults."""
    cfg = _load_config()
    if "tool_defaults" not in cfg:
        cfg["tool_defaults"] = {}
    for k in ("wiki", "weather", "news", "scrape"):
        if k in updates:
            cfg["tool_defaults"][k] = updates[k]
    _save_config(cfg)
    return cfg


def get_cloud_model(model_name: str) -> dict | None:
    """Get a specific cloud model config by name."""
    cfg = _load_config()
    return next((m for m in cfg["cloud_models"] if m["model_name"] == model_name), None)


# ── Cloud Chat API ────────────────────────────────────────────────────────────

def chat_cloud(model_cfg: dict, messages: list, tools: list | None = None) -> dict:
    """
    Chat with a cloud LLM. Supports OpenAI-compatible API, Anthropic, and Gemini.
    Returns: {"content": str, "tool_calls": list | None}
    """
    provider = model_cfg.get("provider", "")
    api_key = model_cfg.get("api_key", "")

    if not api_key:
        return {"error": f"No API key configured for {model_cfg.get('model_name', '?')}"}

    if provider == "anthropic":
        return _chat_anthropic(model_cfg, messages, tools)
    elif provider == "gemini":
        return _chat_gemini(model_cfg, messages, tools)
    else:
        return _chat_openai_compat(model_cfg, messages, tools)


def _chat_openai_compat(model_cfg: dict, messages: list, tools: list | None) -> dict:
    """Chat via OpenAI-compatible API (works for OpenAI, Groq, DeepSeek, Mistral, Together, etc.)."""
    api_base = model_cfg["api_base"].rstrip("/")
    url = f"{api_base}/chat/completions"

    payload = {
        "model": model_cfg.get("model", model_cfg["model_name"]),
        "messages": messages,
        "max_tokens": model_cfg.get("max_tokens", 4096),
        "temperature": model_cfg.get("temperature", 0.7),
    }
    if tools:
        payload["tools"] = tools

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_cfg['api_key']}",
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        timeout = model_cfg.get("request_timeout", 60)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        choice = result.get("choices", [{}])[0]
        msg = choice.get("message", {})
        return {
            "content": msg.get("content", ""),
            "tool_calls": msg.get("tool_calls"),
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"API error ({e.code}): {body}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def _chat_anthropic(model_cfg: dict, messages: list, tools: list | None) -> dict:
    """Chat via Anthropic Messages API."""
    url = f"{model_cfg['api_base'].rstrip('/')}/v1/messages"

    # Convert from OpenAI message format to Anthropic
    system_prompt = ""
    anthropic_messages = []
    for msg in messages:
        if msg["role"] == "system":
            system_prompt = msg["content"]
        elif msg["role"] in ("user", "assistant"):
            anthropic_messages.append({"role": msg["role"], "content": msg["content"]})
        elif msg["role"] == "tool":
            anthropic_messages.append({
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool", "content": msg["content"]}]
            })

    payload = {
        "model": model_cfg.get("model", model_cfg["model_name"]),
        "messages": anthropic_messages,
        "max_tokens": model_cfg.get("max_tokens", 4096),
    }
    if system_prompt:
        payload["system"] = system_prompt
    if tools:
        # Convert OpenAI tool format to Anthropic
        anthropic_tools = []
        for t in tools:
            fn = t.get("function", {})
            anthropic_tools.append({
                "name": fn.get("name", ""),
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {}),
            })
        payload["tools"] = anthropic_tools

    headers = {
        "Content-Type": "application/json",
        "x-api-key": model_cfg["api_key"],
        "anthropic-version": "2023-06-01",
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read().decode("utf-8"))

        # Extract content from Anthropic's response format
        content_parts = result.get("content", [])
        text_parts = [c["text"] for c in content_parts if c.get("type") == "text"]
        tool_uses = [c for c in content_parts if c.get("type") == "tool_use"]

        tool_calls = None
        if tool_uses:
            tool_calls = [{
                "function": {"name": t["name"], "arguments": t.get("input", {})},
            } for t in tool_uses]

        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
        }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        return {"error": f"Anthropic API error ({e.code}): {body}"}
    except Exception as e:
        return {"error": f"Request failed: {str(e)}"}


def _chat_gemini(model_cfg: dict, messages: list, tools: list | None) -> dict:
    """
    Chat via Google Gemini using the google-genai SDK.
    Full multimodal support:
      - Text generation with system instructions
      - Image understanding (JPEG, PNG, WebP, HEIC/HEIF)
      - Document understanding (PDF, up to 1000 pages)
      - Files API for large uploads (auto for files > 20MB)
      - Multi-turn conversations
      - Tool calling via FunctionDeclaration
    
    Messages may include 'attachments' list:
      [{"file_path": "/path/to/file.pdf", "mime_type": "application/pdf"},
       {"url": "https://example.com/image.jpg", "mime_type": "image/jpeg"},
       {"file_path": "/path/to/big.pdf", "use_files_api": true}]
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return {"error": "google-genai package not installed. Run: pip install google-genai"}

    try:
        client = genai.Client(api_key=model_cfg["api_key"])
        model_id = model_cfg.get("model", model_cfg["model_name"])

        # MIME type detection helper
        MIME_MAP = {
            ".pdf": "application/pdf",
            ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
            ".png": "image/png", ".webp": "image/webp",
            ".heic": "image/heic", ".heif": "image/heif",
            ".gif": "image/gif",
            ".txt": "text/plain", ".md": "text/markdown",
            ".html": "text/html", ".htm": "text/html",
            ".csv": "text/csv", ".xml": "text/xml",
            ".json": "application/json",
        }

        def _detect_mime(path: str) -> str:
            ext = os.path.splitext(path)[1].lower()
            return MIME_MAP.get(ext, "application/octet-stream")

        def _load_attachment(att: dict) -> types.Part | None:
            """Convert an attachment dict to a Gemini Part."""
            mime = att.get("mime_type", "")
            use_files_api = att.get("use_files_api", False)

            # URL-based attachment
            if att.get("url"):
                url = att["url"]
                if not mime:
                    mime = _detect_mime(url)
                try:
                    req = urllib.request.Request(url, headers={"User-Agent": "RAG-Agent/1.0"})
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = resp.read()
                except Exception as e:
                    return types.Part.from_text(text=f"[Failed to fetch {url}: {e}]")

                # Use Files API for large downloads (>20MB)
                if len(data) > 20 * 1024 * 1024 or use_files_api:
                    import io
                    uploaded = client.files.upload(
                        file=io.BytesIO(data),
                        config={"mime_type": mime}
                    )
                    return uploaded
                return types.Part.from_bytes(data=data, mime_type=mime)

            # Local file attachment
            if att.get("file_path"):
                fpath = att["file_path"]
                if not os.path.isfile(fpath):
                    return types.Part.from_text(text=f"[File not found: {fpath}]")
                if not mime:
                    mime = _detect_mime(fpath)
                file_size = os.path.getsize(fpath)

                # Files API for large files (>20MB) or if explicitly requested
                if file_size > 20 * 1024 * 1024 or use_files_api:
                    uploaded = client.files.upload(file=fpath)
                    return uploaded

                # Inline for smaller files
                import pathlib
                data = pathlib.Path(fpath).read_bytes()
                return types.Part.from_bytes(data=data, mime_type=mime)

            # Base64 inline data
            if att.get("data"):
                import base64
                data = base64.b64decode(att["data"])
                return types.Part.from_bytes(data=data, mime_type=mime or "application/octet-stream")

            return None

        # Build contents and extract system instruction
        system_instruction = None
        contents = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                parts = [types.Part.from_text(text=msg["content"])]
                # Process attachments (images, PDFs, documents)
                for att in msg.get("attachments", []):
                    part = _load_attachment(att)
                    if part is not None:
                        parts.append(part)
                contents.append(types.Content(role="user", parts=parts))
            elif msg["role"] == "assistant":
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=msg["content"])]
                ))
            elif msg["role"] == "tool":
                contents.append(types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=f"Tool result: {msg['content']}")]
                ))

        # Build generation config
        gen_config = types.GenerateContentConfig(
            temperature=model_cfg.get("temperature", 1.0),
            max_output_tokens=model_cfg.get("max_tokens", 8192),
        )
        if system_instruction:
            gen_config.system_instruction = system_instruction

        # Convert tools to Gemini format
        if tools:
            gemini_tools = []
            for t in tools:
                fn = t.get("function", {})
                params = fn.get("parameters", {})
                properties = {}
                for pname, pinfo in params.get("properties", {}).items():
                    ptype = pinfo.get("type", "string").upper()
                    properties[pname] = types.Schema(
                        type=ptype,
                        description=pinfo.get("description", ""),
                    )
                func_decl = types.FunctionDeclaration(
                    name=fn.get("name", ""),
                    description=fn.get("description", ""),
                    parameters=types.Schema(
                        type="OBJECT",
                        properties=properties,
                        required=params.get("required", []),
                    ) if properties else None,
                )
                gemini_tools.append(func_decl)
            gen_config.tools = [types.Tool(function_declarations=gemini_tools)]

        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=gen_config,
        )

        # Extract text and tool calls from response
        text_parts = []
        tool_calls = None

        if response.candidates:
            for part in response.candidates[0].content.parts:
                if part.text:
                    text_parts.append(part.text)
                elif part.function_call:
                    if tool_calls is None:
                        tool_calls = []
                    tool_calls.append({
                        "function": {
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args) if part.function_call.args else {},
                        }
                    })

        return {
            "content": "\n".join(text_parts),
            "tool_calls": tool_calls,
        }
    except Exception as e:
        return {"error": f"Gemini API error: {str(e)}"}

