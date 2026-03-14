"""
Main chat loop — handles all slash-commands and normal
question/answer turns.  This is the heart of the interactive session.
"""

import os
import re
import json
import datetime

import ollama
import requests
import chromadb
from rich.panel import Panel
from rich.table import Table
from rich import box

from rag.config   import (DOC_FOLDER, CHATS_FOLDER, EXPORTS_FOLDER,
                           SCRAPED_DOCS_FOLDER, REDDIT_JSON_FOLDER,
                           AUTOSAVE_EVERY, TOP_K)
from rag.console  import console
from rag import speech
from rag.vectordb import build_vector_db, add_doc_to_db
from rag.scraper  import add_url_to_db
from rag.export   import export_pdf
from rag.ui       import choose_model, choose_docs, print_help
from rag.query    import run_query


def chat(
    collection: chromadb.Collection,
    model: str,
    selected_paths: list[str],
    doc_chunk_counts: dict[str, int],
):
    doc_names    = ", ".join(os.path.basename(p) for p in selected_paths)
    chunk_offset = sum(doc_chunk_counts.values())

    console.print(f"\n  [info]Loaded:[/]  {doc_names}")
    console.print(f"  [info]Model:[/]   {model}")
    console.print(f"  [info]Voice:[/]   {'on' if speech.voice_enabled else 'off'}")
    console.print("  Type a question, [cmd]/help[/] for commands, or [error]exit[/] to quit.\n")
    console.rule(style="info")

    messages:           list[dict] = []
    last_chunks:        list[dict] = []
    last_question:      str        = ""
    autosave_turn_count: int       = 0
    autosave_stem:       str       = f"autosave_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_start:       datetime.datetime = datetime.datetime.now()

    while True:
        try:
            question = input("\n[You] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not question:
            continue

        cmd = question.lower()

        # ── exit ──────────────────────────────────────────────────────────
        if cmd == "exit":
            break

        # ── /help ─────────────────────────────────────────────────────────
        elif cmd == "/help":
            print_help()

        # ── /status ───────────────────────────────────────────────────────
        elif cmd == "/status":
            elapsed    = datetime.datetime.now() - session_start
            hours, rem = divmod(int(elapsed.total_seconds()), 3600)
            mins, secs = divmod(rem, 60)
            uptime     = f"{hours:02d}:{mins:02d}:{secs:02d}"
            turns      = len([m for m in messages if m["role"] == "user"])
            total_chunks = sum(doc_chunk_counts.values())
            voice_state  = "[system]ON[/]" if speech.voice_enabled else "[error]OFF[/]"

            grid = Table(box=box.ROUNDED, show_header=False, padding=(0, 2), expand=False)
            grid.add_column(style="info",   no_wrap=True)
            grid.add_column(style="cmd",    no_wrap=True)
            grid.add_row("Model",        model)
            grid.add_row("Documents",    ", ".join(doc_chunk_counts.keys()) or "none")
            grid.add_row("Total chunks", str(total_chunks))
            grid.add_row("Top-K",        str(TOP_K))
            grid.add_row("Turns",        str(turns))
            grid.add_row("Auto-save",    f"every {AUTOSAVE_EVERY} turns")
            grid.add_row("Voice",        ("on" if speech.voice_enabled else "off"))
            grid.add_row("Uptime",       uptime)
            grid.add_row("chats/",       os.path.abspath(CHATS_FOLDER))
            grid.add_row("exports/",     os.path.abspath(EXPORTS_FOLDER))
            console.print(Panel(grid, title="[system]  Status  [/]", box=box.ROUNDED))

        # ── /history ──────────────────────────────────────────────────────
        elif cmd == "/history":
            if not messages:
                console.print("  [info]No history yet.[/]")
            else:
                console.print()
                for msg in messages:
                    if msg["role"] == "user":
                        console.print(f"[user]You:[/]  {msg['content']}")
                    else:
                        console.print(f"[ai]AI:[/]   {msg['content']}")
                    console.rule(style="info")

        # ── /clear ────────────────────────────────────────────────────────
        elif cmd == "/clear":
            messages.clear()
            last_chunks.clear()
            last_question = ""
            console.print("  [system]✓ History cleared.[/]")

        # ── /save-chat [name] ─────────────────────────────────────────────
        elif cmd.startswith("/save-chat"):
            if not messages:
                console.print("  [info]Nothing to save yet.[/]")
            else:
                os.makedirs(CHATS_FOLDER, exist_ok=True)
                parts     = question.split(maxsplit=1)
                ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                # Use custom name if given, sanitise it for filesystem
                if len(parts) > 1:
                    safe_name = re.sub(r"[^\w\-]", "_", parts[1].strip())
                    stem      = f"{safe_name}_{ts}"
                else:
                    stem      = f"chat_{ts}"

                md_path   = os.path.join(CHATS_FOLDER, f"{stem}.md")
                json_path = os.path.join(CHATS_FOLDER, f"{stem}.json")

                with open(md_path, "w", encoding="utf-8") as f:
                    f.write(f"# Chat Export — {stem}\n\n")
                    f.write(f"**Model:** {model}  \n**Docs:** {doc_names}\n\n---\n\n")
                    for msg in messages:
                        role = "**You**" if msg["role"] == "user" else "**AI**"
                        f.write(f"{role}\n\n{msg['content']}\n\n---\n\n")

                payload = {"timestamp": ts, "model": model,
                           "docs": doc_names, "messages": messages}
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(payload, f, indent=2, ensure_ascii=False)

                console.print(f"  [system]✓ Saved →[/] [cmd]{md_path}[/]")
                console.print(f"  [info]  (reloadable: {json_path})[/]")

        # ── /search-history <keyword> ──────────────────────────────────────
        elif cmd.startswith("/search-history"):
            parts = question.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                console.print("  [error]Usage: /search-history <keyword>[/]")
            elif not messages:
                console.print("  [info]No history yet.[/]")
            else:
                kw      = parts[1].strip()
                kw_low  = kw.lower()
                matches = 0
                console.print()
                for i, msg in enumerate(messages):
                    if kw_low in msg["content"].lower():
                        matches += 1
                        role_tag = "[user]You:[/]" if msg["role"] == "user" else "[ai]AI:[/]"
                        # Highlight keyword in context (case-insensitive replace)
                        highlighted = re.sub(
                            f"({re.escape(kw)})",
                            r"[bold yellow]\1[/bold yellow]",
                            msg["content"],
                            flags=re.IGNORECASE,
                        )
                        # Only show first 400 chars to keep it readable
                        preview = highlighted[:400] + ("…" if len(highlighted) > 400 else "")
                        console.print(f"{role_tag} [info](turn {i//2 + 1})[/]")
                        console.print(preview)
                        console.rule(style="info")
                if matches == 0:
                    console.print(f"  [info]No matches for '{kw}'.[/]")
                else:
                    console.print(f"  [system]{matches} match(es) for '[cmd]{kw}[/]'.[/]")

        # ── /summarize ────────────────────────────────────────────────────
        elif cmd == "/summarize":
            if not messages:
                console.print("  [info]No conversation to summarize yet.[/]")
            else:
                history_text = "\n".join(
                    f"{'User' if m['role'] == 'user' else 'AI'}: {m['content']}"
                    for m in messages
                )
                summary_prompt = (
                    "Please write a concise summary (3-5 sentences) of the conversation "
                    "below. Cover the main topics discussed and any conclusions reached.\n\n"
                    f"{history_text}"
                )
                console.print("\n[ai]Summary:[/] ", end="")
                stream = ollama.chat(
                    model=model,
                    messages=[{"role": "user", "content": summary_prompt}],
                    stream=True,
                )
                try:
                    for chunk in stream:
                        console.print(chunk["message"]["content"], end="", markup=False)
                except KeyboardInterrupt:
                    speech.stop_speaking()
                console.print()

        # ── /export-pdf [name] ────────────────────────────────────────────
        elif cmd.startswith("/export-pdf"):
            if not messages:
                console.print("  [info]Nothing to export yet.[/]")
            else:
                os.makedirs(EXPORTS_FOLDER, exist_ok=True)
                parts     = question.split(maxsplit=1)
                ts        = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                if len(parts) > 1:
                    safe_name = re.sub(r"[^\w\-]", "_", parts[1].strip())
                    stem      = f"{safe_name}_{ts}"
                else:
                    stem      = f"chat_{ts}"
                pdf_path  = os.path.join(EXPORTS_FOLDER, f"{stem}.pdf")
                try:
                    export_pdf(messages, model, doc_names, pdf_path)
                    console.print(f"  [system]✓ PDF exported →[/] [cmd]{pdf_path}[/]")
                except Exception as e:
                    console.print(f"  [error]PDF export failed: {e}[/]")

        # ── /add-url <url> ────────────────────────────────────────────────
        elif cmd.startswith("/add-url"):
            parts = question.split(maxsplit=1)
            if len(parts) < 2 or not parts[1].strip():
                console.print("  [error]Usage: /add-url <https://...>[/]")
            else:
                url = parts[1].strip()
                try:
                    chunk_offset = add_url_to_db(collection, url, doc_chunk_counts, chunk_offset)
                    doc_names    = ", ".join(list(doc_chunk_counts.keys()))
                except requests.exceptions.RequestException as e:
                    console.print(f"  [error]Failed to fetch URL: {e}[/]")
                except Exception as e:
                    console.print(f"  [error]Error indexing URL: {e}[/]")

        # ── /load-chat ────────────────────────────────────────────────────
        elif cmd == "/load-chat":
            if not os.path.isdir(CHATS_FOLDER):
                console.print("  [info]No chats/ folder found. Save a chat first.[/]")
            else:
                saved = sorted([
                    f for f in os.listdir(CHATS_FOLDER)
                    if f.endswith(".json")
                ], reverse=True)   # newest first

                if not saved:
                    console.print("  [info]No saved chats found. Use /save-chat first.[/]")
                else:
                    table = Table(box=box.ROUNDED, show_header=True, header_style="system")
                    table.add_column("#",        style="cmd",  width=4)
                    table.add_column("File",     style="info")
                    table.add_column("Model",    style="info")
                    table.add_column("Docs",     style="info")
                    table.add_column("Turns",    style="cmd",  justify="right")

                    previews = []
                    for f in saved:
                        try:
                            with open(os.path.join(CHATS_FOLDER, f), encoding="utf-8") as fh:
                                data = json.load(fh)
                            turns = len([m for m in data["messages"] if m["role"] == "user"])
                            previews.append(data)
                            table.add_row(
                                str(len(previews)),
                                f,
                                data.get("model", "?"),
                                data.get("docs",  "?"),
                                str(turns),
                            )
                        except Exception:
                            pass   # skip malformed files

                    if not previews:
                        console.print("  [error]No valid chat files found.[/]")
                    else:
                        console.print(table)
                        console.print("  [info]Enter a number to load, or press Enter to cancel.[/]")
                        pick = input("  Your choice: ").strip()

                        if pick.isdigit():
                            idx = int(pick) - 1
                            if 0 <= idx < len(previews):
                                data = previews[idx]
                                messages      = data["messages"]
                                last_chunks   = []
                                last_question = ""

                                # Show a brief preview of the last exchange
                                last_q = next((m["content"] for m in reversed(messages) if m["role"] == "user"),      "")
                                last_a = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
                                turns  = len([m for m in messages if m["role"] == "user"])

                                console.print(Panel(
                                    f"[info]Model:[/]  {data.get('model', '?')}\n"
                                    f"[info]Docs:[/]   {data.get('docs',  '?')}\n"
                                    f"[info]Turns:[/]  {turns}\n\n"
                                    f"[user]Last Q:[/] {last_q[:120]}{'…' if len(last_q) > 120 else ''}\n"
                                    f"[ai]Last A:[/] {last_a[:120]}{'…' if len(last_a) > 120 else ''}",
                                    title="[system]✓ Chat Loaded[/]",
                                    box=box.ROUNDED,
                                ))
                                console.print(
                                    "  [info]Note: continuing with your currently loaded docs & model.\n"
                                    "  Use /change-docs or /change-model to match the original session.[/]"
                                )
                            else:
                                console.print("  [error]Invalid number.[/]")
                        else:
                            console.print("  [info]Cancelled.[/]")

        # ── /retry ────────────────────────────────────────────────────────
        elif cmd == "/retry":
            if not last_question:
                console.print("  [info]Nothing to retry yet.[/]")
            else:
                # Drop last exchange from history
                if messages and messages[-1]["role"] == "assistant":
                    messages.pop()
                if messages and messages[-1]["role"] == "user":
                    messages.pop()
                console.print(f"  [system]↺ Retrying:[/] {last_question}")
                console.rule(style="info")
                answer, last_chunks = run_query(collection, model, messages, last_question)
                console.rule(style="info")
                messages.append({"role": "user",      "content": last_question})
                messages.append({"role": "assistant", "content": answer})
                if len(messages) > 20:
                    messages = messages[-20:]

        # ── /why ──────────────────────────────────────────────────────────
        elif cmd == "/why":
            if not last_chunks:
                console.print("  [info]Ask a question first.[/]")
            else:
                console.print()
                for i, chunk in enumerate(last_chunks, 1):
                    console.print(Panel(
                        f"[chunk]{chunk['text']}[/]",
                        title=f"[source]Chunk {i} · {chunk['source']}[/]",
                        box=box.ROUNDED,
                    ))

        # ── /list-docs ────────────────────────────────────────────────────
        elif cmd == "/list-docs":
            table = Table(box=box.ROUNDED, show_header=True, header_style="system")
            table.add_column("Document", style="info")
            table.add_column("Chunks",   style="cmd", justify="right")
            for name, count in doc_chunk_counts.items():
                table.add_row(name, str(count))
            table.add_section()
            table.add_row("[bold]TOTAL[/]", str(sum(doc_chunk_counts.values())))
            console.print(table)

        # ── /add-doc ──────────────────────────────────────────────────────
        elif cmd == "/add-doc":
            available = sorted([
                f for f in os.listdir(DOC_FOLDER)
                if os.path.isfile(os.path.join(DOC_FOLDER, f))
                and not f.startswith(".")
                and f not in doc_chunk_counts
            ])
            if not available:
                console.print("  [info]No new documents available to add.[/]")
            else:
                table = Table(box=box.ROUNDED, show_header=True, header_style="system")
                table.add_column("#",    style="cmd",  width=4)
                table.add_column("File", style="info")
                table.add_column("Size", style="info", justify="right")
                for i, f in enumerate(available, 1):
                    size_kb = os.path.getsize(os.path.join(DOC_FOLDER, f)) // 1024
                    table.add_row(str(i), f, f"{size_kb} KB")
                console.print(table)

                raw = input("  Pick a number to add: ").strip()
                if raw.isdigit():
                    idx = int(raw) - 1
                    if 0 <= idx < len(available):
                        path         = os.path.join(DOC_FOLDER, available[idx])
                        chunk_offset = add_doc_to_db(collection, path, doc_chunk_counts, chunk_offset)
                        selected_paths.append(path)
                        doc_names = ", ".join(os.path.basename(p) for p in selected_paths)
                    else:
                        console.print("  [error]Invalid number.[/]")
                else:
                    console.print("  [info]Cancelled.[/]")

        # ── /voice ────────────────────────────────────────────────────────
        elif cmd == "/voice":
            speech.voice_enabled = not speech.voice_enabled
            if not speech.voice_enabled:
                speech.stop_speaking()   # kill anything currently playing
            state = "[system]ON[/]" if speech.voice_enabled else "[error]OFF[/]"
            console.print(f"  Voice → {state}")

        # ── /remove-doc ───────────────────────────────────────────────────
        elif cmd == "/remove-doc":
            if not doc_chunk_counts:
                console.print("  [info]No documents currently loaded.[/]")
            else:
                table = Table(box=box.ROUNDED, show_header=True, header_style="system")
                table.add_column("#",    style="cmd",  width=4)
                table.add_column("File", style="info")
                table.add_column("Chunks", style="info", justify="right")
                
                doc_list = sorted(doc_chunk_counts.keys())
                for i, name in enumerate(doc_list, 1):
                    table.add_row(str(i), name, str(doc_chunk_counts[name]))
                console.print(table)
                
                pick = input("  Pick a number to remove (or 'cancel'): ").strip().lower()
                if pick.isdigit():
                    idx = int(pick) - 1
                    if 0 <= idx < len(doc_list):
                        name = doc_list[idx]
                        
                        # 1. Remove from ChromaDB
                        try:
                            console.print(f"  [system]Removing '{name}' from memory…[/]")
                            collection.delete(where={"source": name})
                            del doc_chunk_counts[name]
                            # Update selected_paths and doc_names
                            selected_paths = [p for p in selected_paths if os.path.basename(p) != name]
                            doc_names = ", ".join(os.path.basename(p) for p in selected_paths)
                            console.print(f"  [system]✓ Removed from session memory.[/]")
                        except Exception as e:
                            console.print(f"  [error]Failed to remove from database: {e}[/]")
                        
                        # 2. Optional: Remove from disk
                        delete_disk = input(f"  Delete '{name}' from filesystem too? (y/n): ").strip().lower()
                        if delete_disk == 'y':
                            # Check all potential folders
                            found = False
                            for folder in [DOC_FOLDER, SCRAPED_DOCS_FOLDER, REDDIT_JSON_FOLDER]:
                                path = os.path.join(folder, name)
                                if os.path.exists(path):
                                    try:
                                        os.remove(path)
                                        console.print(f"  [system]✓ Deleted file:[/] {path}")
                                        found = True
                                        break
                                    except Exception as e:
                                        console.print(f"  [error]Failed to delete file: {e}[/]")
                            if not found:
                                console.print(f"  [info]FileNotFound:[/] Metadata suggested '{name}', but file not in docs, scraped-docs, or reddit-json.")
                        
                    else:
                        console.print("  [error]Invalid number.[/]")
                else:
                    console.print("  [info]Cancelled.[/]")

        # ── /change-model ─────────────────────────────────────────────────
        elif cmd == "/change-model":
            model = choose_model()
            messages.clear()
            console.print("  [system]✓ Model changed. History cleared.[/]")
            console.rule(style="info")

        # ── /change-docs ──────────────────────────────────────────────────
        elif cmd == "/change-docs":
            selected_paths = choose_docs()
            console.print("\n  [system]Rebuilding vector database…[/]")
            collection, doc_chunk_counts = build_vector_db(selected_paths)
            chunk_offset = sum(doc_chunk_counts.values())
            messages.clear()
            last_chunks.clear()
            doc_names = ", ".join(os.path.basename(p) for p in selected_paths)
            console.print("  [system]✓ Docs changed. History cleared.[/]")
            console.rule(style="info")

        # ── Normal question ───────────────────────────────────────────────
        else:
            console.rule(style="info")
            answer, last_chunks = run_query(collection, model, messages, question)
            last_question = question
            console.rule(style="info")

            messages.append({"role": "user",      "content": question})
            messages.append({"role": "assistant", "content": answer})
            if len(messages) > 20:
                messages = messages[-20:]

            # ── Auto-save every N turns ──────────────────────────────────
            autosave_turn_count += 1
            if autosave_turn_count % AUTOSAVE_EVERY == 0:
                try:
                    os.makedirs(CHATS_FOLDER, exist_ok=True)
                    json_path = os.path.join(CHATS_FOLDER, f"{autosave_stem}.json")
                    payload   = {"timestamp": autosave_stem, "model": model,
                                 "docs": doc_names, "messages": messages}
                    with open(json_path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2, ensure_ascii=False)
                    console.print(f"  [info]💾 Auto-saved ({autosave_turn_count} turns)[/]")
                except Exception:
                    pass   # never let autosave crash the session
