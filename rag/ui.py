"""
Interactive UI helpers — model selector, document picker,
and the /help menu.
"""

import os

from rich.table import Table
from rich import box

from rag.config  import (get_available_models, DOC_FOLDER, SCRAPED_DOCS_FOLDER,
                         REDDIT_JSON_FOLDER, SCRAPED_DATA_FOLDER,
                         X_SCRAPED_FOLDER, NEWS_DATA_FOLDER,
                         WIKI_DATA_FOLDER)
from rag.console import console

# Ordered list of (label, folder_path) pairs scanned by the doc picker.
_DATA_SOURCES = [
    ("docs",         DOC_FOLDER),
    ("scraped-docs", SCRAPED_DOCS_FOLDER),
    ("reddit-json",  REDDIT_JSON_FOLDER),
    ("scrapped-data",SCRAPED_DATA_FOLDER),
    ("x-scraped",    X_SCRAPED_FOLDER),
    ("news-data",    NEWS_DATA_FOLDER),
    ("wiki-data",    WIKI_DATA_FOLDER),
]


def choose_model() -> str:
    """Fetch models from Ollama (≥300 MB), display a menu, return chosen name."""
    models = get_available_models()

    if not models:
        console.print("  [error]✗ No qualifying models found (need ≥300 MB via `ollama list`).[/]")
        console.print("  [info]Install a model with: ollama pull <model-name>[/]")
        exit(1)

    table = Table(box=box.ROUNDED, show_header=True, header_style="system",
                  title="[system]Available Models (≥300 MB)[/]")
    table.add_column("#",    style="cmd",    width=4)
    table.add_column("Model", style="info")
    table.add_column("Size",  style="info",  justify="right")

    for i, m in enumerate(models, 1):
        table.add_row(str(i), m["name"], m["size_label"])
    console.print(table)

    choice = input("  Enter model number: ").strip()
    if not choice.isdigit() or int(choice) < 1 or int(choice) > len(models):
        console.print("  [error]Invalid choice. Defaulting to 1.[/]")
        choice = "1"

    model = models[int(choice) - 1]["name"]
    console.print(f"  [system]✓ Using model:[/] {model}\n")
    return model


def _collect_all_files() -> list[tuple[str, str, str]]:
    """Scan all data folders. Returns list of (source_label, filename, abs_path)."""
    entries: list[tuple[str, str, str]] = []
    for label, folder in _DATA_SOURCES:
        if not os.path.isdir(folder):
            continue

        if label in ("news-data", "wiki-data"):
            # These folders have sub-folders; pick files inside each
            prefix = "news" if label == "news-data" else "wiki"
            for sub in sorted(os.listdir(folder)):
                sub_path = os.path.join(folder, sub)
                if os.path.isdir(sub_path):
                    for fname in sorted(os.listdir(sub_path)):
                        fpath = os.path.join(sub_path, fname)
                        if os.path.isfile(fpath) and not fname.startswith("."):
                            entries.append((f"{prefix}/{sub}", fname, fpath))
        else:
            for fname in sorted(os.listdir(folder)):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and not fname.startswith("."):
                    entries.append((label, fname, fpath))
    return entries


def choose_docs() -> list[str]:
    """Display the document picker across all data sources and return selected paths."""
    all_files = _collect_all_files()

    if not all_files:
        console.print("  [error]✗ No files found in any data folder.[/]")
        exit(1)

    table = Table(box=box.ROUNDED, show_header=True, header_style="system")
    table.add_column("#",      style="cmd",  width=4)
    table.add_column("Source", style="system", width=18)
    table.add_column("File",   style="info")
    table.add_column("Size",   style="info", justify="right")

    prev_source = None
    for i, (source, fname, fpath) in enumerate(all_files, 1):
        size_kb = os.path.getsize(fpath) // 1024
        # Show source label only on first file of each group for cleaner look
        display_source = source if source != prev_source else ""
        table.add_row(str(i), display_source, fname, f"{size_kb} KB")
        prev_source = source
    console.print(table)

    console.print("  [info]Enter numbers (spaces/commas), 'all', or 'skip / none' for no context[/]")
    raw = input("  Your selection: ").strip().lower()

    # Flexible matching for skip keywords
    if raw in ["0", "skip", "none", "no", "0/skip"]:
        console.print("  [system]✓ Entering 'Normal Chat' mode (no document context).[/]")
        return []

    if raw == "all":
        selected_indices = list(range(len(all_files)))
    else:
        tokens = raw.replace(",", " ").split()
        selected_indices = []
        for t in tokens:
            if t.isdigit():
                idx = int(t) - 1
                if 0 <= idx < len(all_files):
                    selected_indices.append(idx)
                else:
                    console.print(f"  [error]⚠ Invalid number: {t}[/]")
            else:
                console.print(f"  [error]⚠ Unrecognized token: {t}[/]")

    if not selected_indices:
        console.print("  [error]No valid selection. Loading first doc by default.[/]")
        selected_indices = [0]

    seen = set()
    unique_indices = [x for x in selected_indices if not (x in seen or seen.add(x))]
    selected_files = [all_files[i][2] for i in unique_indices]

    console.print("  [system]✓ Selected:[/]")
    for i in unique_indices:
        source, fname, _ = all_files[i]
        console.print(f"     [info]• [{source}] {fname}[/]")

    return selected_files


# ──────────────────────────────────────────
# Help Menu
# ──────────────────────────────────────────

def print_help():
    """Render the full /help command reference."""
    def section(title: str, rows: list[tuple[str, str]]) -> Table:
        t = Table(box=box.SIMPLE, show_header=False, padding=(0, 2), expand=True)
        t.add_column("Command",     style="cmd",  no_wrap=True, width=26)
        t.add_column("Description", style="info")
        for cmd, desc in rows:
            t.add_row(cmd, desc)
        return t

    console.print()
    console.rule("[system]  Commands  [/]", style="system")

    # ── General ──────────────────────────────────────────────────────────
    console.print("\n [bold white]General[/]")
    console.print(section("general", [
        ("/help",               "Show this menu"),
        ("/status",             "Dashboard — model, docs, chunks, voice, turns"),
        ("exit",                "Quit the program"),
    ]))

    # ── Conversation ─────────────────────────────────────────────────────
    console.print(" [bold white]Conversation[/]")
    console.print(section("conversation", [
        ("/history",            "Print the full conversation so far"),
        ("/search-history <kw>","Search & highlight turns containing a keyword"),
        ("/summarize",          "Ask the model to summarize the conversation"),
        ("/clear",              "Wipe history (keep model & docs)"),
        ("/retry",              "Regenerate the last AI response"),
        ("/why",                "Show the source chunks used for the last answer"),
    ]))

    # ── Save / Load ───────────────────────────────────────────────────────
    console.print(" [bold white]Save / Load[/]")
    console.print(section("save", [
        ("/save-chat [name]",   "Save conversation to chats/  (.md + .json)"),
        ("/load-chat",          "Resume a saved conversation from chats/"),
        ("/export-pdf [name]",  "Export conversation as a formatted PDF to exports/"),
    ]))

    # ── Documents & Knowledge ─────────────────────────────────────────────
    console.print(" [bold white]Documents & Knowledge[/]")
    console.print(section("docs", [
        ("/list-docs",          "Show loaded docs and their chunk counts"),
        ("/add-doc",            "Inject a new doc into the session (no rebuild)"),
        ("/add-url <url>",      "Scrape & index a webpage for RAG context"),
        ("/scrape <url>",       "Scrape to standalone file (no RAG index)"),
        ("/get-news",           "Fetch news from NewsAPI & embed for RAG chat"),
        ("/wiki <topic>",       "Fetch Wikipedia article & embed for RAG chat"),
        ("/weather <city>",     "Fetch weather overview and embed for RAG chat (in-memory)"),
        ("/remove-doc",         "Delete a doc from memory (and optionally disk)"),
        ("/change-docs",        "Swap out all docs and rebuild the vector DB"),
    ]))

    # ── Agent Web UI ──────────────────────────────────────────────────────
    console.print(" [bold white]Agent Web UI[/]")
    console.print(section("agent", [
        ("/agent start",        "Launch browser-based PicoClaw UI (runs alongside CLI)"),
        ("/agent stop",         "Shut down the web UI server"),
    ]))

    # ── Model & Voice ─────────────────────────────────────────────────────
    console.print(" [bold white]Model & Voice[/]")
    console.print(section("model", [
        ("/change-model",       "Switch to a different LLM"),
        ("/voice",              "Toggle TTS on / off"),
    ]))

    console.rule(style="info")
    console.print()
