"""
Interactive UI helpers — model selector, document picker,
and the /help menu.
"""

import os

from rich.panel import Panel
from rich.table import Table
from rich import box

from rag.config  import MODELS, DOC_FOLDER
from rag.console import console


def choose_model() -> str:
    """Display the model menu and return the chosen model identifier."""
    console.print(Panel(
        "[cmd]1[/] → llama3.2 1B uncensored  [info](fast)[/]\n"
        "[cmd]2[/] → drana infinity 0.5B     [info](very fast)[/]\n"
        "[cmd]3[/] → llama3.2 3B uncensored  [info](smarter)[/]",
        title="[system]Choose Model[/]", box=box.ROUNDED
    ))
    choice = input("  Enter model number: ").strip()
    if choice not in MODELS:
        console.print("  [error]Invalid choice. Defaulting to 1.[/]")
        choice = "1"
    model = MODELS[choice]
    console.print(f"  [system]✓ Using model:[/] {model}\n")
    return model


def choose_docs() -> list[str]:
    """Display the document picker and return a list of selected file paths."""
    if not os.path.isdir(DOC_FOLDER):
        console.print(f"  [error]✗ Folder '{DOC_FOLDER}' not found.[/]")
        exit(1)

    files = sorted([
        f for f in os.listdir(DOC_FOLDER)
        if os.path.isfile(os.path.join(DOC_FOLDER, f)) and not f.startswith(".")
    ])

    if not files:
        console.print(f"  [error]✗ No files found in '{DOC_FOLDER}/'.[/]")
        exit(1)

    table = Table(box=box.ROUNDED, show_header=True, header_style="system")
    table.add_column("#",    style="cmd",  width=4)
    table.add_column("File", style="info")
    table.add_column("Size", style="info", justify="right")
    for i, f in enumerate(files, 1):
        size_kb = os.path.getsize(os.path.join(DOC_FOLDER, f)) // 1024
        table.add_row(str(i), f, f"{size_kb} KB")
    console.print(table)

    console.print("  [info]Enter numbers (spaces/commas), 'all', or 'skip / none' for no context[/]")
    raw = input("  Your selection: ").strip().lower()

    # Flexible matching for skip keywords
    if raw in ["0", "skip", "none", "no", "0/skip"]:
        console.print("  [system]✓ Entering 'Normal Chat' mode (no document context).[/]")
        return []

    if raw == "all":
        selected_indices = list(range(len(files)))
    else:
        tokens = raw.replace(",", " ").split()
        selected_indices = []
        for t in tokens:
            if t.isdigit():
                idx = int(t) - 1
                if 0 <= idx < len(files):
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
    selected_files = [os.path.join(DOC_FOLDER, files[i]) for i in unique_indices]

    console.print("  [system]✓ Selected:[/]")
    for f in selected_files:
        console.print(f"     [info]• {os.path.basename(f)}[/]")

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
        ("/add-url <url>",      "Scrape & index a webpage on the spot"),
        ("/remove-doc",         "Delete a doc from memory (and optionally disk)"),
        ("/change-docs",        "Swap out all docs and rebuild the vector DB"),
    ]))

    # ── Model & Voice ─────────────────────────────────────────────────────
    console.print(" [bold white]Model & Voice[/]")
    console.print(section("model", [
        ("/change-model",       "Switch to a different LLM"),
        ("/voice",              "Toggle TTS on / off"),
    ]))

    console.rule(style="info")
    console.print()
