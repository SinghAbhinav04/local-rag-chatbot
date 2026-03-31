"""
Agent Memory — persistent memory system inspired by PicoClaw's MemoryStore.

Long-term memory:  ~/.rag-agent/memory/MEMORY.md
Daily notes:       ~/.rag-agent/memory/YYYYMM/YYYYMMDD.md
"""

import os
import datetime

MEMORY_BASE = os.path.expanduser("~/.rag-agent/memory")
MEMORY_FILE = os.path.join(MEMORY_BASE, "MEMORY.md")


def _ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)


# ── Long-term memory ─────────────────────────────────────────────────────────

def read_long_term() -> str:
    """Read the long-term memory file MEMORY.md."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return f.read()
    return ""


def write_long_term(content: str):
    """Overwrite the long-term memory file."""
    _ensure_dir(MEMORY_FILE)
    with open(MEMORY_FILE, "w") as f:
        f.write(content)


def append_long_term(content: str):
    """Append to the long-term memory file."""
    _ensure_dir(MEMORY_FILE)
    existing = read_long_term()
    with open(MEMORY_FILE, "w") as f:
        if existing:
            f.write(existing.rstrip("\n") + "\n\n" + content + "\n")
        else:
            f.write(content + "\n")


# ── Daily notes ───────────────────────────────────────────────────────────────

def _today_path() -> str:
    now = datetime.datetime.now()
    month_dir = now.strftime("%Y%m")
    day_file = now.strftime("%Y%m%d") + ".md"
    return os.path.join(MEMORY_BASE, month_dir, day_file)


def read_today() -> str:
    """Read today's daily note."""
    path = _today_path()
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read()
    return ""


def append_today(content: str):
    """Append to today's daily note. Creates the file with a date header if new."""
    path = _today_path()
    _ensure_dir(path)

    existing = ""
    if os.path.exists(path):
        with open(path, "r") as f:
            existing = f.read()

    with open(path, "w") as f:
        if not existing:
            header = f"# {datetime.datetime.now().strftime('%Y-%m-%d')}\n\n"
            f.write(header + content + "\n")
        else:
            f.write(existing.rstrip("\n") + "\n\n" + content + "\n")


def get_recent_notes(days: int = 3) -> str:
    """Return daily notes from the last N days, joined with separators."""
    parts = []
    for i in range(days):
        date = datetime.datetime.now() - datetime.timedelta(days=i)
        month_dir = date.strftime("%Y%m")
        day_file = date.strftime("%Y%m%d") + ".md"
        path = os.path.join(MEMORY_BASE, month_dir, day_file)
        if os.path.exists(path):
            with open(path, "r") as f:
                parts.append(f.read())
    return "\n\n---\n\n".join(parts)


# ── Context injection ─────────────────────────────────────────────────────────

def get_memory_context() -> str:
    """Build a memory context block for the system prompt."""
    long_term = read_long_term()
    recent = get_recent_notes(3)

    if not long_term and not recent:
        return ""

    parts = []
    if long_term:
        parts.append("## Long-term Memory\n\n" + long_term)
    if recent:
        parts.append("## Recent Daily Notes\n\n" + recent)
    return "\n\n---\n\n".join(parts)
