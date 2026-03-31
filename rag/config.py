"""
Configuration constants for the RAG system.
All tunables (folders, chunk sizes, model list) live here.
"""

# ──────────────────────────────────────────
# Paths
# ──────────────────────────────────────────

DOC_FOLDER     = "docs"
CHATS_FOLDER   = "chats"
EXPORTS_FOLDER = "exports"
SCRAPED_DOCS_FOLDER = "scraped-docs"
REDDIT_JSON_FOLDER = "reddit-json"
SCRAPED_DATA_FOLDER = "scrapped-data"
X_SCRAPED_FOLDER = "x-scraped"
INSTA_SCRAPED_FOLDER = "insta-scraped"
NEWS_DATA_FOLDER = "news-data"
WIKI_DATA_FOLDER = "wiki-data"

# ──────────────────────────────────────────
# API Keys
# ──────────────────────────────────────────

NEWS_API_KEY   = "3e8b8475002c4f0db03e532bd0e0b721"
WEATHER_API_KEY= "6c53531b9422ef9aed24db06fbf90209"

# ──────────────────────────────────────────
# Chunking
# ──────────────────────────────────────────

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50
TOP_K         = 5

# ──────────────────────────────────────────
# Session
# ──────────────────────────────────────────

AUTOSAVE_EVERY = 5        # silently save every N turns

# ──────────────────────────────────────────
# Models
# ──────────────────────────────────────────

MIN_MODEL_SIZE_MB = 300

EMBED_MODEL = "nomic-embed-text"


def get_available_models() -> list[dict]:
    """
    Query `ollama list` and return models >= MIN_MODEL_SIZE_MB.
    Each entry: {"name": str, "size_mb": float, "size_label": str}
    Sorted by size ascending.
    """
    import subprocess, re

    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []

    models = []
    for line in result.stdout.strip().splitlines()[1:]:      # skip header
        parts = line.split()
        if len(parts) < 3:
            continue

        name = parts[0]

        # find the size token, e.g. "1.3", and unit token, e.g. "GB" / "MB"
        size_mb = None
        for i, tok in enumerate(parts[1:], start=1):
            match = re.match(r"^(\d+(?:\.\d+)?)$", tok)
            if match and i + 1 < len(parts):
                unit = parts[i + 1].upper()
                val = float(match.group(1))
                if unit == "GB":
                    size_mb = val * 1024
                elif unit == "MB":
                    size_mb = val
                break

        if size_mb is None or size_mb < MIN_MODEL_SIZE_MB:
            continue
        if name == EMBED_MODEL or name.startswith(EMBED_MODEL + ":"):
            continue

        if size_mb >= 1024:
            label = f"{size_mb / 1024:.1f} GB"
        else:
            label = f"{size_mb:.0f} MB"

        models.append({"name": name, "size_mb": size_mb, "size_label": label})

    models.sort(key=lambda m: m["size_mb"])
    return models
