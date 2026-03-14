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

MODELS = {
    "1": "sadiq-bd/llama3.2-1b-uncensored:latest",
    "2": "IHA089/drana-infinity-0.5b:0.5b",
    "3": "sadiq-bd/llama3.2-3b-uncensored:latest"
}

EMBED_MODEL = "nomic-embed-text"
