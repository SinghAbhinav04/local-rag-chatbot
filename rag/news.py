"""
News fetcher — pulls articles from NewsAPI (top-headlines, everything,
sources), saves them to dated folders, and injects them into the
active vector DB session for RAG chat.
"""

import os
import json
import datetime

from newsapi import NewsApiClient

from rag.config   import NEWS_API_KEY, NEWS_DATA_FOLDER
from rag.console  import console
from rag.chunking import chunk_text
from rag.vectordb import get_embedding


# ──────────────────────────────────────────
# Initialise client
# ──────────────────────────────────────────

_client = NewsApiClient(api_key=NEWS_API_KEY)


# ──────────────────────────────────────────
# Folder helpers
# ──────────────────────────────────────────

def _make_folder(endpoint: str) -> str:
    """Create and return a dated folder like `news-data/20260315_020300-top_headlines/`."""
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{ts}-{endpoint}"
    path = os.path.join(NEWS_DATA_FOLDER, name)
    os.makedirs(path, exist_ok=True)
    return path


def _save_json(folder: str, data: dict, filename: str = "raw.json"):
    """Dump raw API response to JSON."""
    with open(os.path.join(folder, filename), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _save_text(folder: str, text: str, filename: str = "articles.txt"):
    """Write the formatted plain-text representation."""
    path = os.path.join(folder, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


# ──────────────────────────────────────────
# Formatters
# ──────────────────────────────────────────

def _format_articles(articles: list[dict]) -> str:
    """Convert a list of article dicts into readable plain text."""
    parts = []
    for i, art in enumerate(articles, 1):
        title       = art.get("title")       or "No title"
        author      = art.get("author")      or "Unknown"
        source_name = (art.get("source") or {}).get("name", "Unknown")
        published   = art.get("publishedAt") or ""
        description = art.get("description") or ""
        content     = art.get("content")     or ""
        url         = art.get("url")         or ""

        block = (
            f"=== Article {i} ===\n"
            f"Title: {title}\n"
            f"Author: {author}\n"
            f"Source: {source_name}\n"
            f"Published: {published}\n"
            f"URL: {url}\n"
            f"Description: {description}\n"
            f"Content: {content}\n"
        )
        parts.append(block)
    return "\n\n".join(parts)


def _format_sources(sources: list[dict]) -> str:
    """Convert a list of source dicts into readable plain text."""
    parts = []
    for i, src in enumerate(sources, 1):
        block = (
            f"=== Source {i} ===\n"
            f"Name: {src.get('name', '')}\n"
            f"Description: {src.get('description', '')}\n"
            f"URL: {src.get('url', '')}\n"
            f"Category: {src.get('category', '')}\n"
            f"Language: {src.get('language', '')}\n"
            f"Country: {src.get('country', '')}\n"
        )
        parts.append(block)
    return "\n\n".join(parts)


# ──────────────────────────────────────────
# Fetch endpoints
# ──────────────────────────────────────────

def fetch_top_headlines(query: str) -> tuple[str, str, int]:
    """Fetch top headlines. Returns (folder_path, formatted_text, article_count)."""
    console.print(f"  [system]Fetching top headlines for '{query}'…[/]")
    resp     = _client.get_top_headlines(q=query, language="en")
    articles = resp.get("articles", [])
    folder   = _make_folder("top_headlines")

    _save_json(folder, resp)
    text = _format_articles(articles)
    _save_text(folder, text)

    console.print(f"  [system]✓ {len(articles)} articles saved → {folder}[/]")
    return folder, text, len(articles)


def fetch_everything(query: str) -> tuple[str, str, int]:
    """Fetch everything. Returns (folder_path, formatted_text, article_count)."""
    console.print(f"  [system]Fetching all articles for '{query}'…[/]")
    resp     = _client.get_everything(q=query, language="en", sort_by="publishedAt")
    articles = resp.get("articles", [])
    folder   = _make_folder("everything")

    _save_json(folder, resp)
    text = _format_articles(articles)
    _save_text(folder, text)

    console.print(f"  [system]✓ {len(articles)} articles saved → {folder}[/]")
    return folder, text, len(articles)


def fetch_sources() -> tuple[str, str, int]:
    """Fetch all sources. Returns (folder_path, formatted_text, source_count)."""
    console.print("  [system]Fetching news sources…[/]")
    resp    = _client.get_sources()
    sources = resp.get("sources", [])
    folder  = _make_folder("sources")

    _save_json(folder, resp)
    text = _format_sources(sources)
    _save_text(folder, text, "sources.txt")

    console.print(f"  [system]✓ {len(sources)} sources saved → {folder}[/]")
    return folder, text, len(sources)


# ──────────────────────────────────────────
# Embed into vector DB
# ──────────────────────────────────────────

def news_to_rag(collection, text: str, source_label: str,
                doc_chunk_counts: dict[str, int],
                chunk_offset: int) -> int:
    """Chunk and embed news text into the active collection. Returns updated offset."""
    chunks = chunk_text(text, source_label)
    if not chunks:
        console.print("  [info]No text to embed.[/]")
        return chunk_offset

    console.print(f"  [system]Embedding {len(chunks)} news chunks…[/]")

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        cid = f"chunk_{chunk_offset + i}"
        emb = get_embedding(chunk["text"])
        ids.append(cid)
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"source": source_label})

    collection.add(ids=ids, embeddings=embeddings,
                   documents=documents, metadatas=metadatas)
    doc_chunk_counts[source_label] = len(chunks)
    console.print(f"  [system]✓ {len(chunks)} chunks indexed as '{source_label}'.[/]")
    return chunk_offset + len(chunks)
