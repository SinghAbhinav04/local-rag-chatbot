"""
Wikipedia quick-fetch — pulls a Wikipedia article by topic via the
MediaWiki API, saves it to wiki-data/, and injects it into the
active vector DB session for RAG chat.

Uses `requests` (already a project dependency) so no new packages needed.
"""

import os
import json
import datetime

import requests
from bs4 import BeautifulSoup

from rag.config   import WIKI_DATA_FOLDER
from rag.console  import console
from rag.chunking import chunk_text
from rag.vectordb import get_embedding


# ──────────────────────────────────────────
# MediaWiki API helpers
# ──────────────────────────────────────────

_API_URL = "https://en.wikipedia.org/w/api.php"

# Wikipedia requires a proper User-Agent or it returns 403.
_session = requests.Session()
_session.headers.update({
    "User-Agent": "LocalRAGChatBot/1.0 (https://github.com; educational project)"
})


def _search_title(topic: str) -> str | None:
    """Search Wikipedia and return the best-matching page title, or None."""
    resp = _session.get(_API_URL, params={
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "srlimit": 1,
        "format": "json",
    }, timeout=15)
    resp.raise_for_status()
    results = resp.json().get("query", {}).get("search", [])
    return results[0]["title"] if results else None


def _fetch_page_html(title: str) -> str:
    """Fetch the parsed HTML content of a Wikipedia page."""
    resp = _session.get(_API_URL, params={
        "action": "parse",
        "page": title,
        "prop": "text",
        "format": "json",
    }, timeout=30)
    resp.raise_for_status()
    return resp.json()["parse"]["text"]["*"]


def _html_to_text(html: str) -> str:
    """Strip HTML tags, keep readable plain text."""
    soup = BeautifulSoup(html, "html.parser")
    # Remove style/script/reference tags
    for tag in soup(["style", "script", "sup", "table"]):
        tag.decompose()
    return soup.get_text(separator="\n", strip=True)


def _get_search_suggestions(topic: str) -> list[str]:
    """Fetch top 5 search suggestions for a given topic."""
    try:
        resp = _session.get(_API_URL, params={
            "action": "opensearch",
            "search": topic,
            "limit": 5,
            "namespace": 0,
            "format": "json",
        }, timeout=10)
        resp.raise_for_status()
        # opensearch returns [search_term, [title1, title2...], [desc1, desc2...], [url1, url2...]]
        return resp.json()[1]
    except Exception:
        return []

# ──────────────────────────────────────────
# Public API
# ──────────────────────────────────────────

def fetch_wiki(topic: str) -> tuple[str, str, str]:
    """
    Search & fetch a Wikipedia article.
    Returns (folder_path, plain_text, page_title).
    Raises RuntimeError if no article found.
    """
    console.print(f"  [system]Searching Wikipedia for '{topic}'…[/]")
    title = _search_title(topic)
    if not title:
        raise RuntimeError(f"No Wikipedia article found for '{topic}'.")

    console.print(f"  [system]Fetching '{title}'…[/]")
    html = _fetch_page_html(title)
    text = _html_to_text(html)

    # Save to wiki-data/<timestamp>-<safe_topic>/
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = title.replace(" ", "_").replace("/", "_")[:60]
    folder = os.path.join(WIKI_DATA_FOLDER, f"{ts}-{safe_topic}")
    os.makedirs(folder, exist_ok=True)

    # Save raw text
    txt_path = os.path.join(folder, f"{safe_topic}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Wikipedia: {title}\n{'=' * 60}\n\n{text}")

    # Save metadata
    meta = {"title": title, "topic_query": topic, "fetched_at": ts,
            "url": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}
    with open(os.path.join(folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    console.print(f"  [system]✓ '{title}' saved → {folder}[/]")

    # Check for disambiguation
    if "may refer to:" in text[:500] or "disambiguation" in text[-500:].lower():
        suggestions = _get_search_suggestions(topic)
        sug_text = ""
        if suggestions:
            sug_text = "\n  [info]Here are some specific articles you might have meant:[/]\n"
            for s in suggestions:
                if s.lower() != title.lower():  # Don't suggest the disambiguation page itself
                    sug_text += f"    • /wiki {s}\n"

        console.print(
            "  [error]⚠ Warning: This looks like a disambiguation page (a list of different meanings).[/]\n"
            f"  [info]To get a specific article, try a more exact term.[/]{sug_text}"
        )

    return folder, text, title


def wiki_to_rag(collection, text: str, source_label: str,
                doc_chunk_counts: dict[str, int],
                chunk_offset: int) -> int:
    """Chunk and embed wiki text into the active collection. Returns updated offset."""
    chunks = chunk_text(text, source_label)
    if not chunks:
        console.print("  [info]No text to embed.[/]")
        return chunk_offset

    console.print(f"  [system]Embedding {len(chunks)} wiki chunks…[/]")

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
