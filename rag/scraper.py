"""
Web scraper — fetch a URL, strip boilerplate, chunk and embed
the content into an existing ChromaDB collection.
"""

import re
import urllib.parse

import requests
from bs4 import BeautifulSoup
import chromadb

from rag.console  import console
from rag.chunking import chunk_text
from rag.vectordb import get_embedding


def scrape_url(url: str) -> tuple[str, str]:
    """
    Fetch a URL, strip boilerplate, return (clean_text, label).
    Label is the domain, e.g. 'en.wikipedia.org'.
    """
    headers = {"User-Agent": "Mozilla/5.0 (compatible; RAG-bot/1.0)"}
    resp    = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove navigation, ads, scripts, styles
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    text  = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    text  = re.sub(r"\n{3,}", "\n\n", text).strip()
    label = urllib.parse.urlparse(url).netloc or url
    return text, label


def add_url_to_db(
    collection: chromadb.Collection,
    url: str,
    doc_chunk_counts: dict[str, int],
    chunk_offset: int,
) -> int:
    """Scrape a URL and embed its content. Returns updated chunk offset."""
    console.print(f"  [system]Fetching:[/] {url}")
    text, label = scrape_url(url)
    chunks      = chunk_text(text, label)

    console.print(f"  [system]Embedding {len(chunks)} chunks from '{label}'…[/]")

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        cid = f"chunk_{chunk_offset + i}"
        emb = get_embedding(chunk["text"])
        ids.append(cid)
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"source": label})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    doc_chunk_counts[label] = len(chunks)
    console.print(f"  [system]✓ '{label}' indexed — {len(chunks)} chunks.[/]")
    return chunk_offset + len(chunks)
