"""
Web scraper — fetch a URL, strip boilerplate, chunk and embed
the content into an existing ChromaDB collection.
"""

import re
import urllib.parse
import os
import json

from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup
import chromadb

from rag.config   import SCRAPED_DOCS_FOLDER, REDDIT_JSON_FOLDER
from rag.console  import console
from rag.chunking import chunk_text
from rag.vectordb import get_embedding
from rag.loaders  import load_file


def parse_reddit_json(data) -> str:
    """
    Recursively extract post text and comments from Reddit's listing JSON.
    """
    out = []
    
    # Reddit JSON is usually a list: [post_listing, comments_listing]
    if isinstance(data, list):
        for item in data:
            out.append(parse_reddit_json(item))
        return "\n\n".join(filter(None, out))

    # If it's a listing, look at children
    if isinstance(data, dict):
        kind = data.get("kind")
        obj_data = data.get("data", {})
        
        # Post (t3) or Comment (t1)
        if kind in ["t1", "t3"]:
            title = obj_data.get("title", "")
            body  = obj_data.get("selftext") or obj_data.get("body", "")
            author = obj_data.get("author", "[unknown]")
            
            content = []
            if title: content.append(f"Post Title: {title}")
            if author: content.append(f"Author: {author}")
            if body: content.append(body)
            out.append("\n".join(content))
            
            # Handle replies
            replies = obj_data.get("replies")
            if replies:
                out.append(parse_reddit_json(replies))
                
        # Listing or other container
        children = obj_data.get("children")
        if children:
            for child in children:
                out.append(parse_reddit_json(child))
                
    return "\n\n".join(filter(None, out))


def scrape_url(url: str) -> tuple[str, str, list[str]]:
    """
    Fetch a URL using curl_cffi impersonating chrome (or safari for reddit), 
    strip boilerplate, return (clean_text, label, downloaded_docs).
    Label is the domain, e.g. 'en.wikipedia.org'.
    """
    parsed = urllib.parse.urlparse(url)
    is_reddit = "reddit.com" in parsed.netloc
    impersonate_target = "chrome"
    
    if is_reddit:
        impersonate_target = "safari15_5"
        if not url.endswith(".json"):
            # If it's just the domain or domain/, ensure we join /.json correctly
            if not parsed.path or parsed.path == "/":
                url = urllib.parse.urljoin(url, "/.json")
            else:
                url = url.rstrip("/") + ".json"
            console.print(f"  [system]Reddit Optimization:[/] Switching to .json endpoint for full thread history.")

    # Use curl_cffi to perfectly impersonate browser TLS fingerprints
    resp = cffi_requests.get(url, impersonate=impersonate_target, timeout=15)
    resp.raise_for_status()

    # Handle Reddit JSON specifically
    if is_reddit and (url.endswith(".json") or "application/json" in resp.headers.get("Content-Type", "")):
        try:
            os.makedirs(REDDIT_JSON_FOLDER, exist_ok=True)
            
            # Create a safe filename for the JSON cache based on the URL path
            name_part = parsed.path.strip("/").replace("/", "_") or "homepage"
            json_path = os.path.join(REDDIT_JSON_FOLDER, f"{name_part}.json")
            
            # 1. Download/Save the raw JSON first as requested
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
            
            label = parsed.netloc or "reddit.com"
            console.print(f"  [system]Reddit Saved:[/] Raw JSON stored at {json_path}")
            # For Reddit, we return empty text here because we want to chunk specifically from the FILE
            # in the add_url_to_db loop below.
            return "", label, [json_path]
        except Exception as e:
            console.print(f"  [error]Failed to parse Reddit JSON: {e}. Falling back to HTML scraping.[/]")

    # Standard HTML scraping
    soup = BeautifulSoup(resp.text, "html.parser")

    # Process links for documents
    downloaded_docs = []
    supported_exts = {".pdf", ".docx", ".txt", ".csv", ".md"}
    
    os.makedirs(SCRAPED_DOCS_FOLDER, exist_ok=True)
    
    for a in soup.find_all("a", href=True):
        href = a["href"]
        parsed_href = urllib.parse.urlparse(href)
        ext = os.path.splitext(parsed_href.path)[1].lower()
        if ext in supported_exts:
            # Resolve absolute URL
            abs_url = urllib.parse.urljoin(url, href)
            # Create a safe filename
            filename = os.path.basename(parsed_href.path)
            if not filename:
                continue
            
            save_path = os.path.join(SCRAPED_DOCS_FOLDER, filename)
            
            # Simple deduplication: if file already exists, we skip downloading
            if os.path.exists(save_path):
                if save_path not in downloaded_docs:
                    downloaded_docs.append(save_path)
                continue
                
            try:
                console.print(f"  [system]Downloading:[/] {abs_url}")
                doc_resp = cffi_requests.get(abs_url, impersonate="chrome", timeout=15)
                doc_resp.raise_for_status()
                with open(save_path, "wb") as f:
                    f.write(doc_resp.content)
                downloaded_docs.append(save_path)
            except Exception as e:
                console.print(f"  [error]Failed to download '{abs_url}': {e}[/]")

    # Remove navigation, ads, scripts, styles for standard HTML
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "form", "noscript", "iframe"]):
        tag.decompose()

    text  = soup.get_text(separator="\n")
    # Collapse excessive blank lines
    text  = re.sub(r"\n{3,}", "\n\n", text).strip()
    label = parsed.netloc or url
    return text, label, downloaded_docs


def add_url_to_db(
    collection: chromadb.Collection,
    url: str,
    doc_chunk_counts: dict[str, int],
    chunk_offset: int,
) -> int:
    """Scrape a URL, fetch linked documents, and embed their contents. Returns updated chunk offset."""
    console.print(f"  [system]Fetching:[/] {url}")
    text, label, downloaded_docs = scrape_url(url)
    
    # Only index in-memory text if it exists (standard HTML)
    if text.strip():
        chunks = chunk_text(text, label)
        console.print(f"  [system]Embedding {len(chunks)} chunks from '{label}' ({len(text)} chars)…[/]")
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
        chunk_offset += len(chunks)
    
    # Process downloaded documents (PDFs, DOCX, and Reddit JSON)
    for doc_path in downloaded_docs:
        name = os.path.basename(doc_path)
        try:
            # Special Handling for Reddit JSON: Parse and index specifically from the FILE
            if doc_path.endswith(".json") and REDDIT_JSON_FOLDER in doc_path:
                with open(doc_path, "r", encoding="utf-8") as f:
                    raw_data = json.load(f)
                doc_text = parse_reddit_json(raw_data)
            else:
                doc_text = load_file(doc_path)
            
            doc_chunks = chunk_text(doc_text, name)
            
            if not doc_chunks:
                continue
                
            console.print(f"  [system]Embedding {len(doc_chunks)} chunks from file '{name}' ({len(doc_text)} chars)…[/]")
            doc_ids, doc_embeddings, doc_documents, doc_metadatas = [], [], [], []
            for i, chunk in enumerate(doc_chunks):
                cid = f"chunk_{chunk_offset + i}"
                emb = get_embedding(chunk["text"])
                doc_ids.append(cid)
                doc_embeddings.append(emb)
                doc_documents.append(chunk["text"])
                doc_metadatas.append({"source": name})
                
            collection.add(ids=doc_ids, embeddings=doc_embeddings, documents=doc_documents, metadatas=doc_metadatas)
            doc_chunk_counts[name] = len(doc_chunks)
            console.print(f"  [system]✓ '{name}' indexed — {len(doc_chunks)} chunks.[/]")
            chunk_offset += len(doc_chunks)
        except Exception as e:
            console.print(f"  [error]Failed to process downloaded doc '{name}': {e}[/]")

    return chunk_offset
