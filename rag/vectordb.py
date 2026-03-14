"""
Vector database layer — ChromaDB backed by Ollama embeddings.

Provides:
  - get_embedding()   — embed a single text via Ollama
  - build_vector_db() — load multiple docs, chunk, embed, index
  - add_doc_to_db()   — hot-add one doc to an existing collection
  - retrieve_raw()    — top-K cosine retrieval
"""

import os
import hashlib

import ollama
import chromadb
from chromadb.config import Settings

from rag.config   import EMBED_MODEL, TOP_K
from rag.console  import console
from rag.loaders  import load_file
from rag.chunking import chunk_text


# ──────────────────────────────────────────
# Embedding helper
# ──────────────────────────────────────────

def get_embedding(text: str) -> list[float]:
    resp = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return resp["embedding"]


# ──────────────────────────────────────────
# Build / extend the vector store
# ──────────────────────────────────────────

def build_vector_db(selected_paths: list[str]) -> tuple[chromadb.Collection, dict[str, int]]:
    client = chromadb.Client(Settings(anonymized_telemetry=False))

    if not selected_paths:
        collection = client.get_or_create_collection(name="docs_empty", metadata={"hnsw:space": "cosine"})
        console.print("\n  [system]✓ Started without document context.[/]\n")
        return collection, {}

    col_id = hashlib.md5(",".join(sorted(selected_paths)).encode()).hexdigest()[:8]
    collection = client.get_or_create_collection(
        name=f"docs_{col_id}",
        metadata={"hnsw:space": "cosine"}
    )

    all_chunks: list[dict] = []
    doc_chunk_counts: dict[str, int] = {}

    for path in selected_paths:
        console.print(f"  [system]→ Loading:[/] {os.path.basename(path)}")
        text   = load_file(path)
        chunks = chunk_text(text, os.path.basename(path))
        all_chunks.extend(chunks)
        doc_chunk_counts[os.path.basename(path)] = len(chunks)

    console.print(f"\n  [system]Embedding {len(all_chunks)} chunks…[/]")

    ids, embeddings, documents, metadatas = [], [], [], []

    for i, chunk in enumerate(all_chunks):
        cid = f"chunk_{i}"
        emb = get_embedding(chunk["text"])
        ids.append(cid)
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"source": chunk["source"]})

        if (i + 1) % 20 == 0 or (i + 1) == len(all_chunks):
            console.print(f"  [{i+1}/{len(all_chunks)}] chunks embedded", end="\r")

    print()
    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    console.print(f"\n  [system]✓ Vector DB built — {len(all_chunks)} chunks indexed.[/]\n")
    return collection, doc_chunk_counts


def add_doc_to_db(
    collection: chromadb.Collection,
    path: str,
    doc_chunk_counts: dict[str, int],
    chunk_offset: int,
) -> int:
    """Embed and inject one new doc into an existing collection. Returns updated offset."""
    name   = os.path.basename(path)
    text   = load_file(path)
    chunks = chunk_text(text, name)

    console.print(f"  [system]Embedding {len(chunks)} new chunks for '{name}'…[/]")

    ids, embeddings, documents, metadatas = [], [], [], []
    for i, chunk in enumerate(chunks):
        cid = f"chunk_{chunk_offset + i}"
        emb = get_embedding(chunk["text"])
        ids.append(cid)
        embeddings.append(emb)
        documents.append(chunk["text"])
        metadatas.append({"source": name})

    collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)
    doc_chunk_counts[name] = len(chunks)
    console.print(f"  [system]✓ '{name}' added — {len(chunks)} chunks.[/]")
    return chunk_offset + len(chunks)


# ──────────────────────────────────────────
# Retrieval
# ──────────────────────────────────────────

def retrieve_raw(collection: chromadb.Collection, query: str) -> list[dict]:
    """Return raw list of {source, text} dicts for top-K chunks."""
    query_emb = get_embedding(query)
    results   = collection.query(
        query_embeddings=[query_emb],
        n_results=TOP_K,
        include=["documents", "metadatas"]
    )
    return [
        {"source": meta["source"], "text": doc}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]
