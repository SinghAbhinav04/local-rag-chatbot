"""
Text chunker — splits a document string into overlapping
fixed-size windows for embedding and retrieval.
"""

from rag.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_text(text: str, source: str) -> list[dict]:
    """Split `text` into overlapping chunks, each tagged with `source`."""
    chunks = []
    start = 0
    while start < len(text):
        end   = start + CHUNK_SIZE
        chunk = text[start:end]
        if chunk.strip():
            chunks.append({"text": chunk, "source": source})
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks
