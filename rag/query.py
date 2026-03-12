"""
Core RAG query runner — retrieves context from the vector DB,
builds the prompt, streams the response from Ollama, and
feeds sentences to the TTS queue.
"""

import ollama
import chromadb

from rag.console  import console
from rag.vectordb import retrieve_raw
from rag.speech   import speak, stop_speaking


def run_query(
    collection: chromadb.Collection,
    model: str,
    messages: list[dict],
    question: str,
) -> tuple[str, list[dict]]:
    """Run one RAG query. Returns (answer_text, raw_chunks)."""
    raw_chunks = retrieve_raw(collection, question)
    context    = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in raw_chunks)

    system_prompt = (
        "You are a knowledgeable AI assistant. "
        "The user has loaded documents as reference material. "
        "Use the context below to inform your answers — treat it as extra knowledge. "
        "You are NOT restricted to only the documents; feel free to expand. "
        "Prefer document context when relevant.\n\n"
        f"REFERENCE CONTEXT:\n{context}"
    )

    turn_messages = (
        [{"role": "system", "content": system_prompt}]
        + messages
        + [{"role": "user", "content": question}]
    )

    stream = ollama.chat(model=model, messages=turn_messages, stream=True)

    console.print("\n[ai]AI:[/] ", end="")
    answer   = ""
    sentence = ""

    try:
        for chunk in stream:
            token = chunk["message"]["content"]
            console.print(token, end="", markup=False)
            answer   += token
            sentence += token
            if token in ".!?":
                speak(sentence.strip())
                sentence = ""
    except KeyboardInterrupt:
        stop_speaking()
        console.print("\n  [system]⚠ Interrupted.[/]")

    if sentence.strip():
        speak(sentence.strip())

    console.print()
    return answer, raw_chunks
