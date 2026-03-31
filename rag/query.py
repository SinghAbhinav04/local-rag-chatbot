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
    # Retrieve context
    raw_chunks = []
    if collection.count() == 0:
        context = "No documents have been loaded into the system yet."
    else:
        raw_chunks = retrieve_raw(collection, question)
        if not raw_chunks:
            console.print("  [info](No relevant chunks found in loaded documents for this query.)[/]")
            context = "No relevant context found for this specific query, though documents are loaded."
        else:
            context = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in raw_chunks)

    system_prompt = (
        "You are a helpful assistant. Use the CONTEXT below to answer the user's question. "
        "Include specific details, numbers, and facts from the context in your answer. "
        "Mention the source when possible.\n\n"
        f"CONTEXT:\n{context}\n\n"
        "Answer the question using the context above. Be detailed and specific."
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
