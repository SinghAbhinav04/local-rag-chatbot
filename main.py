#!/usr/bin/env python3
"""
Entry point for the Local Document Chat (Vector RAG) system.
Run:  python3 main.py   or   bash rag.sh
"""

from rich.rule import Rule

from rag.console  import console
from rag.ui       import choose_model, choose_docs
from rag.vectordb import build_vector_db
from rag.chat     import chat


def main():
    console.print(Rule("[system]LOCAL DOC CHAT  (Vector RAG)[/]", style="system"))

    model          = choose_model()
    selected_paths = choose_docs()

    console.print("\n  [system]Building vector database…[/]")
    collection, doc_chunk_counts = build_vector_db(selected_paths)

    chat(collection, model, selected_paths, doc_chunk_counts)

    console.print("\n[system]Goodbye![/]")


if __name__ == "__main__":
    main()
