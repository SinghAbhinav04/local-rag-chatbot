"""
File loaders — read PDF, TXT, Markdown, HTML, DOCX, and CSV
into plain text for downstream chunking.
"""

import pandas as pd
from pypdf import PdfReader
from docx import Document
from bs4 import BeautifulSoup
import markdown


def load_file(path: str) -> str:
    """Return the full text content of a supported document file."""
    ext = path.rsplit(".", 1)[-1].lower()

    if ext == "pdf":
        reader = PdfReader(path)
        return "\n".join([p.extract_text() or "" for p in reader.pages])

    if ext == "txt":
        return open(path, encoding="utf-8").read()

    if ext == "md":
        html = markdown.markdown(open(path, encoding="utf-8").read())
        return BeautifulSoup(html, "html.parser").get_text()

    if ext == "html":
        return BeautifulSoup(open(path, encoding="utf-8"), "html.parser").get_text()

    if ext == "docx":
        doc = Document(path)
        return "\n".join([p.text for p in doc.paragraphs])

    if ext == "csv":
        df = pd.read_csv(path)
        return df.to_string()

    return open(path, encoding="utf-8").read()
