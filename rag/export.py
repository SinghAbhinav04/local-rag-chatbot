"""
PDF export — write a conversation to a nicely formatted PDF
using ReportLab.
"""

import datetime

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors


def export_pdf(messages: list[dict], model: str, doc_names: str, filepath: str):
    """Write the conversation to a formatted PDF using reportlab."""
    doc    = SimpleDocTemplate(
        filepath, pagesize=letter,
        leftMargin=inch, rightMargin=inch,
        topMargin=inch,  bottomMargin=inch,
    )
    base   = getSampleStyleSheet()
    styles = {
        "title":  ParagraphStyle("title",  parent=base["Title"],
                                 fontSize=18, spaceAfter=6),
        "meta":   ParagraphStyle("meta",   parent=base["Normal"],
                                 fontSize=9,  textColor=colors.grey, spaceAfter=16),
        "you":    ParagraphStyle("you",    parent=base["Normal"],
                                 fontSize=10, textColor=colors.HexColor("#0077cc"),
                                 spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"),
        "ai":     ParagraphStyle("ai",     parent=base["Normal"],
                                 fontSize=10, textColor=colors.HexColor("#2a7a2a"),
                                 spaceBefore=4,  spaceAfter=4, fontName="Helvetica-Bold"),
        "body":   ParagraphStyle("body",   parent=base["Normal"],
                                 fontSize=10, leading=14, spaceAfter=8),
    }

    ts    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    story = [
        Paragraph("Chat Export", styles["title"]),
        Paragraph(f"Exported: {ts} &nbsp;&nbsp;|&nbsp;&nbsp; Model: {model} &nbsp;&nbsp;|&nbsp;&nbsp; Docs: {doc_names}", styles["meta"]),
        HRFlowable(width="100%", thickness=1, color=colors.lightgrey, spaceAfter=12),
    ]

    for msg in messages:
        if msg["role"] == "user":
            story.append(Paragraph("You", styles["you"]))
        else:
            story.append(Paragraph("AI", styles["ai"]))
        # Escape special XML chars and preserve newlines
        safe = msg["content"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        for line in safe.split("\n"):
            if line.strip():
                story.append(Paragraph(line, styles["body"]))
        story.append(HRFlowable(width="100%", thickness=0.5,
                                color=colors.HexColor("#eeeeee"), spaceAfter=4))

    doc.build(story)
