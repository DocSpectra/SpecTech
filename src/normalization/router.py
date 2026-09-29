"""Normalization router for heterogeneous document formats."""
from __future__ import annotations

from pathlib import Path

from src.normalization.html import html_to_text
from src.normalization.markdown import markdown_to_text
from src.normalization.rst import rst_to_text


def normalize_document(path: Path, text: str) -> str:
    """Normalize text based on file suffix for SpeciTeller comparisons."""
    suffix = path.suffix.lower()
    if suffix in {".md", ".markdown"}:
        return markdown_to_text(text)
    if suffix in {".rst"}:
        return rst_to_text(text)
    if suffix in {".html", ".htm"}:
        return html_to_text(text)
    return text