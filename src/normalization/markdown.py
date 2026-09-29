"""Markdown normalization utilities."""
from __future__ import annotations

import re

from markdown_it import MarkdownIt

from src.normalization.code_removal import strip_markdown_fenced_blocks


FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)


def markdown_to_text(text: str) -> str:
    """Convert Markdown to plaintext while preserving prose structure.

    - Removes fenced code blocks so SpeciTeller comparisons focus on prose sentences.
    - Keeps inline code spans that carry technical specificity signals.
    - Retains headings and list items as newline-separated text for deterministic
      sentence segmentation (Paper_Strategy.md motivation).
    """
    cleaned = FRONTMATTER_RE.sub("", text)
    cleaned = strip_markdown_fenced_blocks(cleaned)
    md = MarkdownIt("commonmark")
    tokens = md.parse(cleaned)
    lines: list[str] = []
    for token in tokens:
        if token.type in {"inline", "text"} and token.content:
            lines.append(token.content)
        if token.type in {"heading_open", "list_item_open"}:
            lines.append("\n")
    return "\n".join(line for line in lines if line.strip())