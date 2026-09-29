"""reStructuredText normalization utilities."""
from __future__ import annotations

import re

from src.normalization.code_removal import strip_rst_code_blocks


DIRECTIVE_RE = re.compile(r"^\.\.\s+\w+::.*$")


def rst_to_text(text: str) -> str:
    """Convert reStructuredText to plaintext for SpeciTeller comparisons.

    Removes block-level code and directive lines while retaining prose content.
    """
    cleaned = strip_rst_code_blocks(text)
    lines = []
    for line in cleaned.splitlines():
        if DIRECTIVE_RE.match(line.strip()):
            continue
        lines.append(line)
    return "\n".join(line for line in lines if line.strip())