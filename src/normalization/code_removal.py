"""Code removal utilities for block-level code."""
from __future__ import annotations

import re


FENCED_BLOCK_RE = re.compile(r"```.*?```", re.DOTALL)
RST_CODE_BLOCK_RE = re.compile(
    r"\.\.\s+code-block::.*?(\n\S|\Z)",
    re.DOTALL,
)


def strip_markdown_fenced_blocks(text: str) -> str:
    """Remove fenced Markdown code blocks while preserving surrounding prose.

    This supports the paper's goal of comparing SpeciTeller scores over natural
    language sentences (see Paper_Strategy.md). Inline backticks remain untouched
    because we only remove triple-backtick fences.
    """
    return FENCED_BLOCK_RE.sub(" ", text)


def strip_rst_code_blocks(text: str) -> str:
    """Remove reStructuredText code blocks introduced by ``.. code-block::``.

    We drop block-level code to keep the sentence pool aligned with prose used for
    SpeciTeller comparisons in the analysis paper (Paper_Strategy.md).
    """
    def _replace(match: re.Match) -> str:
        trailing = match.group(1)
        return "\n" + trailing if trailing else " "

    return RST_CODE_BLOCK_RE.sub(_replace, text)