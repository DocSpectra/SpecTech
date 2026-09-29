"""Tests for normalization utilities."""
from __future__ import annotations

from src.normalization.code_removal import strip_markdown_fenced_blocks
from src.normalization.html import html_to_text
from src.normalization.markdown import markdown_to_text
from src.normalization.rst import rst_to_text


def test_strip_markdown_fenced_blocks_preserves_inline_code() -> None:
    text = """
Here is inline `code`.

```
def foo():
    return 1
```

Back to prose.
""".strip()
    cleaned = strip_markdown_fenced_blocks(text)
    assert "def foo" not in cleaned
    assert "inline `code`" in cleaned


def test_html_to_text_removes_pre_blocks_keeps_inline_code() -> None:
    html = """
<html><body>
<p>Inline <code>code</code> here.</p>
<pre><code>block code</code></pre>
</body></html>
""".strip()
    cleaned = html_to_text(html)
    assert "block code" not in cleaned
    assert "Inline code here." in cleaned


def test_markdown_frontmatter_removed() -> None:
    text = """
---
title: Demo
---

# Heading
Body text.
""".strip()
    cleaned = markdown_to_text(text)
    assert "title: Demo" not in cleaned
    assert "Heading" in cleaned


def test_rst_directive_removed() -> None:
    text = """
.. note::
   This is a note.

Paragraph text.
""".strip()
    cleaned = rst_to_text(text)
    assert ".. note::" not in cleaned
    assert "Paragraph text" in cleaned