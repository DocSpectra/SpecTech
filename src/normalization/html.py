"""HTML normalization utilities."""
from __future__ import annotations

from bs4 import BeautifulSoup


def html_to_text(html: str) -> str:
    """Convert HTML to plaintext with deterministic cleanup.

    - Removes block-level code in <pre> blocks to focus on prose sentences.
    - Preserves inline code by unwrapping <code> tags (text retained).
    - Removes navigation/boilerplate tags like <nav>, <script>, and <style>.
    - Keeps text suited for SpeciTeller comparisons (Paper_Strategy.md motivation).
    """
    soup = BeautifulSoup(html, "lxml")
    for pre in soup.find_all("pre"):
        pre.decompose()
    for code in soup.find_all("code"):
        code.unwrap()
    for script in soup.find_all(["script", "style", "nav"]):
        script.decompose()
    return " ".join(chunk.strip() for chunk in soup.stripped_strings)