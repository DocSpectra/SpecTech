"""Tokenizer helper for SpeciTeller input."""
from __future__ import annotations

from nltk.tokenize import TreebankWordTokenizer


_TOKENIZER = TreebankWordTokenizer()


def tokenize_for_speciteller(text: str) -> str:
    """Tokenize a sentence using TreebankWordTokenizer.

    Returns a space-delimited string suitable for SpeciTeller input.
    """
    tokens = _TOKENIZER.tokenize(text)
    return " ".join(tokens)