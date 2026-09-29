"""PySBD-based sentence splitter."""
from __future__ import annotations

from typing import Iterable

import pysbd

from src.sentences.repair import repair_sentences


def split_sentences(text: str) -> list[str]:
    """Split text into sentences deterministically using PySBD.

    The output is passed through a repair layer to merge artifact-only fragments
    (e.g., ")" or ":") and keep short bullet-label fragments intact.
    """
    segmenter = pysbd.Segmenter(language="en", clean=False)
    sentences = [s.strip() for s in segmenter.segment(text) if s.strip()]
    return list(repair_sentences(sentences))


def iter_sentences(text: str) -> Iterable[str]:
    for sentence in split_sentences(text):
        yield sentence