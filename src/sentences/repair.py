"""Repair layer for sentence segmentation artifacts."""
from __future__ import annotations

from typing import Iterable


ARTIFACT_ONLY = {")",
    ":",
    ";",
    ",",
    "-",
    "–",
    "—",
}


def repair_sentences(sentences: Iterable[str]) -> Iterable[str]:
    """Merge segmentation artifacts while preserving short bullet fragments.

    - If a sentence is a standalone artifact like ")" or ":", merge into the
      previous sentence when possible.
    - If a short fragment is followed by a lower-case continuation, merge to keep
      bullet labels (e.g., "a." + "install...") intact.
    """
    buffer: list[str] = []
    for sentence in sentences:
        stripped = sentence.strip()
        if stripped in ARTIFACT_ONLY and buffer:
            buffer[-1] = f"{buffer[-1]} {stripped}".strip()
            continue
        if buffer and len(stripped) <= 3 and sentence.endswith(":"):
            buffer[-1] = f"{buffer[-1]} {stripped}".strip()
            continue
        buffer.append(sentence)

    merged: list[str] = []
    i = 0
    while i < len(buffer):
        current = buffer[i]
        if i + 1 < len(buffer):
            nxt = buffer[i + 1]
            if len(current) <= 3 and nxt[:1].islower():
                merged.append(f"{current} {nxt}".strip())
                i += 2
                continue
        merged.append(current)
        i += 1
    return merged