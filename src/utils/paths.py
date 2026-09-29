"""Path helpers for the pipeline."""
from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def ensure_directories() -> None:
    for rel_path in ("data", "data/corpora", "data/manifests", "outputs", "tests"):
        (PROJECT_ROOT / rel_path).mkdir(parents=True, exist_ok=True)