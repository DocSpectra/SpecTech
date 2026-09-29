"""Corpus ingestion interfaces."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from src.ingestion.manifest import CorpusConfig, iter_document_paths


@dataclass(frozen=True)
class RawDocument:
    doc_id: str
    doc_path: Path
    content: str


def load_documents(config: CorpusConfig) -> Iterable[RawDocument]:
    for path in iter_document_paths(config):
        text = path.read_text(encoding="utf-8", errors="ignore")
        yield RawDocument(doc_id=str(path), doc_path=path, content=text)