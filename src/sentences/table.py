"""Sentence table writer and validation."""
from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class SentenceRecord:
    corpus_id: str
    doc_path: str
    sent_idx: int
    sent_text: str
    sent_id: str


def stable_sentence_id(corpus_id: str, doc_path: str, sent_idx: int, sent_text: str) -> str:
    """Generate a stable hash-based sentence ID."""
    payload = f"{corpus_id}\n{doc_path}\n{sent_idx}\n{sent_text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_sentence_records(
    corpus_id: str,
    doc_path: str,
    sentences: Iterable[str],
) -> list[SentenceRecord]:
    records: list[SentenceRecord] = []
    for idx, sent in enumerate(sentences):
        sent_id = stable_sentence_id(corpus_id, doc_path, idx, sent)
        records.append(
            SentenceRecord(
                corpus_id=corpus_id,
                doc_path=doc_path,
                sent_idx=idx,
                sent_text=sent,
                sent_id=sent_id,
            )
        )
    return records


def write_sentence_table(path: Path, records: Iterable[SentenceRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"])
        for record in records:
            writer.writerow(
                [
                    record.corpus_id,
                    record.doc_path,
                    record.sent_idx,
                    record.sent_text,
                    record.sent_id,
                ]
            )


def validate_sentence_records(records: Iterable[SentenceRecord]) -> None:
    seen_ids: set[str] = set()
    for record in records:
        if not record.sent_text.strip():
            raise ValueError("Empty sentence detected")
        if record.sent_id in seen_ids:
            raise ValueError("Duplicate sentence ID detected")
        seen_ids.add(record.sent_id)