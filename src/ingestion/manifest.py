"""Manifest loading and validation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import yaml


@dataclass(frozen=True)
class CorpusConfig:
    corpus_id: str
    source_type: str
    source_ref: str
    snapshot: str
    local_path: Path
    include_globs: list[str]
    exclude_globs: list[str]
    language: str
    notes: str


def load_manifest(path: Path) -> list[CorpusConfig]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    corpora = data.get("corpora", []) if isinstance(data, dict) else []
    configs: list[CorpusConfig] = []
    for entry in corpora:
        configs.append(
            CorpusConfig(
                corpus_id=entry["corpus_id"],
                source_type=entry["source_type"],
                source_ref=entry["source_ref"],
                snapshot=str(entry.get("snapshot", "")),
                local_path=Path(entry["local_path"]),
                include_globs=list(entry.get("include_globs", [])),
                exclude_globs=list(entry.get("exclude_globs", [])),
                language=entry.get("language", ""),
                notes=entry.get("notes", ""),
            )
        )
    return configs


def iter_document_paths(config: CorpusConfig) -> Iterable[Path]:
    root = config.local_path
    for pattern in config.include_globs:
        for path in root.glob(pattern):
            if any(path.match(exclude) for exclude in config.exclude_globs):
                continue
            if path.is_file():
                yield path