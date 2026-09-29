"""Tests for manifest loading."""
from __future__ import annotations

from pathlib import Path

from src.ingestion.manifest import load_manifest


def test_load_manifest(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yml"
    manifest.write_text(
        """
corpora:
  - corpus_id: demo
    source_type: git
    source_ref: https://example.com/repo.git
    snapshot: ""
    local_path: data/corpora/demo
    include_globs:
      - "**/*.md"
    exclude_globs: []
    language: en
    notes: demo corpus
""".strip(),
        encoding="utf-8",
    )

    configs = load_manifest(manifest)
    assert len(configs) == 1
    config = configs[0]
    assert config.corpus_id == "demo"
    assert config.source_type == "git"
    assert config.local_path.as_posix().endswith("data/corpora/demo")