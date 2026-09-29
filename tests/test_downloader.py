"""Tests for download helper detection."""
from __future__ import annotations

from pathlib import Path

from src.ingestion.downloader import has_existing_download
from src.ingestion.manifest import CorpusConfig


def test_has_existing_download_git(tmp_path: Path) -> None:
    target = tmp_path / "repo"
    (target / ".git").mkdir(parents=True)
    config = CorpusConfig(
        corpus_id="repo",
        source_type="git",
        source_ref="",
        snapshot="",
        local_path=target,
        include_globs=["**/*.md"],
        exclude_globs=[],
        language="en",
        notes="",
    )
    assert has_existing_download(config) is True


def test_has_existing_download_html_zip(tmp_path: Path) -> None:
    target = tmp_path / "html"
    target.mkdir(parents=True)
    (target / "corpus.zip").write_bytes(b"zip")
    config = CorpusConfig(
        corpus_id="html",
        source_type="html_zip",
        source_ref="",
        snapshot="",
        local_path=target,
        include_globs=["**/*.html"],
        exclude_globs=[],
        language="en",
        notes="",
    )
    assert has_existing_download(config) is True


def test_has_existing_download_wiki_dump(tmp_path: Path) -> None:
    target = tmp_path / "wiki"
    target.mkdir(parents=True)
    (target / "sample.txt").write_text("hello", encoding="utf-8")
    config = CorpusConfig(
        corpus_id="wiki",
        source_type="wiki_dump",
        source_ref="",
        snapshot="",
        local_path=target,
        include_globs=["**/*.txt"],
        exclude_globs=[],
        language="en",
        notes="",
    )
    assert has_existing_download(config) is True