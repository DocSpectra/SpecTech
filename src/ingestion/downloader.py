"""Download helpers for corpora (optional, used for initial setup only)."""
from __future__ import annotations

import shutil
import subprocess
import stat
from pathlib import Path

import requests

from src.ingestion.manifest import CorpusConfig


def ensure_git_repo(config: CorpusConfig, force: bool = False) -> None:
    target = Path(config.local_path)
    if force and target.exists():
        shutil.rmtree(target, onerror=_handle_remove_readonly)
    if target.exists() and (target / ".git").exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.longpaths=true",
            "clone",
            "--depth",
            "1",
            config.source_ref,
            str(target),
        ],
        check=True,
    )


def has_existing_download(config: CorpusConfig) -> bool:
    target = Path(config.local_path)
    if config.source_type == "git":
        return target.exists() and (target / ".git").exists()
    if config.source_type == "html_zip":
        return (target / "corpus.zip").exists() or any(target.glob("**/*.html"))
    if config.source_type == "wiki_dump":
        return any(target.glob("**/*.txt"))
    return any(target.rglob("*"))


def download_zip(config: CorpusConfig, dest_name: str, force: bool = False) -> Path:
    target_dir = Path(config.local_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    zip_path = target_dir / dest_name
    if force and zip_path.exists():
        zip_path.unlink()
    if zip_path.exists():
        return zip_path
    response = requests.get(config.source_ref, timeout=120)
    response.raise_for_status()
    zip_path.write_bytes(response.content)
    return zip_path


def extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(zip_path), str(target_dir))


def _handle_remove_readonly(func, path, exc):
    del exc
    Path(path).chmod(stat.S_IWRITE)
    func(path)