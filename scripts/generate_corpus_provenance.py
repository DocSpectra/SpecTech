"""Generate PROVENANCE.json for all corpora in the manifest.

This script creates a provenance file under each corpus directory:

    data/corpora/<corpus_id>/PROVENANCE.json

It captures manifest metadata and source-specific details (e.g., git commit SHA,
zip artifact hash/size, or wiki text file counts). Existing provenance files are
left untouched unless --overwrite is supplied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.manifest import CorpusConfig
from src.ingestion.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate per-corpus PROVENANCE.json files")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest.yml"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _build_source_details(config: CorpusConfig) -> dict[str, object]:
    local_path = Path(config.local_path)

    if config.source_type == "git":
        return {
            "git_head": _git_head(local_path),
            "has_git_dir": (local_path / ".git").exists(),
        }

    if config.source_type == "html_zip":
        zip_files = sorted(local_path.glob("*.zip"))
        return {
            "zip_inputs": [
                {
                    "path": str(path).replace("\\", "/"),
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
                for path in zip_files
            ],
            "html_file_count": len(list(local_path.glob("**/*.html"))),
        }

    if config.source_type == "wiki_dump":
        txt_files = sorted(local_path.glob("*.txt"))
        return {
            "txt_file_count": len(txt_files),
            "txt_file_sample": [path.name for path in txt_files[:10]],
        }

    return {
        "file_count": len([p for p in local_path.rglob("*") if p.is_file()]),
    }


def build_provenance(config: CorpusConfig) -> dict[str, object]:
    return {
        "workflow": "manifest_corpus_provenance",
        "corpus_id": config.corpus_id,
        "source_type": config.source_type,
        "source_ref": config.source_ref,
        "snapshot": config.snapshot,
        "local_path": str(config.local_path).replace("\\", "/"),
        "include_globs": config.include_globs,
        "exclude_globs": config.exclude_globs,
        "language": config.language,
        "notes": config.notes,
        "source_details": _build_source_details(config),
    }


def main() -> None:
    args = parse_args()
    configs = load_manifest(args.manifest)

    for config in configs:
        out_path = Path(config.local_path) / "PROVENANCE.json"
        if out_path.exists() and not args.overwrite:
            print(f"Skipping existing: {out_path}")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(build_provenance(config), indent=2),
            encoding="utf-8",
        )
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
