"""Build deterministic wikipedia_en plaintext docs from Wikimedia dump shards.

This is the reproducible path for Wikipedia corpus construction:
- use explicit dump shard files (.bz2)
- extract in deterministic file/page order
- emit fixed naming scheme for output .txt docs
- write provenance metadata (input files + sha256 + parameters)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.ingestion.wiki_dump import extract_plaintext_from_wiki_dump


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build wikipedia_en plaintext corpus from dump shards")
    parser.add_argument(
        "--dump-glob",
        default="data/corpora/wikipedia_en_dumps/*.bz2",
        help="Glob for Wikimedia XML dump shard files",
    )
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/corpora/wikipedia_en"),
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--provenance-path",
        type=Path,
        default=Path("data/corpora/wikipedia_en/PROVENANCE.json"),
    )
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    dump_paths = sorted(Path().glob(args.dump_glob))
    if not dump_paths:
        raise FileNotFoundError(f"No dump shards matched glob: {args.dump_glob}")

    written = extract_plaintext_from_wiki_dump(
        dump_bz2_paths=dump_paths,
        output_dir=args.output_dir,
        target_count=args.target_count,
        reset=args.reset,
    )

    provenance = {
        "workflow": "wikimedia_dump_shard_extraction",
        "dump_glob": args.dump_glob,
        "target_count": args.target_count,
        "reset": args.reset,
        "written_count": written,
        "dump_inputs": [
            {
                "path": str(path).replace("\\", "/"),
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in dump_paths
        ],
    }
    args.provenance_path.parent.mkdir(parents=True, exist_ok=True)
    args.provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"wrote {written} wikipedia plaintext files to {args.output_dir}")
    print(f"wrote provenance metadata to {args.provenance_path}")


if __name__ == "__main__":
    main()
