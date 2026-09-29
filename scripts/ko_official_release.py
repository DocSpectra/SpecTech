"""Prepare or run the approved official author-repository comparator."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ko_specificity.official_release import (
    load_protocol,
    prepare_canonical_inputs,
    run_official_release,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "run"))
    parser.add_argument(
        "--config", type=Path, default=Path("configs/ko_official_release_comparator_v1.json")
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--input-manifest",
        type=Path,
        default=Path("configs/ko_official_release_inputs_v1.csv"),
    )
    parser.add_argument("--corpus-id")
    parser.add_argument("--run-id")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    protocol = load_protocol(args.config)
    repo_root = Path.cwd().resolve()
    experiment_root = args.outputs_root / "round2" / "ko_official_release"
    if args.command == "prepare":
        rows = prepare_canonical_inputs(
            protocol,
            outputs_root=args.outputs_root,
            artifact_checksums_path=Path("configs/round1_artifact_checksums.csv"),
            input_dir=experiment_root / "inputs",
            manifest_path=args.input_manifest,
        )
        for row in rows:
            print(f"{row['corpus_id']}: {row['valid_row_count']} rows {row['input_sha256']}")
        return
    if not args.corpus_id or not args.run_id:
        raise SystemExit("run requires --corpus-id and --run-id")
    metadata = run_official_release(
        protocol,
        corpus_id=args.corpus_id,
        run_id=args.run_id,
        input_manifest_path=args.input_manifest,
        repo_root=repo_root,
        run_root=experiment_root / "runs",
    )
    print(
        f"{metadata['corpus_id']}/{metadata['run_id']}: "
        f"{metadata['row_count']} scores {metadata['score_sha256']}"
    )


if __name__ == "__main__":
    main()
