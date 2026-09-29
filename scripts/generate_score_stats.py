"""Generate corpus-level score summary statistics from canonical score TSV files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.score_stats import compute_and_write_score_stats_for_corpora
from src.ingestion.manifest import load_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate per-corpus and aggregate score summary stats from outputs/speciteller/*_scores.tsv"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest.yml"),
        help="Path to manifest YAML used to determine corpus_ids",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs root directory containing speciteller score files",
    )
    parser.add_argument(
        "--corpus-ids",
        nargs="*",
        default=None,
        help="Optional explicit corpus_ids to process (otherwise all in manifest)",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Run lightweight validation of expected output files/columns after generation",
    )
    return parser.parse_args()


def run_self_check(outputs_root: Path, corpus_ids: list[str]) -> None:
    required_cols = [
        "corpus_id",
        "score_count",
        "score_mean",
        "score_median",
        "score_std",
        "score_min",
        "score_max",
        "score_q1",
        "score_q3",
        "score_iqr",
    ]

    analysis_dir = outputs_root / "analysis" / "tables"
    paper_dir = outputs_root / "paper_pack" / "tables"

    for corpus_id in corpus_ids:
        for base_dir in (analysis_dir, paper_dir):
            path = base_dir / f"{corpus_id}_score_stats.csv"
            if not path.exists():
                raise FileNotFoundError(f"Missing expected stats file: {path}")

            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")
            if header != required_cols:
                raise ValueError(
                    f"Unexpected columns in {path}. Expected {required_cols}, got {header}"
                )

    for base_dir in (analysis_dir, paper_dir):
        aggregate = base_dir / "corpus_score_stats_all.csv"
        if not aggregate.exists():
            raise FileNotFoundError(f"Missing expected aggregate stats file: {aggregate}")


def main() -> None:
    args = parse_args()
    manifest_cfgs = load_manifest(args.manifest)
    corpus_ids = [cfg.corpus_id for cfg in manifest_cfgs]
    if args.corpus_ids:
        corpus_ids = [cid for cid in corpus_ids if cid in set(args.corpus_ids)]

    stats_rows = compute_and_write_score_stats_for_corpora(
        corpus_ids=corpus_ids,
        outputs_root=args.outputs_root,
    )
    print(f"Generated score stats for {len(stats_rows)} corpus/corpora.")

    if args.self_check:
        run_self_check(args.outputs_root, corpus_ids)
        print("Self-check passed.")


if __name__ == "__main__":
    main()
