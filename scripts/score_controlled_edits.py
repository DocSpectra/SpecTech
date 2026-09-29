"""Score completed controlled-edit templates and write Phase 7.5 statistics.

This script is the second step of the controlled-edit workflow:
1) Generate template CSV (with blank ``sentence_edited``)
2) Manually fill ``sentence_edited`` rows
3) Run this script to re-score edited sentences and compute delta statistics

Outputs are written to ``outputs/controlled_edits/`` by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.phase075_controlled_edit_evaluation import compute_controlled_edit_stats
from src.analysis.phase075_controlled_edit_evaluation import load_controlled_edit_template
from src.analysis.phase075_controlled_edit_evaluation import rescore_completed_controlled_edits
from src.analysis.phase075_controlled_edit_evaluation import write_controlled_edit_stats
from src.analysis.phase075_controlled_edit_evaluation import write_scored_controlled_edits
from src.speciteller.config import DEFAULT_SPECITELLER_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-score edited sentences and compute controlled-edit deltas"
    )
    parser.add_argument(
        "--template-path",
        type=Path,
        required=True,
        help="Path to completed controlled-edit template CSV",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root outputs directory",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip container preflight check before scoring",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_rows = load_controlled_edit_template(args.template_path)

    controlled_dir = args.outputs_root / "controlled_edits"
    controlled_dir.mkdir(parents=True, exist_ok=True)

    scored_rows = rescore_completed_controlled_edits(
        template_rows,
        config=DEFAULT_SPECITELLER_CONFIG,
        outputs_dir=controlled_dir,
        preflight=not args.skip_preflight,
    )
    stats_rows = compute_controlled_edit_stats(scored_rows)

    stem = args.template_path.stem
    scored_path = controlled_dir / f"{stem}_scored.csv"
    stats_path = controlled_dir / f"{stem}_stats.csv"
    write_scored_controlled_edits(scored_path, scored_rows)
    write_controlled_edit_stats(stats_path, stats_rows)

    print(f"Scored rows written to {scored_path}")
    print(f"Stats written to {stats_path}")


if __name__ == "__main__":
    main()
