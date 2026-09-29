"""Run the frozen Round 2 existing-pilot model--human comparison."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.pilot_model_human import run_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the frozen existing-pilot SpeciTeller/Ko agreement analysis"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/round2_pilot_model_human_comparison_v1.json"),
    )
    parser.add_argument("--full-dir", type=Path, default=None)
    parser.add_argument("--compact-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = run_analysis(
        config_path=args.config,
        full_dir=args.full_dir,
        compact_dir=args.compact_dir,
    )
    print("Existing-pilot comparison complete.")
    print(f"- coverage rows: {metadata['coverage']['total_rows']}/80")
    print(f"- compact output: {args.compact_dir or 'analysis/round2_pilot_model_human_comparison'}")


if __name__ == "__main__":
    main()
