"""CLI for the Round 2 strict-natural-language preprocessing ablation."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.preprocessing_ablation import RunSettings
from src.analysis.preprocessing_ablation import run_preprocessing_ablation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the Round 1 baseline and run strict_natural_language_v1"
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/round2/preprocessing_ablation"),
    )
    parser.add_argument(
        "--paper-facing-dir",
        type=Path,
        default=Path("analysis/round2_preprocessing_ablation"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/round1_speciteller_baseline.csv"),
    )
    parser.add_argument(
        "--rule-config",
        type=Path,
        default=Path("configs/strict_natural_language_v1.json"),
    )
    parser.add_argument(
        "--artifact-checksums-config",
        type=Path,
        default=Path("configs/round1_artifact_checksums.csv"),
    )
    parser.add_argument(
        "--source-provenance-config",
        type=Path,
        default=Path("configs/round1_source_provenance.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--examples-per-reason", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = " ".join(shlex.quote(value) for value in sys.argv)
    metadata = run_preprocessing_ablation(
        RunSettings(
            outputs_root=args.outputs_root,
            output_dir=args.output_dir,
            paper_facing_dir=args.paper_facing_dir,
            baseline_config=args.baseline_config,
            rule_config=args.rule_config,
            artifact_checksums_config=args.artifact_checksums_config,
            source_provenance_config=args.source_provenance_config,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            examples_per_reason=args.examples_per_reason,
            command=command,
        )
    )
    reconciliation = metadata["selection_reconciliation"]
    print(
        "Baseline passed; "
        f"retained {reconciliation['retained_rows']} of {reconciliation['total_rows']} rows."
    )
    print(f"Wrote full audit pack to {args.output_dir}")
    print(f"Wrote frozen paper-facing pack to {args.paper_facing_dir}")


if __name__ == "__main__":
    main()
