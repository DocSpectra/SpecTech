"""CLI for the frozen Round 2 length-controlled analysis."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.length_controlled import LengthRunSettings, run_length_controlled


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exact-support token-length standardization and regression sensitivity."
    )
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/round2/length_controlled")
    )
    parser.add_argument(
        "--paper-facing-dir",
        type=Path,
        default=Path("analysis/round2_length_control"),
    )
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/round1_speciteller_baseline.csv"),
    )
    parser.add_argument(
        "--method-config",
        type=Path,
        default=Path("configs/round2_length_control_v1.json"),
    )
    parser.add_argument(
        "--artifact-checksums-config",
        type=Path,
        default=Path("configs/round1_artifact_checksums.csv"),
    )
    parser.add_argument(
        "--preprocessing-metadata",
        type=Path,
        default=Path("analysis/round2_preprocessing_ablation/run_metadata.json"),
    )
    parser.add_argument(
        "--source-provenance-config",
        type=Path,
        default=Path("configs/round1_source_provenance.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260809)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metadata = run_length_controlled(
        LengthRunSettings(
            outputs_root=args.outputs_root,
            output_dir=args.output_dir,
            paper_facing_dir=args.paper_facing_dir,
            baseline_config=args.baseline_config,
            method_config=args.method_config,
            artifact_checksums_config=args.artifact_checksums_config,
            preprocessing_metadata=args.preprocessing_metadata,
            source_provenance_config=args.source_provenance_config,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
            command="scripts/length_controlled_analysis.py",
        )
    )
    print(f"baseline_gate_passed={metadata['baseline_gate_passed']}")
    print(f"result_categories={metadata['result_categories']}")


if __name__ == "__main__":
    main()
