"""Run the frozen model-aware official-release comparator analysis."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.model_comparison import ComparisonSettings, run_model_comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/round2/ko_official_release/analysis"))
    parser.add_argument("--paper-facing-dir", type=Path, default=Path("analysis/round2_ko_official_release_comparison"))
    args = parser.parse_args()
    settings = ComparisonSettings(
        protocol_config=Path("configs/ko_official_release_comparator_v1.json"),
        input_manifest=Path("configs/ko_official_release_inputs_v1.csv"),
        outputs_root=args.outputs_root,
        run_root=args.outputs_root / "round2" / "ko_official_release" / "runs",
        preprocessing_manifest=args.outputs_root / "round2" / "preprocessing_ablation" / "strict_natural_language_v1_manifest.csv",
        artifact_checksums=Path("configs/round1_artifact_checksums.csv"),
        baseline_config=Path("configs/round1_speciteller_baseline.csv"),
        length_config=Path("configs/round2_length_control_v1.json"),
        output_dir=args.output_dir,
        paper_facing_dir=args.paper_facing_dir,
        bootstrap_replicates=1000,
        bootstrap_seed=20260809,
        command="python scripts/ko_model_comparison.py",
    )
    metadata = run_model_comparison(settings)
    print(f"comparison complete: {len(metadata['outputs'])} tables")


if __name__ == "__main__":
    main()
