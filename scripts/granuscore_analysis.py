"""Run the frozen Round 2 GranuScore analysis."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.granuscore_comparison import GranuSettings, run_granuscore_analysis
from src.analysis.model_comparison import ComparisonSettings


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", type=Path, default=Path("outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/round2/granuscore/analysis"))
    parser.add_argument("--compact-dir", type=Path, default=Path("analysis/round2_granuscore"))
    args = parser.parse_args()
    model_settings = ComparisonSettings(
        protocol_config=Path("configs/ko_official_release_comparator_v1.json"),
        input_manifest=Path("configs/ko_official_release_inputs_v1.csv"),
        outputs_root=args.outputs_root,
        run_root=args.outputs_root / "round2" / "ko_official_release" / "runs",
        preprocessing_manifest=args.outputs_root / "round2" / "preprocessing_ablation" / "strict_natural_language_v1_manifest.csv",
        artifact_checksums=Path("configs/round1_artifact_checksums.csv"),
        baseline_config=Path("configs/round1_speciteller_baseline.csv"),
        length_config=Path("configs/round2_length_control_v1.json"),
        output_dir=args.outputs_root / "round2" / "granuscore" / "base_unused",
        paper_facing_dir=None,
        bootstrap_replicates=1000,
        bootstrap_seed=20260809,
        command="internal GranuScore base join",
    )
    settings = GranuSettings(
        config_path=Path("configs/round2_granuscore_v1.json"),
        model_settings=model_settings,
        score_root=args.outputs_root / "round2" / "granuscore" / "scores",
        controlled_manifest=args.outputs_root / "round2" / "granuscore" / "controlled_edits" / "scoring_manifest.csv",
        controlled_preparation_metadata=args.outputs_root / "round2" / "granuscore" / "controlled_edits" / "preparation_metadata.json",
        controlled_scores=args.outputs_root / "round2" / "granuscore" / "controlled_edits" / "scores.csv",
        controlled_metadata=args.outputs_root / "round2" / "granuscore" / "controlled_edits" / "scores.metadata.json",
        pilot_config=Path("configs/round2_pilot_model_human_comparison_v1.json"),
        qwen_scores=Path("analysis/round2_qwen_rubric/qwen_scores.csv"),
        qwen_metadata=Path("analysis/round2_qwen_rubric/run_metadata.json"),
        length_config=Path("configs/round2_length_control_v1.json"),
        output_dir=args.output_dir,
        compact_dir=args.compact_dir,
        command="python scripts/granuscore_analysis.py",
    )
    metadata = run_granuscore_analysis(settings)
    print(f"GranuScore analysis complete: {len(metadata['outputs'])} tables")


if __name__ == "__main__":
    main()
