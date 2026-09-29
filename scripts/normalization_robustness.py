"""Generate within-corpus normalization robustness report and CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.normalization_robustness import run_normalization_robustness

try:
    from src.ingestion.manifest import load_manifest
except ModuleNotFoundError:
    load_manifest = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check robustness of SpeciTeller score conclusions under within-corpus normalization."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/manifest.yml"),
        help="Manifest used to identify corpus IDs.",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root containing sentence, score, feature, and controlled-edit outputs.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory for robustness report and CSV.",
    )
    parser.add_argument(
        "--raw-epsilon",
        type=float,
        default=0.05,
        help="Raw-score epsilon for irrelevant_rewrite directional consistency.",
    )
    return parser.parse_args()


def load_corpus_ids(manifest_path: Path) -> list[str]:
    if load_manifest is not None:
        return [cfg.corpus_id for cfg in load_manifest(manifest_path)]

    corpus_ids: list[str] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- corpus_id:"):
            corpus_ids.append(stripped.split(":", 1)[1].strip().strip("'\""))
        elif stripped.startswith("corpus_id:"):
            corpus_ids.append(stripped.split(":", 1)[1].strip().strip("'\""))
    if not corpus_ids:
        raise ValueError(f"Could not load corpus IDs from manifest: {manifest_path}")
    return corpus_ids


def main() -> None:
    args = parse_args()
    corpus_ids = load_corpus_ids(args.manifest)
    report_path = args.analysis_dir / "normalization_robustness.md"
    csv_path = args.analysis_dir / "normalization_robustness_controlled_edits.csv"
    result = run_normalization_robustness(
        corpus_ids=corpus_ids,
        outputs_root=args.outputs_root,
        report_path=report_path,
        csv_path=csv_path,
        raw_epsilon=args.raw_epsilon,
    )
    print(result.report_text)


if __name__ == "__main__":
    main()
