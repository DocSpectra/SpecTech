"""Generate documentation-aware proxy decomposition report and CSV."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.proxy_decomposition import run_proxy_decomposition
from src.features.proxies import TreebankWordTokenizer

try:
    from src.ingestion.manifest import load_manifest
except ModuleNotFoundError:
    load_manifest = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose aggregate technical-token proxies into documentation-aware feature correlations."
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
        help="Root containing sentence and SpeciTeller score outputs.",
    )
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory for the proxy decomposition report and CSV.",
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
    if TreebankWordTokenizer is None:
        raise RuntimeError(
            "The paper-facing proxy decomposition requires pinned NLTK Treebank tokenization; "
            "install requirements.txt instead of using the fallback tokenizer."
        )
    result = run_proxy_decomposition(
        corpus_ids=load_corpus_ids(args.manifest),
        outputs_root=args.outputs_root,
        report_path=args.analysis_dir / "proxy_decomposition_results.md",
        csv_path=args.analysis_dir / "proxy_decomposition_tables.csv",
    )
    print(result.report_text)


if __name__ == "__main__":
    main()
