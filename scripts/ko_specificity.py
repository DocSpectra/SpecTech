"""Blocked project-scoring CLI for the Ko et al. (2019) diagnostic adapter."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ko_specificity.config import DEFAULT_KO_CONFIG
from src.ko_specificity.runner import run_ko_specificity


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ko et al. diagnostic adapter (project scoring is blocked)"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--adaptation-context",
        required=True,
        help="Frozen target-domain adaptation identity (for example corpus_id+config hash)",
    )
    parser.add_argument("--image", default=DEFAULT_KO_CONFIG.image)
    parser.add_argument("--glove-volume", default=DEFAULT_KO_CONFIG.glove_volume)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DEFAULT_KO_CONFIG.__class__(
        **{
            **DEFAULT_KO_CONFIG.__dict__,
            "image": args.image,
            "glove_volume": args.glove_volume,
        }
    )
    run_ko_specificity(
        config,
        args.input,
        args.output,
        adaptation_context=args.adaptation_context,
    )


if __name__ == "__main__":
    main()
