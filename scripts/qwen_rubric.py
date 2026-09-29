#!/usr/bin/env python3
"""Run frozen local-Qwen rubric scoring or analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.qwen_rubric import run_analysis, run_scoring


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("score", "analyze", "run"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/round2_qwen_rubric_v1.json"),
    )
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command in {"score", "run"}:
        result = run_scoring(config_path=args.config, resume=args.resume)
        print(f"Qwen scoring passed: {len(result['scores'])} rows")
    if args.command in {"analyze", "run"}:
        metadata = run_analysis(config_path=args.config)
        print(
            "Qwen rubric analysis passed: "
            f"{metadata['coverage']['total_rows']} rows; "
            f"freeze={metadata['outcome_blind_freeze_commit']}"
        )


if __name__ == "__main__":
    main()
