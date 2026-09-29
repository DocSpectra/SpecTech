#!/usr/bin/env python3
"""Run the frozen QE-A Qwen edit-source workflow."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.qwen_edit_source import (
    prepare_scoring,
    run_analysis,
    run_generation,
    score_granuscore,
    score_ko_checkpoint,
    score_speciteller,
    verify_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("preflight", "generate", "prepare-scoring", "score-speciteller", "score-granuscore", "score-ko", "analyze"),
    )
    parser.add_argument("--config", type=Path, default=Path("configs/round2_qwen_edit_source_v1.json"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--corpus-id", choices=("ansible_docs", "github_docs"))
    parser.add_argument("--run-id", choices=("run01", "run02", "run03"))
    args = parser.parse_args()
    if args.command == "preflight":
        result = verify_preflight(args.config)
    elif args.command == "generate":
        result = run_generation(args.config, resume=args.resume)
    elif args.command == "prepare-scoring":
        result = prepare_scoring(args.config)
    elif args.command == "score-speciteller":
        result = score_speciteller(args.config)
    elif args.command == "score-granuscore":
        result = score_granuscore(args.config)
    elif args.command == "score-ko":
        if not args.corpus_id or not args.run_id:
            raise SystemExit("score-ko requires --corpus-id and --run-id")
        result = score_ko_checkpoint(args.config, args.corpus_id, args.run_id)
    else:
        result = run_analysis(args.config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
