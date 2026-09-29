#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.human_first_reranking import (
    build_packet_and_metrics, finalize_xlsx, load_cases, load_protocol,
    prepare_scoring_and_proxies, run_generation, score_granuscore, score_ko,
    score_speciteller,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["preflight", "generate", "prepare", "score-speciteller", "score-granuscore", "score-ko", "build-packet", "finalize-xlsx"])
    parser.add_argument("--config", type=Path, default=Path("configs/round2_human_first_reranking_v1.json"))
    parser.add_argument("--corpus", choices=["ansible_docs", "github_docs"])
    parser.add_argument("--run", choices=["run01", "run02", "run03"])
    args = parser.parse_args()
    if args.command == "preflight":
        record, digest, freeze = load_protocol(args.config); result = {"config_sha256": digest, "freeze_commit": freeze["method_freeze_commit"], "cases": len(load_cases(record))}
    elif args.command == "generate": result = run_generation(args.config)
    elif args.command == "prepare": result = prepare_scoring_and_proxies(args.config)
    elif args.command == "score-speciteller": result = score_speciteller(args.config)
    elif args.command == "score-granuscore": result = score_granuscore(args.config)
    elif args.command == "score-ko":
        if not args.corpus or not args.run: parser.error("score-ko requires --corpus and --run")
        result = score_ko(args.config, args.corpus, args.run)
    elif args.command == "build-packet": result = build_packet_and_metrics(args.config)
    else: result = finalize_xlsx(args.config)
    print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
