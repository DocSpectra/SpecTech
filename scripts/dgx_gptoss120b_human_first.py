#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.dgx_gptoss120b_human_first import (
    build_packet,
    content_blind_binding,
    finalize,
    prepare,
    score_granuscore,
    score_ko,
    score_speciteller,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["binding", "prepare", "score-speciteller", "score-granuscore", "score-ko", "build-packet", "finalize"])
    parser.add_argument("--config", type=Path, default=Path("configs/round2_dgx_gptoss120b_human_first_v1.json"))
    parser.add_argument("--corpus", choices=["ansible_docs", "github_docs"])
    parser.add_argument("--run", choices=["run01", "run02", "run03"])
    args = parser.parse_args()
    if args.command == "binding":
        result = content_blind_binding(args.config, require_freeze=False)
    elif args.command == "prepare":
        result = prepare(args.config)
    elif args.command == "score-speciteller":
        result = score_speciteller(args.config)
    elif args.command == "score-granuscore":
        result = score_granuscore(args.config)
    elif args.command == "score-ko":
        if not args.corpus or not args.run:
            parser.error("score-ko requires --corpus and --run")
        result = score_ko(args.config, args.corpus, args.run)
    elif args.command == "build-packet":
        result = build_packet(args.config)
    else:
        result = finalize(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
