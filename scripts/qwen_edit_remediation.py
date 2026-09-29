#!/usr/bin/env python3
"""CLI for the QE-R blinded remediation study."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.qwen_edit_remediation import (
    diagnose_v1,
    run_development_candidates,
    run_held_out_confirmation,
    summarize_development_candidates,
    summarize_held_out_confirmation,
    write_split,
)


DEFAULT_CONFIG = Path("configs/round2_qwen_edit_remediation_split_v1.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-split")
    subparsers.add_parser("diagnose-v1")
    generation = subparsers.add_parser("generate-candidates")
    generation.add_argument("--resume", action="store_true")
    subparsers.add_parser("summarize-candidates")
    held = subparsers.add_parser("run-held-out")
    held.add_argument("--resume", action="store_true")
    subparsers.add_parser("summarize-held-out")
    args = parser.parse_args()
    if args.command == "build-split":
        print(json.dumps(write_split(args.config), indent=2, sort_keys=True))
    elif args.command == "diagnose-v1":
        print(json.dumps(diagnose_v1(args.config), indent=2, sort_keys=True))
    elif args.command == "generate-candidates":
        print(json.dumps(run_development_candidates(args.config, resume=args.resume), indent=2, sort_keys=True))
    elif args.command == "summarize-candidates":
        print(json.dumps(summarize_development_candidates(args.config), indent=2, sort_keys=True))
    elif args.command == "run-held-out":
        print(json.dumps(run_held_out_confirmation(args.config, resume=args.resume), indent=2, sort_keys=True))
    elif args.command == "summarize-held-out":
        print(json.dumps(summarize_held_out_confirmation(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
