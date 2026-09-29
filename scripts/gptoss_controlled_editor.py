#!/usr/bin/env python3
"""CLI for the frozen GPT-OSS controlled-editor comparison."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.gptoss_controlled_editor import (
    load_cases,
    load_protocol,
    run_controlled,
    validate_existing_packet,
    verify_ollama_identity,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/round2_gptoss_controlled_editor_v1.json"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    run = sub.add_parser("run")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--timeout", type=float, default=300.0)
    sub.add_parser("validate-packet")
    args = parser.parse_args()
    if args.command == "preflight":
        record, config_hash, freeze = load_protocol(args.config)
        result = {
            "config_sha256": config_hash, "freeze_commit": freeze["method_freeze_commit"],
            "planned_cases": len(load_cases(record)), "identity": verify_ollama_identity(record),
        }
    elif args.command == "run":
        result = run_controlled(args.config, resume=args.resume, timeout=args.timeout)
    else:
        result = validate_existing_packet(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
