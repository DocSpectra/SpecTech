#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.local_rubric_replication import run_analysis, run_scoring


def main() -> None:
    parser = argparse.ArgumentParser(description="Matched local rubric replication")
    parser.add_argument("command", choices=("score", "analyze"))
    parser.add_argument("--config", type=Path, default=Path("configs/round2_local_rubric_replication_v1.json"))
    args = parser.parse_args()
    if args.command == "score":
        results = run_scoring(args.config)
        print(f"completed {len(results)} gated judge matrices")
    else:
        result = run_analysis(args.config)
        print(f"wrote {len(result['outputs'])} compact tables")


if __name__ == "__main__":
    main()
