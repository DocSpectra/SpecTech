#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.dgx_gptoss120b_human_first_analysis import run_analysis


def main() -> int:
    parser = argparse.ArgumentParser(description="Run frozen GPT-OSS-120B human-first Phase D analysis")
    parser.add_argument("--config", type=Path, default=Path("configs/round2_dgx_gptoss120b_human_first_analysis_v1.json"))
    args = parser.parse_args()
    print(json.dumps(run_analysis(args.config), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
