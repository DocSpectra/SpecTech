#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analysis.dgx_gptoss120b_rubric import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the frozen returned GPT-OSS-120B rubric matrix")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/round2_dgx_gptoss120b_rubric_analysis_v1.json"),
    )
    args = parser.parse_args()
    result = run_analysis(args.config)
    print(f"wrote {len(result['outputs'])} aggregate tables")


if __name__ == "__main__":
    main()
