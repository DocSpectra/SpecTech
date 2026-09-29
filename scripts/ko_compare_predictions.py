"""Compare two official smoke runs against the outcome-blind tolerance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ko_specificity.io import parse_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mean-absolute-tolerance", type=float, default=0.02)
    parser.add_argument("--maximum-absolute-tolerance", type=float, default=0.10)
    args = parser.parse_args()
    count = len([x for x in args.first.read_text(encoding="utf-8").splitlines() if x.strip()])
    first = parse_predictions(args.first, count)
    second = parse_predictions(args.second, count)
    differences = [abs(a - b) for a, b in zip(first, second)]
    mean_absolute = sum(differences) / len(differences)
    maximum_absolute = max(differences)
    passed = (
        mean_absolute <= args.mean_absolute_tolerance
        and maximum_absolute <= args.maximum_absolute_tolerance
    )
    result = {
        "schema_version": "ko_smoke_determinism_v1",
        "prediction_count": count,
        "mean_absolute_difference": mean_absolute,
        "maximum_absolute_difference": maximum_absolute,
        "mean_absolute_tolerance": args.mean_absolute_tolerance,
        "maximum_absolute_tolerance": args.maximum_absolute_tolerance,
        "passed": passed,
        "note": "Official code seeds NumPy and Torch but does not seed Python random.",
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise SystemExit("Ko released-data smoke determinism tolerance failed")


if __name__ == "__main__":
    main()
