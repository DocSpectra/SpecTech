"""Fail unless the official GranuScore notebook examples reproduce."""

from __future__ import annotations

import json
import os
import time
import argparse

import numpy as np
import torch
from granuscore import GranuScore


TEXTS = [
    "Tony Hawk was born in San Diego",
    "Tony Hawk was born in California",
    "Tony Hawk was born in the United States",
    "A skateboarder was born in the United States",
    "A sportsman was born in the United States",
]
EXPECTED = np.asarray(
    [29.322155, 40.344215, 54.774826, 74.53445, 81.604416], dtype=np.float64
)
STRICT_GPU_TOLERANCE = 1e-4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=float, default=STRICT_GPU_TOLERANCE)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tolerance <= 0:
        raise SystemExit("tolerance must be positive")
    started = time.perf_counter()
    scorer = GranuScore(model_name=os.environ["GRANUSCORE_MODEL_PATH"], use_cache=False)
    scores = np.asarray(
        scorer(TEXTS, batch_size=32, encoding_batch_size=128), dtype=np.float64
    )
    max_abs_error = float(np.max(np.abs(scores - EXPECTED)))
    print(
        json.dumps(
            {
                "scores": scores.tolist(),
                "expected": EXPECTED.tolist(),
                "max_abs_error": max_abs_error,
                "tolerance": args.tolerance,
                "elapsed_seconds": time.perf_counter() - started,
                "cuda_available": torch.cuda.is_available(),
            },
            sort_keys=True,
        )
    )
    if args.require_cuda and not torch.cuda.is_available():
        raise SystemExit("strict published-example gate requires CUDA")
    if max_abs_error > args.tolerance:
        raise SystemExit("published GranuScore example reproduction failed")


if __name__ == "__main__":
    main()
