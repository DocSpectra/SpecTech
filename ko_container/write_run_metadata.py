"""Write portable evidence for a completed legacy run (Python 3.6 compatible)."""
from __future__ import print_function

import argparse
import hashlib
import json
import platform

import numpy
import scipy
import torch


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--predictions", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()
with open(args.predictions, "r") as handle:
    prediction_count = sum(1 for line in handle if line.strip())
metadata = {
    "schema_version": "ko_reproduction_run_v1",
    "mode": args.mode,
    "upstream_commit": "36f8e835e9dc6087d5b6763accf302db175947b1",
    "cpu_patch": "official no_cuda.zip",
    "prediction_count": prediction_count,
    "prediction_sha256": sha256(args.predictions),
    "teacher_model_sha256": sha256(args.model),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "cuda_available": torch.cuda.is_available(),
}
with open(args.output, "w") as handle:
    json.dump(metadata, handle, indent=2, sort_keys=True)
    handle.write("\n")
