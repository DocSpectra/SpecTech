"""Score a canonical SpecTech sentence CSV with frozen GranuScore defaults."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import granuscore
import numpy as np
import spacy
import torch
from granuscore import GranuScore


REQUIRED_COLUMNS = ("corpus_id", "sent_id", "sent_text")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--corpus-id", required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--encoding-batch-size", type=int, default=256)
    return parser.parse_args()


def batches(rows: list[dict[str, str]], size: int):
    for offset in range(0, len(rows), size):
        yield rows[offset : offset + size]


def main() -> None:
    args = parse_args()
    csv.field_size_limit(sys.maxsize)
    if args.batch_size < 1 or args.encoding_batch_size < 1:
        raise SystemExit("batch sizes must be positive")
    observed_input_hash = sha256(args.input)
    if observed_input_hash != args.expected_input_sha256.lower():
        raise SystemExit("input SHA-256 mismatch")

    with args.input.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or any(c not in reader.fieldnames for c in REQUIRED_COLUMNS):
            raise SystemExit(f"missing required columns: {REQUIRED_COLUMNS}")
        rows = list(reader)
    if not rows:
        raise SystemExit("input contains no rows")
    if any(row["corpus_id"] != args.corpus_id for row in rows):
        raise SystemExit("corpus_id mismatch")
    sent_ids = [row["sent_id"] for row in rows]
    if any(not value for value in sent_ids) or len(set(sent_ids)) != len(sent_ids):
        raise SystemExit("sent_id values must be nonempty and unique")
    if any(not row["sent_text"] for row in rows):
        raise SystemExit("sent_text values must be nonempty")

    scorer = GranuScore(model_name=os.environ["GRANUSCORE_MODEL_PATH"], use_cache=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    started = time.perf_counter()
    no_unit_count = 0
    output_count = 0
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("corpus_id", "sent_id", "granuscore_percentile", "referential_unit_count", "no_referential_unit"),
            lineterminator="\n",
        )
        writer.writeheader()
        for batch in batches(rows, args.batch_size):
            details = scorer(
                [row["sent_text"] for row in batch],
                return_details=True,
                batch_size=args.batch_size,
                encoding_batch_size=args.encoding_batch_size,
                show_progress_bar=False,
            )
            if len(details) != len(batch):
                raise SystemExit("prediction count mismatch")
            for row, detail in zip(batch, details):
                score = float(detail["pooled_percentile"])
                unit_count = sum(len(scope["unit_scores"]) for scope in detail["scopes"])
                no_unit = unit_count == 0
                if not math.isfinite(score) or not 0.0 <= score <= 100.0:
                    raise SystemExit("nonfinite or out-of-range GranuScore")
                writer.writerow(
                    {
                        "corpus_id": args.corpus_id,
                        "sent_id": row["sent_id"],
                        "granuscore_percentile": f"{score:.9f}",
                        "referential_unit_count": unit_count,
                        "no_referential_unit": str(no_unit).lower(),
                    }
                )
                no_unit_count += int(no_unit)
                output_count += 1
    if output_count != len(rows):
        raise SystemExit("incomplete output")
    temporary.replace(args.output)
    metadata = {
        "schema_version": "spectech_granuscore_scores_v1",
        "runner_sha256": sha256(Path(__file__)),
        "command": " ".join(sys.argv),
        "corpus_id": args.corpus_id,
        "input_path": args.input.as_posix(),
        "input_sha256": observed_input_hash,
        "output_path": args.output.as_posix(),
        "output_sha256": sha256(args.output),
        "row_count": output_count,
        "no_referential_unit_count": no_unit_count,
        "elapsed_seconds": time.perf_counter() - started,
        "image": os.environ.get("GRANUSCORE_IMAGE_ID"),
        "python": platform.python_version(),
        "granuscore": "1.0.1",
        "granuscore_module": granuscore.__file__,
        "spacy": spacy.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "batch_size": args.batch_size,
        "encoding_batch_size": args.encoding_batch_size,
        "score_direction": "higher_is_coarser_more_abstract",
        "official_defaults": {
            "predictor_type": "hit",
            "search_method": "random_anchors",
            "random_anchors_k": 999,
            "splitter": "SpacyNounPhraseSplitter_en_core_web_sm_3.8.0",
            "pooling": "mean",
            "pooling_scope": "sentence",
            "scope_pooling_method": "lower_quantile_mean",
            "scope_tail_q": 0.8,
            "percentile_before_pooling": True,
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
