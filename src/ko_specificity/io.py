"""Input preparation and output parsing for the official legacy interface."""
from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path

from src.ko_specificity.config import KoSpecificityConfig


INPUT_COLUMNS = ("sent_id", "corpus_id", "text")
OUTPUT_COLUMNS = (
    "sent_id",
    "corpus_id",
    "model_id",
    "score_raw",
    "score_min",
    "score_max",
    "score_direction",
    "model_version",
    "adaptation_context",
    "upstream_commit",
)
_TENSOR_RE = re.compile(
    r"^\s*(?:tensor\()?([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\)?\s*$"
)


@dataclass(frozen=True)
class KoInputRow:
    sent_id: str
    corpus_id: str
    text: str


def read_input_csv(path: Path) -> list[KoInputRow]:
    """Read the model-neutral input contract and reject ambiguous row identity."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(INPUT_COLUMNS):
            raise ValueError(f"Input columns must be exactly {INPUT_COLUMNS}")
        rows = [KoInputRow(**row) for row in reader]
    if not rows:
        raise ValueError("Ko input must contain at least one row")
    ids: set[str] = set()
    for index, row in enumerate(rows, start=2):
        if not row.sent_id or not row.corpus_id or not row.text.strip():
            raise ValueError(f"Empty required field at CSV row {index}")
        if row.sent_id in ids:
            raise ValueError(f"Duplicate sent_id: {row.sent_id}")
        if "\n" in row.text or "\r" in row.text:
            raise ValueError(f"Embedded newline is unsupported for sent_id {row.sent_id}")
        ids.add(row.sent_id)
    return rows


def write_legacy_target_bundle(rows: list[KoInputRow], output_dir: Path) -> None:
    """Write files expected by the unmodified official ``twitter`` code path.

    The legacy test loader reserves its first test row (``tv=1``) and emits no
    prediction for it. We prepend a duplicate of the first real sentence, then
    map the N predictions to the N input identities. The unlabeled adaptation
    file contains each real target sentence exactly once. Files named ``*l``
    and ``*v`` are non-semantic interface fillers: the official test function
    loads them but never uses them when writing predictions.
    """
    if not rows:
        raise ValueError("Cannot prepare an empty Ko target bundle")
    output_dir.mkdir(parents=True, exist_ok=True)
    test_texts = [rows[0].text, *(row.text for row in rows)]
    adaptation_texts = [row.text for row in rows]
    (output_dir / "twitters.txt").write_text(
        "\n".join(test_texts) + "\n", encoding="utf-8"
    )
    (output_dir / "twitteru.txt").write_text(
        "\n".join(adaptation_texts) + "\n", encoding="utf-8"
    )
    (output_dir / "twitterl.txt").write_text(
        "1\n" * len(test_texts), encoding="utf-8"
    )
    (output_dir / "twitterv.txt").write_text(
        "0.5\n" * len(test_texts), encoding="utf-8"
    )
    with (output_dir / "row_map.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["prediction_index", "sent_id", "corpus_id"])
        for index, row in enumerate(rows):
            writer.writerow([index, row.sent_id, row.corpus_id])


def parse_prediction_line(line: str) -> float:
    match = _TENSOR_RE.match(line)
    if not match:
        raise ValueError(f"Malformed Ko prediction: {line!r}")
    score = float(match.group(1))
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Ko prediction outside native [0,1] scale: {score}")
    return score


def parse_predictions(path: Path, expected_rows: int) -> list[float]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != expected_rows:
        raise ValueError(
            f"Ko output length mismatch: {len(lines)} scores for {expected_rows} input rows"
        )
    return [parse_prediction_line(line) for line in lines]


def write_model_scores(
    output_path: Path,
    rows: list[KoInputRow],
    scores: list[float],
    config: KoSpecificityConfig,
    adaptation_context: str,
) -> None:
    if len(rows) != len(scores):
        raise ValueError("Row/score count mismatch")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row, score in zip(rows, scores):
            writer.writerow(
                {
                    "sent_id": row.sent_id,
                    "corpus_id": row.corpus_id,
                    "model_id": config.model_id,
                    "score_raw": format(score, ".10g"),
                    "score_min": config.score_min,
                    "score_max": config.score_max,
                    "score_direction": config.score_direction,
                    "model_version": config.model_version,
                    "adaptation_context": adaptation_context,
                    "upstream_commit": config.upstream_commit,
                }
            )
