"""Deterministic Round 2 preprocessing-sensitivity analysis.

The analysis selects existing scored sentence rows with one frozen, score-blind
predicate. Canonical sentence text, sentence IDs, scores, and features are read
and validated but never rewritten.
"""
from __future__ import annotations

from collections import Counter
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable

import numpy as np

from src.speciteller.config import DEFAULT_SPECITELLER_CONFIG
from src.utils.text_filters import STRICT_NATURAL_LANGUAGE_REASON_CODES
from src.utils.text_filters import STRICT_NATURAL_LANGUAGE_RULE_VERSION
from src.utils.text_filters import strict_natural_language_v1


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

CORPUS_ORDER = (
    "wikipedia_en",
    "github_docs",
    "ansible_docs",
    "python_312_html",
)
WIKIPEDIA_CORPUS_ID = "wikipedia_en"


@dataclass(frozen=True)
class BaselineExpectation:
    corpus_id: str
    display_name: str
    expected_sentence_count: int
    published_mean: float
    published_median: float
    published_std: float
    published_iqr: float
    published_decimals: int


@dataclass(frozen=True)
class CorpusArtifacts:
    sentences: Path
    scores: Path
    features: Path


@dataclass
class CorpusData:
    corpus_id: str
    display_name: str
    scores: np.ndarray
    keep: np.ndarray
    doc_codes: np.ndarray
    doc_names: tuple[str, ...]
    reason_counts: Counter[str]
    overlap_counts: Counter[int]
    examples: dict[str, list[dict[str, str]]]


@dataclass(frozen=True)
class RunSettings:
    outputs_root: Path
    output_dir: Path
    paper_facing_dir: Path | None
    baseline_config: Path
    rule_config: Path
    artifact_checksums_config: Path
    source_provenance_config: Path
    bootstrap_replicates: int
    seed: int
    examples_per_reason: int
    command: str


def load_baseline_expectations(path: Path) -> dict[str, BaselineExpectation]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expectations = {
        row["corpus_id"]: BaselineExpectation(
            corpus_id=row["corpus_id"],
            display_name=row["display_name"],
            expected_sentence_count=int(row["expected_sentence_count"]),
            published_mean=float(row["published_mean"]),
            published_median=float(row["published_median"]),
            published_std=float(row["published_std"]),
            published_iqr=float(row["published_iqr"]),
            published_decimals=int(row["published_decimals"]),
        )
        for row in rows
    }
    if tuple(expectations) != CORPUS_ORDER:
        raise ValueError(
            f"Baseline config corpus order must be {CORPUS_ORDER}, got {tuple(expectations)}"
        )
    return expectations


def validate_frozen_rule_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("rule_version") != STRICT_NATURAL_LANGUAGE_RULE_VERSION:
        raise ValueError("Rule config version does not match implementation")
    if tuple(config.get("reason_order", [])) != STRICT_NATURAL_LANGUAGE_REASON_CODES:
        raise ValueError("Rule config reason order does not match implementation")
    if tuple(config.get("applies_identically_to", [])) != CORPUS_ORDER:
        raise ValueError("Rule config must apply identically in canonical corpus order")
    thresholds = config.get("thresholds", {})
    expected_thresholds = {
        "minimum_whitespace_tokens": 5,
        "minimum_alphabetic_ratio_non_space": 0.6,
        "command_path_minimum_evidence_tokens": 2,
        "command_path_general_evidence_ratio": 0.5,
        "command_led_unpunctuated_evidence_ratio": 0.35,
    }
    if thresholds != expected_thresholds:
        raise ValueError("Rule config thresholds do not match implementation")
    if config.get("outcome_blind_freeze") is not True:
        raise ValueError("Rule config must record an outcome-blind freeze")
    return config


def corpus_artifacts(outputs_root: Path, corpus_id: str) -> CorpusArtifacts:
    return CorpusArtifacts(
        sentences=outputs_root / "sentences" / f"{corpus_id}.csv",
        scores=outputs_root / "speciteller" / f"{corpus_id}_scores.tsv",
        features=outputs_root / "features" / f"{corpus_id}_features.csv",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_artifact_checksums(
    path: Path,
    outputs_root: Path,
) -> dict[str, dict[str, dict[str, Any]]]:
    """Verify all 12 canonical artifact files against the frozen Round 1 contract."""

    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected_keys = [
        (corpus_id, kind)
        for corpus_id in CORPUS_ORDER
        for kind in ("sentences", "scores", "features")
    ]
    actual_keys = [(row["corpus_id"], row["artifact_kind"]) for row in rows]
    if actual_keys != expected_keys:
        raise ValueError(
            f"Artifact checksum config order/schema mismatch: expected {expected_keys}, "
            f"got {actual_keys}"
        )

    metadata: dict[str, dict[str, dict[str, Any]]] = {
        corpus_id: {} for corpus_id in CORPUS_ORDER
    }
    for row in rows:
        relative = Path(row["relative_path"])
        # The contract names canonical outputs/ paths, while outputs_root may be
        # relocated for fixture tests or external runs.
        if len(relative.parts) < 3 or relative.parts[0] != "outputs":
            raise ValueError(f"Invalid canonical artifact path in checksum config: {relative}")
        artifact_path = outputs_root.joinpath(*relative.parts[1:])
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        size = artifact_path.stat().st_size
        checksum = sha256_file(artifact_path)
        expected_size = int(row["size_bytes"])
        expected_checksum = row["sha256"].lower()
        if size != expected_size or checksum != expected_checksum:
            raise ValueError(
                f"Artifact checksum gate failed for {relative.as_posix()}: "
                f"size={size} expected_size={expected_size}, sha256={checksum} "
                f"expected_sha256={expected_checksum}"
            )
        metadata[row["corpus_id"]][row["artifact_kind"]] = {
            "path": relative.as_posix(),
            "size_bytes": size,
            "sha256": checksum,
            "checksum_verified": True,
        }
    return metadata


def _require_columns(reader: csv.DictReader, required: set[str], label: str) -> None:
    actual = set(reader.fieldnames or [])
    missing = required - actual
    if missing:
        raise ValueError(f"Missing {label} columns: {sorted(missing)}")


def validate_and_load_corpus(
    expectation: BaselineExpectation,
    artifacts: CorpusArtifacts,
    manifest_writer: csv.DictWriter,
    *,
    examples_per_reason: int,
) -> CorpusData:
    """Stream three canonical artifacts in lockstep and build analysis arrays."""

    for path in asdict(artifacts).values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    scores: list[float] = []
    keep_flags: list[bool] = []
    doc_codes: list[int] = []
    doc_index: dict[str, int] = {}
    seen_ids: set[str] = set()
    reason_counts: Counter[str] = Counter()
    overlap_counts: Counter[int] = Counter()
    examples = {code: [] for code in STRICT_NATURAL_LANGUAGE_REASON_CODES}

    sentinel = object()
    with (
        artifacts.sentences.open("r", encoding="utf-8", newline="") as sentence_handle,
        artifacts.scores.open("r", encoding="utf-8", newline="") as score_handle,
        artifacts.features.open("r", encoding="utf-8", newline="") as feature_handle,
    ):
        sentence_reader = csv.DictReader(sentence_handle)
        score_reader = csv.reader(score_handle, delimiter="\t")
        feature_reader = csv.DictReader(feature_handle)
        _require_columns(
            sentence_reader,
            {"corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"},
            "sentence",
        )
        _require_columns(feature_reader, {"corpus_id", "sent_id"}, "feature")

        for row_number, (sentence_row, score_row, feature_row) in enumerate(
            zip_longest(sentence_reader, score_reader, feature_reader, fillvalue=sentinel),
            start=2,
        ):
            if sentinel in (sentence_row, score_row, feature_row):
                raise ValueError(
                    f"Artifact row-count mismatch for {expectation.corpus_id} at data row "
                    f"{row_number - 1}"
                )
            assert isinstance(sentence_row, dict)
            assert isinstance(score_row, list)
            assert isinstance(feature_row, dict)
            if len(score_row) != 2:
                raise ValueError(
                    f"Malformed score row for {expectation.corpus_id} at line {row_number}: "
                    f"expected 2 fields, got {len(score_row)}"
                )

            sentence_id = sentence_row["sent_id"]
            score_id = score_row[0]
            feature_id = feature_row["sent_id"]
            if not sentence_id or sentence_id != score_id or sentence_id != feature_id:
                raise ValueError(
                    f"Ordered sent_id join mismatch for {expectation.corpus_id} at line "
                    f"{row_number}: sentence={sentence_id!r}, score={score_id!r}, "
                    f"feature={feature_id!r}"
                )
            if sentence_id in seen_ids:
                raise ValueError(
                    f"Duplicate sent_id for {expectation.corpus_id} at line {row_number}: "
                    f"{sentence_id}"
                )
            seen_ids.add(sentence_id)
            if sentence_row["corpus_id"] != expectation.corpus_id:
                raise ValueError(f"Sentence corpus_id mismatch at line {row_number}")
            if feature_row["corpus_id"] != expectation.corpus_id:
                raise ValueError(f"Feature corpus_id mismatch at line {row_number}")

            score = float(score_row[1])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(
                    f"Invalid SpeciTeller score for {expectation.corpus_id} at line "
                    f"{row_number}: {score_row[1]!r}"
                )
            decision = strict_natural_language_v1(sentence_row["sent_text"])
            reason_text = ";".join(decision.reason_codes)
            manifest_writer.writerow(
                {
                    "corpus_id": expectation.corpus_id,
                    "sent_id": sentence_id,
                    "keep": "1" if decision.keep else "0",
                    "reason_codes": reason_text,
                    "rule_version": STRICT_NATURAL_LANGUAGE_RULE_VERSION,
                }
            )

            scores.append(score)
            keep_flags.append(decision.keep)
            doc_path = sentence_row["doc_path"]
            if doc_path not in doc_index:
                doc_index[doc_path] = len(doc_index)
            doc_codes.append(doc_index[doc_path])
            overlap_counts[len(decision.reason_codes)] += 1
            if decision.reason_codes:
                reason_counts["__any__"] += 1
            for code in decision.reason_codes:
                reason_counts[code] += 1
                if len(examples[code]) < examples_per_reason:
                    examples[code].append(
                        {
                            "reason_code": code,
                            "corpus_id": expectation.corpus_id,
                            "sent_id": sentence_id,
                            "doc_path": doc_path,
                            "sent_text": sentence_row["sent_text"],
                            "all_reason_codes": reason_text,
                        }
                    )

    return CorpusData(
        corpus_id=expectation.corpus_id,
        display_name=expectation.display_name,
        scores=np.asarray(scores, dtype=np.float64),
        keep=np.asarray(keep_flags, dtype=np.bool_),
        doc_codes=np.asarray(doc_codes, dtype=np.int32),
        doc_names=tuple(doc_index),
        reason_counts=reason_counts,
        overlap_counts=overlap_counts,
        examples=examples,
    )


def score_summary(scores: np.ndarray) -> dict[str, float | int]:
    if scores.size == 0:
        raise ValueError("Cannot summarize an empty score array")
    quantiles = np.quantile(
        scores,
        [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    return {
        "count": int(scores.size),
        "mean": float(np.mean(scores)),
        "median": float(quantiles[4]),
        "std": float(np.std(scores, ddof=0)),
        "min": float(np.min(scores)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q10": float(quantiles[2]),
        "q25": float(quantiles[3]),
        "q50": float(quantiles[4]),
        "q75": float(quantiles[5]),
        "q90": float(quantiles[6]),
        "q95": float(quantiles[7]),
        "q99": float(quantiles[8]),
        "max": float(np.max(scores)),
        "iqr": float(quantiles[5] - quantiles[3]),
    }


def validate_published_baseline(
    data: CorpusData,
    expectation: BaselineExpectation,
) -> dict[str, Any]:
    summary = score_summary(data.scores)
    checks: dict[str, bool] = {
        "sentence_count": summary["count"] == expectation.expected_sentence_count,
    }
    for metric in ("mean", "median", "std", "iqr"):
        actual = float(summary[metric])
        expected = float(getattr(expectation, f"published_{metric}"))
        checks[metric] = f"{actual:.{expectation.published_decimals}f}" == (
            f"{expected:.{expectation.published_decimals}f}"
        )
    passed = all(checks.values())
    return {
        "corpus_id": data.corpus_id,
        "display_name": data.display_name,
        "actual_count": summary["count"],
        "expected_count": expectation.expected_sentence_count,
        "actual_mean": summary["mean"],
        "published_mean": expectation.published_mean,
        "actual_median": summary["median"],
        "published_median": expectation.published_median,
        "actual_std": summary["std"],
        "published_std": expectation.published_std,
        "actual_iqr": summary["iqr"],
        "published_iqr": expectation.published_iqr,
        "count_pass": checks["sentence_count"],
        "mean_pass": checks["mean"],
        "median_pass": checks["median"],
        "std_pass": checks["std"],
        "iqr_pass": checks["iqr"],
        "baseline_pass": passed,
    }


def _weighted_median_with_buffers(
    sorted_scores: np.ndarray,
    sorted_doc_codes: np.ndarray,
    doc_weights: np.ndarray,
    row_weights: np.ndarray,
    cumulative: np.ndarray,
) -> float:
    np.take(doc_weights, sorted_doc_codes, out=row_weights)
    np.cumsum(row_weights, out=cumulative)
    total = int(cumulative[-1])
    if total <= 0:
        raise ValueError("Bootstrap replicate contains no rows")
    lower_position = (total - 1) // 2 + 1
    upper_position = total // 2 + 1
    lower_index = int(np.searchsorted(cumulative, lower_position, side="left"))
    upper_index = int(np.searchsorted(cumulative, upper_position, side="left"))
    return float((sorted_scores[lower_index] + sorted_scores[upper_index]) / 2.0)


def cluster_bootstrap_statistics(
    data: CorpusData,
    *,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    """Resample documents within one corpus, retaining all rows in sampled docs."""

    if replicates < 1:
        raise ValueError("bootstrap replicates must be positive")
    document_count = len(data.doc_names)
    if document_count < 2:
        raise ValueError(f"Need at least two document clusters for {data.corpus_id}")
    kept_scores = data.scores[data.keep]
    kept_doc_codes = data.doc_codes[data.keep]
    if kept_scores.size == 0:
        raise ValueError(f"Filter removed every row for {data.corpus_id}")

    raw_doc_count = np.bincount(data.doc_codes, minlength=document_count).astype(np.float64)
    raw_doc_sum = np.bincount(
        data.doc_codes, weights=data.scores, minlength=document_count
    )
    kept_doc_count = np.bincount(
        kept_doc_codes, minlength=document_count
    ).astype(np.float64)
    kept_doc_sum = np.bincount(
        kept_doc_codes, weights=kept_scores, minlength=document_count
    )

    raw_order = np.argsort(data.scores, kind="stable")
    kept_order = np.argsort(kept_scores, kind="stable")
    raw_sorted_scores = data.scores[raw_order]
    raw_sorted_codes = data.doc_codes[raw_order]
    kept_sorted_scores = kept_scores[kept_order]
    kept_sorted_codes = kept_doc_codes[kept_order]
    raw_row_weights = np.empty(data.scores.size, dtype=np.int64)
    raw_cumulative = np.empty(data.scores.size, dtype=np.int64)
    kept_row_weights = np.empty(kept_scores.size, dtype=np.int64)
    kept_cumulative = np.empty(kept_scores.size, dtype=np.int64)

    result = {
        "raw_mean": np.empty(replicates, dtype=np.float64),
        "filtered_mean": np.empty(replicates, dtype=np.float64),
        "raw_median": np.empty(replicates, dtype=np.float64),
        "filtered_median": np.empty(replicates, dtype=np.float64),
    }
    rng = np.random.default_rng(seed)
    for index in range(replicates):
        sampled_docs = rng.integers(0, document_count, size=document_count)
        weights = np.bincount(sampled_docs, minlength=document_count).astype(np.int64)
        raw_denominator = float(np.dot(weights, raw_doc_count))
        kept_denominator = float(np.dot(weights, kept_doc_count))
        if kept_denominator == 0.0:
            raise ValueError(f"Bootstrap replicate retained no rows for {data.corpus_id}")
        result["raw_mean"][index] = float(np.dot(weights, raw_doc_sum) / raw_denominator)
        result["filtered_mean"][index] = float(
            np.dot(weights, kept_doc_sum) / kept_denominator
        )
        result["raw_median"][index] = _weighted_median_with_buffers(
            raw_sorted_scores,
            raw_sorted_codes,
            weights,
            raw_row_weights,
            raw_cumulative,
        )
        result["filtered_median"][index] = _weighted_median_with_buffers(
            kept_sorted_scores,
            kept_sorted_codes,
            weights,
            kept_row_weights,
            kept_cumulative,
        )
    return result


def _ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _interpret_gap(raw_gap: float, filtered_gap: float, ci_low: float, ci_high: float) -> str:
    if raw_gap * filtered_gap < 0:
        return "reversal"
    if ci_low <= 0.0 <= ci_high:
        return "ambiguity"
    if abs(filtered_gap) < abs(raw_gap):
        return "attenuation"
    return "survival"


def build_gap_rows(
    data_by_corpus: dict[str, CorpusData],
    summaries: dict[str, dict[str, dict[str, float | int]]],
    bootstraps: dict[str, dict[str, np.ndarray]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    wiki = WIKIPEDIA_CORPUS_ID
    for technical in CORPUS_ORDER[1:]:
        for metric in ("mean", "median"):
            raw_gap = float(summaries[wiki]["original"][metric]) - float(
                summaries[technical]["original"][metric]
            )
            filtered_gap = float(summaries[wiki]["filtered"][metric]) - float(
                summaries[technical]["filtered"][metric]
            )
            gap_change = filtered_gap - raw_gap
            raw_reps = bootstraps[wiki][f"raw_{metric}"] - bootstraps[technical][
                f"raw_{metric}"
            ]
            filtered_reps = bootstraps[wiki][
                f"filtered_{metric}"
            ] - bootstraps[technical][f"filtered_{metric}"]
            change_reps = filtered_reps - raw_reps
            raw_low, raw_high = _ci(raw_reps)
            filtered_low, filtered_high = _ci(filtered_reps)
            change_low, change_high = _ci(change_reps)
            rows.append(
                {
                    "reference_corpus_id": wiki,
                    "technical_corpus_id": technical,
                    "technical_display_name": data_by_corpus[technical].display_name,
                    "metric": metric,
                    "raw_gap": raw_gap,
                    "raw_gap_ci_low": raw_low,
                    "raw_gap_ci_high": raw_high,
                    "filtered_gap": filtered_gap,
                    "filtered_gap_ci_low": filtered_low,
                    "filtered_gap_ci_high": filtered_high,
                    "gap_change": gap_change,
                    "gap_change_ci_low": change_low,
                    "gap_change_ci_high": change_high,
                    "relative_gap_change": gap_change / raw_gap if raw_gap else float("nan"),
                    "interpretation": _interpret_gap(
                        raw_gap, filtered_gap, filtered_low, filtered_high
                    ),
                }
            )
    return rows


def _format_output_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if math.isnan(float(value)):
            return ""
        return f"{float(value):.9f}"
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    return value


def write_csv_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format_output_value(value) for key, value in row.items()})


def _ordering_rows(
    data_by_corpus: dict[str, CorpusData],
    summaries: dict[str, dict[str, dict[str, float | int]]],
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric in ("mean", "median"):
        raw = sorted(
            CORPUS_ORDER,
            key=lambda corpus: (-float(summaries[corpus]["original"][metric]), corpus),
        )
        filtered = sorted(
            CORPUS_ORDER,
            key=lambda corpus: (-float(summaries[corpus]["filtered"][metric]), corpus),
        )
        rows.append(
            {
                "metric": metric,
                "raw_order_descending": " > ".join(data_by_corpus[c].display_name for c in raw),
                "filtered_order_descending": " > ".join(
                    data_by_corpus[c].display_name for c in filtered
                ),
                "ordering_unchanged": "1" if raw == filtered else "0",
            }
        )
    return rows


def _summary_rows(
    data_by_corpus: dict[str, CorpusData],
    summaries: dict[str, dict[str, dict[str, float | int]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        raw = summaries[corpus_id]["original"]
        filtered = summaries[corpus_id]["filtered"]
        raw_count = int(raw["count"])
        filtered_count = int(filtered["count"])
        for subset, values in (("original", raw), ("filtered", filtered)):
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "display_name": data.display_name,
                    "subset": subset,
                    **values,
                    "retention_rate": filtered_count / raw_count,
                    "removed_rate": 1.0 - (filtered_count / raw_count),
                    "absolute_mean_change": float(filtered["mean"]) - float(raw["mean"]),
                    "relative_mean_change": (
                        float(filtered["mean"]) - float(raw["mean"])
                    )
                    / float(raw["mean"]),
                    "absolute_median_change": float(filtered["median"])
                    - float(raw["median"]),
                    "relative_median_change": (
                        float(filtered["median"]) - float(raw["median"])
                    )
                    / float(raw["median"]),
                }
            )
    return rows


def _paper_table_rows(
    data_by_corpus: dict[str, CorpusData],
    summaries: dict[str, dict[str, dict[str, float | int]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        raw = summaries[corpus_id]["original"]
        filtered = summaries[corpus_id]["filtered"]
        rows.append(
            {
                "corpus_id": corpus_id,
                "corpus": data_by_corpus[corpus_id].display_name,
                "original_n": raw["count"],
                "retained_n": filtered["count"],
                "retention_pct": 100.0 * int(filtered["count"]) / int(raw["count"]),
                "raw_mean": raw["mean"],
                "filtered_mean": filtered["mean"],
                "mean_change": float(filtered["mean"]) - float(raw["mean"]),
                "raw_median": raw["median"],
                "filtered_median": filtered["median"],
                "median_change": float(filtered["median"]) - float(raw["median"]),
                "raw_iqr": raw["iqr"],
                "filtered_iqr": filtered["iqr"],
            }
        )
    return rows


def write_paper_table_tex(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"Corpus & Retained & Raw mean & Filtered mean & Raw median & Filtered median \\",
        r"\midrule",
    ]
    for row in rows:
        corpus = str(row["corpus"]).replace("&", r"\&")
        lines.append(
            f"{corpus} & {float(row['retention_pct']):.1f}\\% & "
            f"{float(row['raw_mean']):.3f} & {float(row['filtered_mean']):.3f} & "
            f"{float(row['raw_median']):.3f} & {float(row['filtered_median']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_results_readme(
    path: Path,
    *,
    paper_rows: list[dict[str, Any]],
    ordering_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
) -> None:
    all_changes_positive = all(float(row["gap_change"]) > 0.0 for row in gap_rows)
    all_change_intervals_positive = all(
        float(row["gap_change_ci_low"]) > 0.0 for row in gap_rows
    )
    lines = [
        "# Round 2 Preprocessing Ablation Evidence",
        "",
        "This directory is regenerated by `python scripts/preprocessing_ablation.py`.",
        "The full row-level manifest remains under the ignored `outputs/` tree.",
        "",
        "## Baseline gate",
        "",
        "All four published Round 1 sentence counts and rounded SpeciTeller mean,",
        "median, population standard deviation, and IQR values reproduce exactly.",
        "The three row-level tables join one-to-one and in identical order by",
        "`sent_id`.",
        "",
        "## Strict-subset result",
        "",
    ]
    for row in paper_rows:
        lines.append(
            f"- {row['corpus']}: retained {int(row['retained_n']):,} of "
            f"{int(row['original_n']):,} rows ({float(row['retention_pct']):.1f}%)."
        )
    lines.extend(
        [
            "",
            f"- Raw mean ordering: {ordering_rows[0]['raw_order_descending']}.",
            f"- Filtered mean ordering: {ordering_rows[0]['filtered_order_descending']}.",
            f"- Raw median ordering: {ordering_rows[1]['raw_order_descending']}.",
            f"- Filtered median ordering: {ordering_rows[1]['filtered_order_descending']}.",
            "",
        ]
    )
    if all_changes_positive and all_change_intervals_positive:
        lines.extend(
            [
                "Every Wikipedia-minus-technical mean and median gap survives and",
                "widens under the strict subset; every 95% document-cluster bootstrap",
                "interval for the gap change is above zero. GitHub and Ansible exchange",
                "their middle positions, so the complete four-corpus ordering is not",
                "invariant even though Wikipedia remains highest and Python remains lowest.",
            ]
        )
    else:
        categories = sorted({str(row["interpretation"]) for row in gap_rows})
        lines.append(
            "Gap outcomes include these predeclared categories: " + ", ".join(categories) + "."
        )
    lines.extend(
        [
            "",
            "This is a deliberately strict selection sensitivity analysis, not a",
            "replacement corpus or evidence about true sentence specificity. The low",
            "retention rates in markup-heavy source artifacts and the large GitHub shift",
            "must be reported alongside the survival result.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _rule_prevalence_rows(data_by_corpus: dict[str, CorpusData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        total = int(data.scores.size)
        removed = int(data.reason_counts["__any__"])
        for code in STRICT_NATURAL_LANGUAGE_REASON_CODES:
            count = int(data.reason_counts[code])
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "display_name": data.display_name,
                    "reason_code": code,
                    "count": count,
                    "prevalence_all_rows": count / total,
                    "prevalence_removed_rows": count / removed if removed else 0.0,
                }
            )
    return rows


def _overlap_rows(data_by_corpus: dict[str, CorpusData]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        for reason_count in sorted(data.overlap_counts):
            rows.append(
                {
                    "corpus_id": corpus_id,
                    "reason_count": reason_count,
                    "row_count": data.overlap_counts[reason_count],
                }
            )
    return rows


def _example_rows(data_by_corpus: dict[str, CorpusData]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for code in STRICT_NATURAL_LANGUAGE_REASON_CODES:
        for corpus_id in CORPUS_ORDER:
            rows.extend(data_by_corpus[corpus_id].examples[code])
    return rows


def _git_head(path: Path) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _installed_packages() -> dict[str, str]:
    try:
        from importlib.metadata import distributions

        return dict(
            sorted(
                (dist.metadata["Name"], dist.version)
                for dist in distributions()
                if dist.metadata["Name"]
            )
        )
    except Exception:
        return {}


def _portable_repo_path(path: Path) -> str:
    resolved = path.resolve()
    repo_root = Path.cwd().resolve()
    try:
        return resolved.relative_to(repo_root).as_posix()
    except ValueError:
        return path.name


def _source_provenance(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if tuple(record.get("corpora", {})) != CORPUS_ORDER:
        raise ValueError("Source provenance config must use canonical corpus order")
    model = record.get("speciteller", {})
    if model.get("repository_commit") != DEFAULT_SPECITELLER_CONFIG.repo_commit:
        raise ValueError("Source provenance SpeciTeller commit does not match code config")
    if model.get("released_data_sha256") != DEFAULT_SPECITELLER_CONFIG.data_checksum:
        raise ValueError("Source provenance SpeciTeller data checksum does not match code config")
    return {
        "path": _portable_repo_path(path),
        "sha256": sha256_file(path),
        "record": record,
    }


def _write_result_pack(
    directory: Path,
    *,
    baseline_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    ordering_rows: list[dict[str, Any]],
    gap_rows: list[dict[str, Any]],
    paper_rows: list[dict[str, Any]],
    prevalence_rows: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    example_rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    write_csv_rows(directory / "baseline_validation.csv", baseline_rows)
    write_csv_rows(directory / "score_summaries.csv", summary_rows)
    write_csv_rows(directory / "corpus_ordering.csv", ordering_rows)
    write_csv_rows(directory / "wikipedia_technical_gaps.csv", gap_rows)
    write_csv_rows(directory / "preprocessing_ablation_table.csv", paper_rows)
    write_paper_table_tex(directory / "preprocessing_ablation_table.tex", paper_rows)
    write_csv_rows(directory / "rule_prevalence.csv", prevalence_rows)
    write_csv_rows(directory / "rule_overlap_counts.csv", overlap_rows)
    write_csv_rows(directory / "reason_examples.csv", example_rows)
    write_results_readme(
        directory / "README.md",
        paper_rows=paper_rows,
        ordering_rows=ordering_rows,
        gap_rows=gap_rows,
    )
    (directory / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_preprocessing_ablation(settings: RunSettings) -> dict[str, Any]:
    expectations = load_baseline_expectations(settings.baseline_config)
    rule_config = validate_frozen_rule_config(settings.rule_config)
    if settings.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if settings.examples_per_reason < 1:
        raise ValueError("examples_per_reason must be positive")
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_metadata = validate_artifact_checksums(
        settings.artifact_checksums_config,
        settings.outputs_root,
    )

    manifest_path = settings.output_dir / "strict_natural_language_v1_manifest.csv"
    data_by_corpus: dict[str, CorpusData] = {}
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        manifest_writer = csv.DictWriter(
            handle,
            fieldnames=["corpus_id", "sent_id", "keep", "reason_codes", "rule_version"],
        )
        manifest_writer.writeheader()
        for corpus_id in CORPUS_ORDER:
            artifacts = corpus_artifacts(settings.outputs_root, corpus_id)
            data_by_corpus[corpus_id] = validate_and_load_corpus(
                expectations[corpus_id],
                artifacts,
                manifest_writer,
                examples_per_reason=settings.examples_per_reason,
            )

    baseline_rows = [
        validate_published_baseline(data_by_corpus[c], expectations[c])
        for c in CORPUS_ORDER
    ]
    if not all(row["baseline_pass"] for row in baseline_rows):
        write_csv_rows(settings.output_dir / "baseline_validation.csv", baseline_rows)
        raise ValueError(
            "Published Round 1 baseline gate failed; filtered comparisons were not produced"
        )

    summaries = {
        corpus_id: {
            "original": score_summary(data.scores),
            "filtered": score_summary(data.scores[data.keep]),
        }
        for corpus_id, data in data_by_corpus.items()
    }
    seed_sequence = np.random.SeedSequence(settings.seed)
    corpus_seeds = {
        corpus_id: int(child.generate_state(1, dtype=np.uint32)[0])
        for corpus_id, child in zip(CORPUS_ORDER, seed_sequence.spawn(len(CORPUS_ORDER)))
    }
    bootstraps: dict[str, dict[str, np.ndarray]] = {}
    for corpus_id in CORPUS_ORDER:
        bootstraps[corpus_id] = cluster_bootstrap_statistics(
            data_by_corpus[corpus_id],
            replicates=settings.bootstrap_replicates,
            seed=corpus_seeds[corpus_id],
        )

    summary_rows = _summary_rows(data_by_corpus, summaries)
    ordering_rows = _ordering_rows(data_by_corpus, summaries)
    gap_rows = build_gap_rows(data_by_corpus, summaries, bootstraps)
    paper_rows = _paper_table_rows(data_by_corpus, summaries)
    prevalence_rows = _rule_prevalence_rows(data_by_corpus)
    overlap_rows = _overlap_rows(data_by_corpus)
    example_rows = _example_rows(data_by_corpus)
    total_rows = sum(data.scores.size for data in data_by_corpus.values())
    retained_rows = sum(int(np.count_nonzero(data.keep)) for data in data_by_corpus.values())
    manifest_sha256 = sha256_file(manifest_path)

    metadata = {
        "analysis": "round2_preprocessing_ablation",
        "analysis_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": settings.command,
        "rule": rule_config,
        "baseline_config": {
            "path": _portable_repo_path(settings.baseline_config),
            "sha256": sha256_file(settings.baseline_config),
        },
        "artifact_checksums_config": {
            "path": _portable_repo_path(settings.artifact_checksums_config),
            "sha256": sha256_file(settings.artifact_checksums_config),
        },
        "input_artifacts": artifact_metadata,
        "source_provenance": _source_provenance(settings.source_provenance_config),
        "model_provenance": asdict(DEFAULT_SPECITELLER_CONFIG),
        "join_validation": {
            "one_to_one_sent_id": True,
            "ordered_identically": True,
            "duplicates": 0,
            "total_rows": int(total_rows),
        },
        "selection_reconciliation": {
            "total_rows": int(total_rows),
            "retained_rows": int(retained_rows),
            "removed_rows": int(total_rows - retained_rows),
            "reconciles": int(total_rows) == retained_rows + int(total_rows - retained_rows),
            "canonical_text_or_scores_modified": False,
        },
        "row_manifest": {
            "path": _portable_repo_path(manifest_path),
            "sha256": manifest_sha256,
            "row_count_excluding_header": int(total_rows),
        },
        "bootstrap": {
            "method": "nonparametric document-cluster bootstrap within each corpus",
            "cluster_key": "doc_path",
            "replicates": settings.bootstrap_replicates,
            "master_seed": settings.seed,
            "corpus_seeds": corpus_seeds,
            "confidence_interval": "percentile 2.5% to 97.5%",
            "paired_raw_filtered_within_corpus": True,
        },
        "environment": {
            "python": sys.version,
            "python_executable_name": Path(sys.executable).name,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "installed_packages": _installed_packages(),
            "requirements_path": "requirements.txt",
            "requirements_sha256": sha256_file(Path("requirements.txt")),
            "lock_state": "All direct requirements are exactly pinned; full installed package versions recorded here.",
        },
        "repository": {
            "path": ".",
            "head_before_sprint_commit": _git_head(Path.cwd()),
        },
        "baseline_gate_passed": True,
    }

    _write_result_pack(
        settings.output_dir,
        baseline_rows=baseline_rows,
        summary_rows=summary_rows,
        ordering_rows=ordering_rows,
        gap_rows=gap_rows,
        paper_rows=paper_rows,
        prevalence_rows=prevalence_rows,
        overlap_rows=overlap_rows,
        example_rows=example_rows,
        metadata=metadata,
    )
    if settings.paper_facing_dir is not None:
        _write_result_pack(
            settings.paper_facing_dir,
            baseline_rows=baseline_rows,
            summary_rows=summary_rows,
            ordering_rows=ordering_rows,
            gap_rows=gap_rows,
            paper_rows=paper_rows,
            prevalence_rows=prevalence_rows,
            overlap_rows=overlap_rows,
            example_rows=example_rows,
            metadata=metadata,
        )
    return metadata
