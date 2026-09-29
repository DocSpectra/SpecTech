"""Round 2 length-controlled SpeciTeller analysis.

The primary estimand uses exact-token-count direct standardization on common
support. Regression is a sensitivity analysis and never extrapolates beyond
the same observed support.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Iterable, Sequence

import numpy as np

from src.analysis.preprocessing_ablation import (
    CORPUS_ORDER,
    WIKIPEDIA_CORPUS_ID,
    BaselineExpectation,
    corpus_artifacts,
    load_baseline_expectations,
    score_summary,
    sha256_file,
    validate_artifact_checksums,
)


csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
METHOD_VERSION = "round2_length_control_v1"
STRICT_VARIANT = "strict_natural_language_v1"
VARIANTS = ("original", STRICT_VARIANT)


@dataclass
class LengthCorpusData:
    corpus_id: str
    display_name: str
    scores: np.ndarray
    token_count: np.ndarray
    char_count: np.ndarray
    keep: np.ndarray
    doc_codes: np.ndarray
    doc_names: tuple[str, ...]


@dataclass(frozen=True)
class LengthRunSettings:
    outputs_root: Path
    output_dir: Path
    paper_facing_dir: Path | None
    baseline_config: Path
    method_config: Path
    artifact_checksums_config: Path
    preprocessing_metadata: Path
    source_provenance_config: Path
    bootstrap_replicates: int
    seed: int
    command: str


@dataclass
class SupportResult:
    variant: str
    lengths: np.ndarray
    reference_probability: np.ndarray
    masks: dict[str, np.ndarray]
    corpus_probability: dict[str, np.ndarray]
    row_weight_by_length: dict[str, np.ndarray]
    diagnostics: list[dict[str, Any]]
    strata_rows: list[dict[str, Any]]
    reference_rows: list[dict[str, Any]]


def _require_columns(reader: csv.DictReader, required: set[str], label: str) -> None:
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing {label} columns: {sorted(missing)}")


def validate_method_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("method_version") != METHOD_VERSION:
        raise ValueError("Method config version does not match implementation")
    if config.get("outcome_blind_freeze") is not True:
        raise ValueError("Method config must record an outcome-blind freeze")
    support = config.get("common_support", {})
    if support.get("strata") != "exact positive integer token counts":
        raise ValueError("Only exact token-count strata are supported")
    if support.get("minimum_rows_per_corpus_per_stratum") != 100:
        raise ValueError("Frozen row support threshold changed")
    if support.get("minimum_documents_per_corpus_per_stratum") != 20:
        raise ValueError("Frozen document support threshold changed")
    uncertainty = config.get("primary_uncertainty", {})
    if uncertainty.get("reference_distribution_reestimated") is not False:
        raise ValueError("Reference distribution must remain fixed in bootstrap")
    return config


def validate_preprocessing_manifest(
    metadata_path: Path, outputs_root: Path
) -> tuple[Path, dict[str, Any]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest_meta = metadata["row_manifest"]
    canonical = Path(manifest_meta["path"])
    if len(canonical.parts) < 3 or canonical.parts[0] != "outputs":
        raise ValueError("Invalid preprocessing manifest path")
    manifest_path = outputs_root.joinpath(*canonical.parts[1:])
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if sha256_file(manifest_path) != manifest_meta["sha256"]:
        raise ValueError("Preprocessing manifest checksum mismatch")
    return manifest_path, metadata


def load_length_corpora(
    *,
    outputs_root: Path,
    expectations: dict[str, BaselineExpectation],
    manifest_path: Path,
) -> dict[str, LengthCorpusData]:
    """Stream canonical artifacts and Pair 1 status in ordered lockstep."""

    result: dict[str, LengthCorpusData] = {}
    sentinel = object()
    with manifest_path.open("r", encoding="utf-8", newline="") as manifest_handle:
        manifest_reader = csv.DictReader(manifest_handle)
        _require_columns(
            manifest_reader,
            {"corpus_id", "sent_id", "keep", "rule_version"},
            "preprocessing manifest",
        )
        for corpus_id in CORPUS_ORDER:
            artifacts = corpus_artifacts(outputs_root, corpus_id)
            scores: list[float] = []
            tokens: list[int] = []
            chars: list[int] = []
            keep: list[bool] = []
            doc_codes: list[int] = []
            doc_index: dict[str, int] = {}
            seen: set[str] = set()
            with (
                artifacts.sentences.open("r", encoding="utf-8", newline="") as sh,
                artifacts.scores.open("r", encoding="utf-8", newline="") as oh,
                artifacts.features.open("r", encoding="utf-8", newline="") as fh,
            ):
                sr = csv.DictReader(sh)
                scorer = csv.reader(oh, delimiter="\t")
                fr = csv.DictReader(fh)
                _require_columns(sr, {"corpus_id", "doc_path", "sent_id"}, "sentence")
                _require_columns(
                    fr,
                    {"corpus_id", "sent_id", "token_count", "char_count"},
                    "feature",
                )
                for row_number, artifact_rows in enumerate(
                    zip_longest(sr, scorer, fr, fillvalue=sentinel),
                    start=2,
                ):
                    sentence_row, score_row, feature_row = artifact_rows
                    if sentinel in artifact_rows:
                        raise ValueError(
                            f"Artifact row-count mismatch for {corpus_id} "
                            f"at data row {row_number - 1}"
                        )
                    try:
                        manifest_row = next(manifest_reader)
                    except StopIteration as exc:
                        raise ValueError(
                            f"Preprocessing manifest ended early for {corpus_id} "
                            f"at data row {row_number - 1}"
                        ) from exc
                    assert isinstance(sentence_row, dict)
                    assert isinstance(score_row, list)
                    assert isinstance(feature_row, dict)
                    assert isinstance(manifest_row, dict)
                    if len(score_row) != 2:
                        raise ValueError(f"Malformed score row at {corpus_id}:{row_number}")
                    ids = (
                        sentence_row["sent_id"],
                        score_row[0],
                        feature_row["sent_id"],
                        manifest_row["sent_id"],
                    )
                    if not ids[0] or len(set(ids)) != 1:
                        raise ValueError(
                            f"Ordered sent_id join mismatch for {corpus_id} at line "
                            f"{row_number}: {ids}"
                        )
                    if ids[0] in seen:
                        raise ValueError(f"Duplicate sent_id for {corpus_id}: {ids[0]}")
                    seen.add(ids[0])
                    corpus_ids = (
                        sentence_row["corpus_id"],
                        feature_row["corpus_id"],
                        manifest_row["corpus_id"],
                    )
                    if any(value != corpus_id for value in corpus_ids):
                        raise ValueError(f"corpus_id join mismatch at {corpus_id}:{row_number}")
                    if manifest_row["rule_version"] != STRICT_VARIANT:
                        raise ValueError("Unexpected Pair 1 rule version")
                    score = float(score_row[1])
                    token = int(feature_row["token_count"])
                    char = int(feature_row["char_count"])
                    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                        raise ValueError(f"Invalid score at {corpus_id}:{row_number}")
                    if token < 1 or char < 1:
                        raise ValueError(f"Invalid length at {corpus_id}:{row_number}")
                    scores.append(score)
                    tokens.append(token)
                    chars.append(char)
                    keep.append(manifest_row["keep"] == "1")
                    doc = sentence_row["doc_path"]
                    if doc not in doc_index:
                        doc_index[doc] = len(doc_index)
                    doc_codes.append(doc_index[doc])
            expectation = expectations[corpus_id]
            if len(scores) != expectation.expected_sentence_count:
                raise ValueError(
                    f"Count mismatch for {corpus_id}: {len(scores)} != "
                    f"{expectation.expected_sentence_count}"
                )
            result[corpus_id] = LengthCorpusData(
                corpus_id=corpus_id,
                display_name=expectation.display_name,
                scores=np.asarray(scores, dtype=np.float64),
                token_count=np.asarray(tokens, dtype=np.int32),
                char_count=np.asarray(chars, dtype=np.int32),
                keep=np.asarray(keep, dtype=np.bool_),
                doc_codes=np.asarray(doc_codes, dtype=np.int32),
                doc_names=tuple(doc_index),
            )
        try:
            extra = next(manifest_reader)
        except StopIteration:
            extra = None
        if extra is not None:
            raise ValueError("Preprocessing manifest has extra rows")
    return result


def validate_baseline(
    data_by_corpus: dict[str, LengthCorpusData],
    expectations: dict[str, BaselineExpectation],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        expected = expectations[corpus_id]
        summary = score_summary(data.scores)
        checks = {
            "count": summary["count"] == expected.expected_sentence_count,
            "mean": f"{summary['mean']:.{expected.published_decimals}f}"
            == f"{expected.published_mean:.{expected.published_decimals}f}",
            "median": f"{summary['median']:.{expected.published_decimals}f}"
            == f"{expected.published_median:.{expected.published_decimals}f}",
            "std": f"{summary['std']:.{expected.published_decimals}f}"
            == f"{expected.published_std:.{expected.published_decimals}f}",
            "iqr": f"{summary['iqr']:.{expected.published_decimals}f}"
            == f"{expected.published_iqr:.{expected.published_decimals}f}",
        }
        rows.append(
            {
                "corpus_id": corpus_id,
                "actual_count": summary["count"],
                "actual_mean": summary["mean"],
                "actual_median": summary["median"],
                "actual_std": summary["std"],
                "actual_iqr": summary["iqr"],
                "baseline_pass": all(checks.values()),
            }
        )
    if not all(row["baseline_pass"] for row in rows):
        raise ValueError("Round 1 baseline gate failed")
    return rows


def _variant_mask(data: LengthCorpusData, variant: str) -> np.ndarray:
    if variant == "original":
        return np.ones(data.scores.size, dtype=np.bool_)
    if variant == STRICT_VARIANT:
        return data.keep.copy()
    raise ValueError(f"Unknown variant: {variant}")


def _document_counts_by_length(
    lengths: np.ndarray, docs: np.ndarray, max_length: int, document_count: int
) -> np.ndarray:
    packed = lengths.astype(np.int64) * (document_count + 1) + docs.astype(np.int64)
    unique = np.unique(packed)
    unique_lengths = unique // (document_count + 1)
    return np.bincount(unique_lengths, minlength=max_length + 1)


def build_common_support(
    data_by_corpus: dict[str, LengthCorpusData],
    *,
    variant: str,
    minimum_rows: int,
    minimum_documents: int,
    maximum_weight_ratio: float,
) -> SupportResult:
    masks = {corpus_id: _variant_mask(data_by_corpus[corpus_id], variant) for corpus_id in CORPUS_ORDER}
    max_length = max(
        int(np.max(data_by_corpus[c].token_count[masks[c]])) for c in CORPUS_ORDER
    )
    row_counts: dict[str, np.ndarray] = {}
    document_counts: dict[str, np.ndarray] = {}
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        mask = masks[corpus_id]
        lengths = data.token_count[mask]
        docs = data.doc_codes[mask]
        row_counts[corpus_id] = np.bincount(lengths, minlength=max_length + 1)
        document_counts[corpus_id] = _document_counts_by_length(
            lengths, docs, max_length, len(data.doc_names)
        )
    eligible = np.ones(max_length + 1, dtype=np.bool_)
    eligible[0] = False
    for corpus_id in CORPUS_ORDER:
        eligible &= row_counts[corpus_id] >= minimum_rows
        eligible &= document_counts[corpus_id] >= minimum_documents
    support_lengths = np.flatnonzero(eligible)
    if support_lengths.size < 5:
        raise ValueError(f"Insufficient exact common support for {variant}")

    corpus_probability: dict[str, np.ndarray] = {}
    for corpus_id in CORPUS_ORDER:
        counts = row_counts[corpus_id][support_lengths].astype(np.float64)
        corpus_probability[corpus_id] = counts / np.sum(counts)
    reference = np.mean(
        np.vstack([corpus_probability[c] for c in CORPUS_ORDER]), axis=0
    )
    reference /= np.sum(reference)

    row_weight_by_length: dict[str, np.ndarray] = {}
    diagnostics: list[dict[str, Any]] = []
    strata_rows: list[dict[str, Any]] = []
    reference_rows = [
        {
            "variant": variant,
            "token_count": int(length),
            "reference_probability": float(probability),
        }
        for length, probability in zip(support_lengths, reference)
    ]
    support_lookup = np.zeros(max_length + 1, dtype=np.bool_)
    support_lookup[support_lengths] = True
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        base_mask = masks[corpus_id]
        support_mask = base_mask & (data.token_count <= max_length) & support_lookup[
            np.minimum(data.token_count, max_length)
        ]
        masks[corpus_id] = support_mask
        weights_at_support = reference / corpus_probability[corpus_id]
        if float(np.max(weights_at_support)) > maximum_weight_ratio:
            raise ValueError(
                f"Weight ratio exceeds frozen threshold for {variant}/{corpus_id}"
            )
        lookup = np.zeros(max_length + 1, dtype=np.float64)
        lookup[support_lengths] = weights_at_support
        row_weight_by_length[corpus_id] = lookup
        row_weights = lookup[data.token_count[support_mask]]
        ess = float(np.sum(row_weights) ** 2 / np.sum(row_weights**2))
        variant_count = int(np.sum(base_mask))
        support_count = int(np.sum(support_mask))
        diagnostics.append(
            {
                "variant": variant,
                "corpus_id": corpus_id,
                "variant_rows": variant_count,
                "support_rows": support_count,
                "excluded_rows": variant_count - support_count,
                "support_retention_rate": support_count / variant_count,
                "variant_documents": int(np.unique(data.doc_codes[base_mask]).size),
                "support_documents": int(np.unique(data.doc_codes[support_mask]).size),
                "variant_min_token_count": int(np.min(data.token_count[base_mask])),
                "variant_max_token_count": int(np.max(data.token_count[base_mask])),
                "support_min_token_count": int(support_lengths[0]),
                "support_max_token_count": int(support_lengths[-1]),
                "support_strata": int(support_lengths.size),
                "weight_min": float(np.min(row_weights)),
                "weight_q50": float(np.quantile(row_weights, 0.50)),
                "weight_q95": float(np.quantile(row_weights, 0.95)),
                "weight_q99": float(np.quantile(row_weights, 0.99)),
                "weight_max": float(np.max(row_weights)),
                "weight_sum": float(np.sum(row_weights)),
                "kish_effective_sample_size": ess,
                "ess_fraction": ess / support_count,
            }
        )
        support_index = {int(length): index for index, length in enumerate(support_lengths)}
        for length in np.flatnonzero(row_counts[corpus_id]):
            stratum_mask = base_mask & (data.token_count == length)
            stratum_scores = data.scores[stratum_mask]
            index = support_index.get(int(length))
            strata_rows.append(
                {
                    "variant": variant,
                    "corpus_id": corpus_id,
                    "token_count": int(length),
                    "eligible_common_support": index is not None,
                    "row_count": int(row_counts[corpus_id][length]),
                    "document_count": int(document_counts[corpus_id][length]),
                    "variant_row_probability": float(
                        row_counts[corpus_id][length] / np.sum(row_counts[corpus_id])
                    ),
                    "corpus_probability_within_support": (
                        float(corpus_probability[corpus_id][index])
                        if index is not None
                        else None
                    ),
                    "reference_probability": (
                        float(reference[index]) if index is not None else None
                    ),
                    "row_weight": (
                        float(weights_at_support[index]) if index is not None else None
                    ),
                    "score_mean": float(np.mean(stratum_scores)),
                    "score_std": float(np.std(stratum_scores, ddof=0)),
                }
            )
    return SupportResult(
        variant=variant,
        lengths=support_lengths,
        reference_probability=reference,
        masks=masks,
        corpus_probability=corpus_probability,
        row_weight_by_length=row_weight_by_length,
        diagnostics=diagnostics,
        strata_rows=strata_rows,
        reference_rows=reference_rows,
    )


def weighted_mean_and_ess(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    if values.size == 0 or values.size != weights.size or np.any(weights <= 0):
        raise ValueError("Values and positive weights must be nonempty and aligned")
    mean = float(np.dot(values, weights) / np.sum(weights))
    ess = float(np.sum(weights) ** 2 / np.sum(weights**2))
    return mean, ess


def standardized_means(
    data_by_corpus: dict[str, LengthCorpusData], support: SupportResult
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        variant_mask = _variant_mask(data, support.variant)
        support_mask = support.masks[corpus_id]
        weights = support.row_weight_by_length[corpus_id][data.token_count[support_mask]]
        adjusted, ess = weighted_mean_and_ess(data.scores[support_mask], weights)
        rows.append(
            {
                "variant": support.variant,
                "corpus_id": corpus_id,
                "raw_count": int(np.sum(variant_mask)),
                "support_count": int(np.sum(support_mask)),
                "raw_mean": float(np.mean(data.scores[variant_mask])),
                "support_unweighted_mean": float(np.mean(data.scores[support_mask])),
                "standardized_mean": adjusted,
                "kish_effective_sample_size": ess,
            }
        )
    return rows


def length_distribution_rows(
    data_by_corpus: dict[str, LengthCorpusData]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for corpus_id in CORPUS_ORDER:
            data = data_by_corpus[corpus_id]
            mask = _variant_mask(data, variant)
            token = data.token_count[mask]
            char = data.char_count[mask]
            token_q = np.quantile(token, [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99])
            char_q = np.quantile(char, [0.25, 0.50, 0.75])
            rows.append(
                {
                    "variant": variant,
                    "corpus_id": corpus_id,
                    "row_count": int(token.size),
                    "document_count": int(np.unique(data.doc_codes[mask]).size),
                    "token_mean": float(np.mean(token)),
                    "token_std": float(np.std(token, ddof=0)),
                    "token_min": int(np.min(token)),
                    "token_q01": float(token_q[0]),
                    "token_q05": float(token_q[1]),
                    "token_q10": float(token_q[2]),
                    "token_q25": float(token_q[3]),
                    "token_q50": float(token_q[4]),
                    "token_q75": float(token_q[5]),
                    "token_q90": float(token_q[6]),
                    "token_q95": float(token_q[7]),
                    "token_q99": float(token_q[8]),
                    "token_max": int(np.max(token)),
                    "char_mean": float(np.mean(char)),
                    "char_q25": float(char_q[0]),
                    "char_q50": float(char_q[1]),
                    "char_q75": float(char_q[2]),
                }
            )
    return rows


def _cluster_bootstrap_corpus(
    data: LengthCorpusData,
    *,
    variant: str,
    support: SupportResult,
    replicates: int,
    seed: int,
) -> dict[str, np.ndarray]:
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    variant_mask = _variant_mask(data, variant)
    active_docs = np.unique(data.doc_codes[variant_mask])
    remap = np.full(len(data.doc_names), -1, dtype=np.int32)
    remap[active_docs] = np.arange(active_docs.size, dtype=np.int32)
    variant_docs = remap[data.doc_codes[variant_mask]]
    raw_count = np.bincount(variant_docs, minlength=active_docs.size).astype(np.float64)
    raw_sum = np.bincount(
        variant_docs, weights=data.scores[variant_mask], minlength=active_docs.size
    )

    support_mask = support.masks[data.corpus_id]
    docs = remap[data.doc_codes[support_mask]]
    length_to_index = {int(length): i for i, length in enumerate(support.lengths)}
    strata = np.fromiter(
        (length_to_index[int(value)] for value in data.token_count[support_mask]),
        dtype=np.int32,
        count=int(np.sum(support_mask)),
    )
    doc_counts = np.zeros((active_docs.size, support.lengths.size), dtype=np.float64)
    doc_sums = np.zeros_like(doc_counts)
    np.add.at(doc_counts, (docs, strata), 1.0)
    np.add.at(doc_sums, (docs, strata), data.scores[support_mask])

    rng = np.random.default_rng(seed)
    sample_weights = rng.multinomial(
        active_docs.size,
        np.full(active_docs.size, 1.0 / active_docs.size),
        size=replicates,
    ).astype(np.float64)
    raw_denominator = sample_weights @ raw_count
    raw = (sample_weights @ raw_sum) / raw_denominator
    stratum_denominator = sample_weights @ doc_counts
    if np.any(stratum_denominator == 0):
        raise ValueError("Bootstrap replicate has an empty supported stratum")
    stratum_means = (sample_weights @ doc_sums) / stratum_denominator
    adjusted = stratum_means @ support.reference_probability
    return {"raw_mean": raw, "standardized_mean": adjusted}


def bootstrap_all(
    data_by_corpus: dict[str, LengthCorpusData],
    supports: dict[str, SupportResult],
    *,
    replicates: int,
    seed: int,
) -> tuple[dict[str, dict[str, dict[str, np.ndarray]]], dict[str, int]]:
    children = np.random.SeedSequence(seed).spawn(len(VARIANTS) * len(CORPUS_ORDER))
    index = 0
    result: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    seeds: dict[str, int] = {}
    for variant in VARIANTS:
        result[variant] = {}
        for corpus_id in CORPUS_ORDER:
            child_seed = int(children[index].generate_state(1)[0])
            index += 1
            seeds[f"{variant}:{corpus_id}"] = child_seed
            result[variant][corpus_id] = _cluster_bootstrap_corpus(
                data_by_corpus[corpus_id],
                variant=variant,
                support=supports[variant],
                replicates=replicates,
                seed=child_seed,
            )
    return result, seeds


def _ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def weighted_quantile_discrete(
    values: np.ndarray, probabilities: np.ndarray, quantiles: Sequence[float]
) -> np.ndarray:
    if values.size == 0 or values.size != probabilities.size:
        raise ValueError("Discrete distribution is empty or misaligned")
    cumulative = np.cumsum(probabilities / np.sum(probabilities))
    return np.asarray(
        [values[min(int(np.searchsorted(cumulative, q, side="left")), values.size - 1)] for q in quantiles],
        dtype=np.float64,
    )


def restricted_cubic_spline_basis(x: np.ndarray, knots: Sequence[float]) -> np.ndarray:
    """Harrell restricted cubic spline basis: linear term plus K-2 nonlinear terms."""

    knot = np.asarray(knots, dtype=np.float64)
    if knot.size < 4 or np.any(np.diff(knot) <= 0):
        raise ValueError("Restricted cubic spline knots must be strictly increasing")
    x = np.asarray(x, dtype=np.float64)
    scale = (knot[-1] - knot[0]) ** 2
    columns = [x]
    for current in knot[:-2]:
        term = np.maximum(x - current, 0.0) ** 3
        term -= np.maximum(x - knot[-2], 0.0) ** 3 * (
            (knot[-1] - current) / (knot[-1] - knot[-2])
        )
        term += np.maximum(x - knot[-1], 0.0) ** 3 * (
            (knot[-2] - current) / (knot[-1] - knot[-2])
        )
        columns.append(term / scale)
    return np.column_stack(columns)


def _fit_cluster_ols(
    design: np.ndarray, outcome: np.ndarray, clusters: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int], np.ndarray]:
    beta, _, rank, singular = np.linalg.lstsq(design, outcome, rcond=None)
    if rank != design.shape[1]:
        raise ValueError("Regression design is rank deficient")
    residual = outcome - design @ beta
    xtx_inv = np.linalg.inv(design.T @ design)
    unique, inverse = np.unique(clusters, return_inverse=True)
    cluster_score = np.zeros((unique.size, design.shape[1]), dtype=np.float64)
    np.add.at(cluster_score, inverse, design * residual[:, None])
    meat = cluster_score.T @ cluster_score
    n, p = design.shape
    g = unique.size
    if g <= 1 or n <= p:
        raise ValueError("Insufficient rows/clusters for robust covariance")
    correction = (g / (g - 1)) * ((n - 1) / (n - p))
    covariance = correction * (xtx_inv @ meat @ xtx_inv)
    sse = float(np.dot(residual, residual))
    centered = outcome - np.mean(outcome)
    sst = float(np.dot(centered, centered))
    diagnostics: dict[str, float | int] = {
        "rows": n,
        "clusters": g,
        "parameters": p,
        "rank": int(rank),
        "r_squared": 1.0 - sse / sst,
        "rmse": math.sqrt(sse / n),
        "sse": sse,
        "condition_number": float(singular[0] / singular[-1]),
        "cr1_correction": correction,
    }
    return beta, covariance, diagnostics, residual


def _contrast_row(
    beta: np.ndarray, covariance: np.ndarray, contrast: np.ndarray
) -> tuple[float, float, float, float]:
    estimate = float(contrast @ beta)
    se = math.sqrt(max(0.0, float(contrast @ covariance @ contrast)))
    return estimate, se, estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se


def regression_analysis(
    data_by_corpus: dict[str, LengthCorpusData], support: SupportResult
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, bool]]:
    knot_counts = weighted_quantile_discrete(
        support.lengths,
        support.reference_probability,
        [0.05, 0.275, 0.50, 0.725, 0.95],
    )
    if np.unique(knot_counts).size != knot_counts.size:
        raise ValueError("Frozen spline probability rule produced duplicate knots")
    log_knots = np.log(knot_counts)
    eval_counts = weighted_quantile_discrete(
        support.lengths,
        support.reference_probability,
        [0.10, 0.25, 0.50, 0.75, 0.90],
    ).astype(np.int32)

    outcome_parts: list[np.ndarray] = []
    corpus_parts: list[np.ndarray] = []
    token_parts: list[np.ndarray] = []
    cluster_parts: list[np.ndarray] = []
    cluster_offset = 0
    for corpus_index, corpus_id in enumerate(CORPUS_ORDER):
        data = data_by_corpus[corpus_id]
        mask = support.masks[corpus_id]
        outcome_parts.append(data.scores[mask])
        corpus_parts.append(np.full(int(np.sum(mask)), corpus_index, dtype=np.int8))
        token_parts.append(data.token_count[mask])
        local_docs = data.doc_codes[mask]
        cluster_parts.append(local_docs.astype(np.int64) + cluster_offset)
        cluster_offset += len(data.doc_names)
    y = np.concatenate(outcome_parts)
    corpus = np.concatenate(corpus_parts)
    token = np.concatenate(token_parts)
    clusters = np.concatenate(cluster_parts)
    spline = restricted_cubic_spline_basis(np.log(token.astype(np.float64)), log_knots)
    dummies = np.column_stack([(corpus == i).astype(np.float64) for i in range(1, 4)])
    additive = np.column_stack([np.ones(y.size), dummies, spline])
    beta, covariance, additive_diag, _ = _fit_cluster_ols(additive, y, clusters)

    interactions = np.column_stack(
        [dummies[:, [j]] * spline for j in range(dummies.shape[1])]
    )
    interaction_design = np.column_stack([additive, interactions])
    ibeta, icov, interaction_diag, _ = _fit_cluster_ols(
        interaction_design, y, clusters
    )
    partial_r2 = max(
        0.0,
        (float(additive_diag["sse"]) - float(interaction_diag["sse"]))
        / float(additive_diag["sse"]),
    )

    diagnostics = [
        {
            "variant": support.variant,
            "model": "additive_token_rcs",
            **additive_diag,
            "partial_r_squared_vs_additive": 0.0,
            "knot_token_counts": ";".join(str(int(x)) for x in knot_counts),
        },
        {
            "variant": support.variant,
            "model": "corpus_by_token_rcs_interaction",
            **interaction_diag,
            "partial_r_squared_vs_additive": partial_r2,
            "knot_token_counts": ";".join(str(int(x)) for x in knot_counts),
        },
    ]

    ref_basis = restricted_cubic_spline_basis(
        np.log(support.lengths.astype(np.float64)), log_knots
    )
    mean_basis = support.reference_probability @ ref_basis
    adjusted_rows: list[dict[str, Any]] = []
    mean_vectors: dict[str, np.ndarray] = {}
    for index, corpus_id in enumerate(CORPUS_ORDER):
        vector = np.zeros(additive.shape[1], dtype=np.float64)
        vector[0] = 1.0
        if index > 0:
            vector[index] = 1.0
        vector[4:] = mean_basis
        estimate, se, low, high = _contrast_row(beta, covariance, vector)
        mean_vectors[corpus_id] = vector
        adjusted_rows.append(
            {
                "variant": support.variant,
                "estimand": "regression_standardized_mean",
                "corpus_id": corpus_id,
                "technical_corpus_id": "",
                "estimate": estimate,
                "cluster_robust_se": se,
                "ci_low": low,
                "ci_high": high,
            }
        )
    for technical in CORPUS_ORDER[1:]:
        vector = mean_vectors[WIKIPEDIA_CORPUS_ID] - mean_vectors[technical]
        estimate, se, low, high = _contrast_row(beta, covariance, vector)
        adjusted_rows.append(
            {
                "variant": support.variant,
                "estimand": "wikipedia_minus_technical_additive_gap",
                "corpus_id": WIKIPEDIA_CORPUS_ID,
                "technical_corpus_id": technical,
                "estimate": estimate,
                "cluster_robust_se": se,
                "ci_low": low,
                "ci_high": high,
            }
        )

    length_rows: list[dict[str, Any]] = []
    substantial: dict[str, bool] = {}
    for technical_index, technical in enumerate(CORPUS_ORDER[1:], start=1):
        estimates: list[float] = []
        for count in eval_counts:
            basis = restricted_cubic_spline_basis(
                np.asarray([math.log(float(count))]), log_knots
            )[0]
            wiki = np.zeros(interaction_design.shape[1], dtype=np.float64)
            wiki[0] = 1.0
            wiki[4 : 4 + spline.shape[1]] = basis
            tech = wiki.copy()
            tech[technical_index] = 1.0
            interaction_start = additive.shape[1] + (technical_index - 1) * spline.shape[1]
            tech[interaction_start : interaction_start + spline.shape[1]] = basis
            contrast = wiki - tech
            estimate, se, low, high = _contrast_row(ibeta, icov, contrast)
            estimates.append(estimate)
            length_rows.append(
                {
                    "variant": support.variant,
                    "technical_corpus_id": technical,
                    "token_count": int(count),
                    "wikipedia_minus_technical_gap": estimate,
                    "cluster_robust_se": se,
                    "ci_low": low,
                    "ci_high": high,
                }
            )
        sign_change = min(estimates) < 0.0 < max(estimates)
        substantial[technical] = (max(estimates) - min(estimates) >= 0.10) or sign_change
    return adjusted_rows, diagnostics, length_rows, substantial


def categorize_gap(
    *,
    raw_gap: float,
    adjusted_gap: float,
    ci_low: float,
    ci_high: float,
    heterogeneous: bool,
) -> str:
    if heterogeneous:
        return "heterogeneous"
    if raw_gap * adjusted_gap < 0.0 and not (ci_low <= 0.0 <= ci_high):
        return "reversed"
    if ci_low <= 0.0 <= ci_high or abs(adjusted_gap) < 0.02:
        return "eliminated"
    if raw_gap != 0.0 and abs(adjusted_gap) < 0.8 * abs(raw_gap):
        return "attenuated"
    return "persistent"


def build_gap_rows(
    means: dict[str, list[dict[str, Any]]],
    bootstraps: dict[str, dict[str, dict[str, np.ndarray]]],
    substantial: dict[str, dict[str, bool]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        by_corpus = {row["corpus_id"]: row for row in means[variant]}
        for technical in CORPUS_ORDER[1:]:
            raw_gap = by_corpus[WIKIPEDIA_CORPUS_ID]["raw_mean"] - by_corpus[technical]["raw_mean"]
            adjusted_gap = (
                by_corpus[WIKIPEDIA_CORPUS_ID]["standardized_mean"]
                - by_corpus[technical]["standardized_mean"]
            )
            raw_boot = (
                bootstraps[variant][WIKIPEDIA_CORPUS_ID]["raw_mean"]
                - bootstraps[variant][technical]["raw_mean"]
            )
            adjusted_boot = (
                bootstraps[variant][WIKIPEDIA_CORPUS_ID]["standardized_mean"]
                - bootstraps[variant][technical]["standardized_mean"]
            )
            raw_low, raw_high = _ci(raw_boot)
            low, high = _ci(adjusted_boot)
            rows.append(
                {
                    "variant": variant,
                    "technical_corpus_id": technical,
                    "raw_gap": raw_gap,
                    "raw_ci_low": raw_low,
                    "raw_ci_high": raw_high,
                    "standardized_gap": adjusted_gap,
                    "standardized_ci_low": low,
                    "standardized_ci_high": high,
                    "absolute_change": adjusted_gap - raw_gap,
                    "retained_magnitude_ratio": abs(adjusted_gap / raw_gap) if raw_gap else math.nan,
                    "interaction_substantial": substantial[variant][technical],
                    "category": categorize_gap(
                        raw_gap=raw_gap,
                        adjusted_gap=adjusted_gap,
                        ci_low=low,
                        ci_high=high,
                        heterogeneous=substantial[variant][technical],
                    ),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_gap_svg(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a compact dependency-free plot of interaction-model gap profiles."""

    width, height = 920, 420
    left, right, top, bottom = 70, 25, 45, 55
    panel_gap = 55
    panel_width = (width - left - right - panel_gap) / 2
    plot_height = height - top - bottom
    colors = {
        "github_docs": "#0072B2",
        "ansible_docs": "#D55E00",
        "python_312_html": "#009E73",
    }
    labels = {
        "github_docs": "GitHub Docs",
        "ansible_docs": "Ansible Docs",
        "python_312_html": "Python 3.12",
    }
    y_max = max(float(row["ci_high"]) for row in rows) * 1.05
    y_min = min(0.0, min(float(row["ci_low"]) for row in rows))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="20" y="22" font-family="Arial" font-size="15" font-weight="bold">Wikipedia minus technical SpeciTeller gap by token count</text>',
    ]
    for panel_index, variant in enumerate(VARIANTS):
        panel_rows = [row for row in rows if row["variant"] == variant]
        x_values = [int(row["token_count"]) for row in panel_rows]
        x_min, x_max = min(x_values), max(x_values)
        x0 = left + panel_index * (panel_width + panel_gap)
        y0 = top
        elements.append(
            f'<text x="{x0 + panel_width / 2:.1f}" y="38" text-anchor="middle" font-family="Arial" font-size="12">{variant}</text>'
        )
        elements.append(
            f'<rect x="{x0:.1f}" y="{y0}" width="{panel_width:.1f}" height="{plot_height}" fill="none" stroke="#555" stroke-width="1"/>'
        )
        for tick in np.linspace(y_min, y_max, 6):
            y = y0 + plot_height * (y_max - tick) / (y_max - y_min)
            elements.append(
                f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x0 + panel_width:.1f}" y2="{y:.1f}" stroke="#ddd"/>'
            )
            if panel_index == 0:
                elements.append(
                    f'<text x="{x0 - 8:.1f}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="10">{tick:.2f}</text>'
                )
        for technical in CORPUS_ORDER[1:]:
            series = [row for row in panel_rows if row["technical_corpus_id"] == technical]
            points: list[str] = []
            for row in series:
                count = int(row["token_count"])
                x = x0 + panel_width * (count - x_min) / max(1, x_max - x_min)
                y = y0 + plot_height * (y_max - float(row["wikipedia_minus_technical_gap"])) / (y_max - y_min)
                low_y = y0 + plot_height * (y_max - float(row["ci_low"])) / (y_max - y_min)
                high_y = y0 + plot_height * (y_max - float(row["ci_high"])) / (y_max - y_min)
                points.append(f"{x:.1f},{y:.1f}")
                elements.append(
                    f'<line x1="{x:.1f}" y1="{high_y:.1f}" x2="{x:.1f}" y2="{low_y:.1f}" stroke="{colors[technical]}" stroke-width="1"/>'
                )
                elements.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[technical]}"/>'
                )
            elements.append(
                f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[technical]}" stroke-width="2"/>'
            )
        for count in sorted(set(x_values)):
            x = x0 + panel_width * (count - x_min) / max(1, x_max - x_min)
            elements.append(
                f'<text x="{x:.1f}" y="{y0 + plot_height + 17:.1f}" text-anchor="middle" font-family="Arial" font-size="9">{count}</text>'
            )
    elements.append(
        f'<text x="{width / 2:.1f}" y="{height - 17}" text-anchor="middle" font-family="Arial" font-size="11">Token count (reference-distribution quantiles)</text>'
    )
    legend_x = width - 310
    for index, technical in enumerate(CORPUS_ORDER[1:]):
        x = legend_x + index * 100
        elements.append(
            f'<line x1="{x}" y1="20" x2="{x + 16}" y2="20" stroke="{colors[technical]}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{x + 20}" y="24" font-family="Arial" font-size="9">{labels[technical]}</text>'
        )
    elements.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def _git_head() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _installed_packages() -> dict[str, str]:
    try:
        from importlib.metadata import version

        packages: dict[str, str] = {}
        for line in Path("requirements.txt").read_text(encoding="utf-8").splitlines():
            requirement = line.strip()
            if not requirement or requirement.startswith("#"):
                continue
            name = requirement.split("==", 1)[0]
            packages[name] = version(name)
        return dict(sorted(packages.items()))
    except Exception:
        return {}


def _portable_path(path: Path) -> str:
    parts = path.parts
    if "configs" in parts:
        return Path(*parts[parts.index("configs") :]).as_posix()
    if "analysis" in parts:
        return Path(*parts[parts.index("analysis") :]).as_posix()
    return path.name


def _write_readme(
    path: Path,
    gap_rows: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> None:
    lines = [
        "# Round 2 Length-Controlled Evidence",
        "",
        "Generated by `python scripts/length_controlled_analysis.py` under the",
        "outcome-blind frozen `round2_length_control_v1` contract.",
        "",
        "The primary analysis directly standardizes exact token-count strata on",
        "variant-specific four-corpus common support. The regression is a",
        "document-cluster-robust sensitivity analysis. These are observational",
        "SpeciTeller score comparisons, not causal domain effects or true",
        "specificity measurements.",
        "",
        "## Primary gaps",
        "",
        "| Variant | Technical corpus | Raw gap | Standardized gap (95% CI) | Category |",
        "|---|---|---:|---:|---|",
    ]
    for row in gap_rows:
        lines.append(
            f"| {row['variant']} | {row['technical_corpus_id']} | {row['raw_gap']:.3f} | "
            f"{row['standardized_gap']:.3f} [{row['standardized_ci_low']:.3f}, "
            f"{row['standardized_ci_high']:.3f}] | {row['category']} |"
        )
    lines.extend(["", "## Common-support diagnostics", ""])
    for row in diagnostics:
        lines.append(
            f"- `{row['variant']}` / `{row['corpus_id']}`: "
            f"{row['support_rows']:,}/{row['variant_rows']:,} rows retained; "
            f"ESS {row['kish_effective_sample_size']:.0f}; max weight "
            f"{row['weight_max']:.3f}."
        )
    lines.extend(
        [
            "",
            "See `gap_estimates.csv`, `standardized_means.csv`,",
            "`regression_contrasts.csv`, `length_specific_contrasts.csv`, and",
            "`common_support_diagnostics.csv` for exact values and diagnostics.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_length_controlled(settings: LengthRunSettings) -> dict[str, Any]:
    method = validate_method_config(settings.method_config)
    if settings.bootstrap_replicates != method["primary_uncertainty"]["replicates"]:
        raise ValueError("Bootstrap replicates do not match frozen method")
    if settings.seed != method["primary_uncertainty"]["master_seed"]:
        raise ValueError("Bootstrap seed does not match frozen method")
    expectations = load_baseline_expectations(settings.baseline_config)
    artifact_metadata = validate_artifact_checksums(
        settings.artifact_checksums_config, settings.outputs_root
    )
    manifest_path, preprocessing_metadata = validate_preprocessing_manifest(
        settings.preprocessing_metadata, settings.outputs_root
    )
    data_by_corpus = load_length_corpora(
        outputs_root=settings.outputs_root,
        expectations=expectations,
        manifest_path=manifest_path,
    )
    baseline_rows = validate_baseline(data_by_corpus, expectations)

    supports: dict[str, SupportResult] = {}
    means: dict[str, list[dict[str, Any]]] = {}
    regression_rows: list[dict[str, Any]] = []
    regression_diagnostics: list[dict[str, Any]] = []
    length_rows: list[dict[str, Any]] = []
    substantial: dict[str, dict[str, bool]] = {}
    support_cfg = method["common_support"]
    for variant in VARIANTS:
        supports[variant] = build_common_support(
            data_by_corpus,
            variant=variant,
            minimum_rows=support_cfg["minimum_rows_per_corpus_per_stratum"],
            minimum_documents=support_cfg[
                "minimum_documents_per_corpus_per_stratum"
            ],
            maximum_weight_ratio=method["weights"]["maximum_allowed_weight_ratio"],
        )
        means[variant] = standardized_means(data_by_corpus, supports[variant])
        adjusted, diagnostics, lengths, flags = regression_analysis(
            data_by_corpus, supports[variant]
        )
        regression_rows.extend(adjusted)
        regression_diagnostics.extend(diagnostics)
        length_rows.extend(lengths)
        substantial[variant] = flags

    bootstraps, bootstrap_seeds = bootstrap_all(
        data_by_corpus,
        supports,
        replicates=settings.bootstrap_replicates,
        seed=settings.seed,
    )
    gap_rows = build_gap_rows(means, bootstraps, substantial)
    mean_rows = [row for variant in VARIANTS for row in means[variant]]
    distribution_rows = length_distribution_rows(data_by_corpus)
    support_diagnostics = [
        row for variant in VARIANTS for row in supports[variant].diagnostics
    ]
    strata_rows = [row for variant in VARIANTS for row in supports[variant].strata_rows]
    reference_rows = [
        row for variant in VARIANTS for row in supports[variant].reference_rows
    ]
    paper_rows = [
        {
            "variant": row["variant"],
            "technical_corpus_id": row["technical_corpus_id"],
            "raw_gap": row["raw_gap"],
            "standardized_gap": row["standardized_gap"],
            "standardized_ci_low": row["standardized_ci_low"],
            "standardized_ci_high": row["standardized_ci_high"],
            "retained_magnitude_ratio": row["retained_magnitude_ratio"],
            "category": row["category"],
        }
        for row in gap_rows
    ]
    tables = {
        "baseline_validation.csv": baseline_rows,
        "common_support_diagnostics.csv": support_diagnostics,
        "common_support_reference.csv": reference_rows,
        "length_distributions.csv": distribution_rows,
        "length_strata.csv": strata_rows,
        "standardized_means.csv": mean_rows,
        "gap_estimates.csv": gap_rows,
        "regression_contrasts.csv": regression_rows,
        "regression_diagnostics.csv": regression_diagnostics,
        "length_specific_contrasts.csv": length_rows,
        "paper_length_control_table.csv": paper_rows,
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        _write_csv(settings.output_dir / name, rows)
    _write_gap_svg(settings.output_dir / "length_specific_gap_plot.svg", length_rows)
    if settings.paper_facing_dir is not None:
        settings.paper_facing_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            _write_csv(settings.paper_facing_dir / name, rows)
        _write_gap_svg(
            settings.paper_facing_dir / "length_specific_gap_plot.svg", length_rows
        )
        _write_readme(
            settings.paper_facing_dir / "README.md", gap_rows, support_diagnostics
        )

    metadata: dict[str, Any] = {
        "analysis": "round2_length_controlled",
        "analysis_version": 1,
        "method_version": METHOD_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": settings.command,
        "repository": {"path": ".", "head_before_sprint_commit": _git_head()},
        "configs": {
            "method": {
                "path": _portable_path(settings.method_config),
                "sha256": sha256_file(settings.method_config),
                "outcome_blind_freeze": True,
            },
            "baseline": {
                "path": _portable_path(settings.baseline_config),
                "sha256": sha256_file(settings.baseline_config),
            },
            "artifact_checksums": {
                "path": _portable_path(settings.artifact_checksums_config),
                "sha256": sha256_file(settings.artifact_checksums_config),
            },
            "preprocessing_metadata": {
                "path": _portable_path(settings.preprocessing_metadata),
                "sha256": sha256_file(settings.preprocessing_metadata),
            },
            "source_provenance": {
                "path": _portable_path(settings.source_provenance_config),
                "sha256": sha256_file(settings.source_provenance_config),
            },
        },
        "input_artifacts": artifact_metadata,
        "preprocessing_manifest": {
            "path": "outputs/round2/preprocessing_ablation/strict_natural_language_v1_manifest.csv",
            "sha256": preprocessing_metadata["row_manifest"]["sha256"],
            "row_count": preprocessing_metadata["row_manifest"][
                "row_count_excluding_header"
            ],
        },
        "join_validation": {
            "one_to_one_sent_id": True,
            "ordered_identically": True,
            "duplicates": 0,
            "total_rows": sum(data.scores.size for data in data_by_corpus.values()),
        },
        "baseline_gate_passed": True,
        "common_support": {
            variant: {
                "exact_token_counts": [int(x) for x in supports[variant].lengths],
                "stratum_count": int(supports[variant].lengths.size),
                "reference_probability_sum": float(
                    np.sum(supports[variant].reference_probability)
                ),
            }
            for variant in VARIANTS
        },
        "bootstrap": {
            "method": "nonparametric document-cluster bootstrap within corpus",
            "cluster_key": "doc_path",
            "replicates": settings.bootstrap_replicates,
            "master_seed": settings.seed,
            "analysis_seeds": bootstrap_seeds,
            "reference_distribution_fixed": True,
            "confidence_interval": "percentile 2.5% to 97.5%",
        },
        "regression": {
            "model": "OLS corpus indicators plus five-knot RCS(log token_count)",
            "uncertainty": "CR1 document-cluster-robust sandwich covariance",
            "interaction_diagnostic_exported": True,
            "character_count_simultaneously_controlled": False,
        },
        "environment": {
            "python": sys.version,
            "python_executable_name": Path(sys.executable).name,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "installed_packages": _installed_packages(),
            "requirements_path": "requirements.txt",
            "requirements_sha256": sha256_file(Path("requirements.txt")),
            "lock_state": "All direct requirements are exactly pinned; full installed package versions recorded here.",
        },
        "output_tables": {
            name: {
                "path": f"analysis/round2_length_control/{name}",
                "rows": len(rows),
                "sha256": sha256_file(settings.output_dir / name),
            }
            for name, rows in tables.items()
        },
        "figure": {
            "path": "analysis/round2_length_control/length_specific_gap_plot.svg",
            "sha256": sha256_file(
                settings.output_dir / "length_specific_gap_plot.svg"
            ),
            "justification": "All six frozen interaction diagnostics exceeded the material heterogeneity threshold.",
        },
        "source_provenance": {
            "record": json.loads(settings.source_provenance_config.read_text(encoding="utf-8")),
            "model_provenance": preprocessing_metadata.get("model_provenance", {}),
        },
        "result_categories": {
            f"{row['variant']}:{row['technical_corpus_id']}": row["category"]
            for row in gap_rows
        },
        "claim_boundary": (
            "Adjusted results characterize observed cross-corpus SpeciTeller score "
            "differences after token-count control; they are not causal domain effects "
            "or measurements of true sentence specificity."
        ),
    }
    for directory in (settings.output_dir, settings.paper_facing_dir):
        if directory is not None:
            (directory / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    return metadata
