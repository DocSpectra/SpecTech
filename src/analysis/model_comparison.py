"""Model-aware comparison for SpeciTeller and the approved Ko release.

Scientific choices are frozen in ``ko_official_release_comparator_v1``.  The
Ko primary score is the row-wise arithmetic mean of three independently
trained target-adapted runs.  Native scales are retained; cross-model
diagnostics use ranks or within-model pooled standardization.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.analysis.length_controlled import (
    LengthCorpusData,
    VARIANTS,
    bootstrap_all,
    build_common_support,
    build_gap_rows,
    regression_analysis,
    standardized_means,
)
from src.analysis.preprocessing_ablation import (
    CORPUS_ORDER,
    WIKIPEDIA_CORPUS_ID,
    load_baseline_expectations,
    sha256_file,
    validate_artifact_checksums,
)
from src.ko_specificity.official_release import (
    SCORE_COLUMNS,
    OfficialReleaseProtocol,
    load_protocol,
    read_input_manifest,
)


SPECITELLER_ID = "speciteller_frozen_round1"
KO_ID = "ko_author_official_release_se_ad_mean_std"
MODEL_IDS = (SPECITELLER_ID, KO_ID)


@dataclass
class ComparisonCorpus:
    corpus_id: str
    display_name: str
    sent_ids: tuple[str, ...]
    doc_paths: tuple[str, ...]
    doc_codes: np.ndarray
    doc_names: tuple[str, ...]
    token_count: np.ndarray
    char_count: np.ndarray
    keep: np.ndarray
    speciteller: np.ndarray
    ko_runs: np.ndarray

    @property
    def ko_primary(self) -> np.ndarray:
        return aggregate_ko_runs(self.ko_runs)


@dataclass(frozen=True)
class ComparisonSettings:
    protocol_config: Path
    input_manifest: Path
    outputs_root: Path
    run_root: Path
    preprocessing_manifest: Path
    artifact_checksums: Path
    baseline_config: Path
    length_config: Path
    output_dir: Path
    paper_facing_dir: Path | None
    bootstrap_replicates: int
    bootstrap_seed: int
    command: str


def aggregate_ko_runs(run_scores: np.ndarray) -> np.ndarray:
    if run_scores.ndim != 2 or run_scores.shape[0] != 3:
        raise ValueError("Ko aggregation requires exactly three aligned runs")
    if not np.all(np.isfinite(run_scores)) or np.any((run_scores < 0) | (run_scores > 1)):
        raise ValueError("Invalid Ko run score")
    return np.mean(run_scores, axis=0)


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("Percentile ranks require finite one-dimensional values")
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    stops = np.r_[starts[1:], values.size]
    ranks = np.empty(values.size, dtype=np.float64)
    averages = (starts + 1 + stops) / 2.0
    ranks[order] = np.repeat(averages, stops - starts)
    return (ranks - 0.5) / values.size


def fixed_rank_spearman(x: np.ndarray, y: np.ndarray) -> float:
    if x.size != y.size or x.size < 2:
        raise ValueError("Spearman inputs must be aligned")
    return float(np.corrcoef(percentile_ranks(x), percentile_ranks(y))[0, 1])


def _seed(master: int, *parts: str) -> int:
    digest = hashlib.sha256((str(master) + "\0" + "\0".join(parts)).encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _cluster_mean_replicates(
    values: np.ndarray, doc_codes: np.ndarray, document_count: int, replicates: int, seed: int
) -> np.ndarray:
    counts = np.bincount(doc_codes, minlength=document_count).astype(np.float64)
    sums = np.bincount(doc_codes, weights=values, minlength=document_count)
    rng = np.random.default_rng(seed)
    output = np.empty(replicates, dtype=np.float64)
    probability = np.full(document_count, 1.0 / document_count)
    for start in range(0, replicates, 50):
        stop = min(start + 50, replicates)
        weights = rng.multinomial(document_count, probability, size=stop - start)
        output[start:stop] = (weights @ sums) / (weights @ counts)
    return output


def _cluster_rank_correlation_replicates(
    x_rank: np.ndarray,
    y_rank: np.ndarray,
    doc_codes: np.ndarray,
    document_count: int,
    replicates: int,
    seed: int,
) -> np.ndarray:
    count = np.bincount(doc_codes, minlength=document_count).astype(np.float64)
    sx = np.bincount(doc_codes, weights=x_rank, minlength=document_count)
    sy = np.bincount(doc_codes, weights=y_rank, minlength=document_count)
    sxx = np.bincount(doc_codes, weights=x_rank * x_rank, minlength=document_count)
    syy = np.bincount(doc_codes, weights=y_rank * y_rank, minlength=document_count)
    sxy = np.bincount(doc_codes, weights=x_rank * y_rank, minlength=document_count)
    probability = np.full(document_count, 1.0 / document_count)
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 50):
        stop = min(start + 50, replicates)
        weight = rng.multinomial(document_count, probability, size=stop - start)
        n = weight @ count
        x = weight @ sx
        y = weight @ sy
        xx = weight @ sxx
        yy = weight @ syy
        xy = weight @ sxy
        numerator = xy - x * y / n
        denominator = np.sqrt((xx - x * x / n) * (yy - y * y / n))
        result[start:stop] = numerator / denominator
    return result


def _ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def _required_columns(reader: csv.DictReader, required: set[str], label: str) -> None:
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"Missing {label} columns: {sorted(missing)}")


def load_comparison_data(
    *,
    settings: ComparisonSettings,
    protocol: OfficialReleaseProtocol,
) -> tuple[dict[str, ComparisonCorpus], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    expectations = load_baseline_expectations(settings.baseline_config)
    artifact_metadata = validate_artifact_checksums(
        settings.artifact_checksums, settings.outputs_root
    )
    input_manifest = read_input_manifest(settings.input_manifest, protocol)
    manifest_handle = settings.preprocessing_manifest.open(
        "r", encoding="utf-8", newline=""
    )
    manifest_reader = csv.DictReader(manifest_handle)
    _required_columns(
        manifest_reader, {"corpus_id", "sent_id", "keep", "rule_version"}, "strict manifest"
    )
    result: dict[str, ComparisonCorpus] = {}
    coverage: list[dict[str, Any]] = []
    run_metadata: dict[str, dict[str, Any]] = {}
    sentinel = object()
    try:
        for corpus_id in protocol.corpora:
            sentence_path = settings.outputs_root / "sentences" / f"{corpus_id}.csv"
            spec_path = settings.outputs_root / "speciteller" / f"{corpus_id}_scores.tsv"
            feature_path = settings.outputs_root / "features" / f"{corpus_id}_features.csv"
            run_paths = [settings.run_root / corpus_id / run_id / "scores.csv" for run_id in protocol.run_ids]
            handles = [path.open("r", encoding="utf-8", newline="") for path in run_paths]
            try:
                run_readers = [csv.DictReader(handle) for handle in handles]
                for reader in run_readers:
                    if tuple(reader.fieldnames or []) != SCORE_COLUMNS:
                        raise ValueError(f"Ko score schema mismatch for {corpus_id}")
                sent_ids: list[str] = []
                doc_paths: list[str] = []
                tokens: list[int] = []
                chars: list[int] = []
                keep: list[bool] = []
                spec: list[float] = []
                ko: list[list[float]] = [[] for _ in protocol.run_ids]
                doc_index: dict[str, int] = {}
                doc_codes: list[int] = []
                with (
                    sentence_path.open("r", encoding="utf-8", newline="") as sh,
                    spec_path.open("r", encoding="utf-8", newline="") as oh,
                    feature_path.open("r", encoding="utf-8", newline="") as fh,
                ):
                    sr = csv.DictReader(sh)
                    scorer = csv.reader(oh, delimiter="\t")
                    fr = csv.DictReader(fh)
                    _required_columns(sr, {"corpus_id", "sent_id", "doc_path"}, "sentence")
                    _required_columns(fr, {"corpus_id", "sent_id", "token_count", "char_count"}, "feature")
                    iterables: list[Iterable[Any]] = [sr, scorer, fr, *run_readers]
                    seen: set[str] = set()
                    for line, joined in enumerate(zip_longest(*iterables, fillvalue=sentinel), start=2):
                        if sentinel in joined:
                            raise ValueError(f"Ordered comparison join ended unevenly at {corpus_id}:{line}")
                        sentence, spec_row, feature, *run_rows = joined
                        try:
                            strict = next(manifest_reader)
                        except StopIteration as exc:
                            raise ValueError(
                                f"Strict manifest ended early at {corpus_id}:{line}"
                            ) from exc
                        ids = [sentence["sent_id"], spec_row[0], feature["sent_id"], strict["sent_id"]]
                        ids.extend(row["sent_id"] for row in run_rows)
                        if not ids[0] or len(set(ids)) != 1 or ids[0] in seen:
                            raise ValueError(f"Ordered sent_id join failure at {corpus_id}:{line}")
                        seen.add(ids[0])
                        corpus_values = [sentence["corpus_id"], feature["corpus_id"], strict["corpus_id"]]
                        corpus_values.extend(row["corpus_id"] for row in run_rows)
                        if any(value != corpus_id for value in corpus_values):
                            raise ValueError(f"Corpus join failure at {corpus_id}:{line}")
                        if strict["rule_version"] != "strict_natural_language_v1":
                            raise ValueError("Strict rule identity mismatch")
                        for run_id, row in zip(protocol.run_ids, run_rows):
                            if row["run_id"] != run_id:
                                raise ValueError("Ko run identity/order mismatch")
                            if row["model_id"] != KO_ID:
                                raise ValueError("Ko model identity mismatch")
                            if row["comparator_config_sha256"] != protocol.sha256:
                                raise ValueError("Ko config identity mismatch")
                            if row["adaptation_context"] != input_manifest[corpus_id]["adaptation_context"]:
                                raise ValueError("Ko adaptation context mismatch")
                            if row["input_manifest_sha256"] != input_manifest[corpus_id]["input_sha256"]:
                                raise ValueError("Ko input identity mismatch")
                            expected_training = f"ko_official_release_comparator_v1:{corpus_id}:{run_id}"
                            if row["training_run_id"] != expected_training:
                                raise ValueError("Ko training run identity mismatch")
                            if row["upstream_commit"] != protocol.record["identity"]["upstream_commit"]:
                                raise ValueError("Ko upstream commit mismatch")
                            if row["model_version"] != protocol.record["identity"]["model_version"]:
                                raise ValueError("Ko model version mismatch")
                        score = float(spec_row[1])
                        run_values = [float(row["score_raw"]) for row in run_rows]
                        if not math.isfinite(score) or not 0 <= score <= 1:
                            raise ValueError("Invalid SpeciTeller score")
                        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in run_values):
                            raise ValueError("Invalid Ko score")
                        sent_ids.append(ids[0])
                        doc = sentence["doc_path"]
                        doc_paths.append(doc)
                        if doc not in doc_index:
                            doc_index[doc] = len(doc_index)
                        doc_codes.append(doc_index[doc])
                        tokens.append(int(feature["token_count"]))
                        chars.append(int(feature["char_count"]))
                        keep.append(strict["keep"] == "1")
                        spec.append(score)
                        for index, value in enumerate(run_values):
                            ko[index].append(value)
                expected = expectations[corpus_id].expected_sentence_count
                if len(sent_ids) != expected or len(sent_ids) != int(input_manifest[corpus_id]["valid_row_count"]):
                    raise ValueError(f"Complete coverage gate failed for {corpus_id}")
                data = ComparisonCorpus(
                    corpus_id=corpus_id,
                    display_name=expectations[corpus_id].display_name,
                    sent_ids=tuple(sent_ids),
                    doc_paths=tuple(doc_paths),
                    doc_codes=np.asarray(doc_codes, dtype=np.int32),
                    doc_names=tuple(doc_index),
                    token_count=np.asarray(tokens, dtype=np.int32),
                    char_count=np.asarray(chars, dtype=np.int32),
                    keep=np.asarray(keep, dtype=np.bool_),
                    speciteller=np.asarray(spec, dtype=np.float64),
                    ko_runs=np.asarray(ko, dtype=np.float64),
                )
                result[corpus_id] = data
                coverage.append(
                    {
                        "corpus_id": corpus_id,
                        "canonical_rows": expected,
                        "speciteller_rows": data.speciteller.size,
                        "ko_rows_per_run": data.ko_runs.shape[1],
                        "ko_run_count": data.ko_runs.shape[0],
                        "strict_status_rows": data.keep.size,
                        "failed_rows": 0,
                        "coverage_rate": 1.0,
                        "ordered_join_passed": True,
                    }
                )
            finally:
                for handle in handles:
                    handle.close()
            for run_id in protocol.run_ids:
                metadata_path = settings.run_root / corpus_id / run_id / "run_metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["score_sha256"] != sha256_file(settings.run_root / corpus_id / run_id / "scores.csv"):
                    raise ValueError("Ko score checksum failure")
                run_metadata[f"{corpus_id}:{run_id}"] = {
                    **metadata,
                    "metadata_sha256": sha256_file(metadata_path),
                }
        if next(manifest_reader, None) is not None:
            raise ValueError("Strict manifest has extra rows")
    finally:
        manifest_handle.close()
    return result, coverage, {"artifacts": artifact_metadata, "runs": run_metadata}


def _model_values(data: ComparisonCorpus, model_id: str) -> np.ndarray:
    if model_id == SPECITELLER_ID:
        return data.speciteller
    if model_id == KO_ID:
        return data.ko_primary
    raise ValueError(model_id)


def _variant_mask(data: ComparisonCorpus, variant: str) -> np.ndarray:
    return np.ones(data.keep.size, dtype=np.bool_) if variant == "original" else data.keep


def summarize_models(
    data_by_corpus: dict[str, ComparisonCorpus], replicates: int, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    variability: list[dict[str, Any]] = []
    boots: dict[str, dict[str, dict[str, np.ndarray]]] = {model: {} for model in MODEL_IDS}
    pooled: dict[str, np.ndarray] = {
        model: np.concatenate([_model_values(data_by_corpus[c], model) for c in CORPUS_ORDER])
        for model in MODEL_IDS
    }
    pooled_stats = {
        model: (float(np.mean(values)), float(np.std(values, ddof=0)))
        for model, values in pooled.items()
    }
    disagreement_rows: list[dict[str, Any]] = []
    disagreement_summary: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        ko = data.ko_primary
        spec_rank = percentile_ranks(data.speciteller)
        ko_rank = percentile_ranks(ko)
        difference = np.abs(spec_rank - ko_rank)
        top_count = min(25, difference.size)
        top = np.argpartition(difference, -top_count)[-top_count:]
        top = top[np.argsort(-difference[top], kind="stable")]
        for rank_index, index in enumerate(top, start=1):
            disagreement_rows.append(
                {
                    "corpus_id": corpus_id,
                    "diagnostic_rank": rank_index,
                    "sent_id": data.sent_ids[int(index)],
                    "doc_path": data.doc_paths[int(index)],
                    "token_count": int(data.token_count[index]),
                    "strict_keep": bool(data.keep[index]),
                    "speciteller_score": float(data.speciteller[index]),
                    "ko_three_run_mean": float(ko[index]),
                    "speciteller_percentile": float(spec_rank[index]),
                    "ko_percentile": float(ko_rank[index]),
                    "absolute_percentile_difference": float(difference[index]),
                }
            )
        disagreement_summary.append(
            {
                "corpus_id": corpus_id,
                "row_count": difference.size,
                "mean_absolute_percentile_difference": float(np.mean(difference)),
                "median_absolute_percentile_difference": float(np.median(difference)),
                "q90_absolute_percentile_difference": float(np.quantile(difference, 0.9)),
                "q95_absolute_percentile_difference": float(np.quantile(difference, 0.95)),
                "share_difference_at_least_0_50": float(np.mean(difference >= 0.5)),
            }
        )
        for run_index in range(3):
            values = data.ko_runs[run_index]
            variability.append(
                {
                    "corpus_id": corpus_id,
                    "run_id": f"run{run_index + 1:02d}",
                    "row_count": values.size,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=0)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
        row_sd = np.std(data.ko_runs, axis=0, ddof=0)
        variability.append(
            {
                "corpus_id": corpus_id,
                "run_id": "across_run_row_sd",
                "row_count": row_sd.size,
                "mean": float(np.mean(row_sd)),
                "median": float(np.median(row_sd)),
                "std": float(np.std(row_sd, ddof=0)),
                "minimum": float(np.min(row_sd)),
                "maximum": float(np.max(row_sd)),
            }
        )
        for model_id in MODEL_IDS:
            values = _model_values(data, model_id)
            boots[model_id][corpus_id] = {}
            for variant in VARIANTS:
                mask = _variant_mask(data, variant)
                active_docs = np.unique(data.doc_codes[mask])
                remap = np.full(len(data.doc_names), -1, dtype=np.int32)
                remap[active_docs] = np.arange(active_docs.size)
                reps = _cluster_mean_replicates(
                    values[mask], remap[data.doc_codes[mask]], active_docs.size,
                    replicates, _seed(seed, model_id, corpus_id, variant, "mean"),
                )
                boots[model_id][corpus_id][variant] = reps
                mean = float(np.mean(values[mask]))
                median = float(np.median(values[mask]))
                pooled_mean, pooled_sd = pooled_stats[model_id]
                summaries.append(
                    {
                        "model_id": model_id,
                        "variant": variant,
                        "corpus_id": corpus_id,
                        "row_count": int(np.sum(mask)),
                        "mean_native": mean,
                        "median_native": median,
                        "std_native": float(np.std(values[mask], ddof=0)),
                        "pooled_z_mean": (mean - pooled_mean) / pooled_sd,
                    }
                )
        for run_label, values in [("three_run_mean", ko), *[(f"run{i+1:02d}", data.ko_runs[i]) for i in range(3)]]:
            point = fixed_rank_spearman(data.speciteller, values)
            xr = percentile_ranks(data.speciteller)
            yr = percentile_ranks(values)
            reps = _cluster_rank_correlation_replicates(
                xr, yr, data.doc_codes, len(data.doc_names), replicates,
                _seed(seed, corpus_id, run_label, "agreement"),
            )
            low, high = _ci(reps)
            agreements.append(
                {
                    "corpus_id": corpus_id,
                    "ko_score_identity": run_label,
                    "row_count": values.size,
                    "spearman": point,
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "bootstrap_method": "document-cluster resampling of fixed full-sample average ranks",
                }
            )
    contrasts: list[dict[str, Any]] = []
    summary_index = {
        (row["model_id"], row["variant"], row["corpus_id"]): row for row in summaries
    }
    for model_id in MODEL_IDS:
        for variant in VARIANTS:
            ordering = sorted(
                CORPUS_ORDER,
                key=lambda corpus: summary_index[(model_id, variant, corpus)]["mean_native"],
                reverse=True,
            )
            for order, corpus_id in enumerate(ordering, start=1):
                contrasts.append(
                    {
                        "model_id": model_id,
                        "variant": variant,
                        "contrast": "corpus_order",
                        "technical_corpus_id": corpus_id,
                        "native_gap": "",
                        "native_ci_low": "",
                        "native_ci_high": "",
                        "pooled_z_gap": "",
                        "order": order,
                    }
                )
            wiki = summary_index[(model_id, variant, WIKIPEDIA_CORPUS_ID)]
            for technical in CORPUS_ORDER[1:]:
                tech = summary_index[(model_id, variant, technical)]
                gap_reps = (
                    boots[model_id][WIKIPEDIA_CORPUS_ID][variant]
                    - boots[model_id][technical][variant]
                )
                low, high = _ci(gap_reps)
                contrasts.append(
                    {
                        "model_id": model_id,
                        "variant": variant,
                        "contrast": "wikipedia_minus_technical",
                        "technical_corpus_id": technical,
                        "native_gap": wiki["mean_native"] - tech["mean_native"],
                        "native_ci_low": low,
                        "native_ci_high": high,
                        "pooled_z_gap": wiki["pooled_z_mean"] - tech["pooled_z_mean"],
                        "order": "",
                    }
                )
    diagnostics = {
        "disagreement_summary": disagreement_summary,
        "disagreement_rows": disagreement_rows,
    }
    return summaries, contrasts, agreements, {"variability": variability, **diagnostics}


def length_control_models(
    data_by_corpus: dict[str, ComparisonCorpus], length_config: dict[str, Any], replicates: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    outputs: dict[str, list[dict[str, Any]]] = {
        "length_standardized_means.csv": [],
        "length_gap_estimates.csv": [],
        "length_regression_contrasts.csv": [],
        "length_regression_diagnostics.csv": [],
        "length_specific_contrasts.csv": [],
        "length_common_support_diagnostics.csv": [],
    }
    support_cfg = length_config["common_support"]
    for model_id in MODEL_IDS:
        length_data = {
            corpus_id: LengthCorpusData(
                corpus_id=corpus_id,
                display_name=data.display_name,
                scores=_model_values(data, model_id),
                token_count=data.token_count,
                char_count=data.char_count,
                keep=data.keep,
                doc_codes=data.doc_codes,
                doc_names=data.doc_names,
            )
            for corpus_id, data in data_by_corpus.items()
        }
        supports = {}
        means = {}
        substantial = {}
        for variant in VARIANTS:
            support = build_common_support(
                length_data,
                variant=variant,
                minimum_rows=support_cfg["minimum_rows_per_corpus_per_stratum"],
                minimum_documents=support_cfg["minimum_documents_per_corpus_per_stratum"],
                maximum_weight_ratio=length_config["weights"]["maximum_allowed_weight_ratio"],
            )
            supports[variant] = support
            means[variant] = standardized_means(length_data, support)
            adjusted, diagnostics, length_rows, flags = regression_analysis(length_data, support)
            substantial[variant] = flags
            for row in means[variant]:
                outputs["length_standardized_means.csv"].append({"model_id": model_id, **row})
            for row in adjusted:
                outputs["length_regression_contrasts.csv"].append({"model_id": model_id, **row})
            for row in diagnostics:
                outputs["length_regression_diagnostics.csv"].append({"model_id": model_id, **row})
            for row in length_rows:
                outputs["length_specific_contrasts.csv"].append({"model_id": model_id, **row})
            for row in support.diagnostics:
                outputs["length_common_support_diagnostics.csv"].append({"model_id": model_id, **row})
        bootstrap, _ = bootstrap_all(
            length_data, supports, replicates=replicates, seed=_seed(seed, model_id, "length")
        )
        for row in build_gap_rows(means, bootstrap, substantial):
            outputs["length_gap_estimates.csv"].append({"model_id": model_id, **row})
    return outputs


def _format(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return "" if math.isnan(float(value)) else format(float(value), ".10g")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _format(value) for key, value in row.items()} for row in rows)


def run_model_comparison(settings: ComparisonSettings) -> dict[str, Any]:
    protocol = load_protocol(settings.protocol_config)
    if settings.bootstrap_replicates != protocol.record["analysis"]["bootstrap_replicates"]:
        raise ValueError("Bootstrap replicate count differs from frozen comparator protocol")
    if settings.bootstrap_seed != protocol.record["analysis"]["bootstrap_master_seed"]:
        raise ValueError("Bootstrap seed differs from frozen comparator protocol")
    length_config = json.loads(settings.length_config.read_text(encoding="utf-8"))
    data, coverage, provenance = load_comparison_data(settings=settings, protocol=protocol)
    summaries, contrasts, agreements, diagnostics = summarize_models(
        data, settings.bootstrap_replicates, settings.bootstrap_seed
    )
    length_outputs = length_control_models(
        data, length_config, settings.bootstrap_replicates, settings.bootstrap_seed
    )
    tables: dict[str, list[dict[str, Any]]] = {
        "coverage.csv": coverage,
        "model_corpus_summaries.csv": summaries,
        "within_model_contrasts.csv": contrasts,
        "cross_model_agreement.csv": agreements,
        "training_variability.csv": diagnostics["variability"],
        "disagreement_summary.csv": diagnostics["disagreement_summary"],
        "disagreement_rows.csv": diagnostics["disagreement_rows"],
        **length_outputs,
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        _write_csv(settings.output_dir / name, rows)
    if settings.paper_facing_dir is not None:
        settings.paper_facing_dir.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            _write_csv(settings.paper_facing_dir / name, rows)
    metadata = {
        "schema_version": "ko_official_release_comparison_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": settings.command,
        "protocol": {
            "path": str(settings.protocol_config).replace("\\", "/"),
            "sha256": protocol.sha256,
            "outcome_blind_freeze": True,
        },
        "input_manifest": {
            "path": str(settings.input_manifest).replace("\\", "/"),
            "sha256": sha256_file(settings.input_manifest),
        },
        "preprocessing_manifest": {
            "path": str(settings.preprocessing_manifest).replace("\\", "/"),
            "sha256": sha256_file(settings.preprocessing_manifest),
        },
        "length_method": {
            "path": str(settings.length_config).replace("\\", "/"),
            "sha256": sha256_file(settings.length_config),
            "method_version": length_config["method_version"],
        },
        "coverage_gate_passed": all(row["coverage_rate"] == 1.0 for row in coverage),
        "ordered_join_gate_passed": all(row["ordered_join_passed"] for row in coverage),
        "aggregation": protocol.record["protocol"]["primary_ko_aggregation"],
        "bootstrap": {
            "unit": "doc_path",
            "replicates": settings.bootstrap_replicates,
            "master_seed": settings.bootstrap_seed,
        },
        "model_scales": {
            SPECITELLER_ID: {"minimum": 0.0, "maximum": 1.0, "direction": "higher_is_more_specific"},
            KO_ID: protocol.record["protocol"]["score_scale"],
        },
        "claim_boundary": protocol.record["analysis"]["accuracy_boundary"],
        "provenance": _portable_provenance(provenance),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
        "outputs": {
            name: {"rows": len(rows), "sha256": sha256_file(settings.output_dir / name)}
            for name, rows in tables.items()
        },
    }
    metadata_path = settings.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if settings.paper_facing_dir is not None:
        (settings.paper_facing_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return metadata


def _portable_provenance(provenance: dict[str, Any]) -> dict[str, Any]:
    """Retain scientific identities while omitting host-specific run commands."""
    runs = provenance.get("runs", {})
    return {
        "artifacts": provenance.get("artifacts", {}),
        "runs": {
            run_key: {key: value for key, value in record.items() if key != "command"}
            for run_key, record in runs.items()
        },
    }
