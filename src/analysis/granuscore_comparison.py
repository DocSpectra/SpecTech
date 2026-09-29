"""Frozen GranuScore comparison across corpora, pilot labels, and edit pairs."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from src.analysis.model_comparison import (
    ComparisonCorpus,
    ComparisonSettings,
    _ci,
    _cluster_mean_replicates,
    _cluster_rank_correlation_replicates,
    _seed,
    fixed_rank_spearman,
    load_comparison_data,
    percentile_ranks,
    _portable_provenance,
)
from src.analysis.pilot_model_human import (
    ALL_MODELS,
    HUMAN_TARGETS,
    PRIMARY_MODELS,
    _interval,
    _load_corpora,
    _stream_seed,
    bootstrap_correlation_matrices,
)
from src.analysis.preprocessing_ablation import CORPUS_ORDER, WIKIPEDIA_CORPUS_ID, sha256_file
from src.ko_specificity.official_release import load_protocol


GRANUSCORE_NATIVE = "granuscore_native_higher_is_coarser"
GRANUSCORE_ALIGNED = "granuscore_direction_aligned_secondary"
QWEN_MODEL = "qwen3_14b_zero_shot_rubric"
FULL_COMPARATORS = (
    "speciteller_frozen_round1",
    "ko_official_release_three_run_mean",
    "ko_official_release_run01",
    "ko_official_release_run02",
    "ko_official_release_run03",
)
FULL_VARIANTS = ("original_all", "strict_all", "original_units", "strict_units")


@dataclass(frozen=True)
class GranuCorpus:
    base: ComparisonCorpus
    scores: np.ndarray
    unit_count: np.ndarray
    no_unit: np.ndarray


@dataclass(frozen=True)
class GranuSettings:
    config_path: Path
    model_settings: ComparisonSettings
    score_root: Path
    controlled_manifest: Path
    controlled_preparation_metadata: Path
    controlled_scores: Path
    controlled_metadata: Path
    pilot_config: Path
    qwen_scores: Path
    qwen_metadata: Path
    length_config: Path
    output_dir: Path
    compact_dir: Path | None
    command: str


def load_granuscore_config(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_granuscore_v1":
        raise ValueError("unexpected GranuScore config schema")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("GranuScore config is not outcome-blind frozen")
    if record["identity"]["score_direction"] != "higher_is_coarser_more_abstract":
        raise ValueError("GranuScore direction changed")
    return record


def _read_score_file(path: Path, expected_ids: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores: list[float] = []
    units: list[int] = []
    no_units: list[bool] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_columns = [
            "corpus_id", "sent_id", "granuscore_percentile",
            "referential_unit_count", "no_referential_unit",
        ]
        if reader.fieldnames != expected_columns:
            raise ValueError(f"GranuScore schema mismatch: {path}")
        for expected_id, row in zip(expected_ids, reader, strict=True):
            if row["sent_id"] != expected_id:
                raise ValueError(f"GranuScore ordered join failure: {path}")
            score = float(row["granuscore_percentile"])
            unit_count = int(row["referential_unit_count"])
            no_unit = row["no_referential_unit"] == "true"
            if not math.isfinite(score) or not 0 <= score <= 100:
                raise ValueError("invalid GranuScore percentile")
            if unit_count < 0 or no_unit != (unit_count == 0):
                raise ValueError("invalid GranuScore referential-unit audit")
            scores.append(score)
            units.append(unit_count)
            no_units.append(no_unit)
    if len(scores) != len(expected_ids):
        raise ValueError(f"incomplete GranuScore coverage: {path}")
    return (
        np.asarray(scores, dtype=np.float64),
        np.asarray(units, dtype=np.int32),
        np.asarray(no_units, dtype=np.bool_),
    )


def load_full_data(
    config: dict[str, Any], settings: GranuSettings
) -> tuple[dict[str, GranuCorpus], list[dict[str, Any]], dict[str, Any]]:
    protocol = load_protocol(settings.model_settings.protocol_config)
    base_data, _, base_provenance = load_comparison_data(
        settings=settings.model_settings, protocol=protocol
    )
    result: dict[str, GranuCorpus] = {}
    coverage: list[dict[str, Any]] = []
    metadata_records: dict[str, Any] = {}
    expected_runner_sha256 = sha256_file(Path("granuscore_container/score_csv.py"))
    for corpus_id in config["full_corpora"]["canonical_order"]:
        base = base_data[corpus_id]
        score_path = settings.score_root / f"{corpus_id}.csv"
        metadata_path = settings.score_root / f"{corpus_id}.metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["output_sha256"] != sha256_file(score_path):
            raise ValueError(f"GranuScore output hash mismatch: {corpus_id}")
        expected_input = config["full_corpora"]["inputs"][corpus_id]
        if (
            metadata["input_sha256"] != expected_input["sha256"]
            or metadata["row_count"] != expected_input["rows"]
            or metadata["image"] != config["runtime"]["release_image"]
            or metadata["runner_sha256"] != expected_runner_sha256
            or metadata["cuda_available"] is not True
            or metadata["score_direction"] != "higher_is_coarser_more_abstract"
        ):
            raise ValueError(f"GranuScore metadata identity failure: {corpus_id}")
        scores, units, no_unit = _read_score_file(score_path, base.sent_ids)
        result[corpus_id] = GranuCorpus(base, scores, units, no_unit)
        coverage.append(
            {
                "corpus_id": corpus_id,
                "canonical_rows": len(base.sent_ids),
                "granuscore_rows": scores.size,
                "no_referential_unit_rows": int(np.sum(no_unit)),
                "no_referential_unit_rate": float(np.mean(no_unit)),
                "coverage_rate": 1.0,
                "ordered_join_passed": True,
            }
        )
        metadata_records[corpus_id] = {
            "path": metadata_path.as_posix(),
            "sha256": sha256_file(metadata_path),
            "output_sha256": metadata["output_sha256"],
        }
    return result, coverage, {"base": base_provenance, "scores": metadata_records}


def _mask(data: GranuCorpus, variant: str) -> np.ndarray:
    masks = {
        "original_all": np.ones(data.scores.size, dtype=np.bool_),
        "strict_all": data.base.keep,
        "original_units": ~data.no_unit,
        "strict_units": data.base.keep & ~data.no_unit,
    }
    return masks[variant]


def _remap_docs(codes: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    active = np.unique(codes[mask])
    remap = np.full(int(np.max(codes)) + 1, -1, dtype=np.int32)
    remap[active] = np.arange(active.size)
    return remap[codes[mask]], int(active.size)


def _comparators(data: GranuCorpus) -> dict[str, np.ndarray]:
    return {
        FULL_COMPARATORS[0]: data.base.speciteller,
        FULL_COMPARATORS[1]: data.base.ko_primary,
        FULL_COMPARATORS[2]: data.base.ko_runs[0],
        FULL_COMPARATORS[3]: data.base.ko_runs[1],
        FULL_COMPARATORS[4]: data.base.ko_runs[2],
    }


def summarize_full(
    data_by_corpus: dict[str, GranuCorpus], *, replicates: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    agreements: list[dict[str, Any]] = []
    length_agreement: list[dict[str, Any]] = []
    mean_boot: dict[tuple[str, str], np.ndarray] = {}
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        for variant in FULL_VARIANTS:
            mask = _mask(data, variant)
            values = data.scores[mask]
            doc_codes, document_count = _remap_docs(data.base.doc_codes, mask)
            reps = _cluster_mean_replicates(
                values, doc_codes, document_count, replicates,
                _seed(seed, corpus_id, variant, "granuscore_mean"),
            )
            mean_boot[(corpus_id, variant)] = reps
            low, high = _ci(reps)
            summaries.append(
                {
                    "corpus_id": corpus_id,
                    "variant": variant,
                    "row_count": values.size,
                    "document_count": document_count,
                    "mean": float(np.mean(values)),
                    "mean_ci_low": low,
                    "mean_ci_high": high,
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=0)),
                    "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                }
            )
            for comparator_id, comparator in _comparators(data).items():
                point = fixed_rank_spearman(values, comparator[mask])
                reps_corr = _cluster_rank_correlation_replicates(
                    percentile_ranks(values), percentile_ranks(comparator[mask]),
                    doc_codes, document_count, replicates,
                    _seed(seed, corpus_id, variant, comparator_id, "agreement"),
                )
                corr_low, corr_high = _ci(reps_corr)
                agreements.append(
                    {
                        "corpus_id": corpus_id,
                        "variant": variant,
                        "comparator_id": comparator_id,
                        "row_count": values.size,
                        "native_spearman": point,
                        "native_ci_low": corr_low,
                        "native_ci_high": corr_high,
                        "direction_aligned_spearman": -point,
                        "direction_note": "native GranuScore is higher when coarser",
                    }
                )
            length_point = fixed_rank_spearman(values, data.base.token_count[mask].astype(np.float64))
            length_reps = _cluster_rank_correlation_replicates(
                percentile_ranks(values),
                percentile_ranks(data.base.token_count[mask].astype(np.float64)),
                doc_codes, document_count, replicates,
                _seed(seed, corpus_id, variant, "token_count"),
            )
            length_low, length_high = _ci(length_reps)
            length_agreement.append(
                {
                    "corpus_id": corpus_id,
                    "variant": variant,
                    "row_count": values.size,
                    "token_count_spearman": length_point,
                    "ci_low": length_low,
                    "ci_high": length_high,
                }
            )
    gaps: list[dict[str, Any]] = []
    index = {(row["corpus_id"], row["variant"]): row for row in summaries}
    for variant in FULL_VARIANTS:
        wiki = index[(WIKIPEDIA_CORPUS_ID, variant)]
        for technical in CORPUS_ORDER[1:]:
            tech = index[(technical, variant)]
            reps = mean_boot[(WIKIPEDIA_CORPUS_ID, variant)] - mean_boot[(technical, variant)]
            low, high = _ci(reps)
            gaps.append(
                {
                    "variant": variant,
                    "technical_corpus_id": technical,
                    "wikipedia_minus_technical": wiki["mean"] - tech["mean"],
                    "ci_low": low,
                    "ci_high": high,
                    "direction_note": "positive means Wikipedia is coarser on native GranuScore",
                }
            )
    return {
        "full_corpus_summaries.csv": summaries,
        "full_corpus_gaps.csv": gaps,
        "full_model_agreement.csv": agreements,
        "granuscore_token_length_agreement.csv": length_agreement,
    }


def length_control(
    data_by_corpus: dict[str, GranuCorpus], length_config: dict[str, Any], *, replicates: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    length_data = {
        corpus_id: LengthCorpusData(
            corpus_id=corpus_id,
            display_name=data.base.display_name,
            scores=data.scores,
            token_count=data.base.token_count,
            char_count=data.base.char_count,
            keep=data.base.keep,
            doc_codes=data.base.doc_codes,
            doc_names=data.base.doc_names,
        )
        for corpus_id, data in data_by_corpus.items()
    }
    supports: dict[str, Any] = {}
    means: dict[str, Any] = {}
    substantial: dict[str, Any] = {}
    outputs: dict[str, list[dict[str, Any]]] = {
        "length_standardized_means.csv": [],
        "length_regression_contrasts.csv": [],
        "length_regression_diagnostics.csv": [],
        "length_specific_contrasts.csv": [],
        "length_common_support_diagnostics.csv": [],
    }
    common = length_config["common_support"]
    for variant in VARIANTS:
        support = build_common_support(
            length_data,
            variant=variant,
            minimum_rows=common["minimum_rows_per_corpus_per_stratum"],
            minimum_documents=common["minimum_documents_per_corpus_per_stratum"],
            maximum_weight_ratio=length_config["weights"]["maximum_allowed_weight_ratio"],
        )
        supports[variant] = support
        means[variant] = standardized_means(length_data, support)
        adjusted, diagnostics, specific, flags = regression_analysis(length_data, support)
        substantial[variant] = flags
        outputs["length_standardized_means.csv"].extend(means[variant])
        outputs["length_regression_contrasts.csv"].extend(adjusted)
        outputs["length_regression_diagnostics.csv"].extend(diagnostics)
        outputs["length_specific_contrasts.csv"].extend(specific)
        outputs["length_common_support_diagnostics.csv"].extend(support.diagnostics)
    bootstrap, _ = bootstrap_all(length_data, supports, replicates=replicates, seed=seed)
    outputs["length_gap_estimates.csv"] = build_gap_rows(means, bootstrap, substantial)
    return outputs


def _load_qwen(path: Path, metadata_path: Path) -> dict[tuple[str, str], float]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if sha256_file(path) != metadata["outputs"]["qwen_scores.csv"]["sha256"]:
        raise ValueError("Qwen compact score hash mismatch")
    result: dict[tuple[str, str], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["corpus_id"], row["sent_id"])
            if key in result:
                raise ValueError("duplicate Qwen pilot row")
            result[key] = float(row["qwen_score"])
    return result


def build_pilot_paper_table(agreement_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pivot the complete pilot agreement evidence into a manuscript-ready table."""
    agreement_index = {
        (row["corpus_id"], row["target_id"], row["model_id"]): row
        for row in agreement_rows
    }
    paper_models = (
        ("speciteller", "speciteller_frozen_round1"),
        ("ko_run01", "ko_official_release_run01"),
        ("ko_run02", "ko_official_release_run02"),
        ("ko_run03", "ko_official_release_run03"),
        ("ko_mean_secondary", "ko_official_release_run_mean"),
        ("qwen", QWEN_MODEL),
        ("granuscore_native", GRANUSCORE_NATIVE),
        ("granuscore_direction_aligned_secondary", GRANUSCORE_ALIGNED),
    )
    paper_rows: list[dict[str, Any]] = []
    for corpus_id in ("ansible_docs", "github_docs"):
        for target_id in HUMAN_TARGETS:
            row: dict[str, Any] = {"corpus_id": corpus_id, "target_id": target_id, "n": 40}
            for prefix, model_id in paper_models:
                source = agreement_index[(corpus_id, target_id, model_id)]
                row[f"{prefix}_rho"] = source["spearman_rho"]
                row[f"{prefix}_ci_low"] = source["ci_low"]
                row[f"{prefix}_ci_high"] = source["ci_high"]
            paper_rows.append(row)
    return paper_rows


def analyze_pilot(
    data_by_corpus: dict[str, GranuCorpus], settings: GranuSettings, *, replicates: int, seed: int
) -> dict[str, list[dict[str, Any]]]:
    pilot_record = json.loads(settings.pilot_config.read_text(encoding="utf-8"))
    pilot, _ = _load_corpora(pilot_record, settings.pilot_config)
    qwen = _load_qwen(settings.qwen_scores, settings.qwen_metadata)
    agreement_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    for corpus_id, pilot_corpus in pilot.items():
        full = data_by_corpus[corpus_id]
        index = {sent_id: position for position, sent_id in enumerate(full.base.sent_ids)}
        gran = np.asarray([full.scores[index[sent_id]] for sent_id in pilot_corpus.sent_ids])
        qwen_values = np.asarray([qwen[(corpus_id, sent_id)] for sent_id in pilot_corpus.sent_ids])
        score_map = {
            GRANUSCORE_NATIVE: gran,
            GRANUSCORE_ALIGNED: -gran,
            **{model: pilot_corpus.scores[model] for model in ALL_MODELS},
            QWEN_MODEL: qwen_values,
        }
        model_ids = tuple(score_map)
        columns = np.vstack([score_map[name] for name in model_ids] + [pilot_corpus.labels[t] for t in HUMAN_TARGETS])
        point, boot = bootstrap_correlation_matrices(
            columns, replicates=replicates, seed=_stream_seed(seed, corpus_id)
        )
        column_index = {name: i for i, name in enumerate(model_ids + HUMAN_TARGETS)}
        for model_id in model_ids:
            for target in HUMAN_TARGETS:
                mi, ti = column_index[model_id], column_index[target]
                low, high, valid, degenerate = _interval(boot[:, mi, ti], replicates, 0.95)
                agreement_rows.append(
                    {
                        "corpus_id": corpus_id,
                        "model_id": model_id,
                        "target_id": target,
                        "n": len(pilot_corpus.sent_ids),
                        "spearman_rho": float(point[mi, ti]),
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_valid": valid,
                        "bootstrap_degenerate": degenerate,
                        "analysis_role": "primary_native" if model_id == GRANUSCORE_NATIVE else ("secondary_direction_aid" if model_id == GRANUSCORE_ALIGNED else "existing_comparator"),
                    }
                )
        aligned_index = column_index[GRANUSCORE_ALIGNED]
        for target in HUMAN_TARGETS:
            ti = column_index[target]
            for comparator in (*PRIMARY_MODELS, QWEN_MODEL):
                ci = column_index[comparator]
                delta = boot[:, aligned_index, ti] - boot[:, ci, ti]
                low, high, valid, degenerate = _interval(delta, replicates, 0.95)
                contrast_rows.append(
                    {
                        "corpus_id": corpus_id,
                        "target_id": target,
                        "comparison": f"{GRANUSCORE_ALIGNED}_minus_{comparator}",
                        "delta_rho": float(point[aligned_index, ti] - point[ci, ti]),
                        "ci_low": low,
                        "ci_high": high,
                        "bootstrap_valid": valid,
                        "bootstrap_degenerate": degenerate,
                    }
                )
        native_index = column_index[GRANUSCORE_NATIVE]
        for comparator in (*PRIMARY_MODELS, QWEN_MODEL):
            ci = column_index[comparator]
            low, high, valid, degenerate = _interval(boot[:, native_index, ci], replicates, 0.95)
            model_rows.append(
                {
                    "corpus_id": corpus_id,
                    "comparator_id": comparator,
                    "n": len(pilot_corpus.sent_ids),
                    "native_spearman": float(point[native_index, ci]),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_valid": valid,
                    "bootstrap_degenerate": degenerate,
                }
            )
    return {
        "pilot_model_human_agreement.csv": agreement_rows,
        "pilot_direction_aligned_contrasts.csv": contrast_rows,
        "pilot_granuscore_model_agreement.csv": model_rows,
        "pilot_paper_table.csv": build_pilot_paper_table(agreement_rows),
    }


def load_edit_pairs(
    config_path: Path,
    manifest_path: Path,
    preparation_metadata_path: Path,
    score_path: Path,
    metadata_path: Path,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    preparation = json.loads(preparation_metadata_path.read_text(encoding="utf-8"))
    expected_runner_sha256 = sha256_file(Path("granuscore_container/score_csv.py"))
    if (
        preparation["schema_version"] != "granuscore_controlled_edit_preparation_v1"
        or preparation["config_sha256"] != sha256_file(config_path)
        or preparation["manifest_sha256"] != sha256_file(manifest_path)
        or preparation["case_count"] != 60
        or preparation["score_row_count"] != 120
    ):
        raise ValueError("controlled-edit preparation metadata failure")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if (
        metadata["output_sha256"] != sha256_file(score_path)
        or metadata["input_sha256"] != preparation["sentence_sha256"]
        or metadata["row_count"] != 120
        or metadata["image"] != config["runtime"]["release_image"]
        or metadata["runner_sha256"] != expected_runner_sha256
        or metadata["cuda_available"] is not True
    ):
        raise ValueError("controlled-edit GranuScore metadata failure")
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with score_path.open("r", encoding="utf-8", newline="") as handle:
        scores = list(csv.DictReader(handle))
    if len(manifest) != 120 or len(scores) != 120:
        raise ValueError("controlled-edit scoring must contain exactly 120 rows")
    paired: dict[str, dict[str, Any]] = {}
    for mrow, srow in zip(manifest, scores, strict=True):
        if mrow["score_sent_id"] != srow["sent_id"]:
            raise ValueError("controlled-edit ordered join failure")
        case = paired.setdefault(
            mrow["case_id"],
            {
                "case_id": mrow["case_id"],
                "corpus_id": mrow["source_corpus_id"],
                "source_sent_id": mrow["source_sent_id"],
                "edit_type": mrow["edit_type"],
            },
        )
        version = mrow["version"]
        case[f"{version}_score"] = float(srow["granuscore_percentile"])
        case[f"{version}_no_unit"] = srow["no_referential_unit"] == "true"
    rows: list[dict[str, Any]] = []
    for case in paired.values():
        if "original_score" not in case or "edited_score" not in case:
            raise ValueError("incomplete controlled-edit pair")
        case["delta_edited_minus_original"] = case["edited_score"] - case["original_score"]
        case["pair_has_no_unit"] = case["original_no_unit"] or case["edited_no_unit"]
        rows.append(case)
    if len(rows) != 60:
        raise ValueError("controlled-edit pair count mismatch")
    return rows


def summarize_edit_pairs(
    pairs: list[dict[str, Any]], *, replicates: int, seed: int
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for corpus_id in ("ansible_docs", "github_docs"):
        for edit_type in ("add_specific", "de_specify", "irrelevant_rewrite"):
            for variant in ("all", "units_only"):
                selected = [
                    row for row in pairs
                    if row["corpus_id"] == corpus_id
                    and row["edit_type"] == edit_type
                    and (variant == "all" or not row["pair_has_no_unit"])
                ]
                if not selected:
                    continue
                values = np.asarray([row["delta_edited_minus_original"] for row in selected])
                rng = np.random.default_rng(_seed(seed, corpus_id, edit_type, variant))
                draw = rng.integers(0, values.size, size=(replicates, values.size))
                reps = np.mean(values[draw], axis=1)
                low, high = _ci(reps)
                expected = {"add_specific": -1, "de_specify": 1, "irrelevant_rewrite": 0}[edit_type]
                if expected < 0:
                    directional = float(np.mean(values < 0))
                elif expected > 0:
                    directional = float(np.mean(values > 0))
                else:
                    directional = float(np.mean(np.abs(values) <= 5.0))
                output.append(
                    {
                        "corpus_id": corpus_id,
                        "edit_type": edit_type,
                        "variant": variant,
                        "pair_count": values.size,
                        "mean_delta": float(np.mean(values)),
                        "ci_low": low,
                        "ci_high": high,
                        "median_delta": float(np.median(values)),
                        "directional_or_near_zero_share": directional,
                        "direction_note": "negative is finer/more specific; positive is coarser/more abstract",
                    }
                )
    return output


def _format(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return format(float(value), ".10g")
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows({key: _format(value) for key, value in row.items()} for row in rows)


def run_granuscore_analysis(settings: GranuSettings) -> dict[str, Any]:
    config = load_granuscore_config(settings.config_path)
    full_reps = config["full_analysis"]["bootstrap_replicates"]
    full_seed = config["full_analysis"]["bootstrap_master_seed"]
    data, coverage, provenance = load_full_data(config, settings)
    tables = summarize_full(data, replicates=full_reps, seed=full_seed)
    length_config = json.loads(settings.length_config.read_text(encoding="utf-8"))
    tables.update(length_control(data, length_config, replicates=full_reps, seed=full_seed))
    tables.update(
        analyze_pilot(
            data,
            settings,
            replicates=config["pilot"]["bootstrap_replicates"],
            seed=config["pilot"]["bootstrap_master_seed"],
        )
    )
    pairs = load_edit_pairs(
        settings.config_path,
        settings.controlled_manifest,
        settings.controlled_preparation_metadata,
        settings.controlled_scores,
        settings.controlled_metadata,
        config,
    )
    tables["controlled_edit_pair_scores.csv"] = pairs
    tables["controlled_edit_summaries.csv"] = summarize_edit_pairs(
        pairs,
        replicates=config["controlled_edits"]["bootstrap_replicates"],
        seed=config["controlled_edits"]["bootstrap_master_seed"],
    )
    tables["coverage.csv"] = coverage
    for directory in (settings.output_dir, settings.compact_dir):
        if directory is None:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in tables.items():
            _write_csv(directory / name, rows)
    metadata = {
        "schema_version": "round2_granuscore_results_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": settings.command,
        "protocol": {
            "path": settings.config_path.as_posix(),
            "sha256": sha256_file(settings.config_path),
            "outcome_blind_freeze": True,
        },
        "coverage_gate_passed": all(row["coverage_rate"] == 1.0 for row in coverage),
        "ordered_join_gate_passed": all(row["ordered_join_passed"] for row in coverage),
        "score_direction": "higher_is_coarser_more_abstract",
        "construct_boundary": config["identity"]["construct_boundary"],
        "bootstrap": {
            "full": {"replicates": full_reps, "seed": full_seed, "unit": "doc_path"},
            "pilot": {"replicates": config["pilot"]["bootstrap_replicates"], "seed": config["pilot"]["bootstrap_master_seed"], "unit": "sentence"},
            "controlled_edits": {"replicates": config["controlled_edits"]["bootstrap_replicates"], "seed": config["controlled_edits"]["bootstrap_master_seed"], "unit": "pair"},
        },
        "claim_boundary": config["full_analysis"]["accuracy_boundary"],
        "provenance": {
            "base": _portable_provenance(provenance["base"]),
            "scores": provenance["scores"],
            "controlled_edits": {
                "preparation_metadata_sha256": sha256_file(settings.controlled_preparation_metadata),
                "score_metadata_sha256": sha256_file(settings.controlled_metadata),
                "score_sha256": sha256_file(settings.controlled_scores),
            },
            "mechanical_freeze_record": {
                "path": config["runtime"]["mechanical_freeze_record"],
                "sha256": sha256_file(Path(config["runtime"]["mechanical_freeze_record"])),
            },
        },
        "environment": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()},
        "outputs": {
            name: {"rows": len(rows), "sha256": sha256_file(settings.output_dir / name)}
            for name, rows in tables.items()
        },
    }
    for directory in (settings.output_dir, settings.compact_dir):
        if directory is not None:
            (directory / "run_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    return metadata
