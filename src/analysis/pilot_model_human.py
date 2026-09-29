"""Frozen existing-pilot comparison for SpeciTeller and retained Ko runs.

Raw labels and row-level joins stay under the ignored outputs tree.  Only
anonymized aggregate evidence is eligible for the compact tracked pack.
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
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


PRIMARY_MODELS = (
    "speciteller_frozen_round1",
    "ko_official_release_run01",
    "ko_official_release_run02",
    "ko_official_release_run03",
)
SECONDARY_MODEL = "ko_official_release_run_mean"
ALL_MODELS = PRIMARY_MODELS + (SECONDARY_MODEL,)
HUMAN_TARGETS = ("ann_a", "ann_b", "ann_c", "pooled_human_mean")
EXPECTED_KO_MODEL_ID = "ko_author_official_release_se_ad_mean_std"
EXPECTED_KO_VERSION = "official-author-repository-postpublication-36f8e835-cpu-v1"
EXPECTED_UPSTREAM_COMMIT = "36f8e835e9dc6087d5b6763accf302db175947b1"


@dataclass(frozen=True)
class PilotCorpus:
    corpus_id: str
    sent_ids: tuple[str, ...]
    buckets: tuple[str, ...]
    labels: Mapping[str, np.ndarray]
    scores: Mapping[str, np.ndarray]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise ValueError(f"Required artifact is missing: {path.as_posix()}")
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.as_posix()}: expected {expected}, got {actual}"
        )


def _portable(path: Path) -> str:
    value = path.as_posix()
    if path.is_absolute() or ":/" in value:
        raise ValueError(f"Tracked metadata path is not portable: {value}")
    return value


def load_protocol(config_path: Path) -> dict[str, Any]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_pilot_model_human_comparison_v1":
        raise ValueError("Unexpected pilot comparison schema_version")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("Pilot comparison config is not outcome-blind frozen")
    if tuple(row["model_instance_id"] for row in record["score_instances"]["primary"]) != PRIMARY_MODELS:
        raise ValueError("Frozen four-primary-instance identity changed")
    if record["score_instances"]["secondary"]["model_instance_id"] != SECONDARY_MODEL:
        raise ValueError("Frozen secondary Ko aggregate identity changed")
    bootstrap = record["analysis"]["bootstrap"]
    if bootstrap["replicates"] != 10000 or bootstrap["master_seed"] != 20260809:
        raise ValueError("Frozen bootstrap design changed")
    return record


def _load_manifest(path: Path, corpus_order: Iterable[str]) -> dict[str, list[dict[str, str]]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[str, list[dict[str, str]]] = {corpus: [] for corpus in corpus_order}
    for row in rows:
        if row["corpus_id"] not in grouped:
            raise ValueError(f"Unexpected corpus in pilot manifest: {row['corpus_id']}")
        grouped[row["corpus_id"]].append(row)
    for corpus_id, corpus_rows in grouped.items():
        if len(corpus_rows) != 40:
            raise ValueError(f"Pilot manifest must contain exactly 40 {corpus_id} rows")
        positions = [int(row["pilot_position"]) for row in corpus_rows]
        sent_ids = [row["sent_id"] for row in corpus_rows]
        if positions != list(range(1, 41)) or len(set(sent_ids)) != 40:
            raise ValueError(f"Pilot manifest order/uniqueness failure for {corpus_id}")
    return grouped


def _load_label_file(
    path: Path,
    *,
    expected_hash: str,
    expected_corpus: str,
    expected_ids: list[str],
) -> tuple[np.ndarray, list[str], list[float]]:
    _require_hash(path, expected_hash)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row.get("sent_id", "")).strip() for row in rows]
    if ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError(f"Label row identity/order failure: {path.as_posix()}")
    corpora = {str(row.get("corpus_id", "")).strip() for row in rows}
    if corpora != {expected_corpus}:
        raise ValueError(f"Label corpus identity failure: {path.as_posix()}")
    labels: list[float] = []
    legacy_scores: list[float] = []
    texts: list[str] = []
    for row in rows:
        raw_label = str(row.get("human_label", "")).strip()
        try:
            value = float(raw_label)
        except ValueError as exc:
            raise ValueError(f"Missing/non-numeric label in {path.as_posix()}") from exc
        if not value.is_integer() or not 1 <= value <= 5:
            raise ValueError(f"Non-integer/out-of-range label in {path.as_posix()}")
        labels.append(value)
        legacy_scores.append(float(row["score"]))
        texts.append(str(row.get("sent_text", "")))
    return np.asarray(labels, dtype=np.float64), texts, legacy_scores


def _load_speciteller_scores(path: Path, expected_hash: str, sent_ids: list[str]) -> np.ndarray:
    _require_hash(path, expected_hash)
    wanted = set(sent_ids)
    found: dict[str, float] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sent_id, raw = line.split("\t", 1)
            if sent_id not in wanted:
                continue
            if sent_id in found:
                raise ValueError(f"Duplicate target sent_id in {path.as_posix()}: {sent_id}")
            found[sent_id] = float(raw)
    if set(found) != wanted:
        raise ValueError(f"Incomplete SpeciTeller pilot coverage in {path.as_posix()}")
    values = np.asarray([found[sent_id] for sent_id in sent_ids], dtype=np.float64)
    _validate_scores(values, "SpeciTeller")
    return values


def _load_ko_scores(
    path: Path,
    *,
    expected_hash: str,
    expected_run: str,
    expected_corpus: str,
    sent_ids: list[str],
) -> np.ndarray:
    _require_hash(path, expected_hash)
    wanted = set(sent_ids)
    found: dict[str, float] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            sent_id = row["sent_id"]
            if sent_id not in wanted:
                continue
            if sent_id in found:
                raise ValueError(f"Duplicate target sent_id in {path.as_posix()}: {sent_id}")
            if (
                row["corpus_id"] != expected_corpus
                or row["run_id"] != expected_run
                or row["model_id"] != EXPECTED_KO_MODEL_ID
                or row["model_version"] != EXPECTED_KO_VERSION
                or row["upstream_commit"] != EXPECTED_UPSTREAM_COMMIT
                or row["score_direction"] != "higher_is_more_specific"
            ):
                raise ValueError(f"Ko score identity failure in {path.as_posix()}")
            found[sent_id] = float(row["score_raw"])
    if set(found) != wanted:
        raise ValueError(f"Incomplete Ko pilot coverage in {path.as_posix()}")
    values = np.asarray([found[sent_id] for sent_id in sent_ids], dtype=np.float64)
    _validate_scores(values, f"Ko {expected_run}")
    return values


def _validate_scores(values: np.ndarray, label: str) -> None:
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError(f"Nonfinite {label} score")
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError(f"Out-of-range {label} score")


def _audit_sentence_rows(
    path: Path,
    expected_hash: str,
    sent_ids: list[str],
    expected_texts: list[str],
) -> None:
    _require_hash(path, expected_hash)
    wanted = set(sent_ids)
    found: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sent_id = row["sent_id"]
            if sent_id not in wanted:
                continue
            if sent_id in found:
                raise ValueError(f"Duplicate target sentence row: {sent_id}")
            found[sent_id] = row["sent_text"]
    if set(found) != wanted:
        raise ValueError(f"Incomplete canonical sentence pilot coverage in {path.as_posix()}")
    for sent_id, raw_text in zip(sent_ids, expected_texts):
        if found[sent_id] != raw_text:
            raise ValueError(f"Canonical/label sentence text mismatch for {sent_id}")


def _stream_seed(master_seed: int, corpus_id: str) -> int:
    payload = f"{master_seed}:{corpus_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little", signed=False)


def _rank_correlation_matrix(columns: np.ndarray) -> np.ndarray:
    ranks = np.vstack([_average_ranks(row) for row in columns])
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.corrcoef(ranks)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with deterministic stable tie handling."""
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Rank input must be one-dimensional and finite")
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def bootstrap_correlation_matrices(
    columns: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return full correlation matrix and jointly resampled matrices."""
    if columns.ndim != 2 or columns.shape[1] < 2:
        raise ValueError("columns must be shaped (variables, rows) with at least two rows")
    point = _rank_correlation_matrix(columns)
    if not np.all(np.isfinite(point)):
        raise ValueError("Full-sample Spearman is undefined")
    rng = np.random.default_rng(seed)
    boot = np.empty((replicates, columns.shape[0], columns.shape[0]), dtype=np.float64)
    n = columns.shape[1]
    for index in range(replicates):
        draw = rng.integers(0, n, size=n)
        boot[index] = _rank_correlation_matrix(columns[:, draw])
    return point, boot


def _interval(values: np.ndarray, replicates: int, minimum_valid_fraction: float) -> tuple[float, float, int, int]:
    valid = values[np.isfinite(values)]
    valid_count = int(valid.size)
    degenerate = int(replicates - valid_count)
    if valid_count < math.ceil(replicates * minimum_valid_fraction):
        raise ValueError(
            f"Bootstrap valid fraction below frozen minimum: {valid_count}/{replicates}"
        )
    low, high = np.quantile(valid, [0.025, 0.975])
    return float(low), float(high), valid_count, degenerate


def analyze_corpus(
    corpus: PilotCorpus,
    *,
    replicates: int,
    master_seed: int,
    minimum_valid_fraction: float,
) -> dict[str, list[dict[str, Any]]]:
    column_ids = ALL_MODELS + HUMAN_TARGETS
    columns = np.vstack(
        [corpus.scores[model] for model in ALL_MODELS]
        + [corpus.labels[target] for target in HUMAN_TARGETS]
    )
    point, boot = bootstrap_correlation_matrices(
        columns,
        replicates=replicates,
        seed=_stream_seed(master_seed, corpus.corpus_id),
    )
    indices = {name: index for index, name in enumerate(column_ids)}

    model_human: list[dict[str, Any]] = []
    for model in ALL_MODELS:
        for target in HUMAN_TARGETS:
            mi, ti = indices[model], indices[target]
            low, high, valid, degenerate = _interval(
                boot[:, mi, ti], replicates, minimum_valid_fraction
            )
            model_human.append(
                {
                    "corpus_id": corpus.corpus_id,
                    "model_instance_id": model,
                    "target_id": target,
                    "n": len(corpus.sent_ids),
                    "spearman_rho": float(point[mi, ti]),
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_valid": valid,
                    "bootstrap_degenerate": degenerate,
                    "analysis_role": "secondary_sensitivity" if model == SECONDARY_MODEL else "primary",
                }
            )

    contrasts: list[dict[str, Any]] = []
    for target in HUMAN_TARGETS:
        ti = indices[target]
        for model_a, model_b in combinations(PRIMARY_MODELS, 2):
            ai, bi = indices[model_a], indices[model_b]
            delta_boot = boot[:, ai, ti] - boot[:, bi, ti]
            low, high, valid, degenerate = _interval(
                delta_boot, replicates, minimum_valid_fraction
            )
            rho_a = float(point[ai, ti])
            rho_b = float(point[bi, ti])
            contrasts.append(
                {
                    "corpus_id": corpus.corpus_id,
                    "target_id": target,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": len(corpus.sent_ids),
                    "rho_a": rho_a,
                    "rho_b": rho_b,
                    "delta_rho_a_minus_b": rho_a - rho_b,
                    "ci_low": low,
                    "ci_high": high,
                    "bootstrap_valid": valid,
                    "bootstrap_degenerate": degenerate,
                }
            )

    model_model: list[dict[str, Any]] = []
    pairs = list(combinations(PRIMARY_MODELS, 2)) + [
        (model, SECONDARY_MODEL) for model in PRIMARY_MODELS
    ]
    for model_a, model_b in pairs:
        ai, bi = indices[model_a], indices[model_b]
        low, high, valid, degenerate = _interval(
            boot[:, ai, bi], replicates, minimum_valid_fraction
        )
        model_model.append(
            {
                "corpus_id": corpus.corpus_id,
                "model_a": model_a,
                "model_b": model_b,
                "n": len(corpus.sent_ids),
                "spearman_rho": float(point[ai, bi]),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_valid": valid,
                "bootstrap_degenerate": degenerate,
                "analysis_role": "secondary_sensitivity" if SECONDARY_MODEL in {model_a, model_b} else "primary",
            }
        )

    variability: list[dict[str, Any]] = []
    agreement_index = {
        (row["model_instance_id"], row["target_id"]): row for row in model_human
    }
    ko_models = PRIMARY_MODELS[1:]
    for target in HUMAN_TARGETS:
        rhos = np.asarray(
            [agreement_index[(model, target)]["spearman_rho"] for model in ko_models],
            dtype=np.float64,
        )
        variability.append(
            {
                "corpus_id": corpus.corpus_id,
                "target_id": target,
                "n": len(corpus.sent_ids),
                "run01_rho": float(rhos[0]),
                "run02_rho": float(rhos[1]),
                "run03_rho": float(rhos[2]),
                "minimum_rho": float(np.min(rhos)),
                "maximum_rho": float(np.max(rhos)),
                "range_rho": float(np.ptp(rhos)),
                "population_sd_rho": float(np.std(rhos, ddof=0)),
                "ko_mean_sensitivity_rho": float(
                    agreement_index[(SECONDARY_MODEL, target)]["spearman_rho"]
                ),
                "speciteller_rho": float(
                    agreement_index[(PRIMARY_MODELS[0], target)]["spearman_rho"]
                ),
            }
        )

    paper: list[dict[str, Any]] = []
    for target in HUMAN_TARGETS:
        row: dict[str, Any] = {
            "corpus_id": corpus.corpus_id,
            "target_id": target,
            "n": len(corpus.sent_ids),
        }
        for model in ALL_MODELS:
            agreement = agreement_index[(model, target)]
            prefix = {
                PRIMARY_MODELS[0]: "speciteller",
                PRIMARY_MODELS[1]: "ko_run01",
                PRIMARY_MODELS[2]: "ko_run02",
                PRIMARY_MODELS[3]: "ko_run03",
                SECONDARY_MODEL: "ko_mean_secondary",
            }[model]
            row[f"{prefix}_rho"] = agreement["spearman_rho"]
            row[f"{prefix}_ci_low"] = agreement["ci_low"]
            row[f"{prefix}_ci_high"] = agreement["ci_high"]
        variability_row = next(item for item in variability if item["target_id"] == target)
        row["ko_run_range_rho"] = variability_row["range_rho"]
        paper.append(row)

    return {
        "model_human_agreement.csv": model_human,
        "paired_model_human_contrasts.csv": contrasts,
        "model_model_agreement.csv": model_model,
        "ko_run_variability.csv": variability,
        "paper_table.csv": paper,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.10f}" if isinstance(value, float) else value
                    for key, value in row.items()
                }
            )


def _load_corpora(
    record: dict[str, Any], config_path: Path
) -> tuple[dict[str, PilotCorpus], list[dict[str, Any]]]:
    corpus_order = tuple(record["pilot"]["corpus_order"])
    manifest_path = Path(record["pilot"]["manifest"])
    manifest = _load_manifest(manifest_path, corpus_order)
    coverage: list[dict[str, Any]] = []

    label_entries = {
        (entry["corpus_id"], entry["annotator_id"]): entry
        for entry in record["label_inputs"]["files"]
    }
    ko_entries = {
        (entry["corpus_id"], entry["run_id"]): entry
        for entry in record["input_identities"]["ko_runs"]
    }
    corpora: dict[str, PilotCorpus] = {}

    for corpus_id in corpus_order:
        manifest_rows = manifest[corpus_id]
        sent_ids = [row["sent_id"] for row in manifest_rows]
        buckets = [row["bucket"] for row in manifest_rows]
        labels: dict[str, np.ndarray] = {}
        reference_texts: list[str] | None = None
        legacy_scores: list[float] | None = None
        for annotator_id in record["pilot"]["required_annotators"]:
            entry = label_entries[(corpus_id, annotator_id)]
            path = Path(entry["path"])
            values, texts, old_scores = _load_label_file(
                path,
                expected_hash=entry["sha256"],
                expected_corpus=corpus_id,
                expected_ids=sent_ids,
            )
            if reference_texts is None:
                reference_texts = texts
                legacy_scores = old_scores
            elif texts != reference_texts or not np.allclose(old_scores, legacy_scores, atol=0.0, rtol=0.0):
                raise ValueError(f"Annotator row content mismatch for {corpus_id}")
            labels[annotator_id] = values
            coverage.append(
                {
                    "corpus_id": corpus_id,
                    "artifact_type": "label_input",
                    "artifact_id": annotator_id,
                    "expected_rows": 40,
                    "observed_rows": 40,
                    "coverage_rate": 1.0,
                    "sha256": entry["sha256"],
                    "gate_passed": True,
                }
            )
        labels["pooled_human_mean"] = np.mean(
            np.vstack([labels[target] for target in HUMAN_TARGETS[:3]]), axis=0
        )

        sentence_entry = record["input_identities"]["sentences"][corpus_id]
        assert reference_texts is not None and legacy_scores is not None
        _audit_sentence_rows(
            Path(sentence_entry["path"]),
            sentence_entry["sha256"],
            sent_ids,
            reference_texts,
        )
        coverage.append(
            {
                "corpus_id": corpus_id,
                "artifact_type": "canonical_sentences",
                "artifact_id": "canonical",
                "expected_rows": 40,
                "observed_rows": 40,
                "coverage_rate": 1.0,
                "sha256": sentence_entry["sha256"],
                "gate_passed": True,
            }
        )

        spec_entry = record["input_identities"]["speciteller"][corpus_id]
        spec_values = _load_speciteller_scores(
            Path(spec_entry["path"]), spec_entry["sha256"], sent_ids
        )
        if not np.allclose(spec_values, np.asarray(legacy_scores), atol=5.1e-7, rtol=0.0):
            raise ValueError(f"Legacy pilot SpeciTeller reconciliation failed for {corpus_id}")
        scores: dict[str, np.ndarray] = {PRIMARY_MODELS[0]: spec_values}
        coverage.append(
            {
                "corpus_id": corpus_id,
                "artifact_type": "score_input",
                "artifact_id": PRIMARY_MODELS[0],
                "expected_rows": 40,
                "observed_rows": 40,
                "coverage_rate": 1.0,
                "sha256": spec_entry["sha256"],
                "gate_passed": True,
            }
        )

        for run_number, run_id in enumerate(("run01", "run02", "run03"), start=1):
            entry = ko_entries[(corpus_id, run_id)]
            score_path = Path(entry["score_path"])
            checkpoint_path = score_path.parent / "evidence" / "model.pickle"
            _require_hash(checkpoint_path, entry["checkpoint_sha256"])
            values = _load_ko_scores(
                score_path,
                expected_hash=entry["score_sha256"],
                expected_run=run_id,
                expected_corpus=corpus_id,
                sent_ids=sent_ids,
            )
            model_id = PRIMARY_MODELS[run_number]
            scores[model_id] = values
            coverage.extend(
                [
                    {
                        "corpus_id": corpus_id,
                        "artifact_type": "score_input",
                        "artifact_id": model_id,
                        "expected_rows": 40,
                        "observed_rows": 40,
                        "coverage_rate": 1.0,
                        "sha256": entry["score_sha256"],
                        "gate_passed": True,
                    },
                    {
                        "corpus_id": corpus_id,
                        "artifact_type": "checkpoint_identity",
                        "artifact_id": model_id,
                        "expected_rows": 1,
                        "observed_rows": 1,
                        "coverage_rate": 1.0,
                        "sha256": entry["checkpoint_sha256"],
                        "gate_passed": True,
                    },
                ]
            )
        scores[SECONDARY_MODEL] = np.mean(
            np.vstack([scores[model] for model in PRIMARY_MODELS[1:]]), axis=0
        )
        corpora[corpus_id] = PilotCorpus(
            corpus_id=corpus_id,
            sent_ids=tuple(sent_ids),
            buckets=tuple(buckets),
            labels=labels,
            scores=scores,
        )
    return corpora, coverage


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_readme(compact_dir: Path) -> None:
    rows = _read_csv_rows(compact_dir / "paper_table.csv")
    pooled = [row for row in rows if row["target_id"] == "pooled_human_mean"]
    lines = [
        "# Round 2 Existing-Pilot Model--Human Comparison",
        "",
        "All 80 existing pilot rows passed exact coverage for frozen SpeciTeller and Ko run01/run02/run03. "
        "The Ko rowwise mean is secondary, not a fifth independent model.",
        "",
        "## Pooled-label headline (Spearman rho; paired sentence-bootstrap 95% CI)",
        "",
        "| Corpus | SpeciTeller | Ko run01 | Ko run02 | Ko run03 | Ko run range | Ko mean (secondary) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pooled:
        display = "Ansible" if row["corpus_id"] == "ansible_docs" else "GitHub"
        def cell(prefix: str) -> str:
            return f"{float(row[prefix + '_rho']):.3f} [{float(row[prefix + '_ci_low']):.3f}, {float(row[prefix + '_ci_high']):.3f}]"
        lines.append(
            f"| {display} | {cell('speciteller')} | {cell('ko_run01')} | {cell('ko_run02')} | "
            f"{cell('ko_run03')} | {float(row['ko_run_range_rho']):.3f} | {cell('ko_mean_secondary')} |"
        )
    lines.extend(
        [
            "",
            "Full per-annotator estimates, paired contrasts, pairwise model agreement, run variability, "
            "coverage, and hashes are in the adjacent CSV/JSON artifacts.",
            "",
            "## Interpretation boundary",
            "",
            "These are descriptive correspondences to three non-expert ordinal pilot judgments at n=40 per corpus. "
            "Intervals and paired differences can be unstable at this sample size. They do not identify a winner, "
            "do not establish comparative accuracy, and do not turn the pilot labels into a gold standard. The three "
            "Ko rows are stochastic instances of one maintained implementation, not different architectures.",
            "",
        ]
    )
    (compact_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    *,
    config_path: Path,
    full_dir: Path | None = None,
    compact_dir: Path | None = None,
) -> dict[str, Any]:
    record = load_protocol(config_path)
    configured_full = Path(record["outputs"]["full_directory"])
    configured_compact = Path(record["outputs"]["compact_directory"])
    full_dir = configured_full if full_dir is None else full_dir
    compact_dir = configured_compact if compact_dir is None else compact_dir
    corpora, coverage = _load_corpora(record, config_path)

    bootstrap = record["analysis"]["bootstrap"]
    outputs: dict[str, list[dict[str, Any]]] = {
        "coverage.csv": coverage,
        "model_human_agreement.csv": [],
        "paired_model_human_contrasts.csv": [],
        "model_model_agreement.csv": [],
        "ko_run_variability.csv": [],
        "paper_table.csv": [],
    }
    for corpus_id in record["pilot"]["corpus_order"]:
        corpus_outputs = analyze_corpus(
            corpora[corpus_id],
            replicates=int(bootstrap["replicates"]),
            master_seed=int(bootstrap["master_seed"]),
            minimum_valid_fraction=float(bootstrap["minimum_valid_fraction"]),
        )
        for name, rows in corpus_outputs.items():
            outputs[name].extend(rows)

    for directory in (full_dir, compact_dir):
        directory.mkdir(parents=True, exist_ok=True)
        for name, rows in outputs.items():
            _write_csv(directory / name, rows)

    config_hash = sha256_file(config_path)
    manifest_path = Path(record["pilot"]["manifest"])
    metadata: dict[str, Any] = {
        "schema_version": "round2_pilot_model_human_results_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": {"path": _portable(config_path), "sha256": config_hash},
        "outcome_blind_freeze": {
            "commit": record["freeze_commit"],
            "committed_at_utc": record["frozen_at_utc"],
            "reporting_only_timestamp_correction": True,
        },
        "pilot_manifest": {"path": _portable(manifest_path), "sha256": sha256_file(manifest_path)},
        "coverage_gate_passed": True,
        "coverage": {"total_rows": 80, "rows_per_corpus": 40, "primary_instances_per_row": 4, "complete_labels_per_row": 3},
        "score_instances": record["score_instances"],
        "bootstrap": record["analysis"]["bootstrap"],
        "pooling_rule": record["pilot"]["pooling_rule"],
        "claim_boundary": record["analysis"]["claim_boundary"],
        "privacy_boundary": record["pilot"]["privacy"],
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "platform": platform.system()},
        "command": "python scripts/pilot_model_human_comparison.py",
        "inputs": {
            "label_hashes": [
                {"corpus_id": row["corpus_id"], "annotator_id": row["annotator_id"], "sha256": row["sha256"]}
                for row in record["label_inputs"]["files"]
            ],
            "score_and_checkpoint_hashes": record["input_identities"],
        },
        "outputs": {},
    }
    for name in outputs:
        metadata["outputs"][name] = {
            "rows": len(outputs[name]),
            "sha256": sha256_file(compact_dir / name),
        }
    for directory in (full_dir, compact_dir):
        (directory / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    _write_readme(compact_dir)
    metadata["outputs"]["README.md"] = {
        "sha256": sha256_file(compact_dir / "README.md")
    }
    (compact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


__all__ = [
    "ALL_MODELS",
    "HUMAN_TARGETS",
    "PRIMARY_MODELS",
    "PilotCorpus",
    "analyze_corpus",
    "bootstrap_correlation_matrices",
    "load_protocol",
    "run_analysis",
]
