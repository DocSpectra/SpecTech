"""Frozen local-Qwen scoring and pilot-rubric analysis for Round 2 QR-A."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.analysis.pilot_model_human import (
    HUMAN_TARGETS,
    PRIMARY_MODELS,
    PilotCorpus,
    _interval,
    _load_corpora,
    _stream_seed,
    bootstrap_correlation_matrices,
    load_protocol,
    sha256_file,
)

QWEN_ID = "qwen3_14b_zero_shot_rubric"


def _require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _portable(path: Path) -> str:
    value = path.as_posix()
    if path.is_absolute() or ":/" in value:
        raise ValueError(f"Metadata path must be repository-relative: {path}")
    return value


def load_qwen_protocol(config_path: Path) -> dict[str, Any]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_qwen_rubric_v1":
        raise ValueError("Unexpected Qwen rubric protocol schema")
    if not record.get("outcome_blind_freeze"):
        raise ValueError("Qwen rubric protocol is not marked outcome-blind")
    return record


def verify_ollama_identity(record: dict[str, Any]) -> dict[str, str]:
    qwen = record["qwen"]
    version_output = subprocess.run(
        ["ollama", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    if qwen["ollama_version"] not in version_output:
        raise ValueError(f"Ollama version mismatch: {version_output}")
    listing = subprocess.run(
        ["ollama", "list"], check=True, capture_output=True, text=True
    ).stdout
    matching = [line for line in listing.splitlines() if line.split()[:1] == [qwen["model"]]]
    if len(matching) != 1 or qwen["model_list_id"] not in matching[0]:
        raise ValueError("Ollama model name/ID mismatch")
    modelfile = subprocess.run(
        ["ollama", "show", qwen["model"], "--modelfile"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    expected_blob = f"sha256-{qwen['backing_blob_sha256']}"
    if expected_blob not in modelfile:
        raise ValueError("Ollama backing blob mismatch")
    return {
        "ollama_version_output": version_output,
        "model": qwen["model"],
        "model_list_id": qwen["model_list_id"],
        "backing_blob_sha256": qwen["backing_blob_sha256"],
    }


def load_project_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    pilot = record["pilot"]
    manifest_path = Path(pilot["manifest"])
    _require_hash(manifest_path, pilot["manifest_sha256"])
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    expected_total = int(pilot["required_rows_total"])
    if len(manifest) != expected_total:
        raise ValueError(f"Expected {expected_total} manifest rows, found {len(manifest)}")
    corpus_order = list(pilot["corpus_order"])
    order_index = {corpus: index for index, corpus in enumerate(corpus_order)}
    manifest.sort(key=lambda row: (order_index[row["corpus_id"]], int(row["pilot_position"])))
    if len({(row["corpus_id"], row["sent_id"]) for row in manifest}) != expected_total:
        raise ValueError("Duplicate pilot key")

    sentence_maps: dict[str, dict[str, str]] = {}
    for corpus_id in corpus_order:
        entry = pilot["sentence_inputs"][corpus_id]
        path = Path(entry["path"])
        _require_hash(path, entry["sha256"])
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        sentence_map: dict[str, str] = {}
        for row in rows:
            sent_id = row["sent_id"]
            if sent_id in sentence_map:
                raise ValueError(f"Duplicate canonical sentence ID in {corpus_id}: {sent_id}")
            sentence_map[sent_id] = row["sent_text"]
        sentence_maps[corpus_id] = sentence_map

    project_rows: list[dict[str, Any]] = []
    counts = {corpus: 0 for corpus in corpus_order}
    for row in manifest:
        corpus_id = row["corpus_id"]
        sent_id = row["sent_id"]
        text = sentence_maps[corpus_id].get(sent_id)
        if text is None or not text.strip():
            raise ValueError(f"Missing canonical text for {corpus_id}/{sent_id}")
        counts[corpus_id] += 1
        project_rows.append(
            {
                "corpus_id": corpus_id,
                "pilot_position": int(row["pilot_position"]),
                "bucket": row["bucket"],
                "sent_id": sent_id,
                "sent_text": text,
            }
        )
    required = int(pilot["required_rows_per_corpus"])
    if any(count != required for count in counts.values()):
        raise ValueError(f"Pilot corpus coverage mismatch: {counts}")
    prior = pilot["existing_analysis_protocol"]
    _require_hash(Path(prior["path"]), prior["sha256"])
    return project_rows


def build_request(record: dict[str, Any], sent_text: str) -> dict[str, Any]:
    qwen = record["qwen"]
    return {
        "model": qwen["model"],
        "messages": [
            {"role": "system", "content": record["prompt"]["system"]},
            {
                "role": "user",
                "content": record["prompt"]["user_template"].format(sent_text=sent_text),
            },
        ],
        "stream": qwen["stream"],
        "think": qwen["thinking"],
        "format": qwen["response_schema"],
        "options": qwen["options"],
        "keep_alive": qwen["keep_alive"],
    }


def parse_score_response(payload: dict[str, Any], expected_model: str) -> tuple[int, dict[str, Any]]:
    if payload.get("model") != expected_model:
        raise ValueError(f"Ollama response model mismatch: {payload.get('model')}")
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response lacks string message content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"score"}:
        raise ValueError(f"Response must contain only score: {parsed!r}")
    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, int) or score not in range(1, 6):
        raise ValueError(f"Invalid rubric score: {score!r}")
    metadata = {
        "model": payload["model"],
        "created_at": payload.get("created_at"),
        "done_reason": payload.get("done_reason"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "total_duration": payload.get("total_duration"),
        "load_duration": payload.get("load_duration"),
        "prompt_eval_duration": payload.get("prompt_eval_duration"),
        "eval_duration": payload.get("eval_duration"),
        "response_content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
    }
    return score, metadata


def request_score(record: dict[str, Any], sent_text: str, *, timeout: float = 300.0) -> tuple[int, dict[str, Any]]:
    response = requests.post(
        record["qwen"]["api_url"],
        json=build_request(record, sent_text),
        timeout=timeout,
    )
    response.raise_for_status()
    return parse_score_response(response.json(), record["qwen"]["model"])


def run_determinism_smoke(record: dict[str, Any], output_path: Path) -> dict[str, Any]:
    smoke = record["determinism_smoke"]
    results: list[dict[str, Any]] = []
    for fixture_index, text in enumerate(smoke["fixtures"], start=1):
        fixture_scores: list[int] = []
        response_hashes: list[str] = []
        for repetition in range(1, int(smoke["repetitions"]) + 1):
            score, metadata = request_score(record, text)
            fixture_scores.append(score)
            response_hashes.append(metadata["response_content_sha256"])
            results.append(
                {
                    "fixture_id": f"fixture_{fixture_index:02d}",
                    "repetition": repetition,
                    "score": score,
                    "response_content_sha256": metadata["response_content_sha256"],
                }
            )
        if len(set(fixture_scores)) != 1:
            raise ValueError(f"Determinism smoke failed for fixture {fixture_index}: {fixture_scores}")
    output = {
        "schema_version": "round2_qwen_rubric_determinism_smoke_v1",
        "passed": True,
        "model": record["qwen"]["model"],
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _read_raw_scores(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    keys = [(row["corpus_id"], row["sent_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate raw Qwen score key")
    return rows


def score_project_rows(
    record: dict[str, Any],
    rows: list[dict[str, Any]],
    raw_path: Path,
    *,
    resume: bool,
    protocol_sha256: str,
) -> list[dict[str, Any]]:
    existing = _read_raw_scores(raw_path)
    if existing and not resume:
        raise ValueError(f"Raw score output already exists; use --resume after validation: {raw_path}")
    expected = {(row["corpus_id"], row["sent_id"]): row for row in rows}
    for row in existing:
        key = (row["corpus_id"], row["sent_id"])
        if (
            key not in expected
            or row["score"] not in range(1, 6)
            or row.get("protocol_sha256") != protocol_sha256
            or row.get("backing_blob_sha256") != record["qwen"]["backing_blob_sha256"]
        ):
            raise ValueError(f"Invalid resumed Qwen row: {key}")
    completed = {(row["corpus_id"], row["sent_id"]) for row in existing}
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            key = (row["corpus_id"], row["sent_id"])
            if key in completed:
                continue
            score, metadata = request_score(record, row["sent_text"])
            output = {
                "corpus_id": row["corpus_id"],
                "pilot_position": row["pilot_position"],
                "sent_id": row["sent_id"],
                "score": score,
                "protocol_sha256": protocol_sha256,
                "backing_blob_sha256": record["qwen"]["backing_blob_sha256"],
                **metadata,
            }
            handle.write(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
    observed = _read_raw_scores(raw_path)
    if len(observed) != len(rows):
        raise ValueError(f"Incomplete Qwen coverage: {len(observed)}/{len(rows)}")
    observed_keys = {(row["corpus_id"], row["sent_id"]) for row in observed}
    if observed_keys != set(expected):
        raise ValueError("Qwen score keys differ from frozen manifest")
    return observed


def run_scoring(*, config_path: Path, resume: bool = False) -> dict[str, Any]:
    record = load_qwen_protocol(config_path)
    protocol_sha256 = sha256_file(config_path)
    identity = verify_ollama_identity(record)
    rows = load_project_rows(record)
    raw_dir = Path(record["outputs"]["raw_directory"])
    smoke = run_determinism_smoke(record, raw_dir / record["outputs"]["smoke"])
    scores = score_project_rows(
        record,
        rows,
        raw_dir / record["outputs"]["raw_responses"],
        resume=resume,
        protocol_sha256=protocol_sha256,
    )
    return {"identity": identity, "smoke": smoke, "scores": scores}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: f"{value:.10f}" if isinstance(value, float) else value for key, value in row.items()}
            )


def _load_qwen_score_map(record: dict[str, Any]) -> tuple[dict[tuple[str, str], int], Path]:
    raw_path = Path(record["outputs"]["raw_directory"]) / record["outputs"]["raw_responses"]
    rows = _read_raw_scores(raw_path)
    expected_total = int(record["pilot"]["required_rows_total"])
    if len(rows) != expected_total:
        raise ValueError(f"Expected {expected_total} Qwen rows, found {len(rows)}")
    mapping = {(row["corpus_id"], row["sent_id"]): int(row["score"]) for row in rows}
    if any(score not in range(1, 6) for score in mapping.values()):
        raise ValueError("Out-of-range Qwen score")
    return mapping, raw_path


def _analyze_corpus(corpus: PilotCorpus, *, replicates: int, seed: int, minimum_valid_fraction: float) -> dict[str, list[dict[str, Any]]]:
    models = [QWEN_ID, *PRIMARY_MODELS]
    column_ids = models + list(HUMAN_TARGETS)
    columns = np.vstack(
        [corpus.scores[model] for model in models]
        + [corpus.labels[target] for target in HUMAN_TARGETS]
    )
    point, boot = bootstrap_correlation_matrices(
        columns, replicates=replicates, seed=_stream_seed(seed, corpus.corpus_id)
    )
    indices = {name: index for index, name in enumerate(column_ids)}
    human_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    model_rows: list[dict[str, Any]] = []
    ordinal_rows: list[dict[str, Any]] = []

    qi = indices[QWEN_ID]
    for target in HUMAN_TARGETS:
        ti = indices[target]
        low, high, valid, degenerate = _interval(
            boot[:, qi, ti], replicates, minimum_valid_fraction
        )
        human_rows.append(
            {
                "corpus_id": corpus.corpus_id,
                "model_instance_id": QWEN_ID,
                "target_id": target,
                "n": len(corpus.sent_ids),
                "spearman_rho": float(point[qi, ti]),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_valid": valid,
                "bootstrap_degenerate": degenerate,
            }
        )
        differences = np.abs(corpus.scores[QWEN_ID] - corpus.labels[target])
        ordinal_rows.append(
            {
                "corpus_id": corpus.corpus_id,
                "target_id": target,
                "n": len(corpus.sent_ids),
                "mean_absolute_ordinal_distance": float(np.mean(differences)),
                "exact_agreement_rate": "" if target == "pooled_human_mean" else float(np.mean(differences == 0)),
                "within_one_rate": "" if target == "pooled_human_mean" else float(np.mean(differences <= 1)),
            }
        )
        for model in PRIMARY_MODELS:
            mi = indices[model]
            delta_boot = boot[:, qi, ti] - boot[:, mi, ti]
            low_d, high_d, valid_d, degenerate_d = _interval(
                delta_boot, replicates, minimum_valid_fraction
            )
            contrast_rows.append(
                {
                    "corpus_id": corpus.corpus_id,
                    "target_id": target,
                    "model_a": QWEN_ID,
                    "model_b": model,
                    "n": len(corpus.sent_ids),
                    "rho_a": float(point[qi, ti]),
                    "rho_b": float(point[mi, ti]),
                    "delta_rho_a_minus_b": float(point[qi, ti] - point[mi, ti]),
                    "ci_low": low_d,
                    "ci_high": high_d,
                    "bootstrap_valid": valid_d,
                    "bootstrap_degenerate": degenerate_d,
                }
            )
    for model in PRIMARY_MODELS:
        mi = indices[model]
        low, high, valid, degenerate = _interval(
            boot[:, qi, mi], replicates, minimum_valid_fraction
        )
        model_rows.append(
            {
                "corpus_id": corpus.corpus_id,
                "model_a": QWEN_ID,
                "model_b": model,
                "n": len(corpus.sent_ids),
                "spearman_rho": float(point[qi, mi]),
                "ci_low": low,
                "ci_high": high,
                "bootstrap_valid": valid,
                "bootstrap_degenerate": degenerate,
            }
        )
    distribution_rows = []
    values = corpus.scores[QWEN_ID]
    for score in range(1, 6):
        count = int(np.sum(values == score))
        distribution_rows.append(
            {
                "corpus_id": corpus.corpus_id,
                "score": score,
                "count": count,
                "proportion": count / len(values),
            }
        )
    paper_row = {"corpus_id": corpus.corpus_id, "n": len(corpus.sent_ids)}
    for target in HUMAN_TARGETS:
        agreement = next(row for row in human_rows if row["target_id"] == target)
        prefix = target.replace("pooled_human_mean", "pooled")
        paper_row[f"{prefix}_rho"] = agreement["spearman_rho"]
        paper_row[f"{prefix}_ci_low"] = agreement["ci_low"]
        paper_row[f"{prefix}_ci_high"] = agreement["ci_high"]
    return {
        "qwen_human_agreement.csv": human_rows,
        "qwen_vs_model_human_contrasts.csv": contrast_rows,
        "qwen_model_agreement.csv": model_rows,
        "ordinal_agreement.csv": ordinal_rows,
        "score_distribution.csv": distribution_rows,
        "paper_table.csv": [paper_row],
    }


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def run_analysis(*, config_path: Path) -> dict[str, Any]:
    record = load_qwen_protocol(config_path)
    identity = verify_ollama_identity(record)
    project_rows = load_project_rows(record)
    score_map, raw_path = _load_qwen_score_map(record)
    prior_config = Path(record["pilot"]["existing_analysis_protocol"]["path"])
    prior_record = load_protocol(prior_config)
    corpora, _ = _load_corpora(prior_record, prior_config)

    qwen_score_rows: list[dict[str, Any]] = []
    for row in project_rows:
        key = (row["corpus_id"], row["sent_id"])
        qwen_score_rows.append(
            {
                "corpus_id": row["corpus_id"],
                "pilot_position": row["pilot_position"],
                "sent_id": row["sent_id"],
                "qwen_score": score_map[key],
            }
        )
    for corpus_id, corpus in list(corpora.items()):
        values = np.asarray([score_map[(corpus_id, sent_id)] for sent_id in corpus.sent_ids], dtype=float)
        corpora[corpus_id] = replace(corpus, scores={**corpus.scores, QWEN_ID: values})

    bootstrap = record["analysis"]["bootstrap"]
    outputs: dict[str, list[dict[str, Any]]] = {
        "qwen_scores.csv": qwen_score_rows,
        "coverage.csv": [],
        "qwen_human_agreement.csv": [],
        "qwen_vs_model_human_contrasts.csv": [],
        "qwen_model_agreement.csv": [],
        "ordinal_agreement.csv": [],
        "score_distribution.csv": [],
        "paper_table.csv": [],
    }
    outputs["coverage.csv"].extend(
        [
            {
                "corpus_id": "all",
                "artifact_type": "pilot_manifest",
                "expected_rows": 80,
                "observed_rows": len(project_rows),
                "coverage_rate": len(project_rows) / 80,
                "gate_passed": len(project_rows) == 80,
            },
            {
                "corpus_id": "all",
                "artifact_type": "prior_analysis_protocol",
                "expected_rows": 1,
                "observed_rows": 1,
                "coverage_rate": 1.0,
                "gate_passed": True,
            },
        ]
    )
    for corpus_id in record["pilot"]["corpus_order"]:
        observed = sum(1 for row in qwen_score_rows if row["corpus_id"] == corpus_id)
        outputs["coverage.csv"].append(
            {
                "corpus_id": corpus_id,
                "artifact_type": "qwen_score",
                "expected_rows": 40,
                "observed_rows": observed,
                "coverage_rate": observed / 40,
                "gate_passed": observed == 40,
            }
        )
        result = _analyze_corpus(
            corpora[corpus_id],
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["master_seed"]),
            minimum_valid_fraction=float(bootstrap["minimum_valid_fraction"]),
        )
        for name, rows in result.items():
            outputs[name].extend(rows)

    compact_dir = Path(record["outputs"]["compact_directory"])
    compact_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        _write_csv(compact_dir / name, rows)
    smoke_path = Path(record["outputs"]["raw_directory"]) / record["outputs"]["smoke"]
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if smoke.get("passed") is not True:
        raise ValueError("Determinism smoke is not passing")
    compact_smoke_path = compact_dir / "determinism_smoke.json"
    compact_smoke_path.write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    freeze_record_path = Path("configs/round2_qwen_rubric_freeze_record.json")
    freeze_record = json.loads(freeze_record_path.read_text(encoding="utf-8"))
    if freeze_record.get("method_frozen_before_any_qwen_project_output") is not True:
        raise ValueError("Qwen method freeze record is not valid")
    metadata: dict[str, Any] = {
        "schema_version": "round2_qwen_rubric_results_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_blind_freeze_commit": freeze_record["method_freeze_commit"],
        "analysis_execution_commit": _git_head(),
        "freeze_record": {
            "path": _portable(freeze_record_path),
            "sha256": sha256_file(freeze_record_path),
            **freeze_record,
        },
        "protocol": {"path": _portable(config_path), "sha256": sha256_file(config_path)},
        "model": identity,
        "coverage": {"total_rows": 80, "rows_per_corpus": 40, "coverage_rate": 1.0},
        "determinism_smoke_passed": True,
        "determinism_smoke_sha256": sha256_file(smoke_path),
        "raw_response_sha256": sha256_file(raw_path),
        "bootstrap": bootstrap,
        "claim_boundary": record["analysis"]["claim_boundary"],
        "privacy_boundary": record["scoring_contract"]["privacy"],
        "inputs": {
            "pilot_manifest": {
                "path": record["pilot"]["manifest"],
                "sha256": record["pilot"]["manifest_sha256"],
            },
            "canonical_sentences": record["pilot"]["sentence_inputs"],
            "prior_analysis_protocol": record["pilot"]["existing_analysis_protocol"],
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "requests": requests.__version__,
            "platform": platform.system(),
        },
        "outputs": {},
    }
    for name, rows in outputs.items():
        metadata["outputs"][name] = {
            "rows": len(rows),
            "sha256": sha256_file(compact_dir / name),
        }
    metadata["outputs"]["determinism_smoke.json"] = {
        "sha256": sha256_file(compact_smoke_path)
    }
    (compact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    pooled = {
        row["corpus_id"]: row
        for row in outputs["qwen_human_agreement.csv"]
        if row["target_id"] == "pooled_human_mean"
    }
    lines = [
        "# Round 2 Qwen zero-shot rubric comparison",
        "",
        "All 80 frozen pilot rows received one valid deterministic local-Qwen 1--5 score.",
        "",
        "## Pooled-label headline (Spearman rho; paired sentence-bootstrap 95% CI)",
        "",
        "| Corpus | Qwen rho [95% CI] |",
        "| --- | ---: |",
    ]
    for corpus_id in record["pilot"]["corpus_order"]:
        row = pooled[corpus_id]
        label = "Ansible" if corpus_id == "ansible_docs" else "GitHub"
        lines.append(f"| {label} | {row['spearman_rho']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Qwen applies the existing rubric as a zero-shot LLM judge. Correspondence to the three non-expert pilot label sets is descriptive, not gold-standard accuracy or proof that Qwen measures latent specificity.",
            "",
        ]
    )
    (compact_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    metadata["outputs"]["README.md"] = {"sha256": sha256_file(compact_dir / "README.md")}
    (compact_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


__all__ = [
    "QWEN_ID",
    "build_request",
    "load_project_rows",
    "load_qwen_protocol",
    "parse_score_response",
    "run_analysis",
    "run_determinism_smoke",
    "run_scoring",
    "verify_ollama_identity",
]
