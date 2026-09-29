"""Matched Gemma/GPT-OSS replication of the frozen Qwen rubric protocol."""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import subprocess
from dataclasses import replace
from decimal import Decimal
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.analysis.pilot_model_human import (
    HUMAN_TARGETS,
    PRIMARY_MODELS,
    SECONDARY_MODEL,
    _interval,
    _load_corpora,
    _stream_seed,
    bootstrap_correlation_matrices,
    load_protocol,
    sha256_file,
)
from src.analysis.qwen_rubric import load_project_rows

QWEN_ID = "qwen3_14b_zero_shot_rubric"
GRANUSCORE_ALIGNED = "granuscore_direction_aligned_secondary"
INSTALLED_MODELS = ("qwen3:14b", "gemma4:12b", "gpt-oss:20b")
FREEZE_RECORD = Path("configs/round2_local_rubric_replication_freeze_record.json")


def _require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def load_freeze_binding(path: Path = FREEZE_RECORD) -> dict[str, Any]:
    freeze = json.loads(path.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "round2_local_rubric_replication_freeze_record_v1":
        raise ValueError("Unexpected local rubric freeze-record schema")
    if freeze.get("method_frozen_before_any_project_judge_output") is not True:
        raise ValueError("Method is not bound before project judge output")
    if freeze.get("project_judge_calls_before_freeze") != 0:
        raise ValueError("Freeze record reports pre-freeze project calls")
    if freeze.get("project_outcomes_inspected_before_freeze") is not False:
        raise ValueError("Freeze record does not preserve outcome blindness")
    chronology = freeze.get("chronology_audit", {})
    if chronology.get("pre_freeze_project_sentence_calls_after_base") != 0:
        raise ValueError("Chronology audit reports a pre-freeze project sentence call")
    method_commit = freeze.get("method_freeze_commit", "")
    if len(method_commit) != 40:
        raise ValueError("Invalid method-freeze commit identity")
    binding_commit = _git("log", "-1", "--format=%H", "--", path.as_posix())
    if len(binding_commit) != 40:
        raise ValueError("Freeze binding is not committed")
    committed = subprocess.run(
        ["git", "show", f"{binding_commit}:{path.as_posix()}"],
        check=True,
        capture_output=True,
    ).stdout
    if hashlib.sha256(committed).hexdigest() != sha256_file(path):
        raise ValueError("Working freeze record differs from committed binding")
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", method_commit, binding_commit],
        check=True,
        capture_output=True,
    )
    return {**freeze, "freeze_binding_commit": binding_commit}


def load_replication_protocol(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_local_rubric_replication_v1":
        raise ValueError("Unexpected local rubric replication schema")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("Protocol is not marked outcome-blind")
    qwen_entry = record["qwen_protocol"]
    qwen_path = Path(qwen_entry["path"])
    _require_hash(qwen_path, qwen_entry["sha256"])
    qwen = json.loads(qwen_path.read_text(encoding="utf-8"))
    for key in qwen_entry["inherit_exactly"]:
        if key not in qwen:
            raise ValueError(f"Missing inherited Qwen protocol key: {key}")
    for entry in record["comparison_inputs"].values():
        if isinstance(entry, dict) and "path" in entry:
            _require_hash(Path(entry["path"]), entry["sha256"])
    for entry in record["comparison_inputs"]["granuscore_scores"].values():
        _require_hash(Path(entry["path"]), entry["sha256"])
    for judge in record["judges"]:
        _require_hash(Path(judge["identity_source"]["path"]), judge["identity_source"]["sha256"])
    return record, qwen


def _ollama(args: list[str]) -> str:
    return subprocess.run(
        ["ollama", *args],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout.strip()


def stop_all_models() -> None:
    for model in INSTALLED_MODELS:
        _ollama(["stop", model])
    if _ollama(["ps"]).splitlines()[1:]:
        raise ValueError("Ollama residency is not empty after stop gate")


def verify_identity(record: dict[str, Any], judge: dict[str, Any]) -> dict[str, str]:
    version = _ollama(["--version"])
    if record["runtime"]["ollama_version"] not in version:
        raise ValueError(f"Ollama version mismatch: {version}")
    listing = _ollama(["list"])
    lines = [line for line in listing.splitlines() if line.split()[:1] == [judge["model"]]]
    if len(lines) != 1 or judge["model_list_id"] not in lines[0]:
        raise ValueError(f"Installed model identity mismatch: {judge['model']}")
    modelfile = _ollama(["show", judge["model"], "--modelfile"])
    if f"sha256-{judge['model_blob_sha256']}" not in modelfile:
        raise ValueError("Model backing blob mismatch")
    projector = judge.get("projector_blob_sha256")
    if projector and f"sha256-{projector}" not in modelfile:
        raise ValueError("Projector backing blob mismatch")
    return {
        "ollama_version": version,
        "model": judge["model"],
        "model_list_id": judge["model_list_id"],
        "model_tag_digest": judge["model_tag_digest"],
        "model_blob_sha256": judge["model_blob_sha256"],
    }


def verify_sole_residency(judge: dict[str, Any]) -> dict[str, str]:
    lines = _ollama(["ps"]).splitlines()
    rows = [line for line in lines[1:] if line.strip()]
    if len(rows) != 1 or rows[0].split()[:2] != [judge["model"], judge["model_list_id"]]:
        raise ValueError(f"Sole-residency gate failed for {judge['model']}: {rows}")
    return {"model": judge["model"], "model_list_id": judge["model_list_id"], "ps_row_sha256": hashlib.sha256(rows[0].encode()).hexdigest()}


def build_request(record: dict[str, Any], qwen: dict[str, Any], judge: dict[str, Any], text: str) -> dict[str, Any]:
    options = dict(record["runtime"]["options_matched_to_qwen"])
    options["num_predict"] = judge["num_predict"]
    return {
        "model": judge["model"],
        "messages": [
            {"role": "system", "content": qwen["prompt"]["system"]},
            {"role": "user", "content": qwen["prompt"]["user_template"].format(sent_text=text)},
        ],
        "stream": False,
        "think": judge["think"],
        "format": qwen["qwen"]["response_schema"],
        "options": options,
        "keep_alive": record["runtime"]["keep_alive"],
    }


def parse_response(payload: dict[str, Any], expected_model: str) -> tuple[int, dict[str, Any]]:
    if payload.get("model") != expected_model:
        raise ValueError("Ollama response model mismatch")
    message = payload.get("message", {})
    content = message.get("content")
    if not isinstance(content, str):
        raise ValueError("Response lacks string final content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"score"}:
        raise ValueError("Response must contain only score")
    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, int) or score not in range(1, 6):
        raise ValueError(f"Invalid score: {score!r}")
    thinking = message.get("thinking", "")
    if not isinstance(thinking, str):
        raise ValueError("Unexpected reasoning field type")
    return score, {
        "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_char_count": len(thinking),
        "reasoning_sha256": hashlib.sha256(thinking.encode()).hexdigest(),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "done_reason": payload.get("done_reason"),
    }


def request_score(record: dict[str, Any], qwen: dict[str, Any], judge: dict[str, Any], text: str) -> tuple[int, dict[str, Any]]:
    response = requests.post(record["runtime"]["api_url"], json=build_request(record, qwen, judge, text), timeout=600)
    response.raise_for_status()
    return parse_response(response.json(), judge["model"])


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    keys = [(row["corpus_id"], row["sent_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate raw judge row")
    return rows


def run_judge(
    record: dict[str, Any],
    qwen: dict[str, Any],
    judge: dict[str, Any],
    config_hash: str,
    freeze: dict[str, Any],
) -> dict[str, Any]:
    identity = verify_identity(record, judge)
    rows = load_project_rows(qwen)
    raw_dir = Path(record["outputs"]["raw_directory"]) / judge["judge_id"]
    raw_path = raw_dir / "raw_responses.jsonl"
    smoke_path = raw_dir / "determinism_smoke.json"
    judge_metadata_path = raw_dir / "run_metadata.json"
    if raw_path.exists() or smoke_path.exists() or judge_metadata_path.exists():
        raise ValueError(f"Fresh run required; output already exists for {judge['judge_id']}")
    run_started_at_utc = datetime.now(timezone.utc).isoformat()
    stop_all_models()
    try:
        smoke_rows: list[dict[str, Any]] = []
        for fixture_index, text in enumerate(qwen["determinism_smoke"]["fixtures"], 1):
            scores = []
            response_hashes = []
            for repetition in range(1, int(qwen["determinism_smoke"]["repetitions"]) + 1):
                score, metadata = request_score(record, qwen, judge, text)
                if fixture_index == 1 and repetition == 1:
                    residency_after_warmup = verify_sole_residency(judge)
                scores.append(score)
                response_hashes.append(metadata["response_content_sha256"])
                smoke_rows.append({"fixture_id": f"fixture_{fixture_index:02d}", "repetition": repetition, "score": score, **metadata})
            if len(set(scores)) != 1 or len(set(response_hashes)) != 1:
                raise ValueError(
                    f"Determinism smoke failed for {judge['judge_id']} "
                    f"fixture {fixture_index}"
                )
        raw_dir.mkdir(parents=True, exist_ok=True)
        smoke = {"schema_version": "round2_local_rubric_smoke_v1", "judge_id": judge["judge_id"], "passed": True, "residency_after_warmup": residency_after_warmup, "results": smoke_rows}
        smoke_path.write_text(json.dumps(smoke, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        project_scoring_started_at_utc = datetime.now(timezone.utc).isoformat()
        with raw_path.open("x", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                score, metadata = request_score(record, qwen, judge, row["sent_text"])
                output = {"judge_id": judge["judge_id"], "corpus_id": row["corpus_id"], "pilot_position": row["pilot_position"], "sent_id": row["sent_id"], "score": score, "protocol_sha256": config_hash, "model_tag_digest": judge["model_tag_digest"], **metadata}
                handle.write(json.dumps(output, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
        residency_after_matrix = verify_sole_residency(judge)
        observed = _read_jsonl(raw_path)
        expected_keys = {(row["corpus_id"], row["sent_id"]) for row in rows}
        if len(observed) != 80 or {(row["corpus_id"], row["sent_id"]) for row in observed} != expected_keys:
            raise ValueError("Incomplete or mismatched judge coverage")
        judge_metadata = {
            "schema_version": "round2_local_rubric_judge_run_v1",
            "judge_id": judge["judge_id"],
            "run_started_at_utc": run_started_at_utc,
            "project_scoring_started_at_utc": project_scoring_started_at_utc,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "method_freeze_commit": freeze["method_freeze_commit"],
            "freeze_binding_commit": freeze["freeze_binding_commit"],
            "protocol_sha256": config_hash,
            "identity": identity,
            "coverage": {"total": 80, "ansible_docs": 40, "github_docs": 40},
            "determinism_smoke_sha256": sha256_file(smoke_path),
            "raw_responses_sha256": sha256_file(raw_path),
            "residency_after_warmup": residency_after_warmup,
            "residency_after_matrix": residency_after_matrix,
        }
        judge_metadata_path.write_text(
            json.dumps(judge_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return judge_metadata
    finally:
        _ollama(["stop", judge["model"]])
        if _ollama(["ps"]).splitlines()[1:]:
            raise ValueError("Ollama residency is not empty after judge closeout")


def run_scoring(config_path: Path) -> list[dict[str, Any]]:
    freeze = load_freeze_binding()
    record, qwen = load_replication_protocol(config_path)
    config_hash = sha256_file(config_path)
    return [
        run_judge(record, qwen, judge, config_hash, freeze)
        for judge in record["judges"]
    ]


def _load_map(path: Path, score_field: str, expected_hash: str) -> dict[tuple[str, str], float]:
    _require_hash(path, expected_hash)
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {(row["corpus_id"], row["sent_id"]): float(row[score_field]) for row in rows}
    if len(result) != len(rows):
        raise ValueError(f"Duplicate key in {path}")
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.10f}" if isinstance(value, float) else value for key, value in row.items()})


def _exact_quantized_ko_mean(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Preserve decimal ties in the secondary mean of frozen Ko CSV scores."""
    runs = np.vstack([scores[model] for model in PRIMARY_MODELS[1:]])
    return np.asarray(
        [
            float(sum(Decimal(str(value)) for value in runs[:, index]) / Decimal(3))
            for index in range(runs.shape[1])
        ],
        dtype=np.float64,
    )


def run_analysis(config_path: Path) -> dict[str, Any]:
    record, qwen = load_replication_protocol(config_path)
    freeze_path = FREEZE_RECORD
    freeze = load_freeze_binding(freeze_path)
    project_rows = load_project_rows(qwen)
    prior_path = Path(qwen["pilot"]["existing_analysis_protocol"]["path"])
    corpora, _ = _load_corpora(load_protocol(prior_path), prior_path)
    qentry = record["comparison_inputs"]["qwen_scores"]
    qwen_map = _load_map(Path(qentry["path"]), "qwen_score", qentry["sha256"])
    gran_maps = {}
    for corpus_id, entry in record["comparison_inputs"]["granuscore_scores"].items():
        gran_maps.update(_load_map(Path(entry["path"]), "granuscore_percentile", entry["sha256"]))
    judge_maps: dict[str, dict[tuple[str, str], float]] = {}
    score_rows = []
    for judge in record["judges"]:
        path = Path(record["outputs"]["raw_directory"]) / judge["judge_id"] / "raw_responses.jsonl"
        rows = _read_jsonl(path)
        if len(rows) != 80:
            raise ValueError(f"Incomplete raw judge matrix: {judge['judge_id']}")
        judge_maps[judge["judge_id"]] = {(row["corpus_id"], row["sent_id"]): float(row["score"]) for row in rows}
        score_rows.extend({"judge_id": judge["judge_id"], "corpus_id": row["corpus_id"], "pilot_position": row["pilot_position"], "sent_id": row["sent_id"], "score": row["score"]} for row in rows)
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in ("judge_scores.csv", "coverage.csv", "judge_human_agreement.csv", "judge_vs_comparator_human_contrasts.csv", "judge_comparator_agreement.csv", "judge_pair_human_contrasts.csv", "judge_pair_agreement.csv", "ordinal_agreement.csv", "score_distribution.csv", "pooled_summary.csv", "determinism_smoke_variability.csv", "run_variability_scope.csv")}
    outputs["judge_scores.csv"] = score_rows
    bootstrap = qwen["analysis"]["bootstrap"]
    judge_ids = [judge["judge_id"] for judge in record["judges"]]
    comparators = list(record["analysis_extension"]["comparators"])
    if SECONDARY_MODEL in comparators:
        raise ValueError("Secondary Ko mean must not be declared as a primary comparator")
    comparators.append(SECONDARY_MODEL)
    for judge in record["judges"]:
        smoke_path = Path(record["outputs"]["raw_directory"]) / judge["judge_id"] / "determinism_smoke.json"
        smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
        for fixture_id in sorted({row["fixture_id"] for row in smoke["results"]}):
            fixture_rows = [row for row in smoke["results"] if row["fixture_id"] == fixture_id]
            scores = [int(row["score"]) for row in fixture_rows]
            hashes = [row["response_content_sha256"] for row in fixture_rows]
            outputs["determinism_smoke_variability.csv"].append({
                "judge_id": judge["judge_id"],
                "fixture_id": fixture_id,
                "repetitions": len(fixture_rows),
                "unique_score_count": len(set(scores)),
                "minimum_score": min(scores),
                "maximum_score": max(scores),
                "unique_response_hash_count": len(set(hashes)),
                "deterministic_gate_passed": len(set(scores)) == len(set(hashes)) == 1,
            })
        outputs["run_variability_scope.csv"].append({
            "judge_id": judge["judge_id"],
            "project_matrix_runs": 1,
            "project_calls_per_sentence": 1,
            "project_run_variability_estimable": False,
            "non_project_smoke_fixtures": 3,
            "non_project_repetitions_per_fixture": 3,
            "interpretation": "single frozen project matrix; variability evidence is limited to invented determinism fixtures",
        })
    for corpus_id, corpus in corpora.items():
        keys = [(corpus_id, sent_id) for sent_id in corpus.sent_ids]
        extra = {QWEN_ID: np.asarray([qwen_map[key] for key in keys]), GRANUSCORE_ALIGNED: -np.asarray([gran_maps[key] for key in keys])}
        extra.update({judge_id: np.asarray([judge_maps[judge_id][key] for key in keys]) for judge_id in judge_ids})
        corrected_scores = {
            **corpus.scores,
            SECONDARY_MODEL: _exact_quantized_ko_mean(dict(corpus.scores)),
            **extra,
        }
        corpus = replace(corpus, scores=corrected_scores)
        model_ids = tuple(judge_ids + comparators)
        columns = np.vstack([corpus.scores[name] for name in model_ids] + [corpus.labels[target] for target in HUMAN_TARGETS])
        point, boot = bootstrap_correlation_matrices(columns, replicates=int(bootstrap["replicates"]), seed=_stream_seed(int(bootstrap["master_seed"]), corpus_id))
        index = {name: i for i, name in enumerate(model_ids + HUMAN_TARGETS)}
        for judge_id in judge_ids:
            ji = index[judge_id]
            outputs["coverage.csv"].append({"corpus_id": corpus_id, "judge_id": judge_id, "expected_rows": 40, "observed_rows": len(corpus.sent_ids), "coverage_rate": 1.0, "gate_passed": True})
            for target in HUMAN_TARGETS:
                ti = index[target]
                low, high, valid, deg = _interval(boot[:, ji, ti], int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"]))
                agreement = {"corpus_id": corpus_id, "judge_id": judge_id, "target_id": target, "n": len(corpus.sent_ids), "spearman_rho": float(point[ji, ti]), "ci_low": low, "ci_high": high, "bootstrap_valid": valid, "bootstrap_degenerate": deg}
                outputs["judge_human_agreement.csv"].append(agreement)
                differences = np.abs(corpus.scores[judge_id] - corpus.labels[target])
                outputs["ordinal_agreement.csv"].append({"corpus_id": corpus_id, "judge_id": judge_id, "target_id": target, "n": len(corpus.sent_ids), "mean_absolute_ordinal_distance": float(np.mean(differences)), "exact_agreement_rate": "" if target == "pooled_human_mean" else float(np.mean(differences == 0)), "within_one_rate": "" if target == "pooled_human_mean" else float(np.mean(differences <= 1))})
                if target == "pooled_human_mean":
                    outputs["pooled_summary.csv"].append(agreement)
                for comparator in comparators:
                    ci = index[comparator]
                    delta = boot[:, ji, ti] - boot[:, ci, ti]
                    dlo, dhi, dvalid, ddeg = _interval(delta, int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"]))
                    outputs["judge_vs_comparator_human_contrasts.csv"].append({"corpus_id": corpus_id, "target_id": target, "judge_id": judge_id, "comparator_id": comparator, "n": len(corpus.sent_ids), "judge_rho": float(point[ji, ti]), "comparator_rho": float(point[ci, ti]), "delta_rho": float(point[ji, ti]-point[ci, ti]), "ci_low": dlo, "ci_high": dhi, "bootstrap_valid": dvalid, "bootstrap_degenerate": ddeg})
            for comparator in comparators:
                ci = index[comparator]
                low, high, valid, deg = _interval(boot[:, ji, ci], int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"]))
                outputs["judge_comparator_agreement.csv"].append({"corpus_id": corpus_id, "judge_id": judge_id, "comparator_id": comparator, "n": len(corpus.sent_ids), "spearman_rho": float(point[ji, ci]), "ci_low": low, "ci_high": high, "bootstrap_valid": valid, "bootstrap_degenerate": deg})
            for score in range(1, 6):
                count = int(np.sum(corpus.scores[judge_id] == score))
                outputs["score_distribution.csv"].append({"corpus_id": corpus_id, "judge_id": judge_id, "score": score, "count": count, "proportion": count / len(corpus.sent_ids)})
        a, b = index[judge_ids[0]], index[judge_ids[1]]
        low, high, valid, deg = _interval(boot[:, a, b], int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"]))
        outputs["judge_pair_agreement.csv"].append({"corpus_id": corpus_id, "model_a": judge_ids[0], "model_b": judge_ids[1], "n": len(corpus.sent_ids), "spearman_rho": float(point[a,b]), "ci_low": low, "ci_high": high, "bootstrap_valid": valid, "bootstrap_degenerate": deg})
        for target in HUMAN_TARGETS:
            ti = index[target]
            delta = boot[:, a, ti] - boot[:, b, ti]
            dlo, dhi, dvalid, ddeg = _interval(delta, int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"]))
            outputs["judge_pair_human_contrasts.csv"].append({"corpus_id": corpus_id, "target_id": target, "model_a": judge_ids[0], "model_b": judge_ids[1], "delta_rho_a_minus_b": float(point[a,ti]-point[b,ti]), "ci_low": dlo, "ci_high": dhi, "bootstrap_valid": dvalid, "bootstrap_degenerate": ddeg})
    compact = Path(record["outputs"]["compact_directory"])
    compact.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        _write_csv(compact / name, rows)
    pooled = {
        (row["corpus_id"], row["judge_id"]): row
        for row in outputs["pooled_summary.csv"]
    }
    lines = [
        "# Matched Gemma and GPT-OSS zero-shot rubric replication",
        "",
        "Both local judges apply the unchanged Qwen 1--5 rubric once to each of the same 80 pilot sentences.",
        "",
        "## Pooled-human correspondence",
        "",
        "| Corpus | Judge | Spearman rho [95% paired-bootstrap CI] |",
        "| --- | --- | ---: |",
    ]
    for corpus_id in ("ansible_docs", "github_docs"):
        for judge_id in judge_ids:
            row = pooled[(corpus_id, judge_id)]
            lines.append(
                f"| {corpus_id} | {judge_id} | {row['spearman_rho']:.3f} "
                f"[{row['ci_low']:.3f}, {row['ci_high']:.3f}] |"
            )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            record["analysis_extension"]["claim_boundary"],
            "All model-order comparisons are descriptive paired contrasts, not accuracy or superiority tests. The three Ko runs are primary stochastic instances; their row-wise mean is secondary only. Each judge has one project-matrix run, so repeated-run variability is not estimable beyond the invented determinism fixtures.",
            "",
        ]
    )
    readme_path = compact / "README.md"
    readme_path.write_text("\n".join(lines), encoding="utf-8")
    primary_comparators = [
        comparator
        for comparator in record["analysis_extension"]["comparators"]
        if comparator != GRANUSCORE_ALIGNED
    ]
    metadata = {"schema_version": "round2_local_rubric_replication_results_v1", "completed_at_utc": datetime.now(timezone.utc).isoformat(), "method_freeze_commit": freeze["method_freeze_commit"], "freeze_binding_commit": freeze["freeze_binding_commit"], "analysis_execution_commit": _git("rev-parse", "HEAD"), "freeze_record": {"path": freeze_path.as_posix(), "sha256": sha256_file(freeze_path)}, "protocol": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)}, "bootstrap": bootstrap, "comparison_status": {"primary": primary_comparators, "secondary": [GRANUSCORE_ALIGNED, SECONDARY_MODEL]}, "coverage": {"total_scores": 160, "rows_per_judge": 80, "rows_per_corpus_per_judge": 40}, "claim_boundary": record["analysis_extension"]["claim_boundary"], "environment": {"python": platform.python_version(), "numpy": np.__version__, "requests": requests.__version__}, "raw": {}, "outputs": {}}
    for judge in record["judges"]:
        raw = Path(record["outputs"]["raw_directory"]) / judge["judge_id"] / "raw_responses.jsonl"
        smoke = raw.parent / "determinism_smoke.json"
        judge_metadata = raw.parent / "run_metadata.json"
        metadata["raw"][judge["judge_id"]] = {"raw_response_sha256": sha256_file(raw), "determinism_smoke_sha256": sha256_file(smoke), "judge_run_metadata_sha256": sha256_file(judge_metadata)}
    for name, rows in outputs.items():
        metadata["outputs"][name] = {"rows": len(rows), "sha256": sha256_file(compact / name)}
    metadata["outputs"]["README.md"] = {"sha256": sha256_file(readme_path)}
    (compact / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"outputs": outputs, "metadata": metadata}
