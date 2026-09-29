"""Frozen matched analysis for the returned GPT-OSS-120B rubric matrix."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from src.analysis.local_rubric_replication import (
    GRANUSCORE_ALIGNED,
    QWEN_ID,
    _exact_quantized_ko_mean,
    _load_map,
    _read_jsonl,
    _write_csv,
    run_analysis as run_prior_analysis,
)
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


ROOT = Path(__file__).resolve().parents[2]
FREEZE = Path("configs/round2_dgx_gptoss120b_rubric_analysis_freeze_record.json")
PREANALYSIS_CORRECTION = Path("configs/round2_dgx_gptoss120b_rubric_preanalysis_correction_v1.json")
GPT120 = "gptoss_120b_zero_shot_rubric"
OTHER_JUDGES = (
    "qwen3_14b_zero_shot_rubric",
    "gemma4_12b_zero_shot_rubric",
    "gptoss_20b_zero_shot_rubric",
)
PRIMARY_COMPARATORS = PRIMARY_MODELS
SECONDARY_COMPARATORS = (SECONDARY_MODEL, GRANUSCORE_ALIGNED)
ALL_COMPARATORS = PRIMARY_COMPARATORS + SECONDARY_COMPARATORS


def _require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def _bytes_sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _case_order_sha(case_ids: list[str]) -> str:
    return hashlib.sha256(("\n".join(case_ids) + "\n").encode("utf-8")).hexdigest()


def load_frozen_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_dgx_gptoss120b_rubric_analysis_v1":
        raise ValueError("Unexpected GPT-OSS-120B rubric analysis schema")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("Analysis method was not frozen outcome-blind")
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "round2_dgx_gptoss120b_rubric_analysis_freeze_record_v1":
        raise ValueError("Unexpected GPT-OSS-120B rubric freeze-record schema")
    _require_hash(config_path, freeze["method_config"]["sha256"])
    if freeze["phase_a"]["disposition"] != "accepted_with_author_provenance_waiver":
        raise ValueError("Phase A acceptance disposition mismatch")
    entries = [
        record["authorization"]["provenance_config"],
        record["authorization"]["acceptance_summary"],
        record["pilot_inputs"]["pilot_manifest"],
        record["inherited_protocol"]["qwen_protocol"],
        record["inherited_protocol"]["dgx_rubric_delta"],
        record["human_targets"]["protocol"],
        *record["human_targets"]["labels"],
        *record["judge_inputs"],
        *record["predictor_inputs"]["speciteller"].values(),
        *record["predictor_inputs"]["ko_primary_runs"],
        record["predictor_inputs"]["granuscore_secondary"]["ansible_docs"],
        record["predictor_inputs"]["granuscore_secondary"]["github_docs"],
    ]
    for entry in entries:
        _require_hash(Path(entry["path"]), entry["sha256"])
    return record, freeze


def _read_archive_member(archive: Path, member_name: str) -> bytes:
    with tarfile.open(archive, "r:gz") as bundle:
        member = bundle.getmember(member_name)
        if not member.isfile():
            raise ValueError(f"Archive member is not a regular file: {member_name}")
        handle = bundle.extractfile(member)
        if handle is None:
            raise ValueError(f"Could not read archive member: {member_name}")
        return handle.read()


def _csv_bytes(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def load_returned_scores(record: dict[str, Any]) -> list[dict[str, Any]]:
    returned = record["returned_rubric"]
    archive = ROOT.parents[1] / returned["archive_workspace_path"]
    _require_hash(archive, returned["archive_sha256"])
    data = _read_archive_member(archive, returned["member_path"])
    if _bytes_sha(data) != returned["member_sha256"]:
        raise ValueError("Returned rubric member hash mismatch")
    rows = _csv_bytes(data)
    case_ids = [row["rubric_case_id"] for row in rows]
    if len(rows) != returned["expected_rows"] or len(set(case_ids)) != len(rows):
        raise ValueError("Returned rubric coverage or uniqueness mismatch")
    if _case_order_sha(case_ids) != returned["case_id_order_sha256"]:
        raise ValueError("Returned rubric case order mismatch")
    correction = json.loads(PREANALYSIS_CORRECTION.read_text(encoding="utf-8"))
    if correction["method_config_sha256"] != sha256_file(Path("configs/round2_dgx_gptoss120b_rubric_analysis_v1.json")):
        raise ValueError("Preanalysis correction method binding mismatch")
    if correction["freeze_record_sha256"] != sha256_file(FREEZE):
        raise ValueError("Preanalysis correction freeze binding mismatch")
    if correction["frozen_semantic_identity"] != returned["model_identity"]:
        raise ValueError("Preanalysis correction semantic identity mismatch")
    exact_remote_identity = correction["exact_remote_serialization"]
    result = []
    for row in rows:
        if row["run_id"] != returned["run_id"] or row["model_identity"] != exact_remote_identity:
            raise ValueError("Returned rubric run/model identity mismatch")
        value = int(row[returned["required_score_field"]])
        if str(value) != row[returned["required_score_field"]] or value not in returned["valid_scores"]:
            raise ValueError("Returned rubric score is not a valid frozen integer")
        result.append({"rubric_case_id": row["rubric_case_id"], "score": value})
    return result


def _load_judge_maps(record: dict[str, Any]) -> dict[str, dict[tuple[str, str], float]]:
    entries = {entry["judge_id"]: entry for entry in record["judge_inputs"]}
    qwen = _load_map(Path(entries[QWEN_ID]["path"]), "qwen_score", entries[QWEN_ID]["sha256"])
    result = {QWEN_ID: qwen}
    for judge_id in OTHER_JUDGES[1:]:
        entry = entries[judge_id]
        rows = _read_jsonl(Path(entry["path"]))
        if len(rows) != 80:
            raise ValueError(f"Incomplete frozen judge matrix: {judge_id}")
        mapping = {(row["corpus_id"], row["sent_id"]): float(row["score"]) for row in rows}
        if len(mapping) != 80:
            raise ValueError(f"Duplicate frozen judge keys: {judge_id}")
        result[judge_id] = mapping
    return result


def replay_prior_output_invariance(record: dict[str, Any]) -> dict[str, Any]:
    prior_config = json.loads(Path("configs/round2_local_rubric_replication_v1.json").read_text(encoding="utf-8"))
    replay_dir = Path(record["outputs"]["raw_import_directory"]) / "prior_three_judge_replay"
    replay_config = Path(record["outputs"]["raw_import_directory"]) / "prior_replay_config.json"
    prior_config["outputs"]["compact_directory"] = replay_dir.as_posix()
    replay_config.parent.mkdir(parents=True, exist_ok=True)
    replay_config.write_text(json.dumps(prior_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    run_prior_analysis(replay_config)
    metadata_path = Path(record["existing_output_invariance"]["metadata_path"])
    _require_hash(metadata_path, record["existing_output_invariance"]["metadata_sha256"])
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = []
    for name, expected in sorted(metadata["outputs"].items()):
        observed = sha256_file(replay_dir / name)
        rows.append({"path": name, "expected_sha256": expected["sha256"], "observed_sha256": observed, "byte_identical": observed == expected["sha256"]})
    if len(rows) != 13 or not all(row["byte_identical"] for row in rows):
        raise ValueError("Existing three-judge compact outputs are not byte-identical")
    return {"required_files": 13, "byte_identical_files": 13, "files": rows}


def _output_row(
    *, corpus_id: str, left_id: str, right_id: str, n: int,
    point: np.ndarray, boot: np.ndarray, index: dict[str, int],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    left, right = index[left_id], index[right_id]
    low, high, valid, degenerate = _interval(
        boot[:, left, right], int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"])
    )
    return {
        "corpus_id": corpus_id,
        "left_id": left_id,
        "right_id": right_id,
        "n": n,
        "spearman_rho": float(point[left, right]),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_valid": valid,
        "bootstrap_degenerate": degenerate,
    }


def _contrast_row(
    *, corpus_id: str, left_id: str, right_id: str, target_id: str, n: int,
    point: np.ndarray, boot: np.ndarray, index: dict[str, int],
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    left, right, target = index[left_id], index[right_id], index[target_id]
    delta = boot[:, left, target] - boot[:, right, target]
    low, high, valid, degenerate = _interval(
        delta, int(bootstrap["replicates"]), float(bootstrap["minimum_valid_fraction"])
    )
    return {
        "corpus_id": corpus_id,
        "target_id": target_id,
        "left_id": left_id,
        "right_id": right_id,
        "n": n,
        "left_rho": float(point[left, target]),
        "right_rho": float(point[right, target]),
        "delta_rho_left_minus_right": float(point[left, target] - point[right, target]),
        "ci_low": low,
        "ci_high": high,
        "bootstrap_valid": valid,
        "bootstrap_degenerate": degenerate,
    }


def run_analysis(config_path: Path) -> dict[str, Any]:
    record, freeze = load_frozen_config(config_path)
    prior_invariance = replay_prior_output_invariance(record)
    qwen_protocol = json.loads(Path(record["inherited_protocol"]["qwen_protocol"]["path"]).read_text(encoding="utf-8"))
    project_rows = load_project_rows(qwen_protocol)
    returned_rows = load_returned_scores(record)
    if len(project_rows) != len(returned_rows):
        raise ValueError("Project/returned rubric row count mismatch")
    expected_case_ids = [
        hashlib.sha256(f"{row['corpus_id']}\0{row['sent_id']}\0{row['pilot_position']}".encode("utf-8")).hexdigest()
        for row in project_rows
    ]
    if expected_case_ids != [row["rubric_case_id"] for row in returned_rows]:
        raise ValueError("Returned case IDs do not map bijectively to frozen pilot rows")
    gpt120_map = {
        (project["corpus_id"], project["sent_id"]): float(returned["score"])
        for project, returned in zip(project_rows, returned_rows, strict=True)
    }
    judge_maps = _load_judge_maps(record)
    human_protocol_path = Path(record["human_targets"]["protocol"]["path"])
    corpora, _ = _load_corpora(load_protocol(human_protocol_path), human_protocol_path)
    gran_maps: dict[tuple[str, str], float] = {}
    for corpus_id in record["pilot_inputs"]["corpus_order"]:
        entry = record["predictor_inputs"]["granuscore_secondary"][corpus_id]
        gran_maps.update(_load_map(Path(entry["path"]), "granuscore_percentile", entry["sha256"]))

    names = tuple(record["outputs"]["tables"])
    outputs: dict[str, list[dict[str, Any]]] = {name: [] for name in names}
    bootstrap = record["analysis"]["bootstrap"]
    all_score_ids = (GPT120,) + OTHER_JUDGES + ALL_COMPARATORS
    for corpus_id in record["pilot_inputs"]["corpus_order"]:
        corpus = corpora[corpus_id]
        keys = [(corpus_id, sent_id) for sent_id in corpus.sent_ids]
        scores = {
            **corpus.scores,
            SECONDARY_MODEL: _exact_quantized_ko_mean(dict(corpus.scores)),
            GRANUSCORE_ALIGNED: -np.asarray([gran_maps[key] for key in keys]),
            GPT120: np.asarray([gpt120_map[key] for key in keys]),
            **{judge_id: np.asarray([judge_maps[judge_id][key] for key in keys]) for judge_id in OTHER_JUDGES},
        }
        corpus = replace(corpus, scores=scores)
        columns = np.vstack([corpus.scores[name] for name in all_score_ids] + [corpus.labels[target] for target in HUMAN_TARGETS])
        point, boot = bootstrap_correlation_matrices(
            columns,
            replicates=int(bootstrap["replicates"]),
            seed=int(bootstrap["corpus_seeds"][corpus_id]),
        )
        index = {name: index for index, name in enumerate(all_score_ids + HUMAN_TARGETS)}
        n = len(corpus.sent_ids)
        outputs["coverage.csv"].append({
            "corpus_id": corpus_id, "judge_id": GPT120, "expected_rows": 40,
            "observed_rows": n, "coverage_rate": 1.0, "gate_passed": True,
        })
        values = corpus.scores[GPT120]
        for score in range(1, 6):
            count = int(np.sum(values == score))
            outputs["score_distribution.csv"].append({
                "corpus_id": corpus_id, "judge_id": GPT120, "score": score,
                "count": count, "proportion": count / n,
            })
        quartiles = np.quantile(values, [0.25, 0.5, 0.75], method="linear")
        outputs["ordinal_summary.csv"].append({
            "corpus_id": corpus_id, "judge_id": GPT120, "n": n,
            "minimum": float(np.min(values)), "q1_linear": float(quartiles[0]),
            "median_linear": float(quartiles[1]), "q3_linear": float(quartiles[2]),
            "maximum": float(np.max(values)), "unique_scores": len(set(values.tolist())),
        })
        for target in HUMAN_TARGETS:
            outputs["judge_human_agreement.csv"].append(_output_row(
                corpus_id=corpus_id, left_id=GPT120, right_id=target, n=n,
                point=point, boot=boot, index=index, bootstrap=bootstrap,
            ))
        for judge_id in OTHER_JUDGES:
            outputs["judge_pair_agreement.csv"].append(_output_row(
                corpus_id=corpus_id, left_id=GPT120, right_id=judge_id, n=n,
                point=point, boot=boot, index=index, bootstrap=bootstrap,
            ))
            for target in HUMAN_TARGETS:
                outputs["judge_pair_human_contrasts.csv"].append(_contrast_row(
                    corpus_id=corpus_id, left_id=GPT120, right_id=judge_id,
                    target_id=target, n=n, point=point, boot=boot, index=index,
                    bootstrap=bootstrap,
                ))
        for comparator in ALL_COMPARATORS:
            row = _output_row(
                corpus_id=corpus_id, left_id=GPT120, right_id=comparator, n=n,
                point=point, boot=boot, index=index, bootstrap=bootstrap,
            )
            row["analysis_role"] = "primary" if comparator in PRIMARY_COMPARATORS else "secondary"
            outputs["judge_comparator_agreement.csv"].append(row)
            for target in HUMAN_TARGETS:
                contrast = _contrast_row(
                    corpus_id=corpus_id, left_id=GPT120, right_id=comparator,
                    target_id=target, n=n, point=point, boot=boot, index=index,
                    bootstrap=bootstrap,
                )
                contrast["analysis_role"] = "primary" if comparator in PRIMARY_COMPARATORS else "secondary"
                outputs["judge_vs_comparator_human_contrasts.csv"].append(contrast)

    compact = Path(record["outputs"]["compact_directory"])
    compact.mkdir(parents=True, exist_ok=True)
    for name, rows in outputs.items():
        _write_csv(compact / name, rows)
    pooled = [row for row in outputs["judge_human_agreement.csv"] if row["right_id"] == "pooled_human_mean"]
    lines = [
        "# GPT-OSS-120B matched zero-shot rubric analysis",
        "",
        "The returned judge matrix covers the same 40 Ansible and 40 GitHub pilot sentences under the frozen 1--5 rubric.",
        "",
        "## Pooled-human correspondence",
        "",
        "| Corpus | Spearman rho [95% paired sentence-bootstrap CI] |",
        "| --- | ---: |",
    ]
    for row in pooled:
        lines.append(f"| {row['corpus_id']} | {row['spearman_rho']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] |")
    lines.extend([
        "", "## Interpretation boundary", "", record["reporting"]["claim_boundary"],
        "The run is used under the reviewed accepted_with_author_provenance_waiver disposition; missing remote attempt, smoke, post-load identity/residency, state, and receipt-bound rubric evidence remains an audit-completeness limitation.",
        "All frozen estimates, including null, adverse, corpus-dependent, and annotator-dependent patterns, remain in the adjacent aggregate tables.", "",
    ])
    (compact / "README.md").write_text("\n".join(lines), encoding="utf-8")
    (compact / "prior_output_invariance.json").write_text(json.dumps(prior_invariance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "round2_dgx_gptoss120b_rubric_results_v1",
        "protocol": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "freeze_record": {"path": FREEZE.as_posix(), "sha256": sha256_file(FREEZE), "method_freeze_commit": freeze["method_freeze_commit"]},
        "provenance_disposition": record["authorization"]["provenance_disposition"],
        "returned_rubric": {
            "archive_sha256": record["returned_rubric"]["archive_sha256"],
            "member_sha256": record["returned_rubric"]["member_sha256"],
            "case_id_order_sha256": record["returned_rubric"]["case_id_order_sha256"],
        },
        "coverage": {"total_rows": 80, "ansible_docs": 40, "github_docs": 40},
        "bootstrap": bootstrap,
        "comparison_status": {"other_judges": list(OTHER_JUDGES), "primary_predictors": list(PRIMARY_COMPARATORS), "secondary": list(SECONDARY_COMPARATORS)},
        "prior_three_judge_output_invariance": {"required_files": 13, "byte_identical_files": 13},
        "claim_boundary": record["reporting"]["claim_boundary"],
        "outputs": {},
    }
    artifact_names = list(outputs) + ["README.md", "prior_output_invariance.json"]
    for name in artifact_names:
        metadata["outputs"][name] = {"sha256": sha256_file(compact / name)}
        if name in outputs:
            metadata["outputs"][name]["rows"] = len(outputs[name])
    (compact / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "round2_dgx_gptoss120b_rubric_manifest_v1",
        "protocol_sha256": sha256_file(config_path),
        "freeze_record_sha256": sha256_file(FREEZE),
        "files": [
            {"path": name, "sha256": sha256_file(compact / name), "size_bytes": (compact / name).stat().st_size}
            for name in sorted(artifact_names + ["run_metadata.json"])
        ],
    }
    (compact / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"outputs": outputs, "metadata": metadata, "manifest": manifest}


__all__ = ["load_frozen_config", "load_returned_scores", "replay_prior_output_invariance", "run_analysis"]
