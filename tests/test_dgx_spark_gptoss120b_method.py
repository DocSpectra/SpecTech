"""Outcome-blind method and minimum-input contract for the Spark packet."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/round2_human_first_reranking_v1.json"
GENERATION_DELTA = ROOT / "configs/round2_dgx_gptoss120b_generation_delta_v1.json"
RUBRIC_DELTA = ROOT / "configs/round2_dgx_gptoss120b_rubric_delta_v1.json"
FREEZE_RECORD = ROOT / "configs/round2_dgx_gptoss120b_freeze_record.json"
QWEN_RUBRIC = ROOT / "configs/round2_qwen_rubric_v1.json"
LOCAL_RUBRIC = ROOT / "configs/round2_local_rubric_replication_v1.json"
EXPORTER = ROOT / "deploy/dgx_spark_gptoss120b/export_inputs.py"
SCHEMAS = ROOT / "schemas/dgx_gptoss120b"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _overlay(base: dict, overrides: dict[str, object]) -> dict:
    result = json.loads(json.dumps(base))
    for pointer, value in overrides.items():
        current = result
        parts = [part.replace("~1", "/").replace("~0", "~") for part in pointer.split("/")[1:]]
        for part in parts[:-1]:
            current = current[part]
        current[parts[-1]] = value
    return result


def _leaf_differences(left: object, right: object, prefix: str = "") -> set[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        keys = set(left) | set(right)
        return {
            item
            for key in keys
            for item in _leaf_differences(left.get(key), right.get(key), f"{prefix}/{key}")
        }
    return set() if left == right else {prefix}


def _exporter_module():
    spec = importlib.util.spec_from_file_location("dgx_export_inputs", EXPORTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generation_delta_is_exactly_bound_to_human_first_protocol() -> None:
    delta = _load(GENERATION_DELTA)
    base = _load(BASE)
    assert _sha(BASE) == delta["base_protocol"]["sha256"]
    assert delta["base_protocol"]["ordered_case_sha256"] == base["inputs"]["ordered_case_sha256"]
    assert delta["base_protocol"]["required_source_cases"] == 60
    assert delta["base_protocol"]["required_candidate_slots"] == 180
    assert set(delta["authorized_overrides"]) == set(delta["allowed_override_pointers"])
    effective = _overlay(base, delta["authorized_overrides"])
    differences = _leaf_differences(base, effective)
    assert differences <= set(delta["allowed_override_pointers"])
    assert differences
    for section in ("inputs", "prompts", "integrity_inclusion", "automatic_proxy_audit", "scoring", "selection_policies", "blinding", "release_gate"):
        assert effective[section] == base[section]


def test_generation_identity_seed_resource_and_time_freezes() -> None:
    delta = _load(GENERATION_DELTA)
    runtime = delta["runtime_registration"]
    model = delta["model_registration"]
    assert runtime == {
        "version": "0.32.5",
        "platform": "linux/arm64",
        "archive": "ollama-linux-arm64.tar.zst",
        "archive_size_bytes": 1542011985,
        "archive_sha256": "aa7e06b5683ee66c4a3ec68ea7236db43b5a5d0821f0dfe2c5a215f4462bddf4",
        "release_url": "https://github.com/ollama/ollama/releases/download/v0.32.5/ollama-linux-arm64.tar.zst",
        "execution": "packet-owned extracted runtime; no curl-pipe-shell, sudo, service install, or automatic update",
    }
    assert model["ollama_manifest_sha256"] == "a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9"
    assert model["ollama_model_blob_sha256"] == "6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a"
    assert model["reasoning_effort"] == "low" and model["reasoning_text_policy"].startswith("never retain")
    assert delta["authorized_overrides"]["/runtime/master_seed"] == 2026081201
    assert delta["authorized_overrides"]["/runtime/seed_namespace"] == "dgx-spark-gptoss120b-candidate-v1"
    assert delta["resource_policy"]["minimum_memory_total_bytes"] == 128000000000
    assert delta["time_budget"]["hard_stop_seconds"] == 72000


def test_rubric_delta_is_exactly_bound_and_model_only() -> None:
    delta = _load(RUBRIC_DELTA)
    assert [_sha(QWEN_RUBRIC), _sha(LOCAL_RUBRIC)] == [item["sha256"] for item in delta["base_bindings"]]
    assert delta["enabled"] is True
    assert delta["priority"] == "secondary_after_primary_complete_and_time_gate"
    assert set(delta["model_runtime_delta"]) == {item.removeprefix("/") for item in delta["allowed_differences"]}
    assert "exact 80 rows and order" in delta["inherited_without_change"]
    assert delta["release_gate"].startswith("exactly 80 unique valid rows")


def test_minimum_input_export_has_exact_counts_order_and_no_private_columns(tmp_path: Path) -> None:
    exporter = _exporter_module()
    primary_path = tmp_path / "primary.csv"
    rubric_path = tmp_path / "rubric.csv"
    primary = exporter.export_primary(BASE, primary_path)
    rubric = exporter.export_rubric(QWEN_RUBRIC, rubric_path)
    assert primary["rows"] == 60 and rubric["rows"] == 80
    assert primary["ordered_case_sha256"] == "c46c1e7182a34a0f89dbb2826a6fcd2c0eec73f56842a5d484cf7675d95ed7ec"
    assert all(set(item) == {"cell_id", "direction"} for item in primary["cell_map"])
    with primary_path.open(encoding="utf-8", newline="") as handle:
        primary_rows = list(csv.DictReader(handle))
    with rubric_path.open(encoding="utf-8", newline="") as handle:
        rubric_rows = list(csv.DictReader(handle))
    assert [int(row["source_position"]) for row in primary_rows] == list(range(1, 31)) * 2
    assert [int(row["rubric_position"]) for row in rubric_rows] == list(range(1, 81))
    assert len({row["case_id"] for row in primary_rows}) == 60
    assert len({row["rubric_case_id"] for row in rubric_rows}) == 80
    forbidden = {"edited_sentence", "candidate", "score", "rating", "annotator", "note", "corpus_id"}
    assert forbidden.isdisjoint(primary_rows[0]) and forbidden.isdisjoint(rubric_rows[0])


def test_all_packet_schemas_are_strict_and_parseable() -> None:
    paths = sorted(SCHEMAS.glob("*.schema.json"))
    assert {path.name for path in paths} == {
        "capability_compatibility.schema.json", "capability_extension.schema.json",
        "experiment_registration.schema.json", "generation_attempt.schema.json",
        "operator_action.schema.json", "packet_manifest.schema.json",
        "return_manifest.schema.json", "run_receipt.schema.json",
        "spark_capability_report.schema.json",
    }
    for path in paths:
        schema = _load(path)
        assert schema["$schema"].endswith("2020-12/schema")
        assert schema["type"] == "object"
        assert schema["additionalProperties"] is False
        assert schema["required"]


def test_freeze_record_binds_pre_output_method_commit() -> None:
    record = _load(FREEZE_RECORD)
    assert record["method_freeze_commit"] == "d6642412988bf2b002d7283984c944df7a9a136e"
    assert record["preliminary_method_commit"] == "86ffd6fbcb35100445df99171627f48e625be974"
    assert record["method_frozen_before_any_gptoss120b_project_output"] is True
    assert record["gptoss120b_project_sentences_sent_to_model_before_freeze"] is False
    assert record["gptoss120b_project_outputs_inspected_before_freeze"] is False
    for binding in (record["bindings"][name] for name in ("generation_delta", "rubric_delta", "feasibility_spec", "packet_spec")):
        assert _sha(ROOT / binding["path"]) == binding["sha256"]
    for name, digest in record["bindings"]["schemas"].items():
        assert _sha(ROOT / "schemas/dgx_gptoss120b" / name) == digest
    for name, digest in record["bindings"]["invented_fixtures"].items():
        assert _sha(ROOT / "deploy/dgx_spark_gptoss120b/fixtures" / name) == digest
