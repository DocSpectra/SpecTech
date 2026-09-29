"""Outcome-blind equality contract for the GPT-OSS editor comparison."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DELTA = ROOT / "configs" / "round2_gptoss_controlled_editor_v1.json"
ATTEMPT_SCHEMA = ROOT / "schemas" / "round2_gptoss_editor_attempt_v1.schema.json"
REVIEW_SCHEMA = ROOT / "schemas" / "round2_gptoss_blinded_review_v1.schema.json"
FREEZE_RECORD = ROOT / "configs" / "round2_gptoss_controlled_editor_freeze_record.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _delta() -> dict:
    return json.loads(DELTA.read_text(encoding="utf-8"))


def _set_pointer(record: dict, pointer: str, value: object) -> None:
    parts = pointer.strip("/").split("/")
    node = record
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def _merged() -> tuple[dict, dict, dict]:
    delta = _delta()
    base_path = ROOT / delta["scientific_protocol_base"]["path"]
    assert _sha(base_path) == delta["scientific_protocol_base"]["sha256"]
    base = json.loads(base_path.read_text(encoding="utf-8"))
    merged = copy.deepcopy(base)
    overrides = delta["authorized_model_runtime_overrides"]
    assert set(overrides) == set(delta["allowed_override_pointers"])
    for pointer, value in overrides.items():
        _set_pointer(merged, pointer, value)
    return base, merged, delta


def _diff(left: object, right: object, pointer: str = "") -> set[str]:
    if type(left) is not type(right):
        return {pointer or "/"}
    if isinstance(left, dict):
        result: set[str] = set()
        for key in set(left) | set(right):
            child = f"{pointer}/{key}"
            if key not in left or key not in right:
                result.add(child)
            else:
                result.update(_diff(left[key], right[key], child))
        return result
    if isinstance(left, list):
        return set() if left == right else {pointer}
    return set() if left == right else {pointer}


def test_only_authorized_model_runtime_and_namespace_values_differ() -> None:
    base, merged, delta = _merged()
    observed = _diff(base, merged)
    allowed = set(delta["allowed_override_pointers"])
    assert all(any(path == pointer or path.startswith(pointer + "/") for pointer in allowed) for path in observed)


def test_scientific_generation_and_human_controls_are_exactly_identical() -> None:
    base, merged, _ = _merged()
    for key in ("authorization", "isolation", "inputs", "prompts", "automated_gates", "generation_gate", "human_gate", "stop_conditions"):
        assert merged[key] == base[key]
    for key in ("api_url", "stream", "keep_alive", "fresh_conversation_per_attempt", "maximum_attempts_per_case", "attempt_policy", "master_seed", "seed_derivation", "options", "response_schema"):
        assert merged["runtime"][key] == base["runtime"][key]
    for key in ("review_id_seed", "side_assignment_seed", "side_assignment", "row_order_seed", "row_order", "packet_fields", "packet_forbidden_fields", "instructions", "reversible_integrity"):
        assert merged["blinded_review"][key] == base["blinded_review"][key]


def test_exact_gptoss_identity_and_minimum_reasoning_interface_are_frozen() -> None:
    _, merged, delta = _merged()
    runtime = merged["runtime"]
    assert runtime["model"] == "gpt-oss:20b"
    assert runtime["model_tag_digest"] == "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7"
    assert runtime["model_blob_sha256"] == "e7b273f9636059a689e3ddcab3716e4f65abe0143ac978e46673ad0e52d09efb"
    assert runtime["thinking"] == "low"
    assert runtime["options"] == {
        "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        "repeat_penalty": 1.0, "num_ctx": 4096, "num_predict": 256,
    }
    assert delta["interface_controls"]["reasoning_text_policy"].startswith("Never print")
    assert delta["provenance"]["hardware"]["smoke_resident_vram_bytes"] <= 16376 * 1024 * 1024


def test_schemas_are_collision_free_and_review_questions_unchanged() -> None:
    _, merged, _ = _merged()
    attempt = json.loads(ATTEMPT_SCHEMA.read_text(encoding="utf-8"))
    review = json.loads(REVIEW_SCHEMA.read_text(encoding="utf-8"))
    assert attempt["properties"]["model"]["const"] == "gpt-oss:20b"
    assert attempt["properties"]["attempt_index"]["maximum"] == 3
    assert {"thinking_char_count", "thinking_sha256"}.issubset(attempt["required"])
    assert review["properties"]["review_id"]["pattern"].startswith("^GOE-")
    assert set(review["properties"]) == set(merged["blinded_review"]["packet_fields"])
    assert review["properties"]["more_specific"]["enum"] == ["A", "B", "tie", "uncertain"]


def test_paths_are_portable_and_gemma_outputs_are_not_reused() -> None:
    _, merged, _ = _merged()
    assert merged["outputs"]["raw_directory"] == "outputs/round2/gptoss_controlled_editor"
    assert merged["outputs"]["compact_directory"] == "analysis/round2_gptoss_controlled_editor"
    text = DELTA.read_text(encoding="utf-8").casefold()
    assert "outputs/round2/gemma_controlled_editor" not in text
    assert "c:\\users\\" not in text and "c:/users/" not in text


def test_freeze_record_binds_pre_generation_commit_and_smoke() -> None:
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    assert freeze["method_freeze_commit"] == "ca7ab7eeb71e1d4fb9ce9447a9a5ec7e27246491"
    assert freeze["config_sha256"] == _sha(DELTA)
    assert freeze["attempt_schema_sha256"] == _sha(ATTEMPT_SCHEMA)
    assert freeze["review_schema_sha256"] == _sha(REVIEW_SCHEMA)
    assert freeze["method_frozen_before_any_project_gptoss_generation"] is True
    assert freeze["project_gptoss_text_or_outcome_inspected_before_freeze"] is False
    assert freeze["gemma_output_text_or_case_outcome_used_for_freeze"] is False
    smoke = freeze["non_project_smoke"]
    assert smoke["project_inputs_used"] is False
    assert smoke["structured_json_passed"] == 3
    assert smoke["thinking_false_empty_final_content"] == 2
    assert smoke["reasoning_text_printed_or_persisted"] is False
