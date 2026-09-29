"""Outcome-blind tests for matched local rubric replication."""
from __future__ import annotations

import json
from pathlib import Path

from src.analysis.local_rubric_replication import (
    build_request,
    load_freeze_binding,
    parse_response,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_local_rubric_replication_v1.json"
QWEN = ROOT / "configs" / "round2_qwen_rubric_v1.json"
FREEZE = ROOT / "configs" / "round2_local_rubric_replication_freeze_record.json"


def _records():
    return json.loads(CONFIG.read_text(encoding="utf-8")), json.loads(QWEN.read_text(encoding="utf-8"))


def test_exact_qwen_scientific_contract_is_inherited() -> None:
    record, qwen = _records()
    assert record["outcome_blind_freeze"] is True
    assert record["authorization"]["project_outcomes_inspected_before_freeze"] is False
    assert record["qwen_protocol"]["inherit_exactly"] == ["pilot", "rubric", "prompt", "determinism_smoke", "scoring_contract", "analysis"]
    assert qwen["pilot"]["required_rows_total"] == 80
    assert qwen["qwen"]["attempts_per_sentence"] == 1


def test_only_frozen_interface_differences_exist() -> None:
    record, qwen = _records()
    judges = {judge["model"]: judge for judge in record["judges"]}
    gemma = build_request(record, qwen, judges["gemma4:12b"], "Fixture.")
    gptoss = build_request(record, qwen, judges["gpt-oss:20b"], "Fixture.")
    assert gemma["messages"] == gptoss["messages"]
    assert gemma["format"] == gptoss["format"] == qwen["qwen"]["response_schema"]
    assert gemma["think"] is False and gemma["options"]["num_predict"] == 16
    assert gptoss["think"] == "low" and gptoss["options"]["num_predict"] == 128
    for key in ("temperature", "top_k", "top_p", "repeat_penalty", "seed", "num_ctx"):
        assert gemma["options"][key] == gptoss["options"][key] == qwen["qwen"]["options"][key]


def test_blindness_comparators_and_privacy_are_explicit() -> None:
    record, qwen = _records()
    assert "human_label" not in qwen["prompt"]["system"].lower()
    assert set(record["analysis_extension"]["comparators"]) == {
        "qwen3_14b_zero_shot_rubric", "speciteller_frozen_round1",
        "ko_official_release_run01", "ko_official_release_run02",
        "ko_official_release_run03", "granuscore_direction_aligned_secondary",
    }
    assert "not gold-standard accuracy" in record["analysis_extension"]["claim_boundary"]


def test_parser_is_strict_and_does_not_return_reasoning_text() -> None:
    score, metadata = parse_response({"model": "gpt-oss:20b", "message": {"content": "{\"score\":4}", "thinking": "private"}}, "gpt-oss:20b")
    assert score == 4
    assert "thinking" not in metadata
    assert metadata["reasoning_char_count"] == 7
    for content in ('{"score":0}', '{"score":"3"}', '{"score":3,"why":"x"}'):
        try:
            parse_response({"model": "gemma4:12b", "message": {"content": content}}, "gemma4:12b")
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError(f"accepted invalid content {content}")


def test_freeze_record_binds_pre_outcome_method_commit() -> None:
    freeze = load_freeze_binding(FREEZE.relative_to(ROOT))
    assert freeze["method_frozen_before_any_project_judge_output"] is True
    assert freeze["project_judge_calls_before_freeze"] == 0
    assert freeze["project_outcomes_inspected_before_freeze"] is False
    assert freeze["chronology_audit"]["pre_freeze_project_sentence_calls_after_base"] == 0
    assert freeze["chronology_audit"]["pre_freeze_non_project_smoke_calls_after_base"] == 1
    assert len(freeze["method_freeze_commit"]) == 40
    assert len(freeze["freeze_binding_commit"]) == 40


def test_secondary_ko_mean_and_single_run_limit_are_predeclared() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    addendum = freeze["analysis_addendum_frozen_before_project_calls"]
    assert "secondary" in addendum["ko_official_release_run_mean"]
    assert "one frozen call" in addendum["project_run_variability"]


def test_frozen_model_and_human_ids_are_tuple_compatible() -> None:
    from src.analysis.pilot_model_human import HUMAN_TARGETS

    model_ids = ("gemma", "gptoss", "qwen")
    assert model_ids + HUMAN_TARGETS == (
        "gemma", "gptoss", "qwen", "ann_a", "ann_b", "ann_c", "pooled_human_mean"
    )
