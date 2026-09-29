"""Focused QE-A generator/gate/analysis helper tests."""
from __future__ import annotations

import json
from pathlib import Path

from src.analysis.qwen_edit_source import (
    EditCase,
    build_generation_request,
    derive_attempt_seed,
    gate_edit,
    load_cases,
    parse_generation_response,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_qwen_edit_source_v1.json"


def _record() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _case(edit_type: str, original: str = "Restart the service.") -> EditCase:
    return EditCase(
        case_id="a" * 64,
        corpus_id="ansible_docs",
        source_position=1,
        source_sent_id="source-1",
        edit_type=edit_type,
        sentence_original=original,
        sentence_author_edited="Hidden author text.",
        original_sha256="b" * 64,
    )


def test_generation_request_is_direction_matched_and_blind() -> None:
    record = _record()
    case = _case("add_specific")
    request = build_generation_request(record, case, 1)
    rendered = json.dumps(request)
    assert case.sentence_original in rendered
    assert case.sentence_author_edited not in rendered
    assert case.corpus_id not in rendered
    assert case.source_sent_id not in rendered
    assert "speciteller" not in rendered.casefold()
    assert request["think"] is False
    assert request["options"]["seed"] == derive_attempt_seed(20260810, case.case_id, 1)
    assert request["options"]["seed"] != derive_attempt_seed(20260810, case.case_id, 2)


def test_response_parser_accepts_only_exact_string_field() -> None:
    good = {"model": "qwen3:14b", "message": {"content": '{"edited_sentence":"Restart nginx 1.24 on host web-03."}'}}
    edited, metadata = parse_generation_response(good, "qwen3:14b")
    assert edited.startswith("Restart nginx")
    assert len(metadata["response_content_sha256"]) == 64
    bad = [
        {"model": "other", "message": {"content": '{"edited_sentence":"x"}'}},
        {"model": "qwen3:14b", "message": {"content": '{"edited_sentence":"x","why":"y"}'}},
        {"model": "qwen3:14b", "message": {"content": '{"edited_sentence":4}'}},
    ]
    for payload in bad:
        try:
            parse_generation_response(payload, "qwen3:14b")
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError(f"invalid response accepted: {payload}")


def test_add_specific_gate_passes_directional_fixture_and_rejects_negation_change() -> None:
    record = _record()
    original = "Restart the service after deployment."
    edited = "Restart nginx 1.24 service on web-03 after deployment."
    reasons, metrics = gate_edit(record, original, edited, "add_specific")
    assert reasons == []
    assert metrics["edited_token_count"] > metrics["original_token_count"]
    reasons, _ = gate_edit(record, "Do not restart the service.", "Restart the nginx 1.24 service on host web-03.", "add_specific")
    assert "polarity_changed" in reasons


def test_de_specify_gate_passes_directional_fixture_and_rejects_added_marker() -> None:
    record = _record()
    original = "Restart nginx 1.24 with --force on host web-03 after deployment."
    edited = "Restart the service after deployment."
    reasons, metrics = gate_edit(record, original, edited, "de_specify")
    assert reasons == []
    assert metrics["edited_concrete_marker_count"] < metrics["original_concrete_marker_count"]
    reasons, _ = gate_edit(record, "Restart the service after deployment.", "Restart nginx 1.24 on host web-03 after deployment.", "de_specify")
    assert "concrete_marker_count_increased" in reasons


def test_neutral_gate_requires_concrete_marker_multiset() -> None:
    record = _record()
    original = "Restart nginx 1.24 with --force after deployment."
    passing = "After deployment, restart nginx 1.24 using --force."
    reasons, _ = gate_edit(record, original, passing, "irrelevant_rewrite")
    assert reasons == []
    failing = "After deployment, restart the service."
    reasons, _ = gate_edit(record, original, failing, "irrelevant_rewrite")
    assert "neutral_concrete_markers_changed" in reasons


def test_real_cases_load_in_frozen_order_without_duplicates() -> None:
    cases = load_cases(_record())
    assert len(cases) == 60
    assert [case.corpus_id for case in cases[:30]] == ["ansible_docs"] * 30
    assert [case.corpus_id for case in cases[30:]] == ["github_docs"] * 30
    assert len({case.case_id for case in cases}) == 60
