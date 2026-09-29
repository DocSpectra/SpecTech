"""Focused implementation tests for the frozen Gemma editor workflow."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from src.analysis.gemma_controlled_editor import (
    GemmaCase,
    _candidate_sides,
    _review_id,
    build_generation_request,
    derive_attempt_seed,
    gate_edit,
    load_cases,
    parse_generation_response,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_gemma_controlled_editor_v1.json"


def _record() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _case(edit_type: str, original: str = "Restart the service after deployment.") -> GemmaCase:
    return GemmaCase(
        case_id="a" * 64, corpus_id="ansible_docs", source_position=1,
        source_sent_id="fixture-1", edit_type=edit_type, sentence_original=original,
        original_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
    )


def test_request_is_explicit_seeded_thinking_disabled_and_blind() -> None:
    record = _record(); case = _case("add_specific")
    request = build_generation_request(record, case, 1)
    rendered = json.dumps(request)
    assert request["model"] == "gemma4:12b"
    assert request["think"] is False and request["stream"] is False
    assert case.sentence_original in rendered
    assert case.corpus_id not in rendered and case.source_sent_id not in rendered
    assert request["options"]["seed"] == derive_attempt_seed(2026081101, case.case_id, 1)
    assert request["options"]["seed"] != derive_attempt_seed(2026081101, case.case_id, 2)


def test_response_parser_rejects_wrong_model_extra_field_and_thinking() -> None:
    good = {"model": "gemma4:12b", "message": {"content": '{"edited_sentence":"Restart nginx after deployment."}', "thinking": ""}}
    edited, metadata = parse_generation_response(good, "gemma4:12b")
    assert edited.startswith("Restart nginx") and len(metadata["response_content_sha256"]) == 64
    bad = [
        {"model": "other", "message": {"content": '{"edited_sentence":"x"}'}},
        {"model": "gemma4:12b", "message": {"content": '{"edited_sentence":"x","why":"y"}'}},
        {"model": "gemma4:12b", "message": {"content": '{"edited_sentence":"x"}', "thinking": "hidden"}},
    ]
    for payload in bad:
        try:
            parse_generation_response(payload, "gemma4:12b")
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError(f"invalid response accepted: {payload}")


def test_add_de_and_neutral_gate_fixtures() -> None:
    record = _record()
    reasons, _ = gate_edit(record, "Restart the service after deployment.", "Restart the nginx 1.24 service after deployment.", "add_specific")
    assert reasons == []
    reasons, _ = gate_edit(record, "Restart nginx 1.24 after deployment.", "Restart the service after deployment.", "de_specify")
    assert reasons == []
    reasons, _ = gate_edit(record, "Restart nginx 1.24 after deployment.", "After deployment, restart nginx 1.24.", "irrelevant_rewrite")
    assert reasons == []


def test_gate_rejects_modality_polarity_multisentence_and_script_changes() -> None:
    record = _record()
    reasons, _ = gate_edit(record, "You can restart it.", "You should restart it now.", "add_specific")
    assert "modality_changed" in reasons
    reasons, _ = gate_edit(record, "Do not restart it.", "Restart it now.", "add_specific")
    assert "polarity_changed" in reasons
    reasons, _ = gate_edit(record, "Restart it.", "Restart it now. Confirm success.", "add_specific")
    assert "multiple_sentences" in reasons
    reasons, _ = gate_edit(record, "Restart it.", "Restart it for 测试.", "add_specific")
    assert "new_non_latin_script" in reasons


def test_real_cases_and_blinded_identifiers_are_complete_and_balanced() -> None:
    record = _record(); cases = load_cases(record)
    assert len(cases) == 60 and len({case.case_id for case in cases}) == 60
    assert len({_review_id(record, case.case_id) for case in cases}) == 60
    sides = _candidate_sides(record, cases)
    assert list(sides.values()).count("A") == list(sides.values()).count("B") == 30


def test_review_packet_fields_are_blind_and_response_values_are_separate() -> None:
    record = _record()
    fields = set(record["blinded_review"]["packet_fields"])
    forbidden = set(record["blinded_review"]["packet_forbidden_fields"])
    assert not fields.intersection(forbidden)
    assert {"factual_correctness_a", "factual_correctness_b", "semantic_preservation", "grammar_naturalness_a", "grammar_naturalness_b", "more_specific", "confidence", "notes"}.issubset(fields)
