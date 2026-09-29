"""Focused implementation tests for the frozen GPT-OSS editor workflow."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.analysis.gemma_controlled_editor import GemmaCase, derive_attempt_seed, gate_edit, load_cases
from src.analysis.gptoss_controlled_editor import (
    _candidate_sides,
    _review_id,
    build_generation_request,
    load_protocol,
    parse_generation_response,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_gptoss_controlled_editor_v1.json"


def _record() -> dict:
    return load_protocol(CONFIG)[0]


def _case(edit_type: str, original: str = "Restart the service after deployment.") -> GemmaCase:
    return GemmaCase(
        case_id="a" * 64, corpus_id="ansible_docs", source_position=1,
        source_sent_id="fixture-1", edit_type=edit_type, sentence_original=original,
        original_sha256=hashlib.sha256(original.encode("utf-8")).hexdigest(),
    )


def test_request_is_identically_seeded_low_reasoning_and_blind() -> None:
    record = _record(); case = _case("add_specific")
    request = build_generation_request(record, case, 1)
    rendered = json.dumps(request)
    assert request["model"] == "gpt-oss:20b"
    assert request["think"] == "low" and request["stream"] is False
    assert case.sentence_original in rendered
    assert case.corpus_id not in rendered and case.source_sent_id not in rendered
    assert request["options"]["seed"] == derive_attempt_seed(2026081101, case.case_id, 1)


def test_response_parser_hashes_but_does_not_return_reasoning() -> None:
    good = {
        "model": "gpt-oss:20b",
        "message": {"content": '{"edited_sentence":"Restart nginx after deployment."}', "thinking": "invented trace"},
    }
    edited, metadata = parse_generation_response(good, "gpt-oss:20b")
    assert edited.startswith("Restart nginx")
    assert metadata["thinking_char_count"] == len("invented trace")
    assert metadata["thinking_sha256"] == hashlib.sha256(b"invented trace").hexdigest()
    assert "thinking" not in metadata
    for payload in (
        {"model": "other", "message": good["message"]},
        {"model": "gpt-oss:20b", "message": {"content": good["message"]["content"], "thinking": ""}},
        {"model": "gpt-oss:20b", "message": {"content": '{"edited_sentence":"x","why":"y"}', "thinking": "x"}},
    ):
        try:
            parse_generation_response(payload, "gpt-oss:20b")
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError("invalid GPT-OSS response accepted")


def test_unchanged_add_de_and_neutral_gate_fixtures() -> None:
    record = _record()
    reasons, _ = gate_edit(record, "Restart the service after deployment.", "Restart the nginx 1.24 service after deployment.", "add_specific")
    assert reasons == []
    reasons, _ = gate_edit(record, "Restart nginx 1.24 after deployment.", "Restart the service after deployment.", "de_specify")
    assert reasons == []
    reasons, _ = gate_edit(record, "Restart nginx 1.24 after deployment.", "After deployment, restart nginx 1.24.", "irrelevant_rewrite")
    assert reasons == []


def test_real_cases_review_ids_and_sides_are_complete_and_balanced() -> None:
    record = _record(); cases = load_cases(record)
    assert len(cases) == len({case.case_id for case in cases}) == 60
    identifiers = {_review_id(record, case.case_id) for case in cases}
    assert len(identifiers) == 60 and all(value.startswith("GOE-") for value in identifiers)
    sides = _candidate_sides(record, cases)
    assert list(sides.values()).count("A") == list(sides.values()).count("B") == 30


def test_output_namespaces_are_collision_free() -> None:
    record = _record()
    assert record["outputs"]["raw_directory"] == "outputs/round2/gptoss_controlled_editor"
    assert record["outputs"]["compact_directory"] == "analysis/round2_gptoss_controlled_editor"
    assert record["blinded_review"]["review_schema"].endswith("gptoss_blinded_review_v1.schema.json")
