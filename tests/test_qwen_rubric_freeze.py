"""Outcome-blind contract tests for the Qwen rubric experiment."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.analysis.pilot_model_human import HUMAN_TARGETS, PRIMARY_MODELS, PilotCorpus
from src.analysis.qwen_rubric import (
    QWEN_ID,
    _analyze_corpus,
    build_request,
    load_project_rows,
    parse_score_response,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_qwen_rubric_v1.json"
MANIFEST = ROOT / "configs" / "round2_pilot_sent_ids_v1.csv"
FREEZE_RECORD = ROOT / "configs" / "round2_qwen_rubric_freeze_record.json"


def _record() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_protocol_is_outcome_blind_and_exactly_scoped() -> None:
    record = _record()
    assert record["outcome_blind_freeze"] is True
    assert record["authorization"]["qwen_project_outcomes_inspected_before_freeze"] is False
    assert record["pilot"]["required_rows_total"] == 80
    assert record["pilot"]["required_rows_per_corpus"] == 40
    assert record["pilot"]["corpus_order"] == ["ansible_docs", "github_docs"]
    assert record["qwen"]["attempts_per_sentence"] == 1
    assert record["qwen"]["fresh_conversation_per_sentence"] is True


def test_model_and_decoding_are_frozen() -> None:
    record = _record()
    qwen = record["qwen"]
    assert qwen["model"] == "qwen3:14b"
    assert qwen["model_list_id"] == "bdbd181c33f2"
    assert len(qwen["backing_blob_sha256"]) == 64
    assert qwen["thinking"] is False
    assert qwen["stream"] is False
    assert qwen["options"] == {
        "temperature": 0.0,
        "top_k": 1,
        "top_p": 1.0,
        "repeat_penalty": 1.0,
        "seed": 20260809,
        "num_ctx": 4096,
        "num_predict": 16,
    }


def test_prompt_is_zero_shot_and_blind() -> None:
    record = _record()
    prompt = record["prompt"]["system"].lower()
    for term in ("human_label", "speciteller", "ko run", "edge_low", "edge_high"):
        assert term not in prompt
    request = build_request(record, "A held-out fixture sentence.")
    assert request["think"] is False
    assert request["format"]["additionalProperties"] is False
    assert request["messages"][-1]["content"].endswith("A held-out fixture sentence.")


def test_response_parser_accepts_only_one_integer_score() -> None:
    good = {"model": "qwen3:14b", "message": {"content": '{"score":4}'}}
    score, metadata = parse_score_response(good, "qwen3:14b")
    assert score == 4
    assert len(metadata["response_content_sha256"]) == 64

    bad_payloads = [
        {"model": "qwen3:14b", "message": {"content": '{"score":0}'}},
        {"model": "qwen3:14b", "message": {"content": '{"score":3,"why":"x"}'}},
        {"model": "qwen3:14b", "message": {"content": '{"score":"3"}'}},
        {"model": "other", "message": {"content": '{"score":3}'}},
    ]
    for payload in bad_payloads:
        try:
            parse_score_response(payload, "qwen3:14b")
        except (ValueError, json.JSONDecodeError):
            pass
        else:
            raise AssertionError(f"Invalid payload accepted: {payload}")


def test_manifest_has_exact_frozen_shape() -> None:
    lines = MANIFEST.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 81
    assert lines[0] == "corpus_id,pilot_position,bucket,sent_id"


def test_canonical_project_rows_join_exactly() -> None:
    rows = load_project_rows(_record())
    assert len(rows) == 80
    assert [row["corpus_id"] for row in rows[:40]] == ["ansible_docs"] * 40
    assert [row["corpus_id"] for row in rows[40:]] == ["github_docs"] * 40
    assert [row["pilot_position"] for row in rows[:40]] == list(range(1, 41))
    assert all(row["sent_text"].strip() for row in rows)


def test_claim_and_privacy_boundaries_are_explicit() -> None:
    record = _record()
    assert "not gold-standard accuracy" in record["analysis"]["claim_boundary"]
    privacy = record["scoring_contract"]["privacy"]
    assert "no sentence text" in privacy
    assert "human labels" in privacy
    assert record["analysis"]["bootstrap"]["replicates"] == 10000
    assert record["analysis"]["bootstrap"]["master_seed"] == 20260809


def test_freeze_record_preserves_pre_score_and_mechanical_fix_boundary() -> None:
    record = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    assert record["method_frozen_before_any_qwen_project_output"] is True
    assert record["qwen_scores_or_scientific_summaries_inspected_before_mechanical_fix"] is False
    assert len(record["method_freeze_commit"]) == 40
    assert len(record["mechanical_analysis_fix_commit"]) == 40
    assert "no prompt" in record["fix_scope"]


def test_analysis_shape_accepts_frozen_tuple_constants() -> None:
    labels = {
        "ann_a": np.asarray([1, 1, 2, 3, 4, 5, 5, 4], dtype=float),
        "ann_b": np.asarray([1, 2, 2, 3, 3, 4, 5, 5], dtype=float),
        "ann_c": np.asarray([2, 1, 2, 4, 3, 5, 4, 5], dtype=float),
    }
    labels["pooled_human_mean"] = np.mean(
        np.vstack([labels[target] for target in HUMAN_TARGETS[:3]]), axis=0
    )
    scores = {
        QWEN_ID: np.asarray([1, 1, 2, 3, 4, 5, 5, 4], dtype=float),
        PRIMARY_MODELS[0]: np.asarray([0.0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 0.7]),
        PRIMARY_MODELS[1]: np.asarray([0.2, 0.1, 0.3, 0.5, 0.4, 0.7, 0.9, 0.8]),
        PRIMARY_MODELS[2]: np.asarray([0.8, 0.9, 0.7, 0.5, 0.6, 0.3, 0.1, 0.2]),
        PRIMARY_MODELS[3]: np.asarray([0.1, 0.2, 0.3, 0.3, 0.5, 0.8, 0.8, 0.7]),
    }
    corpus = PilotCorpus(
        corpus_id="fixture",
        sent_ids=tuple(f"s{i}" for i in range(8)),
        buckets=tuple("average" for _ in range(8)),
        labels=labels,
        scores=scores,
    )
    result = _analyze_corpus(
        corpus, replicates=100, seed=20260809, minimum_valid_fraction=0.90
    )
    assert len(result["qwen_human_agreement.csv"]) == 4
    assert len(result["qwen_vs_model_human_contrasts.csv"]) == 16
    assert len(result["qwen_model_agreement.csv"]) == 4
    assert len(result["score_distribution.csv"]) == 5
