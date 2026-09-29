"""Mechanical runner tests for the matched local rubric replication."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.analysis import local_rubric_replication as rubric


def _record(tmp_path: Path) -> tuple[dict, dict, dict]:
    record = {
        "outputs": {"raw_directory": tmp_path.as_posix()},
    }
    qwen = {
        "determinism_smoke": {
            "fixtures": ["fixture low", "fixture high", "fixture general"],
            "repetitions": 3,
        }
    }
    judge = {
        "judge_id": "fixture_judge",
        "model": "gemma4:12b",
        "model_tag_digest": "a" * 64,
    }
    return record, qwen, judge


def _metadata(content: str) -> dict:
    return {
        "response_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "reasoning_char_count": 0,
        "reasoning_sha256": hashlib.sha256(b"").hexdigest(),
        "prompt_eval_count": 10,
        "eval_count": 2,
        "done_reason": "stop",
    }


def test_run_judge_writes_complete_text_free_matrix(monkeypatch, tmp_path: Path) -> None:
    record, qwen, judge = _record(tmp_path)
    rows = [
        {
            "corpus_id": "ansible_docs" if index < 40 else "github_docs",
            "pilot_position": (index % 40) + 1,
            "sent_id": f"sent-{index:03d}",
            "sent_text": f"private project text {index}",
        }
        for index in range(80)
    ]
    monkeypatch.setattr(rubric, "verify_identity", lambda *_: {"model": judge["model"]})
    monkeypatch.setattr(rubric, "load_project_rows", lambda *_: rows)
    monkeypatch.setattr(rubric, "stop_all_models", lambda: None)
    monkeypatch.setattr(rubric, "verify_sole_residency", lambda *_: {"model": judge["model"]})
    monkeypatch.setattr(rubric, "_ollama", lambda *_: "NAME ID")

    def score(*args):
        text = args[-1]
        value = 2 if text.startswith("fixture") else 3
        content = json.dumps({"score": value}, separators=(",", ":"))
        return value, _metadata(content)

    monkeypatch.setattr(rubric, "request_score", score)
    freeze = {"method_freeze_commit": "1" * 40, "freeze_binding_commit": "2" * 40}
    result = rubric.run_judge(record, qwen, judge, "3" * 64, freeze)

    raw_path = tmp_path / judge["judge_id"] / "raw_responses.jsonl"
    smoke_path = tmp_path / judge["judge_id"] / "determinism_smoke.json"
    metadata_path = tmp_path / judge["judge_id"] / "run_metadata.json"
    assert len(raw_path.read_text(encoding="utf-8").splitlines()) == 80
    assert len(json.loads(smoke_path.read_text(encoding="utf-8"))["results"]) == 9
    assert result["coverage"] == {"total": 80, "ansible_docs": 40, "github_docs": 40}
    combined = raw_path.read_text(encoding="utf-8") + metadata_path.read_text(encoding="utf-8")
    assert "private project text" not in combined


def test_run_judge_rejects_hash_variability_even_when_scores_match(monkeypatch, tmp_path: Path) -> None:
    record, qwen, judge = _record(tmp_path)
    monkeypatch.setattr(rubric, "verify_identity", lambda *_: {"model": judge["model"]})
    monkeypatch.setattr(rubric, "load_project_rows", lambda *_: [])
    monkeypatch.setattr(rubric, "stop_all_models", lambda: None)
    monkeypatch.setattr(rubric, "verify_sole_residency", lambda *_: {"model": judge["model"]})
    monkeypatch.setattr(rubric, "_ollama", lambda *_: "NAME ID")
    calls = iter(range(9))

    def score(*_):
        content = json.dumps({"score": 2, "nonce": next(calls)})
        return 2, _metadata(content)

    monkeypatch.setattr(rubric, "request_score", score)
    freeze = {"method_freeze_commit": "1" * 40, "freeze_binding_commit": "2" * 40}
    with pytest.raises(ValueError, match="Determinism smoke failed"):
        rubric.run_judge(record, qwen, judge, "3" * 64, freeze)
    assert not (tmp_path / judge["judge_id"] / "raw_responses.jsonl").exists()

