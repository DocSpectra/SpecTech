"""Integrity and privacy checks for the completed GPT-OSS generation gate."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest

from src.analysis.gemma_controlled_editor import derive_attempt_seed


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "gptoss_controlled_editor"
COMPACT = ROOT / "analysis" / "round2_gptoss_controlled_editor"
MANIFEST = COMPACT / "packet_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("local GPT-OSS generation result is not restored")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _attempts() -> list[dict]:
    return [json.loads(line) for line in (RAW / "generation_attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_generation_failure_and_no_packet_disposition_are_exact() -> None:
    manifest = _manifest()
    assert manifest["planned_cases"] == 60
    assert manifest["retained_cases"] == 54
    assert manifest["attempt_rows"] == 91
    assert manifest["generation_gate_passed"] is False
    assert manifest["packet_released"] is False
    assert manifest["scoring_performed"] is False
    assert manifest["rubric_performed"] is False
    assert manifest["human_review_performed"] is False
    assert manifest["reasoning_effort"] == "low"
    assert manifest["reasoning_text_persisted"] is False
    coverage = {
        (row["corpus_id"], row["edit_type"]): (int(row["planned"]), int(row["retained"]))
        for row in _csv(COMPACT / "generation_coverage.csv")
    }
    assert coverage == {
        ("ansible_docs", "add_specific"): (8, 6),
        ("ansible_docs", "de_specify"): (12, 12),
        ("ansible_docs", "irrelevant_rewrite"): (10, 8),
        ("github_docs", "add_specific"): (10, 10),
        ("github_docs", "de_specify"): (10, 10),
        ("github_docs", "irrelevant_rewrite"): (10, 8),
    }
    for name in ("blinded_review_packet.csv", "blinded_review_answer_key.csv", "blinded_review_responses.csv", "blinded_review_instructions.md"):
        assert not (RAW / name).exists()


def test_all_attempts_first_passes_seeds_and_reasoning_metadata_are_accounted() -> None:
    _manifest(); rows = _attempts()
    assert len(rows) == 91
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
        assert row["edited_sha256"] == hashlib.sha256(row["edited_sentence"].encode("utf-8")).hexdigest()
        assert row["status"] == ("accepted" if not row["reason_codes"] else "rejected")
        assert row["model"] == "gpt-oss:20b"
        assert row["model_tag_digest"] == "17052f91a42e97930aa6e28a6c6c06a983e6a58dbb00434885a0cf5313e376f7"
        assert row["seed"] == derive_attempt_seed(2026081101, row["case_id"], row["attempt_index"])
        assert "thinking" not in row
        assert row["thinking_char_count"] >= 0
        assert len(row["thinking_sha256"]) == 64
        if not any(code.startswith("request_or_parse_error:") for code in row["reason_codes"]):
            assert row["thinking_char_count"] > 0
    assert len(by_case) == 60
    accepted = 0
    for values in by_case.values():
        assert [row["attempt_index"] for row in values] == list(range(1, len(values) + 1))
        passing = [row for row in values if row["status"] == "accepted"]
        assert len(passing) <= 1
        if passing:
            assert passing[0] is values[-1]
            accepted += 1
        else:
            assert len(values) == 3
    assert accepted == 54


def test_retained_join_failure_aggregates_and_hashes_match() -> None:
    manifest = _manifest(); rows = _attempts(); retained = _csv(RAW / "retained_candidates.csv")
    accepted = {row["case_id"]: row for row in rows if row["status"] == "accepted"}
    assert len(retained) == len(accepted) == 54
    for row in retained:
        source = accepted[row["case_id"]]
        assert int(row["attempt_index"]) == source["attempt_index"]
        assert int(row["seed"]) == source["seed"]
        assert row["original_sha256"] == source["original_sha256"]
        assert row["edited_sha256"] == source["edited_sha256"]
        assert hashlib.sha256(row["edited_sentence"].encode("utf-8")).hexdigest() == row["edited_sha256"]
    observed: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        for reason in row["reason_codes"]:
            observed[(row["corpus_id"], row["edit_type"], reason)] += 1
    compact = Counter({
        (row["corpus_id"], row["edit_type"], row["reason_code"]): int(row["count"])
        for row in _csv(COMPACT / "generation_failure_reasons.csv")
    })
    assert compact == observed
    assert manifest["raw_hashes"]["attempts_sha256"] == _sha(RAW / "generation_attempts.jsonl")
    assert manifest["raw_hashes"]["retained_candidates_sha256"] == _sha(RAW / "retained_candidates.csv")
    for name, expected in manifest["compact_hashes"].items():
        assert _sha(COMPACT / name) == expected


def test_compact_pack_is_text_and_reasoning_free_and_portable() -> None:
    manifest = _manifest()
    combined = "\n".join(path.read_text(encoding="utf-8-sig") for path in COMPACT.iterdir() if path.is_file()).casefold()
    for field in ("sentence_original", "edited_sentence", "sentence_a", "sentence_b", "thinking_sha256", "thinking_char_count"):
        assert field not in combined
    assert "c:\\users\\" not in combined and "c:/users/" not in combined
    raw_texts = [row["edited_sentence"] for row in _attempts() if len(row["edited_sentence"]) >= 20]
    assert all(text.casefold() not in combined for text in raw_texts)
    assert manifest["residency_after_generation"]["name"] == "gpt-oss:20b"
    assert manifest["residency_after_generation"]["digest"] == manifest["model_tag_digest"]
