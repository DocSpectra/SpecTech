"""Integrity and privacy checks for the completed Gemma generation gate."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "gemma_controlled_editor"
COMPACT = ROOT / "analysis" / "round2_gemma_controlled_editor"
MANIFEST = COMPACT / "packet_manifest.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("local Gemma generation result is not restored")
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _attempts() -> list[dict]:
    return [json.loads(line) for line in (RAW / "generation_attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]


def test_generation_failure_and_no_packet_disposition_are_exact() -> None:
    manifest = _manifest()
    assert manifest["planned_cases"] == 60
    assert manifest["retained_cases"] == 49
    assert manifest["attempt_rows"] == 90
    assert manifest["generation_gate_passed"] is False
    assert manifest["packet_released"] is False
    assert manifest["scoring_performed"] is False
    assert manifest["rubric_performed"] is False
    assert manifest["human_review_performed"] is False
    coverage = {
        (row["corpus_id"], row["edit_type"]): (int(row["planned"]), int(row["retained"]))
        for row in _csv(COMPACT / "generation_coverage.csv")
    }
    assert coverage == {
        ("ansible_docs", "add_specific"): (8, 2),
        ("ansible_docs", "de_specify"): (12, 11),
        ("ansible_docs", "irrelevant_rewrite"): (10, 10),
        ("github_docs", "add_specific"): (10, 7),
        ("github_docs", "de_specify"): (10, 10),
        ("github_docs", "irrelevant_rewrite"): (10, 9),
    }
    for name in ("blinded_review_packet.csv", "blinded_review_answer_key.csv", "blinded_review_responses.csv"):
        assert not (RAW / name).exists()


def test_all_attempts_first_passes_and_failures_are_accounted() -> None:
    _manifest(); rows = _attempts()
    assert len(rows) == 90
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
        assert row["edited_sha256"] == hashlib.sha256(row["edited_sentence"].encode("utf-8")).hexdigest()
        assert row["status"] == ("accepted" if not row["reason_codes"] else "rejected")
        assert row["model"] == "gemma4:12b"
        assert row["model_tag_digest"] == "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
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
    assert accepted == 49


def test_failure_aggregates_and_raw_hashes_match() -> None:
    manifest = _manifest(); rows = _attempts()
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


def test_compact_pack_is_text_free_and_portable() -> None:
    manifest = _manifest()
    combined = "\n".join(path.read_text(encoding="utf-8-sig") for path in COMPACT.iterdir() if path.is_file()).casefold()
    assert "sentence_original" not in combined
    assert "edited_sentence" not in combined
    assert "sentence_a" not in combined and "sentence_b" not in combined
    assert "c:\\users\\" not in combined and "c:/users/" not in combined
    raw_texts = [row["edited_sentence"] for row in _attempts() if len(row["edited_sentence"]) >= 20]
    assert all(text.casefold() not in combined for text in raw_texts)
    assert manifest["residency_after_generation"]["name"] == "gemma4:12b"
    assert manifest["residency_after_generation"]["digest"] == manifest["model_tag_digest"]
