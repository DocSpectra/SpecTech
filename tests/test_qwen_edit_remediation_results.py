"""Integrity and privacy checks for the completed QE-R held-out run."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "qwen_edit_remediation" / "held_out" / "v2"
COMPACT = ROOT / "analysis" / "round2_qwen_edit_remediation" / "held_out_v2"
METADATA = COMPACT / "run_metadata.json"
CHRONOLOGY = ROOT / "configs" / "round2_qwen_edit_remediation_chronology_note.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _metadata() -> dict:
    if not METADATA.exists():
        pytest.skip("local QE-R held-out results are not restored")
    return json.loads(METADATA.read_text(encoding="utf-8"))


def test_held_out_coverage_failure_and_disposition_are_exact() -> None:
    metadata = _metadata()
    assert metadata["planned_cases"] == 30
    assert metadata["accepted_cases"] == 20
    assert metadata["attempt_rows"] == 50
    assert metadata["coverage_gate_passed"] is False
    assert metadata["disposition"] == "fail_recommend_omit_qwen_edit_arm"
    assert metadata["scoring_performed"] is False
    coverage = {
        (row["corpus_id"], row["edit_type"]): (int(row["planned"]), int(row["accepted"]), int(row["minimum_accepted"]))
        for row in _csv(COMPACT / "coverage.csv")
    }
    assert coverage == {
        ("ansible_docs", "add_specific"): (4, 2, 3),
        ("ansible_docs", "de_specify"): (6, 4, 5),
        ("ansible_docs", "irrelevant_rewrite"): (5, 2, 4),
        ("github_docs", "add_specific"): (5, 5, 4),
        ("github_docs", "de_specify"): (5, 5, 4),
        ("github_docs", "irrelevant_rewrite"): (5, 2, 4),
    }


def test_every_attempt_and_first_acceptance_are_accounted() -> None:
    _metadata()
    attempts = [json.loads(line) for line in (RAW / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    accounting = _csv(COMPACT / "attempt_accounting.csv")
    accepted = _csv(COMPACT / "accepted_manifest.csv")
    assert len(attempts) == len(accounting) == 50
    assert len(accepted) == 20
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        by_case[row["case_id"]].append(row)
        assert row["edited_sha256"] == hashlib.sha256(row["edited_sentence"].encode("utf-8")).hexdigest()
        assert row["status"] == ("accepted" if not row["reason_codes"] else "rejected")
    assert len(by_case) == 30
    for rows in by_case.values():
        assert [row["attempt_index"] for row in rows] == list(range(1, len(rows) + 1))
        passing = [row for row in rows if row["status"] == "accepted"]
        assert len(passing) <= 1
        if passing:
            assert passing[0] is rows[-1]
        else:
            assert len(rows) == 3


def test_compact_hashes_privacy_and_no_scoring_hold() -> None:
    metadata = _metadata()
    for name, expected in metadata["outputs"].items():
        assert _sha(COMPACT / name) == expected
    combined = "\n".join(path.read_text(encoding="utf-8") for path in COMPACT.iterdir() if path.is_file())
    lowered = combined.casefold()
    assert "c:\\users\\" not in lowered and "c:/users/" not in lowered
    assert not set(_csv(COMPACT / "attempt_accounting.csv")[0]).intersection(
        {"sentence_original", "sentence_author_edited", "sentence_qwen_edited", "edited_sentence"}
    )
    raw_texts = [row["edited_sentence"] for row in [json.loads(line) for line in (RAW / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line] if len(row["edited_sentence"]) >= 20]
    assert all(text not in combined for text in raw_texts)
    assert "score_original" not in lowered and "score_edited" not in lowered
    assert metadata["held_out_text_printed_or_interactively_inspected"] is False


def test_failure_reason_aggregates_match_raw_attempts() -> None:
    _metadata()
    attempts = [json.loads(line) for line in (RAW / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    observed: Counter[tuple[str, str, str]] = Counter()
    for row in attempts:
        for reason in row["reason_codes"]:
            observed[(row["corpus_id"], row["edit_type"], reason)] += 1
    compact = Counter({
        (row["corpus_id"], row["edit_type"], row["reason_code"]): int(row["count"])
        for row in _csv(COMPACT / "failure_reasons.csv")
    })
    assert compact == observed


def test_authoritative_commit_freezes_precede_model_responses() -> None:
    note = json.loads(CHRONOLOGY.read_text(encoding="utf-8"))
    assert note["recorded_after_run"] is True
    assert note["candidate_chronology"]["freeze_precedes_first_response"] is True
    assert note["held_out_chronology"]["freeze_precedes_first_response"] is True
    assert note["candidate_chronology"]["attempt_rows"] == 137
    assert note["held_out_chronology"]["attempt_rows"] == 50
    assert note["scientific_rule_changed"] is False
    assert note["output_or_disposition_changed"] is False
