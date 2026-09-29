"""Integrity checks for the completed, coverage-gated QE-A evidence."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "qwen_edit_source"
COMPACT = ROOT / "analysis" / "round2_qwen_edit_source"
METADATA = COMPACT / "run_metadata.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_results() -> dict:
    if not METADATA.exists():
        pytest.skip("local QE-A result pack is not restored")
    return json.loads(METADATA.read_text(encoding="utf-8"))


def test_coverage_gate_stops_paper_facing_source_inference() -> None:
    metadata = _require_results()
    assert metadata["generation"]["planned_cases"] == 60
    assert metadata["generation"]["accepted_cases"] == 24
    assert metadata["generation"]["coverage_gate_passed"] is False
    assert metadata["analysis"]["paper_facing_source_comparison_performed"] is False
    assert (COMPACT / "gate_failure.csv").exists()
    assert not (COMPACT / "arm_summaries.csv").exists()
    assert not (COMPACT / "paired_source_contrasts.csv").exists()
    assert not (COMPACT / "paper_table.csv").exists()


def test_every_prepared_score_joins_to_all_primary_and_secondary_instances() -> None:
    metadata = _require_results()
    with (COMPACT / "scores_long.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 504 == metadata["scoring"]["long_rows"]
    counts = Counter(row["model_instance_id"] for row in rows)
    assert set(counts) == set(metadata["scoring"]["models"])
    assert set(counts.values()) == {84}
    assert Counter(row["edit_source"] for row in rows) == {"author": 360, "qwen": 144}
    originals: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in rows:
        originals[(row["case_id"], row["model_instance_id"])].add(row["score_original"])
    assert all(len(values) == 1 for values in originals.values())
    assert metadata["scoring"]["required_join_passed"] is True


def test_compact_output_hashes_and_privacy_boundary_hold() -> None:
    metadata = _require_results()
    for name, entry in metadata["outputs"].items():
        assert _sha(COMPACT / name) == entry["sha256"]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in COMPACT.iterdir() if path.is_file())
    lowered = combined.casefold()
    assert "c:\\users\\" not in lowered
    assert "c:/users/" not in lowered
    with (RAW / "accepted_edits.csv").open(encoding="utf-8", newline="") as handle:
        accepted = list(csv.DictReader(handle))
    for row in accepted:
        for key in ("sentence_original", "sentence_author_edited", "sentence_qwen_edited"):
            text = row[key].strip()
            assert len(text) >= 20
            assert text not in combined


def test_prefixed_grammar_failures_remain_hash_addressed() -> None:
    freeze = json.loads((ROOT / "configs" / "round2_qwen_edit_source_freeze_record.json").read_text(encoding="utf-8"))
    correction = freeze["outcome_blind_mechanical_correction"]
    archive = RAW / "pre_schema_fix"
    assert _sha(archive / "generation_attempts.jsonl") == correction["pre_fix_attempts_sha256"]
    assert _sha(archive / "generation_manifest.csv") == correction["pre_fix_manifest_sha256"]
    assert _sha(archive / "generation_summary.json") == correction["pre_fix_summary_sha256"]


def test_local_scorer_files_have_exact_prepared_coverage() -> None:
    _require_results()
    with (RAW / "scoring_manifest.csv").open(encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(manifest) == 84
    assert sum(1 for _ in (RAW / "scoring/speciteller/scores.tsv").open(encoding="utf-8")) == 84
    with (RAW / "scoring/granuscore/scores.csv").open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 84
    for corpus_id, expected in (("ansible_docs", 41), ("github_docs", 43)):
        for run_id in ("run01", "run02", "run03"):
            path = RAW / "scoring" / "ko" / corpus_id / run_id / "scores.csv"
            with path.open(encoding="utf-8", newline="") as handle:
                assert len(list(csv.DictReader(handle))) == expected
