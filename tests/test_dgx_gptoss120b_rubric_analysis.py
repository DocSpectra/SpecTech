from __future__ import annotations

import csv
import json
from pathlib import Path

from src.analysis.dgx_gptoss120b_rubric import (
    FREEZE,
    GPT120,
    load_frozen_config,
    load_returned_scores,
)
from src.analysis.pilot_model_human import sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/round2_dgx_gptoss120b_rubric_analysis_v1.json"
COMPACT = ROOT / "analysis/round2_dgx_gptoss120b_rubric"


def _rows(name: str) -> list[dict[str, str]]:
    with (COMPACT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_returned_matrix_matches_frozen_identity_and_range() -> None:
    record, _ = load_frozen_config(CONFIG.relative_to(ROOT))
    rows = load_returned_scores(record)
    assert len(rows) == 80
    assert len({row["rubric_case_id"] for row in rows}) == 80
    assert {row["score"] for row in rows} <= {1, 2, 3, 4, 5}


def test_compact_coverage_roles_and_valid_bootstraps() -> None:
    coverage = _rows("coverage.csv")
    assert [(row["corpus_id"], row["judge_id"], row["observed_rows"]) for row in coverage] == [
        ("ansible_docs", GPT120, "40"), ("github_docs", GPT120, "40")
    ]
    comparator = _rows("judge_comparator_agreement.csv")
    assert len(comparator) == 12
    assert sum(row["analysis_role"] == "primary" for row in comparator) == 8
    assert sum(row["analysis_role"] == "secondary" for row in comparator) == 4
    for name in (
        "judge_human_agreement.csv", "judge_pair_agreement.csv",
        "judge_pair_human_contrasts.csv", "judge_comparator_agreement.csv",
        "judge_vs_comparator_human_contrasts.csv",
    ):
        for row in _rows(name):
            assert int(row["bootstrap_valid"]) >= 9500


def test_distribution_and_ordinal_summaries_are_complete() -> None:
    distribution = _rows("score_distribution.csv")
    assert len(distribution) == 10
    for corpus_id in ("ansible_docs", "github_docs"):
        rows = [row for row in distribution if row["corpus_id"] == corpus_id]
        assert [int(row["score"]) for row in rows] == [1, 2, 3, 4, 5]
        assert sum(int(row["count"]) for row in rows) == 40
        assert abs(sum(float(row["proportion"]) for row in rows) - 1.0) < 1e-9
    assert len(_rows("ordinal_summary.csv")) == 2


def test_manifest_hashes_and_prior_output_invariance() -> None:
    manifest = json.loads((COMPACT / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        assert sha256_file(COMPACT / entry["path"]) == entry["sha256"]
    invariance = json.loads((COMPACT / "prior_output_invariance.json").read_text(encoding="utf-8"))
    assert invariance["required_files"] == invariance["byte_identical_files"] == 13
    assert all(row["byte_identical"] for row in invariance["files"])


def test_compact_pack_contains_aggregates_only_and_claim_boundary() -> None:
    forbidden = ("sent_text", "sentence_text", "rubric_case_id", "participant_id", str(ROOT))
    for path in COMPACT.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in forbidden), path
    readme = (COMPACT / "README.md").read_text(encoding="utf-8")
    assert "not accuracy" in readme
    assert "accepted_with_author_provenance_waiver" in readme
    freeze = json.loads((ROOT / FREEZE).read_text(encoding="utf-8"))
    assert freeze["chronology"]["rubric_score_values_inspected_before_binding_record"] is False
