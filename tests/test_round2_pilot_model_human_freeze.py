"""Outcome-blind contract tests for the existing-pilot extension."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


CONFIG = Path("configs/round2_pilot_model_human_comparison_v1.json")
MANIFEST = Path("configs/round2_pilot_sent_ids_v1.csv")


def test_frozen_scope_and_score_roles() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["outcome_blind_freeze"] is True
    assert record["freeze_commit"] == "345458f09bda6ba673aa4dcaf43b47e64ed9b70b"
    assert record["authorization"] == {
        "existing_labels_only": True,
        "existing_scores_only": True,
        "new_annotation": False,
        "relabeling": False,
        "retraining": False,
        "rescoring": False,
        "controlled_edit_expansion": False,
    }
    primary = record["score_instances"]["primary"]
    assert [row["model_instance_id"] for row in primary] == [
        "speciteller_frozen_round1",
        "ko_official_release_run01",
        "ko_official_release_run02",
        "ko_official_release_run03",
    ]
    assert record["score_instances"]["secondary"]["independent_model"] is False


def test_manifest_freezes_exact_40_by_40_unique_ids() -> None:
    with MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == ["corpus_id", "pilot_position", "bucket", "sent_id"]
    counts = Counter(row["corpus_id"] for row in rows)
    assert counts == {"ansible_docs": 40, "github_docs": 40}
    assert len({(row["corpus_id"], row["sent_id"]) for row in rows}) == 80
    for corpus_id in counts:
        positions = [int(row["pilot_position"]) for row in rows if row["corpus_id"] == corpus_id]
        buckets = Counter(row["bucket"] for row in rows if row["corpus_id"] == corpus_id)
        assert positions == list(range(1, 41))
        assert buckets == {"edge_low": 10, "edge_high": 10, "average": 20}


def test_bootstrap_join_and_claim_boundaries_are_explicit() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    bootstrap = record["analysis"]["bootstrap"]
    assert bootstrap["replicates"] == 10000
    assert bootstrap["master_seed"] == 20260809
    assert bootstrap["minimum_valid_fraction"] == 0.95
    assert "jointly" in bootstrap["joint_resampling"]
    assert record["join_contract"]["required_rows_total"] == 80
    assert record["join_contract"]["missing_labels"].startswith("forbidden")
    assert "not accuracy" in record["analysis"]["claim_boundary"]
    assert "no multiplicity-adjusted" in record["analysis"]["multiplicity"]
