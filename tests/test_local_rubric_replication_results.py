"""Result-integrity tests for the completed local rubric replication."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src.analysis.local_rubric_replication import _exact_quantized_ko_mean, _read_jsonl
from src.analysis.pilot_model_human import (
    _load_corpora,
    _rank_correlation_matrix,
    load_protocol,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/round2_local_rubric_replication_v1.json").read_text(encoding="utf-8"))
RAW = ROOT / CONFIG["outputs"]["raw_directory"]
COMPACT = ROOT / CONFIG["outputs"]["compact_directory"]
JUDGES = tuple(judge["judge_id"] for judge in CONFIG["judges"])


def _csv(name: str) -> list[dict[str, str]]:
    with (COMPACT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_raw_matrices_are_complete_unique_and_text_free() -> None:
    for judge_id in JUDGES:
        rows = _read_jsonl(RAW / judge_id / "raw_responses.jsonl")
        assert len(rows) == 80
        assert sum(row["corpus_id"] == "ansible_docs" for row in rows) == 40
        assert sum(row["corpus_id"] == "github_docs" for row in rows) == 40
        assert {row["score"] for row in rows} <= {1, 2, 3, 4, 5}
        assert all("sent_text" not in row and "thinking" not in row for row in rows)
        assert len({(row["corpus_id"], row["sent_id"]) for row in rows}) == 80


def test_smoke_and_run_metadata_bind_hashes_and_empty_claim_scope() -> None:
    for judge_id in JUDGES:
        directory = RAW / judge_id
        smoke = json.loads((directory / "determinism_smoke.json").read_text(encoding="utf-8"))
        metadata = json.loads((directory / "run_metadata.json").read_text(encoding="utf-8"))
        assert smoke["passed"] is True and len(smoke["results"]) == 9
        for fixture_id in {row["fixture_id"] for row in smoke["results"]}:
            rows = [row for row in smoke["results"] if row["fixture_id"] == fixture_id]
            assert len({row["score"] for row in rows}) == 1
            assert len({row["response_content_sha256"] for row in rows}) == 1
        assert metadata["coverage"] == {"total": 80, "ansible_docs": 40, "github_docs": 40}
        assert metadata["determinism_smoke_sha256"] == sha256_file(directory / "determinism_smoke.json")
        assert metadata["raw_responses_sha256"] == sha256_file(directory / "raw_responses.jsonl")
        assert metadata["project_scoring_started_at_utc"] > metadata["run_started_at_utc"]


def test_compact_hash_manifest_and_bootstrap_gates() -> None:
    metadata = json.loads((COMPACT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["coverage"] == {"total_scores": 160, "rows_per_judge": 80, "rows_per_corpus_per_judge": 40}
    assert metadata["comparison_status"]["primary"] == [
        "qwen3_14b_zero_shot_rubric",
        "speciteller_frozen_round1",
        "ko_official_release_run01",
        "ko_official_release_run02",
        "ko_official_release_run03",
    ]
    assert metadata["comparison_status"]["secondary"] == [
        "granuscore_direction_aligned_secondary",
        "ko_official_release_run_mean",
    ]
    for name, entry in metadata["outputs"].items():
        assert sha256_file(COMPACT / name) == entry["sha256"]
    for path in COMPACT.glob("*.csv"):
        for row in _csv(path.name):
            for key, value in row.items():
                if key.endswith("bootstrap_valid"):
                    assert int(value) >= 9500


def test_secondary_ko_mean_preserves_exact_decimal_ties() -> None:
    scores = {
        "speciteller_frozen_round1": np.asarray([0.0, 0.0]),
        "ko_official_release_run01": np.asarray([0.5542, 0.5541]),
        "ko_official_release_run02": np.asarray([0.5823, 0.5779]),
        "ko_official_release_run03": np.asarray([0.5638, 0.5683]),
    }
    mean = _exact_quantized_ko_mean(scores)
    assert mean[0] == mean[1]


def test_independent_pooled_correlations_match_compact_table() -> None:
    prior_path = ROOT / "configs/round2_pilot_model_human_comparison_v1.json"
    corpora, _ = _load_corpora(load_protocol(prior_path), prior_path)
    stored = {
        (row["corpus_id"], row["judge_id"]): float(row["spearman_rho"])
        for row in _csv("pooled_summary.csv")
    }
    for judge_id in JUDGES:
        rows = _read_jsonl(RAW / judge_id / "raw_responses.jsonl")
        score_map = {(row["corpus_id"], row["sent_id"]): row["score"] for row in rows}
        for corpus_id, corpus in corpora.items():
            values = np.asarray([score_map[(corpus_id, sent_id)] for sent_id in corpus.sent_ids])
            point = _rank_correlation_matrix(np.vstack([values, corpus.labels["pooled_human_mean"]]))[0, 1]
            assert abs(float(point) - stored[(corpus_id, judge_id)]) < 5e-10


def test_tracked_compact_pack_is_privacy_safe_and_claim_bounded() -> None:
    forbidden = ("sent_text", "human_label", "participant_id", "thinking", str(ROOT))
    for path in COMPACT.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not any(term in text for term in forbidden)
    readme = (COMPACT / "README.md").read_text(encoding="utf-8")
    assert "not gold-standard accuracy" in readme
    assert "not accuracy or superiority tests" in readme
