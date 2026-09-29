"""Tests for corpus-level score summary statistics generation."""

from __future__ import annotations

import csv

import pytest

from src.analysis.score_stats import compute_and_write_score_stats_for_corpora
from src.analysis.score_stats import compute_score_stats_from_file


def test_compute_score_stats_from_canonical_tsv(tmp_path) -> None:
    score_path = tmp_path / "demo_scores.tsv"
    score_path.write_text("s1\t0.10\ns2\t0.30\ns3\t0.90\n", encoding="utf-8")

    row = compute_score_stats_from_file("demo", score_path)

    assert row.corpus_id == "demo"
    assert row.score_count == 3
    assert row.score_mean == pytest.approx((0.10 + 0.30 + 0.90) / 3)
    assert row.score_median == pytest.approx(0.30)
    assert row.score_min == pytest.approx(0.10)
    assert row.score_max == pytest.approx(0.90)
    assert row.score_q1 == pytest.approx(0.20)
    assert row.score_q3 == pytest.approx(0.60)
    assert row.score_iqr == pytest.approx(0.40)


def test_compute_score_stats_rejects_duplicate_sent_ids(tmp_path) -> None:
    score_path = tmp_path / "dup_scores.tsv"
    score_path.write_text("s1\t0.10\ns1\t0.20\n", encoding="utf-8")

    qa_note = tmp_path / "dup_qa.txt"
    with pytest.raises(ValueError, match="duplicate sent_id"):
        compute_score_stats_from_file("dup", score_path, qa_note_paths=[qa_note])

    assert qa_note.exists()
    assert "duplicate sent_id" in qa_note.read_text(encoding="utf-8")


def test_compute_and_write_score_stats_writes_required_outputs(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "speciteller").mkdir(parents=True, exist_ok=True)
    (outputs / "speciteller" / "a_scores.tsv").write_text("s1\t0.1\ns2\t0.2\n", encoding="utf-8")
    (outputs / "speciteller" / "b_scores.tsv").write_text("s3\t0.8\ns4\t0.9\n", encoding="utf-8")

    rows = compute_and_write_score_stats_for_corpora(["a", "b"], outputs)
    assert len(rows) == 2

    per_corpus_analysis = outputs / "analysis" / "tables" / "a_score_stats.csv"
    per_corpus_paper = outputs / "paper_pack" / "tables" / "b_score_stats.csv"
    aggregate_analysis = outputs / "analysis" / "tables" / "corpus_score_stats_all.csv"
    aggregate_paper = outputs / "paper_pack" / "tables" / "corpus_score_stats_all.csv"

    assert per_corpus_analysis.exists()
    assert per_corpus_paper.exists()
    assert aggregate_analysis.exists()
    assert aggregate_paper.exists()

    with aggregate_analysis.open("r", encoding="utf-8", newline="") as handle:
        rows_csv = list(csv.DictReader(handle))
    assert len(rows_csv) == 2
    assert set(r["corpus_id"] for r in rows_csv) == {"a", "b"}
