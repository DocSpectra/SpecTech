"""Tests for Phase 7 analysis outputs."""
from __future__ import annotations

import csv

import pytest

from src.analysis.phase07_analysis_outputs import run_phase7_for_corpus
from src.analysis.phase07_analysis_outputs import spearman


def test_spearman_monotonic_relationships() -> None:
    xs = [1.0, 2.0, 3.0, 4.0]
    ys_pos = [10.0, 20.0, 30.0, 40.0]
    ys_neg = [40.0, 30.0, 20.0, 10.0]
    assert spearman(xs, ys_pos) == pytest.approx(1.0)
    assert spearman(xs, ys_neg) == pytest.approx(-1.0)


def test_run_phase7_for_corpus_writes_artifacts(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "sentences").mkdir(parents=True, exist_ok=True)
    (outputs / "speciteller").mkdir(parents=True, exist_ok=True)
    (outputs / "features").mkdir(parents=True, exist_ok=True)

    corpus_id = "demo"
    sentence_path = outputs / "sentences" / f"{corpus_id}.csv"
    with sentence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"])
        writer.writerow(["demo", "doc/a.md", 0, "Use --help in module.func v3.12.", "s1"])
        writer.writerow(["demo", "doc/a.md", 1, "This is plain prose.", "s2"])
        writer.writerow(["demo", "doc/b.md", 0, "Call api_client with --force.", "s3"])

    score_path = outputs / "speciteller" / f"{corpus_id}_scores.tsv"
    score_path.write_text("s1\t0.9\ns2\t0.1\ns3\t0.7\n", encoding="utf-8")

    feature_path = outputs / "features" / f"{corpus_id}_features.csv"
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "corpus_id",
                "sent_id",
                "tfidf_mean_nonzero",
                "tfidf_max",
                "technical_token_ratio",
                "token_count",
                "char_count",
            ]
        )
        writer.writerow(["demo", "s1", "0.5", "0.9", "0.5", "8", "31"])
        writer.writerow(["demo", "s2", "0.2", "0.4", "0.0", "5", "20"])
        writer.writerow(["demo", "s3", "0.4", "0.8", "0.4", "6", "30"])

    run_phase7_for_corpus(corpus_id, outputs_root=outputs)

    assert (outputs / "analysis" / "tables" / "demo_corpus_stats.csv").exists()
    assert (outputs / "analysis" / "tables" / "demo_score_feature_spearman.csv").exists()
    assert (outputs / "analysis" / "figures" / "demo_score_distribution.svg").exists()
    assert (outputs / "analysis" / "figures" / "demo_score_vs_tfidf_mean.svg").exists()
    assert (outputs / "analysis" / "figures" / "demo_score_vs_tfidf_max.svg").exists()
    assert (outputs / "analysis" / "figures" / "demo_score_vs_technical_ratio.svg").exists()
    assert (outputs / "analysis" / "samples" / "demo_divergence_samples.csv").exists()
