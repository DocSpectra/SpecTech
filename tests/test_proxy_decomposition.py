"""Tests for documentation-aware proxy decomposition analysis."""

from __future__ import annotations

import csv

from src.analysis.proxy_decomposition import run_proxy_decomposition


def test_run_proxy_decomposition_writes_report_and_csv(tmp_path) -> None:
    outputs = tmp_path / "outputs"
    (outputs / "sentences").mkdir(parents=True)
    (outputs / "speciteller").mkdir(parents=True)

    sentence_path = outputs / "sentences" / "demo.csv"
    with sentence_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"])
        writer.writerow(["demo", "doc/a.md", 0, "Use --help in api_client v3.12.", "s1"])
        writer.writerow(["demo", "doc/a.md", 1, "Set timeout: 30 before running git clone ./repo.", "s2"])
        writer.writerow(["demo", "doc/b.md", 0, "This sentence is plain prose.", "s3"])

    score_path = outputs / "speciteller" / "demo_scores.tsv"
    score_path.write_text("s1\t0.8\ns2\t0.7\ns3\t0.2\n", encoding="utf-8")

    analysis_dir = tmp_path / "analysis"
    result = run_proxy_decomposition(
        corpus_ids=["demo"],
        outputs_root=outputs,
        report_path=analysis_dir / "proxy_decomposition_results.md",
        csv_path=analysis_dir / "proxy_decomposition_tables.csv",
    )

    report_path = analysis_dir / "proxy_decomposition_results.md"
    csv_path = analysis_dir / "proxy_decomposition_tables.csv"
    assert report_path.exists()
    assert csv_path.exists()
    assert "identifier_density" in result.report_text

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    features = {row["feature"] for row in rows}
    assert "technical_token_ratio" in features
    assert "identifier_density" in features
    assert "assignment_parameter_density" in features
