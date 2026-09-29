"""Tests for Phase 7.5 controlled-edit behavioral evaluation."""
from __future__ import annotations

import csv

from src.analysis.phase07_analysis_outputs import AnalysisRow
from src.analysis.phase075_controlled_edit_evaluation import build_controlled_edit_template_rows
from src.analysis.phase075_controlled_edit_evaluation import compute_controlled_edit_stats
from src.analysis.phase075_controlled_edit_evaluation import write_controlled_edit_template


def _analysis_row(idx: int, score: float, token_count: int) -> AnalysisRow:
    sent_id = f"s{idx:03d}"
    return AnalysisRow(
        corpus_id="demo",
        doc_path="docs/example.md",
        sent_id=sent_id,
        sent_text=f"Sentence {idx}",
        score=score,
        tfidf_mean_nonzero=0.1 + idx * 0.001,
        tfidf_max=0.2 + idx * 0.001,
        technical_token_ratio=0.05,
        token_count=token_count,
        char_count=10 + idx,
    )


def test_build_controlled_edit_template_rows_is_deterministic() -> None:
    rows = [_analysis_row(i, score=i / 99.0, token_count=5 + (i % 7)) for i in range(100)]

    first = build_controlled_edit_template_rows(rows, seed=13)
    second = build_controlled_edit_template_rows(rows, seed=13)

    assert first == second
    assert len(first) == 30
    assert {row.edit_type for row in first} == {
        "de_specify",
        "add_specific",
        "irrelevant_rewrite",
    }


def test_build_controlled_edit_template_rows_supports_length_band() -> None:
    rows = [_analysis_row(i, score=i / 99.0, token_count=5 + (i % 20)) for i in range(100)]
    sampled = build_controlled_edit_template_rows(rows, length_count=5, seed=13)
    assert len(sampled) == 35


def test_compute_controlled_edit_stats_directionality_and_abs_delta() -> None:
    class _Row:
        def __init__(self, edit_type: str, delta: float) -> None:
            self.edit_type = edit_type
            self.delta = delta

    rows = [
        _Row("de_specify", -0.1),
        _Row("de_specify", 0.2),
        _Row("add_specific", 0.4),
        _Row("add_specific", -0.1),
        _Row("irrelevant_rewrite", -0.3),
        _Row("irrelevant_rewrite", 0.1),
    ]
    stats = compute_controlled_edit_stats(rows)  # type: ignore[arg-type]
    by_type = {row.edit_type: row for row in stats}

    assert by_type["de_specify"].percent_directionally_correct == 50.0
    assert by_type["add_specific"].percent_directionally_correct == 50.0
    assert by_type["irrelevant_rewrite"].percent_directionally_correct is None
    assert by_type["irrelevant_rewrite"].mean_absolute_delta == 0.2


def test_write_controlled_edit_template_columns(tmp_path) -> None:
    rows = [_analysis_row(i, score=i / 99.0, token_count=5 + (i % 7)) for i in range(100)]
    template_rows = build_controlled_edit_template_rows(rows, seed=13)
    output = tmp_path / "template.csv"

    write_controlled_edit_template(output, template_rows)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert reader.fieldnames == [
        "sent_id",
        "corpus_id",
        "sentence_original",
        "speciteller_score_original",
        "token_count",
        "edit_type",
        "sentence_edited",
    ]
    assert len(written) == 30
