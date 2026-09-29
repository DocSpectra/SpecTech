"""Fixture-level tests for the frozen distribution/saturation implementation."""
from __future__ import annotations

import numpy as np

from src.analysis.distribution_saturation import (
    bootstrap_document_mean,
    boundary_rows,
    cluster_proportion_replicates,
    diagnostic_row,
    document_spans,
    histogram_rows,
    summarize_values,
    bounds_and_sentinel_gate,
)
from src.analysis.granuscore_comparison import GranuCorpus
from src.analysis.model_comparison import ComparisonCorpus


MODEL = {
    "model_instance_id": "fixture_probability",
    "model_family": "fixture",
    "role": "test",
    "minimum": 0.0,
    "maximum": 1.0,
    "direction": "higher_is_more_specific",
}


def test_summary_uses_frozen_quantiles_ranges_and_exact_mass_points() -> None:
    values = np.asarray([0.0, 0.0, 0.1, 0.5, 0.9, 1.0], dtype=np.float64)
    row = summarize_values(values, model=MODEL, corpus_id="fixture", subset="original")
    expected = np.quantile(values, [0.01, 0.05, 0.25, 0.75, 0.95, 0.99], method="linear")
    assert np.isclose(row["p01"], expected[0])
    assert np.isclose(row["p05"], expected[1])
    assert np.isclose(row["iqr"], expected[3] - expected[2])
    assert np.isclose(row["p95_minus_p05"], expected[4] - expected[1])
    assert np.isclose(row["p99_minus_p01"], expected[5] - expected[0])
    assert row["theoretical_range_utilization"] == 1.0
    assert row["exact_unique_count"] == 5
    assert np.isclose(row["tie_rate"], 1 - 5 / 6)
    assert row["top_mass_point_value"] == 0.0
    assert np.isclose(row["top_mass_point_share"], 2 / 6)


def test_boundary_regions_use_theoretical_bounds_and_are_separate() -> None:
    values = np.asarray([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    rows = boundary_rows(values, model=MODEL, corpus_id="fixture", subset="original", fractions=[0.01, 0.05, 0.10])
    index = {(row["fraction"], row["region"]): row for row in rows}
    assert index[(0.01, "lower")]["count"] == 2
    assert index[(0.01, "upper")]["count"] == 2
    assert index[(0.05, "lower")]["count"] == 3
    assert index[(0.05, "upper")]["count"] == 3
    assert index[(0.10, "lower")]["threshold"] == 0.10
    assert index[(0.10, "upper")]["threshold"] == 0.90


def test_histogram_is_fixed_proportion_and_reconciles_last_bound() -> None:
    values = np.asarray([0.0, 0.01, 0.5, 0.999, 1.0])
    rows = histogram_rows(values, model=MODEL, corpus_id="fixture", subset="strict", bins=50)
    assert len(rows) == 50
    assert sum(row["count"] for row in rows) == 5
    assert np.isclose(sum(row["proportion"] for row in rows), 1.0)
    assert rows[-1]["right_edge_inclusive"] is True
    assert rows[-1]["bin_right"] == 1.0


def test_document_cluster_boundary_bootstrap_is_deterministic() -> None:
    indicator = np.asarray([True, False, True, True, False, False])
    codes = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int32)
    mask = np.ones(6, dtype=np.bool_)
    first = cluster_proportion_replicates(indicator, codes, mask, replicates=40, seed=9)
    second = cluster_proportion_replicates(indicator, codes, mask, replicates=40, seed=9)
    assert np.array_equal(first, second)
    assert np.all((first >= 0) & (first <= 1))


def test_document_span_estimand_and_bootstrap_are_explicit() -> None:
    values = np.concatenate([np.linspace(0, 1, 20), np.linspace(0.2, 0.8, 20), [0.4] * 5])
    codes = np.concatenate([np.zeros(20), np.ones(20), np.full(5, 2)]).astype(np.int32)
    mask = np.ones(values.size, dtype=np.bool_)
    spans = document_spans(values, codes, mask, minimum_rows=20)
    assert spans.size == 2
    assert np.all(spans > 0)
    first = bootstrap_document_mean(spans, replicates=50, seed=17)
    second = bootstrap_document_mean(spans, replicates=50, seed=17)
    assert np.array_equal(first, second)


def test_diagnostic_never_equates_broadness_with_smoothness() -> None:
    values = np.linspace(0.1, 0.9, 101)
    summary = summarize_values(values, model=MODEL, corpus_id="fixture", subset="original")
    boundaries = boundary_rows(values, model=MODEL, corpus_id="fixture", subset="original", fractions=[0.01, 0.05, 0.10])
    histogram = histogram_rows(values, model=MODEL, corpus_id="fixture", subset="original", bins=50)
    row = diagnostic_row(summary, boundaries, histogram)
    assert row["dynamic_range_category"] == "broad"
    assert row["broad_graded_appearance"] is True
    assert "smooth" not in row["shape_diagnostic"]


def test_bounds_gate_uses_frozen_granuscore_unit_count_field() -> None:
    config = {
        "corpora": {"row_counts": {corpus: 4 for corpus in ("wikipedia_en", "github_docs", "ansible_docs", "python_312_html")}},
        "model_instances": [
            {"model_instance_id": model_id, "minimum": 0.0, "maximum": 100.0 if model_id == "granuscore_native" else 1.0}
            for model_id in (
                "speciteller_frozen_round1", "ko_official_release_run01", "ko_official_release_run02",
                "ko_official_release_run03", "ko_official_release_run_mean", "granuscore_native",
            )
        ],
    }
    data = {}
    for corpus_id in config["corpora"]["row_counts"]:
        base = ComparisonCorpus(
            corpus_id=corpus_id, display_name=corpus_id,
            sent_ids=tuple(f"{corpus_id}-{index}" for index in range(4)),
            doc_paths=("d0", "d0", "d1", "d1"), doc_codes=np.asarray([0, 0, 1, 1], dtype=np.int32),
            doc_names=("d0", "d1"), token_count=np.ones(4, dtype=np.int32), char_count=np.ones(4, dtype=np.int32),
            keep=np.ones(4, dtype=np.bool_), speciteller=np.asarray([0.1, 0.2, 0.3, 0.4]),
            ko_runs=np.asarray([[0.2, 0.3, 0.4, 0.5]] * 3),
        )
        data[corpus_id] = GranuCorpus(
            base=base, scores=np.asarray([7.0, 20.0, 30.0, 40.0]),
            unit_count=np.asarray([0, 1, 1, 1], dtype=np.int32),
            no_unit=np.asarray([True, False, False, False]),
        )
    rows, sentinel = bounds_and_sentinel_gate(data, config)
    assert len(rows) == 24
    assert sentinel == 7.0
