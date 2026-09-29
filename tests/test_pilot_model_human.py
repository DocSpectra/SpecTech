"""Focused statistics tests for the frozen pilot comparison."""
from __future__ import annotations

import numpy as np

from src.analysis.pilot_model_human import (
    HUMAN_TARGETS,
    PRIMARY_MODELS,
    SECONDARY_MODEL,
    PilotCorpus,
    analyze_corpus,
    bootstrap_correlation_matrices,
)


def _fixture_corpus() -> PilotCorpus:
    labels = {
        "ann_a": np.asarray([1, 1, 2, 3, 4, 5, 5, 4], dtype=float),
        "ann_b": np.asarray([1, 2, 2, 3, 3, 4, 5, 5], dtype=float),
        "ann_c": np.asarray([2, 1, 2, 4, 3, 5, 4, 5], dtype=float),
    }
    labels["pooled_human_mean"] = np.mean(np.vstack(list(labels.values())), axis=0)
    scores = {
        PRIMARY_MODELS[0]: np.asarray([0.0, 0.1, 0.2, 0.4, 0.5, 0.8, 1.0, 0.7]),
        PRIMARY_MODELS[1]: np.asarray([0.2, 0.1, 0.3, 0.5, 0.4, 0.7, 0.9, 0.8]),
        PRIMARY_MODELS[2]: np.asarray([0.8, 0.9, 0.7, 0.5, 0.6, 0.3, 0.1, 0.2]),
        PRIMARY_MODELS[3]: np.asarray([0.1, 0.2, 0.3, 0.3, 0.5, 0.8, 0.8, 0.7]),
    }
    scores[SECONDARY_MODEL] = np.mean(
        np.vstack([scores[model] for model in PRIMARY_MODELS[1:]]), axis=0
    )
    return PilotCorpus(
        corpus_id="fixture",
        sent_ids=tuple(f"s{i}" for i in range(8)),
        buckets=tuple("average" for _ in range(8)),
        labels=labels,
        scores=scores,
    )


def test_bootstrap_is_deterministic_and_joint() -> None:
    columns = np.asarray(
        [[1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]],
        dtype=float,
    )
    point_a, boot_a = bootstrap_correlation_matrices(columns, replicates=100, seed=17)
    point_b, boot_b = bootstrap_correlation_matrices(columns, replicates=100, seed=17)
    np.testing.assert_allclose(point_a, point_b)
    np.testing.assert_allclose(boot_a, boot_b, equal_nan=True)
    assert point_a[0, 1] == 1.0
    assert point_a[0, 2] == -1.0
    np.testing.assert_allclose(boot_a[:, 0, 1], np.ones(100))
    np.testing.assert_allclose(boot_a[:, 0, 2], -np.ones(100))


def test_analysis_emits_frozen_output_shape_and_paired_delta() -> None:
    result = analyze_corpus(
        _fixture_corpus(),
        replicates=200,
        master_seed=20260809,
        minimum_valid_fraction=0.90,
    )
    assert len(result["model_human_agreement.csv"]) == 5 * len(HUMAN_TARGETS)
    assert len(result["paired_model_human_contrasts.csv"]) == 6 * len(HUMAN_TARGETS)
    assert len(result["model_model_agreement.csv"]) == 10
    assert len(result["ko_run_variability.csv"]) == len(HUMAN_TARGETS)
    assert len(result["paper_table.csv"]) == len(HUMAN_TARGETS)
    contrast = result["paired_model_human_contrasts.csv"][0]
    assert np.isclose(
        contrast["delta_rho_a_minus_b"], contrast["rho_a"] - contrast["rho_b"]
    )


def test_ties_use_average_ranks_without_jitter() -> None:
    columns = np.asarray(
        [[1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 3], [3, 3, 2, 2, 1, 1]],
        dtype=float,
    )
    point, _ = bootstrap_correlation_matrices(columns, replicates=20, seed=9)
    assert point[0, 1] == 1.0
    assert point[0, 2] == -1.0
