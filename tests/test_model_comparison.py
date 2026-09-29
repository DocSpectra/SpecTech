"""Tests for model-aware aggregation and rank comparison primitives."""
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.model_comparison import (
    _cluster_rank_correlation_replicates,
    _portable_provenance,
    aggregate_ko_runs,
    fixed_rank_spearman,
    percentile_ranks,
)


def test_paper_provenance_omits_host_specific_run_commands() -> None:
    provenance = {
        "artifacts": {"baseline": {"sha256": "a" * 64}},
        "runs": {
            "demo:run01": {
                "command": [r"docker", r"C:\\private\\target:/target:ro"],
                "score_sha256": "b" * 64,
                "row_count": 50,
            }
        },
    }
    portable = _portable_provenance(provenance)
    assert portable["artifacts"] == provenance["artifacts"]
    assert "command" not in portable["runs"]["demo:run01"]
    assert portable["runs"]["demo:run01"]["score_sha256"] == "b" * 64


def test_three_run_aggregation_is_rowwise_native_mean() -> None:
    runs = np.asarray([[0.1, 0.8], [0.2, 0.7], [0.3, 0.6]])
    assert aggregate_ko_runs(runs) == pytest.approx([0.2, 0.7])
    with pytest.raises(ValueError, match="exactly three"):
        aggregate_ko_runs(runs[:2])


def test_percentile_ranks_average_ties_and_spearman_preserves_order() -> None:
    assert percentile_ranks(np.asarray([1.0, 1.0, 3.0])) == pytest.approx(
        [1.0 / 3.0, 1.0 / 3.0, 5.0 / 6.0]
    )
    assert fixed_rank_spearman(np.arange(10.0), np.arange(10.0)) == pytest.approx(1.0)
    assert fixed_rank_spearman(np.arange(10.0), -np.arange(10.0)) == pytest.approx(-1.0)


def test_cluster_rank_bootstrap_samples_documents_not_rows() -> None:
    x = percentile_ranks(np.asarray([1.0, 2.0, 4.0, 3.0]))
    y = percentile_ranks(np.asarray([1.0, 2.0, 3.0, 4.0]))
    docs = np.asarray([0, 0, 1, 1], dtype=np.int32)
    reps = _cluster_rank_correlation_replicates(x, y, docs, 2, 20, 123)
    assert reps.shape == (20,)
    assert np.all(np.isfinite(reps))
