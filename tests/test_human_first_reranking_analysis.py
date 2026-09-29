import math
from pathlib import Path

import numpy as np

from src.analysis.human_first_reranking_analysis import (
    _case_draws,
    average_ranks,
    build_unblinded_rows,
    direction_utility,
    direction_valid,
    human_delta,
    load_and_validate_reviews,
    load_protocol,
    rating_order_distribution,
    spearman,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_human_first_analysis_v1.json"


def test_human_orientation_respects_candidate_side_and_x():
    assert [human_delta(score, "A") for score in ("1", "2", "3", "4", "5", "X")] == [2, 1, 0, -1, -2, None]
    assert [human_delta(score, "B") for score in ("1", "2", "3", "4", "5", "X")] == [-2, -1, 0, 1, 2, None]
    assert direction_utility(2, "add_specific") == 2
    assert direction_utility(-2, "de_specify") == 2
    assert direction_utility(1, "irrelevant_rewrite") == -1
    assert direction_valid(0, "irrelevant_rewrite") is True
    assert direction_valid(0, "add_specific") is False
    assert direction_valid(None, "de_specify") is False


def test_average_ranks_and_spearman_handle_ties():
    assert np.allclose(average_ranks(np.asarray([10.0, 10.0, 30.0])), [1.5, 1.5, 3.0])
    assert math.isclose(spearman([1, 2, 3], [3, 2, 1]), -1.0)
    assert math.isnan(spearman([1, 1, 1], [1, 2, 3]))


def test_case_bootstrap_is_deterministic_and_uses_cases():
    rows = [
        {"case_id": "a", "corpus_id": "x", "edit_type": "d"},
        {"case_id": "a", "corpus_id": "x", "edit_type": "d"},
        {"case_id": "b", "corpus_id": "x", "edit_type": "d"},
        {"case_id": "b", "corpus_id": "x", "edit_type": "d"},
    ]
    ids_a, draws_a = _case_draws(rows, 20, 17, False)
    ids_b, draws_b = _case_draws(rows, 20, 17, False)
    assert ids_a == ids_b == ["a", "b"]
    assert np.array_equal(draws_a, draws_b)
    assert draws_a.shape == (20, 2)


def test_real_completed_packet_and_hidden_joins_reconcile_without_text_output():
    record, _ = load_protocol(CONFIG)
    reviews = load_and_validate_reviews(record)
    rows, policies = build_unblinded_rows(record, reviews)
    assert len(rows) == 180
    assert len({row["case_id"] for row in rows}) == 60
    assert all(sum(candidate["case_id"] == case_id for candidate in rows) == 3 for case_id in {row["case_id"] for row in rows})
    assert len(policies) == 480
    assert set(row["score"] for row in rows) <= {"1", "2", "3", "4", "5", "X"}
    distributions = rating_order_distribution(rows)
    neutral_three = next(
        row for row in distributions
        if row["distribution_view"] == "unblinded_condition"
        and row["stratum_type"] == "edit_type"
        and row["stratum_value"] == "irrelevant_rewrite"
        and row["score"] == "3"
    )
    assert neutral_three["count"] == 36
