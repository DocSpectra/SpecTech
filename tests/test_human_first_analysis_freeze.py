import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_human_first_analysis_v1.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_completed_packet_and_pre_unblinding_gate_are_frozen():
    config = _config()
    assert config["outcome_blind_method_freeze"] is True
    completed = config["inputs"]["completed_workbook"]
    assert completed["sha256"] == "73312e07427d7d4b9aada98261c91b22d6e287cf134cbfed0c22cf36253430e7"
    assert completed["review_rows"] == 180
    assert completed["allowed_scores"] == ["1", "2", "3", "4", "5", "X"]
    assert config["pre_unblinding_validation"]["status"] == "passed_without_opening_hidden_artifacts"
    assert config["pre_unblinding_validation"]["clerical_corrections"] == []


def test_completed_packet_hash_matches_frozen_bytes():
    config = _config()
    completed = ROOT / config["inputs"]["completed_workbook"]["path"]
    assert hashlib.sha256(completed.read_bytes()).hexdigest() == config["inputs"]["completed_workbook"]["sha256"]


def test_score_orientation_and_annotation_caveat_are_explicit():
    config = _config()
    orientation = config["orientation"]
    assert orientation["candidate_side_A"] == {"1": 2, "2": 1, "3": 0, "4": -1, "5": -2, "X": None}
    assert orientation["candidate_side_B"] == {"1": -2, "2": -1, "3": 0, "4": 1, "5": 2, "X": None}
    assert orientation["primary_direction_validity"]["X"] == "false"
    assert "non-success for add_specific and de_specify" in orientation["score_3_sensitivity_rule"]
    caveat = config["annotation_protocol"]["user_rule_caveat"].casefold()
    assert "subtle meaning difference" in caveat and "not recoded" in caveat
    assert "not proof" in config["annotation_protocol"]["claim_boundary"]


def test_order_drift_analyses_are_frozen_without_score_transformation():
    config = _config()["analysis_sets"]
    assert config["primary"] == {"name": "all_180", "review_positions": "1..180 in completed workbook order"}
    assert config["order_drift_descriptive"] == [
        {"name": "first_60", "review_positions": "1..60"},
        {"name": "middle_60", "review_positions": "61..120"},
        {"name": "final_60", "review_positions": "121..180"},
    ]
    assert config["order_drift_sensitivity"] == [
        {"name": "exclude_first_20", "review_positions": "21..180"},
        {"name": "exclude_first_30", "review_positions": "31..180"},
    ]
    assert "No rescaling, recentering" in config["transformations"]


def test_policy_roles_and_paired_case_bootstrap_are_frozen():
    config = _config()
    assert config["policies"]["reference"] == "unguided_slot01"
    assert config["policies"]["primary"] == [
        "unguided_slot01",
        "speciteller_frozen_round1",
        "ko_run01",
        "ko_run02",
        "ko_run03",
        "granuscore_direction_aligned",
    ]
    assert config["policies"]["secondary"] == [
        "ko_three_run_arithmetic_mean_secondary",
        "rank_consensus_secondary",
    ]
    uncertainty = config["uncertainty"]
    assert uncertainty["replicates"] == 20000
    assert uncertainty["unit"] == "source case"
    assert uncertainty["candidate_independent_resampling_forbidden"] is True
    assert "reuse identical draws" in uncertainty["subset_resampling"]


def test_proxy_confusion_and_claim_boundaries_are_frozen():
    config = _config()
    proxy = config["automatic_proxy_comparison"]
    assert set(proxy["confusion_cells"]) == {"true_accept", "false_accept", "false_reject", "true_reject"}
    assert set(proxy["rates"]) == {"false_accept_rate", "false_reject_rate"}
    assert proxy["strata"] == ["overall", "corpus_id", "edit_type"]
    assert proxy["drift_sensitivity"] == ["all_180", "exclude_first_20", "exclude_first_30"]
    reporting = config["reporting"]
    assert reporting["report_all_results_including_unfavorable"] is True
    assert reporting["preserve_all_three_ko_runs_as_primary"] is True
    assert "Do not claim accuracy" in reporting["one_evaluator_boundary"]
    assert "No manuscript integration" in reporting["next_gate"]


def test_tracked_outputs_remain_aggregate_and_privacy_safe():
    tracked = _config()["outputs"]["tracked"]
    assert all(name.endswith((".md", ".json", ".csv")) for name in tracked)
    privacy = _config()["outputs"]["privacy"].casefold()
    for forbidden in ("sentence text", "review ids", "source/candidate ids", "selector mappings", "row-level human outcomes"):
        assert forbidden in privacy
