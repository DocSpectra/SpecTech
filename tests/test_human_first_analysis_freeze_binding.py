import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "configs" / "round2_human_first_analysis_freeze_record.json"


def test_freeze_record_binds_the_pre_unblinding_method_commit():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["method_freeze_commit"] == "97fa1a1333a61294f937fdaa68d0449c5b157c13"
    bound = {
        "config_sha256": ROOT / "configs" / "round2_human_first_analysis_v1.json",
        "schema_sha256": ROOT / "schemas" / "round2_human_first_analysis_v1.schema.json",
        "spec_sha256": ROOT / "specs" / "round2_human_first_analysis.md",
        "freeze_test_sha256": ROOT / "tests" / "test_human_first_analysis_freeze.py",
    }
    for key, path in bound.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == freeze[key]


def test_freeze_record_confirms_blinded_gate_and_no_hidden_access():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["pre_unblinding_validation_passed"] is True
    assert freeze["validated_review_rows"] == freeze["validated_allowed_score_rows"] == 180
    assert freeze["review_id_a_b_exact_by_released_packet_row"] is True
    assert freeze["clerical_corrections"] == []
    for key in (
        "hidden_answer_key_opened_before_freeze",
        "hidden_predictor_values_opened_before_freeze",
        "hidden_proxy_decisions_opened_before_freeze",
        "hidden_policy_mappings_opened_before_freeze",
        "unblinded_outcomes_computed_before_freeze",
    ):
        assert freeze[key] is False
