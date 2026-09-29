import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_human_first_reranking_v1.json"
FREEZE = ROOT / "configs" / "round2_human_first_reranking_freeze_record.json"


def _config():
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_human_first_generation_is_fresh_complete_and_integrity_only():
    c = _config()
    assert c["outcome_blind_method_freeze"] is True
    assert c["inputs"]["required_source_cases"] == 60
    assert c["inputs"]["candidate_slots_per_source"] == 3
    assert c["inputs"]["required_candidate_slots"] == 180
    assert c["runtime"]["master_seed"] not in {2026081101}
    assert c["authorization"]["prior_generated_outputs_may_be_read_or_reused"] is False
    forbidden = " ".join(c["integrity_inclusion"]["explicitly_forbidden_for_inclusion"]).casefold()
    for term in ("specificity direction", "length ratio", "speciteller", "ko", "granuscore", "author edits", "human labels"):
        assert term in forbidden
    checks = c["integrity_inclusion"]["checks"]
    assert set(checks) == {"exact_json_field_only", "one_nonempty_line", "minimum_characters", "maximum_characters", "must_differ_after_nfkc_strip_whitespace_collapse_casefold", "at_most_one_detected_sentence_boundary", "new_non_latin_letter_script_forbidden_when_original_has_none"}


def test_freeze_record_binds_pre_generation_commit_and_config():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == freeze["config_sha256"]
    assert freeze["method_freeze_commit"] == "1cc96f741cc9cf2a2b2f96a5bd6d96b6007737a5"
    assert freeze["method_frozen_before_any_project_generation_or_candidate_scoring"] is True
    assert freeze["project_generation_calls_before_freeze"] == 0
    assert freeze["candidate_scoring_calls_before_freeze"] == 0


def test_metric_and_proxy_outputs_cannot_gate_or_enter_packet():
    c = _config()
    assert "never gates" in c["automatic_proxy_audit"]["timing"]
    assert "scores never affect inclusion" in c["scoring"]["timing"]
    assert c["blinding"]["packet_columns"] == ["Review ID", "A", "B", "Score"]
    hidden = " ".join(c["blinding"]["forbidden_packet_information"]).casefold()
    for term in ("corpus", "direction", "source/candidate role", "candidate slot", "proxy result", "metric values", "selector mappings"):
        assert term in hidden


def test_all_requested_models_and_selection_policies_are_frozen():
    c = _config()
    assert c["scoring"]["primary"] == ["speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "granuscore_native"]
    assert "ko_three_run_arithmetic_mean" in c["scoring"]["secondary"]
    assert c["scoring"]["directions"]["granuscore_native"] == "higher_is_coarser_more_abstract"
    policies = c["selection_policies"]["policies"]
    assert policies[0] == "unguided_slot01"
    assert all(name in policies for name in ("speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "ko_three_run_arithmetic_mean_secondary", "granuscore_direction_aligned"))
    assert "lowest candidate_slot" in c["selection_policies"]["rule"]


def test_exact_source_and_identity_bindings_are_unchanged():
    c = _config()
    assert c["inputs"]["ordered_case_sha256"] == "c46c1e7182a34a0f89dbb2826a6fcd2c0eec73f56842a5d484cf7675d95ed7ec"
    for entry in c["inputs"]["source_files"].values():
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    for key in ("base_protocol",):
        entry = c["inputs"][key]
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]
    entry = c["scoring"]["identity_source"]
    assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"]


def test_side_balance_and_release_contract_are_exact():
    c = _config()
    counts = []
    for source in c["inputs"]["source_files"].values():
        counts.extend(value * 3 for value in source["direction_counts"].values())
    assert counts == [24, 36, 30, 30, 30, 30]
    assert all(value % 2 == 0 for value in counts)
    assert "90/90 overall" in c["blinding"]["side_assignment"]
    assert "180/180 retained candidates" in c["release_gate"]
    assert c["authorization"]["manuscript_changes_authorized"] is False
