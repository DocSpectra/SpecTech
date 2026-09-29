import hashlib
import json
from pathlib import Path

from src.analysis.dgx_gptoss120b_human_first import content_blind_binding, local_alias, load_method


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_dgx_gptoss120b_human_first_v1.json"
FREEZE = ROOT / "configs" / "round2_dgx_gptoss120b_human_first_freeze_record.json"


def test_content_blind_candidate_population_and_alias_binding():
    result = content_blind_binding(CONFIG, require_freeze=False)
    assert result["source_cases"] == 60
    assert result["candidate_rows"] == 180
    assert result["unique_remote_ids"] == 180
    assert result["unique_local_aliases"] == 180
    assert result["complete_local_integrity_failures"] == 0
    assert result["candidate_structure_sha256"] == "4ec9180830962f6dc549dd3f13a6f48d74921fefefa048bd31aa16517d08d513"
    assert result["remote_alias_bijection_sha256"] == "ca10e13cbd10bc82bf088c08a8ddcf8f08e4bc5992ab1199799b5f9e04b54296"


def test_method_is_fresh_blind_and_audit_only():
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["authorization"]["scorer_outcomes_inspected_before_freeze"] is False
    assert record["authorization"]["human_outcomes_available"] is False
    assert record["blinding"]["review_id_seed"] not in {2026081112}
    assert record["blinding"]["side_seed"] not in {2026081113}
    assert record["blinding"]["row_order_seed"] not in {2026081114}
    assert record["blinding"]["review_id_namespace"] != "human-first-editor-review-v1"
    assert set(record["automatic_proxy_audit"]["forbidden_uses"]) >= {"exclude", "rank", "select"}
    assert record["blinding"]["packet_columns"] == ["Review ID", "A", "B", "Score"]


def test_alias_formula_is_deterministic_and_namespaced():
    record, _ = load_method(CONFIG, require_freeze=False)
    value = local_alias(record, "a" * 64, "b" * 64, 1)
    assert value == local_alias(record, "a" * 64, "b" * 64, 1)
    assert len(value) == 64
    gemma_style = hashlib.sha256(f"{'b' * 64}\0candidate_slot\01".encode()).hexdigest()
    assert value != gemma_style


def test_scorers_policies_ranges_and_review_contract_are_frozen():
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["scoring"]["primary"] == ["speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "granuscore_native"]
    assert record["selection_policies"]["policies"] == ["unguided_slot01", "speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "ko_three_run_arithmetic_mean_secondary", "granuscore_direction_aligned", "rank_consensus_secondary"]
    assert "Decimal" in record["selection_policies"]["ko_mean_rule"]
    assert "lowest candidate slot" in record["selection_policies"]["rule"]
    assert record["returned_candidates"]["hidden_cell_candidate_counts"] == {
        "ansible_docs:add_specific": 24,
        "ansible_docs:de_specify": 36,
        "ansible_docs:irrelevant_rewrite": 30,
        "github_docs:add_specific": 30,
        "github_docs:de_specify": 30,
        "github_docs:irrelevant_rewrite": 30,
    }


def test_freeze_record_binds_committed_method_before_scoring():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["method_freeze_commit"] == "695c68f99744a4ac1fd451bed2d46b456420c7bf"
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == freeze["method_config"]["sha256"]
    assert freeze["chronology"]["scorer_outcomes_before_method_commit"] == 0
    assert freeze["chronology"]["human_outcomes_before_method_commit"] == 0
    assert freeze["chronology"]["new_model_calls"] == 0
