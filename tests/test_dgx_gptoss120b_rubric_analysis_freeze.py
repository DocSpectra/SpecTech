from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/round2_dgx_gptoss120b_rubric_analysis_v1.json"
FREEZE = ROOT / "configs/round2_dgx_gptoss120b_rubric_analysis_freeze_record.json"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_frozen_contract_is_outcome_blind_and_phase_a_authorized() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["outcome_blind_freeze"] is True
    assert record["authorization"]["phase_a_review_commit"] == "982d78717b98a2399da87f35bb3fe64df29675ab"
    assert record["authorization"]["provenance_disposition"] == "accepted_with_author_provenance_waiver"
    assert record["authorization"]["rubric_score_values_inspected_before_freeze"] is False
    assert record["authorization"]["new_model_calls"] is False


def test_all_repository_bindings_exist_and_match() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    entries = [
        record["authorization"]["provenance_config"],
        record["authorization"]["acceptance_summary"],
        record["pilot_inputs"]["pilot_manifest"],
        record["inherited_protocol"]["qwen_protocol"],
        record["inherited_protocol"]["dgx_rubric_delta"],
        record["human_targets"]["protocol"],
        *record["human_targets"]["labels"],
        *record["judge_inputs"],
        *record["predictor_inputs"]["speciteller"].values(),
        *record["predictor_inputs"]["ko_primary_runs"],
        record["predictor_inputs"]["granuscore_secondary"]["ansible_docs"],
        record["predictor_inputs"]["granuscore_secondary"]["github_docs"],
    ]
    for entry in entries:
        assert _sha(ROOT / entry["path"]) == entry["sha256"], entry["path"]


def test_exact_frozen_analysis_family_and_roles() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["human_targets"]["target_order"] == ["ann_a", "ann_b", "ann_c", "pooled_human_mean"]
    assert [item["judge_id"] for item in record["judge_inputs"]] == [
        "qwen3_14b_zero_shot_rubric",
        "gemma4_12b_zero_shot_rubric",
        "gptoss_20b_zero_shot_rubric",
    ]
    assert len(record["predictor_inputs"]["ko_primary_runs"]) == 6
    assert "secondary" in record["predictor_inputs"]["ko_mean_secondary"]
    assert record["analysis"]["ties"] == "exact average ranks; no jitter"
    assert record["analysis"]["bootstrap"]["replicates"] == 10000
    assert record["analysis"]["bootstrap"]["corpus_seeds"] == {
        "ansible_docs": 16090150990932269746,
        "github_docs": 1405329673207500964,
    }


def test_privacy_claim_and_stop_rules_are_explicit() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert record["reporting"]["track_only_aggregates"] is True
    assert record["reporting"]["raw_row_scores_ignored"] is True
    assert record["reporting"]["candidate_text_forbidden"] is True
    assert record["reporting"]["report_favorable_adverse_null_heterogeneous_and_provenance_results"] is True
    assert "not accuracy" in record["reporting"]["claim_boundary"]
    assert len(record["stop_conditions"]) >= 6


def test_separate_binding_record_binds_committed_method_and_return() -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["method_config"]["sha256"] == _sha(CONFIG)
    assert freeze["method_freeze_commit"] == "4a932113ec49fa1e7eb24f6c097080ef7ca5cea7"
    assert freeze["phase_a"]["review_commit"] == record["authorization"]["phase_a_review_commit"]
    assert freeze["phase_a"]["disposition"] == "accepted_with_author_provenance_waiver"
    assert freeze["returned_rubric_binding"]["member_sha256"] == record["returned_rubric"]["member_sha256"]
    assert freeze["returned_rubric_binding"]["case_id_order_sha256"] == record["returned_rubric"]["case_id_order_sha256"]
    assert freeze["chronology"]["rubric_score_values_inspected_before_binding_record"] is False
