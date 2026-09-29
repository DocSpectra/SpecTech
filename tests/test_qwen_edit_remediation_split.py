"""Pre-inspection split and budget freeze tests for QE-R."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from src.analysis.qwen_edit_remediation import (
    _candidate_gate,
    _candidate_prompt,
    _load_candidate_context,
    _v2_gate,
    build_split_rows,
    diagnose_v1,
    ordered_assignment_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_qwen_edit_remediation_split_v1.json"
FREEZE = ROOT / "configs" / "round2_qwen_edit_remediation_split_freeze_record.json"
CANDIDATES = ROOT / "configs" / "round2_qwen_edit_remediation_candidates_v1.json"
CANDIDATE_FREEZE = ROOT / "configs" / "round2_qwen_edit_remediation_candidates_freeze_record.json"
V2 = ROOT / "configs" / "round2_qwen_edit_remediation_v2.json"
V2_ATTEMPT_SCHEMA = ROOT / "schemas" / "round2_qwen_edit_remediation_v2.schema.json"
V2_FREEZE = ROOT / "configs" / "round2_qwen_edit_remediation_v2_freeze_record.json"


def _record() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def test_split_is_exact_stratified_and_deterministic() -> None:
    record = _record()
    rows = build_split_rows(CONFIG)
    assert rows == build_split_rows(CONFIG)
    assert len(rows) == len({row["case_id"] for row in rows}) == 60
    counts = Counter((row["split"], row["corpus_id"], row["edit_type"]) for row in rows)
    for split in ("development", "held_out"):
        expected = record["split"]["expected_counts"][split]
        assert {f"{corpus}:{direction}": counts[(split, corpus, direction)] for corpus, direction in [key.split(":") for key in expected]} == expected


def test_manifest_matches_algorithm_and_contains_no_text_or_outcome() -> None:
    record = _record()
    path = ROOT / record["split"]["manifest_path"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = [{key: str(value) for key, value in row.items()} for row in build_split_rows(CONFIG)]
    assert rows == expected
    assert hashlib.sha256(path.read_bytes()).hexdigest() == record["split"]["manifest_sha256"]
    assert ordered_assignment_sha256(rows) == record["split"]["ordered_assignment_sha256"]
    assert not set(rows[0]).intersection({"sentence_original", "sentence_edited", "accepted", "reason_codes", "score"})


def test_budget_thresholds_and_privacy_are_frozen() -> None:
    record = _record()
    budget = record["development_budget"]
    assert budget["maximum_candidate_configurations"] == 3
    assert budget["maximum_execution_rounds"] == 2
    assert budget["maximum_attempts_per_case_per_candidate_per_round"] == 3
    assert budget["maximum_fresh_generation_calls"] == 360
    assert len(budget["round_master_seeds"]) == 2
    held = record["held_out_confirmation"]
    assert held["single_fresh_run"] is True and held["no_revision_after_start"] is True
    assert held["exact_minimum_accepted"] == {
        "overall:30": 24,
        "ansible_docs:add_specific:4": 3,
        "ansible_docs:de_specify:6": 5,
        "ansible_docs:irrelevant_rewrite:5": 4,
        "github_docs:add_specific:5": 4,
        "github_docs:de_specify:5": 4,
        "github_docs:irrelevant_rewrite:5": 4,
    }
    assert "never be printed" in record["privacy_and_blinding"]["held_out_text_access"]


def test_freeze_record_binds_preinspection_commit() -> None:
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    assert freeze["split_freeze_commit"] == "bca966334e4685cd354710676167991c0e69c785"
    assert hashlib.sha256(CONFIG.read_bytes()).hexdigest() == freeze["split_config_sha256"]
    manifest = ROOT / freeze["split_manifest_path"]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == freeze["split_manifest_sha256"]
    assert freeze["frozen_before_case_level_qe_a_text_or_acceptance_inspection"] is True
    assert freeze["frozen_before_development_positive_control_analysis"] is True
    assert freeze["frozen_before_fresh_development_generation"] is True


def test_v1_diagnosis_is_development_only_score_blind_and_text_free() -> None:
    metadata = diagnose_v1(CONFIG)
    assert metadata["development_cases"] == 30
    assert metadata["held_out_cases_not_exported_or_summarized"] == 30
    assert metadata["prohibited_inputs_used"] is False
    compact = ROOT / "analysis" / "round2_qwen_edit_remediation"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in compact.iterdir() if path.is_file())
    assert not any(field in combined for field in ("sentence_original", "sentence_edited", "edited_sentence"))
    assert "speciteller_score" not in combined.casefold()
    assert "ko_score" not in combined.casefold()
    assert "granuscore_score" not in combined.casefold()
    for name, expected in metadata["outputs"].items():
        assert hashlib.sha256((compact / name).read_bytes()).hexdigest() == expected


def test_candidate_budget_profiles_and_prompt_blindness() -> None:
    candidate, base, cases = _load_candidate_context(CANDIDATES)
    assert candidate["candidate_order"] == ["c1_prompt_only", "c2_proxy_aligned", "c3_micro_edit"]
    assert candidate["round"] == 1 and candidate["maximum_attempts_per_case"] == 3
    assert len(cases) == 30
    case = next(value for value in cases.values() if value["edit_type"] == "add_specific")
    prompt = _candidate_prompt(candidate, base, "c3_micro_edit", case)
    assert case["sentence_original"] in prompt
    assert "sentence_edited" not in prompt
    assert "score" not in prompt.casefold()
    freeze = json.loads(CANDIDATE_FREEZE.read_text(encoding="utf-8"))
    assert freeze["candidate_freeze_commit"] == "c1eb615132875b9a04987720f397dabf0fd65bd7"
    assert hashlib.sha256(CANDIDATES.read_bytes()).hexdigest() == freeze["candidate_config_sha256"]
    assert freeze["frozen_before_fresh_development_generation"] is True


def test_proxy_aligned_gate_fixes_declared_mismatches_but_retains_safeguards() -> None:
    candidate, base, _ = _load_candidate_context(CANDIDATES)
    original = "Dependencies for published collections must be other published collections."
    edited = "Dependencies for certain collections must be other collections."
    v1_reasons, _ = _candidate_gate(candidate, base, "c1_prompt_only", original, edited, "de_specify")
    v2_reasons, _ = _candidate_gate(candidate, base, "c2_proxy_aligned", original, edited, "de_specify")
    assert "insufficient_despecification_proxy" in v1_reasons
    assert v2_reasons == []
    reasons, _ = _candidate_gate(
        candidate,
        base,
        "c2_proxy_aligned",
        "This command can process files.",
        "This command is able to åˆ†æ•£ files.",
        "irrelevant_rewrite",
    )
    assert "new_non_latin_script" in reasons
    reasons, _ = _candidate_gate(candidate, base, "c2_proxy_aligned", "You can run it.", "You should run it.", "irrelevant_rewrite")
    assert "modality_changed" in reasons


def test_v2_protocol_is_complete_and_matches_selected_candidate() -> None:
    protocol = json.loads(V2.read_text(encoding="utf-8"))
    base = json.loads((ROOT / protocol["inputs"]["qe_a_config_path"]).read_text(encoding="utf-8"))
    assert protocol["selection_provenance"]["selected_candidate_id"] == "c3_micro_edit"
    assert protocol["runtime"]["master_seed"] == 2026081099
    assert protocol["runtime"]["maximum_attempts_per_case"] == 3
    assert protocol["held_out_gate"]["exact_minimum_accepted"]["overall:30"] == 24
    assert protocol["automated_gates"]["direction_rules"]["irrelevant_rewrite"]["original_content_anchor_recall_min"] == 0.6
    assert protocol["automated_gates"]["direction_rules"]["de_specify"]["edited_to_original_token_ratio_max"] == 1.1
    reasons, _ = _v2_gate(
        protocol,
        base,
        "Dependencies for published collections must be other published collections.",
        "Dependencies for certain collections must be other collections.",
        "de_specify",
    )
    assert reasons == []
    schema = json.loads(V2_ATTEMPT_SCHEMA.read_text(encoding="utf-8"))
    assert "edited_sentence" in schema["required"]
    assert schema["properties"]["attempt_index"]["maximum"] == 3
    freeze = json.loads(V2_FREEZE.read_text(encoding="utf-8"))
    assert freeze["v2_freeze_commit"] == "d08b6678d28478c229bf7f429720cc8e675586d2"
    assert hashlib.sha256(V2.read_bytes()).hexdigest() == freeze["v2_config_sha256"]
    assert hashlib.sha256(V2_ATTEMPT_SCHEMA.read_bytes()).hexdigest() == freeze["attempt_schema_sha256"]
    assert freeze["frozen_before_held_out_generation"] is True
    assert freeze["held_out_text_inspected_before_freeze"] is False
