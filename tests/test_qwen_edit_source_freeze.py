"""Outcome-blind freeze contract for QE-A."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_qwen_edit_source_v1.json"
FREEZE_RECORD = ROOT / "configs" / "round2_qwen_edit_source_freeze_record.json"


def _record() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(corpus_id: str, sent_id: str, edit_type: str, position: int) -> str:
    value = f"{corpus_id}\0{sent_id}\0{edit_type}\0{position}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def test_protocol_is_authorized_outcome_blind_and_narrow() -> None:
    record = _record()
    assert record["schema_version"] == "round2_qwen_edit_source_v1"
    assert record["outcome_blind_freeze"] is True
    auth = record["authorization"]
    assert auth["qe_a_authorized"] is True
    assert auth["existing_controlled_edit_originals_only"] is True
    assert auth["manual_editing_or_selection"] is False
    assert auth["wikipedia_synthetic_arm"] is False
    assert auth["scores_labels_or_author_edits_visible_to_generation"] is False
    assert auth["project_scorer_outcomes_inspected_before_freeze"] is False


def test_complete_input_hash_order_and_direction_counts() -> None:
    record = _record()
    digest = hashlib.sha256()
    observed_counts: dict[tuple[str, str], int] = {}
    total = 0
    for corpus_id in record["inputs"]["corpus_order"]:
        entry = record["inputs"]["source_files"][corpus_id]
        path = ROOT / entry["path"]
        assert _sha256(path) == entry["sha256"]
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
        assert len(rows) == entry["rows"]
        for position, row in enumerate(rows, start=1):
            assert row["corpus_id"] == corpus_id
            identifier = _case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            original_hash = hashlib.sha256(row["sentence_original"].encode("utf-8")).hexdigest()
            value = "\0".join([
                corpus_id, str(position), identifier, row["sent_id"],
                row["edit_type"], original_hash,
            ])
            digest.update(value.encode("utf-8"))
            digest.update(b"\n")
            key = (corpus_id, row["edit_type"])
            observed_counts[key] = observed_counts.get(key, 0) + 1
            total += 1
        assert {
            edit_type: observed_counts[(corpus_id, edit_type)]
            for edit_type in entry["direction_counts"]
        } == entry["direction_counts"]
    assert total == record["inputs"]["required_total_cases"] == 60
    assert digest.hexdigest() == record["inputs"]["ordered_case_sha256"]


def test_qwen_identity_decoding_attempts_and_prompts_are_complete() -> None:
    record = _record()
    qwen = record["qwen"]
    assert qwen["model"] == "qwen3:14b"
    assert qwen["model_list_id"] == "bdbd181c33f2"
    assert qwen["backing_blob_sha256"] == "a8cc1361f3145dc01f6d77c6c82c9116b9ffe3c97b34716fe20418455876c40e"
    assert qwen["thinking"] is False and qwen["stream"] is False
    assert qwen["maximum_attempts_per_case"] == 3
    assert qwen["fresh_conversation_per_attempt"] is True
    assert qwen["master_seed"] == 20260810
    assert set(record["prompts"]["user_templates"]) == {
        "add_specific", "de_specify", "irrelevant_rewrite",
    }
    combined = json.dumps({
        "system": record["prompts"]["system"],
        "user_templates": record["prompts"]["user_templates"],
    }).casefold()
    for forbidden in ("speciteller score", "ko score", "granuscore score", "human label", "author edit"):
        assert forbidden not in combined


def test_gates_scoring_analysis_and_claim_boundaries_are_fixed() -> None:
    record = _record()
    gates = record["automated_gates"]
    assert gates["coverage_gate"]["minimum_total_acceptance_rate_for_primary_comparison"] == 0.8
    assert gates["coverage_gate"]["minimum_cell_acceptance_rate_for_primary_comparison"] == 0.75
    assert set(gates["direction_rules"]) == {
        "add_specific", "de_specify", "irrelevant_rewrite",
    }
    assert record["scoring"]["score_order"] == [
        "speciteller", "ko_run01", "ko_run02", "ko_run03", "granuscore",
    ]
    assert record["scoring"]["ko"]["primary_instances"] == ["run01", "run02", "run03"]
    assert record["scoring"]["ko"]["secondary_aggregate"].startswith("rowwise arithmetic mean")
    assert record["scoring"]["granuscore"]["direction"] == "higher_is_coarser_more_abstract"
    assert record["scoring"]["qwen_rubric_self_judgment"].startswith("not run")
    analysis = record["analysis"]
    assert analysis["bootstrap"]["replicates"] == 10000
    assert analysis["bootstrap"]["master_seed"] == 20260810
    assert "never compare raw deltas across" in analysis["scale_rule"]
    assert "not human-validated" in analysis["claim_boundary"]


def test_all_frozen_artifact_paths_are_portable_and_hashes_are_well_formed() -> None:
    record = _record()
    paths: list[str] = []
    hashes: list[str] = []
    for entry in record["inputs"]["source_files"].values():
        paths.append(entry["path"]); hashes.append(entry["sha256"])
    for entry in record["scoring"]["speciteller"]["original_score_inputs"].values():
        paths.append(entry["path"]); hashes.append(entry["sha256"])
    for corpus in record["scoring"]["ko"]["runs"].values():
        for run in corpus.values():
            paths.extend([run["checkpoint_path"], run["original_score_path"]])
            hashes.extend([run["checkpoint_sha256"], run["original_score_sha256"]])
    for key in ("author_pair_manifest", "author_pair_scores"):
        entry = record["scoring"]["granuscore"][key]
        paths.append(entry["path"]); hashes.append(entry["sha256"])
    assert all(not Path(path).is_absolute() and ":/" not in path for path in paths)
    assert all(len(value) == 64 and set(value) <= set("0123456789abcdef") for value in hashes)


def test_freeze_record_binds_the_pre_generation_method_commit() -> None:
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    assert freeze["method_freeze_commit"] == "f7030b051a0c6753f8b4071677e9ce0ad724854e"
    assert freeze["config_sha256"] == _sha256(CONFIG)
    assert freeze["method_frozen_before_any_project_generation"] is True
    assert freeze["method_frozen_before_any_new_project_scorer_output"] is True
    assert freeze["qwen_generated_text_or_scorer_outcomes_inspected_before_freeze"] is False
    correction = freeze["outcome_blind_mechanical_correction"]
    assert correction["mechanical_fix_commit"] == "f324f9655e62374bc68a67e89aefc6f1649d7f49"
    assert correction["project_generated_text_before_fix"] is False
    assert correction["new_project_scorer_output_before_fix"] is False
    assert correction["pre_fix_attempt_rows"] == 180
    assert correction["pre_fix_attempts_sha256"] == "047d2dd9ce8a887864abc4ba2722d5e8405a36c20dd8df45a6231e2870d12ae9"
    assert "Removed only unsupported JSON-Schema" in correction["scope"]
    gs_fix = freeze["outcome_blind_granuscore_wrapper_correction"]
    assert gs_fix["granuscore_score_rows_before_fix"] == 0
    assert gs_fix["scientific_or_inclusion_rule_changed"] is False
