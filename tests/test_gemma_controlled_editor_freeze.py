"""Outcome-blind freeze contract for the Gemma controlled-editor pilot."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_gemma_controlled_editor_v1.json"
ATTEMPT_SCHEMA = ROOT / "schemas" / "round2_gemma_editor_attempt_v1.schema.json"
REVIEW_SCHEMA = ROOT / "schemas" / "round2_gemma_blinded_review_v1.schema.json"
FREEZE_RECORD = ROOT / "configs" / "round2_gemma_controlled_editor_freeze_record.json"


def _record() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(corpus_id: str, sent_id: str, edit_type: str, position: int) -> str:
    value = f"{corpus_id}\0{sent_id}\0{edit_type}\0{position}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _review_id(record: dict, case_id: str) -> str:
    review = record["blinded_review"]
    value = f"{review['review_id_seed']}:{review['review_id_namespace']}:{case_id}".encode("utf-8")
    encoded = base64.b32encode(hashlib.sha256(value).digest()).decode("ascii").rstrip("=")
    return "GCE-" + encoded[:12]


def _cases(record: dict) -> list[dict]:
    result: list[dict] = []
    digest = hashlib.sha256()
    for corpus_id in record["inputs"]["corpus_order"]:
        entry = record["inputs"]["source_files"][corpus_id]
        path = ROOT / entry["path"]
        assert _sha(path) == entry["sha256"]
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == entry["rows"]
        for position, row in enumerate(rows, start=1):
            case_id = _case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            original_hash = hashlib.sha256(row["sentence_original"].encode("utf-8")).hexdigest()
            value = "\0".join([
                corpus_id, str(position), case_id, row["sent_id"], row["edit_type"], original_hash,
            ])
            digest.update(value.encode("utf-8")); digest.update(b"\n")
            result.append({"corpus_id": corpus_id, "edit_type": row["edit_type"], "case_id": case_id})
    assert digest.hexdigest() == record["inputs"]["ordered_case_sha256"]
    return result


def test_authorization_and_outcome_isolation_are_narrow() -> None:
    record = _record()
    assert record["schema_version"] == "round2_gemma_controlled_editor_v1"
    assert record["outcome_blind_method_freeze"] is True
    auth = record["authorization"]
    assert auth["required_cases"] == 60
    assert auth["new_source_sampling"] is False
    assert auth["expansion_to_180"] is False
    assert auth["paper_integration"] is False
    assert auth["rubric_study"] is False
    assert auth["scorer_invocation"] is False
    assert auth["manual_generation_selection_or_repair"] is False
    prohibited = " ".join(record["isolation"]["prohibited_method_or_generation_inputs"]).casefold()
    for phrase in ("qwen generated text", "scorer outcome", "human pilot labels", "author-edited sentence"):
        assert phrase in prohibited


def test_frozen_60_case_identity_and_cells_are_exact() -> None:
    record = _record()
    cases = _cases(record)
    assert len(cases) == len({row["case_id"] for row in cases}) == 60
    observed = Counter((row["corpus_id"], row["edit_type"]) for row in cases)
    expected = {
        tuple(key.split(":")): value
        for key, value in record["generation_gate"]["required_exact_cell_counts"].items()
    }
    assert observed == Counter(expected)


def test_exact_gemma_identity_decoding_and_residency_are_frozen() -> None:
    runtime = _record()["runtime"]
    assert runtime["ollama_version"] == "0.32.5"
    assert runtime["model"] == "gemma4:12b"
    assert runtime["model_tag_digest"] == "4eb23ef187e2c5462566d6a1d3bbbc2f1346d0b4327cbb66d58fffbcc9b2b05c"
    assert runtime["model_blob_sha256"] == "1278394b693672ac2799eadc9a83fd98259a6a88a40acfb1dcaa6c6fc895a606"
    assert runtime["projector_blob_sha256"] == "675ad6e68101ca9413ec806855c452362f0213f2dfc5800996b086fdb8119842"
    assert runtime["thinking"] is False and runtime["stream"] is False
    assert runtime["maximum_attempts_per_case"] == 3
    assert runtime["fresh_conversation_per_attempt"] is True
    assert runtime["options"] == {
        "temperature": 1.0, "top_k": 64, "top_p": 0.95,
        "repeat_penalty": 1.0, "num_ctx": 4096, "num_predict": 256,
    }
    assert runtime["residency_control"]["preserve_installed_model"] == "qwen3:14b"
    assert "exactly gemma4:12b" in runtime["residency_control"]["before_run"]
    assert "ollama stop gemma4:12b" in runtime["residency_control"]["after_run"]


def test_prompt_schema_gates_and_generation_coverage_are_complete() -> None:
    record = _record()
    assert set(record["prompts"]["user_templates"]) == {
        "add_specific", "de_specify", "irrelevant_rewrite",
    }
    prompt = json.dumps(record["prompts"]).casefold()
    for forbidden in ("speciteller score", "ko score", "granuscore score", "human label"):
        assert forbidden not in prompt
    gates = record["automated_gates"]
    assert set(gates["direction_rules"]) == {
        "add_specific", "de_specify", "irrelevant_rewrite",
    }
    assert gates["universal"]["polarity_token_multiset_must_match"] is True
    assert gates["universal"]["modal_token_multiset_must_match"] is True
    assert record["generation_gate"]["required_retained_total"] == 60
    attempt = json.loads(ATTEMPT_SCHEMA.read_text(encoding="utf-8"))
    assert attempt["properties"]["model"]["const"] == "gemma4:12b"
    assert attempt["properties"]["attempt_index"]["maximum"] == 3


def test_review_ids_side_assignment_and_row_order_are_stable_and_balanced() -> None:
    record = _record()
    cases = _cases(record)
    review = record["blinded_review"]
    ids = [_review_id(record, row["case_id"]) for row in cases]
    assert len(ids) == len(set(ids)) == 60
    by_cell: dict[tuple[str, str], list[dict]] = {}
    for row in cases:
        by_cell.setdefault((row["corpus_id"], row["edit_type"]), []).append(row)
    sides: Counter[tuple[str, str, str]] = Counter()
    for cell, rows in by_cell.items():
        ranked = sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{review['side_assignment_seed']}:{row['case_id']}".encode("utf-8")
            ).hexdigest(),
        )
        for index, row in enumerate(ranked):
            sides[cell + ("A" if index < len(ranked) // 2 else "B",)] += 1
    for cell, rows in by_cell.items():
        assert sides[cell + ("A",)] == sides[cell + ("B",)] == len(rows) // 2
    assert sum(value for key, value in sides.items() if key[2] == "A") == 30
    order = sorted(
        ids,
        key=lambda value: hashlib.sha256(f"{review['row_order_seed']}:{value}".encode("utf-8")).hexdigest(),
    )
    assert len(order) == len(set(order)) == 60


def test_blinded_review_schema_and_human_gate_do_not_leak_answers() -> None:
    record = _record()
    schema = json.loads(REVIEW_SCHEMA.read_text(encoding="utf-8"))
    fields = set(schema["properties"])
    assert fields == set(record["blinded_review"]["packet_fields"])
    assert not fields.intersection(record["blinded_review"]["packet_forbidden_fields"])
    assert schema["properties"]["more_specific"]["enum"] == ["A", "B", "tie", "uncertain"]
    gate = record["human_gate"]
    assert gate["minimum_overall_passed"] == 48
    assert gate["exact_minimum_passed_by_hidden_cell"] == {
        "ansible_docs:add_specific:8": 6,
        "ansible_docs:de_specify:12": 9,
        "ansible_docs:irrelevant_rewrite:10": 8,
        "github_docs:add_specific:10": 8,
        "github_docs:de_specify:10": 8,
        "github_docs:irrelevant_rewrite:10": 8,
    }
    assert "does not itself authorize expansion" in gate["pass_disposition"]


def test_paths_are_portable_and_scoring_is_absent() -> None:
    record = _record()
    paths = [entry["path"] for entry in record["inputs"]["source_files"].values()]
    paths.extend([
        record["blinded_review"]["review_schema"],
        record["outputs"]["raw_directory"],
        record["outputs"]["compact_directory"],
    ])
    assert all(not Path(path).is_absolute() and ":/" not in path for path in paths)
    text = CONFIG.read_text(encoding="utf-8").casefold()
    assert '"scoring"' not in text
    assert '"analysis"' not in text
    assert "outputs/round2/qwen_edit" not in text


def test_freeze_record_binds_pre_generation_method_commit() -> None:
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    assert freeze["method_freeze_commit"] == "4597b5264125020735c7126ab7af32033e1aa74d"
    assert freeze["config_sha256"] == _sha(CONFIG)
    assert freeze["attempt_schema_sha256"] == _sha(ATTEMPT_SCHEMA)
    assert freeze["review_schema_sha256"] == _sha(REVIEW_SCHEMA)
    assert freeze["method_frozen_before_any_project_gemma_generation"] is True
    assert freeze["project_gemma_text_or_outcome_inspected_before_freeze"] is False
    assert freeze["qwen_output_score_label_or_scorer_outcome_used_for_freeze"] is False
    assert freeze["author_edited_text_used_for_prompt_or_gate_selection"] is False
    assert freeze["model_installation"]["qwen3_14b_preserved"] is True
    assert freeze["model_installation"]["models_resident_after_smoke_and_before_freeze_commit"] == []
    smoke = freeze["non_project_smoke"]
    assert smoke["project_inputs_used"] is False
    assert smoke["structured_json_passed"] == smoke["thinking_disabled_passed"] == 3
