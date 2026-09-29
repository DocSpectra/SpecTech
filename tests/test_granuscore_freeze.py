"""Outcome-blind contract tests for the GranuScore sprint."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_granuscore_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_granuscore_contract_is_outcome_blind_and_direction_explicit() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["outcome_blind_freeze"] is True
    assert config["authorization"]["project_granuscore_outcomes_inspected_before_freeze"] is False
    assert config["identity"]["score_direction"] == "higher_is_coarser_more_abstract"
    assert "not identical" in config["identity"]["construct_boundary"]
    assert config["full_analysis"]["accuracy_boundary"].startswith("without gold labels")


def test_official_reproduction_and_feasibility_gates_pass() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["reproduction_gate"]["passed"] is True
    assert config["reproduction_gate"]["maximum_absolute_error"] <= config["reproduction_gate"]["tolerance"]
    assert config["feasibility_gate"]["passed"] is True
    assert config["feasibility_gate"]["rows_per_second"] >= config["feasibility_gate"]["minimum_rows_per_second_for_full_run"]


def test_all_canonical_inputs_are_hash_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    inputs = config["full_corpora"]["inputs"]
    assert list(inputs) == config["full_corpora"]["canonical_order"]
    assert sum(item["rows"] for item in inputs.values()) == config["full_corpora"]["required_total_rows"]
    for item in inputs.values():
        assert _sha256(ROOT / item["path"]) == item["sha256"]


def test_existing_pilot_and_edit_inputs_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    pilot = config["pilot"]
    assert _sha256(ROOT / pilot["manifest"]) == pilot["manifest_sha256"]
    for item in config["controlled_edits"]["inputs"].values():
        assert _sha256(ROOT / item["path"]) == item["sha256"]


def test_runtime_and_official_artifacts_are_pinned() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert "@sha256:" in config["runtime"]["base_image"]
    assert len(config["identity"]["model_revision"]) == 40
    assert len(config["identity"]["package_wheel_sha256"]) == 64
    assert all(len(value) == 64 for value in config["official_artifacts"].values() if value != "v1.0.0")
    assert _sha256(ROOT / "granuscore_container" / "score_csv.py") == config["runtime"]["runner_sha256"]
    record = json.loads((ROOT / config["runtime"]["mechanical_freeze_record"]).read_text(encoding="utf-8"))
    assert record["scientific_method_changed"] is False
    assert record["fix"]["final_image_id"] == config["runtime"]["release_image_id"]
    assert record["fix"]["runner_sha256"] == config["runtime"]["runner_sha256"]


def test_no_unit_rows_are_retained_and_sensitivity_declared() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["scoring"]["no_referential_unit_rule"].startswith("retain")
    assert "excluding" in config["full_analysis"]["no_unit_sensitivity"]


def test_container_runner_accepts_canonical_large_fields() -> None:
    source = (ROOT / "granuscore_container" / "score_csv.py").read_text(encoding="utf-8")
    assert "csv.field_size_limit(sys.maxsize)" in source
    assert '"runner_sha256"' in source
