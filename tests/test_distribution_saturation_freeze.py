"""Outcome-blind contract tests for Round 2 Pair 4A."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.analysis.distribution_saturation import MODEL_ORDER, PRIMARY_FIGURE_MODELS, load_distribution_config


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "round2_distribution_saturation_v1.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_contract_is_outcome_blind_and_scope_is_exact() -> None:
    config = load_distribution_config(CONFIG)
    assert config["authorization"]["new_distribution_outcomes_inspected_before_freeze"] is False
    assert config["authorization"]["qwen_rubric_excluded_as_pilot_only"] is True
    assert config["authorization"]["qwen_edit_arm_excluded"] is True
    assert config["authorization"]["vago_excluded_by_failed_release_gate"] is True
    assert tuple(item["model_instance_id"] for item in config["model_instances"]) == MODEL_ORDER
    assert tuple(config["histogram"]["included_models"]) == PRIMARY_FIGURE_MODELS
    assert "secondary" in config["comparison_and_claim_rules"]["ko_mean_status"]


def test_every_dependency_and_score_input_is_hash_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    records = list(config["dependency_contracts"].values())
    records += [item for inputs in config["row_inputs"].values() for item in (inputs["sentences"], inputs["features"])]
    records += list(config["score_inputs"]["speciteller"].values())
    records += [item for runs in config["score_inputs"]["ko_runs"].values() for item in runs.values()]
    records += list(config["score_inputs"]["granuscore"].values())
    assert len(records) == 8 + 8 + 4 + 12 + 4
    for item in records:
        assert _sha(ROOT / item["path"]) == item["sha256"]


def test_supported_bounds_directions_and_no_raw_scale_comparison_are_frozen() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for model in config["model_instances"]:
        if model["model_instance_id"] == "granuscore_native":
            assert (model["minimum"], model["maximum"]) == (0.0, 100.0)
            assert model["direction"] == "higher_is_coarser_more_abstract"
        else:
            assert (model["minimum"], model["maximum"]) == (0.0, 1.0)
            assert model["direction"] == "higher_is_more_specific"
    assert config["statistics"]["raw_cross_model_value_comparison"] is False


def test_statistics_boundary_bins_bootstrap_and_stop_rules_are_complete() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    stats = config["statistics"]
    assert stats["quantiles"] == [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    assert stats["effective_dynamic_range"].startswith("p95_minus_p05")
    assert config["boundary_concentration"]["fractions"] == [0.01, 0.05, 0.10]
    assert config["boundary_concentration"]["primary_fraction"] == 0.05
    assert config["histogram"]["bins"] == 50
    assert config["bootstrap"]["replicates"] == 1000
    assert config["bootstrap"]["cluster_key"] == "doc_path"
    assert config["bootstrap"]["minimum_rows_per_document_for_span"] == 20
    assert any("denominator" in item for item in config["stop_conditions"])
    assert any("bound" in item for item in config["stop_conditions"])


def test_schema_and_spec_are_present_and_portable() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    schema_path = ROOT / config["outputs"]["metadata_schema"]
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema["properties"]["schema_version"]["const"] == "round2_distribution_saturation_results_v1"
    spec = (ROOT / "specs" / "round2_distribution_saturation.md").read_text(encoding="utf-8")
    assert "committed\nbefore any new distribution summary" in spec
    assert "latent specificity truth" in spec
    assert "C:\\Users" not in CONFIG.read_text(encoding="utf-8")
    assert "C:\\Users" not in spec
