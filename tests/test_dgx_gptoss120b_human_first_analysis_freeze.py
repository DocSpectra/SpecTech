import json
from pathlib import Path

from src.analysis.dgx_gptoss120b_human_first_analysis import validate_completed_workbook


CONFIG = Path("configs/round2_dgx_gptoss120b_human_first_analysis_v1.json")
SCHEMA = Path("schemas/round2_dgx_gptoss120b_human_first_analysis_v1.schema.json")


def test_phase_d_config_schema_and_required_estimators():
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    assert record["schema_version"] == schema["properties"]["schema_version"]["const"]
    assert set(schema["required"]) <= set(record)
    assert record["outcome_blind_method_freeze"] is True
    assert record["uncertainty"]["replicates"] == 20000
    assert record["uncertainty"]["unit"] == "shared source case"
    assert record["policies"]["primary"] == [
        "unguided_slot01", "speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "granuscore_direction_aligned"
    ]
    assert record["policies"]["secondary"] == ["ko_three_run_arithmetic_mean_secondary", "rank_consensus_secondary"]
    assert record["generator_comparison"]["shared_source_cases"] == 60
    assert record["analysis_sets"]["primary"]["name"] == "all_180"
    assert [x["name"] for x in record["analysis_sets"]["order_sensitivity"]] == ["exclude_first_20", "exclude_first_30"]


def test_completed_workbook_gate_passes_without_hidden_inputs():
    result = validate_completed_workbook(CONFIG)
    assert result["review_rows"] == 180
    assert result["unique_review_ids"] == 180
    assert result["immutable_review_columns_equal"] is True
    assert result["instructions_equal"] is True
    assert result["score_counts"] == {"1": 36, "2": 31, "3": 43, "4": 34, "5": 35, "X": 1}
    assert result["clerical_corrections"] == []
