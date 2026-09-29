import json
from pathlib import Path

from src.analysis.dgx_gptoss120b_human_first_analysis import sha256_file, validate_completed_workbook


FREEZE = Path("configs/round2_dgx_gptoss120b_human_first_analysis_freeze_record.json")


def test_phase_d_freeze_bindings_and_workbook_gate():
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    for key in ("method_config", "schema", "method_spec", "freeze_test"):
        item = freeze[key]
        assert sha256_file(Path(item["path"])) == item["sha256"]
    gate = validate_completed_workbook(Path(freeze["method_config"]["path"]))
    assert gate["completed_workbook_sha256"] == freeze["pre_unblinding_gate"]["completed_workbook_sha256"]
    assert gate["score_counts"] == freeze["pre_unblinding_gate"]["score_counts"]
    assert freeze["chronology"]["analysis_method_frozen_before_unblinding"] is True
    assert not any(value for key, value in freeze["chronology"].items() if key.endswith("opened_before_binding_commit"))
