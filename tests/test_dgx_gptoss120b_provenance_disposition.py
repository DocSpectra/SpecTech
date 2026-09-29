from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/dgx_spark_gptoss120b"
sys.path.insert(0, str(DEPLOY))

import packet_core
import provenance_disposition


CONFIG = ROOT / "configs/round2_dgx_gptoss120b_provenance_disposition_v1.json"
SCHEMA = ROOT / "schemas/dgx_gptoss120b/provenance_disposition.schema.json"
AUDIT = ROOT / "analysis/round2_dgx_gptoss120b_return_run1/audit_summary.json"


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_waiver_config_schema_and_exact_bindings() -> None:
    config = _json(CONFIG)
    packet_core.validate_schema(config, _json(SCHEMA))
    assert config["authorization"]["explicit_author_waiver"] is True
    assert config["bindings"]["original_audit_sha256"] == provenance_disposition.sha_file(AUDIT)
    assert config["bindings"]["run_id"] == "dgx-5411bb15be9f"


def test_explicit_waiver_accepts_without_claiming_missing_checks_passed() -> None:
    audit, config = _json(AUDIT), _json(CONFIG)
    result = provenance_disposition.derive_disposition(
        audit, config, audit_sha256=provenance_disposition.sha_file(AUDIT)
    )
    assert result["disposition"] == provenance_disposition.WAIVER_DISPOSITION
    assert result["interpretation"] == {
        "evidence_of_tampering": False,
        "audit_completeness": "incomplete",
        "missing_checks_claimed_as_passed": False,
        "outputs_may_be_replaced_or_changed": False,
        "authorized_scope": config["authorization"]["authorized_scope"],
    }
    assert result["original_audit"]["preserved_unchanged"] is True


def test_no_waiver_and_no_supplement_preserves_withholding() -> None:
    audit, config = _json(AUDIT), _json(CONFIG)
    config["authorization"]["explicit_author_waiver"] = False
    result = provenance_disposition.derive_disposition(
        audit, config, audit_sha256=provenance_disposition.sha_file(AUDIT)
    )
    assert result["disposition"] == provenance_disposition.WITHHELD_DISPOSITION


def test_passing_no_rerun_supplement_path_takes_precedence() -> None:
    audit, config = _json(AUDIT), _json(CONFIG)
    config["authorization"]["explicit_author_waiver"] = False
    supplement = {
        "run_id": config["bindings"]["run_id"],
        "status": "passed",
        "resolved_blockers": copy.deepcopy(audit["blockers"]),
        "outputs_unchanged": True,
    }
    result = provenance_disposition.derive_disposition(
        audit, config, audit_sha256=provenance_disposition.sha_file(AUDIT), supplement=supplement
    )
    assert result["disposition"] == provenance_disposition.SUPPLEMENT_DISPOSITION
    assert result["interpretation"]["audit_completeness"] == "supplemented"


@pytest.mark.parametrize("field", ["run_id", "return_archive_sha256", "source_packet_sha256"])
def test_binding_drift_is_rejected(field: str) -> None:
    audit, config = _json(AUDIT), _json(CONFIG)
    config["bindings"][field] = "0" * 64 if field != "run_id" else "different-run"
    with pytest.raises(ValueError, match="mismatch"):
        provenance_disposition.derive_disposition(
            audit, config, audit_sha256=provenance_disposition.sha_file(AUDIT)
        )


def test_derived_summary_is_content_free_and_deterministic() -> None:
    audit, config = _json(AUDIT), _json(CONFIG)
    one = provenance_disposition.derive_disposition(
        audit, config, audit_sha256=provenance_disposition.sha_file(AUDIT)
    )
    two = provenance_disposition.derive_disposition(
        audit, config, audit_sha256=provenance_disposition.sha_file(AUDIT)
    )
    assert one == two
    text = json.dumps(one)
    assert "candidate_text" not in text
    assert "score_1_to_5" not in text
