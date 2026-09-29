from __future__ import annotations

import copy
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/dgx_spark_gptoss120b"
sys.path.insert(0, str(DEPLOY))

import audit_run_supplement
import packet_core


CONFIG = ROOT / "configs/round2_dgx_gptoss120b_supplement_ingestion_v1.json"
SCHEMA = ROOT / "schemas/dgx_gptoss120b/supplement_ingestion.schema.json"


def test_supplement_config_is_strict_and_content_blind() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    packet_core.validate_schema(config, json.loads(SCHEMA.read_text(encoding="utf-8")))
    assert config["content_blind"] is True
    assert config["no_model_execution"] is True
    assert config["acceptance_rule"]["disposition_on_pass"] == "accepted_with_author_provenance_waiver"
    assert config["acceptance_rule"]["supplement_status_on_pass"] == "partial_provenance_recovery"
    assert len(config["required_members"]) == len(set(config["required_members"])) == 14


def _write_zip(path: Path, members: list[tuple[str, bytes, int]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data, mode in members:
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = mode << 16
            archive.writestr(info, data)


def test_zip_inspector_rejects_traversal(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    _write_zip(path, [("run/../escape.txt", b"x", 0o100600)])
    with pytest.raises(ValueError, match="unsafe ZIP path"):
        audit_run_supplement.inspect_zip(path, "run")


def test_zip_inspector_rejects_symlink(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    _write_zip(path, [("run/link", b"target", 0o120777)])
    with pytest.raises(ValueError, match="link, device"):
        audit_run_supplement.inspect_zip(path, "run")


def test_zip_inspector_rejects_case_collision(tmp_path: Path) -> None:
    path = tmp_path / "bad.zip"
    _write_zip(path, [("run/A.txt", b"a", 0o100600), ("run/a.txt", b"b", 0o100600)])
    with pytest.raises(ValueError, match="case-colliding"):
        audit_run_supplement.inspect_zip(path, "run")


def test_safe_extraction_discards_archive_modes(tmp_path: Path) -> None:
    path = tmp_path / "safe.zip"
    _write_zip(path, [("run/evidence.json", b"{}", 0o100664)])
    archive, files, summary = audit_run_supplement.inspect_zip(path, "run")
    destination = tmp_path / "validated"
    audit_run_supplement.safe_extract(archive, files, destination)
    archive.close()
    extracted = destination / "run/evidence.json"
    assert extracted.read_bytes() == b"{}"
    assert summary["source_group_or_world_writable_modes_discarded_on_extraction"] == 1
    assert extracted.stat().st_mode & 0o111 == 0


def test_summary_writer_tracks_only_compact_evidence(tmp_path: Path) -> None:
    summary = {
        "run_id": "dgx-5411bb15be9f",
        "disposition": "accepted_with_author_provenance_waiver",
        "supplement": {"sha256": "1" * 64},
        "preservation": {"original_return_sha256": "2" * 64, "original_audit_sha256": "3" * 64, "prior_acceptance_sha256": "4" * 64},
        "remaining_limitations": ["Artifact bytes were unavailable."],
        "derived_chronology": {"derived_full_run_completed_at_utc": "2026-08-12T20:04:16.000Z"},
    }
    audit_run_supplement.write_outputs(tmp_path, summary, CONFIG, SCHEMA)
    text = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.iterdir())
    assert "candidate_text" not in text
    assert "score_1_to_5" not in text
    assert "accepted_with_author_provenance_waiver" in text


@pytest.mark.parametrize(
    ("mutated", "message"),
    [
        ({"case_id": "wrong", "candidate_slot": 1}, "case/slot mismatch"),
        ({"case_id": "case-1", "candidate_slot": 2}, "case/slot mismatch"),
    ],
)
def test_attempt_identity_rejects_case_or_slot_mutation(mutated: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        audit_run_supplement.require_expected_case_slot(mutated, "case-1", 1, "attempt")


def test_retained_metadata_mutation_is_rejected() -> None:
    attempt = {
        "candidate_id": "a" * 64,
        "candidate_slot": 1,
        "attempt_index": 1,
        "seed": 17,
        "candidate_text": "Invented candidate.",
        "candidate_text_sha256": "b" * 64,
    }
    case = {"case_id": "case-1", "source_text_sha256": "c" * 64}
    kept = {
        "run_id": "run-1", "case_id": "case-1", "candidate_id": "a" * 64,
        "candidate_slot": "1", "retained_attempt": "1", "seed": "17",
        "source_text_sha256": "c" * 64, "candidate_text": "Invented candidate.",
        "candidate_text_sha256": "b" * 64, "model_identity": "model-1",
        "config_sha256": "d" * 64, "freeze_id": "e" * 40,
    }
    mutated = copy.deepcopy(kept)
    mutated["source_text_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="retained metadata mismatch: source_text_sha256"):
        audit_run_supplement.validate_retained_metadata(
            mutated, attempt, case, run_id="run-1", config_sha256="d" * 64,
            freeze_id="e" * 40, model_identity="model-1",
        )
