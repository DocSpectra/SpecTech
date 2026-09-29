"""Invented-fixture qualification for the one-shot DGX Spark packet."""
from __future__ import annotations

import csv
import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy/dgx_spark_gptoss120b"
sys.path.insert(0, str(DEPLOY))

import acquire
import audit_return
import collect_return
import packet_core
import runner
import spark_preflight
import verify_return

SCHEMAS = ROOT / "schemas/dgx_gptoss120b"
COMMIT = "1" * 40


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _effective_protocol() -> dict:
    base = _json(ROOT / "configs/round2_human_first_reranking_v1.json")
    delta = _json(ROOT / "configs/round2_dgx_gptoss120b_generation_delta_v1.json")
    for pointer, value in delta["authorized_overrides"].items():
        current = base; parts = pointer.split("/")[1:]
        for part in parts[:-1]: current = current[part]
        current[parts[-1]] = value
    return base


def _packet_root(tmp_path: Path) -> Path:
    root = tmp_path / "packet with spaces"
    for name in ("input", "config", "output/run", "output/return", "scratch", "fixtures"):
        (root / name).mkdir(parents=True, exist_ok=True)
    registration = {
        "schema_version": "spectech_dgx_experiment_registration_v1",
        "experiment_id": packet_core.EXPERIMENT_ID, "freeze_id": "2" * 40,
        "packet_id": packet_core.PACKET_ID, "source_git_commit": COMMIT,
        "base_bindings": [{"path": "a", "sha256": "3" * 64}] * 3,
        "ordered_input_digest": "4" * 64, "input_manifest_sha256": "5" * 64,
        "source_case_count": 60, "candidate_count": 180, "rubric_case_count": 80,
        "model_registration": {}, "runtime_registration": {},
        "effective_primary_protocol": _effective_protocol(),
        "optional_rubric_registration": {"frozen_base_protocol": _json(ROOT / "configs/round2_qwen_rubric_v1.json")}, "allowed_delta_pointers": ["/runtime/model"],
        "resource_policy": {}, "time_budget": {}, "privacy": {}, "release": {},
    }
    packet_core.write_json(root / "input/experiment_registration.json", registration)
    config = ROOT / "configs/round2_dgx_gptoss120b_generation_delta_v1.json"
    (root / "config/round2_dgx_gptoss120b_generation_delta_v1.json").write_bytes(config.read_bytes())
    rubric_config = ROOT / "configs/round2_dgx_gptoss120b_rubric_delta_v1.json"
    (root / "config/round2_dgx_gptoss120b_rubric_delta_v1.json").write_bytes(rubric_config.read_bytes())
    with (root / "input/primary_inputs.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["case_id", "source_position", "direction", "cell_id", "source_text", "source_text_sha256"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader()
        directions = ("add_specific", "de_specify", "irrelevant_rewrite")
        for index in range(1, 61):
            text = f"The invented fixture service handles sample item {index}."
            writer.writerow({"case_id":f"{index:064x}","source_position":index,"direction":directions[(index-1)%3],"cell_id":f"cell{(index-1)%6+1:02d}","source_text":text,"source_text_sha256":packet_core.sha_bytes(text.encode())})
    manifest = {"schema_version":"spectech_dgx_packet_manifest_v1","packet_id":packet_core.PACKET_ID,"packet_version":"v1","created_at_utc":"2026-08-12T00:00:00Z","source_git_commit":COMMIT,"source_worktree_dirty":False,"source_date_epoch":1,"experiment_id":packet_core.EXPERIMENT_ID,"primary_enabled":True,"secondary_rubric_enabled":True,"base_bindings":[{"path":"a","sha256":"3"*64}]*3,"model_registration_sha256":"6"*64,"runtime_registration_sha256":"7"*64,"payload_files":[],"network_default":"disabled","privacy_profile":"no_private_host_or_human_label_data_v1"}
    packet_core.write_json(root / "packet_manifest.json", manifest)
    extension = {"schema_version":"spectech_dgx_capability_extension_v1","packet_id":packet_core.PACKET_ID,"memory_available_bytes":110_000_000_000,"memory_measurement_method":"fixture","inference_runtime":{},"model":{},"gate_status":"passed","failures":[],"diagnostic_categories":[]}
    packet_core.write_json(root / "output/run/spectech_capability_extension.json", extension)
    packet_core.write_json(root / "output/run/spark_capability_report.json", {"fixture":"sanitized"})
    packet_core.write_json(root / "output/run/capability_compatibility.json", {"fixture":"bound"})
    return root


def _schema(name: str) -> dict:
    return _json(SCHEMAS / name)


def _make_tar(path: Path, members: list[tuple[str, bytes, int, str]]) -> None:
    with tarfile.open(path, "w:gz") as archive:
        for name, data, mode, kind in members:
            info = tarfile.TarInfo(name); info.mode = mode
            if kind == "symlink": info.type = tarfile.SYMTYPE; info.linkname = "target"; info.size = 0; archive.addfile(info)
            else: info.size = len(data); archive.addfile(info, io.BytesIO(data))


def test_schema_validator_accepts_positive_and_rejects_negative_capability(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(spark_preflight, "_probe", lambda *args, **kwargs: {"available": True, "return_code": 0})
    base, extension, compatibility = spark_preflight.collect(tmp_path, _json(DEPLOY / "fixtures/spark_pass.json"))
    base_path = tmp_path / "base.json"; extension_path = tmp_path / "extension.json"
    packet_core.write_json(base_path, base); packet_core.write_json(extension_path, extension)
    compatibility["capability_report_sha256"] = packet_core.sha_file(base_path)
    compatibility["extension_sha256"] = packet_core.sha_file(extension_path)
    packet_core.validate_schema(base, _schema("spark_capability_report.schema.json"))
    packet_core.validate_schema(extension, _schema("capability_extension.schema.json"))
    packet_core.validate_schema(compatibility, _schema("capability_compatibility.schema.json"))
    broken = dict(extension); broken["private_hostname"] = "forbidden"
    with pytest.raises(ValueError, match="additional properties"):
        packet_core.validate_schema(broken, _schema("capability_extension.schema.json"))


def test_preflight_failure_is_content_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(spark_preflight, "_probe", lambda *args, **kwargs: {"available": False, "return_code": None})
    _, extension, _ = spark_preflight.collect(tmp_path, _json(DEPLOY / "fixtures/spark_fail_memory.json"))
    assert extension["gate_status"] == "failed"
    assert {item["code"] for item in extension["failures"]} == {"memory_available_low"}
    assert "hostname" not in json.dumps(extension).casefold()


@pytest.mark.parametrize("member", [
    ("../escape", b"x", 0o600, "file"),
    ("root/link", b"", 0o600, "symlink"),
    ("root/world", b"x", 0o666, "file"),
])
def test_archive_rejects_traversal_symlink_and_unsafe_mode(tmp_path: Path, member) -> None:
    path = tmp_path / "bad.tar.gz"; _make_tar(path, [member])
    with pytest.raises(ValueError): packet_core.inspect_archive(path)


def test_archive_rejects_case_collision(tmp_path: Path) -> None:
    path = tmp_path / "collision.tar.gz"
    _make_tar(path, [("root/File", b"a", 0o600, "file"), ("root/file", b"b", 0o600, "file")])
    with pytest.raises(ValueError, match="collision"): packet_core.inspect_archive(path)


def test_deterministic_archive_builder_is_byte_identical(tmp_path: Path) -> None:
    files = [("a.txt", b"a\n", 0o600), ("nested/b.txt", b"b\n", 0o644)]
    one, two = tmp_path / "one.tar.gz", tmp_path / "two.tar.gz"
    packet_core.deterministic_tar_gz(one, "root", files, 123456789)
    packet_core.deterministic_tar_gz(two, "root", reversed(files), 123456789)
    assert one.read_bytes() == two.read_bytes()


def test_privacy_scan_rejects_absolute_paths_credentials_and_labels() -> None:
    for value in (b"C:\\Users\\private\\file", b"password=secret", b'"human_label": 3'):
        with pytest.raises(ValueError): packet_core.privacy_scan_bytes(value, "fixture")


def test_action_chain_registry_note_bounds_and_tamper(tmp_path: Path) -> None:
    path = tmp_path / "actions.jsonl"; started = __import__("time").monotonic()
    first = packet_core.append_action(path, "run", started, stage="collect", event_type="diagnostic", category="memory", command_id="diagnose-memory-v1", status="passed")
    second = packet_core.append_action(path, "run", started, stage="collect", event_type="operator_note", category="operator_note", command_id="operator-note-v1", status="passed", note="sanitized observation")
    assert packet_core.verify_action_chain(path) == (2, second["event_sha256"])
    with pytest.raises(ValueError): packet_core.append_action(path, "run", started, stage="collect", event_type="operator_note", category="operator_note", command_id="note", status="passed", note="x" * 501)
    with pytest.raises(ValueError): packet_core.append_action(path, "run", started, stage="collect", event_type="diagnostic", category="arbitrary", command_id="shell", status="passed")
    path.write_text(path.read_text().replace("sanitized observation", "changed"), encoding="utf-8")
    with pytest.raises(ValueError, match="chain mismatch"): packet_core.verify_action_chain(path)


def test_fake_primary_success_exact_180_and_complete_return_from_path_with_spaces(tmp_path: Path) -> None:
    root = _packet_root(tmp_path)
    receipt = runner.run(root, "fake", {}, False, True)
    assert receipt["release_status"] == "eligible_complete"
    assert receipt["counts"] == {"planned":180,"attempted":180,"accepted":180,"rejected":0,"exhausted":0,"missing":0}
    attempts = [json.loads(line) for line in (root / "output/run/generation_attempts.jsonl").read_text().splitlines()]
    assert len(attempts) == 180 and all(item["reasoning_text_retained"] is False for item in attempts)
    assert all("fixture trace" not in json.dumps(item) for item in attempts)
    packet_core.validate_schema(attempts[0], _schema("generation_attempt.schema.json"))
    packet_core.validate_schema(receipt, _schema("run_receipt.schema.json"))
    archive, _ = collect_return.collect(root)
    manifest = verify_return.verify(archive)
    packet_core.validate_schema(manifest, _schema("return_manifest.schema.json"))


def test_fixture_scenario_never_reads_packet_project_inputs(tmp_path: Path) -> None:
    root = _packet_root(tmp_path)
    (root / "input/primary_inputs.csv").write_text("must_not_be_read\n", encoding="utf-8")
    receipt = runner.run(root, "fake", {"invented_inputs": True}, False, False)
    assert receipt["release_status"] == "eligible_complete"
    assert receipt["rubric_status"] == "complete"
    rows = list(csv.DictReader((root / "output/run/rubric_scores.csv").open(encoding="utf-8")))
    assert len(rows) == 80 and len({row["rubric_case_id"] for row in rows}) == 80


def test_exhausted_slot_suppresses_eligible_release_and_returns_diagnostic(tmp_path: Path) -> None:
    root = _packet_root(tmp_path)
    receipt = runner.run(root, "fake", {"fail_status":"oom","fail_candidates":[7],"fail_attempts":3}, False, True)
    assert receipt["release_status"] == "diagnostic_incomplete" and receipt["counts"]["accepted"] == 179
    archive, _ = collect_return.collect(root)
    with pytest.raises(ValueError, match="requires --allow-diagnostic"): verify_return.verify(archive)
    assert verify_return.verify(archive, True)["release_status"] == "diagnostic_incomplete"


def test_interruption_then_exact_resume_has_no_duplicate_candidate(tmp_path: Path) -> None:
    root = _packet_root(tmp_path)
    with pytest.raises(KeyboardInterrupt): runner.run(root, "fake", {"interrupt_after":10}, False, True)
    receipt = runner.run(root, "fake", {}, True, True)
    assert receipt["release_status"] == "eligible_complete" and receipt["resume_count"] == 1
    accepted = [json.loads(line)["candidate_id"] for line in (root / "output/run/generation_attempts.jsonl").read_text().splitlines() if json.loads(line)["status"] == "accepted"]
    assert len(accepted) == len(set(accepted)) == 180


@pytest.mark.parametrize("status", ["timeout", "oom", "runtime_failed", "interface_failed"])
def test_failure_accounting_is_typed_and_content_free(tmp_path: Path, status: str) -> None:
    root = _packet_root(tmp_path)
    receipt = runner.run(root, "fake", {"fail_status":status,"fail_candidates":[1],"fail_attempts":3}, False, True)
    assert receipt["failure_counts"] == {status:3}
    assert receipt["release_status"] == "diagnostic_incomplete"
    assert "invented fixture service" not in json.dumps(receipt).casefold()


def test_default_acquisition_has_no_network_and_allowlist_is_exact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert acquire.RUNTIME_URL.startswith("https://github.com/ollama/ollama/releases/download/v0.32.5/")
    assert "curl" not in (DEPLOY / "run_all.sh").read_text().casefold()
    calls = []
    monkeypatch.setattr(acquire.urllib.request, "urlopen", lambda *args, **kwargs: calls.append(args) or (_ for _ in ()).throw(AssertionError("network")))
    monkeypatch.setattr(sys, "argv", ["acquire.py", "--root", str(tmp_path)])
    with pytest.raises(SystemExit, match="acquisition requires"): acquire.main()
    assert calls == []


def test_registered_remediation_set_has_only_packet_scoped_changes() -> None:
    assert set(packet_core.REMEDIATIONS) == {"replace-transfer","clear-marker-cache","restart-runtime","release-stale-lock"}
    assert {change for _, change in packet_core.REMEDIATIONS.values()} == {"acquisition_path","packet_cache","packet_runtime","checkpoint_resume"}


def test_return_audit_detects_candidate_namespace_drift_without_reading_outcomes() -> None:
    case_id = "a" * 64
    assert audit_return._runner_candidate_id(case_id, 1) != audit_return._base_candidate_id(case_id, 1)
    assert audit_return._runner_candidate_id(case_id, 1) == runner._candidate_id(case_id, 1)


def test_return_audit_reapplies_frozen_integrity_checks() -> None:
    assert audit_return._integrity_reasons("Restart the service.", "Restart nginx on web-03.") == []
    reasons = audit_return._integrity_reasons("Restart the service.", "Restart the service.")
    assert "unchanged" in reasons
    assert "new_non_latin_script" in audit_return._integrity_reasons("Restart the service.", "Restart the sérvice.")
