#!/usr/bin/env python3
"""Content-blind audit of a DGX return against its sealed source packet.

The audit hashes candidate text but never prints or writes it.  It is stricter
than ``verify_return.py``: scientific eligibility requires complete attempt
accounting and cross-checks against the packet inputs and frozen registration.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from packet_core import (
    MODEL_DIGEST,
    canonical,
    derive_seed,
    inspect_archive,
    privacy_scan_bytes,
    sha_bytes,
    sha_file,
    validate_schema,
    verify_archive,
)
from verify_return import verify as verify_return


EXPECTED_CANDIDATE_FIELDS = (
    "schema_version", "run_id", "case_id", "candidate_id", "candidate_slot",
    "retained_attempt", "seed", "source_text_sha256", "candidate_text",
    "candidate_text_sha256", "model_identity", "config_sha256", "freeze_id",
)
EXPECTED_RUBRIC_FIELDS = (
    "schema_version", "run_id", "rubric_case_id", "score_1_to_5",
    "response_sha256", "model_identity", "prompt_sha256", "config_sha256",
    "started_at_utc", "elapsed_ms", "prompt_tokens", "completion_tokens",
)
SCHEMA_FILES = {
    "return_manifest": "return_manifest.schema.json",
    "run_receipt": "run_receipt.schema.json",
    "capability": "spark_capability_report.schema.json",
    "extension": "capability_extension.schema.json",
    "compatibility": "capability_compatibility.schema.json",
}
SENTENCE_BOUNDARY = re.compile(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)")


def _archive_payload(path: Path) -> tuple[str, dict[str, bytes]]:
    members = inspect_archive(path)
    roots = {member.name.split("/", 1)[0] for member in members}
    if len(roots) != 1:
        raise ValueError("archive has multiple logical roots")
    root = next(iter(roots))
    with tarfile.open(path, "r:gz") as archive:
        payload = {
            member.name.split("/", 1)[1]: archive.extractfile(member).read()
            for member in members
        }
    return root, payload


def _json(payload: dict[str, bytes], name: str) -> dict[str, Any]:
    return json.loads(payload[name].decode("utf-8"))


def _csv(payload: dict[str, bytes], name: str) -> tuple[list[str], list[dict[str, str]]]:
    reader = csv.DictReader(io.StringIO(payload[name].decode("utf-8")))
    return list(reader.fieldnames or []), list(reader)


def _time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _base_candidate_id(case_id: str, slot: int) -> str:
    return sha_bytes(f"{case_id}\0candidate_slot\0{slot}".encode("utf-8"))


def _runner_candidate_id(case_id: str, slot: int) -> str:
    return sha_bytes(f"{case_id}\0{slot}".encode("utf-8"))


def _integrity_reasons(original: str, candidate: str) -> list[str]:
    reasons: list[str] = []
    text = candidate.strip()
    normalized = lambda value: " ".join(unicodedata.normalize("NFKC", value).strip().split()).casefold()
    if not text or "\n" in text or "\r" in text:
        reasons.append("not_one_nonempty_line")
    if len(text) < 5:
        reasons.append("too_short")
    if len(text) > 2000:
        reasons.append("too_long")
    if normalized(text) == normalized(original):
        reasons.append("unchanged")
    if len(SENTENCE_BOUNDARY.findall(text)) > 1:
        reasons.append("multiple_sentences")
    original_nonlatin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in original)
    candidate_nonlatin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in text)
    if not original_nonlatin and candidate_nonlatin:
        reasons.append("new_non_latin_script")
    return reasons


def _verify_action_chain_bytes(data: bytes) -> tuple[int, str | None]:
    previous = None
    count = 0
    for line in data.decode("utf-8").splitlines():
        item = json.loads(line)
        recorded = item.pop("event_sha256")
        if item["previous_event_sha256"] != previous or sha_bytes(canonical(item)) != recorded:
            raise ValueError("operator action chain mismatch")
        previous = recorded
        count += 1
    return count, previous


def audit(return_archive: Path, source_packet: Path, schemas: Path) -> dict[str, Any]:
    # Both built-in verifiers run first; neither extracts or executes content.
    source_manifest = verify_archive(source_packet)
    return_manifest = verify_return(return_archive, allow_diagnostic=True)
    _, returned = _archive_payload(return_archive)
    _, source = _archive_payload(source_packet)

    receipt = _json(returned, "run_receipt.json")
    capability = _json(returned, "spark_capability_report.json")
    extension = _json(returned, "spectech_capability_extension.json")
    compatibility = _json(returned, "capability_compatibility.json")
    values = {
        "return_manifest": return_manifest,
        "run_receipt": receipt,
        "capability": capability,
        "extension": extension,
        "compatibility": compatibility,
    }
    for key, filename in SCHEMA_FILES.items():
        validate_schema(values[key], json.loads((schemas / filename).read_text(encoding="utf-8")))

    for name in (
        "return_manifest.json", "run_receipt.json", "spark_capability_report.json",
        "spectech_capability_extension.json", "capability_compatibility.json",
        "operator_actions.jsonl", "operator_report.md", "return_report.md",
    ):
        privacy_scan_bytes(returned[name], name)

    registration = _json(source, "input/experiment_registration.json")
    primary_fields, primary = _csv(source, "input/primary_inputs.csv")
    rubric_input_fields, rubric_inputs = _csv(source, "input/rubric_inputs.csv")
    candidate_fields, candidates = _csv(returned, "retained_candidates.csv")
    rubric_fields, rubric = _csv(returned, "rubric_scores.csv")

    if primary_fields != ["case_id", "source_position", "direction", "cell_id", "source_text", "source_text_sha256"]:
        raise ValueError("unexpected primary input fields")
    if rubric_input_fields != ["rubric_case_id", "rubric_position", "sentence_text", "sentence_text_sha256"]:
        raise ValueError("unexpected rubric input fields")
    if candidate_fields != list(EXPECTED_CANDIDATE_FIELDS):
        raise ValueError("unexpected retained candidate fields")
    if rubric_fields != list(EXPECTED_RUBRIC_FIELDS):
        raise ValueError("unexpected rubric fields")

    case_map = {row["case_id"]: row for row in primary}
    expected_order = [(row["case_id"], slot) for row in primary for slot in (1, 2, 3)]
    actual_order = [(row["case_id"], int(row["candidate_slot"])) for row in candidates]
    generation_config_sha = sha_bytes(source["config/round2_dgx_gptoss120b_generation_delta_v1.json"])
    expected_model = "gpt-oss:120b@" + MODEL_DIGEST
    integrity = Counter()
    for row in candidates:
        case = case_map.get(row["case_id"])
        if case is None:
            raise ValueError("candidate references unknown case")
        slot = int(row["candidate_slot"])
        attempt = int(row["retained_attempt"])
        text = row["candidate_text"]
        for reason in _integrity_reasons(case["source_text"], text):
            integrity[reason] += 1
        if sha_bytes(text.encode("utf-8")) != row["candidate_text_sha256"]:
            raise ValueError("candidate text hash mismatch")
        if row["source_text_sha256"] != case["source_text_sha256"]:
            raise ValueError("candidate source hash mismatch")
        if int(row["seed"]) != derive_seed(2026081201, "dgx-spark-gptoss120b-candidate-v1", row["case_id"], slot, attempt):
            raise ValueError("candidate seed mismatch")
        if row["config_sha256"] != generation_config_sha:
            raise ValueError("candidate config mismatch")
        if row["freeze_id"] != registration["freeze_id"]:
            raise ValueError("candidate freeze mismatch")
        if row["model_identity"] != expected_model:
            raise ValueError("candidate model identity mismatch")

    rubric_config_sha = sha_bytes(source["config/round2_dgx_gptoss120b_rubric_delta_v1.json"])
    base_rubric = registration["optional_rubric_registration"]["frozen_base_protocol"]
    expected_rubric_order = [row["rubric_case_id"] for row in rubric_inputs]
    for row, source_row in zip(rubric, rubric_inputs):
        if sha_bytes(source_row["sentence_text"].encode("utf-8")) != source_row["sentence_text_sha256"]:
            raise ValueError("rubric input hash mismatch")
        score = int(row["score_1_to_5"])
        if not 1 <= score <= 5:
            raise ValueError("rubric score out of range")
        prompt = base_rubric["prompt"]["system"] + "\n\n" + base_rubric["prompt"]["user_template"].format(sent_text=source_row["sentence_text"])
        if row["prompt_sha256"] != sha_bytes(prompt.encode("utf-8")):
            raise ValueError("rubric prompt mismatch")
        if row["config_sha256"] != rubric_config_sha or row["model_identity"] != expected_model:
            raise ValueError("rubric identity mismatch")

    action_count, action_head = _verify_action_chain_bytes(returned["operator_actions.jsonl"])
    inventory = {item["path"]: item for item in return_manifest["files"]}
    receipt_start, receipt_end = _time(receipt["started_at_utc"]), _time(receipt["ended_at_utc"])
    rubric_times_in_run = all(receipt_start <= _time(row["started_at_utc"]) <= receipt_end for row in rubric)
    candidate_ids_base = sum(
        row["candidate_id"] == _base_candidate_id(row["case_id"], int(row["candidate_slot"]))
        for row in candidates
    )
    candidate_ids_runner = sum(
        row["candidate_id"] == _runner_candidate_id(row["case_id"], int(row["candidate_slot"]))
        for row in candidates
    )
    attempt_hash = receipt.get("output_hashes", {}).get("generation_attempts")
    attempts_present = "generation_attempts.jsonl" in returned

    blockers: list[str] = []
    deviations: list[str] = []
    if not attempts_present:
        blockers.append("generation_attempts_missing_from_return")
    if candidate_ids_base != len(candidates):
        deviations.append("candidate_id_namespace_differs_from_human_first_base_but_is_deterministic_and_join_complete")
    if integrity:
        blockers.append("retained_candidates_fail_frozen_integrity_checks")
    if extension["inference_runtime"].get("status") != "loaded" or not extension["model"].get("present"):
        deviations.append("capability_extension_is_pre_acquisition_by_workflow_order")
        blockers.append("post_acquisition_model_identity_evidence_missing_from_return")
    if "model_smoke_report.json" not in returned:
        blockers.append("model_smoke_report_missing_from_return")
    if receipt.get("output_hashes", {}).get("rubric_scores") is None:
        blockers.append("rubric_output_not_bound_by_run_receipt")
    if not rubric_times_in_run:
        blockers.append("rubric_timestamps_extend_past_run_receipt_end")

    return {
        "schema_version": "spectech_dgx_return_audit_v1",
        "return_archive_sha256": sha_file(return_archive),
        "source_packet_sha256": sha_file(source_packet),
        "packet_id": return_manifest["packet_id"],
        "run_id": return_manifest["run_id"],
        "declared_release_status": return_manifest["release_status"],
        "audit_disposition": "scientific_eligibility_withheld" if blockers else "scientific_eligibility_supported",
        "archive": {
            "member_count": len(returned), "safe_regular_members": True,
            "manifest_inventory_and_hashes": "passed",
        },
        "bindings": {
            "source_packet_manifest": source_manifest["packet_id"],
            "source_git_commit_matches": receipt["source_git_commit"] == source_manifest["source_git_commit"] == registration["source_git_commit"],
            "freeze_matches": receipt["freeze_id"] == registration["freeze_id"],
            "run_id_consistent": all(row["run_id"] == receipt["run_id"] for row in candidates + rubric),
        },
        "primary": {
            "source_rows": len(primary), "candidate_rows": len(candidates),
            "unique_candidate_ids": len({row["candidate_id"] for row in candidates}),
            "exact_case_slot_order": actual_order == expected_order,
            "retained_attempt_counts": dict(sorted(Counter(row["retained_attempt"] for row in candidates).items())),
            "candidate_id_matches_frozen_base": candidate_ids_base,
            "candidate_id_matches_remote_runner": candidate_ids_runner,
            "candidate_text_hashes": "passed", "source_hash_joins": "passed",
            "seed_derivation": "passed", "config_freeze_model_bindings": "passed",
            "independent_frozen_integrity_failures": dict(sorted(integrity.items())),
            "attempt_log_present": attempts_present,
            "attempt_log_expected_sha256": attempt_hash,
            "attempt_accounting_independently_verifiable": attempts_present,
        },
        "rubric": {
            "input_rows": len(rubric_inputs), "score_rows": len(rubric),
            "unique_case_ids": len({row["rubric_case_id"] for row in rubric}),
            "exact_input_order": [row["rubric_case_id"] for row in rubric] == expected_rubric_order,
            "scores_in_range": all(1 <= int(row["score_1_to_5"]) <= 5 for row in rubric),
            "prompt_config_model_bindings": "passed", "timestamps_within_run": rubric_times_in_run,
            "first_started_at_utc": min(row["started_at_utc"] for row in rubric),
            "last_started_at_utc": max(row["started_at_utc"] for row in rubric),
            "total_elapsed_ms": sum(int(row["elapsed_ms"]) for row in rubric),
            "run_receipt_hash_binding": receipt.get("output_hashes", {}).get("rubric_scores"),
        },
        "capability": {
            "architecture": capability["system"]["architecture"],
            "memory_total_bytes": capability["system"]["memory_total_bytes"],
            "memory_available_preflight_bytes": extension["memory_available_bytes"],
            "preflight_gate": extension["gate_status"],
            "preflight_runtime_status": extension["inference_runtime"].get("status"),
            "preflight_model_present": extension["model"].get("present"),
            "receipt_model_residency": receipt["model_residency"],
            "compatibility_hashes": compatibility["capability_report_sha256"] == sha_bytes(returned["spark_capability_report.json"]) and compatibility["extension_sha256"] == sha_bytes(returned["spectech_capability_extension.json"]),
        },
        "chronology": {
            "preflight_before_run": _time(capability["collected_at_utc"]) <= receipt_start,
            "run_start_utc": receipt["started_at_utc"], "run_end_utc": receipt["ended_at_utc"],
            "reported_actual_seconds": receipt["time_budget"]["actual_seconds"],
        },
        "operator_actions": {"count": action_count, "final_chain_sha256": action_head},
        "manifest_flags": {
            "retained_candidates_required_for_import": inventory["retained_candidates.csv"]["required_for_import"],
            "rubric_scores_required_for_import": inventory["rubric_scores.csv"]["required_for_import"],
        },
        "blockers": blockers,
        "nonblocking_deviations": deviations,
        "privacy": {"metadata_scan": "passed", "candidate_text_not_printed_or_exported": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("return_archive", type=Path)
    parser.add_argument("--return-sha256", type=Path)
    parser.add_argument("--source-packet", required=True, type=Path)
    parser.add_argument("--source-sha256", type=Path)
    parser.add_argument("--schemas", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for archive, checksum in ((args.return_archive, args.return_sha256), (args.source_packet, args.source_sha256)):
        if checksum and sha_file(archive) != checksum.read_text(encoding="ascii").split()[0].casefold():
            raise ValueError(f"external checksum mismatch: {archive.name}")
    result = audit(args.return_archive, args.source_packet, args.schemas)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "passed", "run_id": result["run_id"],
        "audit_disposition": result["audit_disposition"],
        "blockers": result["blockers"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
