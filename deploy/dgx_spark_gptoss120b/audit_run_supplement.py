#!/usr/bin/env python3
"""Content-blind audit of the preserved DGX run-directory supplement.

The auditor never executes returned material and never emits sentence,
candidate, rubric-score, reasoning, or private path content.  It validates the
ZIP before optional bounded extraction, reconciles it to the sealed packet and
the original return, and emits compact provenance evidence only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from packet_core import derive_seed, validate_schema


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(data: bytes) -> dict[str, Any]:
    return json.loads(data.decode("utf-8"))


def load_jsonl(data: bytes) -> list[dict[str, Any]]:
    return [json.loads(line) for line in data.decode("utf-8").splitlines() if line]


def csv_rows(data: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"))))


def parse_utc(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def iso_millis(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def packet_files(packet_path: Path) -> dict[str, bytes]:
    with tarfile.open(packet_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        roots = {member.name.split("/", 1)[0] for member in members}
        if len(roots) != 1:
            raise ValueError("source packet root mismatch")
        return {member.name.split("/", 1)[1]: archive.extractfile(member).read() for member in members}


def return_files(return_path: Path) -> dict[str, bytes]:
    with tarfile.open(return_path, "r:gz") as archive:
        members = [member for member in archive.getmembers() if member.isfile()]
        roots = {member.name.split("/", 1)[0] for member in members}
        if len(roots) != 1:
            raise ValueError("original return root mismatch")
        return {member.name.split("/", 1)[1]: archive.extractfile(member).read() for member in members}


def inspect_zip(path: Path, expected_root: str) -> tuple[zipfile.ZipFile, dict[str, zipfile.ZipInfo], dict[str, Any]]:
    archive = zipfile.ZipFile(path)
    seen: set[str] = set()
    folded: set[str] = set()
    files: dict[str, zipfile.ZipInfo] = {}
    directory_count = 0
    discarded_group_write_modes = 0
    for member in archive.infolist():
        name = member.filename
        if "\\" in name or re.match(r"^[A-Za-z]:", name):
            raise ValueError("unsafe ZIP path syntax")
        pure = PurePosixPath(name.rstrip("/"))
        if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
            raise ValueError("unsafe ZIP path")
        if pure.parts[0] != expected_root:
            raise ValueError("unexpected ZIP root")
        key = name.rstrip("/")
        if key in seen or key.casefold() in folded:
            raise ValueError("duplicate or case-colliding ZIP member")
        seen.add(key)
        folded.add(key.casefold())
        mode = (member.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(mode) if mode else 0
        is_directory = member.is_dir() or file_type == stat.S_IFDIR
        is_regular = file_type in {0, stat.S_IFREG} and not member.is_dir()
        if not (is_directory or is_regular):
            raise ValueError("link, device, socket, FIFO, or other special ZIP member")
        if mode & (stat.S_ISUID | stat.S_ISGID | stat.S_ISVTX):
            raise ValueError("unsafe special mode bits")
        if is_regular and mode & 0o111:
            raise ValueError("unexpected executable returned file")
        if mode & 0o022:
            discarded_group_write_modes += 1
        if is_directory:
            directory_count += 1
        else:
            files[name] = member
    return archive, files, {
        "safe": True,
        "regular_file_count": len(files),
        "directory_count": directory_count,
        "absolute_or_traversal_count": 0,
        "link_or_device_count": 0,
        "duplicate_or_case_collision_count": 0,
        "special_mode_count": 0,
        "executable_regular_file_count": 0,
        "source_group_or_world_writable_modes_discarded_on_extraction": discarded_group_write_modes,
    }


def safe_extract(archive: zipfile.ZipFile, files: dict[str, zipfile.ZipInfo], destination: Path) -> None:
    if destination.exists():
        raise ValueError("validated extraction destination already exists")
    destination.mkdir(parents=True, mode=0o700)
    destination = destination.resolve()
    for name, member in sorted(files.items()):
        relative = PurePosixPath(name)
        target = destination.joinpath(*relative.parts).resolve()
        if destination not in target.parents:
            raise ValueError("extraction target escaped destination")
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with target.open("xb") as handle:
            handle.write(archive.read(member))
        os.chmod(target, 0o600)


def _request(protocol: dict[str, Any], text: str, direction: str, seed: int) -> dict[str, Any]:
    user = protocol["prompts"]["user_templates"][direction].format(sentence_original=text)
    return {
        "model": "gpt-oss:120b",
        "messages": [
            {"role": "system", "content": protocol["prompts"]["system"]},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": "low",
        "format": protocol["runtime"]["response_schema"],
        "keep_alive": "30m",
        "options": {**protocol["runtime"]["options"], "seed": seed},
    }


def require_expected_case_slot(row: dict[str, Any], case_id: str, slot: int, label: str) -> None:
    if row.get("case_id") != case_id or int(row.get("candidate_slot", -1)) != slot:
        raise ValueError(f"{label} case/slot mismatch")


def validate_retained_metadata(
    kept: dict[str, str],
    attempt: dict[str, Any],
    case: dict[str, str],
    *,
    run_id: str,
    config_sha256: str,
    freeze_id: str,
    model_identity: str,
) -> None:
    expected = {
        "run_id": run_id,
        "case_id": case["case_id"],
        "candidate_id": attempt["candidate_id"],
        "candidate_slot": str(attempt["candidate_slot"]),
        "retained_attempt": str(attempt["attempt_index"]),
        "seed": str(attempt["seed"]),
        "source_text_sha256": case["source_text_sha256"],
        "candidate_text_sha256": attempt["candidate_text_sha256"],
        "model_identity": model_identity,
        "config_sha256": config_sha256,
        "freeze_id": freeze_id,
    }
    for field, value in expected.items():
        if kept.get(field) != value:
            raise ValueError(f"retained metadata mismatch: {field}")
    if kept.get("candidate_text") != attempt["candidate_text"]:
        raise ValueError("retained candidate text mismatch")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    config = json.loads(args.config.read_text(encoding="utf-8"))
    validate_schema(config, json.loads(args.schema.read_text(encoding="utf-8")))
    if sha_file(args.zip) != config["supplement"]["sha256"]:
        raise ValueError("supplement ZIP SHA-256 mismatch")
    if sha_file(args.original_return) != config["bindings"]["original_return_archive_sha256"]:
        raise ValueError("original return SHA-256 mismatch")
    if sha_file(args.source_packet) != config["bindings"]["source_packet_sha256"]:
        raise ValueError("source packet SHA-256 mismatch")
    if sha_file(args.original_audit) != config["bindings"]["original_audit_sha256"]:
        raise ValueError("original audit SHA-256 mismatch")
    if sha_file(args.prior_acceptance) != config["bindings"]["prior_acceptance_summary_sha256"]:
        raise ValueError("prior acceptance SHA-256 mismatch")

    archive, infos, safety = inspect_zip(args.zip, config["supplement"]["expected_root"])
    if set(infos) != set(config["required_members"]):
        raise ValueError("supplement member inventory mismatch")
    if len(infos) != config["supplement"]["expected_regular_files"]:
        raise ValueError("supplement file count mismatch")
    data = {name: archive.read(info) for name, info in infos.items()}
    member_records = [
        {"path": name, "size_bytes": len(value), "sha256": sha_bytes(value)}
        for name, value in sorted(data.items())
    ]

    original = return_files(args.original_return)
    unchanged = {}
    for name in config["unchanged_output_members"]:
        current = data[f"run/{name}"]
        if name not in original or current != original[name]:
            raise ValueError(f"original-return member changed: {name}")
        unchanged[name] = sha_bytes(current)

    packet = packet_files(args.source_packet)
    registration = load_json(packet["input/experiment_registration.json"])
    if registration["freeze_id"] != config["bindings"]["freeze_id"]:
        raise ValueError("freeze binding mismatch")
    if registration["source_git_commit"] != config["bindings"]["source_git_commit"]:
        raise ValueError("source commit binding mismatch")

    attempt_schema = load_json(packet["schemas/generation_attempt.schema.json"])
    attempts = load_jsonl(data["run/generation_attempts.jsonl"])
    for index, row in enumerate(attempts):
        validate_schema(row, attempt_schema, f"attempt[{index}]")
    primary = csv_rows(packet["input/primary_inputs.csv"])
    retained = csv_rows(data["run/retained_candidates.csv"])
    if len(primary) != 60 or len(attempts) != 180 or len(retained) != 180:
        raise ValueError("primary coverage mismatch")
    retained_by_id = {row["candidate_id"]: row for row in retained}
    expected_entries = []
    reasoning_hash_valid = 0
    for case in primary:
        for slot in (1, 2, 3):
            candidate_id = sha_bytes(f"{case['case_id']}\0{slot}".encode())
            expected_entries.append((candidate_id, case["case_id"], slot))
    expected_order = [item[0] for item in expected_entries]
    if [row["candidate_id"] for row in attempts] != expected_order:
        raise ValueError("attempt order mismatch")
    if [row["candidate_id"] for row in retained] != expected_order:
        raise ValueError("retained order mismatch")
    cases = {row["case_id"]: row for row in primary}
    generation_config_sha256 = sha_bytes(packet["config/round2_dgx_gptoss120b_generation_delta_v1.json"])
    response_model_identity = "gpt-oss:120b@sha256:" + config["bindings"]["model_manifest_sha256"]
    for row, (expected_candidate_id, expected_case_id, expected_slot) in zip(attempts, expected_entries):
        if row["candidate_id"] != expected_candidate_id:
            raise ValueError("attempt candidate ID mismatch")
        require_expected_case_slot(row, expected_case_id, expected_slot, "attempt")
        case = cases[expected_case_id]
        if row["run_id"] != config["run_id"] or row["attempt_index"] != 1 or row["status"] != "accepted":
            raise ValueError("attempt accounting mismatch")
        seed = derive_seed(2026081201, "dgx-spark-gptoss120b-candidate-v1", row["case_id"], row["candidate_slot"], 1)
        request = _request(registration["effective_primary_protocol"], case["source_text"], case["direction"], seed)
        if row["seed"] != seed:
            raise ValueError("attempt seed mismatch")
        if row["input_text_sha256"] != sha_bytes(case["source_text"].encode()):
            raise ValueError("attempt input hash mismatch")
        if row["prompt_sha256"] != sha_bytes(json.dumps(request["messages"], sort_keys=True).encode()):
            raise ValueError("attempt prompt hash mismatch")
        if row["request_sha256"] != sha_bytes(json.dumps(request, sort_keys=True).encode()):
            raise ValueError("attempt request hash mismatch")
        if row["response_model_identity"] != response_model_identity:
            raise ValueError("attempt model identity mismatch")
        if row["reasoning_text_retained"] is not False or "reasoning_text" in row:
            raise ValueError("reasoning retention mismatch")
        if row["reasoning_character_count"] > 0:
            if not isinstance(row["reasoning_sha256"], str) or not re.fullmatch(r"[0-9a-f]{64}", row["reasoning_sha256"]):
                raise ValueError("reasoning hash accounting mismatch")
            reasoning_hash_valid += 1
        candidate = row["candidate_text"]
        if row["candidate_text_sha256"] != sha_bytes(candidate.encode()) or row["candidate_character_count"] != len(candidate):
            raise ValueError("attempt candidate hash mismatch")
        kept = retained_by_id[row["candidate_id"]]
        require_expected_case_slot(kept, expected_case_id, expected_slot, "retained")
        validate_retained_metadata(
            kept, row, case, run_id=config["run_id"], config_sha256=generation_config_sha256,
            freeze_id=config["bindings"]["freeze_id"], model_identity=response_model_identity,
        )

    state = load_json(data["run/state.json"])
    if state["run_id"] != config["run_id"] or state["resume_count"] != 0 or len(state["accepted"]) != 180:
        raise ValueError("state accounting mismatch")
    if state["accepted"] != {row["candidate_id"]: row["candidate_text_sha256"] for row in attempts}:
        raise ValueError("state accepted map mismatch")

    receipt = load_json(data["run/run_receipt.json"])
    validate_schema(receipt, load_json(packet["schemas/run_receipt.schema.json"]), "receipt")
    exact_receipt_bindings = {
        "packet_id": registration["packet_id"],
        "experiment_id": registration["experiment_id"],
        "run_id": config["run_id"],
        "freeze_id": registration["freeze_id"],
        "source_git_commit": registration["source_git_commit"],
        "model_identity": "sha256:" + config["bindings"]["model_manifest_sha256"],
        "runtime_identity": config["bindings"]["runtime_version"],
        "command_identity": "run_all_v1",
        "started_at_utc": state["started_at_utc"],
        "resume_count": state["resume_count"],
        "release_status": "eligible_complete",
        "rubric_status": "complete",
        "privacy_scan": "passed",
        "schema_validation": "passed",
        "final_decision_branch": "return-complete",
    }
    for field, expected in exact_receipt_bindings.items():
        if receipt.get(field) != expected:
            raise ValueError(f"receipt binding mismatch: {field}")
    expected_stages = {name: "passed" for name in ("verify", "preflight", "acquire", "smoke", "estimate", "primary", "rubric", "validate", "collect")}
    if receipt["stage_statuses"] != expected_stages:
        raise ValueError("receipt stage-status mismatch")
    if receipt["model_residency"] != {"before": "verified", "after": "verified"}:
        raise ValueError("receipt model-residency mismatch")
    if receipt["run_id"] != config["run_id"] or receipt["counts"] != {"planned": 180, "attempted": 180, "accepted": 180, "rejected": 0, "exhausted": 0, "missing": 0}:
        raise ValueError("receipt accounting mismatch")
    if receipt["output_hashes"]["generation_attempts"] != sha_bytes(data["run/generation_attempts.jsonl"]):
        raise ValueError("receipt attempt-log hash mismatch")
    if receipt["output_hashes"]["retained_candidates"] != sha_bytes(data["run/retained_candidates.csv"]):
        raise ValueError("receipt retained-candidate hash mismatch")
    if receipt["elapsed_ms"] != sum(row["elapsed_ms"] for row in attempts):
        raise ValueError("receipt elapsed accounting mismatch")
    elapsed_values = sorted(row["elapsed_ms"] for row in attempts)
    if receipt["latency"]["p50_ms"] != (elapsed_values[89] + elapsed_values[90]) / 2:
        raise ValueError("receipt median mismatch")
    if receipt["latency"]["p95_ms"] != elapsed_values[max(0, int(len(elapsed_values) * .95) - 1)]:
        raise ValueError("receipt p95 mismatch")

    smoke = load_json(data["run/model_smoke_report.json"])
    smoke_required = {
        "schema_version": "spectech_dgx_model_smoke_v1",
        "fixture_count": 3,
        "direction_count": 3,
        "structured_valid": 3,
        "response_model_identity": "gpt-oss:120b@sha256:" + config["bindings"]["model_manifest_sha256"],
        "reasoning_text_retained": False,
        "token_rate_minimum_met": True,
        "intentional_interruption_test": "passed",
        "exact_resume_test": "passed",
        "primary_projection_gate": "passed",
    }
    for key, expected in smoke_required.items():
        if smoke.get(key) != expected:
            raise ValueError(f"smoke field mismatch: {key}")
    expected_projection = smoke["p95_ms"] / 1000 * 540 + 2700
    if abs(smoke["projected_primary_seconds"] - expected_projection) > 1e-9 or smoke["projected_primary_seconds"] > 64800:
        raise ValueError("smoke projection mismatch")

    rubric = csv_rows(data["run/rubric_scores.csv"])
    rubric_inputs = csv_rows(packet["input/rubric_inputs.csv"])
    if len(rubric) != 80 or len(rubric_inputs) != 80:
        raise ValueError("rubric coverage mismatch")
    if [row["rubric_case_id"] for row in rubric] != [row["rubric_case_id"] for row in rubric_inputs]:
        raise ValueError("rubric order mismatch")
    rubric_config_hash = sha_bytes(packet["config/round2_dgx_gptoss120b_rubric_delta_v1.json"])
    rubric_base = registration["optional_rubric_registration"]["frozen_base_protocol"]
    for output, source in zip(rubric, rubric_inputs):
        score = int(output["score_1_to_5"])
        if not 1 <= score <= 5:
            raise ValueError("rubric score range mismatch")
        prompt = rubric_base["prompt"]["system"] + "\n\n" + rubric_base["prompt"]["user_template"].format(sent_text=source["sentence_text"])
        if output["prompt_sha256"] != sha_bytes(prompt.encode()):
            raise ValueError("rubric prompt hash mismatch")
        if output["config_sha256"] != rubric_config_hash:
            raise ValueError("rubric config hash mismatch")
        if output["model_identity"] != smoke_required["response_model_identity"] or output["run_id"] != config["run_id"]:
            raise ValueError("rubric run/model binding mismatch")

    actions = data["run/operator_actions.jsonl"]
    if actions or receipt["operator_actions"] != {"count": 0, "final_chain_sha256": None}:
        raise ValueError("operator action accounting mismatch")

    attempt_start = min(parse_utc(row["started_at_utc"]) for row in attempts)
    attempt_end = max(parse_utc(row["started_at_utc"]) + timedelta(milliseconds=row["elapsed_ms"]) for row in attempts)
    rubric_start = min(parse_utc(row["started_at_utc"]) for row in rubric)
    rubric_end = max(parse_utc(row["started_at_utc"]) + timedelta(milliseconds=int(row["elapsed_ms"])) for row in rubric)
    receipt_end = parse_utc(receipt["ended_at_utc"])
    timestamp_overlap_ms = max(0, int((attempt_end - rubric_start).total_seconds() * 1000))
    # started_at_utc is serialized only to whole seconds while elapsed_ms keeps
    # millisecond precision.  The sealed runner is sequential, so tolerate at
    # most the one-second representational overlap implied by that truncation.
    if not (parse_utc(smoke["collected_at_utc"]) <= attempt_start <= attempt_end and rubric_start <= rubric_end):
        raise ValueError("derived chronology order mismatch")
    if timestamp_overlap_ms > 1000:
        raise ValueError("primary/rubric chronology exceeds timestamp precision allowance")
    if receipt_end > rubric_end:
        raise ValueError("receipt no longer precedes rubric completion as expected")

    runtime_log = data["run/runtime_redacted.log"]
    model_blob = config["bindings"]["model_blob_sha256"].encode()
    model_manifest = config["bindings"]["model_manifest_sha256"].encode()
    runtime_archive = config["bindings"]["runtime_archive_sha256"].encode()
    source_values = [row["source_text"].encode() for row in primary] + [row["sentence_text"].encode() for row in rubric_inputs]
    candidate_values = [row["candidate_text"].encode() for row in attempts]
    exact_scientific_text_hits = sum(value in runtime_log for value in source_values + candidate_values)
    private_path_hits = len(re.findall(rb"(?:[A-Za-z]:\\Users\\|/home/|/Users/)[^\s\"']+", runtime_log, re.I))

    shared_unchanged = len(unchanged) == len(config["unchanged_output_members"])
    fully_resolved_blockers = [
        "generation_attempts_missing_from_return",
        "model_smoke_report_missing_from_return",
    ]
    strengthened_not_fully_resolved_blockers = [
        "post_acquisition_model_identity_evidence_missing_from_return",
    ]
    later_bound_not_receipt_resolved_blockers = [
        "rubric_output_not_bound_by_run_receipt",
        "rubric_timestamps_extend_past_run_receipt_end",
    ]
    remaining_limitations = [
        "The original run receipt remains byte-identical and still ends before the optional rubric; the supplement manifest and derived chronology, not the original receipt, bind the rubric.",
        "The small model-manifest file, packet-owned runtime archive, and model blob were not included, so their complete bytes cannot be independently rehashed from this supplement.",
        "Successful manifest-gated execution, exact response tag bindings, and the redacted runtime load-path evidence strengthen run identity, but the stored response digest is derived from the frozen registration rather than returned by the inference API.",
        "The supplement ZIP had no adjacent partner-produced checksum; its author-side computed digest is frozen in this audit.",
        "The returned runtime log contains absolute-path patterns despite its redacted label; it remains ignored raw evidence and no path string is copied into tracked artifacts.",
        "The previously documented remote admission-check and candidate-ID namespace deviations remain unchanged.",
    ]
    result = {
        "schema_version": "spectech_dgx_run_supplement_audit_v1",
        "study_id": config["study_id"],
        "run_id": config["run_id"],
        "disposition": "accepted_with_author_provenance_waiver",
        "supplement_status": "partial_provenance_recovery",
        "basis": "preserved_no_rerun_full_run_directory",
        "content_blind": True,
        "model_calls_or_returned_code_executed": False,
        "supplement": {
            "sha256": sha_file(args.zip),
            "size_bytes": args.zip.stat().st_size,
            "adjacent_checksum_present": config["supplement"]["adjacent_checksum_present"],
            "archive_safety": safety,
            "members": member_records,
        },
        "preservation": {
            "original_audit_sha256": sha_file(args.original_audit),
            "original_audit_disposition": "scientific_eligibility_withheld",
            "prior_acceptance_sha256": sha_file(args.prior_acceptance),
            "prior_acceptance_disposition": "accepted_with_author_provenance_waiver",
            "original_return_sha256": sha_file(args.original_return),
            "original_receipt_sha256": sha_bytes(data["run/run_receipt.json"]),
            "unchanged_original_return_members": len(unchanged),
            "scientific_outputs_unchanged": shared_unchanged,
        },
        "run_accounting": {
            "attempts": len(attempts),
            "accepted": sum(row["status"] == "accepted" for row in attempts),
            "rejected": sum(row["status"] != "accepted" for row in attempts),
            "retries": sum(row["attempt_index"] > 1 for row in attempts),
            "candidate_slots": dict(sorted(Counter(row["candidate_slot"] for row in attempts).items())),
            "state_accepted": len(state["accepted"]),
            "resume_count": state["resume_count"],
            "attempt_log_receipt_hash_match": True,
            "attempt_to_retained_bijection": True,
            "seed_prompt_request_input_candidate_hashes": "passed",
            "reasoning_text_retained": False,
            "reasoning_count_and_hash_rows": reasoning_hash_valid,
        },
        "smoke": {
            "fixture_count": smoke["fixture_count"],
            "structured_valid": smoke["structured_valid"],
            "direction_count": smoke["direction_count"],
            "response_tag_and_registered_digest_binding": "passed",
            "reasoning_text_retained": False,
            "intentional_interruption_test": smoke["intentional_interruption_test"],
            "exact_resume_test": smoke["exact_resume_test"],
            "p50_ms": smoke["p50_ms"],
            "p95_ms": smoke["p95_ms"],
            "projected_primary_seconds": smoke["projected_primary_seconds"],
            "projection_gate": smoke["primary_projection_gate"],
        },
        "rubric": {
            "rows": len(rubric),
            "ordered_identity_prompt_config_model_bindings": "passed",
            "member_sha256": sha_bytes(data["run/rubric_scores.csv"]),
            "derived_supplement_hash_binding": True,
        },
        "derived_chronology": {
            "run_started_at_utc": receipt["started_at_utc"],
            "smoke_collected_at_utc": smoke["collected_at_utc"],
            "primary_first_started_at_utc": iso_millis(attempt_start),
            "primary_last_completed_at_utc": iso_millis(attempt_end),
            "original_receipt_ended_at_utc": receipt["ended_at_utc"],
            "rubric_first_started_at_utc": iso_millis(rubric_start),
            "derived_full_run_completed_at_utc": iso_millis(rubric_end),
            "original_receipt_ends_before_rubric_completion": receipt_end < rubric_end,
            "primary_rubric_apparent_overlap_ms_from_second_precision": timestamp_overlap_ms,
        },
        "model_runtime_evidence": {
            "registered_runtime_version": config["bindings"]["runtime_version"],
            "registered_model_manifest_sha256": config["bindings"]["model_manifest_sha256"],
            "registered_model_blob_sha256": config["bindings"]["model_blob_sha256"],
            "runtime_log_sha256": sha_bytes(runtime_log),
            "runtime_log_contains_registered_model_blob_path": model_blob in runtime_log,
            "runtime_log_contains_registered_model_tag": b"gpt-oss:120b" in runtime_log,
            "runtime_log_contains_registered_runtime_version": config["bindings"]["runtime_version"].encode() in runtime_log,
            "runtime_log_contains_manifest_digest": model_manifest in runtime_log,
            "runtime_log_contains_runtime_archive_digest": runtime_archive in runtime_log,
            "model_manifest_file_present": False,
            "model_blob_file_present": False,
            "runtime_archive_file_present": False,
            "independent_whole_artifact_rehash": "not_available",
        },
        "privacy": {
            "runtime_log_exact_source_candidate_or_rubric_text_hits": exact_scientific_text_hits,
            "runtime_log_private_absolute_path_pattern_hits": private_path_hits,
            "tracked_outputs_contain_raw_scientific_text_or_scores": False,
        },
        "fully_resolved_original_blockers": fully_resolved_blockers,
        "strengthened_not_fully_resolved_blockers": strengthened_not_fully_resolved_blockers,
        "later_bound_not_receipt_resolved_blockers": later_bound_not_receipt_resolved_blockers,
        "remaining_limitations": remaining_limitations,
        "paper_assessment": {
            "narrow_provenance_update_recommended": True,
            "reason": "The manuscript currently describes attempt and post-load records as absent; the recovered supplement makes that wording stale while leaving the whole-artifact rehash limitation.",
            "manuscript_edited_in_this_sprint": False,
        },
    }
    if args.extract_dir:
        safe_extract(archive, infos, args.extract_dir)
        result["supplement"]["validated_extraction_created"] = True
        result["supplement"]["validated_extraction_name"] = args.extract_dir.name
    archive.close()
    return result


def write_outputs(output_dir: Path, summary: dict[str, Any], config: Path, schema: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "supplement_audit_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    chronology_path = output_dir / "derived_chronology.json"
    chronology_path.write_text(json.dumps({"schema_version": "spectech_dgx_derived_chronology_v1", "run_id": summary["run_id"], **summary["derived_chronology"]}, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    readme_path = output_dir / "README.md"
    limitations = "\n".join(f"- {item}" for item in summary["remaining_limitations"])
    readme_path.write_text(
        "# GPT-OSS-120B run-1 supplement audit\n\n"
        "## Disposition\n\n"
        "The preserved no-rerun run directory passes content-blind ingestion as a `partial_provenance_recovery`. "
        "The scientific-use disposition remains `accepted_with_author_provenance_waiver` because the supplement does "
        "not fully resolve the frozen whole-artifact and receipt requirements. The original withheld audit and author-"
        "waiver acceptance remain unchanged as the historical chronology. Candidate and rubric outputs are byte-identical "
        "to the first return.\n\n"
        "## What the supplement resolves\n\n"
        "It restores the 180-row attempt log, smoke report, state file, original receipt, and redacted runtime log. "
        "All 180 candidates were accepted on attempt 1 with no retry or rejection; seeds, prompts, requests, input and "
        "candidate hashes, retained rows, state, receipt totals, and reasoning-nonretention records reconcile. The smoke "
        "records 3/3 structured fixtures, all three directions, exact-resume and interruption checks, the registered model "
        "tag/digest binding, and a passing time projection. A derived hash-bound chronology now includes the exact 80-row rubric.\n\n"
        "## What remains limited\n\n"
        f"{limitations}\n\n"
        "## Paper impact\n\n"
        "No result, rating, estimate, candidate, or manuscript file changed. A later narrow prose correction should replace "
        "the now-stale statement that attempt and post-load records were absent. It should instead say that the full run "
        "directory recovered attempt/smoke/state and run-level identity evidence, while complete runtime/model artifact "
        "rehashing was unavailable.\n",
        encoding="utf-8", newline="\n",
    )
    records = []
    for path, role in ((summary_path, "supplement_audit"), (chronology_path, "derived_chronology"), (readme_path, "human_readable_summary")):
        records.append({"path": path.name, "role": role, "size_bytes": path.stat().st_size, "sha256": sha_file(path)})
    manifest = {
        "schema_version": "spectech_dgx_run_supplement_manifest_v1",
        "run_id": summary["run_id"],
        "disposition": summary["disposition"],
        "bindings": {
            "config_sha256": sha_file(config),
            "schema_sha256": sha_file(schema),
            "supplement_zip_sha256": summary["supplement"]["sha256"],
            "original_return_sha256": summary["preservation"]["original_return_sha256"],
            "original_audit_sha256": summary["preservation"]["original_audit_sha256"],
            "prior_acceptance_sha256": summary["preservation"]["prior_acceptance_sha256"],
        },
        "files": records,
    }
    (output_dir / "supplement_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, type=Path)
    parser.add_argument("--original-return", required=True, type=Path)
    parser.add_argument("--source-packet", required=True, type=Path)
    parser.add_argument("--original-audit", required=True, type=Path)
    parser.add_argument("--prior-acceptance", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--extract-dir", type=Path)
    args = parser.parse_args()
    summary = audit(args)
    write_outputs(args.output_dir, summary, args.config, args.schema)
    print(json.dumps({"status": "passed", "run_id": summary["run_id"], "disposition": summary["disposition"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
