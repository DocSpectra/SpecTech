#!/usr/bin/env python3
"""Shared, standard-library-only packet safety and evidence primitives."""
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

PACKET_ID = "spectech-dgx-gptoss120b-v1"
EXPERIMENT_ID = "round2_gptoss120b_generator_replication_v1"
MODEL_DIGEST = "sha256:a951a23b46a1f6093dafee2ea481d634b4e31ac720a8a16f3f91e04f5a40ecd9"
MODEL_BLOB = "sha256:6be6d66a3f546d8c19b130dc41dc24b2fc159f84ffbc76a0ee0676205083cf5a"
RUNTIME_VERSION = "0.32.5"
MAX_NOTE = 500
ACTION_CATEGORIES = {
    "checksum", "archive", "disk", "memory", "docker", "nvidia", "arm64_image",
    "model_load", "network_download", "timeout_process", "structured_interface",
    "checkpoint", "operator_note",
}
REMEDIATIONS = {
    "replace-transfer": ("checksum", "acquisition_path"),
    "clear-marker-cache": ("disk", "packet_cache"),
    "restart-runtime": ("model_load", "packet_runtime"),
    "release-stale-lock": ("checkpoint", "checkpoint_resume"),
}
FORBIDDEN_TEXT = ("human_label", "predictor_score", "author_edit", "authorization", "private_key", "aws_secret", "password")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False).encode("utf-8") + b"\n")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_relative(value: str) -> PurePosixPath:
    if "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"unsafe path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe path: {value!r}")
    return path


def require_under(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes declared root: {candidate}")
    return candidate


def privacy_scan_bytes(data: bytes, label: str) -> None:
    text = data.decode("utf-8", errors="ignore").casefold()
    for needle in FORBIDDEN_TEXT:
        serialized_key = rf"(?m)^\s*[\"']?{re.escape(needle)}[\"']?\s*[:=,]\s*(?!false\b|null\b|none\b)"
        if re.search(serialized_key, text):
            raise ValueError(f"private value in {label}: {needle}")
    absolute_value = r"(?m)^\s*(?:[\"'][^\"']+[\"']\s*:\s*[\"'])?(?:[A-Za-z]:\\Users\\|/home/|/Users/)[^\s\"']+"
    if re.search(absolute_value, text, re.I):
        raise ValueError(f"private absolute path in {label}")


def validate_schema(value: Any, schema: dict[str, Any], where: str = "$") -> None:
    if "$ref" in schema:
        raise ValueError(f"unresolved $ref at {where}")
    if "const" in schema and value != schema["const"]:
        raise ValueError(f"const mismatch at {where}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"enum mismatch at {where}")
    types = schema.get("type")
    if types:
        types = [types] if isinstance(types, str) else types
        checks = {
            "object": lambda x: isinstance(x, dict), "array": lambda x: isinstance(x, list),
            "string": lambda x: isinstance(x, str), "integer": lambda x: isinstance(x, int) and not isinstance(x, bool),
            "number": lambda x: isinstance(x, (int, float)) and not isinstance(x, bool),
            "boolean": lambda x: isinstance(x, bool), "null": lambda x: x is None,
        }
        if not any(checks[item](value) for item in types):
            raise ValueError(f"type mismatch at {where}: expected {types}")
    if isinstance(value, dict):
        required = schema.get("required", [])
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"missing at {where}: {missing}")
        properties = schema.get("properties", {})
        if schema.get("additionalProperties") is False:
            extra = sorted(set(value) - set(properties))
            if extra:
                raise ValueError(f"additional properties at {where}: {extra}")
        for key, item in value.items():
            if key in properties:
                validate_schema(item, properties[key], f"{where}.{key}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ValueError(f"too few items at {where}")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                validate_schema(item, item_schema, f"{where}[{index}]")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            raise ValueError(f"short string at {where}")
        if "pattern" in schema and re.search(schema["pattern"], value) is None:
            raise ValueError(f"pattern mismatch at {where}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < schema.get("minimum", value):
            raise ValueError(f"below minimum at {where}")
        if value > schema.get("maximum", value):
            raise ValueError(f"above maximum at {where}")


def deterministic_tar_gz(output: Path, logical_root: str, files: Iterable[tuple[str, bytes, int]], epoch: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for relative, data, mode in sorted(files):
            safe_relative(relative)
            info = tarfile.TarInfo(f"{logical_root}/{relative}")
            info.size = len(data); info.mtime = epoch; info.mode = mode
            info.uid = 0; info.gid = 0; info.uname = ""; info.gname = ""
            archive.addfile(info, io.BytesIO(data))
    with output.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=epoch, compresslevel=9) as zipped:
            zipped.write(buffer.getvalue())


def inspect_archive(path: Path, expected_root: str | None = None) -> list[tarfile.TarInfo]:
    seen: set[str] = set(); folded: set[str] = set(); members: list[tarfile.TarInfo] = []
    with tarfile.open(path, "r:gz") as archive:
        for member in archive.getmembers():
            safe_relative(member.name)
            if expected_root and member.name.split("/", 1)[0] != expected_root:
                raise ValueError("unexpected logical root")
            if not member.isfile():
                raise ValueError(f"non-regular archive member: {member.name}")
            if member.mode & 0o022 or member.mode & 0o7000:
                raise ValueError(f"unsafe archive mode: {member.name}")
            folded_name = member.name.casefold()
            if member.name in seen or folded_name in folded:
                raise ValueError(f"archive collision: {member.name}")
            seen.add(member.name); folded.add(folded_name); members.append(member)
    return members


def verify_archive(path: Path) -> dict[str, Any]:
    members = inspect_archive(path)
    root = members[0].name.split("/", 1)[0]
    with tarfile.open(path, "r:gz") as archive:
        manifest_member = archive.getmember(f"{root}/packet_manifest.json")
        manifest = json.loads(archive.extractfile(manifest_member).read())
        expected = {item["path"]: item for item in manifest["payload_files"]}
        actual = {member.name.split("/", 1)[1]: member for member in members if member.name != manifest_member.name}
        if set(expected) != set(actual):
            raise ValueError("payload inventory mismatch")
        for relative, record in expected.items():
            data = archive.extractfile(actual[relative]).read()
            if len(data) != record["size_bytes"] or sha_bytes(data) != record["sha256"]:
                raise ValueError(f"payload mismatch: {relative}")
            if record["content_class"] != "input":
                privacy_scan_bytes(data, relative)
    return manifest


def derive_seed(master: int, namespace: str, case_id: str, slot: int, attempt: int) -> int:
    material = f"{master}:{namespace}:{case_id}:{slot}:{attempt}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big") & 0x7FFFFFFF


def append_action(path: Path, run_id: str, run_started: float, *, stage: str, event_type: str,
                  category: str, command_id: str, status: str, action_id: str | None = None,
                  note: str | None = None, failure_code: str | None = None,
                  state_change: str = "none", result_summary: str = "recorded") -> dict[str, Any]:
    if category not in ACTION_CATEGORIES:
        raise ValueError("unregistered diagnostic category")
    if note is not None:
        note = note.strip()
        if not note or len(note) > MAX_NOTE or any(token in note.casefold() for token in FORBIDDEN_TEXT):
            raise ValueError("unsafe operator note")
    records = []
    if path.exists():
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    previous = records[-1]["event_sha256"] if records else None
    body = {
        "schema_version": "spectech_dgx_operator_action_v1", "run_id": run_id,
        "event_sequence": len(records) + 1, "event_id": f"event-{len(records)+1:04d}",
        "previous_event_sha256": previous, "recorded_at_utc": utc_now(),
        "elapsed_from_run_start_ms": max(0, int((time.monotonic() - run_started) * 1000)),
        "stage": stage, "event_type": event_type, "failure_code": failure_code,
        "category": category, "registered_command_id": command_id,
        "registered_action_id": action_id, "sanitized_parameters": {},
        "target_scope": "packet_owned" if category not in {"nvidia", "memory", "docker"} else "read_only_host_probe",
        "status": status, "return_code": 0 if status == "passed" else 1,
        "duration_ms": 0, "state_change": state_change,
        "identity_before_sha256": None, "identity_after_sha256": None,
        "checkpoint_sha256": None, "operator_note": note,
        "redaction_applied": True, "result_summary": result_summary[:200],
    }
    body["event_sha256"] = sha_bytes(canonical(body))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(body, sort_keys=True, ensure_ascii=False) + "\n")
    return body


def verify_action_chain(path: Path) -> tuple[int, str | None]:
    previous = None; count = 0
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        item = json.loads(line); recorded = item.pop("event_sha256")
        if item["previous_event_sha256"] != previous or sha_bytes(canonical(item)) != recorded:
            raise ValueError("operator action chain mismatch")
        previous = recorded; count += 1
    return count, previous


def operator_report(actions: Path, receipt: dict[str, Any]) -> str:
    count, head = verify_action_chain(actions)
    return "\n".join([
        "# Operator report", "", f"## Decision\n\n{receipt['final_decision_branch']} ({receipt['release_status']}).",
        f"## Budget\n\nDiagnostic seconds used: {receipt['diagnostic_budget']['used_seconds']}; remediation actions: {receipt['diagnostic_budget']['remediation_actions_used']}.",
        f"## Timeline\n\n{count} registered events; final chain hash `{head or 'none'}`.",
        "## Diagnoses\n\nSee the sanitized hash-chained action log.",
        "## Remediations\n\nOnly registry-owned packet-scoped actions are permitted.",
        "## Resume\n\nResume count is recorded in the receipt.",
        "## Redaction\n\nRaw host output, credentials, private paths, sentence text, and reasoning text are excluded.",
        "## Integrity\n\nAction chain and receipt hashes are listed in the return manifest.",
        "## Question after return\n\nNone.", "",
    ])
