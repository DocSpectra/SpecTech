#!/usr/bin/env python3
"""Build the deterministic, small source/config packet from a clean commit."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from export_inputs import export_primary, export_rubric
from packet_core import EXPERIMENT_ID, PACKET_ID, deterministic_tar_gz, sha_bytes, sha_file, write_json

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
GENERATION = ROOT / "configs/round2_dgx_gptoss120b_generation_delta_v1.json"
RUBRIC = ROOT / "configs/round2_dgx_gptoss120b_rubric_delta_v1.json"
FREEZE = ROOT / "configs/round2_dgx_gptoss120b_freeze_record.json"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip()


def registration(commit: str, input_manifest: dict) -> dict:
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    rubric = json.loads(RUBRIC.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    effective = json.loads((ROOT / generation["base_protocol"]["path"]).read_text(encoding="utf-8"))
    for pointer, value in generation["authorized_overrides"].items():
        current = effective; parts = pointer.split("/")[1:]
        for part in parts[:-1]: current = current[part]
        current[parts[-1]] = value
    return {
        "schema_version": "spectech_dgx_experiment_registration_v1",
        "experiment_id": EXPERIMENT_ID, "freeze_id": freeze["method_freeze_commit"],
        "packet_id": PACKET_ID, "source_git_commit": commit,
        "base_bindings": [generation["base_protocol"], *rubric["base_bindings"]],
        "ordered_input_digest": input_manifest["primary"]["ordered_case_sha256"],
        "input_manifest_sha256": sha_bytes(json.dumps(input_manifest, sort_keys=True).encode()),
        "source_case_count": 60, "candidate_count": 180, "rubric_case_count": 80,
        "model_registration": generation["model_registration"],
        "runtime_registration": generation["runtime_registration"],
        "effective_primary_protocol": effective,
        "optional_rubric_registration": {"delta": rubric, "frozen_base_protocol": json.loads((ROOT / "configs/round2_qwen_rubric_v1.json").read_text(encoding="utf-8"))},
        "allowed_delta_pointers": generation["allowed_override_pointers"],
        "resource_policy": generation["resource_policy"], "time_budget": generation["time_budget"],
        "privacy": generation["privacy"], "release": generation["release"],
    }


def payload_sources() -> list[tuple[Path, str, str, int]]:
    result: list[tuple[Path, str, str, int]] = []
    for path in sorted(HERE.glob("*.py")):
        result.append((path, f"scripts/{path.name}", "code", 0o755))
    for name in ("run_all.sh", "README_OPERATOR.md", "TROUBLESHOOTING_RETURN.md"):
        path = HERE / name
        result.append((path, name if name.endswith(".sh") else f"docs/{name}", "code" if name.endswith(".sh") else "doc", 0o755 if name.endswith(".sh") else 0o644))
    for path in sorted((HERE / "fixtures").glob("*")):
        if path.is_file(): result.append((path, f"fixtures/{path.name}", "fixture", 0o644))
    for path in (GENERATION, RUBRIC, FREEZE):
        result.append((path, f"config/{path.name}", "config", 0o644))
    for path in sorted((ROOT / "schemas/dgx_gptoss120b").glob("*.json")):
        result.append((path, f"schemas/{path.name}", "schema", 0o644))
    return result


def build(output: Path) -> tuple[Path, str]:
    if git("status", "--porcelain"):
        raise RuntimeError("packet build requires a clean worktree")
    commit = git("rev-parse", "HEAD")
    epoch = int(git("show", "-s", "--format=%ct", "HEAD"))
    logical_root = f"spectech-dgx-gptoss120b-v1"
    files: list[tuple[str, bytes, int]] = []
    inventory: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="spectech-dgx-export-") as raw:
        temporary = Path(raw)
        input_manifest = {
            "primary": export_primary(ROOT / "configs/round2_human_first_reranking_v1.json", temporary / "primary_inputs.csv"),
            "rubric": export_rubric(ROOT / "configs/round2_qwen_rubric_v1.json", temporary / "rubric_inputs.csv"),
        }
        write_json(temporary / "input_manifest.json", input_manifest)
        write_json(temporary / "experiment_registration.json", registration(commit, input_manifest))
        for source, relative, content_class, mode in payload_sources():
            data = source.read_bytes(); files.append((relative, data, mode))
            inventory.append({"path": relative, "size_bytes": len(data), "sha256": sha_bytes(data), "content_class": content_class})
        for name in ("primary_inputs.csv", "rubric_inputs.csv", "input_manifest.json", "experiment_registration.json"):
            data = (temporary / name).read_bytes(); relative = f"input/{name}"
            files.append((relative, data, 0o600)); inventory.append({"path": relative, "size_bytes": len(data), "sha256": sha_bytes(data), "content_class": "input"})
    generation = json.loads(GENERATION.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "spectech_dgx_packet_manifest_v1", "packet_id": PACKET_ID,
        "packet_version": "v1", "created_at_utc": "2026-08-12T00:00:00Z",
        "source_git_commit": commit, "source_worktree_dirty": False, "source_date_epoch": epoch,
        "experiment_id": EXPERIMENT_ID, "primary_enabled": True, "secondary_rubric_enabled": True,
        "base_bindings": [generation["base_protocol"], *json.loads(RUBRIC.read_text(encoding="utf-8"))["base_bindings"]],
        "model_registration_sha256": sha_bytes(json.dumps(generation["model_registration"], sort_keys=True).encode()),
        "runtime_registration_sha256": sha_bytes(json.dumps(generation["runtime_registration"], sort_keys=True).encode()),
        "payload_files": sorted(inventory, key=lambda item: item["path"]), "network_default": "disabled",
        "privacy_profile": "no_private_host_or_human_label_data_v1",
    }
    files.append(("packet_manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n", 0o644))
    deterministic_tar_gz(output, logical_root, files, epoch)
    digest = sha_file(output); output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="ascii")
    return output, digest


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); path, digest = build(args.output)
    print(json.dumps({"archive": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": digest}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
