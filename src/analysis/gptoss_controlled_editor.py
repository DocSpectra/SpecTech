"""Frozen GPT-OSS controlled-editor comparison and blinded packet workflow."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from src.analysis.gemma_controlled_editor import (
    GemmaCase,
    _candidate_sides,
    _canonical_sha,
    _prompt_values,
    _read_csv,
    _read_jsonl,
    _running_models,
    _sha_text,
    _utc_now,
    _write_csv,
    derive_attempt_seed,
    gate_edit,
    load_cases,
    stop_ollama_model,
)
from src.analysis.pilot_model_human import sha256_file


SCHEMA_VERSION = "round2_gptoss_controlled_editor_v1"
ATTEMPT_SCHEMA_VERSION = "round2_gptoss_editor_attempt_v1"
FREEZE_RECORD = Path("configs/round2_gptoss_controlled_editor_freeze_record.json")


def _set_pointer(record: dict[str, Any], pointer: str, value: Any) -> None:
    node: dict[str, Any] = record
    parts = pointer.strip("/").split("/")
    for part in parts[:-1]:
        node = node[part]
    node[parts[-1]] = value


def load_protocol(config_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    delta = json.loads(config_path.read_text(encoding="utf-8"))
    if delta.get("schema_version") != SCHEMA_VERSION or delta.get("outcome_blind_method_freeze") is not True:
        raise ValueError("unexpected or unfrozen GPT-OSS protocol")
    config_hash = sha256_file(config_path)
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != config_hash:
        raise ValueError("GPT-OSS freeze record/config mismatch")
    if freeze.get("method_frozen_before_any_project_gptoss_generation") is not True:
        raise ValueError("GPT-OSS pre-generation freeze gate is absent")
    base_ref = delta["scientific_protocol_base"]
    base_path = config_path.parents[1] / base_ref["path"]
    if sha256_file(base_path) != base_ref["sha256"]:
        raise ValueError("frozen Gemma scientific-protocol base changed")
    base = json.loads(base_path.read_text(encoding="utf-8"))
    overrides = delta["authorized_model_runtime_overrides"]
    if set(overrides) != set(delta["allowed_override_pointers"]):
        raise ValueError("GPT-OSS override allowlist mismatch")
    merged = copy.deepcopy(base)
    for pointer, value in overrides.items():
        _set_pointer(merged, pointer, value)
    if merged["schema_version"] != SCHEMA_VERSION:
        raise ValueError("GPT-OSS merged protocol identity mismatch")
    return merged, config_hash, freeze


def _ollama_version() -> str:
    return subprocess.run(["ollama", "--version"], check=True, capture_output=True, text=True).stdout.strip()


def _api_json(method: str, url: str, *, payload: dict[str, Any] | None = None, timeout: float = 30.0) -> dict[str, Any]:
    response = requests.request(method, url, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError(f"non-object Ollama response from {url}")
    return value


def verify_ollama_identity(record: dict[str, Any]) -> dict[str, Any]:
    runtime = record["runtime"]
    version = _ollama_version()
    if runtime["ollama_version"] not in version:
        raise ValueError(f"Ollama version mismatch: {version}")
    root_url = runtime["api_url"].rsplit("/api/", 1)[0]
    tags = _api_json("GET", f"{root_url}/api/tags")
    matches = [row for row in tags.get("models", []) if row.get("name") == runtime["model"]]
    if len(matches) != 1:
        raise ValueError("exactly one pinned GPT-OSS tag is required")
    tag = matches[0]
    if tag.get("digest") != runtime["model_tag_digest"] or int(tag.get("size", -1)) != runtime["installed_size_bytes"]:
        raise ValueError("GPT-OSS tag digest or installed size mismatch")
    details = tag.get("details", {})
    expected = {
        "family": runtime["family"], "parameter_size": runtime["parameter_size"],
        "quantization_level": runtime["quantization"], "format": runtime["format"],
    }
    for key, value in expected.items():
        if details.get(key) != value:
            raise ValueError(f"GPT-OSS tag detail mismatch: {key}")
    show = _api_json("POST", f"{root_url}/api/show", payload={"model": runtime["model"], "verbose": False})
    if int(show.get("model_info", {}).get("general.parameter_count", -1)) != runtime["parameter_count"]:
        raise ValueError("GPT-OSS parameter-count mismatch")
    if set(show.get("capabilities", [])) != set(runtime["required_capabilities"]):
        raise ValueError("GPT-OSS capability mismatch")
    if "temperature                    1" not in show.get("parameters", ""):
        raise ValueError("GPT-OSS default temperature mismatch")
    modelfile = subprocess.run(
        ["ollama", "show", runtime["model"], "--modelfile"], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout
    if f"sha256-{runtime['model_blob_sha256']}" not in modelfile:
        raise ValueError("GPT-OSS blob identity mismatch")
    license_text = subprocess.run(
        ["ollama", "show", runtime["model"], "--license"], check=True, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    ).stdout
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("GPT-OSS license mismatch")
    return {
        "ollama_version_output": version, "model": runtime["model"],
        "model_tag_digest": runtime["model_tag_digest"],
        "installed_size_bytes": runtime["installed_size_bytes"],
        "model_blob_sha256": runtime["model_blob_sha256"],
    }


def require_only_pinned_gptoss_resident(record: dict[str, Any]) -> dict[str, Any]:
    rows = _running_models(record)
    runtime = record["runtime"]
    if len(rows) != 1:
        raise ValueError(f"expected one resident model, observed {len(rows)}")
    row = rows[0]
    if row.get("name") != runtime["model"] or row.get("digest") != runtime["model_tag_digest"]:
        raise ValueError("resident model is not exact pinned GPT-OSS")
    size_vram = int(row.get("size_vram") or 0)
    if size_vram <= 0 or size_vram > 16376 * 1024 * 1024:
        raise ValueError("GPT-OSS residency does not fit the frozen 16 GB hardware gate")
    return {
        "name": row.get("name"), "digest": row.get("digest"), "size": row.get("size"),
        "size_vram": size_vram, "context_length": row.get("context_length"),
    }


def build_generation_request(record: dict[str, Any], case: GemmaCase, attempt_index: int) -> dict[str, Any]:
    runtime = record["runtime"]
    options = dict(runtime["options"])
    options["seed"] = derive_attempt_seed(int(runtime["master_seed"]), case.case_id, attempt_index)
    return {
        "model": runtime["model"],
        "messages": [
            {"role": "system", "content": record["prompts"]["system"]},
            {"role": "user", "content": record["prompts"]["user_templates"][case.edit_type].format(**_prompt_values(case))},
        ],
        "stream": runtime["stream"], "think": runtime["thinking"],
        "format": runtime["response_schema"], "options": options,
        "keep_alive": runtime["keep_alive"],
    }


def parse_generation_response(payload: dict[str, Any], expected_model: str) -> tuple[str, dict[str, Any]]:
    if payload.get("model") != expected_model:
        raise ValueError("Ollama response model mismatch")
    message = payload.get("message", {})
    content = message.get("content")
    thinking = message.get("thinking")
    if not isinstance(content, str) or not isinstance(thinking, str) or not thinking:
        raise ValueError("GPT-OSS response lacks final content or low-reasoning trace")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"edited_sentence"} or not isinstance(parsed["edited_sentence"], str):
        raise ValueError("response must contain only a string edited_sentence")
    return parsed["edited_sentence"], {
        "response_created_at": str(payload.get("created_at") or ""),
        "done_reason": str(payload.get("done_reason") or ""),
        "load_duration_ns": int(payload.get("load_duration") or 0),
        "prompt_eval_count": int(payload.get("prompt_eval_count") or 0),
        "prompt_eval_duration_ns": int(payload.get("prompt_eval_duration") or 0),
        "eval_count": int(payload.get("eval_count") or 0),
        "eval_duration_ns": int(payload.get("eval_duration") or 0),
        "response_content_sha256": _sha_text(content),
        "thinking_char_count": len(thinking), "thinking_sha256": _sha_text(thinking),
    }


def _warm_pinned_gptoss(record: dict[str, Any], timeout: float) -> dict[str, Any]:
    runtime = record["runtime"]
    request = {
        "model": runtime["model"],
        "messages": [{"role": "user", "content": "Return JSON with edited_sentence equal to: Runtime fixture complete."}],
        "stream": False, "think": runtime["thinking"], "format": runtime["response_schema"],
        "options": {**runtime["options"], "seed": 2026081100}, "keep_alive": runtime["keep_alive"],
    }
    response = requests.post(runtime["api_url"], json=request, timeout=timeout)
    response.raise_for_status()
    _, metadata = parse_generation_response(response.json(), runtime["model"])
    return {"request_sha256": _canonical_sha(request), **metadata, "residency": require_only_pinned_gptoss_resident(record)}


def _validate_resume(rows: list[dict[str, Any]], cases: list[GemmaCase], record: dict[str, Any], config_hash: str) -> None:
    case_map = {case.case_id: case for case in cases}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case = case_map.get(row.get("case_id"))
        if case is None or row.get("config_sha256") != config_hash:
            raise ValueError("invalid GPT-OSS resume identity")
        if row.get("model_tag_digest") != record["runtime"]["model_tag_digest"] or row.get("original_sha256") != case.original_sha256:
            raise ValueError("invalid GPT-OSS resume model/original identity")
        by_case[case.case_id].append(row)
    for values in by_case.values():
        attempts = [int(row["attempt_index"]) for row in values]
        if attempts != list(range(1, len(values) + 1)):
            raise ValueError("GPT-OSS resumed attempts are not contiguous")
        passing = [row for row in values if row["status"] == "accepted"]
        if len(passing) > 1 or (passing and passing[0] is not values[-1]):
            raise ValueError("GPT-OSS resumed first-pass policy violated")


def _review_id(record: dict[str, Any], case_id: str) -> str:
    review = record["blinded_review"]
    value = f"{review['review_id_seed']}:{review['review_id_namespace']}:{case_id}".encode("utf-8")
    encoded = base64.b32encode(hashlib.sha256(value).digest()).decode("ascii").rstrip("=")
    return "GOE-" + encoded[:12]


def _build_packet(record: dict[str, Any], cases: list[GemmaCase], accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = Path(record["outputs"]["raw_directory"])
    sides = _candidate_sides(record, cases)
    review_ids = {case.case_id: _review_id(record, case.case_id) for case in cases}
    ordered = sorted(cases, key=lambda case: _sha_text(f"{record['blinded_review']['row_order_seed']}:{review_ids[case.case_id]}"))
    packet_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for position, case in enumerate(ordered, start=1):
        candidate = accepted[case.case_id]
        candidate_text = candidate["edited_sentence"]
        candidate_side = sides[case.case_id]
        sentence_a = candidate_text if candidate_side == "A" else case.sentence_original
        sentence_b = case.sentence_original if candidate_side == "A" else candidate_text
        review_id = review_ids[case.case_id]
        packet_rows.append({
            "review_row": position, "review_id": review_id, "sentence_a": sentence_a, "sentence_b": sentence_b,
            "factual_correctness_a": "", "factual_correctness_b": "", "semantic_preservation": "",
            "grammar_naturalness_a": "", "grammar_naturalness_b": "", "more_specific": "",
            "confidence": "", "notes": "",
        })
        key_rows.append({
            "review_row": position, "review_id": review_id, "case_id": case.case_id,
            "corpus_id": case.corpus_id, "edit_type": case.edit_type,
            "source_position": case.source_position, "source_sent_id": case.source_sent_id,
            "original_side": "B" if candidate_side == "A" else "A", "candidate_side": candidate_side,
            "retained_attempt": candidate["attempt_index"], "original_sha256": case.original_sha256,
            "candidate_sha256": candidate["edited_sha256"], "sentence_a_sha256": _sha_text(sentence_a),
            "sentence_b_sha256": _sha_text(sentence_b),
        })
    _write_csv(raw / record["outputs"]["review_packet"], packet_rows, record["blinded_review"]["packet_fields"])
    _write_csv(raw / record["outputs"]["review_answer_key"], key_rows)
    _write_csv(raw / record["outputs"]["review_responses"], packet_rows, record["blinded_review"]["packet_fields"])
    instructions = """# Blinded controlled-edit review

Review all 60 rows without trying to infer which sentence is the original or the intended direction. Use only these exact response values:

- factual_correctness_a / factual_correctness_b: correct, incorrect, uncertain, not_assessable
- semantic_preservation: preserved, mostly_preserved, not_preserved, uncertain
- grammar_naturalness_a / grammar_naturalness_b: natural, minor_issue, major_issue, uncertain
- more_specific: A, B, tie, uncertain
- confidence: integer 1 through 5
- notes: optional free text

Judge factual correctness and grammar/naturalness separately for A and B. Judge semantic preservation for the pair. Do not open the answer key before every response is complete. Fill blinded_review_responses.csv; leave blinded_review_packet.csv unchanged.
"""
    (raw / "blinded_review_instructions.md").write_text(instructions, encoding="utf-8")
    return validate_packet_integrity(record, cases, accepted)


def validate_packet_integrity(record: dict[str, Any], cases: list[GemmaCase], accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = Path(record["outputs"]["raw_directory"])
    packet_path = raw / record["outputs"]["review_packet"]
    key_path = raw / record["outputs"]["review_answer_key"]
    response_path = raw / record["outputs"]["review_responses"]
    packet, keys = _read_csv(packet_path), _read_csv(key_path)
    if len(packet) != 60 or len(keys) != 60:
        raise ValueError("packet/key must each contain 60 rows")
    packet_by_id = {row["review_id"]: row for row in packet}
    key_by_id = {row["review_id"]: row for row in keys}
    if len(packet_by_id) != 60 or set(packet_by_id) != set(key_by_id):
        raise ValueError("packet/key review-ID bijection failed")
    case_map = {case.case_id: case for case in cases}
    side_counts: Counter[tuple[str, str, str]] = Counter()
    for review_id, key in key_by_id.items():
        case = case_map.get(key["case_id"])
        if case is None:
            raise ValueError("answer key references unknown case")
        candidate = accepted[case.case_id]
        row = packet_by_id[review_id]
        if int(row["review_row"]) != int(key["review_row"]):
            raise ValueError("packet/key row mismatch")
        if _sha_text(row["sentence_a"]) != key["sentence_a_sha256"] or _sha_text(row["sentence_b"]) != key["sentence_b_sha256"]:
            raise ValueError("packet sentence hash mismatch")
        candidate_text = row["sentence_a"] if key["candidate_side"] == "A" else row["sentence_b"]
        original_text = row["sentence_a"] if key["original_side"] == "A" else row["sentence_b"]
        if _sha_text(candidate_text) != candidate["edited_sha256"] or _sha_text(original_text) != case.original_sha256:
            raise ValueError("packet side reversal failed")
        side_counts[(case.corpus_id, case.edit_type, key["candidate_side"])] += 1
    for corpus, direction in {(case.corpus_id, case.edit_type) for case in cases}:
        if side_counts[(corpus, direction, "A")] != side_counts[(corpus, direction, "B")]:
            raise ValueError("packet candidate-side balance failed")
    if sum(value for key, value in side_counts.items() if key[2] == "A") != 30:
        raise ValueError("packet overall candidate-side balance failed")
    if set(record["blinded_review"]["packet_forbidden_fields"]).intersection(packet[0]):
        raise ValueError("blinded packet leaks a forbidden field")
    return {
        "packet_rows": 60, "key_rows": 60, "response_template_rows": len(_read_csv(response_path)),
        "unique_review_ids": 60, "candidate_side_a": 30, "candidate_side_b": 30,
        "packet_sha256": sha256_file(packet_path), "answer_key_sha256": sha256_file(key_path),
        "response_template_sha256": sha256_file(response_path),
        "instructions_sha256": sha256_file(raw / "blinded_review_instructions.md"),
        "reversible_key_integrity": True,
    }


def _write_compact(record: dict[str, Any], config_hash: str, cases: list[GemmaCase], rows: list[dict[str, Any]], accepted: dict[str, dict[str, Any]], packet: dict[str, Any] | None, residency: dict[str, Any], started: str, completed: str) -> dict[str, Any]:
    raw = Path(record["outputs"]["raw_directory"])
    compact = Path(record["outputs"]["compact_directory"])
    compact.mkdir(parents=True, exist_ok=True)
    planned: Counter[tuple[str, str]] = Counter((case.corpus_id, case.edit_type) for case in cases)
    retained: Counter[tuple[str, str]] = Counter((case.corpus_id, case.edit_type) for case in cases if case.case_id in accepted)
    coverage_rows = [
        {"corpus_id": corpus, "edit_type": direction, "planned": count,
         "retained": retained[(corpus, direction)], "retention_rate": f"{retained[(corpus, direction)] / count:.6f}"}
        for (corpus, direction), count in sorted(planned.items())
    ]
    failures: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        for reason in row["reason_codes"]:
            failures[(row["corpus_id"], row["edit_type"], reason)] += 1
    failure_rows = [
        {"corpus_id": corpus, "edit_type": direction, "reason_code": reason, "count": count}
        for (corpus, direction, reason), count in sorted(failures.items())
    ]
    _write_csv(compact / "generation_coverage.csv", coverage_rows)
    _write_csv(compact / "generation_failure_reasons.csv", failure_rows, ["corpus_id", "edit_type", "reason_code", "count"])
    raw_hashes = {
        "attempts_sha256": sha256_file(raw / record["outputs"]["attempts"]),
        "retained_candidates_sha256": sha256_file(raw / record["outputs"]["retained_candidates"]),
    }
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    manifest = {
        "schema_version": "round2_gptoss_controlled_editor_packet_manifest_v1",
        "config_sha256": config_hash, "method_freeze_commit": freeze["method_freeze_commit"],
        "model": record["runtime"]["model"], "model_tag_digest": record["runtime"]["model_tag_digest"],
        "model_blob_sha256": record["runtime"]["model_blob_sha256"],
        "started_at_utc": started, "completed_at_utc": completed,
        "planned_cases": len(cases), "retained_cases": len(accepted), "attempt_rows": len(rows),
        "generation_gate_passed": len(accepted) == int(record["generation_gate"]["required_retained_total"]),
        "packet_released": packet is not None, "residency_after_generation": residency,
        "raw_hashes": {**raw_hashes, **(packet or {})},
        "reasoning_effort": "low", "reasoning_text_persisted": False,
        "scoring_performed": False, "rubric_performed": False, "human_review_performed": False,
        "content_boundary": "No sentence text, reasoning text, A/B answer, review response, score, label, participant field, or host path is tracked.",
    }
    (compact / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = f"""# GPT-OSS controlled-editor comparison generation

- Model: `{record['runtime']['model']}` (`{record['runtime']['model_list_id']}`)
- Planned/retained: {len(cases)}/{len(accepted)}
- Attempt rows: {len(rows)}
- Generation gate: {'passed' if manifest['generation_gate_passed'] else 'failed'}
- Blinded packet: {'released with 60 rows and reversible key integrity' if packet else 'not released'}
- Scoring, rubric, and human review: not run

This tracked directory contains aggregate non-content diagnostics only. Raw attempts, generated text, reasoning text, source text, A/B packet, answer key, and review template remain ignored under `outputs/`. Mechanical acceptance is not factual, semantic, grammatical, or manipulation validation.
"""
    (compact / "README.md").write_text(readme, encoding="utf-8")
    manifest["compact_hashes"] = {
        name: sha256_file(compact / name)
        for name in ("generation_coverage.csv", "generation_failure_reasons.csv", "README.md")
    }
    (compact / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def generate(record: dict[str, Any], config_hash: str, cases: list[GemmaCase], *, resume: bool, timeout: float) -> dict[str, Any]:
    raw = Path(record["outputs"]["raw_directory"])
    raw.mkdir(parents=True, exist_ok=True)
    attempts_path = raw / record["outputs"]["attempts"]
    existing = _read_jsonl(attempts_path)
    if existing and not resume:
        raise ValueError("GPT-OSS outputs exist; use --resume only for an interrupted frozen run")
    _validate_resume(existing, cases, record, config_hash)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        by_case[row["case_id"]].append(row)
    with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            prior = by_case[case.case_id]
            if any(row["status"] == "accepted" for row in prior):
                continue
            for attempt_index in range(len(prior) + 1, int(record["runtime"]["maximum_attempts_per_case"]) + 1):
                request = build_generation_request(record, case, attempt_index)
                try:
                    response = requests.post(record["runtime"]["api_url"], json=request, timeout=timeout)
                    response.raise_for_status()
                    edited, response_meta = parse_generation_response(response.json(), record["runtime"]["model"])
                    reasons, metrics = gate_edit(record, case.sentence_original, edited, case.edit_type)
                except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                    edited, metrics = "", {}
                    reasons = [f"request_or_parse_error:{type(exc).__name__}"]
                    response_meta = {
                        "response_created_at": "", "done_reason": "", "load_duration_ns": 0,
                        "prompt_eval_count": 0, "prompt_eval_duration_ns": 0, "eval_count": 0,
                        "eval_duration_ns": 0, "response_content_sha256": _sha_text(""),
                        "thinking_char_count": 0, "thinking_sha256": _sha_text(""),
                    }
                stripped = edited.strip()
                row = {
                    "schema_version": ATTEMPT_SCHEMA_VERSION, "config_sha256": config_hash,
                    "case_id": case.case_id, "corpus_id": case.corpus_id, "source_position": case.source_position,
                    "source_sent_id": case.source_sent_id, "edit_type": case.edit_type,
                    "original_sha256": case.original_sha256, "attempt_index": attempt_index,
                    "seed": request["options"]["seed"], "model": record["runtime"]["model"],
                    "model_tag_digest": record["runtime"]["model_tag_digest"],
                    "request_sha256": _canonical_sha(request), "edited_sentence": stripped,
                    "edited_sha256": _sha_text(stripped), "status": "accepted" if not reasons else "rejected",
                    "reason_codes": reasons, "gate_metrics": metrics, **response_meta,
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n")
                handle.flush(); by_case[case.case_id].append(row)
                if not reasons:
                    break
    rows = _read_jsonl(attempts_path)
    _validate_resume(rows, cases, record, config_hash)
    accepted = {row["case_id"]: row for row in rows if row["status"] == "accepted"}
    retained_rows = [
        {"case_id": case.case_id, "corpus_id": case.corpus_id, "source_position": case.source_position,
         "source_sent_id": case.source_sent_id, "edit_type": case.edit_type,
         "original_sha256": case.original_sha256, "attempt_index": accepted[case.case_id]["attempt_index"],
         "seed": accepted[case.case_id]["seed"], "edited_sentence": accepted[case.case_id]["edited_sentence"],
         "edited_sha256": accepted[case.case_id]["edited_sha256"]}
        for case in cases if case.case_id in accepted
    ]
    _write_csv(raw / record["outputs"]["retained_candidates"], retained_rows)
    packet = _build_packet(record, cases, accepted) if len(accepted) == 60 else None
    return {"rows": rows, "accepted": accepted, "packet": packet}


def run_controlled(config_path: Path, *, resume: bool = False, timeout: float = 300.0) -> dict[str, Any]:
    record, config_hash, freeze = load_protocol(config_path)
    cases = load_cases(record)
    identity = verify_ollama_identity(record)
    started = _utc_now()
    for model in ("qwen3:14b", "gemma4:12b", record["runtime"]["model"]):
        stop_ollama_model(model)
    try:
        warmup = _warm_pinned_gptoss(record, timeout)
        result = generate(record, config_hash, cases, resume=resume, timeout=timeout)
        residency = require_only_pinned_gptoss_resident(record)
        completed = _utc_now()
        manifest = _write_compact(
            record, config_hash, cases, result["rows"], result["accepted"], result["packet"],
            residency, started, completed,
        )
        raw_metadata = {
            "schema_version": "round2_gptoss_controlled_editor_run_metadata_v1",
            "config_sha256": config_hash, "method_freeze_commit": freeze["method_freeze_commit"],
            "started_at_utc": started, "completed_at_utc": completed,
            "identity": identity, "non_project_warmup": warmup, "residency_after_generation": residency,
            "planned_cases": len(cases), "retained_cases": len(result["accepted"]),
            "attempt_rows": len(result["rows"]), "packet_released": result["packet"] is not None,
            "reasoning_effort": "low", "reasoning_text_persisted": False,
            "scoring_performed": False, "rubric_performed": False, "human_review_performed": False,
        }
        raw = Path(record["outputs"]["raw_directory"])
        (raw / record["outputs"]["raw_run_metadata"]).write_text(
            json.dumps(raw_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        stop_ollama_model(record["runtime"]["model"])
        if _running_models(record):
            raise RuntimeError("Ollama model remained resident after GPT-OSS controlled run")


def validate_existing_packet(config_path: Path) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path)
    cases = load_cases(record)
    rows = _read_jsonl(Path(record["outputs"]["raw_directory"]) / record["outputs"]["attempts"])
    accepted = {row["case_id"]: row for row in rows if row["status"] == "accepted"}
    if len(accepted) != 60:
        raise ValueError("no complete GPT-OSS packet exists")
    return validate_packet_integrity(record, cases, accepted)
