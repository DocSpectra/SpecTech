"""Frozen Qwen edit-source generation, scoring, and paired analysis for QE-A."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests

from src.analysis.pilot_model_human import sha256_file
from src.ko_specificity.io import KoInputRow, parse_predictions
from src.speciteller.config import DEFAULT_SPECITELLER_CONFIG
from src.speciteller.runner import preflight_speciteller, run_speciteller
from src.speciteller.tokenize import tokenize_for_speciteller


SCHEMA_VERSION = "round2_qwen_edit_source_v1"
RESULTS_VERSION = "round2_qwen_edit_source_results_v1"
FREEZE_RECORD = Path("configs/round2_qwen_edit_source_freeze_record.json")
EDIT_TYPES = ("add_specific", "de_specify", "irrelevant_rewrite")
PRIMARY_MODELS = (
    "speciteller_frozen_round1",
    "ko_run01",
    "ko_run02",
    "ko_run03",
    "granuscore_native",
)
SECONDARY_MODEL = "ko_three_run_arithmetic_mean"

STOPWORDS = frozenset(
    "a an and are as at be been being but by for from had has have he her hers him his i if in into is it its me my no nor not of on or our ours she so than that the their theirs them they this those to too us was we were what when where which who why will with you your yours".split()
)
WORD_RE = re.compile(r"(?u)\b\w+(?:[-./:]\w+)*\b")
URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
PATH_RE = re.compile(r"(?<!\w)(?:[A-Za-z]:\\[^\s,;]+|/(?:[^\s,;]+))")
FLAG_RE = re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9-]*")
NUMBER_RE = re.compile(r"(?<!\w)\d+(?:\.\d+)*(?!\w)")
SNAKE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b")
CAMEL_RE = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b")
BACKTICK_RE = re.compile(r"`+([^`]+?)`+")


@dataclass(frozen=True)
class EditCase:
    case_id: str
    corpus_id: str
    source_position: int
    source_sent_id: str
    edit_type: str
    sentence_original: str
    sentence_author_edited: str
    original_sha256: str


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def case_id(corpus_id: str, sent_id: str, edit_type: str, position: int) -> str:
    return _sha_text(f"{corpus_id}\0{sent_id}\0{edit_type}\0{position}")


def _portable(path: Path) -> str:
    value = path.as_posix()
    if path.is_absolute() or ":/" in value:
        raise ValueError(f"path is not repository-relative: {path}")
    return value


def _require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def load_protocol(config_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION or record.get("outcome_blind_freeze") is not True:
        raise ValueError("unexpected or unfrozen QE-A protocol")
    protocol_hash = sha256_file(config_path)
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != protocol_hash:
        raise ValueError("QE-A freeze record/config mismatch")
    if freeze.get("method_frozen_before_any_project_generation") is not True:
        raise ValueError("QE-A pre-generation freeze gate is absent")
    return record, protocol_hash, freeze


def load_cases(record: dict[str, Any]) -> list[EditCase]:
    cases: list[EditCase] = []
    ordered_digest = hashlib.sha256()
    counts: Counter[tuple[str, str]] = Counter()
    seen: set[str] = set()
    for corpus_id in record["inputs"]["corpus_order"]:
        entry = record["inputs"]["source_files"][corpus_id]
        source = Path(entry["path"])
        _require_hash(source, entry["sha256"])
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != int(entry["rows"]):
            raise ValueError(f"QE-A source row count mismatch: {corpus_id}")
        for position, row in enumerate(rows, start=1):
            if row["corpus_id"] != corpus_id or row["edit_type"] not in EDIT_TYPES:
                raise ValueError("QE-A source corpus/direction mismatch")
            identifier = case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            if identifier in seen or not row["sentence_original"].strip() or not row["sentence_edited"].strip():
                raise ValueError("QE-A duplicate or empty source case")
            seen.add(identifier)
            original_hash = _sha_text(row["sentence_original"])
            digest_row = "\0".join(
                [corpus_id, str(position), identifier, row["sent_id"], row["edit_type"], original_hash]
            )
            ordered_digest.update(digest_row.encode("utf-8"))
            ordered_digest.update(b"\n")
            counts[(corpus_id, row["edit_type"])] += 1
            cases.append(
                EditCase(
                    case_id=identifier,
                    corpus_id=corpus_id,
                    source_position=position,
                    source_sent_id=row["sent_id"],
                    edit_type=row["edit_type"],
                    sentence_original=row["sentence_original"],
                    sentence_author_edited=row["sentence_edited"],
                    original_sha256=original_hash,
                )
            )
        observed = {key: counts[(corpus_id, key)] for key in entry["direction_counts"]}
        if observed != entry["direction_counts"]:
            raise ValueError(f"QE-A direction counts changed: {corpus_id}")
    if len(cases) != int(record["inputs"]["required_total_cases"]):
        raise ValueError("QE-A planned-case count mismatch")
    if ordered_digest.hexdigest() != record["inputs"]["ordered_case_sha256"]:
        raise ValueError("QE-A ordered-case digest mismatch")
    return cases


def verify_ollama_identity(record: dict[str, Any]) -> dict[str, str]:
    qwen = record["qwen"]
    version = subprocess.run(["ollama", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if qwen["ollama_version"] not in version:
        raise ValueError(f"Ollama version mismatch: {version}")
    listing = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True).stdout
    rows = [line for line in listing.splitlines() if line.split()[:1] == [qwen["model"]]]
    if len(rows) != 1 or qwen["model_list_id"] not in rows[0]:
        raise ValueError("Qwen model ID mismatch")
    modelfile = subprocess.run(
        ["ollama", "show", qwen["model"], "--modelfile"],
        check=True, capture_output=True, text=True,
    ).stdout
    if f"sha256-{qwen['backing_blob_sha256']}" not in modelfile:
        raise ValueError("Qwen backing-blob mismatch")
    return {
        "ollama_version": qwen["ollama_version"],
        "ollama_version_output": version,
        "model": qwen["model"],
        "model_list_id": qwen["model_list_id"],
        "backing_blob_sha256": qwen["backing_blob_sha256"],
    }


def _docker_image_id(image: str) -> str:
    return subprocess.run(
        ["docker", "inspect", image, "--format", "{{.Id}}"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def verify_preflight(config_path: Path, *, inspect_containers: bool = True) -> dict[str, Any]:
    record, protocol_hash, freeze = load_protocol(config_path)
    cases = load_cases(record)
    identity = verify_ollama_identity(record)
    artifact_count = 0
    for entry in record["scoring"]["speciteller"]["original_score_inputs"].values():
        _require_hash(Path(entry["path"]), entry["sha256"]); artifact_count += 1
    for corpus in record["scoring"]["ko"]["runs"].values():
        for run in corpus.values():
            _require_hash(Path(run["checkpoint_path"]), run["checkpoint_sha256"])
            _require_hash(Path(run["original_score_path"]), run["original_score_sha256"])
            artifact_count += 2
    for key in ("author_pair_manifest", "author_pair_scores"):
        entry = record["scoring"]["granuscore"][key]
        _require_hash(Path(entry["path"]), entry["sha256"]); artifact_count += 1
    images: dict[str, str] = {}
    if inspect_containers:
        for key in ("speciteller", "ko", "granuscore"):
            scorer = record["scoring"][key]
            observed = _docker_image_id(scorer["image"])
            if observed != scorer["image_id"]:
                raise ValueError(f"{key} image mismatch: {observed}")
            images[key] = observed
    return {
        "protocol_sha256": protocol_hash,
        "freeze_commit": freeze["method_freeze_commit"],
        "planned_cases": len(cases),
        "verified_artifacts": artifact_count,
        "qwen": identity,
        "images": images,
    }


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(unicodedata.normalize("NFKC", value))]


def _content_tokens(value: str) -> set[str]:
    return {token for token in _tokens(value) if len(token) >= 3 and token not in STOPWORDS}


def _multiset_from_vocab(value: str, vocabulary: Iterable[str]) -> Counter[str]:
    tokens = _tokens(value)
    vocab = set(vocabulary)
    return Counter(token for token in tokens if token in vocab)


def _markers(value: str) -> Counter[str]:
    found: list[str] = []
    for regex in (URL_RE, PATH_RE, FLAG_RE, NUMBER_RE, SNAKE_RE, CAMEL_RE):
        found.extend(match.group(0).casefold().rstrip(".,;:)\"]'") for match in regex.finditer(value))
    found.extend(match.group(1).strip().casefold() for match in BACKTICK_RE.finditer(value))
    return Counter(token for token in found if token)


def _counter_recall(original: Counter[str], edited: Counter[str]) -> float:
    total = sum(original.values())
    if total == 0:
        return 1.0
    retained = sum(min(count, edited.get(key, 0)) for key, count in original.items())
    return retained / total


def gate_edit(record: dict[str, Any], original: str, edited: str, edit_type: str) -> tuple[list[str], dict[str, float | int]]:
    reasons: list[str] = []
    universal = record["automated_gates"]["universal"]
    stripped = edited.strip()
    if not stripped or "\n" in stripped or "\r" in stripped or "\t" in stripped:
        reasons.append("not_one_nonempty_line")
    if len(stripped) < int(universal["minimum_characters"]):
        reasons.append("too_short_chars")
    if len(stripped) > int(universal["maximum_characters"]):
        reasons.append("too_long_chars")
    if _normalized(stripped).casefold() == _normalized(original).casefold():
        reasons.append("unchanged_normalized")
    if any(stripped.casefold().startswith(prefix) for prefix in universal["forbidden_meta_prefixes_casefold"]):
        reasons.append("meta_prefix")
    original_tokens = _tokens(original)
    edited_tokens = _tokens(stripped)
    ratio = len(edited_tokens) / max(1, len(original_tokens))
    original_content = _content_tokens(original)
    edited_content = _content_tokens(stripped)
    content_recall = len(original_content & edited_content) / max(1, len(original_content))
    original_markers = _markers(original)
    edited_markers = _markers(stripped)
    marker_recall = _counter_recall(original_markers, edited_markers)
    polarity_original = _multiset_from_vocab(original, record["automated_gates"]["polarity_tokens"])
    polarity_edited = _multiset_from_vocab(stripped, record["automated_gates"]["polarity_tokens"])
    modal_original = _multiset_from_vocab(original, record["automated_gates"]["modal_tokens"])
    modal_edited = _multiset_from_vocab(stripped, record["automated_gates"]["modal_tokens"])
    if polarity_original != polarity_edited:
        reasons.append("polarity_changed")
    if modal_original != modal_edited:
        reasons.append("modality_changed")
    rule = record["automated_gates"]["direction_rules"][edit_type]
    if ratio < float(rule["edited_to_original_token_ratio_min"]):
        reasons.append("token_ratio_below_min")
    if ratio > float(rule["edited_to_original_token_ratio_max"]):
        reasons.append("token_ratio_above_max")
    if content_recall < float(rule["original_content_anchor_recall_min"]):
        reasons.append("content_anchor_recall_below_min")
    original_marker_count = sum(original_markers.values())
    edited_marker_count = sum(edited_markers.values())
    if edit_type == "add_specific":
        if marker_recall < float(rule["original_concrete_marker_recall_min"]):
            reasons.append("original_concrete_marker_lost")
        if edited_marker_count < original_marker_count:
            reasons.append("concrete_marker_count_decreased")
        if not (len(edited_tokens) >= len(original_tokens) + 2 or edited_marker_count >= original_marker_count + 1):
            reasons.append("insufficient_specificity_addition_proxy")
    elif edit_type == "de_specify":
        if edited_marker_count > original_marker_count:
            reasons.append("concrete_marker_count_increased")
        if not (len(edited_tokens) <= len(original_tokens) - 2 or edited_marker_count <= original_marker_count - 1):
            reasons.append("insufficient_despecification_proxy")
    else:
        if original_markers != edited_markers:
            reasons.append("neutral_concrete_markers_changed")
    metrics: dict[str, float | int] = {
        "original_token_count": len(original_tokens),
        "edited_token_count": len(edited_tokens),
        "token_ratio": ratio,
        "content_anchor_recall": content_recall,
        "original_concrete_marker_count": original_marker_count,
        "edited_concrete_marker_count": edited_marker_count,
        "original_concrete_marker_recall": marker_recall,
    }
    return sorted(set(reasons)), metrics


def derive_attempt_seed(master_seed: int, identifier: str, attempt_index: int) -> int:
    digest = hashlib.sha256(f"{master_seed}:{identifier}:{attempt_index}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_generation_request(record: dict[str, Any], case: EditCase, attempt_index: int) -> dict[str, Any]:
    qwen = record["qwen"]
    options = dict(qwen["options"])
    options["seed"] = derive_attempt_seed(int(qwen["master_seed"]), case.case_id, attempt_index)
    return {
        "model": qwen["model"],
        "messages": [
            {"role": "system", "content": record["prompts"]["system"]},
            {
                "role": "user",
                "content": record["prompts"]["user_templates"][case.edit_type].format(
                    sentence_original=case.sentence_original
                ),
            },
        ],
        "stream": qwen["stream"],
        "think": qwen["thinking"],
        "format": qwen["response_schema"],
        "options": options,
        "keep_alive": qwen["keep_alive"],
    }


def parse_generation_response(payload: dict[str, Any], expected_model: str) -> tuple[str, dict[str, Any]]:
    if payload.get("model") != expected_model:
        raise ValueError("Ollama response model mismatch")
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response lacks string content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"edited_sentence"}:
        raise ValueError("response must contain only edited_sentence")
    edited = parsed["edited_sentence"]
    if not isinstance(edited, str):
        raise ValueError("edited_sentence must be a string")
    return edited, {
        "created_at": payload.get("created_at"),
        "done_reason": payload.get("done_reason"),
        "prompt_eval_count": payload.get("prompt_eval_count"),
        "eval_count": payload.get("eval_count"),
        "total_duration": payload.get("total_duration"),
        "response_content_sha256": _sha_text(content),
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _validate_resume_rows(rows: list[dict[str, Any]], cases: list[EditCase], record: dict[str, Any], protocol_hash: str) -> None:
    case_map = {case.case_id: case for case in cases}
    seen_attempts: dict[str, list[int]] = defaultdict(list)
    accepted: set[str] = set()
    for row in rows:
        identifier = row.get("case_id")
        case = case_map.get(identifier)
        if case is None or row.get("protocol_sha256") != protocol_hash:
            raise ValueError("invalid resumed generation identity")
        if row.get("model_blob_sha256") != record["qwen"]["backing_blob_sha256"]:
            raise ValueError("invalid resumed Qwen blob identity")
        attempt = int(row["attempt_index"])
        if identifier in accepted or attempt in seen_attempts[identifier]:
            raise ValueError("duplicate or post-acceptance resumed attempt")
        seen_attempts[identifier].append(attempt)
        if row["status"] == "accepted":
            accepted.add(identifier)
    for attempts in seen_attempts.values():
        if attempts != list(range(1, len(attempts) + 1)):
            raise ValueError("resumed attempts are not contiguous")


def _write_generation_exports(record: dict[str, Any], cases: list[EditCase], rows: list[dict[str, Any]]) -> dict[str, Any]:
    raw_dir = Path(record["outputs"]["raw_directory"])
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    manifest_rows = []
    accepted_rows = []
    cell_planned: Counter[tuple[str, str]] = Counter()
    cell_accepted: Counter[tuple[str, str]] = Counter()
    for case in cases:
        attempts = by_case.get(case.case_id, [])
        accepted = next((row for row in attempts if row["status"] == "accepted"), None)
        cell = (case.corpus_id, case.edit_type)
        cell_planned[cell] += 1
        if accepted:
            cell_accepted[cell] += 1
            accepted_rows.append(
                {
                    "case_id": case.case_id,
                    "corpus_id": case.corpus_id,
                    "source_position": case.source_position,
                    "source_sent_id": case.source_sent_id,
                    "edit_type": case.edit_type,
                    "sentence_original": case.sentence_original,
                    "sentence_author_edited": case.sentence_author_edited,
                    "sentence_qwen_edited": accepted["edited_sentence"],
                    "accepted_attempt_index": accepted["attempt_index"],
                    "original_sha256": case.original_sha256,
                    "author_edit_sha256": _sha_text(case.sentence_author_edited),
                    "qwen_edit_sha256": _sha_text(accepted["edited_sentence"]),
                }
            )
        reason_union = sorted({reason for row in attempts for reason in row.get("reason_codes", [])})
        manifest_rows.append(
            {
                "case_id": case.case_id,
                "corpus_id": case.corpus_id,
                "source_position": case.source_position,
                "source_sent_id": case.source_sent_id,
                "edit_type": case.edit_type,
                "original_sha256": case.original_sha256,
                "attempts_generated": len(attempts),
                "accepted": str(accepted is not None).lower(),
                "accepted_attempt_index": "" if accepted is None else accepted["attempt_index"],
                "failure_reason_codes": "|".join(reason_union),
                "qwen_edit_sha256": "" if accepted is None else _sha_text(accepted["edited_sentence"]),
            }
        )
    _write_csv(raw_dir / record["outputs"]["generation_manifest"], manifest_rows)
    if accepted_rows:
        _write_csv(raw_dir / record["outputs"]["accepted_edits"], accepted_rows)
    total_rate = len(accepted_rows) / len(cases)
    min_total = float(record["automated_gates"]["coverage_gate"]["minimum_total_acceptance_rate_for_primary_comparison"])
    min_cell = float(record["automated_gates"]["coverage_gate"]["minimum_cell_acceptance_rate_for_primary_comparison"])
    cells = [
        {
            "corpus_id": corpus,
            "edit_type": edit_type,
            "planned": count,
            "accepted": cell_accepted[(corpus, edit_type)],
            "acceptance_rate": cell_accepted[(corpus, edit_type)] / count,
        }
        for (corpus, edit_type), count in sorted(cell_planned.items())
    ]
    gate_passed = total_rate >= min_total and all(row["acceptance_rate"] >= min_cell for row in cells)
    summary = {
        "planned": len(cases),
        "accepted": len(accepted_rows),
        "failed": len(cases) - len(accepted_rows),
        "attempt_rows": len(rows),
        "acceptance_rate": total_rate,
        "coverage_gate_passed": gate_passed,
        "cells": cells,
    }
    (raw_dir / "generation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def run_generation(config_path: Path, *, resume: bool = False, timeout: float = 300.0) -> dict[str, Any]:
    record, protocol_hash, _ = load_protocol(config_path)
    verify_preflight(config_path)
    cases = load_cases(record)
    raw_dir = Path(record["outputs"]["raw_directory"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / record["outputs"]["raw_attempts"]
    existing = _read_jsonl(raw_path)
    if existing and not resume:
        raise ValueError("generation output exists; use --resume after validation")
    _validate_resume_rows(existing, cases, record, protocol_hash)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        by_case[row["case_id"]].append(row)
    with raw_path.open("a", encoding="utf-8", newline="\n") as handle:
        for case in cases:
            prior = by_case[case.case_id]
            if any(row["status"] == "accepted" for row in prior):
                continue
            for attempt in range(len(prior) + 1, int(record["qwen"]["maximum_attempts_per_case"]) + 1):
                request = build_generation_request(record, case, attempt)
                try:
                    response = requests.post(record["qwen"]["api_url"], json=request, timeout=timeout)
                    response.raise_for_status()
                    edited, metadata = parse_generation_response(response.json(), record["qwen"]["model"])
                    reasons, metrics = gate_edit(record, case.sentence_original, edited, case.edit_type)
                except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                    edited = ""
                    reasons = [f"request_or_parse_error:{type(exc).__name__}"]
                    metrics = {}
                    metadata = {"response_content_sha256": _sha_text("")}
                row = {
                    "case_id": case.case_id,
                    "corpus_id": case.corpus_id,
                    "source_sent_id": case.source_sent_id,
                    "edit_type": case.edit_type,
                    "attempt_index": attempt,
                    "attempt_seed": request["options"]["seed"],
                    "status": "accepted" if not reasons else "rejected",
                    "reason_codes": reasons,
                    "edited_sentence": edited.strip(),
                    "protocol_sha256": protocol_hash,
                    "model_blob_sha256": record["qwen"]["backing_blob_sha256"],
                    **metrics,
                    **metadata,
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                by_case[case.case_id].append(row)
                if not reasons:
                    break
    rows = _read_jsonl(raw_path)
    _validate_resume_rows(rows, cases, record, protocol_hash)
    return _write_generation_exports(record, cases, rows)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: f"{value:.10f}" if isinstance(value, float) else value for key, value in row.items()})


def _read_accepted(record: dict[str, Any], cases: list[EditCase]) -> dict[str, dict[str, str]]:
    path = Path(record["outputs"]["raw_directory"]) / record["outputs"]["accepted_edits"]
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    case_map = {case.case_id: case for case in cases}
    accepted: dict[str, dict[str, str]] = {}
    for row in rows:
        case = case_map.get(row["case_id"])
        if case is None or row["original_sha256"] != case.original_sha256:
            raise ValueError("accepted edit identity mismatch")
        if row["author_edit_sha256"] != _sha_text(case.sentence_author_edited):
            raise ValueError("accepted author edit hash mismatch")
        if row["qwen_edit_sha256"] != _sha_text(row["sentence_qwen_edited"]):
            raise ValueError("accepted Qwen edit hash mismatch")
        accepted[row["case_id"]] = row
    return accepted


def prepare_scoring(config_path: Path) -> dict[str, Any]:
    record, protocol_hash, _ = load_protocol(config_path)
    cases = load_cases(record)
    accepted = _read_accepted(record, cases)
    raw_dir = Path(record["outputs"]["raw_directory"])
    scoring_dir = raw_dir / "scoring"
    manifest_rows: list[dict[str, Any]] = []
    score_texts: list[tuple[str, str, str]] = []
    for case in cases:
        sources = [("author", case.sentence_author_edited)]
        if case.case_id in accepted:
            sources.append(("qwen", accepted[case.case_id]["sentence_qwen_edited"]))
        for source, text in sources:
            score_id = _sha_text(f"{case.case_id}\0{source}\0edited")
            manifest_rows.append(
                {
                    "score_id": score_id,
                    "case_id": case.case_id,
                    "corpus_id": case.corpus_id,
                    "source_sent_id": case.source_sent_id,
                    "edit_type": case.edit_type,
                    "edit_source": source,
                    "text_sha256": _sha_text(text),
                }
            )
            score_texts.append((score_id, case.corpus_id, text))
    manifest_path = raw_dir / record["outputs"]["scoring_manifest"]
    _write_csv(manifest_path, manifest_rows)
    spec_dir = scoring_dir / "speciteller"
    spec_dir.mkdir(parents=True, exist_ok=True)
    with (spec_dir / "input.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for score_id, _, text in score_texts:
            handle.write(f"{score_id}\t{tokenize_for_speciteller(text)}\n")
    gran_dir = scoring_dir / "granuscore"
    gran_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(
        gran_dir / "input.csv",
        [
            {"corpus_id": "qe_controlled_edits", "sent_id": score_id, "sent_text": text}
            for score_id, _, text in score_texts
        ],
    )
    for corpus_id in record["inputs"]["corpus_order"]:
        corpus_rows = [(score_id, text) for score_id, corpus, text in score_texts if corpus == corpus_id]
        bundle = scoring_dir / "ko" / corpus_id / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        query_texts = [text for _, text in corpus_rows]
        (bundle / "twitters.txt").write_text("\n".join([query_texts[0], *query_texts]) + "\n", encoding="utf-8")
        (bundle / "twitterl.txt").write_text("1\n" * (len(query_texts) + 1), encoding="utf-8")
        (bundle / "twitterv.txt").write_text("0.5\n" * (len(query_texts) + 1), encoding="utf-8")
        canonical = Path("outputs/round2/ko_official_release/inputs") / f"{corpus_id}.csv"
        with canonical.open("r", encoding="utf-8", newline="") as source:
            target_rows = list(csv.DictReader(source))
        (bundle / "twitteru.txt").write_text(
            "\n".join(row["text"] for row in target_rows) + "\n", encoding="utf-8"
        )
        _write_csv(
            bundle / "row_map.csv",
            [{"prediction_index": i, "score_id": score_id} for i, (score_id, _) in enumerate(corpus_rows)],
        )
    metadata = {
        "schema_version": "round2_qwen_edit_scoring_preparation_v1",
        "protocol_sha256": protocol_hash,
        "planned_cases": len(cases),
        "accepted_qwen_cases": len(accepted),
        "score_text_rows": len(score_texts),
        "manifest_sha256": sha256_file(manifest_path),
        "speciteller_input_sha256": sha256_file(spec_dir / "input.tsv"),
        "granuscore_input_sha256": sha256_file(gran_dir / "input.csv"),
        "ko_bundles": {
            corpus: {
                name: sha256_file(scoring_dir / "ko" / corpus / "bundle" / name)
                for name in ("twitters.txt", "twitteru.txt", "twitterl.txt", "twitterv.txt", "row_map.csv")
            }
            for corpus in record["inputs"]["corpus_order"]
        },
    }
    (scoring_dir / "preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def score_speciteller(config_path: Path) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path)
    raw_dir = Path(record["outputs"]["raw_directory"])
    directory = raw_dir / "scoring" / "speciteller"
    preflight_speciteller(DEFAULT_SPECITELLER_CONFIG)
    run_speciteller(DEFAULT_SPECITELLER_CONFIG, directory / "input.tsv", directory / "scores.tsv")
    expected = sum(1 for _ in (directory / "input.tsv").open(encoding="utf-8"))
    observed = sum(1 for _ in (directory / "scores.tsv").open(encoding="utf-8"))
    if observed != expected:
        raise ValueError("SpeciTeller QE-A score coverage mismatch")
    return {"rows": observed, "sha256": sha256_file(directory / "scores.tsv")}


def score_granuscore(config_path: Path) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path)
    scorer = record["scoring"]["granuscore"]
    raw_dir = Path(record["outputs"]["raw_directory"])
    directory = raw_dir / "scoring" / "granuscore"
    input_path = directory / "input.csv"
    command = [
        "docker", "run", "--rm", "--gpus", "all",
        "-v", f"{Path.cwd().resolve()}:/work", "-w", "/work",
        scorer["image"],
        "--input", f"/work/{input_path.as_posix()}",
        "--output", f"/work/{(directory / 'scores.csv').as_posix()}",
        "--metadata", f"/work/{(directory / 'scores.metadata.json').as_posix()}",
        "--corpus-id", "qe_controlled_edits",
        "--expected-input-sha256", sha256_file(input_path),
        "--batch-size", "256", "--encoding-batch-size", "256",
    ]
    subprocess.run(command, check=True)
    metadata = json.loads((directory / "scores.metadata.json").read_text(encoding="utf-8"))
    if metadata["runner_sha256"] != scorer["runner_sha256"]:
        raise ValueError("GranuScore QE-A runner mismatch")
    return {"rows": metadata["row_count"], "sha256": metadata["output_sha256"]}


def score_ko_checkpoint(config_path: Path, corpus_id: str, run_id: str) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path)
    ko = record["scoring"]["ko"]
    run = ko["runs"][corpus_id][run_id]
    checkpoint = Path(run["checkpoint_path"])
    _require_hash(checkpoint, run["checkpoint_sha256"])
    raw_dir = Path(record["outputs"]["raw_directory"])
    base = raw_dir / "scoring" / "ko" / corpus_id
    bundle = base / "bundle"
    output = base / run_id
    output.mkdir(parents=True, exist_ok=True)
    shell = (
        "set -euo pipefail; "
        "test \"$(sha256sum /artifacts/glove.840B.300d.txt | cut -d' ' -f1)\" = \"$GLOVE_TXT_SHA256\"; "
        "cp -a /opt/ko /tmp/ko; cd /tmp/ko; "
        "ln -s /artifacts/glove.840B.300d.txt glove.840B.300d.txt; "
        "cp /target/twitters.txt /target/twitteru.txt /target/twitterl.txt /target/twitterv.txt dataset/data/; "
        "cp /checkpoint/model.pickle savedir/3osmodel.pickle; "
        "python test.py --gpu_id 0 --test_data twitter > /output/test.log 2>&1; "
        "cp predictions.txt /output/predictions.txt"
    )
    command = [
        "docker", "run", "--rm",
        "-e", f"GLOVE_TXT_SHA256={ko['glove_text_sha256']}",
        "-v", f"{bundle.resolve()}:/target:ro",
        "-v", f"{output.resolve()}:/output",
        "-v", f"{checkpoint.resolve()}:/checkpoint/model.pickle:ro",
        "-v", f"{ko['glove_volume']}:/artifacts:ro",
        "--entrypoint", "bash", ko["image"], "-lc", shell,
    ]
    subprocess.run(command, check=True)
    with (bundle / "row_map.csv").open("r", encoding="utf-8", newline="") as handle:
        mapping = list(csv.DictReader(handle))
    scores = parse_predictions(output / "predictions.txt", len(mapping))
    _write_csv(
        output / "scores.csv",
        [
            {
                "score_id": row["score_id"], "corpus_id": corpus_id,
                "run_id": run_id, "score_raw": score,
                "checkpoint_sha256": run["checkpoint_sha256"],
            }
            for row, score in zip(mapping, scores)
        ],
    )
    metadata = {
        "schema_version": "round2_qwen_edit_ko_checkpoint_scores_v1",
        "corpus_id": corpus_id,
        "run_id": run_id,
        "row_count": len(scores),
        "checkpoint_sha256": run["checkpoint_sha256"],
        "bundle_hashes": {name: sha256_file(bundle / name) for name in ("twitters.txt", "twitteru.txt", "twitterl.txt", "twitterv.txt", "row_map.csv")},
        "prediction_sha256": sha256_file(output / "predictions.txt"),
        "score_sha256": sha256_file(output / "scores.csv"),
    }
    (output / "scores.metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": len(scores), "sha256": metadata["score_sha256"]}


def _read_speciteller_original(path: Path) -> dict[str, float]:
    output: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                identifier, value = line.rstrip("\n").split("\t")[:2]
                output[identifier] = float(value)
    return output


def _read_csv_map(path: Path, key: str, value: str) -> dict[str, float]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = {row[key]: float(row[value]) for row in rows}
    if len(output) != len(rows):
        raise ValueError(f"duplicate score key in {path}")
    return output


def _stream_seed(master: int, *parts: str) -> int:
    digest = hashlib.sha256((str(master) + ":" + ":".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big")


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    valid = values[np.isfinite(values)]
    if len(valid) == 0:
        return math.nan, math.nan
    return float(np.quantile(valid, 0.025)), float(np.quantile(valid, 0.975))


def _direction_values(deltas: np.ndarray, edit_type: str, model: str) -> np.ndarray:
    if edit_type == "irrelevant_rewrite":
        return np.full(len(deltas), np.nan)
    gran = model == "granuscore_native"
    expected_positive = (edit_type == "add_specific" and not gran) or (edit_type == "de_specify" and gran)
    return (deltas > 0).astype(float) if expected_positive else (deltas < 0).astype(float)


def _load_score_rows(record: dict[str, Any], cases: list[EditCase]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_dir = Path(record["outputs"]["raw_directory"])
    with (raw_dir / record["outputs"]["scoring_manifest"]).open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    manifest_by_id = {row["score_id"]: row for row in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("duplicate QE-A scoring manifest ID")
    spec_new = {}
    with (raw_dir / "scoring/speciteller/scores.tsv").open("r", encoding="utf-8") as handle:
        for line in handle:
            identifier, value = line.rstrip("\n").split("\t")
            spec_new[identifier] = float(value)
    gran_new = _read_csv_map(raw_dir / "scoring/granuscore/scores.csv", "sent_id", "granuscore_percentile")
    gran_no_unit: dict[str, bool] = {}
    with (raw_dir / "scoring/granuscore/scores.csv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            gran_no_unit[row["sent_id"]] = row["no_referential_unit"].lower() == "true"
    ko_new: dict[tuple[str, str], dict[str, float]] = {}
    for corpus_id in record["inputs"]["corpus_order"]:
        for run_id in ("run01", "run02", "run03"):
            ko_new[(corpus_id, run_id)] = _read_csv_map(
                raw_dir / f"scoring/ko/{corpus_id}/{run_id}/scores.csv", "score_id", "score_raw"
            )
    expected_new = set(manifest_by_id)
    for name, mapping in [("speciteller", spec_new), ("granuscore", gran_new)]:
        if set(mapping) != expected_new:
            raise ValueError(f"{name} accepted/edit score join mismatch")
    for key, mapping in ko_new.items():
        expected = {score_id for score_id, row in manifest_by_id.items() if row["corpus_id"] == key[0]}
        if set(mapping) != expected:
            raise ValueError(f"Ko score join mismatch: {key}")
    original_spec = {
        corpus: _read_speciteller_original(Path(entry["path"]))
        for corpus, entry in record["scoring"]["speciteller"]["original_score_inputs"].items()
    }
    original_ko = {
        (corpus, run): _read_csv_map(Path(entry["original_score_path"]), "sent_id", "score_raw")
        for corpus, runs in record["scoring"]["ko"]["runs"].items()
        for run, entry in runs.items()
    }
    gran_manifest_path = Path(record["scoring"]["granuscore"]["author_pair_manifest"]["path"])
    gran_scores_path = Path(record["scoring"]["granuscore"]["author_pair_scores"]["path"])
    with gran_manifest_path.open("r", encoding="utf-8", newline="") as handle:
        gran_manifest = list(csv.DictReader(handle))
    gran_scores = _read_csv_map(gran_scores_path, "sent_id", "granuscore_percentile")
    original_gran = {
        row["case_id"]: gran_scores[row["score_sent_id"]]
        for row in gran_manifest if row["version"] == "original"
    }
    score_rows: list[dict[str, Any]] = []
    case_by_id = {case.case_id: case for case in cases}
    for score_id, item in manifest_by_id.items():
        case = case_by_id[item["case_id"]]
        values: dict[str, tuple[float, float, bool]] = {
            "speciteller_frozen_round1": (
                original_spec[case.corpus_id][case.source_sent_id], spec_new[score_id], False
            ),
            "granuscore_native": (original_gran[case.case_id], gran_new[score_id], gran_no_unit[score_id]),
        }
        for run_id in ("run01", "run02", "run03"):
            values[f"ko_{run_id}"] = (
                original_ko[(case.corpus_id, run_id)][case.source_sent_id],
                ko_new[(case.corpus_id, run_id)][score_id], False,
            )
        ko_original_mean = float(np.mean([values[f"ko_{run}"][0] for run in ("run01", "run02", "run03")]))
        ko_edited_mean = float(np.mean([values[f"ko_{run}"][1] for run in ("run01", "run02", "run03")]))
        values[SECONDARY_MODEL] = (ko_original_mean, ko_edited_mean, False)
        for model, (original, edited, no_unit) in values.items():
            if not (math.isfinite(original) and math.isfinite(edited)):
                raise ValueError("nonfinite QE-A score")
            upper = 100.0 if model == "granuscore_native" else 1.0
            if not (0.0 <= original <= upper and 0.0 <= edited <= upper):
                raise ValueError("out-of-range QE-A score")
            score_rows.append(
                {
                    "case_id": case.case_id,
                    "corpus_id": case.corpus_id,
                    "source_sent_id": case.source_sent_id,
                    "edit_type": case.edit_type,
                    "edit_source": item["edit_source"],
                    "model_instance_id": model,
                    "score_original": original,
                    "score_edited": edited,
                    "delta_edited_minus_original": edited - original,
                    "score_direction": "higher_is_coarser_more_abstract" if model == "granuscore_native" else "higher_is_more_specific",
                    "no_referential_unit_edited": str(no_unit).lower(),
                }
            )
    expected = len(manifest) * (len(PRIMARY_MODELS) + 1)
    if len(score_rows) != expected:
        raise ValueError("QE-A long-score coverage mismatch")
    return score_rows, {"required_join_passed": True, "manifest_rows": len(manifest), "long_rows": len(score_rows)}


def _scoring_provenance(record: dict[str, Any]) -> dict[str, Any]:
    raw_dir = Path(record["outputs"]["raw_directory"])
    scoring = raw_dir / "scoring"
    preparation = scoring / "preparation_metadata.json"
    gs_metadata_path = scoring / "granuscore" / "scores.metadata.json"
    gs_metadata = json.loads(gs_metadata_path.read_text(encoding="utf-8"))
    ko_runs: dict[str, Any] = {}
    for corpus_id in record["inputs"]["corpus_order"]:
        for run_id in ("run01", "run02", "run03"):
            path = scoring / "ko" / corpus_id / run_id / "scores.metadata.json"
            metadata = json.loads(path.read_text(encoding="utf-8"))
            ko_runs[f"{corpus_id}:{run_id}"] = {
                "rows": metadata["row_count"],
                "score_sha256": metadata["score_sha256"],
                "prediction_sha256": metadata["prediction_sha256"],
                "checkpoint_sha256": metadata["checkpoint_sha256"],
            }
    return {
        "preparation_metadata_sha256": sha256_file(preparation),
        "speciteller": {
            "rows": sum(1 for _ in (scoring / "speciteller" / "scores.tsv").open(encoding="utf-8")),
            "input_sha256": sha256_file(scoring / "speciteller" / "input.tsv"),
            "score_sha256": sha256_file(scoring / "speciteller" / "scores.tsv"),
            "image_id": record["scoring"]["speciteller"]["image_id"],
        },
        "granuscore": {
            "rows": gs_metadata["row_count"],
            "input_sha256": gs_metadata["input_sha256"],
            "score_sha256": gs_metadata["output_sha256"],
            "runner_sha256": gs_metadata["runner_sha256"],
            "image_id": record["scoring"]["granuscore"]["image_id"],
            "native_direction": gs_metadata["score_direction"],
        },
        "ko_runs": ko_runs,
    }


def run_analysis(config_path: Path) -> dict[str, Any]:
    record, protocol_hash, freeze = load_protocol(config_path)
    cases = load_cases(record)
    raw_dir = Path(record["outputs"]["raw_directory"])
    compact_dir = Path(record["outputs"]["compact_directory"])
    compact_dir.mkdir(parents=True, exist_ok=True)
    generation = json.loads((raw_dir / "generation_summary.json").read_text(encoding="utf-8"))
    score_rows, join = _load_score_rows(record, cases)
    _write_csv(compact_dir / "scores_long.csv", score_rows)
    case_map = {case.case_id: case for case in cases}
    accepted = {row["case_id"] for row in score_rows if row["edit_source"] == "qwen"}
    coverage_rows: list[dict[str, Any]] = []
    for cell in generation["cells"]:
        coverage_rows.append({**cell, "failed": cell["planned"] - cell["accepted"], "gate_passed": cell["acceptance_rate"] >= record["automated_gates"]["coverage_gate"]["minimum_cell_acceptance_rate_for_primary_comparison"]})
    _write_csv(compact_dir / "coverage.csv", coverage_rows)
    attempts = _read_jsonl(raw_dir / record["outputs"]["raw_attempts"])
    failure_counts: Counter[tuple[str, str, str]] = Counter()
    for row in attempts:
        if row["status"] == "rejected":
            for reason in row["reason_codes"]:
                failure_counts[(row["corpus_id"], row["edit_type"], reason)] += 1
    failure_rows = [
        {"corpus_id": corpus, "edit_type": edit_type, "reason_code": reason, "failed_attempt_count": count}
        for (corpus, edit_type, reason), count in sorted(failure_counts.items())
    ] or [{"corpus_id": "all", "edit_type": "all", "reason_code": "none", "failed_attempt_count": 0}]
    _write_csv(compact_dir / "failure_reasons.csv", failure_rows)
    if not generation["coverage_gate_passed"]:
        gate_rows = [{
            "status": "primary_source_comparison_stopped",
            "planned_cases": generation["planned"],
            "accepted_cases": generation["accepted"],
            "acceptance_rate": generation["acceptance_rate"],
            "required_total_acceptance_rate": record["automated_gates"]["coverage_gate"]["minimum_total_acceptance_rate_for_primary_comparison"],
            "required_cell_acceptance_rate": record["automated_gates"]["coverage_gate"]["minimum_cell_acceptance_rate_for_primary_comparison"],
            "reason": "frozen generation coverage gate failed; no paper-facing paired source estimands computed",
        }]
        _write_csv(compact_dir / "gate_failure.csv", gate_rows)
        outputs = {}
        for name in ("scores_long.csv", "coverage.csv", "failure_reasons.csv", "gate_failure.csv"):
            with (compact_dir / name).open("r", encoding="utf-8", newline="") as handle:
                count = sum(1 for _ in csv.DictReader(handle))
            outputs[name] = {"rows": count, "sha256": sha256_file(compact_dir / name)}
        metadata = {
            "schema_version": RESULTS_VERSION,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "outcome_blind_freeze_commit": freeze["method_freeze_commit"],
            "protocol": {"path": _portable(config_path), "sha256": protocol_hash},
            "generation": {
                "planned_cases": generation["planned"], "accepted_cases": generation["accepted"],
                "failed_cases": generation["failed"], "attempt_rows": generation["attempt_rows"],
                "coverage_gate_passed": False,
                "raw_attempts_sha256": sha256_file(raw_dir / record["outputs"]["raw_attempts"]),
                "accepted_edits_sha256": sha256_file(raw_dir / record["outputs"]["accepted_edits"]),
            },
            "scoring": {
                **join,
                "models": [*PRIMARY_MODELS, SECONDARY_MODEL],
                **_scoring_provenance(record),
            },
            "analysis": {
                "bootstrap_replicates": int(record["analysis"]["bootstrap"]["replicates"]),
                "bootstrap_master_seed": int(record["analysis"]["bootstrap"]["master_seed"]),
                "analysis_set_cases": len(accepted),
                "paper_facing_source_comparison_performed": False,
                "stop_reason": "frozen generation coverage gate failed",
            },
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "requests": requests.__version__},
            "outputs": outputs,
            "claim_boundary": record["analysis"]["claim_boundary"],
            "privacy_boundary": record["outputs"]["release_boundary"],
        }
        readme = [
            "# Round 2 Qwen edit-source comparison", "",
            "## Gate decision", "",
            f"Generation accepted {generation['accepted']}/{generation['planned']} frozen cases ({100.0 * generation['acceptance_rate']:.1f}%). The predeclared 80% total and 75% per corpus-by-direction coverage gate failed.", "",
            "| Corpus | Direction | Accepted / planned |", "| --- | --- | ---: |",
        ]
        for cell in generation["cells"]:
            readme.append(f"| {cell['corpus_id']} | {cell['edit_type']} | {cell['accepted']}/{cell['planned']} |")
        readme.extend([
            "", "The corrected generation pass contained 137 attempts: 24 accepted and 113 rejected. Leading failed-attempt reason counts were content-anchor recall below minimum (45), token ratio above maximum (43), insufficient de-specification proxy (26), and modality change (23).", "",
            "All accepted outputs were nevertheless scored with frozen SpeciTeller, Ko run01/run02/run03, and native higher-is-coarser GranuScore. Exact identity, range, coverage, and join gates passed. Per the frozen stop rule, no paper-facing paired author-versus-Qwen source estimand or bootstrap interval was computed.", "",
            "Scoring coverage was 84/84 for SpeciTeller and GranuScore, 41/41 per retained Ko run for Ansible, and 43/43 per retained Ko run for GitHub. The compact long table contains 504 rows including the explicitly secondary Ko three-run mean.", "",
            "Prompts, attempts, gates, and inclusion rules were not tuned after generation.", "",
            "The complete pre-fix 180-call HTTP-grammar failure pass remains hash-addressed in the freeze record; it produced no generated text. A later GranuScore wrapper correction wrote no score before the fix and changed only duplicated host entrypoint arguments.", "",
            "## Reproduction", "",
            "From the SpecTech repository root, run `python scripts/qwen_edit_source.py preflight`, then `generate`, `prepare-scoring`, each frozen scorer command, and finally `analyze`. Text-bearing raw artifacts and nonredistributable Ko checkpoints remain ignored; compact evidence is checksum-addressed in `run_metadata.json`.", "",
            "## Interpretation boundary", "", record["analysis"]["claim_boundary"], "",
        ])
        (compact_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
        metadata["outputs"]["README.md"] = {"sha256": sha256_file(compact_dir / "README.md")}
        (compact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in score_rows:
        if row["case_id"] in accepted:
            grouped[(row["model_instance_id"], row["corpus_id"], row["edit_type"], row["edit_source"])].append(row)
    arm_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    reps = int(record["analysis"]["bootstrap"]["replicates"])
    master = int(record["analysis"]["bootstrap"]["master_seed"])
    for corpus_id in record["inputs"]["corpus_order"]:
        for edit_type in EDIT_TYPES:
            case_ids = sorted(case.case_id for case in cases if case.corpus_id == corpus_id and case.edit_type == edit_type and case.case_id in accepted)
            if not case_ids:
                continue
            rng = np.random.default_rng(_stream_seed(master, corpus_id, edit_type))
            indices = rng.integers(0, len(case_ids), size=(reps, len(case_ids)))
            for model in (*PRIMARY_MODELS, SECONDARY_MODEL):
                arrays: dict[str, np.ndarray] = {}
                for source in ("author", "qwen"):
                    lookup = {row["case_id"]: float(row["delta_edited_minus_original"]) for row in grouped[(model, corpus_id, edit_type, source)]}
                    if set(lookup) != set(case_ids):
                        raise ValueError("paired source analysis join mismatch")
                    values = np.asarray([lookup[case] for case in case_ids], dtype=float)
                    arrays[source] = values
                    boot = values[indices]
                    mean_boot = np.mean(boot, axis=1)
                    abs_boot = np.mean(np.abs(boot), axis=1)
                    low, high = _percentile_interval(mean_boot)
                    abs_low, abs_high = _percentile_interval(abs_boot)
                    direction = _direction_values(values, edit_type, model)
                    if edit_type == "irrelevant_rewrite":
                        direction_rate = ci_dir_low = ci_dir_high = ""
                    else:
                        direction_rate = float(np.mean(direction))
                        ci_dir_low, ci_dir_high = _percentile_interval(np.mean(direction[indices], axis=1))
                    arm_rows.append(
                        {
                            "model_instance_id": model,
                            "model_role": "secondary" if model == SECONDARY_MODEL else "primary",
                            "corpus_id": corpus_id,
                            "edit_type": edit_type,
                            "edit_source": source,
                            "n": len(values),
                            "mean_delta": float(np.mean(values)),
                            "mean_delta_ci_low": low,
                            "mean_delta_ci_high": high,
                            "median_delta": float(np.median(values)),
                            "mean_absolute_delta": float(np.mean(np.abs(values))),
                            "mean_absolute_delta_ci_low": abs_low,
                            "mean_absolute_delta_ci_high": abs_high,
                            "expected_direction_rate": direction_rate,
                            "expected_direction_rate_ci_low": ci_dir_low,
                            "expected_direction_rate_ci_high": ci_dir_high,
                        }
                    )
                author = arrays["author"]
                qwen = arrays["qwen"]
                signed_diff = qwen - author
                magnitude_diff = np.abs(qwen) - np.abs(author)
                signed_low, signed_high = _percentile_interval(np.mean(signed_diff[indices], axis=1))
                mag_low, mag_high = _percentile_interval(np.mean(magnitude_diff[indices], axis=1))
                if edit_type == "irrelevant_rewrite":
                    rate_diff = rate_low = rate_high = concordance = ""
                else:
                    author_direction = _direction_values(author, edit_type, model)
                    qwen_direction = _direction_values(qwen, edit_type, model)
                    rate_values = qwen_direction - author_direction
                    rate_diff = float(np.mean(rate_values))
                    rate_low, rate_high = _percentile_interval(np.mean(rate_values[indices], axis=1))
                    concordance = float(np.mean(author_direction == qwen_direction))
                contrast_rows.append(
                    {
                        "model_instance_id": model,
                        "model_role": "secondary" if model == SECONDARY_MODEL else "primary",
                        "corpus_id": corpus_id,
                        "edit_type": edit_type,
                        "n": len(case_ids),
                        "mean_qwen_minus_author_delta": float(np.mean(signed_diff)),
                        "mean_qwen_minus_author_delta_ci_low": signed_low,
                        "mean_qwen_minus_author_delta_ci_high": signed_high,
                        "mean_absolute_qwen_minus_author_delta": float(np.mean(magnitude_diff)),
                        "mean_absolute_qwen_minus_author_delta_ci_low": mag_low,
                        "mean_absolute_qwen_minus_author_delta_ci_high": mag_high,
                        "expected_direction_rate_qwen_minus_author": rate_diff,
                        "expected_direction_rate_difference_ci_low": rate_low,
                        "expected_direction_rate_difference_ci_high": rate_high,
                        "direction_response_concordance_rate": concordance,
                    }
                )
    _write_csv(compact_dir / "arm_summaries.csv", arm_rows)
    _write_csv(compact_dir / "paired_source_contrasts.csv", contrast_rows)
    paper_rows = [
        {
            "model_instance_id": row["model_instance_id"],
            "corpus_id": row["corpus_id"],
            "edit_type": row["edit_type"],
            "n": row["n"],
            "qwen_minus_author_delta": row["mean_qwen_minus_author_delta"],
            "ci_low": row["mean_qwen_minus_author_delta_ci_low"],
            "ci_high": row["mean_qwen_minus_author_delta_ci_high"],
        }
        for row in contrast_rows if row["model_role"] == "primary"
    ]
    _write_csv(compact_dir / "paper_table.csv", paper_rows)
    outputs = {}
    for name in ("scores_long.csv", "coverage.csv", "failure_reasons.csv", "arm_summaries.csv", "paired_source_contrasts.csv", "paper_table.csv"):
        with (compact_dir / name).open("r", encoding="utf-8", newline="") as handle:
            count = sum(1 for _ in csv.DictReader(handle))
        outputs[name] = {"rows": count, "sha256": sha256_file(compact_dir / name)}
    metadata = {
        "schema_version": RESULTS_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_blind_freeze_commit": freeze["method_freeze_commit"],
        "protocol": {"path": _portable(config_path), "sha256": protocol_hash},
        "generation": {
            "planned_cases": generation["planned"], "accepted_cases": generation["accepted"],
            "failed_cases": generation["failed"], "attempt_rows": generation["attempt_rows"],
            "coverage_gate_passed": generation["coverage_gate_passed"],
            "raw_attempts_sha256": sha256_file(raw_dir / record["outputs"]["raw_attempts"]),
            "accepted_edits_sha256": sha256_file(raw_dir / record["outputs"]["accepted_edits"]),
        },
        "scoring": {**join, "models": [*PRIMARY_MODELS, SECONDARY_MODEL]},
        "analysis": {"bootstrap_replicates": reps, "bootstrap_master_seed": master, "analysis_set_cases": len(accepted)},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "requests": requests.__version__},
        "outputs": outputs,
        "claim_boundary": record["analysis"]["claim_boundary"],
        "privacy_boundary": record["outputs"]["release_boundary"],
    }
    (compact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Round 2 Qwen edit-source comparison", "",
        f"Generation accepted {generation['accepted']}/{generation['planned']} frozen cases; coverage gate passed: {generation['coverage_gate_passed']}.", "",
        "All comparative tables were produced only after exact joins passed for SpeciTeller, Ko run01/run02/run03, and native higher-is-coarser GranuScore.", "",
        "## Interpretation boundary", "",
        record["analysis"]["claim_boundary"], "",
        "Ko run instances are primary and their arithmetic mean is secondary. Raw scales are not compared across models. No Qwen self-judgment or human validation was performed.", "",
    ]
    (compact_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    metadata["outputs"]["README.md"] = {"sha256": sha256_file(compact_dir / "README.md")}
    (compact_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


__all__ = [
    "EditCase", "build_generation_request", "case_id", "derive_attempt_seed",
    "gate_edit", "load_cases", "load_protocol", "parse_generation_response",
    "prepare_scoring", "run_analysis", "run_generation", "score_granuscore",
    "score_ko_checkpoint", "score_speciteller", "verify_preflight",
]
