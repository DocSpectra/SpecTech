"""Blinded split and bounded remediation helpers for the QE-R study."""
from __future__ import annotations

import csv
import copy
import hashlib
import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import requests

from src.analysis.pilot_model_human import sha256_file
from src.analysis.qwen_edit_source import case_id as qe_a_case_id
from src.analysis.qwen_edit_source import derive_attempt_seed, gate_edit, parse_generation_response, verify_ollama_identity


SCHEMA_VERSION = "round2_qwen_edit_remediation_split_v1"
QE_A_SCHEMA_VERSION = "round2_qwen_edit_source_v1"
CANDIDATE_SCHEMA_VERSION = "round2_qwen_edit_remediation_candidates_v1"
CANDIDATE_FREEZE_RECORD = Path("configs/round2_qwen_edit_remediation_candidates_freeze_record.json")
V2_SCHEMA_VERSION = "round2_qwen_edit_remediation_v2"
V2_FREEZE_RECORD = Path("configs/round2_qwen_edit_remediation_v2_freeze_record.json")
WORD_RE = re.compile(r"(?u)\b\w+(?:[-./:]\w+)*\b")
STOPWORDS = frozenset(
    "a an and are as at be been being but by for from had has have he her hers him his i if in into is it its me my no nor not of on or our ours she so than that the their theirs them they this those to too us was we were what when where which who why will with you your yours".split()
)


def _case_id(corpus_id: str, sent_id: str, edit_type: str, position: int) -> str:
    value = f"{corpus_id}\0{sent_id}\0{edit_type}\0{position}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _ranking_key(seed: int, corpus_id: str, edit_type: str, case_id: str, position: int) -> str:
    value = f"{seed}:{corpus_id}:{edit_type}:{case_id}:{position}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_split_rows(config_path: Path) -> list[dict[str, Any]]:
    """Build assignments without reading text, QE-A attempts, gates, or outcomes."""
    config = _load_json(config_path)
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected QE-R split schema")
    qe_a_path = Path(config["inputs"]["qe_a_config_path"])
    if sha256_file(qe_a_path) != config["inputs"]["qe_a_config_sha256"]:
        raise ValueError("QE-A config hash mismatch")
    qe_a = _load_json(qe_a_path)
    if qe_a.get("schema_version") != QE_A_SCHEMA_VERSION:
        raise ValueError("unexpected QE-A config schema")
    if qe_a["inputs"]["ordered_case_sha256"] != config["inputs"]["qe_a_ordered_case_sha256"]:
        raise ValueError("QE-A ordered-case digest mismatch")

    seed = int(config["split"]["seed"])
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    total = 0
    for corpus_id in qe_a["inputs"]["corpus_order"]:
        source_entry = qe_a["inputs"]["source_files"][corpus_id]
        source_path = Path(source_entry["path"])
        if sha256_file(source_path) != source_entry["sha256"]:
            raise ValueError(f"source hash mismatch: {corpus_id}")
        # Deliberately project only non-text split fields from the source rows.
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for position, row in enumerate(reader, start=1):
                edit_type = row["edit_type"]
                identifier = _case_id(corpus_id, row["sent_id"], edit_type, position)
                item = {
                    "case_id": identifier,
                    "corpus_id": corpus_id,
                    "edit_type": edit_type,
                    "source_position": position,
                    "ranking_key": _ranking_key(seed, corpus_id, edit_type, identifier, position),
                }
                cells.setdefault((corpus_id, edit_type), []).append(item)
                total += 1
    if total != int(config["inputs"]["required_total_cases"]):
        raise ValueError("QE-R split input count mismatch")

    assigned: list[dict[str, Any]] = []
    for key in sorted(cells):
        items = sorted(cells[key], key=lambda item: (item["ranking_key"], item["case_id"]))
        if len(items) % 2:
            raise ValueError(f"QE-R requires an even stratum size: {key}")
        midpoint = len(items) // 2
        for index, item in enumerate(items):
            assigned.append({**item, "split": "development" if index < midpoint else "held_out"})
    assigned.sort(key=lambda item: (item["corpus_id"], item["source_position"]))
    return assigned


def ordered_assignment_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        value = "\0".join(
            [
                row["corpus_id"],
                str(row["source_position"]),
                row["case_id"],
                row["edit_type"],
                row["split"],
                row["ranking_key"],
            ]
        )
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_split(config_path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    config = _load_json(config_path)
    counts = Counter(f"{row['corpus_id']}:{row['edit_type']}" for row in rows if row["split"] == "development")
    held = Counter(f"{row['corpus_id']}:{row['edit_type']}" for row in rows if row["split"] == "held_out")
    if dict(counts) != config["split"]["expected_counts"]["development"]:
        raise ValueError("development split counts mismatch")
    if dict(held) != config["split"]["expected_counts"]["held_out"]:
        raise ValueError("held-out split counts mismatch")
    if len(rows) != 60 or len({row["case_id"] for row in rows}) != 60:
        raise ValueError("QE-R split identity mismatch")
    digest = ordered_assignment_sha256(rows)
    expected = config["split"]["ordered_assignment_sha256"]
    if expected != "TO_BE_FROZEN" and digest != expected:
        raise ValueError("QE-R assignment digest mismatch")
    return {"rows": len(rows), "development": sum(counts.values()), "held_out": sum(held.values()), "ordered_assignment_sha256": digest}


def write_split(config_path: Path) -> dict[str, Any]:
    config = _load_json(config_path)
    rows = build_split_rows(config_path)
    summary = validate_split(config_path, rows)
    output = Path(config["split"]["manifest_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary["manifest_sha256"] = sha256_file(output)
    return summary


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * probability
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _metric_summary(source: str, corpus: str, direction: str, metric: str, values: list[float]) -> dict[str, Any]:
    return {
        "source": source,
        "corpus_id": corpus,
        "edit_type": direction,
        "metric": metric,
        "n": len(values),
        "minimum": f"{min(values):.6f}",
        "q10": f"{_quantile(values, 0.10):.6f}",
        "median": f"{statistics.median(values):.6f}",
        "q90": f"{_quantile(values, 0.90):.6f}",
        "maximum": f"{max(values):.6f}",
    }


def diagnose_v1(split_config_path: Path) -> dict[str, Any]:
    """Diagnose QE-A v1 using development text only and score-blind controls."""
    split_config = _load_json(split_config_path)
    split_path = Path(split_config["split"]["manifest_path"])
    if sha256_file(split_path) != split_config["split"]["manifest_sha256"]:
        raise ValueError("QE-R split manifest hash mismatch")
    split_rows = _read_csv(split_path)
    development_ids = {row["case_id"] for row in split_rows if row["split"] == "development"}
    held_ids = {row["case_id"] for row in split_rows if row["split"] == "held_out"}
    if len(development_ids) != 30 or len(held_ids) != 30 or development_ids & held_ids:
        raise ValueError("QE-R split separation failed")

    qe_a_path = Path(split_config["inputs"]["qe_a_config_path"])
    qe_a = _load_json(qe_a_path)
    raw_dir = Path(qe_a["outputs"]["raw_directory"])
    source_by_id: dict[str, dict[str, str]] = {}
    for corpus_id in qe_a["inputs"]["corpus_order"]:
        source_path = Path(qe_a["inputs"]["source_files"][corpus_id]["path"])
        for position, row in enumerate(_read_csv(source_path), start=1):
            identifier = qe_a_case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            if identifier in development_ids:
                source_by_id[identifier] = {
                    **row,
                    "corpus_id": corpus_id,
                    "source_position": str(position),
                }
    if set(source_by_id) != development_ids:
        raise ValueError("development source projection mismatch")

    attempt_rows: list[dict[str, Any]] = []
    with (raw_dir / qe_a["outputs"]["raw_attempts"]).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["case_id"] in held_ids:
                continue
            if row["case_id"] not in development_ids:
                raise ValueError("unknown QE-A attempt case")
            attempt_rows.append(row)

    coverage: Counter[tuple[str, str, str]] = Counter()
    attempt_by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failure_counts: Counter[tuple[str, str, str, str]] = Counter()
    metric_values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in attempt_rows:
        case = source_by_id[row["case_id"]]
        cell = (case["corpus_id"], case["edit_type"])
        attempt_by_case[row["case_id"]].append(row)
        for reason in row["reason_codes"]:
            failure_counts[("qe_a_attempt", *cell, reason)] += 1
        if row["edited_sentence"]:
            reasons, metrics = gate_edit(qe_a, case["sentence_original"], row["edited_sentence"], case["edit_type"])
            if reasons != row["reason_codes"]:
                raise ValueError("development QE-A trace does not reproduce")
            for metric, value in metrics.items():
                metric_values[("qe_a_attempt", *cell, metric)].append(float(value))

    for identifier, case in source_by_id.items():
        cell = (case["corpus_id"], case["edit_type"])
        coverage[("planned", *cell)] += 1
        if any(row["status"] == "accepted" for row in attempt_by_case[identifier]):
            coverage[("accepted", *cell)] += 1
        reasons, metrics = gate_edit(qe_a, case["sentence_original"], case["sentence_edited"], case["edit_type"])
        for reason in reasons:
            failure_counts[("author_positive_control", *cell, reason)] += 1
        if not reasons:
            coverage[("positive_control_pass", *cell)] += 1
        for metric, value in metrics.items():
            metric_values[("author_positive_control", *cell, metric)].append(float(value))

    cells = sorted({(row["corpus_id"], row["edit_type"]) for row in source_by_id.values()})
    coverage_rows = []
    for corpus, direction in cells:
        planned = coverage[("planned", corpus, direction)]
        accepted = coverage[("accepted", corpus, direction)]
        control_pass = coverage[("positive_control_pass", corpus, direction)]
        coverage_rows.append(
            {
                "corpus_id": corpus,
                "edit_type": direction,
                "development_cases": planned,
                "qe_a_accepted_cases": accepted,
                "qe_a_acceptance_rate": f"{accepted / planned:.6f}",
                "author_positive_control_passes_v1": control_pass,
                "author_positive_control_pass_rate_v1": f"{control_pass / planned:.6f}",
            }
        )
    failure_rows = [
        {
            "source": source,
            "corpus_id": corpus,
            "edit_type": direction,
            "reason_code": reason,
            "count": count,
        }
        for (source, corpus, direction, reason), count in sorted(failure_counts.items())
    ]
    metric_rows = [
        _metric_summary(source, corpus, direction, metric, values)
        for (source, corpus, direction, metric), values in sorted(metric_values.items())
    ]

    compact = Path("analysis/round2_qwen_edit_remediation")
    _write_csv(compact / "development_v1_coverage.csv", coverage_rows)
    _write_csv(compact / "development_v1_failures.csv", failure_rows)
    _write_csv(compact / "development_v1_metrics.csv", metric_rows)

    # Local-only report is deliberately development-only and excludes author wording.
    report_lines = ["# QE-R development-only QE-A text review", ""]
    for identifier in sorted(development_ids):
        case = source_by_id[identifier]
        report_lines.extend(
            [
                f"## {identifier} | {case['corpus_id']} | {case['edit_type']}",
                "",
                f"Original: {case['sentence_original']}",
                "",
            ]
        )
        for attempt in attempt_by_case[identifier]:
            report_lines.extend(
                [
                    f"Attempt {attempt['attempt_index']} ({attempt['status']}): {attempt['edited_sentence']}",
                    f"Reasons: {', '.join(attempt['reason_codes']) if attempt['reason_codes'] else 'none'}",
                    f"Metrics: token_ratio={attempt.get('token_ratio', '')}; anchor={attempt.get('content_anchor_recall', '')}; markers={attempt.get('original_concrete_marker_count', '')}->{attempt.get('edited_concrete_marker_count', '')}",
                    "",
                ]
            )
    local_report = Path("outputs/round2/qwen_edit_remediation/development/v1_text_review.md")
    local_report.parent.mkdir(parents=True, exist_ok=True)
    local_report.write_text("\n".join(report_lines), encoding="utf-8")
    metadata = {
        "schema_version": "round2_qwen_edit_remediation_development_v1_diagnosis_v1",
        "split_manifest_sha256": sha256_file(split_path),
        "qe_a_attempts_sha256": sha256_file(raw_dir / qe_a["outputs"]["raw_attempts"]),
        "development_cases": 30,
        "held_out_cases_not_exported_or_summarized": 30,
        "development_attempt_rows": len(attempt_rows),
        "development_text_report_sha256": sha256_file(local_report),
        "positive_controls": "development-half author edits only; gate metrics and reasons aggregated; author wording never shown to generation",
        "prohibited_inputs_used": False,
        "outputs": {
            name: sha256_file(compact / name)
            for name in ("development_v1_coverage.csv", "development_v1_failures.csv", "development_v1_metrics.csv")
        },
    }
    (compact / "development_v1_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(unicodedata.normalize("NFKC", value))]


def _content_token_set(value: str) -> set[str]:
    return {token for token in _tokens(value) if len(token) >= 3 and token not in STOPWORDS}


def _load_candidate_context(candidate_config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    candidate = _load_json(candidate_config_path)
    if candidate.get("schema_version") != CANDIDATE_SCHEMA_VERSION:
        raise ValueError("unexpected QE-R candidate schema")
    if len(candidate["candidate_order"]) > 3 or int(candidate["round"]) > 2:
        raise ValueError("QE-R candidate budget exceeded")
    freeze = _load_json(CANDIDATE_FREEZE_RECORD)
    portable_candidate_path = candidate_config_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    if freeze["candidate_config_path"] != portable_candidate_path or freeze["candidate_config_sha256"] != sha256_file(candidate_config_path):
        raise ValueError("QE-R candidate freeze record mismatch")
    if freeze["frozen_before_fresh_development_generation"] is not True:
        raise ValueError("QE-R candidate pre-generation freeze missing")
    if sha256_file(Path(candidate["split_config_path"])) != candidate["split_config_sha256"]:
        raise ValueError("QE-R candidate split config mismatch")
    if sha256_file(Path(candidate["qe_a_config_path"])) != candidate["qe_a_config_sha256"]:
        raise ValueError("QE-R candidate QE-A config mismatch")
    base = _load_json(Path(candidate["qe_a_config_path"]))
    split = _load_json(Path(candidate["split_config_path"]))
    split_rows = _read_csv(Path(split["split"]["manifest_path"]))
    development_ids = {row["case_id"] for row in split_rows if row["split"] == "development"}
    cases: dict[str, dict[str, str]] = {}
    for corpus_id in base["inputs"]["corpus_order"]:
        for position, row in enumerate(_read_csv(Path(base["inputs"]["source_files"][corpus_id]["path"])), start=1):
            identifier = qe_a_case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            if identifier in development_ids:
                cases[identifier] = {
                    "case_id": identifier,
                    "corpus_id": corpus_id,
                    "edit_type": row["edit_type"],
                    "source_position": str(position),
                    "source_sent_id": row["sent_id"],
                    "sentence_original": row["sentence_original"],
                }
    if len(cases) != int(candidate["development_cases"]) or set(cases) != development_ids:
        raise ValueError("QE-R candidate development projection mismatch")
    return candidate, base, cases


def _candidate_gate(candidate: dict[str, Any], base: dict[str, Any], candidate_id: str, original: str, edited: str, edit_type: str) -> tuple[list[str], dict[str, Any]]:
    profile = candidate["candidate_profiles"][candidate_id]
    gate_variant = candidate["gate_variants"][profile["gate_variant"]]
    effective = copy.deepcopy(base)
    effective["automated_gates"]["direction_rules"]["irrelevant_rewrite"]["original_content_anchor_recall_min"] = gate_variant["neutral_content_anchor_recall_min"]
    effective["automated_gates"]["direction_rules"]["de_specify"]["edited_to_original_token_ratio_max"] = gate_variant["de_token_ratio_max"]
    reasons, metrics = gate_edit(effective, original, edited, edit_type)
    if edit_type == "de_specify" and profile["gate_variant"] == "proxy_aligned" and "insufficient_despecification_proxy" in reasons:
        original_content = _content_token_set(original)
        edited_content = _content_token_set(edited)
        proxy_pass = (
            metrics["edited_token_count"] <= metrics["original_token_count"] - 1
            or metrics["edited_concrete_marker_count"] < metrics["original_concrete_marker_count"]
            or (
                metrics["content_anchor_recall"] <= 0.9
                and len(edited_content) <= len(original_content)
            )
        )
        if proxy_pass:
            reasons.remove("insufficient_despecification_proxy")
    original_has_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in original)
    edited_has_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in edited)
    if not original_has_non_latin and edited_has_non_latin:
        reasons.append("new_non_latin_script")
    boundaries = re.findall(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)", edited.strip())
    if len(boundaries) > 1:
        reasons.append("multiple_sentences")
    return sorted(set(reasons)), {**metrics, "sentence_boundary_count": len(boundaries)}


def _candidate_prompt(candidate: dict[str, Any], base: dict[str, Any], candidate_id: str, case: dict[str, str]) -> str:
    profile = candidate["candidate_profiles"][candidate_id]
    template = candidate["prompt_variants"][profile["prompt_variant"]][case["edit_type"]]
    count = len(_tokens(case["sentence_original"]))
    gate = candidate["gate_variants"][profile["gate_variant"]]
    values = {
        "sentence_original": case["sentence_original"],
        "original_word_count": count,
        "add_word_limit": max(count + 4, math.floor(1.75 * count)),
        "micro_add_word_limit": max(count + 3, math.floor(1.60 * count)),
        "de_word_limit": max(1, math.floor(float(gate["de_token_ratio_max"]) * count)),
        "neutral_word_min": max(1, math.ceil(0.75 * count)),
        "neutral_word_max": max(1, math.floor(1.35 * count)),
    }
    return template.format(**values)


def _candidate_request(candidate: dict[str, Any], base: dict[str, Any], candidate_id: str, case: dict[str, str], attempt: int) -> dict[str, Any]:
    runtime = candidate["runtime"]
    options = dict(runtime["options"])
    options["seed"] = derive_attempt_seed(int(candidate["round_master_seed"]), f"{candidate_id}:{case['case_id']}", attempt)
    return {
        "model": runtime["model"],
        "messages": [
            {"role": "system", "content": candidate["system_prompt"]},
            {"role": "user", "content": _candidate_prompt(candidate, base, candidate_id, case)},
        ],
        "stream": runtime["stream"],
        "think": runtime["thinking"],
        "format": runtime["response_schema"],
        "options": options,
        "keep_alive": runtime["keep_alive"],
    }


def run_development_candidates(candidate_config_path: Path, *, resume: bool = False, timeout: float = 300.0) -> dict[str, Any]:
    candidate, base, cases = _load_candidate_context(candidate_config_path)
    identity_record = copy.deepcopy(base)
    identity_record["qwen"].update(candidate["runtime"])
    verify_ollama_identity(identity_record)
    config_hash = sha256_file(candidate_config_path)
    output_root = Path(candidate["raw_output_directory"])
    summaries: dict[str, Any] = {}
    for candidate_id in candidate["candidate_order"]:
        directory = output_root / candidate_id
        directory.mkdir(parents=True, exist_ok=True)
        attempts_path = directory / "attempts.jsonl"
        rows = []
        if attempts_path.exists():
            rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line]
            if rows and not resume:
                raise ValueError(f"candidate output exists; use --resume: {candidate_id}")
        by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            if row["candidate_config_sha256"] != config_hash or row["case_id"] not in cases:
                raise ValueError("invalid candidate resume identity")
            by_case[row["case_id"]].append(row)
        with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
            for identifier in sorted(cases):
                case = cases[identifier]
                prior = by_case[identifier]
                if any(row["status"] == "accepted" for row in prior):
                    continue
                for attempt in range(len(prior) + 1, int(candidate["maximum_attempts_per_case"]) + 1):
                    request = _candidate_request(candidate, base, candidate_id, case, attempt)
                    try:
                        response = requests.post(candidate["runtime"]["api_url"], json=request, timeout=timeout)
                        response.raise_for_status()
                        edited, response_meta = parse_generation_response(response.json(), candidate["runtime"]["model"])
                        reasons, metrics = _candidate_gate(candidate, base, candidate_id, case["sentence_original"], edited, case["edit_type"])
                    except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                        edited = ""
                        reasons = [f"request_or_parse_error:{type(exc).__name__}"]
                        metrics = {}
                        response_meta = {"response_content_sha256": hashlib.sha256(b"").hexdigest()}
                    row = {
                        "candidate_id": candidate_id,
                        "case_id": identifier,
                        "corpus_id": case["corpus_id"],
                        "edit_type": case["edit_type"],
                        "attempt_index": attempt,
                        "attempt_seed": request["options"]["seed"],
                        "status": "accepted" if not reasons else "rejected",
                        "reason_codes": reasons,
                        "edited_sentence": edited.strip(),
                        "candidate_config_sha256": config_hash,
                        "model_blob_sha256": candidate["runtime"]["backing_blob_sha256"],
                        **metrics,
                        **response_meta,
                    }
                    handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                    handle.flush()
                    by_case[identifier].append(row)
                    if not reasons:
                        break
        final_rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line]
        accepted = {row["case_id"]: row for row in final_rows if row["status"] == "accepted"}
        accepted_path = directory / "accepted.csv"
        if accepted:
            _write_csv(
                accepted_path,
                [
                    {
                        "case_id": identifier,
                        "corpus_id": cases[identifier]["corpus_id"],
                        "edit_type": cases[identifier]["edit_type"],
                        "attempt_index": row["attempt_index"],
                        "edited_sentence": row["edited_sentence"],
                        "edited_sha256": hashlib.sha256(row["edited_sentence"].encode("utf-8")).hexdigest(),
                    }
                    for identifier, row in sorted(accepted.items())
                ],
            )
        cell_counts: Counter[tuple[str, str, str]] = Counter()
        for case in cases.values():
            cell_counts[("planned", case["corpus_id"], case["edit_type"])] += 1
            if case["case_id"] in accepted:
                cell_counts[("accepted", case["corpus_id"], case["edit_type"])] += 1
        cells = [
            {
                "corpus_id": corpus,
                "edit_type": direction,
                "planned": cell_counts[("planned", corpus, direction)],
                "accepted": cell_counts[("accepted", corpus, direction)],
                "acceptance_rate": cell_counts[("accepted", corpus, direction)] / cell_counts[("planned", corpus, direction)],
            }
            for corpus, direction in sorted({(case["corpus_id"], case["edit_type"]) for case in cases.values()})
        ]
        summary = {
            "candidate_id": candidate_id,
            "planned": len(cases),
            "accepted": len(accepted),
            "attempt_rows": len(final_rows),
            "cells": cells,
            "attempts_sha256": sha256_file(attempts_path),
            "accepted_sha256": sha256_file(accepted_path) if accepted else None,
        }
        (directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        summaries[candidate_id] = summary
    return summaries


def summarize_development_candidates(candidate_config_path: Path) -> dict[str, Any]:
    candidate, _, cases = _load_candidate_context(candidate_config_path)
    raw_root = Path(candidate["raw_output_directory"])
    compact = Path(candidate["compact_output_directory"])
    compact.mkdir(parents=True, exist_ok=True)
    coverage_rows: list[dict[str, Any]] = []
    failure_counts: Counter[tuple[str, str, str, str]] = Counter()
    selection_rows: list[dict[str, Any]] = []
    for candidate_id in candidate["candidate_order"]:
        summary = _load_json(raw_root / candidate_id / "summary.json")
        attempts = [json.loads(line) for line in (raw_root / candidate_id / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
        for cell in summary["cells"]:
            coverage_rows.append({"candidate_id": candidate_id, **cell})
        for row in attempts:
            for reason in row["reason_codes"]:
                failure_counts[(candidate_id, row["corpus_id"], row["edit_type"], reason)] += 1
        rates = [float(cell["acceptance_rate"]) for cell in summary["cells"]]
        accepted_attempts = [int(row["attempt_index"]) for row in attempts if row["status"] == "accepted"]
        distinct_reasons = len({reason for row in attempts for reason in row["reason_codes"]})
        selection_rows.append(
            {
                "candidate_id": candidate_id,
                "minimum_cell_coverage": f"{min(rates):.6f}",
                "overall_coverage": f"{summary['accepted'] / len(cases):.6f}",
                "accepted_cases": summary["accepted"],
                "mean_attempt_index_of_accepts": f"{statistics.mean(accepted_attempts):.6f}" if accepted_attempts else "",
                "distinct_failure_reason_codes": distinct_reasons,
                "method_change_count": candidate["candidate_profiles"][candidate_id]["method_change_count"],
            }
        )
    failure_rows = [
        {"candidate_id": cid, "corpus_id": corpus, "edit_type": direction, "reason_code": reason, "count": count}
        for (cid, corpus, direction, reason), count in sorted(failure_counts.items())
    ]
    _write_csv(compact / "coverage.csv", coverage_rows)
    _write_csv(compact / "failure_reasons.csv", failure_rows)
    _write_csv(compact / "selection_metrics.csv", selection_rows)
    metadata = {
        "schema_version": "round2_qwen_edit_remediation_development_candidates_results_v1",
        "candidate_config_sha256": sha256_file(candidate_config_path),
        "development_cases": len(cases),
        "execution_rounds_used": 1,
        "candidate_configurations_used": len(candidate["candidate_order"]),
        "scorer_rubric_human_or_held_out_outcomes_used": False,
        "outputs": {name: sha256_file(compact / name) for name in ("coverage.csv", "failure_reasons.csv", "selection_metrics.csv")},
    }
    (compact / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def _load_v2_context(v2_config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, str]]]:
    protocol = _load_json(v2_config_path)
    if protocol.get("schema_version") != V2_SCHEMA_VERSION:
        raise ValueError("unexpected QE-R v2 schema")
    freeze = _load_json(V2_FREEZE_RECORD)
    portable = v2_config_path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    if freeze["v2_config_path"] != portable or freeze["v2_config_sha256"] != sha256_file(v2_config_path):
        raise ValueError("QE-R v2 freeze record mismatch")
    if freeze["frozen_before_held_out_generation"] is not True:
        raise ValueError("QE-R v2 pre-held-out freeze missing")
    for key in ("split_config_path", "split_manifest_path", "qe_a_config_path"):
        hash_key = key.replace("_path", "_sha256")
        if sha256_file(Path(protocol["inputs"][key])) != protocol["inputs"][hash_key]:
            raise ValueError(f"QE-R v2 input mismatch: {key}")
    split_rows = _read_csv(Path(protocol["inputs"]["split_manifest_path"]))
    held_ids = {row["case_id"] for row in split_rows if row["split"] == "held_out"}
    base = _load_json(Path(protocol["inputs"]["qe_a_config_path"]))
    cases: dict[str, dict[str, str]] = {}
    for corpus_id in base["inputs"]["corpus_order"]:
        for position, row in enumerate(_read_csv(Path(base["inputs"]["source_files"][corpus_id]["path"])), start=1):
            identifier = qe_a_case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            if identifier in held_ids:
                cases[identifier] = {
                    "case_id": identifier,
                    "corpus_id": corpus_id,
                    "edit_type": row["edit_type"],
                    "source_position": str(position),
                    "source_sent_id": row["sent_id"],
                    "sentence_original": row["sentence_original"],
                }
    if len(cases) != int(protocol["inputs"]["required_cases"]) or set(cases) != held_ids:
        raise ValueError("QE-R v2 held-out projection mismatch")
    return protocol, base, cases


def _v2_gate(protocol: dict[str, Any], base: dict[str, Any], original: str, edited: str, edit_type: str) -> tuple[list[str], dict[str, Any]]:
    effective = copy.deepcopy(base)
    effective["automated_gates"]["universal"].update(copy.deepcopy(protocol["automated_gates"]["universal"]))
    effective["automated_gates"]["direction_rules"] = copy.deepcopy(protocol["automated_gates"]["direction_rules"])
    # QE-A's gate implementation consumes the same fully specified universal and
    # direction keys; new safeguards and the revised de proxy are applied below.
    reasons, metrics = gate_edit(effective, original, edited, edit_type)
    if edit_type == "de_specify" and "insufficient_despecification_proxy" in reasons:
        original_content = _content_token_set(original)
        edited_content = _content_token_set(edited)
        proxy_pass = (
            metrics["edited_token_count"] <= metrics["original_token_count"] - 1
            or metrics["edited_concrete_marker_count"] < metrics["original_concrete_marker_count"]
            or (metrics["content_anchor_recall"] <= 0.9 and len(edited_content) <= len(original_content))
        )
        if proxy_pass:
            reasons.remove("insufficient_despecification_proxy")
    original_has_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in original)
    edited_has_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in edited)
    if not original_has_non_latin and edited_has_non_latin:
        reasons.append("new_non_latin_script")
    boundaries = re.findall(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)", edited.strip())
    if len(boundaries) > 1:
        reasons.append("multiple_sentences")
    return sorted(set(reasons)), {**metrics, "sentence_boundary_count": len(boundaries)}


def _v2_request(protocol: dict[str, Any], case: dict[str, str], attempt: int) -> dict[str, Any]:
    count = len(_tokens(case["sentence_original"]))
    values = {
        "sentence_original": case["sentence_original"],
        "original_word_count": count,
        "micro_add_word_limit": max(count + 3, math.floor(1.60 * count)),
        "de_word_limit": max(1, math.floor(1.10 * count)),
        "neutral_word_min": max(1, math.ceil(0.75 * count)),
        "neutral_word_max": max(1, math.floor(1.35 * count)),
    }
    runtime = protocol["runtime"]
    options = dict(runtime["options"])
    options["seed"] = derive_attempt_seed(int(runtime["master_seed"]), f"v2-held-out:{case['case_id']}", attempt)
    return {
        "model": runtime["model"],
        "messages": [
            {"role": "system", "content": protocol["prompts"]["system"]},
            {"role": "user", "content": protocol["prompts"]["user_templates"][case["edit_type"]].format(**values)},
        ],
        "stream": runtime["stream"],
        "think": runtime["thinking"],
        "format": runtime["response_schema"],
        "options": options,
        "keep_alive": runtime["keep_alive"],
    }


def run_held_out_confirmation(v2_config_path: Path, *, resume: bool = False, timeout: float = 300.0) -> dict[str, Any]:
    protocol, base, cases = _load_v2_context(v2_config_path)
    identity_record = copy.deepcopy(base)
    identity_record["qwen"].update(protocol["runtime"])
    verify_ollama_identity(identity_record)
    raw = Path(protocol["outputs"]["raw_directory"])
    raw.mkdir(parents=True, exist_ok=True)
    attempts_path = raw / "attempts.jsonl"
    existing = []
    if attempts_path.exists():
        existing = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line]
        if existing and not resume:
            raise ValueError("held-out output exists; protocol cannot be rerun or revised")
    protocol_hash = sha256_file(v2_config_path)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in existing:
        if row["protocol_sha256"] != protocol_hash or row["case_id"] not in cases:
            raise ValueError("invalid held-out resume identity")
        by_case[row["case_id"]].append(row)
    with attempts_path.open("a", encoding="utf-8", newline="\n") as handle:
        for identifier in sorted(cases):
            case = cases[identifier]
            prior = by_case[identifier]
            if any(row["status"] == "accepted" for row in prior):
                continue
            for attempt in range(len(prior) + 1, int(protocol["runtime"]["maximum_attempts_per_case"]) + 1):
                request = _v2_request(protocol, case, attempt)
                try:
                    response = requests.post(protocol["runtime"]["api_url"], json=request, timeout=timeout)
                    response.raise_for_status()
                    edited, response_meta = parse_generation_response(response.json(), protocol["runtime"]["model"])
                    reasons, metrics = _v2_gate(protocol, base, case["sentence_original"], edited, case["edit_type"])
                except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                    edited = ""
                    reasons = [f"request_or_parse_error:{type(exc).__name__}"]
                    metrics = {}
                    response_meta = {"response_content_sha256": hashlib.sha256(b"").hexdigest()}
                row = {
                    "case_id": identifier,
                    "corpus_id": case["corpus_id"],
                    "edit_type": case["edit_type"],
                    "attempt_index": attempt,
                    "attempt_seed": request["options"]["seed"],
                    "status": "accepted" if not reasons else "rejected",
                    "reason_codes": reasons,
                    "edited_sentence": edited.strip(),
                    "edited_sha256": hashlib.sha256(edited.strip().encode("utf-8")).hexdigest(),
                    "protocol_sha256": protocol_hash,
                    "model_blob_sha256": protocol["runtime"]["backing_blob_sha256"],
                    **metrics,
                    **response_meta,
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
                handle.flush()
                by_case[identifier].append(row)
                if not reasons:
                    break
    rows = [json.loads(line) for line in attempts_path.read_text(encoding="utf-8").splitlines() if line]
    accepted = {row["case_id"]: row for row in rows if row["status"] == "accepted"}
    _write_csv(
        raw / "accepted.csv",
        [
            {
                "case_id": identifier,
                "corpus_id": cases[identifier]["corpus_id"],
                "edit_type": cases[identifier]["edit_type"],
                "attempt_index": row["attempt_index"],
                "edited_sentence": row["edited_sentence"],
                "edited_sha256": row["edited_sha256"],
            }
            for identifier, row in sorted(accepted.items())
        ],
    )
    cell_counts: Counter[tuple[str, str, str]] = Counter()
    for case in cases.values():
        cell_counts[("planned", case["corpus_id"], case["edit_type"])] += 1
        if case["case_id"] in accepted:
            cell_counts[("accepted", case["corpus_id"], case["edit_type"])] += 1
    cells = [
        {
            "corpus_id": corpus,
            "edit_type": direction,
            "planned": cell_counts[("planned", corpus, direction)],
            "accepted": cell_counts[("accepted", corpus, direction)],
            "acceptance_rate": cell_counts[("accepted", corpus, direction)] / cell_counts[("planned", corpus, direction)],
        }
        for corpus, direction in sorted({(case["corpus_id"], case["edit_type"]) for case in cases.values()})
    ]
    passed = len(accepted) >= 24 and all(cell["accepted"] >= math.ceil(0.75 * cell["planned"]) for cell in cells)
    summary = {
        "planned": len(cases),
        "accepted": len(accepted),
        "attempt_rows": len(rows),
        "coverage_gate_passed": passed,
        "cells": cells,
        "attempts_sha256": sha256_file(attempts_path),
        "accepted_sha256": sha256_file(raw / "accepted.csv"),
    }
    (raw / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    # Return aggregate confirmation only; held-out text is never printed.
    return summary


def summarize_held_out_confirmation(v2_config_path: Path) -> dict[str, Any]:
    protocol, _, cases = _load_v2_context(v2_config_path)
    raw = Path(protocol["outputs"]["raw_directory"])
    compact = Path(protocol["outputs"]["compact_directory"])
    compact.mkdir(parents=True, exist_ok=True)
    summary = _load_json(raw / "summary.json")
    attempts = [json.loads(line) for line in (raw / "attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    if {row["case_id"] for row in attempts} != set(cases):
        raise ValueError("held-out attempt accounting incomplete")
    coverage_rows = [
        {
            **cell,
            "minimum_accepted": math.ceil(0.75 * int(cell["planned"])),
            "cell_gate_passed": int(cell["accepted"]) >= math.ceil(0.75 * int(cell["planned"])),
        }
        for cell in summary["cells"]
    ]
    failure_counts: Counter[tuple[str, str, str]] = Counter()
    accounting_rows = []
    accepted_rows = []
    for row in attempts:
        for reason in row["reason_codes"]:
            failure_counts[(row["corpus_id"], row["edit_type"], reason)] += 1
        accounting_rows.append(
            {
                "case_id": row["case_id"],
                "corpus_id": row["corpus_id"],
                "edit_type": row["edit_type"],
                "attempt_index": row["attempt_index"],
                "attempt_seed": row["attempt_seed"],
                "status": row["status"],
                "reason_codes": "|".join(row["reason_codes"]),
                "edited_sha256": row["edited_sha256"],
                "token_ratio": row.get("token_ratio", ""),
                "content_anchor_recall": row.get("content_anchor_recall", ""),
                "original_concrete_marker_count": row.get("original_concrete_marker_count", ""),
                "edited_concrete_marker_count": row.get("edited_concrete_marker_count", ""),
            }
        )
        if row["status"] == "accepted":
            accepted_rows.append(
                {
                    "case_id": row["case_id"],
                    "corpus_id": row["corpus_id"],
                    "edit_type": row["edit_type"],
                    "accepted_attempt_index": row["attempt_index"],
                    "edited_sha256": row["edited_sha256"],
                }
            )
    failure_rows = [
        {"corpus_id": corpus, "edit_type": direction, "reason_code": reason, "count": count}
        for (corpus, direction, reason), count in sorted(failure_counts.items())
    ] or [{"corpus_id": "all", "edit_type": "all", "reason_code": "none", "count": 0}]
    _write_csv(compact / "coverage.csv", coverage_rows)
    _write_csv(compact / "failure_reasons.csv", failure_rows)
    _write_csv(compact / "attempt_accounting.csv", accounting_rows)
    _write_csv(compact / "accepted_manifest.csv", accepted_rows)
    disposition = "pass_recommend_separate_scoring_review" if summary["coverage_gate_passed"] else "fail_recommend_omit_qwen_edit_arm"
    metadata = {
        "schema_version": "round2_qwen_edit_remediation_held_out_results_v1",
        "v2_protocol_sha256": sha256_file(v2_config_path),
        "v2_freeze_commit": _load_json(V2_FREEZE_RECORD)["v2_freeze_commit"],
        "planned_cases": summary["planned"],
        "accepted_cases": summary["accepted"],
        "attempt_rows": summary["attempt_rows"],
        "coverage_gate_passed": summary["coverage_gate_passed"],
        "disposition": disposition,
        "scoring_performed": False,
        "held_out_text_printed_or_interactively_inspected": False,
        "raw_attempts_sha256": summary["attempts_sha256"],
        "raw_accepted_sha256": summary["accepted_sha256"],
        "outputs": {name: sha256_file(compact / name) for name in ("coverage.csv", "failure_reasons.csv", "attempt_accounting.csv", "accepted_manifest.csv")},
        "privacy_boundary": protocol["outputs"]["privacy_boundary"],
    }
    (compact / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# QE-R v2 held-out confirmation", "",
        f"The single frozen held-out run accepted {summary['accepted']}/{summary['planned']} cases. Coverage gate passed: {str(summary['coverage_gate_passed']).lower()}.", "",
        "| Corpus | Direction | Accepted / planned | Required | Pass |", "| --- | --- | ---: | ---: | --- |",
    ]
    for cell in coverage_rows:
        lines.append(f"| {cell['corpus_id']} | {cell['edit_type']} | {cell['accepted']}/{cell['planned']} | {cell['minimum_accepted']} | {str(cell['cell_gate_passed']).lower()} |")
    lines.extend(["", f"Disposition: `{disposition}`.", "", "No SpeciTeller, Ko, GranuScore, Qwen-rubric, or human scoring/judgment was performed. Raw held-out text remains ignored and was not printed or interactively inspected.", ""])
    (compact / "README.md").write_text("\n".join(lines), encoding="utf-8")
    metadata["outputs"]["README.md"] = sha256_file(compact / "README.md")
    (compact / "run_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


__all__ = [
    "build_split_rows", "diagnose_v1", "ordered_assignment_sha256",
    "run_held_out_confirmation", "summarize_held_out_confirmation",
    "run_development_candidates", "summarize_development_candidates",
    "validate_split", "write_split",
]
