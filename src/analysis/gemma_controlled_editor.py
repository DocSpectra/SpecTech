"""Frozen Gemma 4 controlled-editor generation and blinded packet workflow."""
from __future__ import annotations

import base64
import csv
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import requests

from src.analysis.pilot_model_human import sha256_file


SCHEMA_VERSION = "round2_gemma_controlled_editor_v1"
ATTEMPT_SCHEMA_VERSION = "round2_gemma_editor_attempt_v1"
FREEZE_RECORD = Path("configs/round2_gemma_controlled_editor_freeze_record.json")
EDIT_TYPES = ("add_specific", "de_specify", "irrelevant_rewrite")

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
BOUNDARY_RE = re.compile(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)")


@dataclass(frozen=True)
class GemmaCase:
    case_id: str
    corpus_id: str
    source_position: int
    source_sent_id: str
    edit_type: str
    sentence_original: str
    original_sha256: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _case_id(corpus_id: str, sent_id: str, edit_type: str, position: int) -> str:
    return _sha_text(f"{corpus_id}\0{sent_id}\0{edit_type}\0{position}")


def _require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


def load_protocol(config_path: Path) -> tuple[dict[str, Any], str, dict[str, Any]]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION or record.get("outcome_blind_method_freeze") is not True:
        raise ValueError("unexpected or unfrozen Gemma protocol")
    protocol_hash = sha256_file(config_path)
    freeze = json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))
    if freeze.get("config_sha256") != protocol_hash:
        raise ValueError("Gemma freeze record/config mismatch")
    if freeze.get("method_frozen_before_any_project_gemma_generation") is not True:
        raise ValueError("Gemma pre-generation freeze gate is absent")
    return record, protocol_hash, freeze


def load_cases(record: dict[str, Any]) -> list[GemmaCase]:
    cases: list[GemmaCase] = []
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
            raise ValueError(f"Gemma source row count mismatch: {corpus_id}")
        for position, row in enumerate(rows, start=1):
            if row["corpus_id"] != corpus_id or row["edit_type"] not in EDIT_TYPES:
                raise ValueError("Gemma source corpus/direction mismatch")
            identifier = _case_id(corpus_id, row["sent_id"], row["edit_type"], position)
            original = row["sentence_original"]
            if identifier in seen or not original.strip():
                raise ValueError("Gemma duplicate or empty source case")
            seen.add(identifier)
            original_hash = _sha_text(original)
            digest_row = "\0".join(
                [corpus_id, str(position), identifier, row["sent_id"], row["edit_type"], original_hash]
            )
            ordered_digest.update(digest_row.encode("utf-8")); ordered_digest.update(b"\n")
            counts[(corpus_id, row["edit_type"])] += 1
            cases.append(
                GemmaCase(
                    case_id=identifier,
                    corpus_id=corpus_id,
                    source_position=position,
                    source_sent_id=row["sent_id"],
                    edit_type=row["edit_type"],
                    sentence_original=original,
                    original_sha256=original_hash,
                )
            )
        observed = {key: counts[(corpus_id, key)] for key in entry["direction_counts"]}
        if observed != entry["direction_counts"]:
            raise ValueError(f"Gemma direction counts changed: {corpus_id}")
    if len(cases) != int(record["inputs"]["required_total_cases"]):
        raise ValueError("Gemma planned-case count mismatch")
    if ordered_digest.hexdigest() != record["inputs"]["ordered_case_sha256"]:
        raise ValueError("Gemma ordered-case digest mismatch")
    return cases


def _ollama_version() -> str:
    return subprocess.run(
        ["ollama", "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()


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
        raise ValueError("exactly one pinned Gemma tag is required")
    tag = matches[0]
    if tag.get("digest") != runtime["model_tag_digest"] or int(tag.get("size", -1)) != runtime["installed_size_bytes"]:
        raise ValueError("Gemma tag digest or installed size mismatch")
    details = tag.get("details", {})
    expected_details = {
        "family": runtime["family"],
        "parameter_size": runtime["parameter_size"],
        "quantization_level": runtime["quantization"],
        "format": runtime["format"],
    }
    for key, expected in expected_details.items():
        if details.get(key) != expected:
            raise ValueError(f"Gemma tag detail mismatch: {key}")
    show = _api_json("POST", f"{root_url}/api/show", payload={"model": runtime["model"], "verbose": True})
    if int(show.get("model_info", {}).get("general.parameter_count", -1)) != runtime["parameter_count"]:
        raise ValueError("Gemma parameter-count mismatch")
    if int(show.get("projector_info", {}).get("general.parameter_count", -1)) != runtime["projector_parameter_count"]:
        raise ValueError("Gemma projector parameter-count mismatch")
    parameters = show.get("parameters", "")
    for line in ("temperature                    1", "top_k                          64", "top_p                          0.95"):
        if line not in parameters:
            raise ValueError("Gemma default parameter mismatch")
    modelfile = subprocess.run(
        ["ollama", "show", runtime["model"], "--modelfile"], check=True, capture_output=True, text=True
    ).stdout
    for blob in (runtime["model_blob_sha256"], runtime["projector_blob_sha256"]):
        if f"sha256-{blob}" not in modelfile:
            raise ValueError("Gemma blob identity mismatch")
    license_text = subprocess.run(
        ["ollama", "show", runtime["model"], "--license"], check=True, capture_output=True, text=True
    ).stdout
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Gemma license mismatch")
    return {
        "ollama_version_output": version,
        "model": runtime["model"],
        "model_tag_digest": runtime["model_tag_digest"],
        "installed_size_bytes": runtime["installed_size_bytes"],
        "model_blob_sha256": runtime["model_blob_sha256"],
        "projector_blob_sha256": runtime["projector_blob_sha256"],
    }


def _running_models(record: dict[str, Any]) -> list[dict[str, Any]]:
    root_url = record["runtime"]["api_url"].rsplit("/api/", 1)[0]
    return list(_api_json("GET", f"{root_url}/api/ps").get("models", []))


def require_only_pinned_gemma_resident(record: dict[str, Any]) -> dict[str, Any]:
    rows = _running_models(record)
    runtime = record["runtime"]
    if len(rows) != 1:
        raise ValueError(f"expected one resident model, observed {len(rows)}")
    row = rows[0]
    if row.get("name") != runtime["model"] or row.get("digest") != runtime["model_tag_digest"]:
        raise ValueError("resident model is not exact pinned Gemma")
    return {
        "name": row.get("name"), "digest": row.get("digest"),
        "size": row.get("size"), "size_vram": row.get("size_vram"),
        "context_length": row.get("context_length"),
    }


def stop_ollama_model(model: str) -> None:
    subprocess.run(["ollama", "stop", model], check=True, capture_output=True, text=True)


def derive_attempt_seed(master_seed: int, identifier: str, attempt_index: int) -> int:
    value = f"{master_seed}:gemma-editor-v1:{identifier}:{attempt_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "big") & 0x7FFFFFFF


def _normalized(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).strip().split())


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(unicodedata.normalize("NFKC", value))]


def _content_tokens(value: str) -> set[str]:
    return {token for token in _tokens(value) if len(token) >= 3 and token not in STOPWORDS}


def _multiset_from_vocab(value: str, vocabulary: Iterable[str]) -> Counter[str]:
    vocab = set(vocabulary)
    return Counter(token for token in _tokens(value) if token in vocab)


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
    return sum(min(count, edited.get(key, 0)) for key, count in original.items()) / total


def gate_edit(record: dict[str, Any], original: str, edited: str, edit_type: str) -> tuple[list[str], dict[str, Any]]:
    reasons: list[str] = []
    universal = record["automated_gates"]["universal"]
    stripped = edited.strip()
    if not stripped or any(char in stripped for char in "\n\r\t"):
        reasons.append("not_one_nonempty_line")
    if len(stripped) < int(universal["minimum_characters"]):
        reasons.append("too_short_chars")
    if len(stripped) > int(universal["maximum_characters"]):
        reasons.append("too_long_chars")
    if _normalized(stripped).casefold() == _normalized(original).casefold():
        reasons.append("unchanged_normalized")
    if any(stripped.casefold().startswith(prefix) for prefix in universal["forbidden_meta_prefixes_casefold"]):
        reasons.append("meta_prefix")
    boundaries = BOUNDARY_RE.findall(stripped)
    if len(boundaries) > 1:
        reasons.append("multiple_sentences")
    original_has_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in original)
    edited_has_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in stripped)
    if not original_has_non_latin and edited_has_non_latin:
        reasons.append("new_non_latin_script")
    original_tokens, edited_tokens = _tokens(original), _tokens(stripped)
    ratio = len(edited_tokens) / max(1, len(original_tokens))
    original_content, edited_content = _content_tokens(original), _content_tokens(stripped)
    content_recall = len(original_content & edited_content) / max(1, len(original_content))
    original_markers, edited_markers = _markers(original), _markers(stripped)
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
    original_marker_count, edited_marker_count = sum(original_markers.values()), sum(edited_markers.values())
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
        proxy_pass = (
            len(edited_tokens) <= len(original_tokens) - 1
            or edited_marker_count < original_marker_count
            or (content_recall <= 0.9 and len(edited_content) <= len(original_content))
        )
        if not proxy_pass:
            reasons.append("insufficient_despecification_proxy")
    elif original_markers != edited_markers:
        reasons.append("neutral_concrete_markers_changed")
    metrics = {
        "original_token_count": len(original_tokens),
        "edited_token_count": len(edited_tokens),
        "token_ratio": ratio,
        "content_anchor_recall": content_recall,
        "original_content_token_count": len(original_content),
        "edited_content_token_count": len(edited_content),
        "original_concrete_marker_count": original_marker_count,
        "edited_concrete_marker_count": edited_marker_count,
        "original_concrete_marker_recall": marker_recall,
        "sentence_boundary_count": len(boundaries),
    }
    return sorted(set(reasons)), metrics


def _prompt_values(case: GemmaCase) -> dict[str, Any]:
    count = len(_tokens(case.sentence_original))
    return {
        "sentence_original": case.sentence_original,
        "original_word_count": count,
        "add_word_limit": max(count + 3, math.floor(1.60 * count)),
        "de_word_limit": max(1, math.floor(1.10 * count)),
        "neutral_word_min": max(1, math.ceil(0.75 * count)),
        "neutral_word_max": max(1, math.floor(1.35 * count)),
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
        "stream": runtime["stream"],
        "think": runtime["thinking"],
        "format": runtime["response_schema"],
        "options": options,
        "keep_alive": runtime["keep_alive"],
    }


def parse_generation_response(payload: dict[str, Any], expected_model: str) -> tuple[str, dict[str, Any]]:
    if payload.get("model") != expected_model:
        raise ValueError("Ollama response model mismatch")
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("Ollama response lacks string content")
    thinking = payload.get("message", {}).get("thinking")
    if thinking not in (None, ""):
        raise ValueError("thinking content present despite frozen think=false")
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
    }


def _warm_pinned_gemma(record: dict[str, Any], timeout: float) -> dict[str, Any]:
    runtime = record["runtime"]
    request = {
        "model": runtime["model"],
        "messages": [{"role": "user", "content": "Return JSON with edited_sentence equal to: Runtime fixture complete."}],
        "stream": False,
        "think": False,
        "format": runtime["response_schema"],
        "options": {**runtime["options"], "seed": 2026081100},
        "keep_alive": runtime["keep_alive"],
    }
    response = requests.post(runtime["api_url"], json=request, timeout=timeout)
    response.raise_for_status()
    _, metadata = parse_generation_response(response.json(), runtime["model"])
    return {"request_sha256": _canonical_sha(request), **metadata, "residency": require_only_pinned_gemma_resident(record)}


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fieldnames or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _validate_resume(rows: list[dict[str, Any]], cases: list[GemmaCase], record: dict[str, Any], config_hash: str) -> None:
    case_map = {case.case_id: case for case in cases}
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        case = case_map.get(row.get("case_id"))
        if case is None or row.get("config_sha256") != config_hash:
            raise ValueError("invalid Gemma resume identity")
        if row.get("model_tag_digest") != record["runtime"]["model_tag_digest"]:
            raise ValueError("invalid Gemma resume model identity")
        if row.get("original_sha256") != case.original_sha256:
            raise ValueError("invalid Gemma resume original identity")
        by_case[case.case_id].append(row)
    for values in by_case.values():
        attempts = [int(row["attempt_index"]) for row in values]
        if attempts != list(range(1, len(values) + 1)):
            raise ValueError("Gemma resumed attempts are not contiguous")
        passing = [row for row in values if row["status"] == "accepted"]
        if len(passing) > 1 or (passing and passing[0] is not values[-1]):
            raise ValueError("Gemma resumed first-pass policy violated")


def _review_id(record: dict[str, Any], case_id: str) -> str:
    review = record["blinded_review"]
    value = f"{review['review_id_seed']}:{review['review_id_namespace']}:{case_id}".encode("utf-8")
    encoded = base64.b32encode(hashlib.sha256(value).digest()).decode("ascii").rstrip("=")
    return "GCE-" + encoded[:12]


def _candidate_sides(record: dict[str, Any], cases: list[GemmaCase]) -> dict[str, str]:
    seed = record["blinded_review"]["side_assignment_seed"]
    groups: dict[tuple[str, str], list[GemmaCase]] = defaultdict(list)
    for case in cases:
        groups[(case.corpus_id, case.edit_type)].append(case)
    sides: dict[str, str] = {}
    for values in groups.values():
        ranked = sorted(values, key=lambda case: _sha_text(f"{seed}:{case.case_id}"))
        if len(ranked) % 2:
            raise ValueError("exact within-cell A/B balance requires even cell size")
        for index, case in enumerate(ranked):
            sides[case.case_id] = "A" if index < len(ranked) // 2 else "B"
    return sides


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
            "review_row": position, "review_id": review_id,
            "sentence_a": sentence_a, "sentence_b": sentence_b,
            "factual_correctness_a": "", "factual_correctness_b": "",
            "semantic_preservation": "", "grammar_naturalness_a": "",
            "grammar_naturalness_b": "", "more_specific": "", "confidence": "", "notes": "",
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
    packet_path = raw / record["outputs"]["review_packet"]
    key_path = raw / record["outputs"]["review_answer_key"]
    _write_csv(packet_path, packet_rows, record["blinded_review"]["packet_fields"])
    _write_csv(key_path, key_rows)
    response_path = raw / record["outputs"]["review_responses"]
    _write_csv(response_path, packet_rows, record["blinded_review"]["packet_fields"])
    instructions = """# Blinded controlled-edit review\n\nReview all 60 rows without trying to infer which sentence is the original or the intended direction. Use only these exact response values:\n\n- factual_correctness_a / factual_correctness_b: correct, incorrect, uncertain, not_assessable\n- semantic_preservation: preserved, mostly_preserved, not_preserved, uncertain\n- grammar_naturalness_a / grammar_naturalness_b: natural, minor_issue, major_issue, uncertain\n- more_specific: A, B, tie, uncertain\n- confidence: integer 1 through 5\n- notes: optional free text\n\nJudge factual correctness and grammar/naturalness separately for A and B. Judge semantic preservation for the pair. Do not open the answer key before every response is complete. Fill blinded_review_responses.csv; leave blinded_review_packet.csv unchanged.\n"""
    (raw / "blinded_review_instructions.md").write_text(instructions, encoding="utf-8")
    return validate_packet_integrity(record, cases, accepted)


def validate_packet_integrity(record: dict[str, Any], cases: list[GemmaCase], accepted: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw = Path(record["outputs"]["raw_directory"])
    packet_path = raw / record["outputs"]["review_packet"]
    key_path = raw / record["outputs"]["review_answer_key"]
    response_path = raw / record["outputs"]["review_responses"]
    packet, keys = _read_csv(packet_path), _read_csv(key_path)
    if len(packet) != len(keys) != 60:
        raise ValueError("invalid packet/key length")
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
    forbidden = set(record["blinded_review"]["packet_forbidden_fields"])
    if forbidden.intersection(packet[0]):
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
        {"corpus_id": corpus, "edit_type": direction, "planned": count, "retained": retained[(corpus, direction)], "retention_rate": f"{retained[(corpus, direction)] / count:.6f}"}
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
    manifest = {
        "schema_version": "round2_gemma_controlled_editor_packet_manifest_v1",
        "config_sha256": config_hash,
        "method_freeze_commit": json.loads(FREEZE_RECORD.read_text(encoding="utf-8"))["method_freeze_commit"],
        "model": record["runtime"]["model"], "model_tag_digest": record["runtime"]["model_tag_digest"],
        "model_blob_sha256": record["runtime"]["model_blob_sha256"],
        "projector_blob_sha256": record["runtime"]["projector_blob_sha256"],
        "started_at_utc": started, "completed_at_utc": completed,
        "planned_cases": len(cases), "retained_cases": len(accepted), "attempt_rows": len(rows),
        "generation_gate_passed": len(accepted) == int(record["generation_gate"]["required_retained_total"]),
        "packet_released": packet is not None, "residency_after_generation": residency,
        "raw_hashes": {**raw_hashes, **(packet or {})},
        "scoring_performed": False, "rubric_performed": False, "human_review_performed": False,
        "content_boundary": "No sentence text, A/B answer, review response, score, label, participant field, or host path is tracked.",
    }
    (compact / "packet_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    readme = f"""# Gemma controlled-editor pilot generation\n\n- Model: `{record['runtime']['model']}` (`{record['runtime']['model_list_id']}`)\n- Planned/retained: {len(cases)}/{len(accepted)}\n- Attempt rows: {len(rows)}\n- Generation gate: {'passed' if manifest['generation_gate_passed'] else 'failed'}\n- Blinded packet: {'released with 60 rows and reversible key integrity' if packet else 'not released'}\n- Scoring, rubric, and human review: not run\n\nThis tracked directory contains aggregate non-content diagnostics only. Raw attempts, generated text, source text, A/B packet, answer key, and review template remain ignored under `outputs/`. Mechanical acceptance is not factual, semantic, grammatical, or manipulation validation.\n"""
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
        raise ValueError("Gemma outputs exist; use --resume only for an interrupted frozen run")
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
                request_hash = _canonical_sha(request)
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
                    }
                stripped = edited.strip()
                row = {
                    "schema_version": ATTEMPT_SCHEMA_VERSION, "config_sha256": config_hash,
                    "case_id": case.case_id, "corpus_id": case.corpus_id,
                    "source_position": case.source_position, "source_sent_id": case.source_sent_id,
                    "edit_type": case.edit_type, "original_sha256": case.original_sha256,
                    "attempt_index": attempt_index, "seed": request["options"]["seed"],
                    "model": record["runtime"]["model"], "model_tag_digest": record["runtime"]["model_tag_digest"],
                    "request_sha256": request_hash, "edited_sentence": stripped,
                    "edited_sha256": _sha_text(stripped), "status": "accepted" if not reasons else "rejected",
                    "reason_codes": reasons, "gate_metrics": metrics, **response_meta,
                }
                handle.write(json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"); handle.flush()
                by_case[case.case_id].append(row)
                if not reasons:
                    break
    rows = _read_jsonl(attempts_path)
    _validate_resume(rows, cases, record, config_hash)
    accepted = {row["case_id"]: row for row in rows if row["status"] == "accepted"}
    retained_rows = [
        {
            "case_id": case.case_id, "corpus_id": case.corpus_id, "source_position": case.source_position,
            "source_sent_id": case.source_sent_id, "edit_type": case.edit_type,
            "original_sha256": case.original_sha256, "attempt_index": accepted[case.case_id]["attempt_index"],
            "seed": accepted[case.case_id]["seed"], "edited_sentence": accepted[case.case_id]["edited_sentence"],
            "edited_sha256": accepted[case.case_id]["edited_sha256"],
        }
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
    stop_ollama_model("qwen3:14b"); stop_ollama_model(record["runtime"]["model"])
    warmup: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
    unloaded = False
    try:
        warmup = _warm_pinned_gemma(record, timeout)
        result = generate(record, config_hash, cases, resume=resume, timeout=timeout)
        residency = require_only_pinned_gemma_resident(record)
        completed = _utc_now()
        manifest = _write_compact(
            record, config_hash, cases, result["rows"], result["accepted"], result["packet"],
            residency, started, completed,
        )
        raw_metadata = {
            "schema_version": "round2_gemma_controlled_editor_run_metadata_v1",
            "config_sha256": config_hash, "method_freeze_commit": freeze["method_freeze_commit"],
            "started_at_utc": started, "completed_at_utc": completed,
            "identity": identity, "non_project_warmup": warmup,
            "residency_after_generation": residency,
            "planned_cases": len(cases), "retained_cases": len(result["accepted"]),
            "attempt_rows": len(result["rows"]), "packet_released": result["packet"] is not None,
            "scoring_performed": False, "rubric_performed": False, "human_review_performed": False,
        }
        raw = Path(record["outputs"]["raw_directory"])
        (raw / record["outputs"]["raw_run_metadata"]).write_text(
            json.dumps(raw_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return manifest
    finally:
        stop_ollama_model(record["runtime"]["model"])
        unloaded = not _running_models(record)
        if not unloaded:
            raise RuntimeError("Ollama model remained resident after controlled run")


def validate_existing_packet(config_path: Path) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path)
    cases = load_cases(record)
    raw = Path(record["outputs"]["raw_directory"])
    rows = _read_jsonl(raw / record["outputs"]["attempts"])
    accepted = {row["case_id"]: row for row in rows if row["status"] == "accepted"}
    if len(accepted) != 60:
        raise ValueError("a complete 60-case retained set is required")
    return validate_packet_integrity(record, cases, accepted)
