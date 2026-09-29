"""Independent primary-review checks for the coverage-gated QE-A run."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "outputs" / "round2" / "qwen_edit_source"
COMPACT = ROOT / "analysis" / "round2_qwen_edit_source"
CONFIG = ROOT / "configs" / "round2_qwen_edit_source_v1.json"
FREEZE = ROOT / "configs" / "round2_qwen_edit_source_freeze_record.json"

STOPWORDS = frozenset(
    "a an and are as at be been being but by for from had has have he her hers "
    "him his i if in into is it its me my no nor not of on or our ours she so "
    "than that the their theirs them they this those to too us was we were what "
    "when where which who why will with you your yours".split()
)
WORD_RE = re.compile(r"(?u)\b\w+(?:[-./:]\w+)*\b")
MARKER_RES = (
    re.compile(r"https?://\S+", re.IGNORECASE),
    re.compile(r"(?<!\w)(?:[A-Za-z]:\\[^\s,;]+|/(?:[^\s,;]+))"),
    re.compile(r"(?<!\w)--?[A-Za-z][A-Za-z0-9-]*"),
    re.compile(r"(?<!\w)\d+(?:\.\d+)*(?!\w)"),
    re.compile(r"\b[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+\b"),
    re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b"),
)
BACKTICK_RE = re.compile(r"`+([^`]+?)`+")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _tokens(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", value)
    return [match.group(0).casefold() for match in WORD_RE.finditer(normalized)]


def _multiset(value: str, vocabulary: list[str]) -> Counter[str]:
    allowed = set(vocabulary)
    return Counter(token for token in _tokens(value) if token in allowed)


def _markers(value: str) -> Counter[str]:
    found: list[str] = []
    for regex in MARKER_RES:
        found.extend(match.group(0).casefold().rstrip(".,;:)\"]'") for match in regex.finditer(value))
    found.extend(match.group(1).strip().casefold() for match in BACKTICK_RE.finditer(value))
    return Counter(item for item in found if item)


def _marker_recall(original: Counter[str], edited: Counter[str]) -> float:
    total = sum(original.values())
    return 1.0 if total == 0 else sum(min(count, edited.get(key, 0)) for key, count in original.items()) / total


def _independent_gate(config: dict, original: str, edited: str, edit_type: str) -> list[str]:
    """Recompute frozen reasons without importing the implementation under review."""
    reasons: list[str] = []
    universal = config["automated_gates"]["universal"]
    stripped = edited.strip()
    if not stripped or any(char in stripped for char in "\n\r\t"):
        reasons.append("not_one_nonempty_line")
    if len(stripped) < int(universal["minimum_characters"]):
        reasons.append("too_short_chars")
    if len(stripped) > int(universal["maximum_characters"]):
        reasons.append("too_long_chars")
    normalize = lambda value: " ".join(unicodedata.normalize("NFKC", value).strip().split()).casefold()
    if normalize(stripped) == normalize(original):
        reasons.append("unchanged_normalized")
    if any(stripped.casefold().startswith(prefix) for prefix in universal["forbidden_meta_prefixes_casefold"]):
        reasons.append("meta_prefix")

    original_tokens = _tokens(original)
    edited_tokens = _tokens(stripped)
    ratio = len(edited_tokens) / max(1, len(original_tokens))
    original_content = {token for token in original_tokens if len(token) >= 3 and token not in STOPWORDS}
    edited_content = {token for token in edited_tokens if len(token) >= 3 and token not in STOPWORDS}
    content_recall = len(original_content & edited_content) / max(1, len(original_content))
    original_markers = _markers(original)
    edited_markers = _markers(stripped)
    if _multiset(original, config["automated_gates"]["polarity_tokens"]) != _multiset(
        stripped, config["automated_gates"]["polarity_tokens"]
    ):
        reasons.append("polarity_changed")
    if _multiset(original, config["automated_gates"]["modal_tokens"]) != _multiset(
        stripped, config["automated_gates"]["modal_tokens"]
    ):
        reasons.append("modality_changed")

    rule = config["automated_gates"]["direction_rules"][edit_type]
    if ratio < float(rule["edited_to_original_token_ratio_min"]):
        reasons.append("token_ratio_below_min")
    if ratio > float(rule["edited_to_original_token_ratio_max"]):
        reasons.append("token_ratio_above_max")
    if content_recall < float(rule["original_content_anchor_recall_min"]):
        reasons.append("content_anchor_recall_below_min")
    original_marker_count = sum(original_markers.values())
    edited_marker_count = sum(edited_markers.values())
    if edit_type == "add_specific":
        if _marker_recall(original_markers, edited_markers) < float(rule["original_concrete_marker_recall_min"]):
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
    elif original_markers != edited_markers:
        reasons.append("neutral_concrete_markers_changed")
    return sorted(set(reasons))


def _source_cases(config: dict) -> dict[str, dict[str, str]]:
    cases: dict[str, dict[str, str]] = {}
    for corpus_id in config["inputs"]["corpus_order"]:
        for position, row in enumerate(_csv(ROOT / config["inputs"]["source_files"][corpus_id]["path"]), start=1):
            payload = f"{corpus_id}\0{row['sent_id']}\0{row['edit_type']}\0{position}".encode()
            cases[hashlib.sha256(payload).hexdigest()] = row
    return cases


def test_primary_review_recomputes_prefixed_failure_and_corrected_lifecycle() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    pre = _jsonl(RAW / "pre_schema_fix" / "generation_attempts.jsonl")
    assert len(pre) == 180
    assert len({row["case_id"] for row in pre}) == 60
    assert set(Counter(row["case_id"] for row in pre).values()) == {3}
    assert {row["status"] for row in pre} == {"rejected"}
    assert {row["edited_sentence"] for row in pre} == {""}
    assert {tuple(row["reason_codes"]) for row in pre} == {("request_or_parse_error:HTTPError",)}
    assert _sha(RAW / "pre_schema_fix" / "generation_attempts.jsonl") == freeze["outcome_blind_mechanical_correction"]["pre_fix_attempts_sha256"]

    attempts = _jsonl(RAW / "generation_attempts.jsonl")
    cases = _source_cases(config)
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in attempts:
        by_case[row["case_id"]].append(row)
        assert row["protocol_sha256"] == _sha(CONFIG)
        assert row["model_blob_sha256"] == config["qwen"]["backing_blob_sha256"]
        assert row["reason_codes"] == _independent_gate(
            config, cases[row["case_id"]]["sentence_original"], row["edited_sentence"], row["edit_type"]
        )
        assert (row["status"] == "accepted") == (not row["reason_codes"])
    assert len(attempts) == 137 and set(by_case) == set(cases)
    assert Counter(row["status"] for row in attempts) == {"rejected": 113, "accepted": 24}
    for rows in by_case.values():
        assert [row["attempt_index"] for row in rows] == list(range(1, len(rows) + 1))
        passing = [row for row in rows if not row["reason_codes"]]
        assert len(passing) <= 1
        if passing:
            assert passing[0] is rows[-1]
        else:
            assert len(rows) == 3

    accepted = _csv(RAW / "accepted_edits.csv")
    accepted_by_case = {row["case_id"]: row for row in accepted}
    assert len(accepted_by_case) == 24
    for identifier, row in accepted_by_case.items():
        first_pass = next(item for item in by_case[identifier] if item["status"] == "accepted")
        assert row["accepted_attempt_index"] == str(first_pass["attempt_index"])
        assert row["sentence_qwen_edited"] == first_pass["edited_sentence"]


def test_primary_review_recomputes_coverage_reasons_and_frozen_stop() -> None:
    attempts = _jsonl(RAW / "generation_attempts.jsonl")
    manifest = _csv(RAW / "generation_manifest.csv")
    expected_coverage = {
        ("ansible_docs", "add_specific"): (8, 4),
        ("ansible_docs", "de_specify"): (12, 6),
        ("ansible_docs", "irrelevant_rewrite"): (10, 1),
        ("github_docs", "add_specific"): (10, 4),
        ("github_docs", "de_specify"): (10, 6),
        ("github_docs", "irrelevant_rewrite"): (10, 3),
    }
    observed: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for row in manifest:
        key = (row["corpus_id"], row["edit_type"])
        observed[key][0] += 1
        observed[key][1] += row["accepted"] == "true"
    assert {key: tuple(value) for key, value in observed.items()} == expected_coverage
    assert sum(value[1] for value in observed.values()) == 24
    assert all(accepted / planned < 0.75 for planned, accepted in observed.values())

    global_reasons: Counter[str] = Counter()
    cell_reasons: Counter[tuple[str, str, str]] = Counter()
    for row in attempts:
        if row["status"] == "rejected":
            for reason in row["reason_codes"]:
                global_reasons[reason] += 1
                cell_reasons[(row["corpus_id"], row["edit_type"], reason)] += 1
    assert global_reasons == {
        "content_anchor_recall_below_min": 45,
        "token_ratio_above_max": 43,
        "insufficient_despecification_proxy": 26,
        "modality_changed": 23,
        "token_ratio_below_min": 6,
        "neutral_concrete_markers_changed": 6,
        "unchanged_normalized": 3,
        "polarity_changed": 3,
    }
    compact_reasons = Counter({
        (row["corpus_id"], row["edit_type"], row["reason_code"]): int(row["failed_attempt_count"])
        for row in _csv(COMPACT / "failure_reasons.csv")
    })
    assert compact_reasons == cell_reasons
    compact_coverage = {
        (row["corpus_id"], row["edit_type"]): (int(row["planned"]), int(row["accepted"]))
        for row in _csv(COMPACT / "coverage.csv")
    }
    assert compact_coverage == expected_coverage
    assert not (COMPACT / "arm_summaries.csv").exists()
    assert not (COMPACT / "paired_source_contrasts.csv").exists()
    assert not (COMPACT / "paper_table.csv").exists()


def test_primary_review_recomputes_scorer_coverage_join_and_privacy() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    metadata = json.loads((COMPACT / "run_metadata.json").read_text(encoding="utf-8"))
    manifest = _csv(RAW / "scoring_manifest.csv")
    assert len(manifest) == len({row["score_id"] for row in manifest}) == 84
    assert Counter(row["edit_source"] for row in manifest) == {"author": 60, "qwen": 24}
    assert _sha(RAW / "scoring_manifest.csv") == "ef0bef5479a938568cb16bbe6cdad5343dd29b5b0885a3ba5f978338ffedafdc"
    manifest_ids = {row["score_id"] for row in manifest}

    spec_rows = [line.split("\t", 1) for line in (RAW / "scoring/speciteller/scores.tsv").read_text(encoding="utf-8").splitlines()]
    assert len(spec_rows) == 84 and {row[0] for row in spec_rows} == manifest_ids
    assert _sha(RAW / "scoring/speciteller/scores.tsv") == metadata["scoring"]["speciteller"]["score_sha256"]
    gran_rows = _csv(RAW / "scoring/granuscore/scores.csv")
    assert len(gran_rows) == 84 and {row["sent_id"] for row in gran_rows} == manifest_ids
    assert _sha(RAW / "scoring/granuscore/scores.csv") == metadata["scoring"]["granuscore"]["score_sha256"]
    assert metadata["scoring"]["granuscore"]["native_direction"] == "higher_is_coarser_more_abstract"

    corpus_counts = Counter(row["corpus_id"] for row in manifest)
    assert corpus_counts == {"ansible_docs": 41, "github_docs": 43}
    for corpus_id, expected in corpus_counts.items():
        for run_id in ("run01", "run02", "run03"):
            rows = _csv(RAW / "scoring" / "ko" / corpus_id / run_id / "scores.csv")
            assert len(rows) == expected
            assert {row["score_id"] for row in rows} == {
                row["score_id"] for row in manifest if row["corpus_id"] == corpus_id
            }
            run_meta = metadata["scoring"]["ko_runs"][f"{corpus_id}:{run_id}"]
            assert _sha(RAW / "scoring" / "ko" / corpus_id / run_id / "scores.csv") == run_meta["score_sha256"]
            assert run_meta["checkpoint_sha256"] == config["scoring"]["ko"]["runs"][corpus_id][run_id]["checkpoint_sha256"]

    long_rows = _csv(COMPACT / "scores_long.csv")
    assert len(long_rows) == 504
    joined: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in long_rows:
        key = (row["case_id"], row["edit_source"])
        assert row["model_instance_id"] not in joined[key]
        joined[key][row["model_instance_id"]] = row
        original = float(row["score_original"])
        edited = float(row["score_edited"])
        upper = 100.0 if row["model_instance_id"] == "granuscore_native" else 1.0
        assert 0.0 <= original <= upper and 0.0 <= edited <= upper
        assert math.isclose(edited - original, float(row["delta_edited_minus_original"]), abs_tol=2e-10)
    assert len(joined) == 84
    expected_models = set(metadata["scoring"]["models"])
    for models in joined.values():
        assert set(models) == expected_models
        ko_values = [float(models[f"ko_run0{i}"]["score_edited"]) for i in (1, 2, 3)]
        assert math.isclose(float(models["ko_three_run_arithmetic_mean"]["score_edited"]), sum(ko_values) / 3, abs_tol=1e-10)
        assert models["granuscore_native"]["score_direction"] == "higher_is_coarser_more_abstract"

    compact_files = [path for path in COMPACT.iterdir() if path.is_file()]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in compact_files)
    lowered = combined.casefold()
    assert "c:\\users\\" not in lowered and "c:/users/" not in lowered
    for path in (COMPACT / "coverage.csv", COMPACT / "failure_reasons.csv", COMPACT / "gate_failure.csv", COMPACT / "scores_long.csv"):
        headers = set(_csv(path)[0])
        assert not headers.intersection({"sentence_original", "sentence_author_edited", "sentence_qwen_edited", "human_label", "participant"})
    for row in _csv(RAW / "accepted_edits.csv"):
        for field in ("sentence_original", "sentence_author_edited", "sentence_qwen_edited"):
            assert row[field] not in combined
