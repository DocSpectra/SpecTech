"""Independent primary-review checks for the blinded QE-R remediation study.

The checks deliberately avoid importing the QE-R implementation under review.
Held-out text is parsed only inside the test process and is never printed.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import median


ROOT = Path(__file__).resolve().parents[1]
SPLIT_CONFIG = ROOT / "configs" / "round2_qwen_edit_remediation_split_v1.json"
SPLIT_MANIFEST = ROOT / "configs" / "round2_qwen_edit_remediation_split_v1.csv"
CANDIDATES = ROOT / "configs" / "round2_qwen_edit_remediation_candidates_v1.json"
V2 = ROOT / "configs" / "round2_qwen_edit_remediation_v2.json"
V2_SCHEMA = ROOT / "schemas" / "round2_qwen_edit_remediation_v2.schema.json"
CHRONOLOGY = ROOT / "configs" / "round2_qwen_edit_remediation_chronology_note.json"
QE_A = ROOT / "configs" / "round2_qwen_edit_source_v1.json"
QE_A_RAW = ROOT / "outputs" / "round2" / "qwen_edit_source" / "generation_attempts.jsonl"
DEV_RAW = ROOT / "outputs" / "round2" / "qwen_edit_remediation" / "development" / "round1"
HELD_RAW = ROOT / "outputs" / "round2" / "qwen_edit_remediation" / "held_out" / "v2"
COMPACT = ROOT / "analysis" / "round2_qwen_edit_remediation"

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


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_id(corpus: str, sent_id: str, direction: str, position: int) -> str:
    return hashlib.sha256(f"{corpus}\0{sent_id}\0{direction}\0{position}".encode()).hexdigest()


def _tokens(value: str) -> list[str]:
    return [match.group(0).casefold() for match in WORD_RE.finditer(unicodedata.normalize("NFKC", value))]


def _markers(value: str) -> Counter[str]:
    found: list[str] = []
    for regex in MARKER_RES:
        found.extend(match.group(0).casefold().rstrip(".,;:)\"]'") for match in regex.finditer(value))
    found.extend(match.group(1).strip().casefold() for match in BACKTICK_RE.finditer(value))
    return Counter(item for item in found if item)


def _vocab_multiset(value: str, vocabulary: list[str]) -> Counter[str]:
    allowed = set(vocabulary)
    return Counter(token for token in _tokens(value) if token in allowed)


def _independent_gate(
    base: dict,
    original: str,
    edited: str,
    direction: str,
    *,
    neutral_anchor: float | None = None,
    de_ratio_max: float | None = None,
    proxy_aligned: bool = False,
    extra_safeguards: bool = False,
) -> tuple[list[str], dict[str, float | int]]:
    """Reproduce the declared gates without calling QE-A/QE-R gate code."""
    universal = base["automated_gates"]["universal"]
    rule = dict(base["automated_gates"]["direction_rules"][direction])
    if neutral_anchor is not None and direction == "irrelevant_rewrite":
        rule["original_content_anchor_recall_min"] = neutral_anchor
    if de_ratio_max is not None and direction == "de_specify":
        rule["edited_to_original_token_ratio_max"] = de_ratio_max

    reasons: list[str] = []
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
    anchor = len(original_content & edited_content) / max(1, len(original_content))
    original_markers = _markers(original)
    edited_markers = _markers(stripped)
    original_marker_count = sum(original_markers.values())
    edited_marker_count = sum(edited_markers.values())
    marker_recall = (
        1.0
        if original_marker_count == 0
        else sum(min(count, edited_markers.get(key, 0)) for key, count in original_markers.items())
        / original_marker_count
    )

    if _vocab_multiset(original, base["automated_gates"]["polarity_tokens"]) != _vocab_multiset(
        stripped, base["automated_gates"]["polarity_tokens"]
    ):
        reasons.append("polarity_changed")
    if _vocab_multiset(original, base["automated_gates"]["modal_tokens"]) != _vocab_multiset(
        stripped, base["automated_gates"]["modal_tokens"]
    ):
        reasons.append("modality_changed")
    if ratio < float(rule["edited_to_original_token_ratio_min"]):
        reasons.append("token_ratio_below_min")
    if ratio > float(rule["edited_to_original_token_ratio_max"]):
        reasons.append("token_ratio_above_max")
    if anchor < float(rule["original_content_anchor_recall_min"]):
        reasons.append("content_anchor_recall_below_min")

    if direction == "add_specific":
        if marker_recall < float(rule["original_concrete_marker_recall_min"]):
            reasons.append("original_concrete_marker_lost")
        if edited_marker_count < original_marker_count:
            reasons.append("concrete_marker_count_decreased")
        if not (len(edited_tokens) >= len(original_tokens) + 2 or edited_marker_count >= original_marker_count + 1):
            reasons.append("insufficient_specificity_addition_proxy")
    elif direction == "de_specify":
        if edited_marker_count > original_marker_count:
            reasons.append("concrete_marker_count_increased")
        if proxy_aligned:
            proxy_pass = (
                len(edited_tokens) <= len(original_tokens) - 1
                or edited_marker_count < original_marker_count
                or (anchor <= 0.9 and len(edited_content) <= len(original_content))
            )
        else:
            proxy_pass = len(edited_tokens) <= len(original_tokens) - 2 or edited_marker_count <= original_marker_count - 1
        if not proxy_pass:
            reasons.append("insufficient_despecification_proxy")
    elif original_markers != edited_markers:
        reasons.append("neutral_concrete_markers_changed")

    if extra_safeguards:
        original_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in original)
        edited_non_latin = any(unicodedata.category(char).startswith("L") and ord(char) > 127 for char in stripped)
        if not original_non_latin and edited_non_latin:
            reasons.append("new_non_latin_script")
        boundaries = re.findall(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)", stripped)
        if len(boundaries) > 1:
            reasons.append("multiple_sentences")

    metrics = {
        "original_token_count": len(original_tokens),
        "edited_token_count": len(edited_tokens),
        "token_ratio": ratio,
        "content_anchor_recall": anchor,
        "original_concrete_marker_count": original_marker_count,
        "edited_concrete_marker_count": edited_marker_count,
        "original_concrete_marker_recall": marker_recall,
    }
    return sorted(set(reasons)), metrics


def _source_cases() -> dict[str, dict[str, str]]:
    base = _json(QE_A)
    cases: dict[str, dict[str, str]] = {}
    for corpus in base["inputs"]["corpus_order"]:
        for position, row in enumerate(_csv(ROOT / base["inputs"]["source_files"][corpus]["path"]), start=1):
            identifier = _case_id(corpus, row["sent_id"], row["edit_type"], position)
            cases[identifier] = {**row, "corpus_id": corpus, "source_position": str(position)}
    return cases


def _commit_time(commit: str) -> datetime:
    value = subprocess.run(
        ["git", "show", "-s", "--format=%cI", commit],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _response_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def test_primary_review_recomputes_source_blind_split_and_signed_chronology() -> None:
    config = _json(SPLIT_CONFIG)
    manifest = _csv(SPLIT_MANIFEST)
    cases = _source_cases()
    seed = int(config["split"]["seed"])
    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for identifier, case in cases.items():
        payload = f"{seed}:{case['corpus_id']}:{case['edit_type']}:{identifier}:{case['source_position']}"
        ranking = hashlib.sha256(payload.encode()).hexdigest()
        cells[(case["corpus_id"], case["edit_type"])].append(
            {
                "case_id": identifier,
                "corpus_id": case["corpus_id"],
                "edit_type": case["edit_type"],
                "source_position": case["source_position"],
                "ranking_key": ranking,
            }
        )
    expected: list[dict[str, str]] = []
    for key in sorted(cells):
        ordered = sorted(cells[key], key=lambda row: (row["ranking_key"], row["case_id"]))
        midpoint = len(ordered) // 2
        expected.extend({**row, "split": "development" if index < midpoint else "held_out"} for index, row in enumerate(ordered))
    expected.sort(key=lambda row: (row["corpus_id"], int(row["source_position"])))
    assert manifest == expected
    assert set(manifest[0]) == {"case_id", "corpus_id", "edit_type", "source_position", "ranking_key", "split"}
    assert _sha(SPLIT_MANIFEST) == "54dd2f4d4fad287a6125cdd303196fce82c08ebf2c7f6d535b7cc17381db44f5"
    assert Counter((row["split"], row["corpus_id"], row["edit_type"]) for row in manifest) == Counter(
        {
            ("development", "ansible_docs", "add_specific"): 4,
            ("held_out", "ansible_docs", "add_specific"): 4,
            ("development", "ansible_docs", "de_specify"): 6,
            ("held_out", "ansible_docs", "de_specify"): 6,
            ("development", "ansible_docs", "irrelevant_rewrite"): 5,
            ("held_out", "ansible_docs", "irrelevant_rewrite"): 5,
            ("development", "github_docs", "add_specific"): 5,
            ("held_out", "github_docs", "add_specific"): 5,
            ("development", "github_docs", "de_specify"): 5,
            ("held_out", "github_docs", "de_specify"): 5,
            ("development", "github_docs", "irrelevant_rewrite"): 5,
            ("held_out", "github_docs", "irrelevant_rewrite"): 5,
        }
    )
    assert _commit_time("bca966334e4685cd354710676167991c0e69c785") < _commit_time(
        "ed136dbb3a27a4c35dfbceaca8b91dc6b4c70d3e"
    )


def test_primary_review_recomputes_development_diagnosis_and_positive_controls() -> None:
    base = _json(QE_A)
    cases = _source_cases()
    development = {row["case_id"] for row in _csv(SPLIT_MANIFEST) if row["split"] == "development"}
    attempts = [row for row in _jsonl(QE_A_RAW) if row["case_id"] in development]
    by_case: dict[str, list[dict]] = defaultdict(list)
    qwen_ratios: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in attempts:
        case = cases[row["case_id"]]
        reasons, metrics = _independent_gate(base, case["sentence_original"], row["edited_sentence"], row["edit_type"])
        assert reasons == row["reason_codes"]
        by_case[row["case_id"]].append(row)
        qwen_ratios[(case["corpus_id"], case["edit_type"])].append(float(metrics["token_ratio"]))
    assert sum(any(row["status"] == "accepted" for row in by_case[identifier]) for identifier in development) == 11
    assert math.isclose(median(qwen_ratios[("ansible_docs", "add_specific")]), 3.0555555556, abs_tol=1e-9)
    assert math.isclose(median(qwen_ratios[("github_docs", "add_specific")]), 2.7142857143, abs_tol=1e-9)

    author_pass: Counter[str] = Counter()
    author_fail: Counter[tuple[str, str]] = Counter()
    neutral_anchors: list[float] = []
    author_add_ratios: dict[str, list[float]] = defaultdict(list)
    for identifier in development:
        case = cases[identifier]
        reasons, metrics = _independent_gate(base, case["sentence_original"], case["sentence_edited"], case["edit_type"])
        if not reasons:
            author_pass[case["edit_type"]] += 1
        for reason in reasons:
            author_fail[(case["edit_type"], reason)] += 1
        if case["edit_type"] == "irrelevant_rewrite":
            neutral_anchors.append(float(metrics["content_anchor_recall"]))
        if case["edit_type"] == "add_specific":
            author_add_ratios[case["corpus_id"]].append(float(metrics["token_ratio"]))
    assert author_pass["add_specific"] == 9
    assert author_fail[("irrelevant_rewrite", "content_anchor_recall_below_min")] == 2
    assert author_fail[("irrelevant_rewrite", "token_ratio_below_min")] == 1
    assert author_fail[("de_specify", "insufficient_despecification_proxy")] == 3
    assert math.isclose(min(neutral_anchors), 0.6363636364, abs_tol=1e-9)
    assert math.isclose(median(author_add_ratios["ansible_docs"]), 1.2828282828, abs_tol=1e-9)
    assert math.isclose(median(author_add_ratios["github_docs"]), 1.2727272727, abs_tol=1e-9)


def test_primary_review_recomputes_candidate_budget_gates_and_selection() -> None:
    candidate = _json(CANDIDATES)
    base = _json(QE_A)
    cases = _source_cases()
    development = {row["case_id"] for row in _csv(SPLIT_MANIFEST) if row["split"] == "development"}
    expected_totals = {"c1_prompt_only": 17, "c2_proxy_aligned": 25, "c3_micro_edit": 27}
    observed_selection: dict[str, tuple[float, float]] = {}
    total_calls = 0
    first_response: datetime | None = None
    last_response: datetime | None = None
    for candidate_id in candidate["candidate_order"]:
        profile = candidate["candidate_profiles"][candidate_id]
        proxy = profile["gate_variant"] == "proxy_aligned"
        neutral_anchor = 0.6 if proxy else 0.7
        de_ratio = 1.1 if proxy else 1.05
        rows = _jsonl(DEV_RAW / candidate_id / "attempts.jsonl")
        total_calls += len(rows)
        assert {row["case_id"] for row in rows} == development
        by_case: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            case = cases[row["case_id"]]
            reasons, _ = _independent_gate(
                base,
                case["sentence_original"],
                row["edited_sentence"],
                row["edit_type"],
                neutral_anchor=neutral_anchor,
                de_ratio_max=de_ratio,
                proxy_aligned=proxy,
                extra_safeguards=True,
            )
            assert reasons == row["reason_codes"]
            by_case[row["case_id"]].append(row)
            created = _response_time(row["created_at"])
            first_response = created if first_response is None else min(first_response, created)
            last_response = created if last_response is None else max(last_response, created)
        accepted = {identifier for identifier, items in by_case.items() if any(row["status"] == "accepted" for row in items)}
        assert len(accepted) == expected_totals[candidate_id]
        cells = Counter((cases[identifier]["corpus_id"], cases[identifier]["edit_type"]) for identifier in accepted)
        planned = Counter((cases[identifier]["corpus_id"], cases[identifier]["edit_type"]) for identifier in development)
        rates = [cells[key] / planned[key] for key in planned]
        observed_selection[candidate_id] = (min(rates), len(accepted) / 30)
    assert total_calls == 137 <= 360
    assert observed_selection == {
        "c1_prompt_only": (0.2, 17 / 30),
        "c2_proxy_aligned": (0.4, 25 / 30),
        "c3_micro_edit": (4 / 6, 27 / 30),
    }
    assert max(observed_selection, key=lambda key: observed_selection[key]) == "c3_micro_edit"
    assert not (DEV_RAW.parent / "round2").exists()
    assert first_response is not None and last_response is not None
    assert _commit_time("be1bc3a64cfbf858a9c8296b922d663d5cac2edb") < first_response
    note = _json(CHRONOLOGY)["candidate_chronology"]
    assert _response_time(note["first_saved_model_response_utc"]) == first_response
    assert _response_time(note["last_saved_model_response_utc"]) == last_response


def test_primary_review_confirms_v2_is_exact_selected_protocol_copy() -> None:
    candidate = _json(CANDIDATES)
    protocol = _json(V2)
    profile = candidate["candidate_profiles"]["c3_micro_edit"]
    assert protocol["prompts"]["system"] == candidate["system_prompt"]
    assert protocol["prompts"]["user_templates"] == candidate["prompt_variants"][profile["prompt_variant"]]
    for key in ("model", "model_list_id", "backing_blob_sha256", "api_url", "thinking", "stream", "keep_alive", "fresh_conversation_per_attempt", "options", "response_schema"):
        assert protocol["runtime"][key] == candidate["runtime"][key]
    gate = candidate["gate_variants"][profile["gate_variant"]]
    assert protocol["automated_gates"]["direction_rules"]["irrelevant_rewrite"]["original_content_anchor_recall_min"] == gate["neutral_content_anchor_recall_min"]
    assert protocol["automated_gates"]["direction_rules"]["de_specify"]["edited_to_original_token_ratio_max"] == gate["de_token_ratio_max"]
    assert protocol["runtime"]["master_seed"] == 2026081099
    assert protocol["runtime"]["maximum_attempts_per_case"] == 3
    assert _sha(V2) == "0d2b715ea015273bbf4be90f2c657a1e9a9f4624ab90c78de840bd3a7971761b"
    assert _sha(V2_SCHEMA) == "738c6b034edba187d09bdce18269b39f2354d1837420b741a82eee8f98e63c73"


def test_primary_review_recomputes_held_out_without_printing_text() -> None:
    base = _json(QE_A)
    cases = _source_cases()
    held = {row["case_id"] for row in _csv(SPLIT_MANIFEST) if row["split"] == "held_out"}
    rows = _jsonl(HELD_RAW / "attempts.jsonl")
    assert len(rows) == 50 and {row["case_id"] for row in rows} == held
    by_case: dict[str, list[dict]] = defaultdict(list)
    failures: Counter[tuple[str, str, str]] = Counter()
    for row in rows:
        case = cases[row["case_id"]]
        reasons, metrics = _independent_gate(
            base,
            case["sentence_original"],
            row["edited_sentence"],
            row["edit_type"],
            neutral_anchor=0.6,
            de_ratio_max=1.1,
            proxy_aligned=True,
            extra_safeguards=True,
        )
        assert reasons == row["reason_codes"]
        assert row["edited_sha256"] == hashlib.sha256(row["edited_sentence"].encode()).hexdigest()
        for name in ("token_ratio", "content_anchor_recall", "original_concrete_marker_count", "edited_concrete_marker_count"):
            assert math.isclose(float(row[name]), float(metrics[name]), abs_tol=1e-12)
        by_case[row["case_id"]].append(row)
        for reason in reasons:
            failures[(row["corpus_id"], row["edit_type"], reason)] += 1
    accepted: set[str] = set()
    for identifier, attempts in by_case.items():
        assert [row["attempt_index"] for row in attempts] == list(range(1, len(attempts) + 1))
        passing = [row for row in attempts if row["status"] == "accepted"]
        assert len(passing) <= 1
        if passing:
            assert passing[0] is attempts[-1]
            accepted.add(identifier)
        else:
            assert len(attempts) == 3
    assert len(accepted) == 20
    coverage = Counter((cases[identifier]["corpus_id"], cases[identifier]["edit_type"]) for identifier in accepted)
    assert coverage == Counter(
        {
            ("ansible_docs", "add_specific"): 2,
            ("ansible_docs", "de_specify"): 4,
            ("ansible_docs", "irrelevant_rewrite"): 2,
            ("github_docs", "add_specific"): 5,
            ("github_docs", "de_specify"): 5,
            ("github_docs", "irrelevant_rewrite"): 2,
        }
    )
    assert failures == Counter(
        {
            ("ansible_docs", "add_specific", "polarity_changed"): 6,
            ("ansible_docs", "de_specify", "insufficient_despecification_proxy"): 6,
            ("ansible_docs", "de_specify", "unchanged_normalized"): 6,
            ("ansible_docs", "irrelevant_rewrite", "content_anchor_recall_below_min"): 3,
            ("ansible_docs", "irrelevant_rewrite", "modality_changed"): 3,
            ("ansible_docs", "irrelevant_rewrite", "unchanged_normalized"): 3,
            ("github_docs", "irrelevant_rewrite", "content_anchor_recall_below_min"): 3,
            ("github_docs", "irrelevant_rewrite", "modality_changed"): 3,
            ("github_docs", "irrelevant_rewrite", "neutral_concrete_markers_changed"): 6,
            ("github_docs", "irrelevant_rewrite", "token_ratio_below_min"): 3,
        }
    )
    assert _sha(HELD_RAW / "attempts.jsonl") == "a8f021c8d2aed0ff99f8577b899d2518e0b2a6f91c13183578303bfc0c6dce3f"
    assert _sha(HELD_RAW / "accepted.csv") == "89cc504637b160e33a57fe4522fc13b7b2054833c7af8e1e6fcbdc426d180c1f"


def test_primary_review_confirms_held_out_chronology_privacy_and_no_scoring() -> None:
    rows = _jsonl(HELD_RAW / "attempts.jsonl")
    first_response = min(_response_time(row["created_at"]) for row in rows)
    last_response = max(_response_time(row["created_at"]) for row in rows)
    assert _commit_time("746a3ada926df1fc87a8dd443250caa54092c37f") < first_response
    note = _json(CHRONOLOGY)["held_out_chronology"]
    assert _response_time(note["first_saved_model_response_utc"]) == first_response
    assert _response_time(note["last_saved_model_response_utc"]) == last_response

    compact_files = [path for path in (COMPACT / "held_out_v2").iterdir() if path.is_file()]
    combined = "\n".join(path.read_text(encoding="utf-8") for path in compact_files)
    lowered = combined.casefold()
    assert "c:\\users\\" not in lowered and "c:/users/" not in lowered
    assert "score_original" not in lowered and "score_edited" not in lowered
    assert not set(_csv(COMPACT / "held_out_v2" / "attempt_accounting.csv")[0]).intersection(
        {"sentence_original", "sentence_author_edited", "sentence_qwen_edited", "edited_sentence"}
    )
    raw_texts = [row["edited_sentence"] for row in rows if len(row["edited_sentence"]) >= 20]
    assert all(text not in combined for text in raw_texts)
    metadata = _json(COMPACT / "held_out_v2" / "run_metadata.json")
    assert metadata["scoring_performed"] is False
    assert metadata["held_out_text_printed_or_interactively_inspected"] is False
    assert metadata["disposition"] == "fail_recommend_omit_qwen_edit_arm"
