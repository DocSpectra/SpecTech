"""Deterministic analysis of the frozen human-first reranking review packet."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence
from xml.etree import ElementTree as ET

import numpy as np

from src.analysis.pilot_model_human import sha256_file


CONFIG = Path("configs/round2_human_first_analysis_v1.json")
FREEZE = Path("configs/round2_human_first_analysis_freeze_record.json")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
POLICY_ORDER = [
    "unguided_slot01",
    "speciteller_frozen_round1",
    "ko_run01",
    "ko_run02",
    "ko_run03",
    "granuscore_direction_aligned",
    "ko_three_run_arithmetic_mean_secondary",
    "rank_consensus_secondary",
]
METRIC_ORDER = [
    "speciteller_frozen_round1",
    "ko_run01",
    "ko_run02",
    "ko_run03",
    "granuscore_direction_aligned",
    "ko_three_run_arithmetic_mean_secondary",
]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(fields or rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _file_sha(path: Path) -> str:
    return sha256_file(path)


def _stable_seed(master: int, *parts: str) -> int:
    payload = ":".join([str(master), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def load_protocol(config_path: Path = CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_human_first_analysis_v1":
        raise ValueError("unexpected analysis protocol")
    if record.get("outcome_blind_method_freeze") is not True:
        raise ValueError("analysis protocol is not outcome blind")
    if _file_sha(config_path) != freeze["config_sha256"]:
        raise ValueError("analysis config changed after freeze")
    bindings = {
        "schema_sha256": Path("schemas/round2_human_first_analysis_v1.schema.json"),
        "spec_sha256": Path("specs/round2_human_first_analysis.md"),
        "freeze_test_sha256": Path("tests/test_human_first_analysis_freeze.py"),
    }
    for key, path in bindings.items():
        if _file_sha(path) != freeze[key]:
            raise ValueError(f"analysis freeze binding changed: {path}")
    if not freeze["pre_unblinding_validation_passed"]:
        raise ValueError("pre-unblinding validation not frozen")
    return record, freeze


def _xlsx_cells(path: Path, sheet_entry: str) -> dict[str, str]:
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in si.findall(".//s:t", NS)) for si in shared_root.findall("s:si", NS)]
        root = ET.fromstring(archive.read(sheet_entry))
    cells: dict[str, str] = {}
    for cell in root.findall(".//s:sheetData/s:row/s:c", NS):
        ref = cell.attrib["r"]
        cell_type = cell.attrib.get("t", "")
        value_node = cell.find("s:v", NS)
        if cell_type == "s":
            value = "" if value_node is None else shared[int(value_node.text or "0")]
        elif cell_type == "inlineStr":
            value = "".join(node.text or "" for node in cell.findall(".//s:t", NS))
        else:
            value = "" if value_node is None else value_node.text or ""
        cells[ref] = value
    return cells


def load_and_validate_reviews(record: dict[str, Any]) -> list[dict[str, str]]:
    released = Path(record["inputs"]["released_workbook"]["path"])
    completed = Path(record["inputs"]["completed_workbook"]["path"])
    if _file_sha(released) != record["inputs"]["released_workbook"]["sha256"]:
        raise ValueError("released workbook hash mismatch")
    if _file_sha(completed) != record["inputs"]["completed_workbook"]["sha256"]:
        raise ValueError("completed workbook hash mismatch")
    released_review = _xlsx_cells(released, "xl/worksheets/sheet1.xml")
    completed_review = _xlsx_cells(completed, "xl/worksheets/sheet1.xml")
    released_instructions = _xlsx_cells(released, "xl/worksheets/sheet2.xml")
    completed_instructions = _xlsx_cells(completed, "xl/worksheets/sheet2.xml")
    if released_instructions != completed_instructions:
        raise ValueError("instructions content changed")
    headers = [completed_review.get(f"{column}1", "") for column in "ABCD"]
    if headers != ["Review ID", "A", "B", "Score"]:
        raise ValueError("review headers changed")
    allowed = set(record["inputs"]["completed_workbook"]["allowed_scores"])
    rows: list[dict[str, str]] = []
    for position, sheet_row in enumerate(range(2, 182), 1):
        immutable = [completed_review.get(f"{column}{sheet_row}", "") for column in "ABC"]
        expected = [released_review.get(f"{column}{sheet_row}", "") for column in "ABC"]
        if immutable != expected:
            raise ValueError(f"released review row changed at position {position}")
        score = completed_review.get(f"D{sheet_row}", "")
        if score not in allowed:
            raise ValueError(f"invalid score at review position {position}")
        rows.append({"review_position": str(position), "review_id": immutable[0], "A": immutable[1], "B": immutable[2], "score": score})
    if len(rows) != 180 or len({row["review_id"] for row in rows}) != 180:
        raise ValueError("review row count or ID uniqueness mismatch")
    return rows


def human_delta(score: str, candidate_side: str) -> int | None:
    if score == "X":
        return None
    a_side = {"1": 2, "2": 1, "3": 0, "4": -1, "5": -2}
    if candidate_side not in {"A", "B"}:
        raise ValueError(f"invalid candidate side: {candidate_side}")
    value = a_side[score]
    return value if candidate_side == "A" else -value


def direction_utility(delta: int | None, edit_type: str) -> int | None:
    if delta is None:
        return None
    if edit_type == "add_specific":
        return delta
    if edit_type == "de_specify":
        return -delta
    if edit_type == "irrelevant_rewrite":
        return -abs(delta)
    raise ValueError(f"unexpected edit type: {edit_type}")


def direction_valid(delta: int | None, edit_type: str) -> bool:
    if delta is None:
        return False
    if edit_type == "add_specific":
        return delta > 0
    if edit_type == "de_specify":
        return delta < 0
    if edit_type == "irrelevant_rewrite":
        return delta == 0
    raise ValueError(f"unexpected edit type: {edit_type}")


def _text_sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_unblinded_rows(record: dict[str, Any], reviews: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    key_rows = _read_csv(Path(record["inputs"]["private_review_key"]))
    metric_rows = _read_csv(Path(record["inputs"]["candidate_metrics"]))
    proxy_rows = _read_csv(Path(record["inputs"]["automatic_proxy_audit"]))
    policy_rows = _read_csv(Path(record["inputs"]["policy_selections"]))
    if not (len(key_rows) == len(metric_rows) == len(proxy_rows) == 180 and len(policy_rows) == 480):
        raise ValueError("hidden artifact row counts changed")
    key_by_review = {row["review_id"]: row for row in key_rows}
    metrics = {row["candidate_id"]: row for row in metric_rows}
    proxies = {row["candidate_id"]: row for row in proxy_rows}
    if len(key_by_review) != 180 or len(metrics) != 180 or len(proxies) != 180:
        raise ValueError("hidden artifact IDs are not unique")
    joined: list[dict[str, Any]] = []
    for review in reviews:
        key = key_by_review.get(review["review_id"])
        if key is None:
            raise ValueError("review/key bijection failed")
        candidate_id = key["candidate_id"]
        metric = metrics.get(candidate_id)
        proxy = proxies.get(candidate_id)
        if metric is None or proxy is None:
            raise ValueError("candidate metric/proxy join failed")
        if metric["case_id"] != key["case_id"] or metric["candidate_slot"] != key["candidate_slot"]:
            raise ValueError("candidate metric identity mismatch")
        if proxy["proxy_pass"] != key["proxy_pass"]:
            raise ValueError("proxy decision mismatch")
        if _text_sha(review["A"]) != key["sentence_a_sha256"] or _text_sha(review["B"]) != key["sentence_b_sha256"]:
            raise ValueError("review side hash reversibility failed")
        delta = human_delta(review["score"], key["candidate_side"])
        utility = direction_utility(delta, key["edit_type"])
        row: dict[str, Any] = {
            "review_position": int(review["review_position"]),
            "review_id": review["review_id"],
            "case_id": key["case_id"],
            "candidate_id": candidate_id,
            "corpus_id": key["corpus_id"],
            "edit_type": key["edit_type"],
            "candidate_slot": int(key["candidate_slot"]),
            "candidate_side": key["candidate_side"],
            "score": review["score"],
            "human_delta": delta,
            "direction_utility": utility,
            "direction_valid": direction_valid(delta, key["edit_type"]),
            "proxy_pass": proxy["proxy_pass"].casefold() == "true",
            "proxy_reason_codes": proxy["reason_codes"],
        }
        for model in METRIC_ORDER:
            row[model] = float(metric[model])
            key_value = float(key[f"metric_{model}"])
            if not math.isclose(row[model], key_value, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f"metric/key mismatch for {model}")
        joined.append(row)
    if len({row["candidate_id"] for row in joined}) != 180:
        raise ValueError("candidate join is not bijective")
    by_case = Counter(row["case_id"] for row in joined)
    if len(by_case) != 60 or set(by_case.values()) != {3}:
        raise ValueError("expected three candidates for each of 60 cases")
    policy_counts = Counter(row["policy"] for row in policy_rows)
    if set(policy_counts) != set(POLICY_ORDER) or set(policy_counts.values()) != {60}:
        raise ValueError("policy selection coverage changed")
    if {row["selected_candidate_id"] for row in policy_rows} - {row["candidate_id"] for row in joined}:
        raise ValueError("policy selected an unknown candidate")
    return joined, policy_rows


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def spearman(values_x: Sequence[float], values_y: Sequence[float]) -> float:
    x = np.asarray(values_x, dtype=float)
    y = np.asarray(values_y, dtype=float)
    if x.size < 2 or y.size != x.size:
        return math.nan
    rx = average_ranks(x)
    ry = average_ranks(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _ci(values: Iterable[float]) -> tuple[float | None, float | None, int]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return None, None, 0
    low, high = np.quantile(array, [0.025, 0.975])
    return float(low), float(high), int(array.size)


def _display(value: float | int | None) -> float | int | str:
    if value is None or isinstance(value, float) and not math.isfinite(value):
        return ""
    return value


def _case_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    return groups


def _case_draws(rows: list[dict[str, Any]], replicates: int, seed: int, stratified: bool) -> tuple[list[str], np.ndarray]:
    groups = _case_groups(rows)
    case_ids = sorted(groups)
    if not case_ids:
        return [], np.empty((replicates, 0), dtype=int)
    rng = np.random.default_rng(seed)
    if stratified:
        strata: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, case_id in enumerate(case_ids):
            first = groups[case_id][0]
            strata[(first["corpus_id"], first["edit_type"])].append(index)
        blocks = []
        for key in sorted(strata):
            indices = np.asarray(strata[key], dtype=int)
            blocks.append(indices[rng.integers(0, len(indices), size=(replicates, len(indices)))])
        draws = np.concatenate(blocks, axis=1)
    else:
        draws = rng.integers(0, len(case_ids), size=(replicates, len(case_ids)))
    return case_ids, draws


def _strata(rows: list[dict[str, Any]]) -> list[tuple[str, str, list[dict[str, Any]]]]:
    result = [("overall", "all", rows)]
    for corpus in sorted({row["corpus_id"] for row in rows}):
        result.append(("corpus_id", corpus, [row for row in rows if row["corpus_id"] == corpus]))
    for direction in sorted({row["edit_type"] for row in rows}):
        result.append(("edit_type", direction, [row for row in rows if row["edit_type"] == direction]))
    return result


def rating_order_distribution(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sets = [
        ("all_180", lambda p: 1 <= p <= 180),
        ("first_60", lambda p: 1 <= p <= 60),
        ("middle_60", lambda p: 61 <= p <= 120),
        ("final_60", lambda p: 121 <= p <= 180),
    ]
    output = []
    for name, predicate in sets:
        selected = [row for row in rows if predicate(row["review_position"])]
        counts = Counter(row["score"] for row in selected)
        for score in ("1", "2", "3", "4", "5", "X"):
            output.append({
                "distribution_view": "review_order",
                "analysis_set": name,
                "stratum_type": "review_order_segment",
                "stratum_value": name,
                "score": score,
                "count": counts[score],
                "rate": counts[score] / len(selected),
                "rows": len(selected),
            })
    for stratum_type, stratum_value, selected in _strata(rows):
        if stratum_type == "overall":
            continue
        counts = Counter(row["score"] for row in selected)
        for score in ("1", "2", "3", "4", "5", "X"):
            output.append({
                "distribution_view": "unblinded_condition",
                "analysis_set": "all_180",
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "score": score,
                "count": counts[score],
                "rate": counts[score] / len(selected),
                "rows": len(selected),
            })
    return output


def _policy_selected(rows: list[dict[str, Any]], policy_rows: list[dict[str, str]], policy: str, cutoff: int) -> list[dict[str, Any]]:
    by_candidate = {row["candidate_id"]: row for row in rows}
    selected = []
    for mapping in policy_rows:
        if mapping["policy"] != policy:
            continue
        row = by_candidate[mapping["selected_candidate_id"]]
        if row["review_position"] > cutoff:
            selected.append(row)
    return selected


def _bootstrap_policy_summary(rows: list[dict[str, Any]], replicates: int, seed: int, stratified: bool) -> tuple[tuple[float | None, float | None, int], tuple[float | None, float | None, int]]:
    case_ids, draws = _case_draws(rows, replicates, seed, stratified)
    by_case = {case_id: _case_groups(rows)[case_id][0] for case_id in case_ids}
    success = np.asarray([float(by_case[case_id]["direction_valid"]) for case_id in case_ids])
    utility = np.asarray([math.nan if by_case[case_id]["direction_utility"] is None else float(by_case[case_id]["direction_utility"]) for case_id in case_ids])
    success_reps = success[draws].mean(axis=1) if case_ids else np.full(replicates, math.nan)
    with np.errstate(invalid="ignore"):
        utility_reps = np.nanmean(utility[draws], axis=1) if case_ids else np.full(replicates, math.nan)
    return _ci(success_reps), _ci(utility_reps)


def build_policy_outputs(record: dict[str, Any], rows: list[dict[str, Any]], policy_rows: list[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    sets = [("all_180", 0), ("exclude_first_20", 20), ("exclude_first_30", 30)]
    summaries: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    selected_cache: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for set_name, cutoff in sets:
        for policy in POLICY_ORDER:
            selected_cache[(set_name, policy)] = _policy_selected(rows, policy_rows, policy, cutoff)
            for stratum_type, stratum_value, subset in _strata(selected_cache[(set_name, policy)]):
                if not subset:
                    continue
                success = sum(bool(row["direction_valid"]) for row in subset)
                x_count = sum(row["score"] == "X" for row in subset)
                utilities = [float(row["direction_utility"]) for row in subset if row["direction_utility"] is not None]
                seed = _stable_seed(master, "policy-summary", set_name, policy, stratum_type, stratum_value)
                success_ci, utility_ci = _bootstrap_policy_summary(subset, reps, seed, stratum_type == "overall")
                role = "reference" if policy == "unguided_slot01" else "secondary" if policy in record["policies"]["secondary"] else "primary"
                summaries.append({
                    "analysis_set": set_name,
                    "stratum_type": stratum_type,
                    "stratum_value": stratum_value,
                    "policy": policy,
                    "policy_role": role,
                    "case_count": len(subset),
                    "direction_valid_count": success,
                    "direction_valid_rate": success / len(subset),
                    "direction_valid_ci_low": _display(success_ci[0]),
                    "direction_valid_ci_high": _display(success_ci[1]),
                    "x_count": x_count,
                    "x_rate": x_count / len(subset),
                    "comparable_count": len(utilities),
                    "mean_direction_utility": _display(float(np.mean(utilities)) if utilities else None),
                    "mean_utility_ci_low": _display(utility_ci[0]),
                    "mean_utility_ci_high": _display(utility_ci[1]),
                    "bootstrap_valid_replicates": success_ci[2],
                })
        reference = selected_cache[(set_name, "unguided_slot01")]
        reference_by_case = {row["case_id"]: row for row in reference}
        for policy in POLICY_ORDER[1:]:
            guided_by_case = {row["case_id"]: row for row in selected_cache[(set_name, policy)]}
            common_ids = set(reference_by_case) & set(guided_by_case)
            paired_all = [{"reference": reference_by_case[cid], "guided": guided_by_case[cid]} for cid in sorted(common_ids)]
            pseudo_rows = [pair["guided"] for pair in paired_all]
            for stratum_type, stratum_value, subset_guided in _strata(pseudo_rows):
                subset_ids = {row["case_id"] for row in subset_guided}
                pairs = [pair for pair in paired_all if pair["guided"]["case_id"] in subset_ids]
                if not pairs:
                    continue
                case_rows = [pair["guided"] for pair in pairs]
                case_ids, draws = _case_draws(case_rows, reps, _stable_seed(master, "policy-contrast", set_name, policy, stratum_type, stratum_value), stratum_type == "overall")
                pair_by_case = {pair["guided"]["case_id"]: pair for pair in pairs}
                diffs = np.asarray([float(pair_by_case[cid]["guided"]["direction_valid"]) - float(pair_by_case[cid]["reference"]["direction_valid"]) for cid in case_ids])
                boot = diffs[draws].mean(axis=1) if case_ids else np.full(reps, math.nan)
                low, high, valid = _ci(boot)
                wins = losses = ties = comparable_pairs = 0
                for pair in pairs:
                    gu = pair["guided"]["direction_utility"]
                    ru = pair["reference"]["direction_utility"]
                    if gu is None or ru is None:
                        continue
                    comparable_pairs += 1
                    if gu > ru:
                        wins += 1
                    elif gu < ru:
                        losses += 1
                    else:
                        ties += 1
                guided_success = sum(pair["guided"]["direction_valid"] for pair in pairs)
                reference_success = sum(pair["reference"]["direction_valid"] for pair in pairs)
                contrasts.append({
                    "analysis_set": set_name,
                    "stratum_type": stratum_type,
                    "stratum_value": stratum_value,
                    "policy": policy,
                    "reference_policy": "unguided_slot01",
                    "paired_case_count": len(pairs),
                    "guided_valid_count": guided_success,
                    "reference_valid_count": reference_success,
                    "valid_rate_difference": (guided_success - reference_success) / len(pairs),
                    "difference_ci_low": _display(low),
                    "difference_ci_high": _display(high),
                    "ordinal_comparable_pairs": comparable_pairs,
                    "ordinal_wins": wins,
                    "ordinal_losses": losses,
                    "ordinal_ties": ties,
                    "bootstrap_valid_replicates": valid,
                })
    return summaries, contrasts


def build_predictor_agreement(record: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    comparable = [row for row in rows if row["human_delta"] is not None]
    output: list[dict[str, Any]] = []
    for stratum_type, stratum_value, subset in _strata(comparable):
        groups = _case_groups(subset)
        case_rows = [values[0] for _, values in sorted(groups.items())]
        for model in METRIC_ORDER:
            point = spearman([row[model] for row in subset], [row["human_delta"] for row in subset])
            case_ids, draws = _case_draws(case_rows, reps, _stable_seed(master, "predictor", model, stratum_type, stratum_value), stratum_type == "overall")
            reps_out = []
            for draw in draws:
                sampled = [row for index in draw for row in groups[case_ids[int(index)]]]
                reps_out.append(spearman([row[model] for row in sampled], [row["human_delta"] for row in sampled]))
            low, high, valid = _ci(reps_out)
            output.append({
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "model": model,
                "model_role": "secondary" if model == "ko_three_run_arithmetic_mean_secondary" else "primary",
                "candidate_count": len(subset),
                "case_count": len(groups),
                "spearman_rho": _display(point),
                "ci_low": _display(low),
                "ci_high": _display(high),
                "bootstrap_valid_replicates": valid,
            })
    return output


def _confusion_counts(rows: list[dict[str, Any]]) -> np.ndarray:
    counts = np.zeros(6, dtype=float)
    for row in rows:
        proxy = bool(row["proxy_pass"])
        human = bool(row["direction_valid"])
        if proxy and human:
            counts[0] += 1
        elif proxy and not human:
            counts[1] += 1
        elif not proxy and human:
            counts[2] += 1
        else:
            counts[3] += 1
        counts[4] += row["score"] == "X"
        counts[5] += 1
    return counts


def _rates_from_confusion(counts: np.ndarray) -> tuple[float, float]:
    true_accept, false_accept, false_reject, true_reject = counts[:4]
    far_denom = false_accept + true_reject
    frr_denom = true_accept + false_reject
    far = false_accept / far_denom if far_denom else math.nan
    frr = false_reject / frr_denom if frr_denom else math.nan
    return float(far), float(frr)


def build_proxy_confusion(record: dict[str, Any], rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    output = []
    for set_name, cutoff in (("all_180", 0), ("exclude_first_20", 20), ("exclude_first_30", 30)):
        selected = [row for row in rows if row["review_position"] > cutoff]
        for stratum_type, stratum_value, subset in _strata(selected):
            counts = _confusion_counts(subset)
            far, frr = _rates_from_confusion(counts)
            groups = _case_groups(subset)
            case_rows = [values[0] for _, values in sorted(groups.items())]
            case_ids, draws = _case_draws(case_rows, reps, _stable_seed(master, "proxy", set_name, stratum_type, stratum_value), stratum_type == "overall")
            case_counts = {cid: _confusion_counts(groups[cid]) for cid in case_ids}
            far_reps = []
            frr_reps = []
            for draw in draws:
                total = np.sum([case_counts[case_ids[int(index)]] for index in draw], axis=0)
                far_value, frr_value = _rates_from_confusion(total)
                far_reps.append(far_value)
                frr_reps.append(frr_value)
            far_low, far_high, far_valid = _ci(far_reps)
            frr_low, frr_high, frr_valid = _ci(frr_reps)
            output.append({
                "analysis_set": set_name,
                "stratum_type": stratum_type,
                "stratum_value": stratum_value,
                "candidate_count": int(counts[5]),
                "case_count": len(groups),
                "true_accept": int(counts[0]),
                "false_accept": int(counts[1]),
                "false_reject": int(counts[2]),
                "true_reject": int(counts[3]),
                "x_count": int(counts[4]),
                "false_accept_rate": _display(far),
                "false_accept_ci_low": _display(far_low),
                "false_accept_ci_high": _display(far_high),
                "false_reject_rate": _display(frr),
                "false_reject_ci_low": _display(frr_low),
                "false_reject_ci_high": _display(frr_high),
                "far_bootstrap_valid_replicates": far_valid,
                "frr_bootstrap_valid_replicates": frr_valid,
            })
    return output


def _privacy_scan(compact: Path, forbidden_values: list[str]) -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in compact.iterdir() if path.is_file())
    lowered = text.casefold()
    for token in ("sentence_original", "sentence_candidate", "review_id", "candidate_id", "case_id", "source_sent_id"):
        if token in lowered:
            raise ValueError(f"privacy-forbidden field in compact evidence: {token}")
    for value in forbidden_values:
        if value and value in text:
            raise ValueError("row identifier leaked into compact evidence")


def _readme(rows: list[dict[str, Any]], policy_summary: list[dict[str, Any]], policy_contrasts: list[dict[str, Any]], proxy: list[dict[str, Any]]) -> str:
    score_counts = Counter(row["score"] for row in rows)
    score_three_by_direction = Counter(row["edit_type"] for row in rows if row["score"] == "3")
    primary = [row for row in policy_summary if row["analysis_set"] == "all_180" and row["stratum_type"] == "overall"]
    contrast = [row for row in policy_contrasts if row["analysis_set"] == "all_180" and row["stratum_type"] == "overall"]
    proxy_all = next(row for row in proxy if row["analysis_set"] == "all_180" and row["stratum_type"] == "overall")
    lines = [
        "# Human-first reranking analysis",
        "",
        "The frozen 180-row packet was unblinded only after the drift-aware analysis protocol was committed. The judgments cover three candidates for each of 60 source cases and come from one evaluator.",
        "",
        "A rating of 3 means equal specificity under the evaluator's operational rule and may include a small number of subtle semantic shifts; X was reserved for cases without a usable meaning comparison. Scores were not recoded. Accordingly, direction-validity below is specificity agreement, not full semantic edit validity, accuracy, or truth.",
        "",
        "## Aggregate rating counts",
        "",
        f"- 1: {score_counts['1']}; 2: {score_counts['2']}; 3: {score_counts['3']}; 4: {score_counts['4']}; 5: {score_counts['5']}; X: {score_counts['X']}.",
        f"- Score 3 occurred in {score_three_by_direction['add_specific']} add-specific, {score_three_by_direction['de_specify']} de-specific, and {score_three_by_direction['irrelevant_rewrite']} neutral rows. Under the frozen rule, only the neutral score-3 rows count as direction-valid; none establish semantic preservation.",
        "",
        "## Overall frozen-policy results",
        "",
    ]
    for row in primary:
        lines.append(f"- {row['policy']}: {row['direction_valid_count']}/{row['case_count']} direction-valid ({100*float(row['direction_valid_rate']):.1f}%), X={row['x_count']}.")
    lines.extend(["", "Guided-minus-unguided paired differences:", ""])
    for row in contrast:
        lines.append(f"- {row['policy']}: {100*float(row['valid_rate_difference']):+.1f} percentage points over {row['paired_case_count']} paired source cases (95% bootstrap CI {100*float(row['difference_ci_low']):+.1f} to {100*float(row['difference_ci_high']):+.1f}).")
    lines.extend([
        "",
        "## Automatic proxy diagnostic",
        "",
        f"Overall confusion counts were true accept={proxy_all['true_accept']}, false accept={proxy_all['false_accept']}, false reject={proxy_all['false_reject']}, and true reject={proxy_all['true_reject']}. The false-accept rate was {100*float(proxy_all['false_accept_rate']):.1f}% and the false-reject rate was {100*float(proxy_all['false_reject_rate']):.1f}%.",
        "",
        "See the CSV artifacts for corpus, direction, order-drift, all three Ko-run, secondary Ko-mean/consensus, predictor-association, uncertainty, and denominator details. All results are descriptive and report unfavorable outcomes. No manuscript integration is authorized by this analysis.",
        "",
    ])
    return "\n".join(lines)


def run_analysis(config_path: Path = CONFIG) -> dict[str, Any]:
    record, freeze = load_protocol(config_path)
    reviews = load_and_validate_reviews(record)
    rows, policy_rows = build_unblinded_rows(record, reviews)
    rating_rows = rating_order_distribution(rows)
    policy_summary, policy_contrasts = build_policy_outputs(record, rows, policy_rows)
    predictor = build_predictor_agreement(record, rows)
    proxy = build_proxy_confusion(record, rows)
    raw = Path(record["outputs"]["raw_directory"])
    compact = Path(record["outputs"]["compact_directory"])
    raw.mkdir(parents=True, exist_ok=True)
    compact.mkdir(parents=True, exist_ok=True)
    raw_fields = [
        "review_position", "review_id", "case_id", "candidate_id", "corpus_id", "edit_type", "candidate_slot", "candidate_side", "score",
        "human_delta", "direction_utility", "direction_valid", "proxy_pass", "proxy_reason_codes", *METRIC_ORDER,
    ]
    _write_csv(raw / "unblinded_candidate_rows.csv", rows, raw_fields)
    _write_csv(compact / "rating_order_distribution.csv", rating_rows)
    _write_csv(compact / "policy_summary.csv", policy_summary)
    _write_csv(compact / "policy_contrasts.csv", policy_contrasts)
    _write_csv(compact / "predictor_agreement.csv", predictor)
    _write_csv(compact / "proxy_confusion.csv", proxy)
    (compact / "README.md").write_text(_readme(rows, policy_summary, policy_contrasts, proxy), encoding="utf-8")
    input_paths = {
        "released_workbook": Path(record["inputs"]["released_workbook"]["path"]),
        "completed_workbook": Path(record["inputs"]["completed_workbook"]["path"]),
        "private_review_key": Path(record["inputs"]["private_review_key"]),
        "candidate_metrics": Path(record["inputs"]["candidate_metrics"]),
        "policy_selections": Path(record["inputs"]["policy_selections"]),
        "automatic_proxy_audit": Path(record["inputs"]["automatic_proxy_audit"]),
    }
    input_hashes = {name: _file_sha(path) for name, path in input_paths.items()}
    raw_metadata = {
        "schema_version": "round2_human_first_analysis_private_run_metadata_v1",
        "input_sha256": input_hashes,
        "raw_join_sha256": _file_sha(raw / "unblinded_candidate_rows.csv"),
        "review_rows": len(rows),
        "source_cases": len({row["case_id"] for row in rows}),
    }
    (raw / "analysis_run_metadata.json").write_text(json.dumps(raw_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "round2_human_first_analysis_manifest_v1",
        "analysis_protocol_sha256": _file_sha(config_path),
        "method_freeze_commit": freeze["method_freeze_commit"],
        "method_freeze_binding_commit": "4fd823ab16b46292c01b65c9030da77bdb52ebe5",
        "method_freeze_binding_commit_precedes_unblinding": True,
        "completed_workbook_sha256": record["inputs"]["completed_workbook"]["sha256"],
        "released_workbook_sha256": record["inputs"]["released_workbook"]["sha256"],
        "hidden_input_hashes_recorded_in_ignored_run_metadata": True,
        "review_rows": len(rows),
        "source_cases": len({row["case_id"] for row in rows}),
        "candidate_rows_per_source": 3,
        "score_counts": dict(sorted(Counter(row["score"] for row in rows).items())),
        "bootstrap": record["uncertainty"],
        "policy_roles": {"primary": record["policies"]["primary"], "secondary": record["policies"]["secondary"]},
        "annotation_caveat": record["annotation_protocol"]["user_rule_caveat"],
        "claim_boundary": record["annotation_protocol"]["claim_boundary"],
        "outputs": {path.name: _file_sha(path) for path in sorted(compact.iterdir()) if path.is_file() and path.name != "analysis_manifest.json"},
        "raw_join_hash_recorded_in_ignored_run_metadata": True,
        "privacy": record["outputs"]["privacy"],
        "manuscript_changes_authorized": False,
    }
    (compact / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    _privacy_scan(compact, [row["review_id"] for row in rows] + [row["candidate_id"] for row in rows] + [row["case_id"] for row in rows])
    return manifest


__all__ = [
    "average_ranks",
    "build_proxy_confusion",
    "build_unblinded_rows",
    "direction_utility",
    "direction_valid",
    "human_delta",
    "load_and_validate_reviews",
    "load_protocol",
    "run_analysis",
    "spearman",
]
