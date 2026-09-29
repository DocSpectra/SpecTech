"""Frozen Phase D validation and analysis for the GPT-OSS-120B edit arm."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np

from src.analysis import human_first_reranking_analysis as inherited

CONFIG = Path("configs/round2_dgx_gptoss120b_human_first_analysis_v1.json")
FREEZE = Path("configs/round2_dgx_gptoss120b_human_first_analysis_freeze_record.json")
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
INSTRUCTIONS = {
    "A1": "Score", "B1": "Meaning", "A2": "1", "B2": "A clearly more specific",
    "A3": "2", "B3": "A slightly more specific", "A4": "3", "B4": "About the same specificity",
    "A5": "4", "B5": "B slightly more specific", "A6": "5", "B6": "B clearly more specific",
    "A7": "X", "B7": "Not comparable because meaning, factuality, or grammar is not acceptably preserved",
    "A8": "How to use", "B8": "Enter exactly one value (1, 2, 3, 4, 5, or X) in each Review-sheet Score cell.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sheet_cells(path: Path, entry: str) -> tuple[dict[str, str], list[str]]:
    with zipfile.ZipFile(path) as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = ["".join(node.text or "" for node in si.findall(".//s:t", NS)) for si in shared_root.findall("s:si", NS)]
        root = ET.fromstring(archive.read(entry))
    cells: dict[str, str] = {}
    formulas: list[str] = []
    for cell in root.findall(".//s:sheetData/s:row/s:c", NS):
        ref = cell.attrib["r"]
        if cell.find("s:f", NS) is not None:
            formulas.append(ref)
        cell_type = cell.attrib.get("t", "")
        value_node = cell.find("s:v", NS)
        if cell_type == "s":
            value = "" if value_node is None else shared[int(value_node.text or "0")]
        elif cell_type == "inlineStr":
            value = "".join(node.text or "" for node in cell.findall(".//s:t", NS))
        else:
            value = "" if value_node is None else value_node.text or ""
        cells[ref] = value
    return cells, formulas


def validate_completed_workbook(config_path: Path = CONFIG) -> dict[str, Any]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    completed = Path(record["inputs"]["completed_workbook"]["path"])
    released_csv = Path(record["inputs"]["released_packet"]["csv_path"])
    if sha256_file(completed) != record["inputs"]["completed_workbook"]["sha256"]:
        raise ValueError("completed workbook hash mismatch")
    if sha256_file(released_csv) != record["inputs"]["released_packet"]["csv_sha256"]:
        raise ValueError("released CSV hash mismatch")
    with zipfile.ZipFile(completed) as archive:
        names = set(archive.namelist())
        forbidden = [name for name in names if name.endswith("vbaProject.bin") or name.startswith("xl/externalLinks/") or name.startswith("customXml/")]
        if forbidden:
            raise ValueError("workbook contains forbidden active/external content")
        workbook_xml = archive.read("xl/workbook.xml").decode("utf-8")
        if re.search(r'state="(?:hidden|veryHidden)"', workbook_xml):
            raise ValueError("workbook contains hidden sheet")
    review, review_formulas = _sheet_cells(completed, "xl/worksheets/sheet1.xml")
    instructions, instruction_formulas = _sheet_cells(completed, "xl/worksheets/sheet2.xml")
    if review_formulas or instruction_formulas:
        raise ValueError("workbook contains formulas")
    allowed_review_refs = {f"{column}{row}" for row in range(1, 182) for column in "ABCD"}
    unexpected = {ref: value for ref, value in review.items() if value and ref not in allowed_review_refs}
    if unexpected:
        raise ValueError("unexpected nonempty Review cells")
    unexpected_instructions = {ref: value for ref, value in instructions.items() if value and ref not in INSTRUCTIONS}
    if unexpected_instructions or any(instructions.get(ref, "") != value for ref, value in INSTRUCTIONS.items()):
        raise ValueError("instructions changed")
    with released_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        released = list(csv.DictReader(handle))
    if len(released) != 180:
        raise ValueError("released CSV row count changed")
    headers = [review.get(f"{column}1", "") for column in "ABCD"]
    if headers != ["Review ID", "A", "B", "Score"]:
        raise ValueError("review headers changed")
    allowed_scores = set(record["inputs"]["completed_workbook"]["allowed_scores"])
    scores: list[str] = []
    review_ids: list[str] = []
    for index, released_row in enumerate(released, 2):
        actual = [review.get(f"{column}{index}", "") for column in "ABC"]
        expected = [released_row["Review ID"], released_row["A"], released_row["B"]]
        if actual != expected:
            raise ValueError(f"released row changed at position {index - 1}")
        score = review.get(f"D{index}", "")
        if score not in allowed_scores:
            raise ValueError(f"invalid or missing score at position {index - 1}")
        review_ids.append(actual[0])
        scores.append(score)
    if len(set(review_ids)) != 180:
        raise ValueError("Review IDs are not unique")
    return {
        "schema_version": "round2_dgx_gptoss120b_completed_workbook_gate_v1",
        "completed_workbook_sha256": sha256_file(completed),
        "released_csv_sha256": sha256_file(released_csv),
        "review_rows": 180,
        "unique_review_ids": 180,
        "immutable_review_columns_equal": True,
        "instructions_equal": True,
        "score_counts": dict(sorted(Counter(scores).items())),
        "clerical_corrections": [],
        "formulas": 0,
        "forbidden_active_or_external_members": 0,
    }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_frozen_protocol(config_path: Path = CONFIG) -> tuple[dict[str, Any], dict[str, Any]]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if record["schema_version"] != "round2_dgx_gptoss120b_human_first_analysis_v1":
        raise ValueError("unexpected Phase D protocol")
    for key in ("method_config", "schema", "method_spec", "freeze_test"):
        item = freeze[key]
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Phase D binding changed: {item['path']}")
    gate = validate_completed_workbook(config_path)
    if gate["completed_workbook_sha256"] != freeze["pre_unblinding_gate"]["completed_workbook_sha256"]:
        raise ValueError("completed-workbook freeze drift")
    if gate["score_counts"] != freeze["pre_unblinding_gate"]["score_counts"]:
        raise ValueError("completed-workbook score-count drift")
    return record, freeze


def _verify_bound_file(item: dict[str, str]) -> Path:
    path = Path(item["path"])
    if sha256_file(path) != item["sha256"]:
        raise ValueError(f"bound hidden input changed: {path}")
    return path


def _gpt_reviews(record: dict[str, Any]) -> list[dict[str, str]]:
    completed = Path(record["inputs"]["completed_workbook"]["path"])
    review, _ = _sheet_cells(completed, "xl/worksheets/sheet1.xml")
    return [
        {
            "review_position": str(position),
            "review_id": review[f"A{position + 1}"],
            "A": review[f"B{position + 1}"],
            "B": review[f"C{position + 1}"],
            "score": review[f"D{position + 1}"],
        }
        for position in range(1, 181)
    ]


def _adapter_record(record: dict[str, Any]) -> dict[str, Any]:
    inherited_record = json.loads(Path("configs/round2_human_first_analysis_v1.json").read_text(encoding="utf-8"))
    hidden = record["inputs"]["gptoss120b_hidden"]
    inherited_record["inputs"]["private_review_key"] = hidden["private_review_key"]["path"]
    inherited_record["inputs"]["candidate_metrics"] = hidden["candidate_metrics"]["path"]
    inherited_record["inputs"]["policy_selections"] = hidden["policy_selections"]["path"]
    inherited_record["inputs"]["automatic_proxy_audit"] = hidden["automatic_proxy_audit"]["path"]
    inherited_record["uncertainty"] = deepcopy(record["uncertainty"])
    inherited_record["policies"] = deepcopy(record["policies"])
    inherited_record["outputs"] = deepcopy(record["outputs"])
    return inherited_record


def _load_arms(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, str]], list[dict[str, Any]], list[dict[str, str]]]:
    for item in record["inputs"]["gptoss120b_hidden"].values():
        _verify_bound_file(item)
    for name, item in record["inputs"]["gemma_reviewed_arm"].items():
        if isinstance(item, dict) and "path" in item and "sha256" in item:
            _verify_bound_file(item)
    gpt_adapter = _adapter_record(record)
    gpt_rows, gpt_policies = inherited.build_unblinded_rows(gpt_adapter, _gpt_reviews(record))
    gemma_record, _ = inherited.load_protocol(Path(record["inputs"]["gemma_reviewed_arm"]["analysis_config"]["path"]))
    gemma_reviews = inherited.load_and_validate_reviews(gemma_record)
    gemma_rows, gemma_policies = inherited.build_unblinded_rows(gemma_record, gemma_reviews)
    for row in gpt_rows:
        row["generator_arm"] = "gpt_oss_120b"
    for row in gemma_rows:
        row["generator_arm"] = "gemma4_12b"
    if len(gpt_rows) != 180 or len(gemma_rows) != 180:
        raise ValueError("generator-arm row count mismatch")
    if Counter((row["corpus_id"], row["edit_type"]) for row in gpt_rows) != Counter((row["corpus_id"], row["edit_type"]) for row in gemma_rows):
        raise ValueError("generator-arm strata mismatch")
    if set(row["case_id"] for row in gpt_rows) != set(row["case_id"] for row in gemma_rows):
        raise ValueError("shared-source case IDs mismatch")
    return gpt_rows, gpt_policies, gemma_rows, gemma_policies


def _stable_seed(master: int, *parts: str) -> int:
    payload = ":".join([str(master), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _ci(values: np.ndarray) -> tuple[float | None, float | None, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return None, None, 0
    low, high = np.quantile(finite, [0.025, 0.975])
    return float(low), float(high), int(len(finite))


def _display(value: float | None) -> str:
    return "" if value is None or not np.isfinite(value) else f"{value:.12g}"


def _case_meta(rows: list[dict[str, Any]]) -> dict[str, tuple[str, str]]:
    out: dict[str, tuple[str, str]] = {}
    for row in rows:
        current = (row["corpus_id"], row["edit_type"])
        if row["case_id"] in out and out[row["case_id"]] != current:
            raise ValueError("case stratum changed")
        out[row["case_id"]] = current
    return out


def _draws(case_ids: list[str], meta: dict[str, tuple[str, str]], reps: int, seed: int, stratified: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    if not case_ids:
        return np.empty((reps, 0), dtype=int)
    if not stratified:
        return rng.integers(0, len(case_ids), size=(reps, len(case_ids)))
    by_cell: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, case_id in enumerate(case_ids):
        by_cell[meta[case_id]].append(index)
    chunks = [rng.choice(indices, size=(reps, len(indices)), replace=True) for _, indices in sorted(by_cell.items())]
    return np.concatenate(chunks, axis=1)


def _subset_cases(rows: list[dict[str, Any]], stratum_type: str, stratum_value: str) -> list[str]:
    if stratum_type == "overall":
        return sorted({row["case_id"] for row in rows})
    return sorted({row["case_id"] for row in rows if row[stratum_type] == stratum_value})


def _strata(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    return [("overall", "all")] + [("corpus_id", value) for value in sorted({r["corpus_id"] for r in rows})] + [("edit_type", value) for value in sorted({r["edit_type"] for r in rows})]


def _case_candidate_summaries(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["case_id"]].append(row)
    out = {}
    for case_id, values in grouped.items():
        if len(values) != 3:
            raise ValueError("expected three candidates per source")
        utilities = [float(row["direction_utility"]) for row in values if row["direction_utility"] is not None]
        out[case_id] = {
            "valid_fraction": sum(bool(row["direction_valid"]) for row in values) / 3,
            "x_fraction": sum(row["score"] == "X" for row in values) / 3,
            "mean_utility": float(np.mean(utilities)) if utilities else np.nan,
        }
    return out


def _bootstrap_paired_scalar(gpt: np.ndarray, gemma: np.ndarray, draws: np.ndarray) -> tuple[tuple[float, float, int], tuple[float, float, int], tuple[float, float, int]]:
    with np.errstate(invalid="ignore"):
        g_reps = np.nanmean(gpt[draws], axis=1)
        m_reps = np.nanmean(gemma[draws], axis=1)
    return _ci(g_reps), _ci(m_reps), _ci(g_reps - m_reps)


def build_generator_candidate_comparison(record: dict[str, Any], gpt_rows: list[dict[str, Any]], gemma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    meta = _case_meta(gpt_rows)
    gsum = _case_candidate_summaries(gpt_rows)
    msum = _case_candidate_summaries(gemma_rows)
    output: list[dict[str, Any]] = []
    for stratum_type, stratum_value in _strata(gpt_rows):
        case_ids = _subset_cases(gpt_rows, stratum_type, stratum_value)
        draws = _draws(case_ids, meta, reps, _stable_seed(master, "generator-candidate", stratum_type, stratum_value), stratum_type == "overall")
        for estimand in ("valid_fraction", "x_fraction", "mean_utility"):
            gpt = np.asarray([gsum[cid][estimand] for cid in case_ids])
            gemma = np.asarray([msum[cid][estimand] for cid in case_ids])
            gci, mci, dci = _bootstrap_paired_scalar(gpt, gemma, draws)
            output.append({
                "comparison_family": "per_source_candidate_summary", "stratum_type": stratum_type, "stratum_value": stratum_value,
                "estimand": estimand, "metric": "", "case_count": len(case_ids),
                "gpt_oss_120b_value": _display(float(np.nanmean(gpt))), "gpt_oss_120b_ci_low": _display(gci[0]), "gpt_oss_120b_ci_high": _display(gci[1]),
                "gemma4_12b_value": _display(float(np.nanmean(gemma))), "gemma4_12b_ci_low": _display(mci[0]), "gemma4_12b_ci_high": _display(mci[1]),
                "paired_difference_120b_minus_gemma": _display(float(np.nanmean(gpt - gemma))), "difference_ci_low": _display(dci[0]), "difference_ci_high": _display(dci[1]),
                "bootstrap_valid_replicates": dci[2],
            })
    return output


def _selected(rows: list[dict[str, Any]], policies: list[dict[str, str]], policy: str) -> dict[str, dict[str, Any]]:
    by_candidate = {row["candidate_id"]: row for row in rows}
    selected = {}
    for item in policies:
        if item["policy"] == policy:
            row = by_candidate[item["selected_candidate_id"]]
            if row["case_id"] != item["case_id"]:
                raise ValueError("policy mapping case mismatch")
            selected[row["case_id"]] = row
    if len(selected) != 60:
        raise ValueError("policy mapping coverage mismatch")
    return selected


def build_generator_policy_comparison(record: dict[str, Any], gpt_rows: list[dict[str, Any]], gpt_policies: list[dict[str, str]], gemma_rows: list[dict[str, Any]], gemma_policies: list[dict[str, str]]) -> list[dict[str, Any]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    meta = _case_meta(gpt_rows)
    policy_order = record["policies"]["primary"] + record["policies"]["secondary"]
    gsel = {p: _selected(gpt_rows, gpt_policies, p) for p in policy_order}
    msel = {p: _selected(gemma_rows, gemma_policies, p) for p in policy_order}
    output: list[dict[str, Any]] = []
    for stratum_type, stratum_value in _strata(gpt_rows):
        case_ids = _subset_cases(gpt_rows, stratum_type, stratum_value)
        for policy in policy_order:
            draws = _draws(case_ids, meta, reps, _stable_seed(master, "generator-policy", policy, stratum_type, stratum_value), stratum_type == "overall")
            gv = np.asarray([float(gsel[policy][cid]["direction_valid"]) for cid in case_ids])
            mv = np.asarray([float(msel[policy][cid]["direction_valid"]) for cid in case_ids])
            _, _, arm_ci = _bootstrap_paired_scalar(gv, mv, draws)
            if policy == "unguided_slot01":
                gain_g = gain_m = gain_diff = np.full(len(case_ids), np.nan)
                gain_ci = (None, None, 0)
            else:
                gain_g = gv - np.asarray([float(gsel["unguided_slot01"][cid]["direction_valid"]) for cid in case_ids])
                gain_m = mv - np.asarray([float(msel["unguided_slot01"][cid]["direction_valid"]) for cid in case_ids])
                gain_diff = gain_g - gain_m
                gain_ci = _ci(gain_diff[draws].mean(axis=1))
            output.append({
                "stratum_type": stratum_type, "stratum_value": stratum_value, "policy": policy,
                "policy_role": "reference" if policy == "unguided_slot01" else ("secondary" if policy in record["policies"]["secondary"] else "primary"),
                "case_count": len(case_ids), "gpt_oss_120b_valid_rate": _display(float(np.mean(gv))), "gemma4_12b_valid_rate": _display(float(np.mean(mv))),
                "valid_rate_difference_120b_minus_gemma": _display(float(np.mean(gv - mv))), "valid_difference_ci_low": _display(arm_ci[0]), "valid_difference_ci_high": _display(arm_ci[1]),
                "gpt_oss_120b_guided_gain": _display(float(np.nanmean(gain_g)) if policy != "unguided_slot01" else None),
                "gemma4_12b_guided_gain": _display(float(np.nanmean(gain_m)) if policy != "unguided_slot01" else None),
                "paired_guided_gain_difference": _display(float(np.nanmean(gain_diff)) if policy != "unguided_slot01" else None),
                "gain_difference_ci_low": _display(gain_ci[0]), "gain_difference_ci_high": _display(gain_ci[1]),
                "bootstrap_valid_replicates": arm_ci[2], "gain_bootstrap_valid_replicates": gain_ci[2],
            })
    return output


def _candidate_spearman(rows: list[dict[str, Any]], metric: str) -> float:
    valid = [row for row in rows if row["human_delta"] is not None]
    return inherited.spearman([row[metric] for row in valid], [row["human_delta"] for row in valid])


def build_generator_association_comparison(record: dict[str, Any], gpt_rows: list[dict[str, Any]], gemma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    meta = _case_meta(gpt_rows)
    output = []
    for stratum_type, stratum_value in _strata(gpt_rows):
        case_ids = _subset_cases(gpt_rows, stratum_type, stratum_value)
        draws = _draws(case_ids, meta, reps, _stable_seed(master, "generator-association", stratum_type, stratum_value), stratum_type == "overall")
        gb = defaultdict(list)
        mb = defaultdict(list)
        for row in gpt_rows: gb[row["case_id"]].append(row)
        for row in gemma_rows: mb[row["case_id"]].append(row)
        for metric in record["candidate_associations"]["metrics"]:
            gp = _candidate_spearman([r for cid in case_ids for r in gb[cid]], metric)
            mp = _candidate_spearman([r for cid in case_ids for r in mb[cid]], metric)
            gr, mr = [], []
            for draw in draws:
                sampled = [case_ids[int(i)] for i in draw]
                gr.append(_candidate_spearman([r for cid in sampled for r in gb[cid]], metric))
                mr.append(_candidate_spearman([r for cid in sampled for r in mb[cid]], metric))
            ga, ma = np.asarray(gr), np.asarray(mr)
            gci, mci, dci = _ci(ga), _ci(ma), _ci(ga - ma)
            output.append({
                "comparison_family": "candidate_metric_association", "stratum_type": stratum_type, "stratum_value": stratum_value,
                "estimand": "spearman_rho", "metric": metric, "case_count": len(case_ids),
                "gpt_oss_120b_value": _display(gp), "gpt_oss_120b_ci_low": _display(gci[0]), "gpt_oss_120b_ci_high": _display(gci[1]),
                "gemma4_12b_value": _display(mp), "gemma4_12b_ci_low": _display(mci[0]), "gemma4_12b_ci_high": _display(mci[1]),
                "paired_difference_120b_minus_gemma": _display(gp - mp), "difference_ci_low": _display(dci[0]), "difference_ci_high": _display(dci[1]),
                "bootstrap_valid_replicates": dci[2],
            })
    return output


def build_generator_proxy_comparison(record: dict[str, Any], gpt_rows: list[dict[str, Any]], gemma_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reps = int(record["uncertainty"]["replicates"])
    master = int(record["uncertainty"]["master_seed"])
    meta = _case_meta(gpt_rows)
    gb = defaultdict(list); mb = defaultdict(list)
    for row in gpt_rows: gb[row["case_id"]].append(row)
    for row in gemma_rows: mb[row["case_id"]].append(row)
    output = []
    for stratum_type, stratum_value in _strata(gpt_rows):
        case_ids = _subset_cases(gpt_rows, stratum_type, stratum_value)
        draws = _draws(case_ids, meta, reps, _stable_seed(master, "generator-proxy", stratum_type, stratum_value), stratum_type == "overall")
        gc = inherited._confusion_counts([r for cid in case_ids for r in gb[cid]])
        mc = inherited._confusion_counts([r for cid in case_ids for r in mb[cid]])
        gfar, gfrr = inherited._rates_from_confusion(gc); mfar, mfrr = inherited._rates_from_confusion(mc)
        g_case = {cid: inherited._confusion_counts(gb[cid]) for cid in case_ids}; m_case = {cid: inherited._confusion_counts(mb[cid]) for cid in case_ids}
        far_diffs, frr_diffs = [], []
        for draw in draws:
            gtotal = np.sum([g_case[case_ids[int(i)]] for i in draw], axis=0); mtotal = np.sum([m_case[case_ids[int(i)]] for i in draw], axis=0)
            gf, gr = inherited._rates_from_confusion(gtotal); mf, mr = inherited._rates_from_confusion(mtotal)
            far_diffs.append(gf - mf); frr_diffs.append(gr - mr)
        for estimand, gv, mv, diffs in (("false_accept_rate", gfar, mfar, far_diffs), ("false_reject_rate", gfrr, mfrr, frr_diffs)):
            dci = _ci(np.asarray(diffs))
            output.append({
                "comparison_family": "automatic_proxy", "stratum_type": stratum_type, "stratum_value": stratum_value,
                "estimand": estimand, "metric": "", "case_count": len(case_ids),
                "gpt_oss_120b_value": _display(gv), "gpt_oss_120b_ci_low": "", "gpt_oss_120b_ci_high": "",
                "gemma4_12b_value": _display(mv), "gemma4_12b_ci_low": "", "gemma4_12b_ci_high": "",
                "paired_difference_120b_minus_gemma": _display(gv - mv), "difference_ci_low": _display(dci[0]), "difference_ci_high": _display(dci[1]),
                "bootstrap_valid_replicates": dci[2],
            })
    return output


def _privacy_scan(compact: Path, forbidden: list[str]) -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in compact.iterdir() if path.is_file()).casefold()
    for token in ("review_id", "candidate_id", "case_id", "source_sent_id", "sentence_original", "sentence_candidate"):
        if token in text:
            raise ValueError(f"privacy-forbidden field in compact evidence: {token}")
    for value in forbidden:
        if value and value.casefold() in text:
            raise ValueError("private value leaked into compact evidence")


def _readme(gpt_rows: list[dict[str, Any]], summaries: list[dict[str, Any]], contrasts: list[dict[str, Any]], generator: list[dict[str, Any]], gains: list[dict[str, Any]]) -> str:
    counts = Counter(row["score"] for row in gpt_rows)
    overall = [row for row in summaries if row["analysis_set"] == "all_180" and row["stratum_type"] == "overall"]
    overall_contrasts = [row for row in contrasts if row["analysis_set"] == "all_180" and row["stratum_type"] == "overall"]
    candidate = {row["estimand"]: row for row in generator if row["comparison_family"] == "per_source_candidate_summary" and row["stratum_type"] == "overall"}
    lines = [
        "# GPT-OSS-120B human-first analysis", "",
        "The exact 180-row completed workbook passed the frozen released-packet gate before mechanical unblinding. The judgments cover three candidates for each of 60 source cases and were made by the same evaluator as the Gemma arm in a separate blinded session.", "",
        f"Ratings: 1={counts['1']}, 2={counts['2']}, 3={counts['3']}, 4={counts['4']}, 5={counts['5']}, X={counts['X']}.", "",
        "## GPT-OSS-120B policy results", "",
    ]
    for row in overall:
        lines.append(f"- {row['policy']}: {row['direction_valid_count']}/{row['case_count']} direction-valid ({100*float(row['direction_valid_rate']):.1f}%), X={row['x_count']}.")
    lines.extend(["", "Guided minus unguided:", ""])
    for row in overall_contrasts:
        lines.append(f"- {row['policy']}: {100*float(row['valid_rate_difference']):+.1f} pp (95% CI {100*float(row['difference_ci_low']):+.1f} to {100*float(row['difference_ci_high']):+.1f}).")
    lines.extend(["", "## Generator-arm comparison", ""])
    for name in ("valid_fraction", "x_fraction", "mean_utility"):
        row = candidate[name]
        lines.append(f"- {name}: GPT-OSS-120B={row['gpt_oss_120b_value']}, Gemma={row['gemma4_12b_value']}, paired difference={row['paired_difference_120b_minus_gemma']} [{row['difference_ci_low']}, {row['difference_ci_high']}].")
    lines.extend(["", "Guided-gain differences between generator arms remain policy-specific:", ""])
    for row in gains:
        if row["stratum_type"] == "overall" and row["policy"] != "unguided_slot01":
            lines.append(f"- {row['policy']}: 120B-minus-Gemma guided-gain difference={row['paired_guided_gain_difference']} [{row['gain_difference_ci_low']}, {row['gain_difference_ci_high']}].")
    lines.extend(["", "All estimates are one-evaluator results with source-case clustering. Separate review sessions, model family/size, runtime, and quantization prevent causal size or general editing-superiority claims. Null, adverse, direction-specific, order-sensitive, and proxy results are retained in the CSV evidence. Phase E remains unauthorized pending independent review.", ""])
    return "\n".join(lines)


def run_analysis(config_path: Path = CONFIG) -> dict[str, Any]:
    record, freeze = load_frozen_protocol(config_path)
    gpt_rows, gpt_policies, gemma_rows, gemma_policies = _load_arms(record)
    adapter = _adapter_record(record)
    rating = inherited.rating_order_distribution(gpt_rows)
    summaries, contrasts = inherited.build_policy_outputs(adapter, gpt_rows, gpt_policies)
    predictor = inherited.build_predictor_agreement(adapter, gpt_rows)
    proxy = inherited.build_proxy_confusion(adapter, gpt_rows)
    generator = build_generator_candidate_comparison(record, gpt_rows, gemma_rows)
    generator.extend(build_generator_association_comparison(record, gpt_rows, gemma_rows))
    generator.extend(build_generator_proxy_comparison(record, gpt_rows, gemma_rows))
    gains = build_generator_policy_comparison(record, gpt_rows, gpt_policies, gemma_rows, gemma_policies)
    raw = Path(record["outputs"]["raw_directory"]); compact = Path(record["outputs"]["compact_directory"])
    raw.mkdir(parents=True, exist_ok=True); compact.mkdir(parents=True, exist_ok=True)
    raw_fields = ["review_position", "review_id", "case_id", "candidate_id", "corpus_id", "edit_type", "candidate_slot", "candidate_side", "score", "human_delta", "direction_utility", "direction_valid", "proxy_pass", "proxy_reason_codes", *inherited.METRIC_ORDER, "generator_arm"]
    inherited._write_csv(raw / "gptoss120b_unblinded_candidate_rows.csv", gpt_rows, raw_fields)
    _write_csv(compact / "rating_order_distribution.csv", rating)
    _write_csv(compact / "policy_summary.csv", summaries)
    _write_csv(compact / "policy_contrasts.csv", contrasts)
    _write_csv(compact / "predictor_agreement.csv", predictor)
    _write_csv(compact / "proxy_confusion.csv", proxy)
    _write_csv(compact / "generator_comparison.csv", generator)
    _write_csv(compact / "generator_policy_gain_contrasts.csv", gains)
    (compact / "README.md").write_text(_readme(gpt_rows, summaries, contrasts, generator, gains), encoding="utf-8")
    private_hashes = {
        "gptoss120b_raw_join_sha256": sha256_file(raw / "gptoss120b_unblinded_candidate_rows.csv"),
        "gptoss120b_hidden_inputs": {name: sha256_file(Path(item["path"])) for name, item in record["inputs"]["gptoss120b_hidden"].items()},
        "gemma_hidden_inputs": {name: sha256_file(Path(item["path"])) for name, item in record["inputs"]["gemma_reviewed_arm"].items() if isinstance(item, dict) and "path" in item},
    }
    (raw / "analysis_run_metadata.json").write_text(json.dumps(private_hashes, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "round2_dgx_gptoss120b_human_first_analysis_manifest_v1",
        "method_config_sha256": sha256_file(config_path), "method_freeze_commit": freeze["method_freeze_commit"],
        "freeze_binding_commit": "4562641", "released_workbook_sha256": record["inputs"]["released_packet"]["released_xlsx_sha256"],
        "completed_workbook_sha256": record["inputs"]["completed_workbook"]["sha256"], "review_rows": 180, "source_cases": 60,
        "score_counts": dict(sorted(Counter(row["score"] for row in gpt_rows).items())), "bootstrap": record["uncertainty"],
        "policy_roles": {"primary": record["policies"]["primary"], "secondary": record["policies"]["secondary"]},
        "generator_arms": record["generator_comparison"]["arms"], "separate_blinded_sessions": True,
        "outputs": {path.name: sha256_file(path) for path in sorted(compact.iterdir()) if path.is_file() and path.name != "analysis_manifest.json"},
        "raw_hashes_recorded_only_in_ignored_metadata": True, "privacy": record["outputs"]["privacy"],
        "paper_use_disposition": "eligible_for_independent_primary_review_regardless_of_result_sign", "phase_e_authorized": False,
    }
    (compact / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    forbidden = [row["review_id"] for row in gpt_rows + gemma_rows] + [row["candidate_id"] for row in gpt_rows + gemma_rows] + [row["case_id"] for row in gpt_rows + gemma_rows]
    _privacy_scan(compact, forbidden)
    return manifest


__all__ = ["load_frozen_protocol", "run_analysis", "sha256_file", "validate_completed_workbook"]
