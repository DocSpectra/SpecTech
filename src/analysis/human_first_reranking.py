"""Frozen human-first controlled-edit generation, scoring, and blinding workflow."""
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import requests

from src.analysis.gemma_controlled_editor import (
    GemmaCase,
    gate_edit as legacy_proxy,
    load_cases as load_gemma_cases,
    require_only_pinned_gemma_resident,
    stop_ollama_model,
)
from src.analysis.pilot_model_human import sha256_file
from src.analysis.qwen_edit_source import _docker_image_id
from src.ko_specificity.io import parse_predictions
from src.speciteller.config import DEFAULT_SPECITELLER_CONFIG
from src.speciteller.runner import preflight_speciteller, run_speciteller
from src.speciteller.tokenize import tokenize_for_speciteller


SCHEMA_VERSION = "round2_human_first_reranking_v1"
CONFIG = Path("configs/round2_human_first_reranking_v1.json")
FREEZE = Path("configs/round2_human_first_reranking_freeze_record.json")
BOUNDARY_RE = re.compile(r"[.!?](?:[\"')\]]*)?(?=\s+[A-Z]|\s*$)")


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False))


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    names = fields or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_protocol(config_path: Path = CONFIG) -> tuple[dict[str, Any], str, dict[str, Any]]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION or record.get("outcome_blind_method_freeze") is not True:
        raise ValueError("unexpected/unfrozen human-first protocol")
    digest = sha256_file(config_path)
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze["config_sha256"] != digest or not freeze["method_frozen_before_any_project_generation_or_candidate_scoring"]:
        raise ValueError("human-first freeze binding failed")
    return record, digest, freeze


def load_cases(record: dict[str, Any]) -> list[GemmaCase]:
    base = json.loads(Path(record["inputs"]["base_protocol"]["path"]).read_text(encoding="utf-8"))
    if sha256_file(Path(record["inputs"]["base_protocol"]["path"])) != record["inputs"]["base_protocol"]["sha256"]:
        raise ValueError("base Gemma protocol hash changed")
    cases = load_gemma_cases(base)
    if len(cases) != 60:
        raise ValueError("expected 60 frozen originals")
    return cases


def candidate_id(case_id: str, slot: int) -> str:
    return _sha(f"{case_id}\0candidate_slot\0{slot}")


def derive_seed(record: dict[str, Any], case_id_value: str, slot: int, attempt: int) -> int:
    runtime = record["runtime"]
    value = f"{runtime['master_seed']}:{runtime['seed_namespace']}:{case_id_value}:{slot}:{attempt}"
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], "big") & 0x7FFFFFFF


def build_request(record: dict[str, Any], case: GemmaCase, slot: int, attempt: int) -> dict[str, Any]:
    runtime = record["runtime"]
    options = dict(runtime["options"])
    options["seed"] = derive_seed(record, case.case_id, slot, attempt)
    return {
        "model": runtime["model"],
        "messages": [
            {"role": "system", "content": record["prompts"]["system"]},
            {"role": "user", "content": record["prompts"]["user_templates"][case.edit_type].format(sentence_original=case.sentence_original)},
        ],
        "stream": False,
        "think": False,
        "format": runtime["response_schema"],
        "options": options,
        "keep_alive": runtime["keep_alive"],
    }


def parse_response(payload: dict[str, Any], expected_model: str) -> tuple[str, dict[str, Any]]:
    if payload.get("model") != expected_model:
        raise ValueError("response model mismatch")
    content = payload.get("message", {}).get("content")
    if not isinstance(content, str):
        raise ValueError("missing response content")
    parsed = json.loads(content)
    if not isinstance(parsed, dict) or set(parsed) != {"edited_sentence"} or not isinstance(parsed["edited_sentence"], str):
        raise ValueError("response is not exact edited_sentence JSON")
    if payload.get("message", {}).get("thinking") not in (None, ""):
        raise ValueError("unexpected thinking content")
    return parsed["edited_sentence"], {
        "response_content_sha256": _sha(content),
        "created_at": str(payload.get("created_at") or ""),
        "done_reason": str(payload.get("done_reason") or ""),
        "prompt_eval_count": int(payload.get("prompt_eval_count") or 0),
        "eval_count": int(payload.get("eval_count") or 0),
    }


def integrity_reasons(original: str, candidate: str) -> list[str]:
    reasons: list[str] = []
    text = candidate.strip()
    if not text or any(char in text for char in "\n\r\t"):
        reasons.append("not_one_nonempty_line")
    if len(text) < 5:
        reasons.append("too_short_chars")
    if len(text) > 2000:
        reasons.append("too_long_chars")
    norm = lambda s: " ".join(unicodedata.normalize("NFKC", s).strip().split()).casefold()
    if norm(text) == norm(original):
        reasons.append("unchanged_normalized")
    if len(BOUNDARY_RE.findall(text)) > 1:
        reasons.append("multiple_sentences")
    original_nonlatin = any(unicodedata.category(c).startswith("L") and ord(c) > 127 for c in original)
    candidate_nonlatin = any(unicodedata.category(c).startswith("L") and ord(c) > 127 for c in text)
    if not original_nonlatin and candidate_nonlatin:
        reasons.append("new_non_latin_script")
    return sorted(set(reasons))


def _api_post(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise ValueError("non-object response")
    return value


def _warm(record: dict[str, Any], timeout: float) -> dict[str, Any]:
    runtime = record["runtime"]
    request = {
        "model": runtime["model"], "messages": [{"role": "user", "content": "Return JSON with edited_sentence equal to: Runtime fixture complete."}],
        "stream": False, "think": False, "format": runtime["response_schema"],
        "options": {**runtime["options"], "seed": 2026081110}, "keep_alive": runtime["keep_alive"],
    }
    text, meta = parse_response(_api_post(runtime["api_url"], request, timeout), runtime["model"])
    if text.strip() != "Runtime fixture complete.":
        raise ValueError("Gemma warm fixture failed")
    return {**meta, "request_sha256": _canonical_sha(request), "residency": require_only_pinned_gemma_resident(record)}


def _verify_gemma_identity(record: dict[str, Any]) -> dict[str, Any]:
    runtime = record["runtime"]
    version = subprocess.run(["ollama", "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if runtime["ollama_version"] not in version:
        raise ValueError("Ollama version mismatch")
    listing = subprocess.run(["ollama", "list"], check=True, capture_output=True, text=True).stdout
    lines = [line for line in listing.splitlines() if line.split()[:1] == [runtime["model"]]]
    if len(lines) != 1 or runtime["model_list_id"] not in lines[0]:
        raise ValueError("Gemma list identity mismatch")
    modelfile = subprocess.run(["ollama", "show", runtime["model"], "--modelfile"], check=True, capture_output=True, text=True).stdout
    for blob in (runtime["model_blob_sha256"], runtime["projector_blob_sha256"]):
        if f"sha256-{blob}" not in modelfile:
            raise ValueError("Gemma blob identity mismatch")
    license_text = subprocess.run(["ollama", "show", runtime["model"], "--license"], check=True, capture_output=True, text=True).stdout
    if "Apache License" not in license_text or "Version 2.0" not in license_text:
        raise ValueError("Gemma license mismatch")
    return {"ollama_version_output": version, "model": runtime["model"], "model_tag_digest": runtime["model_tag_digest"], "model_blob_sha256": runtime["model_blob_sha256"], "projector_blob_sha256": runtime["projector_blob_sha256"]}


def run_generation(config_path: Path = CONFIG, timeout: float = 300.0) -> dict[str, Any]:
    record, protocol_sha, freeze = load_protocol(config_path)
    cases = load_cases(record)
    raw = Path(record["outputs"]["raw_directory"])
    if raw.exists() and any(raw.iterdir()):
        raise ValueError("fresh human-first output namespace must be empty")
    raw.mkdir(parents=True, exist_ok=True)
    for model in ("qwen3:14b", "gpt-oss:20b", "gemma4:12b"):
        stop_ollama_model(model)
    identity = _verify_gemma_identity(record)
    warm = _warm(record, timeout)
    attempts_path = raw / "generation_attempts.jsonl"
    candidates: list[dict[str, Any]] = []
    attempts = 0
    started = _utc()
    try:
        with attempts_path.open("w", encoding="utf-8", newline="\n") as handle:
            for case in cases:
                for slot in range(1, 4):
                    cid = candidate_id(case.case_id, slot)
                    accepted = None
                    for attempt in range(1, 4):
                        request = build_request(record, case, slot, attempt)
                        request_sha = _canonical_sha(request)
                        seed = request["options"]["seed"]
                        status = "interface_error"
                        reasons: list[str] = []
                        edited = ""
                        meta: dict[str, Any] = {}
                        try:
                            edited, meta = parse_response(_api_post(record["runtime"]["api_url"], request, timeout), record["runtime"]["model"])
                            reasons = integrity_reasons(case.sentence_original, edited)
                            status = "retained" if not reasons else "integrity_rejected"
                        except Exception as exc:  # bounded interface retry; preserve only class, not message/text
                            reasons = [f"interface_error:{type(exc).__name__}"]
                        row = {
                            "schema_version": "round2_human_first_attempt_v1", "protocol_sha256": protocol_sha,
                            "case_id": case.case_id, "candidate_id": cid, "corpus_id": case.corpus_id,
                            "source_sent_id": case.source_sent_id, "source_position": case.source_position,
                            "edit_type": case.edit_type, "candidate_slot": slot, "attempt_index": attempt,
                            "seed": seed, "status": status, "reason_codes": reasons,
                            "original_sha256": case.original_sha256,
                            "edited_sentence": edited.strip(), "candidate_sha256": _sha(edited.strip()) if edited else "",
                            "request_sha256": request_sha, **meta,
                        }
                        handle.write(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")
                        handle.flush(); attempts += 1
                        if status == "retained":
                            accepted = row; break
                    if accepted is not None:
                        candidates.append({
                            "case_id": case.case_id, "candidate_id": cid, "corpus_id": case.corpus_id,
                            "source_sent_id": case.source_sent_id, "source_position": case.source_position,
                            "edit_type": case.edit_type, "candidate_slot": slot,
                            "sentence_original": case.sentence_original, "sentence_candidate": accepted["edited_sentence"],
                            "original_sha256": case.original_sha256, "candidate_sha256": accepted["candidate_sha256"],
                            "retained_attempt_index": accepted["attempt_index"], "seed": accepted["seed"],
                        })
        _write_csv(raw / "retained_candidates.csv", candidates)
        complete = len(candidates) == 180 and len({r["candidate_id"] for r in candidates}) == 180
        metadata = {
            "schema_version": "round2_human_first_generation_v1", "protocol_sha256": protocol_sha,
            "method_freeze_commit": freeze["method_freeze_commit"], "started_at_utc": started,
            "completed_at_utc": _utc(), "identity": identity, "warmup": warm,
            "residency_after_generation": require_only_pinned_gemma_resident(record),
            "source_cases": len(cases), "planned_candidates": 180, "retained_candidates": len(candidates),
            "attempt_rows": attempts, "generation_complete": complete,
            "attempts_sha256": sha256_file(attempts_path), "candidates_sha256": sha256_file(raw / "retained_candidates.csv"),
        }
        (raw / "generation_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return metadata
    finally:
        stop_ollama_model("gemma4:12b")


def _require_hash(path: Path, expected: str) -> None:
    if sha256_file(path) != expected:
        raise ValueError(f"hash mismatch: {path}")


def _scorer_config(record: dict[str, Any]) -> dict[str, Any]:
    entry = record["scoring"]["identity_source"]
    path = Path(entry["path"]); _require_hash(path, entry["sha256"])
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_scoring_and_proxies(config_path: Path = CONFIG) -> dict[str, Any]:
    record, protocol_sha, _ = load_protocol(config_path)
    cases = load_cases(record); case_map = {c.case_id: c for c in cases}
    raw = Path(record["outputs"]["raw_directory"])
    candidates = _read_csv(raw / "retained_candidates.csv")
    if len(candidates) != 180 or len({r["candidate_id"] for r in candidates}) != 180:
        raise ValueError("candidate completeness gate failed")
    base = json.loads(Path(record["inputs"]["base_protocol"]["path"]).read_text(encoding="utf-8"))
    proxy_rows = []
    for row in candidates:
        case = case_map[row["case_id"]]
        reasons, metrics = legacy_proxy(base, case.sentence_original, row["sentence_candidate"], case.edit_type)
        proxy_rows.append({"candidate_id": row["candidate_id"], "proxy_pass": str(not reasons).lower(), "reason_codes": "|".join(reasons), **metrics})
    _write_csv(raw / "automatic_proxy_audit.csv", proxy_rows)
    scoring = raw / "scoring"; scoring.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict[str, Any]] = []
    for case in cases:
        sid = _sha(f"{case.case_id}\0original")
        score_rows.append({"score_id": sid, "case_id": case.case_id, "candidate_id": "", "corpus_id": case.corpus_id, "role": "original", "candidate_slot": "", "text_sha256": case.original_sha256, "text": case.sentence_original})
        for row in sorted((r for r in candidates if r["case_id"] == case.case_id), key=lambda r: int(r["candidate_slot"])):
            score_rows.append({"score_id": _sha(f"{row['candidate_id']}\0candidate"), "case_id": case.case_id, "candidate_id": row["candidate_id"], "corpus_id": case.corpus_id, "role": "candidate", "candidate_slot": row["candidate_slot"], "text_sha256": row["candidate_sha256"], "text": row["sentence_candidate"]})
    _write_csv(raw / "scoring_manifest.csv", [{k: v for k, v in row.items() if k != "text"} for row in score_rows])
    spec = scoring / "speciteller"; spec.mkdir(parents=True, exist_ok=True)
    with (spec / "input.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for row in score_rows: handle.write(f"{row['score_id']}\t{tokenize_for_speciteller(row['text'])}\n")
    gs = scoring / "granuscore"; gs.mkdir(parents=True, exist_ok=True)
    # The frozen GranuScore runner requires a BOM-free UTF-8 header.
    gs_rows = [{"corpus_id": "human_first_edits", "sent_id": r["score_id"], "sent_text": r["text"]} for r in score_rows]
    with (gs / "input.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corpus_id", "sent_id", "sent_text"], lineterminator="\n")
        writer.writeheader(); writer.writerows(gs_rows)
    scorer = _scorer_config(record)
    for corpus in record["inputs"]["corpus_order"]:
        rows = [r for r in score_rows if r["corpus_id"] == corpus]
        bundle = scoring / "ko" / corpus / "bundle"; bundle.mkdir(parents=True, exist_ok=True)
        texts = [r["text"] for r in rows]
        (bundle / "twitters.txt").write_text("\n".join([texts[0], *texts]) + "\n", encoding="utf-8")
        (bundle / "twitterl.txt").write_text("1\n" * (len(texts) + 1), encoding="utf-8")
        (bundle / "twitterv.txt").write_text("0.5\n" * (len(texts) + 1), encoding="utf-8")
        canonical = Path("outputs/round2/ko_official_release/inputs") / f"{corpus}.csv"
        target = _read_csv(canonical)
        (bundle / "twitteru.txt").write_text("\n".join(r["text"] for r in target) + "\n", encoding="utf-8")
        _write_csv(bundle / "row_map.csv", [{"prediction_index": i, "score_id": r["score_id"]} for i, r in enumerate(rows)])
    metadata = {"schema_version": "round2_human_first_scoring_preparation_v1", "protocol_sha256": protocol_sha, "rows": len(score_rows), "candidate_rows": 180, "original_rows": 60, "proxy_rows": len(proxy_rows), "manifest_sha256": sha256_file(raw / "scoring_manifest.csv"), "proxy_sha256": sha256_file(raw / "automatic_proxy_audit.csv"), "scorer_identity_config_sha256": record["scoring"]["identity_source"]["sha256"]}
    (scoring / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def score_speciteller(config_path: Path = CONFIG) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path); scorer = _scorer_config(record)["scoring"]["speciteller"]
    if _docker_image_id(scorer["image"]) != scorer["image_id"]: raise ValueError("SpeciTeller image mismatch")
    directory = Path(record["outputs"]["raw_directory"]) / "scoring" / "speciteller"
    preflight_speciteller(DEFAULT_SPECITELLER_CONFIG)
    run_speciteller(DEFAULT_SPECITELLER_CONFIG, directory / "input.tsv", directory / "scores.tsv")
    rows = sum(1 for _ in (directory / "scores.tsv").open(encoding="utf-8"))
    if rows != 240: raise ValueError("SpeciTeller coverage mismatch")
    return {"rows": rows, "sha256": sha256_file(directory / "scores.tsv")}


def score_granuscore(config_path: Path = CONFIG) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path); scorer = _scorer_config(record)["scoring"]["granuscore"]
    if _docker_image_id(scorer["image"]) != scorer["image_id"]: raise ValueError("GranuScore image mismatch")
    directory = Path(record["outputs"]["raw_directory"]) / "scoring" / "granuscore"; inp = directory / "input.csv"
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{Path.cwd().resolve()}:/work", "-w", "/work", scorer["image"], "--input", f"/work/{inp.as_posix()}", "--output", f"/work/{(directory/'scores.csv').as_posix()}", "--metadata", f"/work/{(directory/'scores.metadata.json').as_posix()}", "--corpus-id", "human_first_edits", "--expected-input-sha256", sha256_file(inp), "--batch-size", "256", "--encoding-batch-size", "256"]
    subprocess.run(cmd, check=True)
    meta = json.loads((directory / "scores.metadata.json").read_text(encoding="utf-8"))
    if meta["row_count"] != 240 or meta["runner_sha256"] != scorer["runner_sha256"]: raise ValueError("GranuScore identity/coverage mismatch")
    return {"rows": 240, "sha256": meta["output_sha256"]}


def score_ko(config_path: Path, corpus: str, run_id: str) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path); ko = _scorer_config(record)["scoring"]["ko"]
    if _docker_image_id(ko["image"]) != ko["image_id"]: raise ValueError("Ko image mismatch")
    run = ko["runs"][corpus][run_id]; checkpoint = Path(run["checkpoint_path"]); _require_hash(checkpoint, run["checkpoint_sha256"])
    base = Path(record["outputs"]["raw_directory"]) / "scoring" / "ko" / corpus; bundle = base / "bundle"; output = base / run_id; output.mkdir(parents=True, exist_ok=True)
    shell = "set -euo pipefail; test \"$(sha256sum /artifacts/glove.840B.300d.txt | cut -d' ' -f1)\" = \"$GLOVE_TXT_SHA256\"; cp -a /opt/ko /tmp/ko; cd /tmp/ko; ln -s /artifacts/glove.840B.300d.txt glove.840B.300d.txt; cp /target/twitters.txt /target/twitteru.txt /target/twitterl.txt /target/twitterv.txt dataset/data/; cp /checkpoint/model.pickle savedir/3osmodel.pickle; python test.py --gpu_id 0 --test_data twitter > /output/test.log 2>&1; cp predictions.txt /output/predictions.txt"
    cmd = ["docker", "run", "--rm", "-e", f"GLOVE_TXT_SHA256={ko['glove_text_sha256']}", "-v", f"{bundle.resolve()}:/target:ro", "-v", f"{output.resolve()}:/output", "-v", f"{checkpoint.resolve()}:/checkpoint/model.pickle:ro", "-v", f"{ko['glove_volume']}:/artifacts:ro", "--entrypoint", "bash", ko["image"], "-lc", shell]
    subprocess.run(cmd, check=True)
    mapping = _read_csv(bundle / "row_map.csv"); scores = parse_predictions(output / "predictions.txt", len(mapping))
    rows = [{"score_id": row["score_id"], "corpus_id": corpus, "run_id": run_id, "score_raw": score, "checkpoint_sha256": run["checkpoint_sha256"]} for row, score in zip(mapping, scores)]
    _write_csv(output / "scores.csv", rows)
    meta = {"row_count": len(rows), "checkpoint_sha256": run["checkpoint_sha256"], "prediction_sha256": sha256_file(output / "predictions.txt"), "score_sha256": sha256_file(output / "scores.csv")}
    (output / "scores.metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": len(rows), "sha256": meta["score_sha256"]}


def _score_map(path: Path, key: str, value: str) -> dict[str, float]:
    rows = _read_csv(path); out = {r[key]: float(r[value]) for r in rows}
    if len(out) != len(rows): raise ValueError(f"duplicate score IDs: {path}")
    return out


def _review_id(record: dict[str, Any], cid: str) -> str:
    b = record["blinding"]; digest = hashlib.sha256(f"{b['review_id_seed']}:{b['review_id_namespace']}:{cid}".encode()).digest()
    return "HFE-" + base64.b32encode(digest).decode().rstrip("=")[:12]


def _sort_hash(seed: int, value: str) -> str:
    return _sha(f"{seed}:{value}")


def build_packet_and_metrics(config_path: Path = CONFIG) -> dict[str, Any]:
    record, protocol_sha, freeze = load_protocol(config_path); cases = load_cases(record); case_map = {c.case_id: c for c in cases}
    raw = Path(record["outputs"]["raw_directory"]); candidates = _read_csv(raw / "retained_candidates.csv"); manifest = _read_csv(raw / "scoring_manifest.csv")
    if len(candidates) != 180 or len(manifest) != 240: raise ValueError("generation/scoring manifest incomplete")
    score_ids = {r["score_id"] for r in manifest}
    spec = {}; 
    for line in (raw / "scoring/speciteller/scores.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            bits = line.split("\t"); spec[bits[0]] = float(bits[1])
    gs_rows = _read_csv(raw / "scoring/granuscore/scores.csv"); gs = {r["sent_id"]: float(r["granuscore_percentile"]) for r in gs_rows}
    scorer_cfg = _scorer_config(record); kos: dict[str, dict[str, float]] = {}
    for corpus in record["inputs"]["corpus_order"]:
        for run in ("run01", "run02", "run03"):
            kos[f"{corpus}:{run}"] = _score_map(raw / f"scoring/ko/{corpus}/{run}/scores.csv", "score_id", "score_raw")
    if set(spec) != score_ids or set(gs) != score_ids: raise ValueError("global scorer coverage mismatch")
    for corpus in record["inputs"]["corpus_order"]:
        expected = {r["score_id"] for r in manifest if r["corpus_id"] == corpus}
        for run in ("run01", "run02", "run03"):
            if set(kos[f"{corpus}:{run}"]) != expected: raise ValueError("Ko coverage mismatch")
    original_score_id = {r["case_id"]: r["score_id"] for r in manifest if r["role"] == "original"}
    candidate_score_id = {r["candidate_id"]: r["score_id"] for r in manifest if r["role"] == "candidate"}
    metric_rows = []
    for row in candidates:
        case = case_map[row["case_id"]]; osid = original_score_id[case.case_id]; csid = candidate_score_id[row["candidate_id"]]
        values = {"speciteller_frozen_round1": spec[csid] - spec[osid], "ko_run01": kos[f"{case.corpus_id}:run01"][csid] - kos[f"{case.corpus_id}:run01"][osid], "ko_run02": kos[f"{case.corpus_id}:run02"][csid] - kos[f"{case.corpus_id}:run02"][osid], "ko_run03": kos[f"{case.corpus_id}:run03"][csid] - kos[f"{case.corpus_id}:run03"][osid], "granuscore_direction_aligned": -(gs[csid] - gs[osid])}
        values["ko_three_run_arithmetic_mean_secondary"] = float(np.mean([values[f"ko_run0{i}"] for i in (1,2,3)]))
        metric_rows.append({"candidate_id": row["candidate_id"], "case_id": case.case_id, "candidate_slot": int(row["candidate_slot"]), "edit_type": case.edit_type, **values, "granuscore_native_original": gs[osid], "granuscore_native_candidate": gs[csid]})
    _write_csv(raw / "candidate_metric_deltas.csv", metric_rows)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in metric_rows: by_case[r["case_id"]].append(r)
    policy_rows = []
    primary = ["speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "granuscore_direction_aligned"]
    for case in cases:
        pool = sorted(by_case[case.case_id], key=lambda r: r["candidate_slot"])
        chosen: dict[str, int] = {"unguided_slot01": 1}
        for metric in [*primary, "ko_three_run_arithmetic_mean_secondary"]:
            if case.edit_type == "add_specific": key = lambda r: (-r[metric], r["candidate_slot"])
            elif case.edit_type == "de_specify": key = lambda r: (r[metric], r["candidate_slot"])
            else: key = lambda r: (abs(r[metric]), r["candidate_slot"])
            chosen[metric] = sorted(pool, key=key)[0]["candidate_slot"]
        rank_sums = Counter()
        for metric in primary:
            if case.edit_type == "add_specific": ordered = sorted(pool, key=lambda r: (-r[metric], r["candidate_slot"]))
            elif case.edit_type == "de_specify": ordered = sorted(pool, key=lambda r: (r[metric], r["candidate_slot"]))
            else: ordered = sorted(pool, key=lambda r: (abs(r[metric]), r["candidate_slot"]))
            for rank, item in enumerate(ordered, 1): rank_sums[item["candidate_slot"]] += rank
        chosen["rank_consensus_secondary"] = min(rank_sums, key=lambda slot: (rank_sums[slot], slot))
        for policy, slot in chosen.items(): policy_rows.append({"case_id": case.case_id, "policy": policy, "selected_candidate_slot": slot, "selected_candidate_id": candidate_id(case.case_id, slot)})
    _write_csv(raw / "policy_selections.csv", policy_rows)
    sides: dict[str, str] = {}
    cells: dict[tuple[str,str], list[dict[str,str]]] = defaultdict(list)
    for r in candidates: cells[(r["corpus_id"], r["edit_type"])].append(r)
    for cell, rows in cells.items():
        ordered = sorted(rows, key=lambda r: _sort_hash(record["blinding"]["side_seed"], r["candidate_id"]))
        if len(ordered) % 2: raise ValueError(f"odd hidden cell: {cell}")
        for i, row in enumerate(ordered): sides[row["candidate_id"]] = "A" if i < len(ordered)//2 else "B"
    packet = []; key_rows = []
    proxy = {r["candidate_id"]: r for r in _read_csv(raw / "automatic_proxy_audit.csv")}
    metric = {r["candidate_id"]: r for r in metric_rows}
    selected = defaultdict(list)
    for r in policy_rows: selected[r["selected_candidate_id"]].append(r["policy"])
    for row in candidates:
        case = case_map[row["case_id"]]; side = sides[row["candidate_id"]]; rid = _review_id(record, row["candidate_id"])
        a = row["sentence_candidate"] if side == "A" else case.sentence_original; b = case.sentence_original if side == "A" else row["sentence_candidate"]
        packet.append({"Review ID": rid, "A": a, "B": b, "Score": ""})
        key_rows.append({**row, "review_id": rid, "candidate_side": side, "sentence_a_sha256": _sha(a), "sentence_b_sha256": _sha(b), "proxy_pass": proxy[row["candidate_id"]]["proxy_pass"], "proxy_reason_codes": proxy[row["candidate_id"]]["reason_codes"], **{f"metric_{k}": v for k,v in metric[row["candidate_id"]].items() if k not in {"candidate_id","case_id","candidate_slot","edit_type"}}, "selected_by_policies": "|".join(sorted(selected[row["candidate_id"]]))})
    packet.sort(key=lambda r: _sort_hash(record["blinding"]["row_order_seed"], r["Review ID"]))
    _write_csv(raw / record["outputs"]["review_csv"], packet, ["Review ID", "A", "B", "Score"])
    _write_csv(raw / "private_review_key.csv", key_rows)
    # Mechanical reversibility and privacy checks without printing content.
    if len({r["Review ID"] for r in packet}) != 180 or len({r["review_id"] for r in key_rows}) != 180: raise ValueError("review ID bijection failed")
    if Counter(sides.values()) != Counter({"A":90,"B":90}): raise ValueError("overall side balance failed")
    for rows in cells.values():
        if Counter(sides[r["candidate_id"]] for r in rows)["A"] != len(rows)//2: raise ValueError("cell side balance failed")
    manifest_out = {"schema_version": "round2_human_first_packet_manifest_v1", "protocol_sha256": protocol_sha, "method_freeze_commit": freeze["method_freeze_commit"], "source_cases": 60, "candidate_rows": 180, "review_rows": 180, "score_manifest_rows": 240, "metric_rows": len(metric_rows), "policy_rows": len(policy_rows), "proxy_rows": len(proxy), "candidate_side_counts": dict(Counter(sides.values())), "hidden_cell_candidate_counts": {f"{a}:{b}": len(v) for (a,b),v in sorted(cells.items())}, "review_csv_sha256": sha256_file(raw / record["outputs"]["review_csv"]), "private_key_sha256": sha256_file(raw / "private_review_key.csv"), "metric_sha256": sha256_file(raw / "candidate_metric_deltas.csv"), "policy_sha256": sha256_file(raw / "policy_selections.csv"), "proxy_sha256": sha256_file(raw / "automatic_proxy_audit.csv"), "xlsx_pending": True, "release_gate_passed": False}
    (raw / "packet_manifest_pre_xlsx.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_out


def finalize_xlsx(config_path: Path = CONFIG) -> dict[str, Any]:
    record, _, _ = load_protocol(config_path); raw = Path(record["outputs"]["raw_directory"]); pre = json.loads((raw / "packet_manifest_pre_xlsx.json").read_text(encoding="utf-8"))
    xlsx = raw / record["outputs"]["review_xlsx"]
    if not xlsx.exists(): raise ValueError("XLSX missing")
    pre["review_xlsx_sha256"] = sha256_file(xlsx); pre["xlsx_pending"] = False; pre["release_gate_passed"] = True; pre["completed_at_utc"] = _utc()
    (raw / "packet_manifest.json").write_text(json.dumps(pre, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = Path(record["outputs"]["compact_directory"]); compact.mkdir(parents=True, exist_ok=True)
    attempts = [json.loads(line) for line in (raw / "generation_attempts.jsonl").read_text(encoding="utf-8").splitlines() if line]
    coverage = []
    candidates = _read_csv(raw / "retained_candidates.csv")
    for (corpus, direction), count in sorted(Counter((r["corpus_id"],r["edit_type"]) for r in candidates).items()): coverage.append({"corpus_id": corpus, "edit_type": direction, "planned_candidates": count, "retained_candidates": count, "complete": "true"})
    _write_csv(compact / "generation_coverage.csv", coverage)
    proxy_rows = _read_csv(raw / "automatic_proxy_audit.csv"); proxy_agg=[]
    lookup = {r["candidate_id"]: r for r in candidates}
    for (corpus,direction,passed), count in sorted(Counter((lookup[r["candidate_id"]]["corpus_id"],lookup[r["candidate_id"]]["edit_type"],r["proxy_pass"]) for r in proxy_rows).items()): proxy_agg.append({"corpus_id":corpus,"edit_type":direction,"proxy_pass":passed,"candidate_count":count})
    _write_csv(compact / "proxy_aggregate.csv", proxy_agg)
    compact_manifest = {k:v for k,v in pre.items() if k != "private_key_sha256"}
    compact_manifest["private_key_sha256_recorded_in_ignored_manifest"] = True
    (compact / "packet_manifest.json").write_text(json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (compact / "README.md").write_text("# Human-first controlled-edit reranking packet\n\nThe frozen workflow generated 180/180 fresh Gemma candidates (three per source), computed the prior automatic proxies without gating, completed exact SpeciTeller, three-run Ko, and native GranuScore coverage, and created a reversible 180-row blinded packet. The user-facing packet contains only Review ID, A, B, and Score. Predictor values, proxy outcomes, hidden conditions, roles, and policy selections remain private until all reviews are frozen. No human outcomes or manuscript claims were analyzed.\n", encoding="utf-8")
    pre["compact_outputs"] = {p.name: sha256_file(p) for p in compact.iterdir() if p.is_file()}
    return pre


__all__ = ["build_packet_and_metrics", "build_request", "candidate_id", "derive_seed", "finalize_xlsx", "integrity_reasons", "load_cases", "load_protocol", "prepare_scoring_and_proxies", "run_generation", "score_granuscore", "score_ko", "score_speciteller"]
