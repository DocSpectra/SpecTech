"""Content-blind scoring and packet construction for returned GPT-OSS-120B edits."""
from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import math
import subprocess
import tarfile
from collections import Counter, defaultdict
from decimal import Decimal
from pathlib import Path
from typing import Any

from src.analysis.gemma_controlled_editor import gate_edit as legacy_proxy
from src.analysis.human_first_reranking import integrity_reasons
from src.analysis.pilot_model_human import sha256_file
from src.analysis.qwen_edit_source import _docker_image_id
from src.ko_specificity.io import parse_predictions
from src.speciteller.config import DEFAULT_SPECITELLER_CONFIG
from src.speciteller.runner import preflight_speciteller, run_speciteller
from src.speciteller.tokenize import tokenize_for_speciteller


ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parents[1]
CONFIG = Path("configs/round2_dgx_gptoss120b_human_first_v1.json")
FREEZE = Path("configs/round2_dgx_gptoss120b_human_first_freeze_record.json")
SCHEMA_VERSION = "round2_dgx_gptoss120b_human_first_v1"


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch for {path}: {observed} != {expected}")


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


def _archive_member(archive_path: Path, member_path: str) -> bytes:
    with tarfile.open(archive_path, "r:gz") as archive:
        member = archive.getmember(member_path)
        if not member.isfile():
            raise ValueError("returned candidate member is not a regular file")
        handle = archive.extractfile(member)
        if handle is None:
            raise ValueError("returned candidate member cannot be read")
        return handle.read()


def _base_cases(record: dict[str, Any]) -> list[Any]:
    inherited = record["inherited_protocol"]
    path = Path(inherited["path"])
    _require_hash(path, inherited["sha256"])
    from src.analysis.human_first_reranking import load_cases

    protocol = json.loads(path.read_text(encoding="utf-8"))
    cases = load_cases(protocol)
    source_serialization = "".join(f"{case.case_id}\0{case.original_sha256}\n" for case in cases)
    if len(cases) != inherited["source_cases"]:
        raise ValueError("source case count mismatch")
    if _sha_text(source_serialization) != inherited["ordered_source_hashes_sha256"]:
        raise ValueError("ordered source/hash binding mismatch")
    return cases


def load_method(config_path: Path = CONFIG, *, require_freeze: bool = True) -> tuple[dict[str, Any], dict[str, Any] | None]:
    record = json.loads(config_path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION or record.get("outcome_blind_method_freeze") is not True:
        raise ValueError("unexpected or unfrozen GPT-OSS-120B packet protocol")
    for entry in (
        record["authorization"]["provenance_config"],
        record["authorization"]["acceptance_summary"],
        record["inherited_protocol"],
        record["scoring"]["identity_source"],
        record["automatic_proxy_audit"]["definition_source"],
    ):
        _require_hash(Path(entry["path"]), entry["sha256"])
    for source in record["inherited_protocol"]["source_files"].values():
        _require_hash(Path(source["path"]), source["sha256"])
    if not require_freeze:
        return record, None
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    if freeze.get("schema_version") != "round2_dgx_gptoss120b_human_first_freeze_record_v1":
        raise ValueError("unexpected packet freeze record")
    _require_hash(config_path, freeze["method_config"]["sha256"])
    if freeze["chronology"]["scorer_outcomes_before_method_commit"] != 0:
        raise ValueError("invalid scoring chronology")
    return record, freeze


def local_alias(record: dict[str, Any], remote_id: str, case_id: str, slot: int) -> str:
    namespace = record["local_alias"]["namespace"]
    return _sha_text(f"{namespace}\0{remote_id}\0{case_id}\0{slot}")


def content_blind_binding(config_path: Path = CONFIG, *, require_freeze: bool = False) -> dict[str, Any]:
    record, _ = load_method(config_path, require_freeze=require_freeze)
    returned = record["returned_candidates"]
    archive = WORKSPACE / returned["archive_workspace_path"]
    _require_hash(archive, returned["archive_sha256"])
    data = _archive_member(archive, returned["member_path"])
    if len(data) != returned["member_bytes"] or _sha_bytes(data) != returned["member_sha256"]:
        raise ValueError("returned candidate member binding mismatch")
    rows = list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))
    cases = _base_cases(record)
    case_map = {case.case_id: case for case in cases}
    expected_pairs = [(case.case_id, slot) for case in cases for slot in (1, 2, 3)]
    observed_pairs: list[tuple[str, int]] = []
    aliases: list[str] = []
    structure: list[str] = []
    alias_records: list[str] = []
    remote_ids: list[str] = []
    full_integrity_failures = 0
    for row in rows:
        slot = int(row["candidate_slot"])
        case = case_map.get(row["case_id"])
        if case is None or row["source_text_sha256"] != case.original_sha256:
            raise ValueError("returned source join/hash mismatch")
        if row["candidate_id"] != _sha_text(f"{row['case_id']}\0{slot}"):
            raise ValueError("remote candidate ID mismatch")
        if _sha_text(row["candidate_text"]) != row["candidate_text_sha256"]:
            raise ValueError("returned candidate text hash mismatch")
        if row["run_id"] != returned["run_id"] or row["model_identity"] != returned["model_identity"]:
            raise ValueError("returned run/model mismatch")
        if row["config_sha256"] != returned["generation_config_sha256"] or row["freeze_id"] != returned["generation_freeze_id"]:
            raise ValueError("returned generation config/freeze mismatch")
        if int(row["retained_attempt"]) != returned["required_retained_attempt"]:
            raise ValueError("returned retained-attempt mismatch")
        full_integrity_failures += int(bool(integrity_reasons(case.sentence_original, row["candidate_text"])))
        alias = local_alias(record, row["candidate_id"], row["case_id"], slot)
        aliases.append(alias)
        remote_ids.append(row["candidate_id"])
        observed_pairs.append((row["case_id"], slot))
        structure.append(f"{row['case_id']}\0{slot}\0{row['candidate_id']}\0{row['source_text_sha256']}\0{row['candidate_text_sha256']}\n")
        alias_records.append(f"{row['candidate_id']}\0{alias}\n")
    if observed_pairs != expected_pairs:
        raise ValueError("returned case/slot order mismatch")
    if len(rows) != returned["expected_rows"] or len(set(remote_ids)) != returned["expected_unique_remote_ids"]:
        raise ValueError("returned candidate coverage/uniqueness mismatch")
    if len(set(aliases)) != record["local_alias"]["expected_unique_aliases"]:
        raise ValueError("local alias bijection mismatch")
    if _sha_text("".join(structure)) != returned["candidate_structure_sha256"]:
        raise ValueError("candidate structure digest mismatch")
    if _sha_text("".join(alias_records)) != record["local_alias"]["remote_alias_bijection_sha256"]:
        raise ValueError("remote/local alias digest mismatch")
    if full_integrity_failures:
        raise ValueError("complete local integrity gate failed")
    cells = Counter((case_map[row["case_id"]].corpus_id, case_map[row["case_id"]].edit_type) for row in rows)
    observed_cells = {f"{a}:{b}": value for (a, b), value in sorted(cells.items())}
    if observed_cells != returned["hidden_cell_candidate_counts"]:
        raise ValueError("hidden-cell population mismatch")
    return {
        "archive_sha256": returned["archive_sha256"],
        "member_sha256": returned["member_sha256"],
        "source_cases": len(cases),
        "candidate_rows": len(rows),
        "unique_remote_ids": len(set(remote_ids)),
        "unique_local_aliases": len(set(aliases)),
        "candidate_structure_sha256": returned["candidate_structure_sha256"],
        "remote_alias_bijection_sha256": record["local_alias"]["remote_alias_bijection_sha256"],
        "complete_local_integrity_failures": full_integrity_failures,
        "hidden_cell_candidate_counts": observed_cells,
    }


def _load_candidates(record: dict[str, Any]) -> list[dict[str, str]]:
    returned = record["returned_candidates"]
    data = _archive_member(WORKSPACE / returned["archive_workspace_path"], returned["member_path"])
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def _scorer_config(record: dict[str, Any]) -> dict[str, Any]:
    entry = record["scoring"]["identity_source"]
    _require_hash(Path(entry["path"]), entry["sha256"])
    return json.loads(Path(entry["path"]).read_text(encoding="utf-8"))


def prepare(config_path: Path = CONFIG) -> dict[str, Any]:
    record, freeze = load_method(config_path)
    binding = content_blind_binding(config_path, require_freeze=True)
    cases = _base_cases(record)
    case_map = {case.case_id: case for case in cases}
    returned_rows = _load_candidates(record)
    raw = Path(record["outputs"]["raw_directory"])
    raw.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for row in returned_rows:
        case = case_map[row["case_id"]]
        slot = int(row["candidate_slot"])
        candidates.append({
            "case_id": case.case_id,
            "corpus_id": case.corpus_id,
            "edit_type": case.edit_type,
            "candidate_slot": slot,
            "remote_candidate_id": row["candidate_id"],
            "candidate_id": local_alias(record, row["candidate_id"], case.case_id, slot),
            "candidate_sha256": row["candidate_text_sha256"],
            "sentence_candidate": row["candidate_text"],
        })
    _write_csv(raw / "retained_candidates.csv", candidates)
    base = json.loads(Path(record["inherited_protocol"]["path"]).read_text(encoding="utf-8"))
    proxy_base = json.loads(Path(base["inputs"]["base_protocol"]["path"]).read_text(encoding="utf-8"))
    proxy_rows: list[dict[str, Any]] = []
    for row in candidates:
        case = case_map[row["case_id"]]
        reasons, metrics = legacy_proxy(proxy_base, case.sentence_original, row["sentence_candidate"], case.edit_type)
        proxy_rows.append({"candidate_id": row["candidate_id"], "proxy_pass": str(not reasons).lower(), "reason_codes": "|".join(reasons), **metrics})
    _write_csv(raw / "automatic_proxy_audit.csv", proxy_rows)
    score_rows: list[dict[str, Any]] = []
    for case in cases:
        score_rows.append({"score_id": _sha_text(f"{case.case_id}\0original"), "case_id": case.case_id, "candidate_id": "", "corpus_id": case.corpus_id, "role": "original", "candidate_slot": "", "text_sha256": case.original_sha256, "text": case.sentence_original})
        for row in (item for item in candidates if item["case_id"] == case.case_id):
            score_rows.append({"score_id": _sha_text(f"{row['candidate_id']}\0candidate"), "case_id": case.case_id, "candidate_id": row["candidate_id"], "corpus_id": case.corpus_id, "role": "candidate", "candidate_slot": row["candidate_slot"], "text_sha256": row["candidate_sha256"], "text": row["sentence_candidate"]})
    _write_csv(raw / "scoring_manifest.csv", [{key: value for key, value in row.items() if key != "text"} for row in score_rows])
    spec = raw / "scoring" / "speciteller"
    spec.mkdir(parents=True, exist_ok=True)
    with (spec / "input.tsv").open("w", encoding="utf-8", newline="\n") as handle:
        for row in score_rows:
            handle.write(f"{row['score_id']}\t{tokenize_for_speciteller(row['text'])}\n")
    gs = raw / "scoring" / "granuscore"
    gs.mkdir(parents=True, exist_ok=True)
    with (gs / "input.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corpus_id", "sent_id", "sent_text"], lineterminator="\n")
        writer.writeheader()
        writer.writerows({"corpus_id": "dgx_gptoss120b_human_first", "sent_id": row["score_id"], "sent_text": row["text"]} for row in score_rows)
    for corpus in record["inherited_protocol"]["corpus_order"]:
        corpus_rows = [row for row in score_rows if row["corpus_id"] == corpus]
        bundle = raw / "scoring" / "ko" / corpus / "bundle"
        bundle.mkdir(parents=True, exist_ok=True)
        texts = [row["text"] for row in corpus_rows]
        (bundle / "twitters.txt").write_text("\n".join([texts[0], *texts]) + "\n", encoding="utf-8")
        (bundle / "twitterl.txt").write_text("1\n" * (len(texts) + 1), encoding="utf-8")
        (bundle / "twitterv.txt").write_text("0.5\n" * (len(texts) + 1), encoding="utf-8")
        target = _read_csv(Path("outputs/round2/ko_official_release/inputs") / f"{corpus}.csv")
        (bundle / "twitteru.txt").write_text("\n".join(row["text"] for row in target) + "\n", encoding="utf-8")
        _write_csv(bundle / "row_map.csv", [{"prediction_index": index, "score_id": row["score_id"]} for index, row in enumerate(corpus_rows)])
    metadata = {
        "schema_version": "round2_dgx_gptoss120b_scoring_preparation_v1",
        "method_config_sha256": sha256_file(config_path),
        "method_freeze_commit": freeze["method_freeze_commit"],
        "source_cases": 60,
        "candidate_rows": 180,
        "score_rows": len(score_rows),
        "proxy_rows": len(proxy_rows),
        "scoring_manifest_sha256": sha256_file(raw / "scoring_manifest.csv"),
        "proxy_sha256": sha256_file(raw / "automatic_proxy_audit.csv"),
        "candidate_binding": binding,
    }
    (raw / "scoring" / "preparation_metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return metadata


def score_speciteller(config_path: Path = CONFIG) -> dict[str, Any]:
    record, _ = load_method(config_path)
    scorer = _scorer_config(record)["scoring"]["speciteller"]
    if _docker_image_id(scorer["image"]) != scorer["image_id"]:
        raise ValueError("SpeciTeller image mismatch")
    directory = Path(record["outputs"]["raw_directory"]) / "scoring" / "speciteller"
    preflight_speciteller(DEFAULT_SPECITELLER_CONFIG)
    run_speciteller(DEFAULT_SPECITELLER_CONFIG, directory / "input.tsv", directory / "scores.tsv")
    rows = [line for line in (directory / "scores.tsv").read_text(encoding="utf-8").splitlines() if line]
    values = [float(line.split("\t")[1]) for line in rows]
    if len(rows) != 240 or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("SpeciTeller coverage/range mismatch")
    return {"rows": len(rows), "sha256": sha256_file(directory / "scores.tsv")}


def score_granuscore(config_path: Path = CONFIG) -> dict[str, Any]:
    record, _ = load_method(config_path)
    scorer = _scorer_config(record)["scoring"]["granuscore"]
    if _docker_image_id(scorer["image"]) != scorer["image_id"]:
        raise ValueError("GranuScore image mismatch")
    directory = Path(record["outputs"]["raw_directory"]) / "scoring" / "granuscore"
    inp = directory / "input.csv"
    output = directory / "scores.csv"
    metadata = directory / "scores.metadata.json"
    cmd = ["docker", "run", "--rm", "--gpus", "all", "-v", f"{Path.cwd().resolve()}:/work", "-w", "/work", scorer["image"], "--input", f"/work/{inp.as_posix()}", "--output", f"/work/{output.as_posix()}", "--metadata", f"/work/{metadata.as_posix()}", "--corpus-id", "dgx_gptoss120b_human_first", "--expected-input-sha256", sha256_file(inp), "--batch-size", "256", "--encoding-batch-size", "256"]
    subprocess.run(cmd, check=True)
    meta = json.loads(metadata.read_text(encoding="utf-8"))
    rows = _read_csv(output)
    values = [float(row["granuscore_percentile"]) for row in rows]
    if len(rows) != 240 or meta["row_count"] != 240 or meta["runner_sha256"] != scorer["runner_sha256"] or any(not math.isfinite(value) or not 0.0 <= value <= 100.0 for value in values):
        raise ValueError("GranuScore identity/coverage/range mismatch")
    return {"rows": 240, "sha256": meta["output_sha256"]}


def score_ko(config_path: Path, corpus: str, run_id: str) -> dict[str, Any]:
    record, _ = load_method(config_path)
    ko = _scorer_config(record)["scoring"]["ko"]
    if _docker_image_id(ko["image"]) != ko["image_id"]:
        raise ValueError("Ko image mismatch")
    run = ko["runs"][corpus][run_id]
    checkpoint = Path(run["checkpoint_path"])
    _require_hash(checkpoint, run["checkpoint_sha256"])
    base = Path(record["outputs"]["raw_directory"]) / "scoring" / "ko" / corpus
    bundle = base / "bundle"
    output = base / run_id
    output.mkdir(parents=True, exist_ok=True)
    shell = "set -euo pipefail; test \"$(sha256sum /artifacts/glove.840B.300d.txt | cut -d' ' -f1)\" = \"$GLOVE_TXT_SHA256\"; cp -a /opt/ko /tmp/ko; cd /tmp/ko; ln -s /artifacts/glove.840B.300d.txt glove.840B.300d.txt; cp /target/twitters.txt /target/twitteru.txt /target/twitterl.txt /target/twitterv.txt dataset/data/; cp /checkpoint/model.pickle savedir/3osmodel.pickle; python test.py --gpu_id 0 --test_data twitter > /output/test.log 2>&1; cp predictions.txt /output/predictions.txt"
    cmd = ["docker", "run", "--rm", "-e", f"GLOVE_TXT_SHA256={ko['glove_text_sha256']}", "-v", f"{bundle.resolve()}:/target:ro", "-v", f"{output.resolve()}:/output", "-v", f"{checkpoint.resolve()}:/checkpoint/model.pickle:ro", "-v", f"{ko['glove_volume']}:/artifacts:ro", "--entrypoint", "bash", ko["image"], "-lc", shell]
    subprocess.run(cmd, check=True)
    mapping = _read_csv(bundle / "row_map.csv")
    scores = parse_predictions(output / "predictions.txt", len(mapping))
    if len(scores) != 120 or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores):
        raise ValueError("Ko coverage/range mismatch")
    rows = [{"score_id": row["score_id"], "corpus_id": corpus, "run_id": run_id, "score_raw": repr(score), "checkpoint_sha256": run["checkpoint_sha256"]} for row, score in zip(mapping, scores)]
    _write_csv(output / "scores.csv", rows)
    meta = {"row_count": len(rows), "checkpoint_sha256": run["checkpoint_sha256"], "prediction_sha256": sha256_file(output / "predictions.txt"), "score_sha256": sha256_file(output / "scores.csv")}
    (output / "scores.metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"rows": len(rows), "sha256": meta["score_sha256"]}


def _score_map(path: Path, key: str, value: str) -> tuple[dict[str, float], dict[str, str]]:
    rows = _read_csv(path)
    floats = {row[key]: float(row[value]) for row in rows}
    strings = {row[key]: row[value] for row in rows}
    if len(floats) != len(rows) or any(not math.isfinite(item) for item in floats.values()):
        raise ValueError(f"duplicate or nonfinite score rows: {path}")
    return floats, strings


def _review_id(record: dict[str, Any], alias: str) -> str:
    blind = record["blinding"]
    digest = hashlib.sha256(f"{blind['review_id_seed']}:{blind['review_id_namespace']}:{alias}".encode("utf-8")).digest()
    return "DHF-" + base64.b32encode(digest).decode("ascii").rstrip("=")[:12]


def _sort_hash(seed: int, value: str) -> str:
    return _sha_text(f"{seed}:{value}")


def build_packet(config_path: Path = CONFIG) -> dict[str, Any]:
    record, freeze = load_method(config_path)
    raw = Path(record["outputs"]["raw_directory"])
    cases = _base_cases(record)
    case_map = {case.case_id: case for case in cases}
    candidates = _read_csv(raw / "retained_candidates.csv")
    manifest = _read_csv(raw / "scoring_manifest.csv")
    if len(candidates) != 180 or len(manifest) != 240:
        raise ValueError("candidate/scoring manifest incomplete")
    score_ids = {row["score_id"] for row in manifest}
    spec: dict[str, float] = {}
    for line in (raw / "scoring/speciteller/scores.tsv").read_text(encoding="utf-8").splitlines():
        if line.strip():
            score_id, value = line.split("\t")[:2]
            spec[score_id] = float(value)
    gs_rows = _read_csv(raw / "scoring/granuscore/scores.csv")
    gs = {row["sent_id"]: float(row["granuscore_percentile"]) for row in gs_rows}
    ko_float: dict[str, dict[str, float]] = {}
    ko_text: dict[str, dict[str, str]] = {}
    for corpus in record["inherited_protocol"]["corpus_order"]:
        for run in ("run01", "run02", "run03"):
            key = f"{corpus}:{run}"
            ko_float[key], ko_text[key] = _score_map(raw / f"scoring/ko/{corpus}/{run}/scores.csv", "score_id", "score_raw")
    if set(spec) != score_ids or set(gs) != score_ids:
        raise ValueError("global scorer coverage mismatch")
    if any(not 0.0 <= value <= 1.0 for value in spec.values()) or any(not 0.0 <= value <= 100.0 for value in gs.values()):
        raise ValueError("global scorer range mismatch")
    for corpus in record["inherited_protocol"]["corpus_order"]:
        expected = {row["score_id"] for row in manifest if row["corpus_id"] == corpus}
        for run in ("run01", "run02", "run03"):
            values = ko_float[f"{corpus}:{run}"]
            if set(values) != expected or any(not 0.0 <= value <= 1.0 for value in values.values()):
                raise ValueError("Ko coverage/range mismatch")
    original_score_id = {row["case_id"]: row["score_id"] for row in manifest if row["role"] == "original"}
    candidate_score_id = {row["candidate_id"]: row["score_id"] for row in manifest if row["role"] == "candidate"}
    metric_rows: list[dict[str, Any]] = []
    for row in candidates:
        case = case_map[row["case_id"]]
        osid = original_score_id[case.case_id]
        csid = candidate_score_id[row["candidate_id"]]
        values: dict[str, Any] = {
            "speciteller_frozen_round1": spec[csid] - spec[osid],
            "ko_run01": ko_float[f"{case.corpus_id}:run01"][csid] - ko_float[f"{case.corpus_id}:run01"][osid],
            "ko_run02": ko_float[f"{case.corpus_id}:run02"][csid] - ko_float[f"{case.corpus_id}:run02"][osid],
            "ko_run03": ko_float[f"{case.corpus_id}:run03"][csid] - ko_float[f"{case.corpus_id}:run03"][osid],
            "granuscore_direction_aligned": -(gs[csid] - gs[osid]),
        }
        candidate_mean = sum((Decimal(ko_text[f"{case.corpus_id}:run0{i}"][csid]) for i in (1, 2, 3)), Decimal(0)) / Decimal(3)
        original_mean = sum((Decimal(ko_text[f"{case.corpus_id}:run0{i}"][osid]) for i in (1, 2, 3)), Decimal(0)) / Decimal(3)
        values["ko_three_run_arithmetic_mean_secondary"] = float(candidate_mean - original_mean)
        metric_rows.append({"candidate_id": row["candidate_id"], "case_id": case.case_id, "candidate_slot": int(row["candidate_slot"]), "edit_type": case.edit_type, **values, "granuscore_native_original": gs[osid], "granuscore_native_candidate": gs[csid]})
    _write_csv(raw / "candidate_metric_deltas.csv", metric_rows)
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        by_case[row["case_id"]].append(row)
    primary = ["speciteller_frozen_round1", "ko_run01", "ko_run02", "ko_run03", "granuscore_direction_aligned"]
    policy_rows: list[dict[str, Any]] = []
    for case in cases:
        pool = sorted(by_case[case.case_id], key=lambda row: row["candidate_slot"])
        chosen: dict[str, int] = {"unguided_slot01": 1}
        for metric in [*primary, "ko_three_run_arithmetic_mean_secondary"]:
            if case.edit_type == "add_specific":
                key = lambda row, metric=metric: (-row[metric], row["candidate_slot"])
            elif case.edit_type == "de_specify":
                key = lambda row, metric=metric: (row[metric], row["candidate_slot"])
            else:
                key = lambda row, metric=metric: (abs(row[metric]), row["candidate_slot"])
            chosen[metric] = sorted(pool, key=key)[0]["candidate_slot"]
        rank_sums: Counter[int] = Counter()
        for metric in primary:
            if case.edit_type == "add_specific":
                ordered = sorted(pool, key=lambda row, metric=metric: (-row[metric], row["candidate_slot"]))
            elif case.edit_type == "de_specify":
                ordered = sorted(pool, key=lambda row, metric=metric: (row[metric], row["candidate_slot"]))
            else:
                ordered = sorted(pool, key=lambda row, metric=metric: (abs(row[metric]), row["candidate_slot"]))
            for rank, item in enumerate(ordered, 1):
                rank_sums[item["candidate_slot"]] += rank
        chosen["rank_consensus_secondary"] = min(rank_sums, key=lambda slot: (rank_sums[slot], slot))
        pool_ids = {int(row["candidate_slot"]): row["candidate_id"] for row in pool}
        for policy, slot in chosen.items():
            policy_rows.append({"case_id": case.case_id, "policy": policy, "selected_candidate_slot": slot, "selected_candidate_id": pool_ids[slot]})
    _write_csv(raw / "policy_selections.csv", policy_rows)
    cells: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        cells[(row["corpus_id"], row["edit_type"])].append(row)
    sides: dict[str, str] = {}
    for cell, rows in cells.items():
        ordered = sorted(rows, key=lambda row: _sort_hash(record["blinding"]["side_seed"], row["candidate_id"]))
        if len(ordered) % 2:
            raise ValueError(f"odd hidden cell: {cell}")
        for index, row in enumerate(ordered):
            sides[row["candidate_id"]] = "A" if index < len(ordered) // 2 else "B"
    proxy = {row["candidate_id"]: row for row in _read_csv(raw / "automatic_proxy_audit.csv")}
    metrics = {row["candidate_id"]: row for row in metric_rows}
    selected: dict[str, list[str]] = defaultdict(list)
    for row in policy_rows:
        selected[row["selected_candidate_id"]].append(row["policy"])
    packet: list[dict[str, str]] = []
    key_rows: list[dict[str, Any]] = []
    for row in candidates:
        case = case_map[row["case_id"]]
        side = sides[row["candidate_id"]]
        review_id = _review_id(record, row["candidate_id"])
        a = row["sentence_candidate"] if side == "A" else case.sentence_original
        b = case.sentence_original if side == "A" else row["sentence_candidate"]
        packet.append({"Review ID": review_id, "A": a, "B": b, "Score": ""})
        key_rows.append({**row, "review_id": review_id, "candidate_side": side, "sentence_a_sha256": _sha_text(a), "sentence_b_sha256": _sha_text(b), "proxy_pass": proxy[row["candidate_id"]]["proxy_pass"], "proxy_reason_codes": proxy[row["candidate_id"]]["reason_codes"], **{f"metric_{key}": value for key, value in metrics[row["candidate_id"]].items() if key not in {"candidate_id", "case_id", "candidate_slot", "edit_type"}}, "selected_by_policies": "|".join(sorted(selected[row["candidate_id"]]))})
    packet.sort(key=lambda row: _sort_hash(record["blinding"]["row_order_seed"], row["Review ID"]))
    _write_csv(raw / record["outputs"]["review_csv"], packet, ["Review ID", "A", "B", "Score"])
    _write_csv(raw / "private_review_key.csv", key_rows)
    if len({row["Review ID"] for row in packet}) != 180 or len({row["review_id"] for row in key_rows}) != 180:
        raise ValueError("review ID bijection failed")
    if Counter(sides.values()) != Counter({"A": 90, "B": 90}):
        raise ValueError("overall side balance failed")
    for rows in cells.values():
        if Counter(sides[row["candidate_id"]] for row in rows) != Counter({"A": len(rows) // 2, "B": len(rows) // 2}):
            raise ValueError("hidden-cell side balance failed")
    manifest_out = {
        "schema_version": "round2_dgx_gptoss120b_packet_manifest_v1",
        "method_config_sha256": sha256_file(config_path),
        "method_freeze_commit": freeze["method_freeze_commit"],
        "run_id": record["returned_candidates"]["run_id"],
        "provenance_disposition": record["authorization"]["provenance_disposition"],
        "source_cases": 60,
        "candidate_rows": 180,
        "review_rows": 180,
        "score_manifest_rows": 240,
        "metric_rows": len(metric_rows),
        "policy_rows": len(policy_rows),
        "proxy_rows": len(proxy),
        "candidate_side_counts": dict(Counter(sides.values())),
        "hidden_cell_candidate_counts": {f"{a}:{b}": len(rows) for (a, b), rows in sorted(cells.items())},
        "hidden_cell_side_counts": {f"{a}:{b}": dict(Counter(sides[row["candidate_id"]] for row in rows)) for (a, b), rows in sorted(cells.items())},
        "review_csv_sha256": sha256_file(raw / record["outputs"]["review_csv"]),
        "private_key_sha256": sha256_file(raw / "private_review_key.csv"),
        "metric_sha256": sha256_file(raw / "candidate_metric_deltas.csv"),
        "policy_sha256": sha256_file(raw / "policy_selections.csv"),
        "proxy_sha256": sha256_file(raw / "automatic_proxy_audit.csv"),
        "xlsx_pending": True,
        "release_gate_passed": False,
    }
    (raw / "packet_manifest_pre_xlsx.json").write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_out


def finalize(config_path: Path = CONFIG) -> dict[str, Any]:
    record, _ = load_method(config_path)
    raw = Path(record["outputs"]["raw_directory"])
    pre = json.loads((raw / "packet_manifest_pre_xlsx.json").read_text(encoding="utf-8"))
    xlsx = raw / record["outputs"]["review_xlsx"]
    if not xlsx.exists():
        raise ValueError("XLSX missing")
    pre["review_xlsx_sha256"] = sha256_file(xlsx)
    pre["review_xlsx_bytes"] = xlsx.stat().st_size
    pre["xlsx_pending"] = False
    pre["release_gate_passed"] = True
    (raw / "packet_manifest.json").write_text(json.dumps(pre, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    compact = Path(record["outputs"]["compact_directory"])
    compact.mkdir(parents=True, exist_ok=True)
    candidates = _read_csv(raw / "retained_candidates.csv")
    coverage = []
    for (corpus, direction), count in sorted(Counter((row["corpus_id"], row["edit_type"]) for row in candidates).items()):
        coverage.append({"corpus_id": corpus, "edit_type": direction, "planned_candidates": count, "retained_candidates": count, "complete": "true"})
    _write_csv(compact / "generation_coverage.csv", coverage)
    proxy_rows = _read_csv(raw / "automatic_proxy_audit.csv")
    lookup = {row["candidate_id"]: row for row in candidates}
    proxy_aggregate = []
    for (corpus, direction, passed), count in sorted(Counter((lookup[row["candidate_id"]]["corpus_id"], lookup[row["candidate_id"]]["edit_type"], row["proxy_pass"]) for row in proxy_rows).items()):
        proxy_aggregate.append({"corpus_id": corpus, "edit_type": direction, "proxy_pass": passed, "candidate_count": count})
    _write_csv(compact / "proxy_aggregate.csv", proxy_aggregate)
    compact_manifest = {key: value for key, value in pre.items() if key != "private_key_sha256"}
    compact_manifest["private_key_sha256_recorded_only_in_ignored_manifest"] = True
    (compact / "packet_manifest.json").write_text(json.dumps(compact_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (compact / "README.md").write_text(
        "# GPT-OSS-120B human-first review packet\n\n"
        "The accepted returned set contains 180/180 candidates for 60 exact sources. "
        "All originals and candidates received complete frozen SpeciTeller, three-run Ko, and native GranuScore scoring. "
        "The prior automatic proxy was computed for audit only. Frozen selector mappings and a reversible key remain ignored. "
        "The released packet contains only Review ID, A, B, and blank Score; no human outcome has been analyzed.\n",
        encoding="utf-8",
    )
    return pre


__all__ = ["build_packet", "content_blind_binding", "finalize", "load_method", "local_alias", "prepare", "score_granuscore", "score_ko", "score_speciteller"]
