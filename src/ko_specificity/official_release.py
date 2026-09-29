"""Frozen project-scoring path for the approved official repository release.

This module is intentionally separate from the fail-closed 2019 reproduction
configuration.  It authorizes only the outcome-blind protocol identity in
``configs/ko_official_release_comparator_v1.json``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.analysis.preprocessing_ablation import sha256_file
from src.ko_specificity.config import KoSpecificityConfig
from src.ko_specificity.runner import run_ko_specificity


CONFIG_VERSION = "ko_official_release_comparator_v1"
INPUT_MANIFEST_COLUMNS = (
    "corpus_id",
    "canonical_sentence_path",
    "canonical_sentence_sha256",
    "canonical_sentence_size_bytes",
    "canonical_row_count",
    "valid_row_count",
    "invalid_row_count",
    "input_path",
    "input_sha256",
    "input_size_bytes",
    "ordered_sent_id_sha256",
    "first50_sent_id_text_sha256",
    "adaptation_context",
)
SCORE_COLUMNS = (
    "sent_id",
    "corpus_id",
    "model_id",
    "score_raw",
    "score_min",
    "score_max",
    "score_direction",
    "model_version",
    "adaptation_context",
    "upstream_commit",
    "run_id",
    "training_run_id",
    "input_manifest_sha256",
    "comparator_config_sha256",
)


@dataclass(frozen=True)
class OfficialReleaseProtocol:
    path: Path
    record: dict[str, Any]
    sha256: str

    @property
    def corpora(self) -> tuple[str, ...]:
        return tuple(self.record["protocol"]["canonical_corpus_order"])

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(self.record["protocol"]["run_ids"])

    def runtime_config(self) -> KoSpecificityConfig:
        identity = self.record["identity"]
        runtime = self.record["runtime"]
        scale = self.record["protocol"]["score_scale"]
        return KoSpecificityConfig(
            image=runtime["image"],
            model_id=identity["model_id"],
            model_version=identity["model_version"],
            upstream_commit=identity["upstream_commit"],
            glove_volume=runtime["glove_volume"],
            project_scoring_authorized=True,
            score_min=float(scale["minimum"]),
            score_max=float(scale["maximum"]),
            score_direction=scale["direction"],
        )


def load_protocol(path: Path) -> OfficialReleaseProtocol:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != CONFIG_VERSION:
        raise ValueError("Unexpected official-release comparator config version")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("Comparator config must remain outcome blind")
    if record.get("project_scoring_authorized") is not True:
        raise ValueError("Official-release comparator authorization is absent")
    identity = record.get("identity", {})
    if identity.get("upstream_commit") != "36f8e835e9dc6087d5b6763accf302db175947b1":
        raise ValueError("Upstream comparator identity changed")
    blocker = record.get("immutable_blocker_reference", {})
    if blocker.get("config") != "configs/ko_2019_reproduction_v1.json":
        raise ValueError("Immutable 2019 blocker reference changed")
    if tuple(record["protocol"]["run_ids"]) != ("run01", "run02", "run03"):
        raise ValueError("Frozen three-run protocol changed")
    if record["coverage_gates"]["required_coverage_rate"] != 1.0:
        raise ValueError("Comparator coverage gate must remain complete")
    return OfficialReleaseProtocol(path=path, record=record, sha256=sha256_file(path))


def _update_digest(digest: "hashlib._Hash", *values: str) -> None:
    for value in values:
        digest.update(value.encode("utf-8"))
        digest.update(b"\0")


def prepare_canonical_inputs(
    protocol: OfficialReleaseProtocol,
    *,
    outputs_root: Path,
    artifact_checksums_path: Path,
    input_dir: Path,
    manifest_path: Path,
) -> list[dict[str, str | int]]:
    """Freeze exact valid canonical text and order without reading scores."""
    expected: dict[tuple[str, str], dict[str, str]] = {}
    with artifact_checksums_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            expected[(row["corpus_id"], row["artifact_kind"])] = row
    input_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, str | int]] = []
    for corpus_id in protocol.corpora:
        frozen = expected[(corpus_id, "sentences")]
        sentence_path = outputs_root.parent / frozen["relative_path"]
        if not sentence_path.is_file():
            raise FileNotFoundError(sentence_path)
        if sentence_path.stat().st_size != int(frozen["size_bytes"]):
            raise ValueError(f"Canonical sentence size mismatch: {corpus_id}")
        sentence_sha = sha256_file(sentence_path)
        if sentence_sha != frozen["sha256"]:
            raise ValueError(f"Canonical sentence checksum mismatch: {corpus_id}")
        input_path = input_dir / f"{corpus_id}.csv"
        id_digest = hashlib.sha256()
        first50_digest = hashlib.sha256()
        seen: set[str] = set()
        canonical_count = 0
        invalid: list[str] = []
        with (
            sentence_path.open("r", encoding="utf-8", newline="") as source,
            input_path.open("w", encoding="utf-8", newline="") as target,
        ):
            reader = csv.DictReader(source)
            required = {"corpus_id", "sent_id", "sent_text"}
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"Missing canonical columns for {corpus_id}")
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(["sent_id", "corpus_id", "text"])
            for row_number, row in enumerate(reader, start=2):
                canonical_count += 1
                sent_id = row["sent_id"]
                text = row["sent_text"]
                if row["corpus_id"] != corpus_id:
                    raise ValueError(f"Corpus mismatch at {corpus_id}:{row_number}")
                if not sent_id or sent_id in seen:
                    raise ValueError(f"Missing/duplicate sent_id at {corpus_id}:{row_number}")
                seen.add(sent_id)
                if not text.strip() or "\r" in text or "\n" in text:
                    invalid.append(sent_id)
                    continue
                writer.writerow([sent_id, corpus_id, text])
                _update_digest(id_digest, sent_id)
                if canonical_count <= 50:
                    _update_digest(first50_digest, sent_id, text)
        if invalid:
            raise ValueError(
                f"Frozen complete-coverage gate failed for {corpus_id}: "
                f"{len(invalid)} invalid canonical rows (first={invalid[0]})"
            )
        if canonical_count < int(protocol.record["protocol"]["minimum_valid_rows"]):
            raise ValueError(f"Too few target rows for {corpus_id}")
        input_sha = sha256_file(input_path)
        adaptation_context = (
            f"{CONFIG_VERSION}:{corpus_id}:input-{input_sha[:16]}:per-corpus"
        )
        rows_out.append(
            {
                "corpus_id": corpus_id,
                "canonical_sentence_path": frozen["relative_path"],
                "canonical_sentence_sha256": sentence_sha,
                "canonical_sentence_size_bytes": sentence_path.stat().st_size,
                "canonical_row_count": canonical_count,
                "valid_row_count": canonical_count,
                "invalid_row_count": 0,
                "input_path": f"outputs/round2/ko_official_release/inputs/{corpus_id}.csv",
                "input_sha256": input_sha,
                "input_size_bytes": input_path.stat().st_size,
                "ordered_sent_id_sha256": id_digest.hexdigest(),
                "first50_sent_id_text_sha256": first50_digest.hexdigest(),
                "adaptation_context": adaptation_context,
            }
        )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INPUT_MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    return rows_out


def read_input_manifest(path: Path, protocol: OfficialReleaseProtocol) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or []) != INPUT_MANIFEST_COLUMNS:
            raise ValueError("Official-release input manifest columns changed")
        rows = list(reader)
    if [row["corpus_id"] for row in rows] != list(protocol.corpora):
        raise ValueError("Official-release input manifest corpus order changed")
    for row in rows:
        if row["canonical_row_count"] != row["valid_row_count"] or row["invalid_row_count"] != "0":
            raise ValueError("Complete canonical coverage gate failed")
    return {row["corpus_id"]: row for row in rows}


def _write_enriched_scores(
    legacy_path: Path,
    output_path: Path,
    *,
    protocol: OfficialReleaseProtocol,
    corpus_id: str,
    run_id: str,
    adaptation_context: str,
    input_manifest_sha256: str,
) -> int:
    training_run_id = f"{CONFIG_VERSION}:{corpus_id}:{run_id}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with (
        legacy_path.open("r", encoding="utf-8", newline="") as source,
        output_path.open("w", encoding="utf-8", newline="") as target,
    ):
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=SCORE_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in reader:
            count += 1
            if row["corpus_id"] != corpus_id:
                raise ValueError("Legacy adapter corpus mismatch")
            score = float(row["score_raw"])
            if not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError("Invalid Ko score")
            row.update(
                {
                    "run_id": run_id,
                    "training_run_id": training_run_id,
                    "input_manifest_sha256": input_manifest_sha256,
                    "comparator_config_sha256": protocol.sha256,
                }
            )
            writer.writerow({key: row[key] for key in SCORE_COLUMNS})
    return count


def run_official_release(
    protocol: OfficialReleaseProtocol,
    *,
    corpus_id: str,
    run_id: str,
    input_manifest_path: Path,
    repo_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    if corpus_id not in protocol.corpora or run_id not in protocol.run_ids:
        raise ValueError("Corpus/run is not part of the frozen official-release protocol")
    manifest = read_input_manifest(input_manifest_path, protocol)[corpus_id]
    input_path = repo_root / manifest["input_path"]
    if sha256_file(input_path) != manifest["input_sha256"]:
        raise ValueError("Prepared Ko input checksum mismatch")
    run_dir = run_root / corpus_id / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = run_dir / "legacy_scores.csv"
    final_path = run_dir / "scores.csv"
    evidence_dir = run_dir / "evidence"
    recovered_completed_run = (
        final_path.is_file()
        and (evidence_dir / "run_metadata.json").is_file()
        and (evidence_dir / "model.pickle").is_file()
    )
    if recovered_completed_run:
        command = [
            "completed frozen Docker run finalized idempotently from retained evidence"
        ]
        with final_path.open("r", encoding="utf-8", newline="") as handle:
            row_count = sum(1 for _ in csv.DictReader(handle))
    else:
        command = run_ko_specificity(
            protocol.runtime_config(),
            input_path,
            legacy_path,
            manifest["adaptation_context"],
            keep_run_dir=run_dir,
        )
        row_count = _write_enriched_scores(
            legacy_path,
            final_path,
            protocol=protocol,
            corpus_id=corpus_id,
            run_id=run_id,
            adaptation_context=manifest["adaptation_context"],
            input_manifest_sha256=manifest["input_sha256"],
        )
    expected = int(manifest["valid_row_count"])
    if row_count != expected:
        raise ValueError(f"Ko score coverage mismatch: {row_count} != {expected}")
    container_metadata_path = run_dir / "evidence" / "run_metadata.json"
    container_metadata = json.loads(container_metadata_path.read_text(encoding="utf-8"))
    if container_metadata["prediction_count"] != expected:
        raise ValueError("Container prediction coverage mismatch")
    checkpoint_path = run_dir / "evidence" / "model.pickle"
    if not checkpoint_path.is_file():
        raise ValueError("Comparator checkpoint was not retained")
    evidence_artifacts = {}
    for name in (
        "predictions.txt",
        "model.pickle",
        "train.log",
        "test.log",
        "run_metadata.json",
        "model.sha256",
    ):
        path = evidence_dir / name
        if not path.is_file():
            raise ValueError(f"Missing retained run evidence: {name}")
        evidence_artifacts[name] = {
            "path": str(path.resolve().relative_to(repo_root)).replace("\\", "/"),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    metadata = {
        "schema_version": "ko_official_release_run_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_id": corpus_id,
        "run_id": run_id,
        "training_run_id": f"{CONFIG_VERSION}:{corpus_id}:{run_id}",
        "adaptation_context": manifest["adaptation_context"],
        "row_count": row_count,
        "input_path": manifest["input_path"],
        "input_sha256": manifest["input_sha256"],
        "score_path": str(final_path.resolve().relative_to(repo_root)).replace("\\", "/"),
        "score_sha256": sha256_file(final_path),
        "prediction_sha256": container_metadata["prediction_sha256"],
        "teacher_model_sha256": container_metadata["teacher_model_sha256"],
        "checkpoint_sha256_verified": sha256_file(checkpoint_path)
        == container_metadata["teacher_model_sha256"],
        "comparator_config_sha256": protocol.sha256,
        "input_manifest_file_sha256": sha256_file(input_manifest_path),
        "upstream_commit": protocol.record["identity"]["upstream_commit"],
        "model_id": protocol.record["identity"]["model_id"],
        "model_version": protocol.record["identity"]["model_version"],
        "container": container_metadata,
        "evidence_artifacts": evidence_artifacts,
        "command": command,
        "host_finalization_recovered": recovered_completed_run,
        "redistribution": "do not redistribute upstream checkpoint, image, data, or source absent permission",
    }
    metadata_path = run_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if legacy_path.exists():
        legacy_path.unlink()
    return metadata


def copy_compact_run_metadata(run_root: Path, output_path: Path, protocol: OfficialReleaseProtocol) -> None:
    records = []
    for corpus_id in protocol.corpora:
        for run_id in protocol.run_ids:
            path = run_root / corpus_id / run_id / "run_metadata.json"
            records.append(json.loads(path.read_text(encoding="utf-8")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8")
