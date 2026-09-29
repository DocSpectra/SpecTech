"""Outcome-blind contract tests for the approved official-release comparator."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.ko_specificity.config import DEFAULT_KO_CONFIG
from src.ko_specificity.official_release import (
    CONFIG_VERSION,
    INPUT_MANIFEST_COLUMNS,
    load_protocol,
    prepare_canonical_inputs,
    read_input_manifest,
)


CONFIG = Path("configs/ko_official_release_comparator_v1.json")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_identity_authorizes_only_separate_official_release_path() -> None:
    protocol = load_protocol(CONFIG)
    runtime = protocol.runtime_config()
    assert protocol.record["immutable_blocker_reference"]["config"] == (
        "configs/ko_2019_reproduction_v1.json"
    )
    assert runtime.project_scoring_authorized is True
    assert runtime.model_id == "ko_author_official_release_se_ad_mean_std"
    assert DEFAULT_KO_CONFIG.project_scoring_authorized is False
    assert DEFAULT_KO_CONFIG.model_id == "ko2019_se_ad_mean_std"


def test_protocol_rejects_changed_three_run_design(tmp_path: Path) -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    record["protocol"]["run_ids"] = ["only-one"]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="three-run"):
        load_protocol(changed)


def test_prepare_freezes_canonical_order_and_first_50(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG)
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    record["protocol"]["canonical_corpus_order"] = ["fixture"]
    fixture_config = tmp_path / "protocol.json"
    fixture_config.write_text(json.dumps(record), encoding="utf-8")
    protocol = load_protocol(fixture_config)
    outputs = tmp_path / "outputs"
    sentences = outputs / "sentences" / "fixture.csv"
    sentences.parent.mkdir(parents=True)
    with sentences.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"])
        for index in range(55):
            writer.writerow(["fixture", "doc", index, f"Canonical row {index}.", f"s{index:03d}"])
    checksums = tmp_path / "checksums.csv"
    with checksums.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["corpus_id", "artifact_kind", "relative_path", "size_bytes", "sha256"])
        writer.writerow(
            [
                "fixture",
                "sentences",
                "outputs/sentences/fixture.csv",
                sentences.stat().st_size,
                _sha(sentences),
            ]
        )
    manifest = tmp_path / "manifest.csv"
    rows = prepare_canonical_inputs(
        protocol,
        outputs_root=outputs,
        artifact_checksums_path=checksums,
        input_dir=outputs / "round2" / "inputs",
        manifest_path=manifest,
    )
    assert rows[0]["canonical_row_count"] == 55
    assert rows[0]["invalid_row_count"] == 0
    with (outputs / "round2" / "inputs" / "fixture.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        prepared = list(csv.DictReader(handle))
    assert [row["sent_id"] for row in prepared] == [f"s{i:03d}" for i in range(55)]
    parsed = read_input_manifest(manifest, protocol)
    assert tuple(next(csv.reader(manifest.open(encoding="utf-8")))) == INPUT_MANIFEST_COLUMNS
    assert parsed["fixture"]["adaptation_context"].startswith(f"{CONFIG_VERSION}:fixture:")


def test_prepare_fails_instead_of_repairing_multiline_text(tmp_path: Path) -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    record["protocol"]["canonical_corpus_order"] = ["fixture"]
    config = tmp_path / "protocol.json"
    config.write_text(json.dumps(record), encoding="utf-8")
    protocol = load_protocol(config)
    outputs = tmp_path / "outputs"
    sentences = outputs / "sentences" / "fixture.csv"
    sentences.parent.mkdir(parents=True)
    with sentences.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus_id", "sent_text", "sent_id"])
        for index in range(50):
            text = "has\nnewline" if index == 2 else f"Row {index}."
            writer.writerow(["fixture", text, f"s{index:03d}"])
    checksums = tmp_path / "checksums.csv"
    checksums.write_text(
        "corpus_id,artifact_kind,relative_path,size_bytes,sha256\n"
        f"fixture,sentences,outputs/sentences/fixture.csv,{sentences.stat().st_size},{_sha(sentences)}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="complete-coverage"):
        prepare_canonical_inputs(
            protocol,
            outputs_root=outputs,
            artifact_checksums_path=checksums,
            input_dir=outputs / "inputs",
            manifest_path=tmp_path / "manifest.csv",
        )
