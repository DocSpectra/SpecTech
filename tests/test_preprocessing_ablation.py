"""Focused tests for the Round 2 preprocessing-sensitivity analysis."""
from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path
import re

import numpy as np
import pytest

from src.analysis.preprocessing_ablation import BaselineExpectation
from src.analysis.preprocessing_ablation import CorpusArtifacts
from src.analysis.preprocessing_ablation import CorpusData
from src.analysis.preprocessing_ablation import CORPUS_ORDER
from src.analysis.preprocessing_ablation import RunSettings
from src.analysis.preprocessing_ablation import cluster_bootstrap_statistics
from src.analysis.preprocessing_ablation import run_preprocessing_ablation
from src.analysis.preprocessing_ablation import score_summary
from src.analysis.preprocessing_ablation import sha256_file
from src.analysis.preprocessing_ablation import validate_and_load_corpus
from src.analysis.preprocessing_ablation import validate_artifact_checksums
from src.analysis.preprocessing_ablation import validate_published_baseline


def _write_fixture_artifacts(root: Path, *, mismatch_feature_id: bool = False) -> CorpusArtifacts:
    sentence_path = root / "sentences.csv"
    score_path = root / "scores.tsv"
    feature_path = root / "features.csv"
    sentence_path.write_text(
        "corpus_id,doc_path,sent_idx,sent_text,sent_id\n"
        "demo,doc-a,0,This complete sentence remains in the analysis.,s1\n"
        "demo,doc-b,0,Too short now,s2\n",
        encoding="utf-8",
    )
    score_path.write_text("s1\t0.25\ns2\t0.75\n", encoding="utf-8")
    second_feature_id = "wrong" if mismatch_feature_id else "s2"
    feature_path.write_text(
        "corpus_id,sent_id,token_count,char_count\n"
        f"demo,s1,7,47\ndemo,{second_feature_id},3,13\n",
        encoding="utf-8",
    )
    return CorpusArtifacts(sentence_path, score_path, feature_path)


def _expectation() -> BaselineExpectation:
    return BaselineExpectation(
        corpus_id="demo",
        display_name="Demo",
        expected_sentence_count=2,
        published_mean=0.5,
        published_median=0.5,
        published_std=0.25,
        published_iqr=0.25,
        published_decimals=3,
    )


def test_streaming_join_writes_one_manifest_row_per_input(tmp_path: Path) -> None:
    artifacts = _write_fixture_artifacts(tmp_path)
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["corpus_id", "sent_id", "keep", "reason_codes", "rule_version"],
    )
    writer.writeheader()
    data = validate_and_load_corpus(
        _expectation(), artifacts, writer, examples_per_reason=2
    )
    rows = list(csv.DictReader(StringIO(buffer.getvalue())))
    assert len(rows) == 2
    assert rows[0]["keep"] == "1"
    assert rows[1]["keep"] == "0"
    assert rows[1]["reason_codes"] == "very_short_fragment"
    assert data.reason_counts["__any__"] == 1
    assert data.overlap_counts[0] == 1
    assert data.overlap_counts[1] == 1


def test_streaming_join_rejects_sent_id_mismatch(tmp_path: Path) -> None:
    artifacts = _write_fixture_artifacts(tmp_path, mismatch_feature_id=True)
    buffer = StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["corpus_id", "sent_id", "keep", "reason_codes", "rule_version"],
    )
    with pytest.raises(ValueError, match="Ordered sent_id join mismatch"):
        validate_and_load_corpus(
            _expectation(), artifacts, writer, examples_per_reason=1
        )


def test_score_summary_and_published_baseline_validation() -> None:
    data = CorpusData(
        corpus_id="demo",
        display_name="Demo",
        scores=np.array([0.25, 0.75]),
        keep=np.array([True, False]),
        doc_codes=np.array([0, 1], dtype=np.int32),
        doc_names=("a", "b"),
        reason_counts={},  # type: ignore[arg-type]
        overlap_counts={},  # type: ignore[arg-type]
        examples={},
    )
    summary = score_summary(data.scores)
    assert summary["mean"] == 0.5
    assert summary["median"] == 0.5
    assert summary["std"] == 0.25
    assert summary["iqr"] == 0.25
    validation = validate_published_baseline(data, _expectation())
    assert validation["baseline_pass"]


def test_document_cluster_bootstrap_is_deterministic_and_paired() -> None:
    data = CorpusData(
        corpus_id="demo",
        display_name="Demo",
        scores=np.array([0.0, 0.2, 0.8, 1.0]),
        keep=np.array([False, True, True, True]),
        doc_codes=np.array([0, 0, 1, 1], dtype=np.int32),
        doc_names=("doc-a", "doc-b"),
        reason_counts={},  # type: ignore[arg-type]
        overlap_counts={},  # type: ignore[arg-type]
        examples={},
    )
    first = cluster_bootstrap_statistics(data, replicates=20, seed=19)
    second = cluster_bootstrap_statistics(data, replicates=20, seed=19)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    assert np.all(first["filtered_mean"] >= first["raw_mean"])
    assert set(first) == {"raw_mean", "filtered_mean", "raw_median", "filtered_median"}


def test_end_to_end_fixture_writes_portable_reproducible_pack(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    checksum_rows: list[dict[str, str]] = []
    baseline_rows: list[dict[str, str]] = []
    for corpus_id in CORPUS_ORDER:
        sentence_path = outputs_root / "sentences" / f"{corpus_id}.csv"
        score_path = outputs_root / "speciteller" / f"{corpus_id}_scores.tsv"
        feature_path = outputs_root / "features" / f"{corpus_id}_features.csv"
        for path in (sentence_path, score_path, feature_path):
            path.parent.mkdir(parents=True, exist_ok=True)
        sentence_path.write_text(
            "corpus_id,doc_path,sent_idx,sent_text,sent_id\n"
            f"{corpus_id},doc-a,0,This complete sentence remains for analysis.,{corpus_id}-1\n"
            f"{corpus_id},doc-a,1,Too short now,{corpus_id}-2\n"
            f"{corpus_id},doc-b,0,Another complete sentence remains for analysis.,{corpus_id}-3\n"
            f"{corpus_id},doc-b,1,Tiny row now,{corpus_id}-4\n",
            encoding="utf-8",
        )
        score_path.write_text(
            f"{corpus_id}-1\t0.00\n"
            f"{corpus_id}-2\t0.25\n"
            f"{corpus_id}-3\t0.75\n"
            f"{corpus_id}-4\t1.00\n",
            encoding="utf-8",
        )
        feature_path.write_text(
            "corpus_id,sent_id,token_count,char_count\n"
            f"{corpus_id},{corpus_id}-1,6,44\n"
            f"{corpus_id},{corpus_id}-2,3,13\n"
            f"{corpus_id},{corpus_id}-3,6,46\n"
            f"{corpus_id},{corpus_id}-4,3,12\n",
            encoding="utf-8",
        )
        for kind, path in (
            ("sentences", sentence_path),
            ("scores", score_path),
            ("features", feature_path),
        ):
            canonical_relative = {
                "sentences": f"outputs/sentences/{corpus_id}.csv",
                "scores": f"outputs/speciteller/{corpus_id}_scores.tsv",
                "features": f"outputs/features/{corpus_id}_features.csv",
            }[kind]
            checksum_rows.append(
                {
                    "corpus_id": corpus_id,
                    "artifact_kind": kind,
                    "relative_path": canonical_relative,
                    "size_bytes": str(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        baseline_rows.append(
            {
                "corpus_id": corpus_id,
                "display_name": corpus_id,
                "expected_sentence_count": "4",
                "published_mean": "0.500",
                "published_median": "0.500",
                "published_std": "0.395",
                "published_iqr": "0.625",
                "published_decimals": "3",
            }
        )

    checksum_config = tmp_path / "artifact_checksums.csv"
    with checksum_config.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(checksum_rows[0]))
        writer.writeheader()
        writer.writerows(checksum_rows)
    baseline_config = tmp_path / "baseline.csv"
    with baseline_config.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(baseline_rows[0]))
        writer.writeheader()
        writer.writerows(baseline_rows)
    source_config = tmp_path / "source_provenance.json"
    source_config.write_text(
        json.dumps(
            {
                "corpora": {corpus_id: {} for corpus_id in CORPUS_ORDER},
                "speciteller": {
                    "repository_commit": "218cb5a389b3e51d393a76e72714ff65e7e81f47",
                    "released_data_sha256": "dd70443a77b6fc576e3fff88ddc09e6e0777bea3537465a11f339b9de8105fbe",
                },
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "audit"
    paper_dir = tmp_path / "paper"
    metadata = run_preprocessing_ablation(
        RunSettings(
            outputs_root=outputs_root,
            output_dir=output_dir,
            paper_facing_dir=paper_dir,
            baseline_config=baseline_config,
            rule_config=Path("configs/strict_natural_language_v1.json"),
            artifact_checksums_config=checksum_config,
            source_provenance_config=source_config,
            bootstrap_replicates=10,
            seed=20260808,
            examples_per_reason=1,
            command="python scripts/preprocessing_ablation.py",
        )
    )
    assert metadata["baseline_gate_passed"]
    assert metadata["selection_reconciliation"]["total_rows"] == 16
    assert metadata["selection_reconciliation"]["retained_rows"] == 8
    assert (output_dir / "strict_natural_language_v1_manifest.csv").is_file()
    assert (paper_dir / "preprocessing_ablation_table.csv").is_file()
    assert (paper_dir / "README.md").is_file()
    tracked_metadata = (paper_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert re.search(r"(?<![A-Za-z])[A-Za-z]:[/\\](?!/)", tracked_metadata) is None

    tampered = outputs_root / "speciteller" / f"{CORPUS_ORDER[0]}_scores.tsv"
    tampered.write_text(tampered.read_text(encoding="utf-8") + "tamper\t0.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Artifact checksum gate failed"):
        validate_artifact_checksums(checksum_config, outputs_root)
