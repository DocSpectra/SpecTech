"""Outcome-blind tests for the frozen released Yelp/Movie diagnostic."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.ko_specificity.released_domain_diagnostic import (
    classify_condition,
    load_protocol,
    run_domain,
    summarize,
    validate_container_runtime,
    verify_immutable_twitter,
)


CONFIG = Path("configs/ko_released_review_diagnostic_v1.json")
COMPACT = Path("analysis/round2_ko_released_review_diagnostic")


def test_protocol_freezes_released_mapping_counts_and_three_runs() -> None:
    protocol = load_protocol(CONFIG)
    assert protocol.domains == ("yelp", "movie")
    assert protocol.run_ids == ("run01", "run02", "run03")
    yelp = protocol.record["released_conditions"]["yelp"]
    movie = protocol.record["released_conditions"]["movie"]
    assert (yelp["annotated_rows"], yelp["unlabeled_rows"], yelp["expected_prediction_rows"]) == (
        845,
        95650,
        844,
    )
    assert (
        movie["annotated_rows"],
        movie["unlabeled_rows"],
        movie["expected_prediction_rows"],
    ) == (920, 11855, 919)


def test_protocol_rejects_changed_mapping(tmp_path: Path) -> None:
    record = json.loads(CONFIG.read_text(encoding="utf-8"))
    record["protocol"]["domains"] = ["movie", "yelp"]
    changed = tmp_path / "changed.json"
    changed.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_protocol(changed)


def test_classification_boundaries_are_exactly_frozen() -> None:
    assert classify_condition(0.72, 0.03, 0.75, 0.016) == "match"
    assert classify_condition(0.716, 0.03, 0.75, 0.016) == "indeterminate"
    assert classify_condition(0.68, 0.03, 0.75, 0.016) == "shift"
    assert classify_condition(0.75, 0.06, 0.75, 0.016) == "indeterminate"
    assert "dispersion mismatch alone is never a shift" in json.loads(
        CONFIG.read_text(encoding="utf-8")
    )["classification"]["indeterminate"]


def test_immutable_twitter_hash_gate_passes() -> None:
    protocol = load_protocol(CONFIG)
    verified = verify_immutable_twitter(protocol, Path.cwd())
    assert len(verified) == 12
    assert all(len(expected) == 64 for expected in verified.values())


def test_config_hash_is_stable_hex() -> None:
    protocol = load_protocol(CONFIG)
    assert protocol.sha256 == hashlib.sha256(CONFIG.read_bytes()).hexdigest()


def test_run_refuses_nonempty_attempt_directory(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG)
    attempt = tmp_path / "runs" / "yelp" / "run01" / "attempt01"
    attempt.mkdir(parents=True)
    (attempt / "attempt.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_domain(
            protocol,
            domain="yelp",
            run_id="run01",
            attempt_id="attempt01",
            repo_root=Path.cwd(),
            run_root=tmp_path / "runs",
        )


def test_incomplete_domains_package_as_indeterminate(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG)
    compact = tmp_path / "compact"
    comparisons = summarize(
        protocol,
        repo_root=Path.cwd(),
        run_root=tmp_path / "runs",
        compact_root=compact,
    )
    assert len(comparisons) == 6
    assert {row["classification"] for row in comparisons} == {"indeterminate"}
    assert all(row["observed_mean"] == "" for row in comparisons)
    evidence = json.loads((compact / "run_metadata.json").read_text(encoding="utf-8"))
    assert evidence["domain_validity"] == {"movie": False, "yelp": False}
    assert evidence["attempts"] == []


def test_container_runtime_gate_rejects_changed_dependency() -> None:
    protocol = load_protocol(CONFIG)
    runtime = protocol.record["runtime"]
    observed = {
        "upstream_commit": protocol.record["identity"]["upstream_commit"],
        "python": runtime["python"],
        "torch": runtime["torch"],
        "numpy": runtime["numpy"],
        "scipy": "changed",
        "cuda_available": runtime["cuda_available"],
    }
    with pytest.raises(ValueError, match="scipy"):
        validate_container_runtime(protocol, observed)


def test_failed_attempt_inventory_hashes_present_logs(tmp_path: Path) -> None:
    protocol = load_protocol(CONFIG)
    attempt_dir = tmp_path / "runs" / "yelp" / "run01"
    attempt_dir.mkdir(parents=True)
    (attempt_dir / "attempt.json").write_text(
        json.dumps(
            {
                "domain": "yelp",
                "run_id": "run01",
                "status": "failed",
                "failure_message": "returned non-zero exit status 137",
            }
        ),
        encoding="utf-8",
    )
    (attempt_dir / "train.log").write_bytes(b"")
    compact = tmp_path / "compact"
    summarize(protocol, repo_root=Path.cwd(), run_root=tmp_path / "runs", compact_root=compact)
    evidence = json.loads((compact / "run_metadata.json").read_text(encoding="utf-8"))
    train_log = evidence["attempts"][0]["artifact_inventory"]["train.log"]
    assert train_log == {
        "present": True,
        "size_bytes": 0,
        "sha256": hashlib.sha256(b"").hexdigest(),
    }


def test_compact_released_domain_evidence_is_complete() -> None:
    metadata = json.loads((COMPACT / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["config_sha256"] == load_protocol(CONFIG).sha256
    assert metadata["run_count"] == 6
    assert metadata["attempt_count"] == 8
    assert metadata["domain_validity"] == {"movie": True, "yelp": True}
    assert sum(row["status"] == "completed" for row in metadata["attempts"]) == 6
    failed = [row for row in metadata["attempts"] if row["status"] == "failed"]
    assert len(failed) == 2
    for row in failed:
        assert row["exit_code"] == 137
        assert row["artifact_inventory"]["train.log"] == {
            "present": True,
            "size_bytes": 0,
            "sha256": hashlib.sha256(b"").hexdigest(),
        }
        for name in ("predictions.txt", "metrics.json", "teacher_model.pickle"):
            assert row["artifact_inventory"][name] == {"present": False}
    comparisons = (COMPACT / "comparison.csv").read_text(encoding="utf-8").splitlines()
    assert len(comparisons) == 7
    assert all(line.endswith(",shift") for line in comparisons[1:])
