"""Frozen supplemental diagnostic for the released Yelp and Movie domains."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "ko_released_review_diagnostic_v1"
METRICS = ("spearman", "kendall_tau", "mae")
RAW_ARTIFACTS = (
    "predictions.txt",
    "teacher_model.pickle",
    "model.sha256",
    "train.log",
    "test.log",
    "metrics.json",
    "container_run_metadata.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ReleasedDiagnosticProtocol:
    path: Path
    record: dict[str, Any]
    sha256: str

    @property
    def domains(self) -> tuple[str, ...]:
        return tuple(self.record["protocol"]["domains"])

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(self.record["protocol"]["run_ids"])


def load_protocol(path: Path) -> ReleasedDiagnosticProtocol:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unexpected released-domain diagnostic schema")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("Diagnostic protocol must remain outcome blind")
    if record["identity"]["upstream_commit"] != "36f8e835e9dc6087d5b6763accf302db175947b1":
        raise ValueError("Pinned upstream identity changed")
    if tuple(record["protocol"]["domains"]) != ("yelp", "movie"):
        raise ValueError("Released diagnostic domain mapping changed")
    if tuple(record["protocol"]["run_ids"]) != ("run01", "run02", "run03"):
        raise ValueError("Released diagnostic must use exactly three runs")
    for domain in ("yelp", "movie"):
        condition = record["released_conditions"][domain]
        if condition["expected_prediction_rows"] != condition["annotated_rows"] - 1:
            raise ValueError("Frozen loader withholding count changed")
    return ReleasedDiagnosticProtocol(path, record, sha256_file(path))


def verify_immutable_twitter(protocol: ReleasedDiagnosticProtocol, repo_root: Path) -> dict[str, str]:
    frozen = protocol.record["immutable_twitter_gate"]
    paths = {frozen["config"]: frozen["config_sha256"]}
    for name, expected in frozen["evidence_sha256"].items():
        paths[f"{frozen['evidence_directory']}/{name}"] = expected
    for relative, expected in paths.items():
        actual = sha256_file(repo_root / relative)
        if actual != expected:
            raise ValueError(f"Immutable Twitter artifact changed: {relative}")
    return paths


def validate_container_runtime(
    protocol: ReleasedDiagnosticProtocol, container: dict[str, Any]
) -> None:
    frozen_runtime = protocol.record["runtime"]
    expected_runtime = {
        "upstream_commit": protocol.record["identity"]["upstream_commit"],
        "python": frozen_runtime["python"],
        "torch": frozen_runtime["torch"],
        "numpy": frozen_runtime["numpy"],
        "scipy": frozen_runtime["scipy"],
        "cuda_available": frozen_runtime["cuda_available"],
    }
    for field, expected_value in expected_runtime.items():
        if container.get(field) != expected_value:
            raise ValueError(f"Frozen container runtime mismatch: {field}")


def validate_completed_run_record(
    protocol: ReleasedDiagnosticProtocol, metadata_path: Path
) -> dict[str, Any]:
    record = json.loads(metadata_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed" or record.get("config_sha256") != protocol.sha256:
        raise ValueError(f"Invalid completed run identity: {metadata_path}")
    attempt_path = metadata_path.parent / "attempt.json"
    attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
    if attempt.get("status") != "completed":
        raise ValueError(f"Attempt is not completed: {attempt_path}")
    if attempt.get("run_metadata_sha256") != sha256_file(metadata_path):
        raise ValueError(f"Attempt/run metadata checksum mismatch: {metadata_path}")
    for name in RAW_ARTIFACTS:
        path = metadata_path.parent / name
        if not path.is_file():
            raise ValueError(f"Missing completed run artifact: {path}")
        frozen = record["artifacts"][name]
        if path.stat().st_size != frozen["size_bytes"] or sha256_file(path) != frozen["sha256"]:
            raise ValueError(f"Completed run artifact checksum mismatch: {path}")
    raw_metrics = json.loads((metadata_path.parent / "metrics.json").read_text(encoding="utf-8"))
    if any(float(raw_metrics[name]) != float(record["metrics"][name]) for name in METRICS):
        raise ValueError(f"Run metric record mismatch: {metadata_path}")
    container = json.loads(
        (metadata_path.parent / "container_run_metadata.json").read_text(encoding="utf-8")
    )
    validate_container_runtime(protocol, container)
    expected = protocol.record["released_conditions"][record["domain"]][
        "expected_prediction_rows"
    ]
    if record["prediction_count"] != expected or container["prediction_count"] != expected:
        raise ValueError(f"Completed run prediction count mismatch: {metadata_path}")
    if record["prediction_sha256"] != container["prediction_sha256"]:
        raise ValueError(f"Completed run prediction hash mismatch: {metadata_path}")
    if record["teacher_model_sha256"] != container["teacher_model_sha256"]:
        raise ValueError(f"Completed run checkpoint hash mismatch: {metadata_path}")
    return record


def classify_condition(
    observed_mean: float,
    observed_std: float,
    paper_mean: float,
    paper_std: float,
    *,
    rounding_allowance: float = 0.0005,
) -> str:
    mean_delta = abs(observed_mean - paper_mean)
    std_delta = abs(observed_std - paper_std)
    if (
        mean_delta <= 2.0 * paper_std + rounding_allowance
        and std_delta <= 2.0 * paper_std + rounding_allowance
    ):
        return "match"
    if mean_delta > 4.0 * paper_std + rounding_allowance:
        return "shift"
    return "indeterminate"


def _portable_command(protocol: ReleasedDiagnosticProtocol, domain: str) -> str:
    return (
        "docker run --rm --volume "
        f"{protocol.record['runtime']['glove_volume']}:/artifacts:ro "
        "--volume <absolute-run-dir>:/output "
        "--volume <repo>/ko_container:/diagnostic:ro "
        f"{protocol.record['runtime']['image_id']} "
        "/bin/bash /diagnostic/run_released_domain_diagnostic.sh "
        f"{domain}"
    )


def run_domain(
    protocol: ReleasedDiagnosticProtocol,
    *,
    domain: str,
    run_id: str,
    attempt_id: str,
    repo_root: Path,
    run_root: Path,
    command_runner=subprocess.run,
) -> dict[str, Any]:
    if domain not in protocol.domains or run_id not in protocol.run_ids:
        raise ValueError("Domain/run is outside the frozen diagnostic")
    if not re.fullmatch(r"attempt[0-9]{2}", attempt_id):
        raise ValueError("attempt_id must match attemptNN")
    run_dir = (run_root / domain / run_id / attempt_id).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty attempt directory: {run_dir}")
    verify_immutable_twitter(protocol, repo_root)
    run_dir.mkdir(parents=True, exist_ok=True)
    attempt_path = run_dir / "attempt.json"
    attempt = {
        "schema_version": "ko_released_review_attempt_v1",
        "domain": domain,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "status": "started",
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": protocol.sha256,
        "portable_command": _portable_command(protocol, domain),
    }
    attempt_path.write_text(json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    diagnostic_dir = (repo_root / "ko_container").resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--volume",
        f"{protocol.record['runtime']['glove_volume']}:/artifacts:ro",
        "--volume",
        f"{run_dir}:/output",
        "--volume",
        f"{diagnostic_dir}:/diagnostic:ro",
        protocol.record["runtime"]["image_id"],
        "/bin/bash",
        "/diagnostic/run_released_domain_diagnostic.sh",
        domain,
    ]
    try:
        command_runner(command, check=True)
        missing = [name for name in RAW_ARTIFACTS if not (run_dir / name).is_file()]
        if missing:
            raise ValueError(f"Missing run artifacts: {missing}")
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        container = json.loads(
            (run_dir / "container_run_metadata.json").read_text(encoding="utf-8")
        )
        expected = protocol.record["released_conditions"][domain]["expected_prediction_rows"]
        if metrics["prediction_count"] != expected or container["prediction_count"] != expected:
            raise ValueError("Frozen prediction coverage gate failed")
        if container["teacher_model_sha256"] != sha256_file(run_dir / "teacher_model.pickle"):
            raise ValueError("Checkpoint checksum gate failed")
        validate_container_runtime(protocol, container)
        for metric in METRICS:
            if not math.isfinite(float(metrics[metric])):
                raise ValueError(f"Nonfinite run metric: {metric}")
        artifacts = {
            name: {
                "path": (
                    "outputs/round2/ko_released_review_diagnostic/runs/"
                    f"{domain}/{run_id}/{attempt_id}/{name}"
                ),
                "size_bytes": (run_dir / name).stat().st_size,
                "sha256": sha256_file(run_dir / name),
            }
            for name in RAW_ARTIFACTS
        }
        metadata = {
            "schema_version": "ko_released_review_run_v1",
            "domain": domain,
            "run_id": run_id,
            "attempt_id": attempt_id,
            "status": "completed",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": protocol.sha256,
            "upstream_commit": protocol.record["identity"]["upstream_commit"],
            "image": protocol.record["runtime"]["image"],
            "image_id": protocol.record["runtime"]["image_id"],
            "prediction_count": expected,
            "metrics": {key: metrics[key] for key in METRICS},
            "prediction_summary": {
                key: metrics[key]
                for key in (
                    "prediction_mean",
                    "prediction_population_std",
                    "prediction_min",
                    "prediction_max",
                )
            },
            "prediction_sha256": container["prediction_sha256"],
            "teacher_model_sha256": container["teacher_model_sha256"],
            "artifacts": artifacts,
            "portable_command": _portable_command(protocol, domain),
            "redistribution": "raw released data, predictions, checkpoint, logs, image, and GloVe remain local and untracked",
        }
        (run_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        attempt.update(
            status="completed",
            completed_at_utc=metadata["completed_at_utc"],
            run_metadata_sha256=sha256_file(run_dir / "run_metadata.json"),
        )
        attempt_path.write_text(
            json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return metadata
    except Exception as exc:
        attempt.update(
            status="failed",
            failed_at_utc=datetime.now(timezone.utc).isoformat(),
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
        attempt_path.write_text(
            json.dumps(attempt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


def summarize(
    protocol: ReleasedDiagnosticProtocol,
    *,
    repo_root: Path,
    run_root: Path,
    compact_root: Path,
) -> list[dict[str, Any]]:
    verify_immutable_twitter(protocol, repo_root)
    attempt_inventory: list[dict[str, Any]] = []
    for attempt_path in sorted(run_root.glob("*/*/**/attempt.json")):
        attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
        attempt_dir = attempt_path.parent
        exit_match = re.search(r"exit status ([0-9]+)", attempt.get("failure_message", ""))
        artifact_inventory = {}
        for name in RAW_ARTIFACTS:
            path = attempt_dir / name
            artifact_inventory[name] = {"present": path.is_file()}
            if path.is_file():
                artifact_inventory[name].update(
                    size_bytes=path.stat().st_size,
                    sha256=sha256_file(path),
                )
        attempt_inventory.append(
            {
                "domain": attempt.get("domain"),
                "run_id": attempt.get("run_id"),
                "attempt_id": attempt.get("attempt_id", "legacy_attempt01"),
                "status": attempt.get("status"),
                "started_at_utc": attempt.get("started_at_utc"),
                "finished_at_utc": attempt.get("completed_at_utc", attempt.get("failed_at_utc")),
                "config_sha256": attempt.get("config_sha256"),
                "attempt_path": attempt_path.resolve().relative_to(repo_root).as_posix(),
                "attempt_sha256": sha256_file(attempt_path),
                "failure_type": attempt.get("failure_type", ""),
                "exit_code": int(exit_match.group(1)) if exit_match else None,
                "artifact_inventory": artifact_inventory,
            }
        )

    run_records: list[dict[str, Any]] = []
    domain_validity: dict[str, bool] = {}
    for domain in protocol.domains:
        domain_records: list[dict[str, Any]] = []
        valid_run_ids: set[str] = set()
        for run_id in protocol.run_ids:
            completed: list[dict[str, Any]] = []
            for path in sorted((run_root / domain / run_id).glob("*/run_metadata.json")):
                completed.append(validate_completed_run_record(protocol, path))
            if len(completed) == 1:
                domain_records.extend(completed)
                valid_run_ids.add(run_id)
            else:
                domain_records.extend(completed)
        domain_validity[domain] = (
            len(domain_records) == 3 and valid_run_ids == set(protocol.run_ids)
        )
        run_records.extend(domain_records)

    compact_root.mkdir(parents=True, exist_ok=True)
    run_columns = (
        "domain",
        "run_id",
        "prediction_count",
        "spearman",
        "kendall_tau",
        "mae",
        "prediction_sha256",
        "teacher_model_sha256",
    )
    with (compact_root / "run_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_columns, lineterminator="\n")
        writer.writeheader()
        for record in sorted(run_records, key=lambda row: (row["domain"], row["run_id"])):
            writer.writerow(
                {
                    "domain": record["domain"],
                    "run_id": record["run_id"],
                    "prediction_count": record["prediction_count"],
                    **record["metrics"],
                    "prediction_sha256": record["prediction_sha256"],
                    "teacher_model_sha256": record["teacher_model_sha256"],
                }
            )

    comparisons: list[dict[str, Any]] = []
    rounding = float(protocol.record["classification"]["rounding_allowance"])
    for domain in protocol.domains:
        domain_runs = [record for record in run_records if record["domain"] == domain]
        for metric in METRICS:
            paper = protocol.record["released_conditions"][domain]["paper_table_2"][metric]
            if not domain_validity[domain]:
                comparisons.append(
                    {
                        "domain": domain,
                        "metric": metric,
                        "paper_mean": paper["mean"],
                        "paper_std": paper["std"],
                        "observed_mean": "",
                        "observed_population_std": "",
                        "absolute_mean_difference": "",
                        "absolute_std_difference": "",
                        "mean_difference_in_paper_std": "",
                        "classification": "indeterminate",
                    }
                )
                continue
            values = [float(record["metrics"][metric]) for record in domain_runs]
            observed_mean = statistics.fmean(values)
            observed_std = statistics.pstdev(values)
            classification = classify_condition(
                observed_mean,
                observed_std,
                float(paper["mean"]),
                float(paper["std"]),
                rounding_allowance=rounding,
            )
            comparisons.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "paper_mean": paper["mean"],
                    "paper_std": paper["std"],
                    "observed_mean": observed_mean,
                    "observed_population_std": observed_std,
                    "absolute_mean_difference": abs(observed_mean - float(paper["mean"])),
                    "absolute_std_difference": abs(observed_std - float(paper["std"])),
                    "mean_difference_in_paper_std": (
                        abs(observed_mean - float(paper["mean"])) / float(paper["std"])
                    ),
                    "classification": classification,
                }
            )
    comparison_columns = tuple(comparisons[0].keys())
    with (compact_root / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparisons)

    domain_summaries = []
    for domain in protocol.domains:
        rows = [row for row in comparisons if row["domain"] == domain]
        shifted = [row for row in rows if row["classification"] == "shift"]
        if shifted:
            domain_classification = "shift"
        elif all(row["classification"] == "match" for row in rows):
            domain_classification = "match"
        else:
            domain_classification = "indeterminate"
        directions = []
        for row in shifted:
            delta = float(row["observed_mean"]) - float(row["paper_mean"])
            if row["metric"] == "mae":
                direction = "higher_error" if delta > 0 else "lower_error"
            else:
                direction = "higher_correlation" if delta > 0 else "lower_correlation"
            directions.append(f"{row['metric']}:{direction}")
        domain_summaries.append(
            {
                "domain": domain,
                "classification": domain_classification,
                "shifted_metric_directions": ";".join(directions),
            }
        )
    with (compact_root / "domain_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("domain", "classification", "shifted_metric_directions"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(domain_summaries)

    portable_records = []
    for record in run_records:
        portable_records.append({key: value for key, value in record.items() if key != "completed_at_utc"})
    evidence = {
        "schema_version": "ko_released_review_compact_evidence_v1",
        "config_path": str(protocol.path).replace("\\", "/"),
        "config_sha256": protocol.sha256,
        "immutable_twitter_verified": True,
        "upstream_commit": protocol.record["identity"]["upstream_commit"],
        "image": protocol.record["runtime"]["image"],
        "image_id": protocol.record["runtime"]["image_id"],
        "run_count": len(run_records),
        "domain_validity": domain_validity,
        "attempt_count": len(attempt_inventory),
        "attempts": attempt_inventory,
        "runs": portable_records,
        "table_sha256": {
            "run_metrics.csv": sha256_file(compact_root / "run_metrics.csv"),
            "comparison.csv": sha256_file(compact_root / "comparison.csv"),
            "domain_summary.csv": sha256_file(compact_root / "domain_summary.csv"),
        },
    }
    (compact_root / "run_metadata.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return comparisons
