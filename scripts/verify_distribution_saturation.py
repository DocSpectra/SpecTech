"""Independent representative recomputation for the Pair 4 evidence pack.

This verifier intentionally does not import the analysis implementation. It
reads frozen score files directly, recomputes selected values, and verifies the
committed/full aggregate and figure hashes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FULL = ROOT / "outputs" / "round2" / "distribution_saturation"
COMPACT = ROOT / "analysis" / "round2_distribution_saturation"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probability_scores(path: Path, *, delimiter: str, column: str | int) -> np.ndarray:
    values: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        if isinstance(column, int):
            for row in csv.reader(handle, delimiter=delimiter):
                values.append(float(row[column]))
        else:
            for row in csv.DictReader(handle, delimiter=delimiter):
                values.append(float(row[column]))
    return np.asarray(values, dtype=np.float64)


def granu_scores(path: Path) -> tuple[np.ndarray, np.ndarray]:
    scores: list[float] = []
    no_unit: list[bool] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            scores.append(float(row["granuscore_percentile"]))
            no_unit.append(row["no_referential_unit"] == "true")
    return np.asarray(scores), np.asarray(no_unit, dtype=np.bool_)


def summary(values: np.ndarray, lower: float, upper: float) -> dict[str, float]:
    p05, p50, p95 = np.quantile(values, [0.05, 0.50, 0.95], method="linear")
    span = upper - lower
    return {
        "p05": float(p05), "p50": float(p50), "p95": float(p95),
        "effective_dynamic_range": float((p95 - p05) / span),
        "lower_outer_5pct_share": float(np.mean(values <= lower + 0.05 * span)),
        "upper_outer_5pct_share": float(np.mean(values >= upper - 0.05 * span)),
    }


def assert_close(actual: float, expected: str, label: str, tolerance: float = 5e-9) -> None:
    if not np.isclose(actual, float(expected), rtol=0, atol=tolerance):
        raise AssertionError(f"{label}: recomputed {actual}, stored {expected}")


def main() -> None:
    with (COMPACT / "paper_distribution_table.csv").open("r", encoding="utf-8", newline="") as handle:
        paper = {(row["model_instance_id"], row["corpus_id"]): row for row in csv.DictReader(handle)}
    with (COMPACT / "boundary_comparisons.csv").open("r", encoding="utf-8", newline="") as handle:
        boundary = list(csv.DictReader(handle))
    with (COMPACT / "robust_span_comparisons.csv").open("r", encoding="utf-8", newline="") as handle:
        robust = list(csv.DictReader(handle))
    with (COMPACT / "granuscore_unit_sensitivity.csv").open("r", encoding="utf-8", newline="") as handle:
        units = {(row["corpus_id"], row["subset"]): row for row in csv.DictReader(handle)}

    checks: list[tuple[str, dict[str, float], dict[str, str]]] = []
    wikipedia_spec = probability_scores(
        ROOT / "outputs/speciteller/wikipedia_en_scores.tsv", delimiter="\t", column=1
    )
    checks.append(("SpeciTeller Wikipedia", summary(wikipedia_spec, 0, 1), paper[("speciteller_frozen_round1", "wikipedia_en")]))
    python_spec = probability_scores(
        ROOT / "outputs/speciteller/python_312_html_scores.tsv", delimiter="\t", column=1
    )
    checks.append(("SpeciTeller Python", summary(python_spec, 0, 1), paper[("speciteller_frozen_round1", "python_312_html")]))
    python_ko02 = probability_scores(
        ROOT / "outputs/round2/ko_official_release/runs/python_312_html/run02/scores.csv",
        delimiter=",", column="score_raw",
    )
    checks.append(("Ko run02 Python", summary(python_ko02, 0, 1), paper[("ko_official_release_run02", "python_312_html")]))
    ansible_granu, ansible_no_unit = granu_scores(
        ROOT / "outputs/round2/granuscore/scores/ansible_docs.csv"
    )
    checks.append(("GranuScore Ansible", summary(ansible_granu, 0, 100), paper[("granuscore_native", "ansible_docs")]))
    checks.append(("GranuScore Ansible unit-only", summary(ansible_granu[~ansible_no_unit], 0, 100), units[("ansible_docs", "original_units")]))
    for label, computed, stored in checks:
        for metric, value in computed.items():
            assert_close(value, stored[metric], f"{label} {metric}")

    if np.unique(ansible_granu[ansible_no_unit]).tolist() != [100.0]:
        raise AssertionError("Ansible no-unit sentinel is not the exact 100 mass point")
    assert_close(float(np.mean(ansible_no_unit)), paper[("granuscore_native", "ansible_docs")]["top_mass_point_share"], "Ansible no-unit/top-mass share")

    granu_github = next(row for row in robust if row["model_instance_id"] == "granuscore_native" and row["technical_corpus_id"] == "github_docs" and row["subset"] == "original")
    wikipedia_granu, _ = granu_scores(ROOT / "outputs/round2/granuscore/scores/wikipedia_en.csv")
    github_granu, _ = granu_scores(ROOT / "outputs/round2/granuscore/scores/github_docs.csv")
    wiki_span = float(np.quantile(wikipedia_granu, 0.95) - np.quantile(wikipedia_granu, 0.05))
    github_span = float(np.quantile(github_granu, 0.95) - np.quantile(github_granu, 0.05))
    assert_close(github_span / wiki_span, granu_github["pooled_span_ratio"], "GranuScore GitHub/Wikipedia span ratio")

    spec_python_lower = next(row for row in boundary if row["model_instance_id"] == "speciteller_frozen_round1" and row["technical_corpus_id"] == "python_312_html" and row["subset"] == "original" and row["region"] == "lower")
    assert_close(float(np.mean(python_spec <= 0.05) - np.mean(wikipedia_spec <= 0.05)), spec_python_lower["technical_minus_wikipedia"], "SpeciTeller Python-Wikipedia lower-bound difference")

    metadata = json.loads((COMPACT / "run_metadata.json").read_text(encoding="utf-8"))
    for name, record in metadata["outputs"].items():
        if sha256(FULL / name) != record["sha256"] or sha256(COMPACT / name) != record["sha256"]:
            raise AssertionError(f"aggregate hash mismatch: {name}")
    for kind in ("svg", "png"):
        name = metadata["figure"][f"{kind}_path"]
        expected = metadata["figure"][f"{kind}_sha256"]
        if sha256(FULL / name) != expected or sha256(COMPACT / name) != expected:
            raise AssertionError(f"figure hash mismatch: {name}")

    print("Independent Pair 4 verification passed:")
    print("  representative quantiles/dynamic ranges/boundary rates: 5 views")
    print("  sentinel/top-mass identity and pooled robust-span ratio: passed")
    print(f"  bootstrap contrast spot check: {spec_python_lower['technical_minus_wikipedia']}")
    print(f"  aggregate and figure hashes: {len(metadata['outputs']) + 2} artifacts")


if __name__ == "__main__":
    main()
