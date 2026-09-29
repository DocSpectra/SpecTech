"""Integrity tests for the compact Pair 4A evidence pack."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "analysis" / "round2_distribution_saturation"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(name: str) -> list[dict[str, str]]:
    with (PACK / name).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_release_cardinalities_and_all_scientific_gates() -> None:
    assert len(_rows("coverage.csv")) == 4
    assert len(_rows("distribution_summaries.csv")) == 48
    assert len(_rows("boundary_concentration.csv")) == 288
    assert len(_rows("granuscore_unit_sensitivity.csv")) == 8
    assert len(_rows("boundary_comparisons.csv")) == 72
    assert len(_rows("robust_span_comparisons.csv")) == 36
    assert len(_rows("histogram_bins.csv")) == 2400
    assert len(_rows("paper_distribution_table.csv")) == 20
    assert all(row["baseline_pass"] == "1" for row in _rows("baseline_validation.csv"))
    assert all(row["finite_and_in_bounds"] == "1" for row in _rows("bounds_audit.csv"))
    assert all(row["ordered_join_passed"] == "1" and row["coverage_rate"] == "1.000000000" for row in _rows("coverage.csv"))


def test_release_metadata_hashes_every_table_and_figure() -> None:
    metadata = json.loads((PACK / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["join_validation"]["total_rows"] == 1295205
    assert metadata["baseline_gate"] == {"passed": True, "strict_counts_passed": True}
    assert metadata["granuscore_no_unit_gate"]["native_mass_point_value"] == 100.0
    for name, record in metadata["outputs"].items():
        assert _sha(PACK / name) == record["sha256"]
    assert _sha(PACK / metadata["figure"]["svg_path"]) == metadata["figure"]["svg_sha256"]
    assert _sha(PACK / metadata["figure"]["png_path"]) == metadata["figure"]["png_sha256"]
    ElementTree.parse(PACK / metadata["figure"]["svg_path"])


def test_release_is_portable_private_text_free_and_scope_bounded() -> None:
    for path in PACK.iterdir():
        if path.suffix.lower() in {".png"}:
            continue
        text = path.read_text(encoding="utf-8")
        assert "C:\\Users" not in text
        assert "/home/" not in text
    models = {row["model_instance_id"] for row in _rows("distribution_summaries.csv")}
    assert not any("qwen" in model.lower() or "vago" in model.lower() for model in models)
    readme = (PACK / "README.md").read_text(encoding="utf-8")
    assert "not latent specificity truth" in readme
    assert "All 18 prespecified" in readme
