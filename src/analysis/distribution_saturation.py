"""Frozen model-aware distribution and boundary-concentration analysis.

The scientific contract lives in ``round2_distribution_saturation_v1``.  No
result artifact is created until exact input identities, complete ordered joins,
the Round 1 baseline, strict-subset counts, score bounds, and GranuScore's
no-unit sentinel behavior all pass.
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
from typing import Any, Iterable

import numpy as np

from src.analysis.granuscore_comparison import (
    GranuCorpus,
    GranuSettings,
    load_full_data,
    load_granuscore_config,
)
from src.analysis.preprocessing_ablation import (
    CORPUS_ORDER,
    load_baseline_expectations,
    sha256_file,
)


MODEL_ORDER = (
    "speciteller_frozen_round1",
    "ko_official_release_run01",
    "ko_official_release_run02",
    "ko_official_release_run03",
    "ko_official_release_run_mean",
    "granuscore_native",
)
PRIMARY_FIGURE_MODELS = (
    "speciteller_frozen_round1",
    "ko_official_release_run01",
    "ko_official_release_run02",
    "ko_official_release_run03",
    "granuscore_native",
)
SUBSETS = ("original", "strict")
QUANTILES = (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
QUANTILE_NAMES = ("p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99")


@dataclass(frozen=True)
class DistributionSettings:
    config_path: Path
    granuscore_settings: GranuSettings
    output_dir: Path
    compact_dir: Path
    command: str


def load_distribution_config(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != "round2_distribution_saturation_v1":
        raise ValueError("unexpected distribution config schema")
    if record.get("outcome_blind_freeze") is not True:
        raise ValueError("distribution config is not outcome-blind frozen")
    if tuple(record["corpora"]["order"]) != CORPUS_ORDER:
        raise ValueError("distribution corpus order changed")
    instances = record["model_instances"]
    if tuple(item["model_instance_id"] for item in instances) != MODEL_ORDER:
        raise ValueError("distribution model-instance order changed")
    if tuple(record["corpora"]["subsets"]) != SUBSETS:
        raise ValueError("distribution subset order changed")
    if tuple(record["statistics"]["quantiles"]) != QUANTILES:
        raise ValueError("distribution quantiles changed")
    if record["bootstrap"]["replicates"] != 1000:
        raise ValueError("bootstrap replicate count changed")
    if record["histogram"]["bins"] != 50:
        raise ValueError("histogram bin count changed")
    return record


def _iter_frozen_files(config: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for label, item in config["dependency_contracts"].items():
        yield f"dependency:{label}", item
    for corpus_id, inputs in config["row_inputs"].items():
        yield f"sentences:{corpus_id}", inputs["sentences"]
        yield f"features:{corpus_id}", inputs["features"]
    for corpus_id, item in config["score_inputs"]["speciteller"].items():
        yield f"speciteller:{corpus_id}", item
    for corpus_id, runs in config["score_inputs"]["ko_runs"].items():
        for run_id, item in runs.items():
            yield f"ko:{corpus_id}:{run_id}", item
    for corpus_id, item in config["score_inputs"]["granuscore"].items():
        yield f"granuscore:{corpus_id}", item


def validate_input_hashes(config: dict[str, Any], root: Path = Path(".")) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, item in _iter_frozen_files(config):
        path = root / item["path"]
        actual = sha256_file(path)
        if actual != item["sha256"]:
            raise ValueError(f"frozen input hash mismatch for {label}: {path}")
        rows.append({"input_id": label, "path": item["path"], "sha256": actual, "hash_passed": True})
    manifest_path = root / config["dependency_contracts"]["ko_input_manifest"]["path"]
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = {row["corpus_id"]: row for row in csv.DictReader(handle)}
    for corpus_id in CORPUS_ORDER:
        if manifest[corpus_id]["ordered_sent_id_sha256"] != config["row_inputs"][corpus_id]["ordered_sent_id_sha256"]:
            raise ValueError(f"ordered sent_id identity mismatch: {corpus_id}")
    return rows


def _model_record(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    return next(item for item in config["model_instances"] if item["model_instance_id"] == model_id)


def model_values(data: GranuCorpus, model_id: str) -> np.ndarray:
    if model_id == "speciteller_frozen_round1":
        return data.base.speciteller
    if model_id.startswith("ko_official_release_run0"):
        run_index = int(model_id[-1]) - 1
        return data.base.ko_runs[run_index]
    if model_id == "ko_official_release_run_mean":
        return np.mean(data.base.ko_runs, axis=0)
    if model_id == "granuscore_native":
        return data.scores
    raise ValueError(f"unknown model instance: {model_id}")


def subset_mask(data: GranuCorpus, subset: str, *, units_only: bool = False) -> np.ndarray:
    if subset not in SUBSETS:
        raise ValueError(subset)
    mask = np.ones(data.scores.size, dtype=np.bool_) if subset == "original" else data.base.keep.copy()
    if units_only:
        mask &= ~data.no_unit
    return mask


def _rounds_like_published(actual: float, expected: float, decimals: int) -> bool:
    return f"{actual:.{decimals}f}" == f"{expected:.{decimals}f}"


def baseline_gate(
    data_by_corpus: dict[str, GranuCorpus], config: dict[str, Any], baseline_path: Path
) -> list[dict[str, Any]]:
    expectations = load_baseline_expectations(baseline_path)
    rows: list[dict[str, Any]] = []
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        values = data.base.speciteller
        q25, median, q75 = np.quantile(values, [0.25, 0.50, 0.75], method="linear")
        exp = expectations[corpus_id]
        actual = {
            "count": int(values.size),
            "mean": float(np.mean(values)),
            "median": float(median),
            "std": float(np.std(values, ddof=0)),
            "iqr": float(q75 - q25),
        }
        checks = {
            "count": actual["count"] == exp.expected_sentence_count == config["corpora"]["row_counts"][corpus_id],
            "mean": _rounds_like_published(actual["mean"], exp.published_mean, exp.published_decimals),
            "median": _rounds_like_published(actual["median"], exp.published_median, exp.published_decimals),
            "std": _rounds_like_published(actual["std"], exp.published_std, exp.published_decimals),
            "iqr": _rounds_like_published(actual["iqr"], exp.published_iqr, exp.published_decimals),
            "strict_count": int(np.sum(data.base.keep)) == config["corpora"]["strict_row_counts"][corpus_id],
        }
        rows.append({
            "corpus_id": corpus_id,
            "actual_count": actual["count"],
            "expected_count": exp.expected_sentence_count,
            "strict_count": int(np.sum(data.base.keep)),
            "expected_strict_count": config["corpora"]["strict_row_counts"][corpus_id],
            "actual_mean": actual["mean"],
            "published_mean": exp.published_mean,
            "actual_median": actual["median"],
            "published_median": exp.published_median,
            "actual_std": actual["std"],
            "published_std": exp.published_std,
            "actual_iqr": actual["iqr"],
            "published_iqr": exp.published_iqr,
            **{f"{name}_pass": passed for name, passed in checks.items()},
            "baseline_pass": all(checks.values()),
        })
    if not all(row["baseline_pass"] for row in rows):
        raise ValueError("Round 1 baseline or strict-count gate failed")
    return rows


def bounds_and_sentinel_gate(
    data_by_corpus: dict[str, GranuCorpus], config: dict[str, Any]
) -> tuple[list[dict[str, Any]], float]:
    rows: list[dict[str, Any]] = []
    no_unit_values: list[np.ndarray] = []
    for model_id in MODEL_ORDER:
        record = _model_record(config, model_id)
        lower, upper = float(record["minimum"]), float(record["maximum"])
        for corpus_id in CORPUS_ORDER:
            values = model_values(data_by_corpus[corpus_id], model_id)
            passed = bool(
                values.size == config["corpora"]["row_counts"][corpus_id]
                and np.all(np.isfinite(values))
                and np.all(values >= lower)
                and np.all(values <= upper)
            )
            rows.append({
                "model_instance_id": model_id,
                "corpus_id": corpus_id,
                "rows": int(values.size),
                "minimum_bound": lower,
                "maximum_bound": upper,
                "finite_and_in_bounds": passed,
            })
            if not passed:
                raise ValueError(f"score bound/coverage gate failed: {model_id} {corpus_id}")
    for corpus_id in CORPUS_ORDER:
        data = data_by_corpus[corpus_id]
        if not np.array_equal(data.no_unit, data.unit_count == 0):
            raise ValueError(f"GranuScore no-unit flag/count mismatch: {corpus_id}")
        if np.any(data.no_unit):
            no_unit_values.append(data.scores[data.no_unit])
    sentinel_values = np.unique(np.concatenate(no_unit_values))
    if sentinel_values.size != 1:
        raise ValueError("GranuScore no-unit rows do not form one exact mass point")
    return rows, float(sentinel_values[0])


def summarize_values(
    values: np.ndarray, *, model: dict[str, Any], corpus_id: str, subset: str
) -> dict[str, Any]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("summary requires nonempty finite values")
    q = np.quantile(values, QUANTILES, method="linear")
    unique, counts = np.unique(values, return_counts=True)
    lower, upper = float(model["minimum"]), float(model["maximum"])
    theoretical_range = upper - lower
    if theoretical_range <= 0:
        raise ValueError("model theoretical range must be positive")
    occupied = float(np.max(values) - np.min(values))
    robust95 = float(q[7] - q[1])
    robust99 = float(q[8] - q[0])
    row: dict[str, Any] = {
        "model_instance_id": model["model_instance_id"],
        "model_family": model["model_family"],
        "model_role": model["role"],
        "score_direction": model["direction"],
        "corpus_id": corpus_id,
        "subset": subset,
        "n": int(values.size),
        "minimum": float(np.min(values)),
        **{name: float(value) for name, value in zip(QUANTILE_NAMES, q, strict=True)},
        "maximum": float(np.max(values)),
        "iqr": float(q[5] - q[3]),
        "p95_minus_p05": robust95,
        "p99_minus_p01": robust99,
        "occupied_range": occupied,
        "theoretical_minimum": lower,
        "theoretical_maximum": upper,
        "theoretical_range_utilization": occupied / theoretical_range,
        "effective_dynamic_range": robust95 / theoretical_range,
        "secondary_effective_dynamic_range": robust99 / theoretical_range,
        "exact_unique_count": int(unique.size),
        "tie_rate": 1.0 - (unique.size / values.size),
        "top_mass_point_value": float(unique[int(np.argmax(counts))]),
        "top_mass_point_share": float(np.max(counts) / values.size),
        "mean_secondary": float(np.mean(values)),
        "population_std_secondary": float(np.std(values, ddof=0)),
    }
    return row


def boundary_rows(
    values: np.ndarray, *, model: dict[str, Any], corpus_id: str, subset: str, fractions: Iterable[float]
) -> list[dict[str, Any]]:
    lower, upper = float(model["minimum"]), float(model["maximum"])
    span = upper - lower
    rows: list[dict[str, Any]] = []
    for fraction in fractions:
        low_threshold = lower + fraction * span
        high_threshold = upper - fraction * span
        rows.extend([
            {
                "model_instance_id": model["model_instance_id"], "corpus_id": corpus_id,
                "subset": subset, "fraction": fraction, "region": "lower",
                "threshold": low_threshold, "inclusive_operator": "<=", "n": int(values.size),
                "count": int(np.sum(values <= low_threshold)), "share": float(np.mean(values <= low_threshold)),
            },
            {
                "model_instance_id": model["model_instance_id"], "corpus_id": corpus_id,
                "subset": subset, "fraction": fraction, "region": "upper",
                "threshold": high_threshold, "inclusive_operator": ">=", "n": int(values.size),
                "count": int(np.sum(values >= high_threshold)), "share": float(np.mean(values >= high_threshold)),
            },
        ])
    return rows


def _seed(master: int, *parts: str) -> int:
    digest = hashlib.sha256((str(master) + "\0" + "\0".join(parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def _remap_codes(codes: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, int]:
    selected = codes[mask]
    _, inverse = np.unique(selected, return_inverse=True)
    return inverse.astype(np.int32), int(np.max(inverse) + 1)


def cluster_proportion_replicates(
    indicator: np.ndarray, codes: np.ndarray, mask: np.ndarray, *, replicates: int, seed: int
) -> np.ndarray:
    remapped, documents = _remap_codes(codes, mask)
    selected = indicator[mask].astype(np.float64)
    counts = np.bincount(remapped, minlength=documents).astype(np.float64)
    sums = np.bincount(remapped, weights=selected, minlength=documents)
    rng = np.random.default_rng(seed)
    probability = np.full(documents, 1.0 / documents)
    result = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 50):
        stop = min(replicates, start + 50)
        weights = rng.multinomial(documents, probability, size=stop - start)
        denominator = weights @ counts
        if np.any(denominator <= 0):
            raise ValueError("zero cluster-bootstrap boundary denominator")
        result[start:stop] = (weights @ sums) / denominator
    return result


def document_spans(
    values: np.ndarray, codes: np.ndarray, mask: np.ndarray, *, minimum_rows: int
) -> np.ndarray:
    selected_codes = codes[mask]
    selected_values = values[mask]
    order = np.argsort(selected_codes, kind="stable")
    sorted_codes = selected_codes[order]
    sorted_values = selected_values[order]
    starts = np.r_[0, np.flatnonzero(sorted_codes[1:] != sorted_codes[:-1]) + 1]
    stops = np.r_[starts[1:], sorted_codes.size]
    spans = [
        float(np.quantile(sorted_values[start:stop], 0.95, method="linear") -
              np.quantile(sorted_values[start:stop], 0.05, method="linear"))
        for start, stop in zip(starts, stops, strict=True)
        if stop - start >= minimum_rows
    ]
    return np.asarray(spans, dtype=np.float64)


def bootstrap_document_mean(values: np.ndarray, *, replicates: int, seed: int) -> np.ndarray:
    if values.size == 0:
        raise ValueError("document bootstrap requires eligible documents")
    rng = np.random.default_rng(seed)
    result = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 100):
        stop = min(replicates, start + 100)
        indexes = rng.integers(0, values.size, size=(stop - start, values.size))
        result[start:stop] = np.mean(values[indexes], axis=1)
    return result


def _ci(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975], method="linear")
    return float(low), float(high)


def histogram_rows(
    values: np.ndarray, *, model: dict[str, Any], corpus_id: str, subset: str, bins: int
) -> list[dict[str, Any]]:
    edges = np.linspace(float(model["minimum"]), float(model["maximum"]), bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    if int(np.sum(counts)) != values.size:
        raise ValueError("histogram count reconciliation failed")
    return [{
        "model_instance_id": model["model_instance_id"], "corpus_id": corpus_id, "subset": subset,
        "bin_index": index, "bin_left": float(edges[index]), "bin_right": float(edges[index + 1]),
        "right_edge_inclusive": index == bins - 1, "count": int(count), "proportion": float(count / values.size),
    } for index, count in enumerate(counts)]


def _category(value: float, low: float, high: float, labels: tuple[str, str, str]) -> str:
    if value <= low:
        return labels[0]
    if value >= high:
        return labels[2]
    return labels[1]


def diagnostic_row(summary: dict[str, Any], boundaries: list[dict[str, Any]], hist: list[dict[str, Any]]) -> dict[str, Any]:
    primary = {row["region"]: row["share"] for row in boundaries if math.isclose(row["fraction"], 0.05)}
    effective = float(summary["effective_dynamic_range"])
    if effective < 0.40:
        dynamic = "compressed"
    elif effective < 0.70:
        dynamic = "intermediate"
    else:
        dynamic = "broad"
    top_mass = float(summary["top_mass_point_share"])
    max_bin = max(float(row["proportion"]) for row in hist)
    dominant = top_mass >= 0.10
    boundary = primary["lower"] >= 0.20 or primary["upper"] >= 0.20
    histogram_concentrated = max_bin >= 0.20
    graded = dynamic == "broad" and not dominant and not boundary and not histogram_concentrated
    if dominant:
        shape = "dominant_exact_mass_point"
    elif boundary:
        shape = "theoretical_boundary_concentrated"
    elif histogram_concentrated:
        shape = "interior_bin_concentrated"
    elif graded:
        shape = "broad_graded_appearance"
    else:
        shape = "compressed_or_intermediate_without_dominant_concentration"
    return {
        "model_instance_id": summary["model_instance_id"], "corpus_id": summary["corpus_id"],
        "subset": summary["subset"], "dynamic_range_category": dynamic,
        "dominant_mass_point": dominant, "boundary_concentrated": boundary,
        "histogram_concentrated": histogram_concentrated, "broad_graded_appearance": graded,
        "shape_diagnostic": shape, "lower_outer_5pct_share": primary["lower"],
        "upper_outer_5pct_share": primary["upper"], "maximum_2pct_bin_share": max_bin,
        "effective_dynamic_range": effective, "top_mass_point_share": top_mass,
    }


def _format(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return ""
        return f"{float(value):.9f}"
    if isinstance(value, (bool, np.bool_)):
        return "1" if bool(value) else "0"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _format(value) for key, value in row.items()})


def _table_index(rows: list[dict[str, Any]], *keys: str) -> dict[tuple[Any, ...], dict[str, Any]]:
    return {tuple(row[key] for key in keys): row for row in rows}


def render_histogram_figure(
    path_svg: Path, path_png: Path, rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = "round2_distribution_saturation_v1"
    import matplotlib.pyplot as plt

    labels = {
        "speciteller_frozen_round1": "SpeciTeller",
        "ko_official_release_run01": "Ko run01",
        "ko_official_release_run02": "Ko run02",
        "ko_official_release_run03": "Ko run03",
        "granuscore_native": "GranuScore",
    }
    corpus_labels = {"wikipedia_en": "Wikipedia", "github_docs": "GitHub", "ansible_docs": "Ansible", "python_312_html": "Python"}
    index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        index.setdefault((row["model_instance_id"], row["corpus_id"], row["subset"]), []).append(row)
    fig, axes = plt.subplots(5, 4, figsize=(7.2, 8.0), squeeze=False)
    for row_index, model_id in enumerate(PRIMARY_FIGURE_MODELS):
        model = _model_record(config, model_id)
        lower, upper = float(model["minimum"]), float(model["maximum"])
        row_max = max(
            float(item["proportion"])
            for corpus_id in CORPUS_ORDER for subset in SUBSETS
            for item in index[(model_id, corpus_id, subset)]
        )
        y_max = min(1.0, max(0.02, 1.08 * row_max))
        for column_index, corpus_id in enumerate(CORPUS_ORDER):
            ax = axes[row_index, column_index]
            original = sorted(index[(model_id, corpus_id, "original")], key=lambda item: item["bin_index"])
            strict = sorted(index[(model_id, corpus_id, "strict")], key=lambda item: item["bin_index"])
            left = np.asarray([item["bin_left"] for item in original])
            right = np.asarray([item["bin_right"] for item in original])
            centers = (left + right) / 2
            widths = right - left
            original_prop = np.asarray([item["proportion"] for item in original])
            strict_prop = np.asarray([item["proportion"] for item in strict])
            ax.bar(centers, original_prop, width=widths, color="#4C78A8", alpha=0.42, linewidth=0, label="original")
            ax.stairs(strict_prop, np.r_[left, right[-1]], color="#E45756", linewidth=0.8, label="strict")
            span = upper - lower
            ax.axvline(lower + 0.05 * span, color="#555555", linestyle=":", linewidth=0.55)
            ax.axvline(upper - 0.05 * span, color="#555555", linestyle=":", linewidth=0.55)
            ax.set_xlim(lower, upper)
            ax.set_ylim(0, y_max)
            ax.tick_params(labelsize=5.5, length=2, pad=1)
            ax.grid(axis="y", color="#dddddd", linewidth=0.35)
            if row_index == 0:
                ax.set_title(corpus_labels[corpus_id], fontsize=7.2, pad=2)
            if column_index == 0:
                ax.set_ylabel(f"{labels[model_id]}\nproportion", fontsize=6.5)
            if lower == 0.0 and upper == 1.0:
                ax.set_xticks([0.0, 0.5, 1.0])
            else:
                ax.set_xticks([0.0, 50.0, 100.0])
            if row_index == 4:
                ax.set_xlabel("native score", fontsize=6.2)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.997))
    fig.suptitle("Model-aware score distributions (within-panel proportions)", fontsize=8.5, y=0.999)
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.972), h_pad=0.7, w_pad=0.5)
    path_svg.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_svg, format="svg", metadata={"Date": None, "Creator": "SpecTech round2_distribution_saturation_v1"})
    svg_text = path_svg.read_text(encoding="utf-8")
    path_svg.write_text(
        "\n".join(line.rstrip() for line in svg_text.splitlines()) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    fig.savefig(path_png, format="png", dpi=300, metadata={"Software": "SpecTech round2_distribution_saturation_v1"})
    plt.close(fig)


def write_readme(
    path: Path,
    diagnostics: list[dict[str, Any]],
    consistency: list[dict[str, Any]],
    unit_rows: list[dict[str, Any]],
    sentinel: float,
) -> None:
    original = [row for row in diagnostics if row["subset"] == "original" and row["model_instance_id"] in PRIMARY_FIGURE_MODELS]
    shape_counts: dict[str, int] = {}
    for row in original:
        shape_counts[row["shape_diagnostic"]] = shape_counts.get(row["shape_diagnostic"], 0) + 1
    heterogeneous = sum(row["cross_model_result"] == "heterogeneous" for row in consistency)
    index = _table_index(diagnostics, "model_instance_id", "corpus_id", "subset")
    unit_index = _table_index(unit_rows, "corpus_id", "subset")
    spec_wiki = index[("speciteller_frozen_round1", "wikipedia_en", "original")]
    spec_python = index[("speciteller_frozen_round1", "python_312_html", "original")]
    ko_python = [index[(f"ko_official_release_run0{run}", "python_312_html", "original")] for run in (1, 2, 3)]
    granu_github = index[("granuscore_native", "github_docs", "original")]
    granu_ansible = index[("granuscore_native", "ansible_docs", "original")]
    granu_ansible_units = unit_index[("ansible_docs", "original_units")]
    lines = [
        "# Round 2 distribution and saturation evidence", "",
        "This compact pack was generated only after every frozen hash, ordered join,",
        "coverage, Round 1 baseline, strict-count, model-bound, and GranuScore",
        "no-unit gate passed.", "", "## Audit summary", "",
        f"- Complete ordered coverage: 1,295,205 rows for every required score instance.",
        f"- GranuScore official no-unit mass point (native scale): {sentinel:.9f}.",
        f"- Primary original model-by-corpus facets audited: {len(original)}.",
        f"- Cross-model relative-category rows labeled heterogeneous: {heterogeneous} of {len(consistency)}.",
    ]
    for name in sorted(shape_counts):
        lines.append(f"- `{name}`: {shape_counts[name]} original facets.")
    lines.extend([
        "", "## Headline distribution result", "",
        f"- SpeciTeller is broad but oppositely boundary-concentrated: Wikipedia has "
        f"{100 * float(spec_wiki['upper_outer_5pct_share']):.1f}% in the upper outer 5%, while "
        f"Python has {100 * float(spec_python['lower_outer_5pct_share']):.1f}% in the lower outer 5%.",
        "- Ko is interior-compressed and has essentially no upper-bound mass. Python's "
        "effective p95-p05 range varies materially across run01/run02/run03: "
        + ", ".join(f"{float(row['effective_dynamic_range']):.3f}" for row in ko_python) + ".",
        f"- Native GranuScore is compressed on GitHub (effective range "
        f"{float(granu_github['effective_dynamic_range']):.3f}). Ansible's all-row effective range is "
        f"{float(granu_ansible['effective_dynamic_range']):.3f}, with a "
        f"{100 * float(granu_ansible['top_mass_point_share']):.2f}% exact 100-point no-unit mass; "
        f"the unit-only range is {float(granu_ansible_units['effective_dynamic_range']):.3f} and its "
        f"upper-5% share is {100 * float(granu_ansible_units['upper_outer_5pct_share']):.2f}%.",
        f"- All {len(consistency)} prespecified cross-model relative-category rows are heterogeneous.",
    ])
    lines.extend(["", "## Interpretation boundary", "",
        "Quantiles, dynamic range, mass points, and theoretical-boundary shares",
        "describe predictor output behavior. They are not latent specificity truth,",
        "semantic-granularity truth, comparative accuracy, documentation quality, or",
        "a causal domain effect. Native raw scales are never compared across models.", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def run_distribution_saturation(settings: DistributionSettings) -> dict[str, Any]:
    config = load_distribution_config(settings.config_path)
    config_sha = sha256_file(settings.config_path)
    input_hash_rows = validate_input_hashes(config)
    granu_config = load_granuscore_config(settings.granuscore_settings.config_path)
    data, coverage, _ = load_full_data(granu_config, settings.granuscore_settings)
    if sum(len(data[corpus].base.sent_ids) for corpus in CORPUS_ORDER) != config["corpora"]["required_total_rows"]:
        raise ValueError("total ordered coverage gate failed")
    baseline_rows = baseline_gate(data, config, Path(config["dependency_contracts"]["round1_baseline"]["path"]))
    bounds_rows, sentinel = bounds_and_sentinel_gate(data, config)

    # No directory or partial result is created before the complete gates above.
    summary_rows: list[dict[str, Any]] = []
    boundary_table: list[dict[str, Any]] = []
    unit_rows: list[dict[str, Any]] = []
    hist_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    summary_index: dict[tuple[str, str, str], dict[str, Any]] = {}
    boundary_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    hist_index: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    fractions = config["boundary_concentration"]["fractions"]
    bins = int(config["histogram"]["bins"])
    for model_id in MODEL_ORDER:
        model = _model_record(config, model_id)
        for corpus_id in CORPUS_ORDER:
            corpus = data[corpus_id]
            all_values = model_values(corpus, model_id)
            for subset in SUBSETS:
                mask = subset_mask(corpus, subset)
                values = all_values[mask]
                summary = summarize_values(values, model=model, corpus_id=corpus_id, subset=subset)
                boundaries = boundary_rows(values, model=model, corpus_id=corpus_id, subset=subset, fractions=fractions)
                hist = histogram_rows(values, model=model, corpus_id=corpus_id, subset=subset, bins=bins)
                summary_rows.append(summary)
                boundary_table.extend(boundaries)
                hist_rows.extend(hist)
                summary_index[(model_id, corpus_id, subset)] = summary
                boundary_index[(model_id, corpus_id, subset)] = boundaries
                hist_index[(model_id, corpus_id, subset)] = hist
                diagnostic_rows.append(diagnostic_row(summary, boundaries, hist))

    granu_model = _model_record(config, "granuscore_native")
    for corpus_id in CORPUS_ORDER:
        corpus = data[corpus_id]
        for subset in SUBSETS:
            mask = subset_mask(corpus, subset, units_only=True)
            values = corpus.scores[mask]
            summary = summarize_values(values, model=granu_model, corpus_id=corpus_id, subset=f"{subset}_units")
            boundaries = boundary_rows(values, model=granu_model, corpus_id=corpus_id, subset=f"{subset}_units", fractions=fractions)
            primary = {row["region"]: row["share"] for row in boundaries if math.isclose(row["fraction"], 0.05)}
            unit_rows.append({**summary, "lower_outer_5pct_share": primary["lower"], "upper_outer_5pct_share": primary["upper"], "no_unit_rows_removed": int(np.sum(subset_mask(corpus, subset) & corpus.no_unit))})

    replicates = int(config["bootstrap"]["replicates"])
    master_seed = int(config["bootstrap"]["master_seed"])
    primary_fraction = float(config["boundary_concentration"]["primary_fraction"])
    boundary_boot: dict[tuple[str, str, str, str], np.ndarray] = {}
    boundary_point: dict[tuple[str, str, str, str], float] = {}
    span_boot: dict[tuple[str, str, str], np.ndarray] = {}
    span_point: dict[tuple[str, str, str], float] = {}
    span_doc_count: dict[tuple[str, str, str], int] = {}
    min_rows = int(config["bootstrap"]["minimum_rows_per_document_for_span"])
    min_docs = int(config["bootstrap"]["minimum_eligible_documents"])
    for model_id in MODEL_ORDER:
        model = _model_record(config, model_id)
        lower, upper = float(model["minimum"]), float(model["maximum"])
        span = upper - lower
        for corpus_id in CORPUS_ORDER:
            corpus = data[corpus_id]
            values = model_values(corpus, model_id)
            for subset in SUBSETS:
                mask = subset_mask(corpus, subset)
                for region, threshold, indicator in (
                    ("lower", lower + primary_fraction * span, values <= lower + primary_fraction * span),
                    ("upper", upper - primary_fraction * span, values >= upper - primary_fraction * span),
                ):
                    key = (model_id, corpus_id, subset, region)
                    boundary_point[key] = float(np.mean(indicator[mask]))
                    boundary_boot[key] = cluster_proportion_replicates(
                        indicator, corpus.base.doc_codes, mask, replicates=replicates,
                        seed=_seed(master_seed, model_id, corpus_id, subset, region, "boundary"),
                    )
                doc_values = document_spans(values, corpus.base.doc_codes, mask, minimum_rows=min_rows)
                if doc_values.size < min_docs:
                    raise ValueError(f"too few eligible documents for span bootstrap: {model_id} {corpus_id} {subset}")
                span_key = (model_id, corpus_id, subset)
                span_doc_count[span_key] = int(doc_values.size)
                span_point[span_key] = float(np.mean(doc_values))
                span_boot[span_key] = bootstrap_document_mean(
                    doc_values, replicates=replicates,
                    seed=_seed(master_seed, model_id, corpus_id, subset, "document_span"),
                )

    boundary_comparisons: list[dict[str, Any]] = []
    robust_comparisons: list[dict[str, Any]] = []
    reference = config["corpora"]["reference_corpus"]
    for model_id in MODEL_ORDER:
        for subset in SUBSETS:
            for technical in CORPUS_ORDER[1:]:
                for region in ("lower", "upper"):
                    wiki_key = (model_id, reference, subset, region)
                    tech_key = (model_id, technical, subset, region)
                    difference = boundary_boot[tech_key] - boundary_boot[wiki_key]
                    low, high = _ci(difference)
                    point = boundary_point[tech_key] - boundary_point[wiki_key]
                    boundary_comparisons.append({
                        "model_instance_id": model_id, "subset": subset, "technical_corpus_id": technical,
                        "reference_corpus_id": reference, "region": region, "fraction": primary_fraction,
                        "wikipedia_share": boundary_point[wiki_key], "technical_share": boundary_point[tech_key],
                        "technical_minus_wikipedia": point, "ci_low": low, "ci_high": high,
                        "relative_category": _category(point, -0.05, 0.05, ("less", "similar_band", "more")),
                    })
                wiki_summary = summary_index[(model_id, reference, subset)]
                tech_summary = summary_index[(model_id, technical, subset)]
                wiki_pooled = float(wiki_summary["p95_minus_p05"])
                tech_pooled = float(tech_summary["p95_minus_p05"])
                if wiki_pooled <= 0:
                    raise ValueError("nonpositive pooled Wikipedia robust span")
                wiki_span_key = (model_id, reference, subset)
                tech_span_key = (model_id, technical, subset)
                wiki_reps = span_boot[wiki_span_key]
                if span_point[wiki_span_key] <= 0 or np.any(wiki_reps <= 0):
                    raise ValueError("nonpositive Wikipedia document-span denominator")
                diff_reps = span_boot[tech_span_key] - wiki_reps
                ratio_reps = span_boot[tech_span_key] / wiki_reps
                diff_low, diff_high = _ci(diff_reps)
                ratio_low, ratio_high = _ci(ratio_reps)
                pooled_ratio = tech_pooled / wiki_pooled
                robust_comparisons.append({
                    "model_instance_id": model_id, "subset": subset, "technical_corpus_id": technical,
                    "reference_corpus_id": reference, "wikipedia_pooled_p95_minus_p05": wiki_pooled,
                    "technical_pooled_p95_minus_p05": tech_pooled,
                    "pooled_span_difference": tech_pooled - wiki_pooled, "pooled_span_ratio": pooled_ratio,
                    "pooled_relative_category": _category(pooled_ratio, 0.80, 1.25, ("narrower", "similar_band", "broader")),
                    "wikipedia_eligible_documents": span_doc_count[wiki_span_key],
                    "technical_eligible_documents": span_doc_count[tech_span_key],
                    "wikipedia_document_average_span": span_point[wiki_span_key],
                    "technical_document_average_span": span_point[tech_span_key],
                    "document_average_span_difference": span_point[tech_span_key] - span_point[wiki_span_key],
                    "document_average_difference_ci_low": diff_low, "document_average_difference_ci_high": diff_high,
                    "document_average_span_ratio": span_point[tech_span_key] / span_point[wiki_span_key],
                    "document_average_ratio_ci_low": ratio_low, "document_average_ratio_ci_high": ratio_high,
                })

    robust_index = _table_index(robust_comparisons, "model_instance_id", "technical_corpus_id", "subset")
    boundary_comparison_index = _table_index(boundary_comparisons, "model_instance_id", "technical_corpus_id", "subset", "region")
    consistency_rows: list[dict[str, Any]] = []
    primary_models = PRIMARY_FIGURE_MODELS
    for technical in CORPUS_ORDER[1:]:
        for subset in SUBSETS:
            measures = {
                "pooled_robust_span": [robust_index[(model, technical, subset)]["pooled_relative_category"] for model in primary_models],
                "lower_boundary": [boundary_comparison_index[(model, technical, subset, "lower")]["relative_category"] for model in primary_models],
                "upper_boundary": [boundary_comparison_index[(model, technical, subset, "upper")]["relative_category"] for model in primary_models],
            }
            for measure, categories in measures.items():
                consistency_rows.append({
                    "technical_corpus_id": technical, "subset": subset, "measure": measure,
                    **{f"{model}_category": category for model, category in zip(primary_models, categories, strict=True)},
                    "cross_model_result": "consistent" if len(set(categories)) == 1 else "heterogeneous",
                    "shared_category": categories[0] if len(set(categories)) == 1 else "",
                })

    diagnostic_index = _table_index(diagnostic_rows, "model_instance_id", "corpus_id", "subset")
    boundary_point_index = _table_index(boundary_table, "model_instance_id", "corpus_id", "subset", "fraction", "region")
    paper_rows: list[dict[str, Any]] = []
    for model_id in PRIMARY_FIGURE_MODELS:
        for corpus_id in CORPUS_ORDER:
            summary = summary_index[(model_id, corpus_id, "original")]
            diagnostic = diagnostic_index[(model_id, corpus_id, "original")]
            paper_rows.append({
                "model_instance_id": model_id, "corpus_id": corpus_id, "n": summary["n"],
                "p05": summary["p05"], "p50": summary["p50"], "p95": summary["p95"],
                "effective_dynamic_range": summary["effective_dynamic_range"],
                "lower_outer_5pct_share": boundary_point_index[(model_id, corpus_id, "original", 0.05, "lower")]["share"],
                "upper_outer_5pct_share": boundary_point_index[(model_id, corpus_id, "original", 0.05, "upper")]["share"],
                "top_mass_point_share": summary["top_mass_point_share"],
                "shape_diagnostic": diagnostic["shape_diagnostic"],
            })

    output_tables = {
        "coverage.csv": [
            {"corpus_id": row["corpus_id"], "canonical_rows": row["canonical_rows"],
             "speciteller_rows": row["canonical_rows"], "ko_rows_per_run": row["canonical_rows"],
             "ko_run_count": 3, "granuscore_rows": row["granuscore_rows"],
             "strict_status_rows": row["canonical_rows"], "coverage_rate": row["coverage_rate"],
             "ordered_join_passed": row["ordered_join_passed"]}
            for row in coverage
        ],
        "baseline_validation.csv": baseline_rows,
        "distribution_summaries.csv": summary_rows,
        "boundary_concentration.csv": boundary_table,
        "granuscore_unit_sensitivity.csv": unit_rows,
        "boundary_comparisons.csv": boundary_comparisons,
        "robust_span_comparisons.csv": robust_comparisons,
        "histogram_bins.csv": hist_rows,
        "model_corpus_diagnostics.csv": diagnostic_rows,
        "cross_model_consistency.csv": consistency_rows,
        "paper_distribution_table.csv": paper_rows,
        "input_identity_audit.csv": input_hash_rows,
        "bounds_audit.csv": bounds_rows,
    }
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in output_tables.items():
        write_csv(settings.output_dir / name, rows)
    svg_path = settings.output_dir / config["outputs"]["figure_svg"]
    png_path = settings.output_dir / config["outputs"]["figure_png"]
    render_histogram_figure(svg_path, png_path, hist_rows, config)
    readme_path = settings.output_dir / "README.md"
    write_readme(readme_path, diagnostic_rows, consistency_rows, unit_rows, sentinel)

    output_records = {
        name: {"path": name, "sha256": sha256_file(settings.output_dir / name), "rows": len(rows)}
        for name, rows in output_tables.items()
    }
    output_records["README.md"] = {
        "path": "README.md", "sha256": sha256_file(readme_path),
        "rows": len(readme_path.read_text(encoding="utf-8").splitlines()),
    }
    metadata = {
        "schema_version": "round2_distribution_saturation_results_v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "command": settings.command,
        "config": {"path": settings.config_path.as_posix(), "sha256": config_sha,
                   "schema_version": config["schema_version"], "outcome_blind_freeze": True},
        "join_validation": {"total_rows": config["corpora"]["required_total_rows"],
                            "corpus_order": list(CORPUS_ORDER), "ordered_sent_id_join": True,
                            "coverage_rate": 1.0, "input_hashes_passed": True},
        "baseline_gate": {"passed": True, "strict_counts_passed": True},
        "bounds_gate": {"passed": True, "model_instances": list(MODEL_ORDER)},
        "granuscore_no_unit_gate": {"passed": True, "single_mass_point": True,
                                    "native_mass_point_value": sentinel},
        "bootstrap": {"replicates": replicates, "master_seed": master_seed, "cluster_key": "doc_path",
                      "confidence_interval": "percentile 2.5% to 97.5%"},
        "outputs": output_records,
        "figure": {"svg_path": svg_path.name, "svg_sha256": sha256_file(svg_path),
                   "png_path": png_path.name, "png_sha256": sha256_file(png_path),
                   "facets": 20, "bins": bins},
        "claim_boundary": "Model-output distributions are not latent specificity truth, comparative accuracy, documentation quality, or causal domain effects.",
    }
    metadata_path = settings.output_dir / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    settings.compact_dir.mkdir(parents=True, exist_ok=True)
    for path in [*(settings.output_dir / name for name in output_tables), readme_path, svg_path, png_path, metadata_path]:
        shutil.copyfile(path, settings.compact_dir / path.name)
        if sha256_file(path) != sha256_file(settings.compact_dir / path.name):
            raise ValueError(f"compact copy hash mismatch: {path.name}")
    return metadata
