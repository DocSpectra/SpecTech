"""Tests for the frozen Round 2 length-controlled analysis."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import re

import numpy as np
import pytest

from src.analysis.length_controlled import (
    LengthCorpusData,
    LengthRunSettings,
    VARIANTS,
    bootstrap_all,
    build_common_support,
    categorize_gap,
    regression_analysis,
    restricted_cubic_spline_basis,
    run_length_controlled,
    standardized_means,
    validate_method_config,
    weighted_mean_and_ess,
)
from src.analysis.preprocessing_ablation import CORPUS_ORDER, sha256_file


def _data(corpus_id: str, counts: dict[int, int], offset: float = 0.0) -> LengthCorpusData:
    tokens: list[int] = []
    docs: list[int] = []
    scores: list[float] = []
    for length, count in counts.items():
        tokens.extend([length] * count)
        docs.extend(index % 25 for index in range(count))
        scores.extend(min(1.0, offset + length / 100.0) for _ in range(count))
    array_tokens = np.asarray(tokens, dtype=np.int32)
    return LengthCorpusData(
        corpus_id=corpus_id,
        display_name=corpus_id,
        scores=np.asarray(scores, dtype=np.float64),
        token_count=array_tokens,
        char_count=array_tokens * 5,
        keep=np.ones(len(tokens), dtype=np.bool_),
        doc_codes=np.asarray(docs, dtype=np.int32),
        doc_names=tuple(f"doc-{i}" for i in range(25)),
    )


def _four_corpora(counts_by_corpus: dict[str, dict[int, int]] | None = None):
    if counts_by_corpus is None:
        counts_by_corpus = {
            corpus_id: {length: 125 for length in range(5, 10)}
            for corpus_id in CORPUS_ORDER
        }
    return {
        corpus_id: _data(corpus_id, counts_by_corpus[corpus_id], index * 0.05)
        for index, corpus_id in enumerate(CORPUS_ORDER)
    }


def test_frozen_method_config_validates() -> None:
    config = validate_method_config(Path("configs/round2_length_control_v1.json"))
    assert config["outcome_blind_freeze"] is True
    assert config["common_support"]["minimum_rows_per_corpus_per_stratum"] == 100


def test_common_support_enforces_row_and_document_boundaries() -> None:
    counts = {
        corpus_id: {5: 100, 6: 99, 7: 120, 8: 120, 9: 120, 10: 120}
        for corpus_id in CORPUS_ORDER
    }
    data = _four_corpora(counts)
    support = build_common_support(
        data,
        variant="original",
        minimum_rows=100,
        minimum_documents=20,
        maximum_weight_ratio=10.0,
    )
    assert support.lengths.tolist() == [5, 7, 8, 9, 10]


def test_common_support_rejects_extreme_weights() -> None:
    counts = {
        CORPUS_ORDER[0]: {length: 100 for length in range(5, 10)},
        CORPUS_ORDER[1]: {length: (10000 if length == 5 else 100) for length in range(5, 10)},
        CORPUS_ORDER[2]: {length: 100 for length in range(5, 10)},
        CORPUS_ORDER[3]: {length: 100 for length in range(5, 10)},
    }
    data = _four_corpora(counts)
    with pytest.raises(ValueError, match="Weight ratio exceeds"):
        build_common_support(
            data,
            variant="original",
            minimum_rows=100,
            minimum_documents=20,
            maximum_weight_ratio=2.0,
        )


def test_standardization_math_weights_and_ess() -> None:
    mean, ess = weighted_mean_and_ess(
        np.asarray([0.0, 1.0, 1.0]), np.asarray([2.0, 1.0, 1.0])
    )
    assert mean == pytest.approx(0.5)
    assert ess == pytest.approx(16.0 / 6.0)

    counts = {
        corpus_id: {length: 125 for length in range(5, 10)}
        for corpus_id in CORPUS_ORDER
    }
    data = _four_corpora(counts)
    support = build_common_support(
        data,
        variant="original",
        minimum_rows=100,
        minimum_documents=20,
        maximum_weight_ratio=10.0,
    )
    rows = standardized_means(data, support)
    assert rows[0]["standardized_mean"] == pytest.approx(rows[0]["support_unweighted_mean"])
    assert rows[0]["kish_effective_sample_size"] == pytest.approx(625.0)


def test_cluster_bootstrap_is_deterministic() -> None:
    data = _four_corpora()
    supports = {
        variant: build_common_support(
            data,
            variant=variant,
            minimum_rows=100,
            minimum_documents=20,
            maximum_weight_ratio=10.0,
        )
        for variant in VARIANTS
    }
    first, first_seeds = bootstrap_all(data, supports, replicates=20, seed=123)
    second, second_seeds = bootstrap_all(data, supports, replicates=20, seed=123)
    assert first_seeds == second_seeds
    for variant in VARIANTS:
        for corpus_id in CORPUS_ORDER:
            assert np.array_equal(
                first[variant][corpus_id]["standardized_mean"],
                second[variant][corpus_id]["standardized_mean"],
            )


def test_restricted_cubic_spline_is_linear_beyond_boundary() -> None:
    x = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    basis = restricted_cubic_spline_basis(x, [0.0, 1.0, 2.0, 3.0, 4.0])
    assert basis.shape == (7, 4)
    second_difference = basis[-1] - 2 * basis[-2] + basis[-3]
    assert np.max(np.abs(second_difference)) < 1e-10
    with pytest.raises(ValueError, match="strictly increasing"):
        restricted_cubic_spline_basis(x, [0.0, 1.0, 1.0, 2.0])


def test_regression_design_and_interaction_diagnostics() -> None:
    data = _four_corpora()
    support = build_common_support(
        data,
        variant="original",
        minimum_rows=100,
        minimum_documents=20,
        maximum_weight_ratio=10.0,
    )
    contrasts, diagnostics, length_rows, flags = regression_analysis(data, support)
    assert len(diagnostics) == 2
    assert diagnostics[0]["rank"] == diagnostics[0]["parameters"]
    assert diagnostics[0]["clusters"] == 100
    assert len(length_rows) == 15
    assert set(flags) == set(CORPUS_ORDER[1:])
    gaps = [row for row in contrasts if row["estimand"].endswith("additive_gap")]
    assert len(gaps) == 3


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"raw_gap": 0.2, "adjusted_gap": 0.2, "ci_low": 0.1, "ci_high": 0.3, "heterogeneous": True}, "heterogeneous"),
        ({"raw_gap": 0.2, "adjusted_gap": -0.1, "ci_low": -0.2, "ci_high": -0.05, "heterogeneous": False}, "reversed"),
        ({"raw_gap": 0.2, "adjusted_gap": 0.01, "ci_low": 0.0, "ci_high": 0.02, "heterogeneous": False}, "eliminated"),
        ({"raw_gap": 0.2, "adjusted_gap": 0.1, "ci_low": 0.05, "ci_high": 0.15, "heterogeneous": False}, "attenuated"),
        ({"raw_gap": 0.2, "adjusted_gap": 0.18, "ci_low": 0.1, "ci_high": 0.25, "heterogeneous": False}, "persistent"),
    ],
)
def test_categorization_boundaries(kwargs, expected) -> None:
    assert categorize_gap(**kwargs) == expected


def _write_fixture(tmp_path: Path) -> LengthRunSettings:
    outputs = tmp_path / "outputs"
    manifest_rows: list[dict[str, str]] = []
    checksums: list[dict[str, str]] = []
    baseline: list[dict[str, str]] = []
    for corpus_index, corpus_id in enumerate(CORPUS_ORDER):
        sentence = outputs / "sentences" / f"{corpus_id}.csv"
        score = outputs / "speciteller" / f"{corpus_id}_scores.tsv"
        feature = outputs / "features" / f"{corpus_id}_features.csv"
        for path in (sentence, score, feature):
            path.parent.mkdir(parents=True, exist_ok=True)
        score_values: list[float] = []
        with (
            sentence.open("w", encoding="utf-8", newline="") as sh,
            score.open("w", encoding="utf-8", newline="") as oh,
            feature.open("w", encoding="utf-8", newline="") as fh,
        ):
            sw = csv.writer(sh)
            fw = csv.writer(fh)
            sw.writerow(["corpus_id", "doc_path", "sent_idx", "sent_text", "sent_id"])
            fw.writerow(["corpus_id", "sent_id", "token_count", "char_count"])
            for length in range(5, 10):
                for index in range(100):
                    sent_id = f"{corpus_id}-{length}-{index}"
                    value = 0.20 + corpus_index * 0.05 + length * 0.01
                    score_values.append(value)
                    sw.writerow([corpus_id, f"doc-{index % 20}", index, "Fixture sentence text.", sent_id])
                    oh.write(f"{sent_id}\t{value}\n")
                    fw.writerow([corpus_id, sent_id, length, 5 * length])
                    manifest_rows.append(
                        {
                            "corpus_id": corpus_id,
                            "sent_id": sent_id,
                            "keep": "1",
                            "reason_codes": "",
                            "rule_version": "strict_natural_language_v1",
                        }
                    )
        for kind, path in (("sentences", sentence), ("scores", score), ("features", feature)):
            checksums.append(
                {
                    "corpus_id": corpus_id,
                    "artifact_kind": kind,
                    "relative_path": f"outputs/{path.parent.name}/{path.name}",
                    "size_bytes": str(path.stat().st_size),
                    "sha256": sha256_file(path),
                }
            )
        values = np.asarray(score_values)
        q25, q50, q75 = np.quantile(values, [0.25, 0.50, 0.75])
        baseline.append(
            {
                "corpus_id": corpus_id,
                "display_name": corpus_id,
                "expected_sentence_count": "500",
                "published_mean": f"{np.mean(values):.3f}",
                "published_median": f"{q50:.3f}",
                "published_std": f"{np.std(values):.3f}",
                "published_iqr": f"{q75-q25:.3f}",
                "published_decimals": "3",
            }
        )
    manifest = outputs / "round2" / "preprocessing_ablation" / "strict_natural_language_v1_manifest.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    checksum_config = tmp_path / "checksums.csv"
    baseline_config = tmp_path / "baseline.csv"
    write_rows(checksum_config, checksums)
    write_rows(baseline_config, baseline)
    pre_meta = tmp_path / "preprocessing_metadata.json"
    pre_meta.write_text(
        json.dumps(
            {
                "row_manifest": {
                    "path": "outputs/round2/preprocessing_ablation/strict_natural_language_v1_manifest.csv",
                    "sha256": sha256_file(manifest),
                    "row_count_excluding_header": len(manifest_rows),
                }
            }
        ),
        encoding="utf-8",
    )
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    return LengthRunSettings(
        outputs_root=outputs,
        output_dir=tmp_path / "audit",
        paper_facing_dir=tmp_path / "paper",
        baseline_config=baseline_config,
        method_config=Path("configs/round2_length_control_v1.json"),
        artifact_checksums_config=checksum_config,
        preprocessing_metadata=pre_meta,
        source_provenance_config=source,
        bootstrap_replicates=1000,
        seed=20260809,
        command="scripts/length_controlled_analysis.py",
    )


def test_end_to_end_fixture_reconciles_and_writes_portable_pack(tmp_path: Path) -> None:
    settings = _write_fixture(tmp_path)
    metadata = run_length_controlled(settings)
    assert metadata["baseline_gate_passed"] is True
    assert metadata["join_validation"]["total_rows"] == 2000
    assert metadata["common_support"]["original"]["stratum_count"] == 5
    assert (settings.paper_facing_dir / "paper_length_control_table.csv").is_file()
    text = (settings.paper_facing_dir / "run_metadata.json").read_text(encoding="utf-8")
    assert re.search(r"(?<![A-Za-z])[A-Za-z]:[/\\](?!/)", text) is None


def test_join_filter_reconciliation_rejects_manifest_mismatch(tmp_path: Path) -> None:
    settings = _write_fixture(tmp_path)
    manifest = settings.outputs_root / "round2" / "preprocessing_ablation" / "strict_natural_language_v1_manifest.csv"
    rows = manifest.read_text(encoding="utf-8").splitlines()
    rows[1] = rows[1].replace(CORPUS_ORDER[0], CORPUS_ORDER[1], 1)
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8")
    metadata = json.loads(settings.preprocessing_metadata.read_text(encoding="utf-8"))
    metadata["row_manifest"]["sha256"] = sha256_file(manifest)
    settings.preprocessing_metadata.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="corpus_id join mismatch"):
        run_length_controlled(settings)
