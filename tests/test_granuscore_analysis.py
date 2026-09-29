import numpy as np

from src.analysis.granuscore_comparison import (
    GRANUSCORE_ALIGNED,
    GRANUSCORE_NATIVE,
    GranuCorpus,
    build_pilot_paper_table,
    summarize_edit_pairs,
    summarize_full,
)
from src.analysis.model_comparison import ComparisonCorpus
from src.analysis.preprocessing_ablation import CORPUS_ORDER


def test_controlled_edit_summary_preserves_native_direction() -> None:
    pairs = []
    for corpus_id in ("ansible_docs", "github_docs"):
        for edit_type, deltas in {
            "add_specific": [-10.0, -5.0],
            "de_specify": [4.0, 8.0],
            "irrelevant_rewrite": [-1.0, 1.0],
        }.items():
            for index, delta in enumerate(deltas):
                pairs.append({
                    "case_id": f"{corpus_id}-{edit_type}-{index}",
                    "corpus_id": corpus_id,
                    "edit_type": edit_type,
                    "delta_edited_minus_original": delta,
                    "pair_has_no_unit": False,
                })
    rows = summarize_edit_pairs(pairs, replicates=100, seed=7)
    index = {(row["corpus_id"], row["edit_type"], row["variant"]): row for row in rows}
    assert index[("ansible_docs", "add_specific", "all")]["mean_delta"] < 0
    assert index[("ansible_docs", "de_specify", "all")]["mean_delta"] > 0
    assert np.isclose(index[("ansible_docs", "irrelevant_rewrite", "all")]["mean_delta"], 0)
    assert all("negative is finer" in row["direction_note"] for row in rows)


def test_full_summary_covers_frozen_variants_and_comparators() -> None:
    data = {}
    for corpus_index, corpus_id in enumerate(CORPUS_ORDER):
        n = 12
        base_values = np.linspace(0.05, 0.95, n)
        base = ComparisonCorpus(
            corpus_id=corpus_id,
            display_name=corpus_id,
            sent_ids=tuple(f"{corpus_id}-{i}" for i in range(n)),
            doc_paths=tuple(f"doc-{i // 4}" for i in range(n)),
            doc_codes=np.repeat(np.arange(3, dtype=np.int32), 4),
            doc_names=("doc-0", "doc-1", "doc-2"),
            token_count=np.arange(1, n + 1, dtype=np.int32),
            char_count=np.arange(20, 20 + n, dtype=np.int32),
            keep=np.asarray([True, True, False] * 4, dtype=np.bool_),
            speciteller=base_values,
            ko_runs=np.vstack([base_values, base_values**2, np.sqrt(base_values)]),
        )
        scores = 100 - 50 * base_values + corpus_index
        no_unit = np.zeros(n, dtype=np.bool_)
        no_unit[2] = True
        units = np.full(n, 2, dtype=np.int32)
        units[2] = 0
        data[corpus_id] = GranuCorpus(base, scores, units, no_unit)
    tables = summarize_full(data, replicates=30, seed=11)
    assert len(tables["full_corpus_summaries.csv"]) == 16
    assert len(tables["full_model_agreement.csv"]) == 80
    assert len(tables["full_corpus_gaps.csv"]) == 12
    assert {row["variant"] for row in tables["full_corpus_summaries.csv"]} == {
        "original_all", "strict_all", "original_units", "strict_units"
    }


def test_granuscore_pilot_direction_labels_are_unambiguous() -> None:
    assert GRANUSCORE_NATIVE == "granuscore_native_higher_is_coarser"
    assert GRANUSCORE_ALIGNED == "granuscore_direction_aligned_secondary"


def test_pilot_paper_table_unifies_all_models_and_human_targets() -> None:
    models = (
        "speciteller_frozen_round1",
        "ko_official_release_run01",
        "ko_official_release_run02",
        "ko_official_release_run03",
        "ko_official_release_run_mean",
        "qwen3_14b_zero_shot_rubric",
        GRANUSCORE_NATIVE,
        GRANUSCORE_ALIGNED,
    )
    targets = ("ann_a", "ann_b", "ann_c", "pooled_human_mean")
    agreements = [
        {
            "corpus_id": corpus_id,
            "target_id": target_id,
            "model_id": model_id,
            "spearman_rho": 0.25,
            "ci_low": 0.1,
            "ci_high": 0.4,
        }
        for corpus_id in ("ansible_docs", "github_docs")
        for target_id in targets
        for model_id in models
    ]
    rows = build_pilot_paper_table(agreements)
    assert len(rows) == 8
    assert {(row["corpus_id"], row["target_id"]) for row in rows} == {
        (corpus_id, target_id)
        for corpus_id in ("ansible_docs", "github_docs")
        for target_id in targets
    }
    assert all(row["granuscore_native_rho"] == 0.25 for row in rows)
    assert all(row["qwen_ci_high"] == 0.4 for row in rows)
