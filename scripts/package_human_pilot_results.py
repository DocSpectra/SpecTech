"""Package annotated spot-check pilot results into paper-ready artifacts.

This script performs deterministic descriptive aggregation over manually annotated
spot-check CSV files and writes outputs under ``outputs/paper_pack/``.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
import re
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package human pilot spot-check annotations into descriptive paper artifacts"
    )
    parser.add_argument(
        "--inputs-glob",
        default="outputs/spot_checks/*__*.csv",
        help="Glob pattern for annotated spot-check CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_pack"),
        help="Output directory for packaged artifacts",
    )
    parser.add_argument(
        "--min-annotators",
        type=int,
        default=2,
        help="Minimum annotators required for a corpus to be included",
    )
    parser.add_argument("--label-col", default="human_label", help="Human label column name")
    parser.add_argument("--score-col", default="score", help="Model score column name")
    parser.add_argument("--sent-id-col", default="sent_id", help="Sentence id column name")
    parser.add_argument(
        "--write-per-corpus-figures",
        action="store_true",
        help="Also write one score-by-label figure per corpus",
    )
    parser.add_argument(
        "--drop-missing-labels",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop missing labels during agreement/alignment computations",
    )
    parser.add_argument(
        "--agreement-metric",
        default="both",
        choices=["both", "percent", "kappa"],
        help="Agreement metric(s) to compute",
    )
    parser.add_argument(
        "--kappa-weighting",
        default="quadratic",
        choices=["quadratic", "linear"],
        help="Weighting for weighted Cohen's kappa",
    )
    return parser.parse_args()


def parse_annotated_filename(path: Path) -> tuple[str, str] | None:
    stem = path.stem
    match = re.match(r"^(?P<corpus>.+?)_spot_check_.*__(?P<annotator>.+)$", stem)
    if not match:
        return None
    return match.group("corpus"), match.group("annotator")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_vals = values[order]
    ranks = np.zeros(len(values), dtype=float)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_vals[j] == sorted_vals[i]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def spearman_rho(x: pd.Series, y: pd.Series) -> float:
    xy = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(xy) < 2:
        return float("nan")
    x_arr = xy["x"].to_numpy(dtype=float)
    y_arr = xy["y"].to_numpy(dtype=float)
    x_rank = rankdata(x_arr)
    y_rank = rankdata(y_arr)
    if np.std(x_rank) == 0 or np.std(y_rank) == 0:
        return float("nan")
    corr = np.corrcoef(x_rank, y_rank)[0, 1]
    return float(corr)


def _weights_matrix(size: int, mode: str) -> np.ndarray:
    if size <= 1:
        return np.zeros((size, size), dtype=float)
    idx = np.arange(size, dtype=float)
    ii, jj = np.meshgrid(idx, idx, indexing="ij")
    if mode == "linear":
        return np.abs(ii - jj) / (size - 1)
    return ((ii - jj) ** 2) / ((size - 1) ** 2)


def cohen_kappa(a: pd.Series, b: pd.Series, weighted: str | None = None) -> float:
    pair = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(pair) == 0:
        return float("nan")

    labels = sorted(set(pair["a"].tolist()) | set(pair["b"].tolist()))
    n = len(labels)
    index = {label: i for i, label in enumerate(labels)}

    obs = np.zeros((n, n), dtype=float)
    for row in pair.itertuples(index=False):
        obs[index[row.a], index[row.b]] += 1.0
    obs = obs / obs.sum()

    row_marg = obs.sum(axis=1)
    col_marg = obs.sum(axis=0)
    exp = np.outer(row_marg, col_marg)

    if weighted is None:
        weights = np.ones((n, n), dtype=float)
        np.fill_diagonal(weights, 0.0)
    else:
        weights = _weights_matrix(n, weighted)

    num = float((weights * obs).sum())
    den = float((weights * exp).sum())
    if den == 0:
        return float("nan")
    return float(1.0 - (num / den))


def sorted_label_values(series: pd.Series) -> list[Any]:
    vals = [v for v in series.dropna().tolist()]
    if not vals:
        return []
    numeric_vals = pd.to_numeric(pd.Series(vals), errors="coerce")
    if numeric_vals.notna().all():
        return sorted(set(float(v) for v in numeric_vals.tolist()))
    return sorted(set(str(v) for v in vals))


def draw_score_by_label(ax: Axes, df: pd.DataFrame, corpus_id: str) -> None:
    valid = df.dropna(subset=["human_label_mean", "score"]).copy()
    valid["human_label_mean"] = pd.to_numeric(valid["human_label_mean"], errors="coerce")
    valid["score"] = pd.to_numeric(valid["score"], errors="coerce")
    valid = valid.dropna(subset=["human_label_mean", "score"])

    labels = sorted_label_values(valid["human_label_mean"])
    if not labels:
        ax.set_title(corpus_id)
        ax.set_xlabel("Pooled human label")
        ax.set_ylabel("Model score")
        ax.yaxis.grid(True, linestyle=":", linewidth=0.8)
        return

    grouped_scores: list[list[float]] = []
    for label in labels:
        series = valid.loc[valid["human_label_mean"] == label, "score"]
        numeric = pd.Series(pd.to_numeric(series, errors="coerce")).dropna()
        grouped_scores.append([float(v) for v in numeric.tolist()])
    tick_labels = [str(label) for label in labels]

    ax.boxplot(grouped_scores, tick_labels=tick_labels)
    ax.set_title(corpus_id)
    ax.set_xlabel("Pooled human label")
    ax.set_ylabel("Model score")
    ax.yaxis.grid(True, linestyle=":", linewidth=0.8)


def write_combined_figure(merged_by_corpus: dict[str, pd.DataFrame], output_path: Path) -> None:
    corpora = sorted(merged_by_corpus.keys())
    fig, axes = plt.subplots(1, len(corpora), figsize=(6 * len(corpora), 5), sharey=True)
    if len(corpora) == 1:
        axes = [axes]
    for ax, corpus_id in zip(axes, corpora):
        draw_score_by_label(ax, merged_by_corpus[corpus_id], corpus_id)
    fig.suptitle("Human Pilot: Model Score by Pooled Human Label")
    fig.tight_layout()
    fig.savefig(str(output_path), format="svg")
    plt.close(fig)


def write_per_corpus_figures(merged_by_corpus: dict[str, pd.DataFrame], output_dir: Path) -> list[Path]:
    written: list[Path] = []
    for corpus_id in sorted(merged_by_corpus.keys()):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        draw_score_by_label(ax, merged_by_corpus[corpus_id], corpus_id)
        fig.suptitle("Human Pilot: Model Score by Pooled Human Label")
        fig.tight_layout()
        out_path = output_dir / f"human_pilot_score_by_label_{corpus_id}.svg"
        fig.savefig(str(out_path), format="svg")
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(Path().glob(args.inputs_glob))
    if not input_paths:
        raise FileNotFoundError(f"No input files matched --inputs-glob: {args.inputs_glob}")

    required_cols = {
        args.sent_id_col,
        args.label_col,
        args.score_col,
        "bucket",
        "sent_text",
    }

    corpus_to_inputs: dict[str, list[tuple[str, Path, pd.DataFrame]]] = {}
    for path in input_paths:
        parsed = parse_annotated_filename(path)
        if parsed is None:
            print(f"Skipping unmatched filename pattern: {path}")
            continue
        corpus_id, annotator = parsed
        df = pd.read_csv(path)
        missing = sorted(required_cols - set(df.columns))
        if missing:
            print(f"Skipping {path} due to missing required columns: {missing}")
            continue

        if "corpus_id" in df.columns:
            df["corpus_id"] = df["corpus_id"].fillna(corpus_id).astype(str)
        else:
            df["corpus_id"] = corpus_id

        df[args.sent_id_col] = df[args.sent_id_col].astype(str)
        df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
        df[args.label_col] = pd.to_numeric(df[args.label_col], errors="coerce")
        df["bucket"] = df["bucket"].astype(str)
        df["sent_text"] = df["sent_text"].astype(str)

        corpus_to_inputs.setdefault(corpus_id, []).append((annotator, path, df))

    summary_rows: list[dict[str, Any]] = []
    label_dist_rows: list[dict[str, Any]] = []
    agreement_rows: list[dict[str, Any]] = []
    alignment_rows: list[dict[str, Any]] = []
    merged_rows_all: list[pd.DataFrame] = []
    trend_rows: list[dict[str, Any]] = []

    merged_by_corpus: dict[str, pd.DataFrame] = {}

    included_corpora: list[str] = []
    for corpus_id in sorted(corpus_to_inputs.keys()):
        entries = sorted(corpus_to_inputs[corpus_id], key=lambda x: x[0])
        annotators = [annot for annot, _, _ in entries]
        if len(annotators) < args.min_annotators:
            print(
                f"Skipping corpus '{corpus_id}' (annotators={len(annotators)} < min={args.min_annotators})."
            )
            continue

        if len(annotators) > 2:
            print(
                f"Corpus '{corpus_id}' has >2 annotators; Krippendorff alpha (ordinal) not computed in this script."
            )

        included_corpora.append(corpus_id)
        canonical_annotator, _, canonical_df = entries[0]

        base_cols = [args.sent_id_col, "bucket", args.score_col, "sent_text"]
        merged = canonical_df[base_cols].copy()
        merged = merged.rename(columns={args.score_col: "score", args.sent_id_col: "sent_id"})
        merged["corpus_id"] = corpus_id

        label_col_name = f"human_label__{canonical_annotator}"
        merged[label_col_name] = canonical_df[args.label_col].to_numpy()

        for annotator, path, df in entries[1:]:
            compare_base = canonical_df[[args.sent_id_col, "bucket", args.score_col, "sent_text"]]
            compare_other = df[[args.sent_id_col, "bucket", args.score_col, "sent_text"]]
            cmp = compare_base.merge(compare_other, on=args.sent_id_col, how="inner", suffixes=("_a", "_b"))

            bucket_mismatch = int((cmp["bucket_a"] != cmp["bucket_b"]).sum())
            sent_text_mismatch = int((cmp["sent_text_a"] != cmp["sent_text_b"]).sum())
            score_mismatch = int(
                (~np.isclose(cmp[f"{args.score_col}_a"], cmp[f"{args.score_col}_b"], equal_nan=True)).sum()
            )
            if bucket_mismatch or sent_text_mismatch or score_mismatch:
                print(
                    f"Corpus '{corpus_id}' mismatch vs canonical ({annotator}): "
                    f"bucket={bucket_mismatch}, sent_text={sent_text_mismatch}, score={score_mismatch}"
                )

            other_small = df[[args.sent_id_col, args.label_col]].rename(
                columns={
                    args.sent_id_col: "sent_id",
                    args.label_col: f"human_label__{annotator}",
                }
            )
            merged = merged.merge(other_small, on="sent_id", how="inner")

        label_cols = [f"human_label__{annot}" for annot in annotators]
        merged = merged.sort_values(by=["sent_id"], kind="mergesort").reset_index(drop=True)
        merged["human_label_mean"] = merged[label_cols].mean(axis=1, skipna=True)
        merged["human_label_std"] = merged[label_cols].std(axis=1, ddof=0, skipna=True)
        merged["n_labels_present"] = merged[label_cols].notna().sum(axis=1)

        # Descriptive label distributions and missingness.
        missing_total = 0
        missing_by_annotator: list[str] = []
        label_values_by_annotator: list[str] = []
        for annot in annotators:
            col = f"human_label__{annot}"
            missing_count = int(merged[col].isna().sum())
            missing_total += missing_count
            missing_by_annotator.append(f"{annot}:{missing_count}")

            labeled = merged[col].dropna().astype(float)
            label_values = sorted(set(float(v) for v in labeled.tolist()))
            label_values_by_annotator.append(
                f"{annot}:{'|'.join(str(v) for v in label_values)}"
            )
            n_labeled = len(labeled)
            counts = labeled.value_counts(dropna=True).sort_index()
            for label, count in counts.items():
                label_num = pd.to_numeric(pd.Series([label]), errors="coerce").iloc[0]
                label_out: float | str = (
                    float(label_num) if pd.notna(label_num) else str(label)
                )
                pct = (float(count) / float(n_labeled) * 100.0) if n_labeled > 0 else float("nan")
                label_dist_rows.append(
                    {
                        "corpus_id": corpus_id,
                        "annotator": annot,
                        "label": label_out,
                        "count": int(count),
                        "pct": pct,
                    }
                )

        pooled_labels = merged["human_label_mean"].dropna().astype(float)
        pooled_unique = sorted_label_values(pooled_labels)
        labels_unique_text = ",".join(str(v) for v in pooled_unique)

        summary_rows.append(
            {
                "corpus_id": corpus_id,
                "annotators": ";".join(annotators),
                "n_sentences": int(len(merged)),
                "n_annotators": int(len(annotators)),
                "labels_min": float(pooled_labels.min()) if len(pooled_labels) else float("nan"),
                "labels_max": float(pooled_labels.max()) if len(pooled_labels) else float("nan"),
                "labels_unique": labels_unique_text,
                "missing_labels_total": int(missing_total),
                "missing_labels_by_annotator": ";".join(missing_by_annotator),
                "label_values_by_annotator": ";".join(label_values_by_annotator),
                "pooled_spearman_rho": spearman_rho(merged["human_label_mean"], merged["score"]),
            }
        )

        # Pairwise agreement.
        for annot_a, annot_b in combinations(annotators, 2):
            col_a = f"human_label__{annot_a}"
            col_b = f"human_label__{annot_b}"
            pair = merged[[col_a, col_b]].copy()
            if args.drop_missing_labels:
                pair = pair.dropna()
            overlap = len(pair)

            pct_agree = float("nan")
            kappa_unweighted = float("nan")
            kappa_weighted = float("nan")

            if overlap > 0 and args.agreement_metric in {"both", "percent"}:
                pct_agree = float((pair[col_a] == pair[col_b]).mean() * 100.0)
            if overlap > 0 and args.agreement_metric in {"both", "kappa"}:
                kappa_unweighted = cohen_kappa(pair[col_a], pair[col_b], weighted=None)
                kappa_weighted = cohen_kappa(pair[col_a], pair[col_b], weighted=args.kappa_weighting)

            agreement_rows.append(
                {
                    "corpus_id": corpus_id,
                    "annotator_a": annot_a,
                    "annotator_b": annot_b,
                    "n_overlap": int(overlap),
                    "pct_agreement": pct_agree,
                    "cohen_kappa": kappa_unweighted,
                    "weighted_kappa_quadratic": kappa_weighted,
                }
            )

        # Model↔human alignment by annotator.
        for annot in annotators:
            col = f"human_label__{annot}"
            align_df = merged[[col, "score"]].copy()
            if args.drop_missing_labels:
                align_df = align_df.dropna()
            alignment_rows.append(
                {
                    "corpus_id": corpus_id,
                    "annotator": annot,
                    "n_labeled": int(len(align_df.dropna(subset=[col]))),
                    "spearman_rho": spearman_rho(align_df[col], align_df["score"]),
                }
            )

        # Mean score by pooled label (trend summary).
        trend_df = merged[["human_label_mean", "score"]].dropna().copy()
        if not trend_df.empty:
            trend_df["human_label_mean"] = pd.to_numeric(trend_df["human_label_mean"], errors="coerce")
            trend_df["score"] = pd.to_numeric(trend_df["score"], errors="coerce")
            trend_df = trend_df.dropna()
            grouped = trend_df.groupby("human_label_mean", sort=True)
            for pooled_label, group in grouped:
                pooled_numeric = pd.to_numeric(pd.Series([pooled_label]), errors="coerce").iloc[0]
                trend_rows.append(
                    {
                        "corpus_id": corpus_id,
                        "pooled_label": float(cast(Any, pooled_numeric)),
                        "n": int(len(group)),
                        "mean_score": float(cast(Any, group["score"].mean())),
                    }
                )

        ordered_cols = ["corpus_id", "sent_id", "bucket", "score", "sent_text"] + label_cols + [
            "human_label_mean",
            "human_label_std",
            "n_labels_present",
        ]
        merged = merged[ordered_cols]
        merged_rows_all.append(merged)
        merged_by_corpus[corpus_id] = merged.copy()

    if not included_corpora:
        raise ValueError("No corpora met inclusion criteria after parsing and filtering.")

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["corpus_id"], kind="mergesort")
    label_dist_df = pd.DataFrame(label_dist_rows).sort_values(
        by=["corpus_id", "annotator", "label"], kind="mergesort"
    )
    agreement_df = pd.DataFrame(agreement_rows).sort_values(
        by=["corpus_id", "annotator_a", "annotator_b"], kind="mergesort"
    )
    alignment_df = pd.DataFrame(alignment_rows).sort_values(
        by=["corpus_id", "annotator"], kind="mergesort"
    )
    merged_df = pd.concat(merged_rows_all, ignore_index=True).sort_values(
        by=["corpus_id", "sent_id"], kind="mergesort"
    )
    trend_df = pd.DataFrame(trend_rows).sort_values(by=["corpus_id", "pooled_label"], kind="mergesort")

    summary_path = output_dir / "human_pilot_summary_by_corpus.csv"
    dist_path = output_dir / "human_pilot_label_distributions.csv"
    agreement_path = output_dir / "human_pilot_agreement_pairwise.csv"
    alignment_path = output_dir / "human_pilot_model_alignment_by_annotator.csv"
    merged_path = output_dir / "human_pilot_merged_sentences.csv"
    trend_path = output_dir / "human_pilot_score_mean_by_pooled_label.csv"
    figure_path = output_dir / "human_pilot_score_by_label.svg"

    summary_df.to_csv(summary_path, index=False)
    label_dist_df.to_csv(dist_path, index=False)
    agreement_df.to_csv(agreement_path, index=False)
    alignment_df.to_csv(alignment_path, index=False)
    merged_df.to_csv(merged_path, index=False)
    trend_df.to_csv(trend_path, index=False)
    write_combined_figure(merged_by_corpus, figure_path)

    per_corpus_figure_paths: list[Path] = []
    if args.write_per_corpus_figures:
        per_corpus_figure_paths = write_per_corpus_figures(merged_by_corpus, output_dir)

    print("Human pilot packaging complete.")
    print(f"Input files matched: {len(input_paths)}")
    print(f"Corpora included: {', '.join(sorted(included_corpora))}")
    for corpus_id in sorted(included_corpora):
        annotators = sorted([annot for annot, _, _ in corpus_to_inputs[corpus_id]])
        print(f"- {corpus_id}: annotators={', '.join(annotators)}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {dist_path}")
    print(f"Wrote: {agreement_path}")
    print(f"Wrote: {alignment_path}")
    print(f"Wrote: {merged_path}")
    print(f"Wrote: {trend_path}")
    print(f"Wrote: {figure_path}")
    for p in per_corpus_figure_paths:
        print(f"Wrote: {p}")


if __name__ == "__main__":
    main()
