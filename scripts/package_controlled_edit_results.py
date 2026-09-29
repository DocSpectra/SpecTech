"""Package Phase 7.5 controlled-edit scored CSVs into paper-ready artifacts.

Reads existing ``*_scored.csv`` files, computes deterministic summary statistics,
and writes neutral analysis artifacts under ``outputs/paper_pack/``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re
from typing import Any
from typing import cast

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd


EDIT_TYPE_ORDER = ["add_specific", "de_specify", "irrelevant_rewrite"]
EDIT_TYPE_TO_RULE = {
    "add_specific": "delta > 0",
    "de_specify": "delta < 0",
    "irrelevant_rewrite": "abs(delta) <= epsilon",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package controlled-edit scored CSVs into paper-ready summary artifacts"
    )
    parser.add_argument(
        "--inputs-glob",
        default="outputs/controlled_edits/*_scored.csv",
        help="Glob pattern for scored controlled-edit CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/paper_pack"),
        help="Directory for packaged analysis artifacts",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Tolerance for irrelevant_rewrite directional consistency: abs(delta) <= epsilon",
    )
    parser.add_argument(
        "--outlier-abs-delta",
        type=float,
        default=0.50,
        help="Absolute delta threshold for optional outlier row export",
    )
    parser.add_argument(
        "--write-per-corpus-figures",
        action="store_true",
        help="Also write one delta-distribution SVG per corpus",
    )
    return parser.parse_args()


def infer_corpus_id_from_filename(path: Path) -> str:
    stem = path.stem
    for marker in ("_controlled_", "_template_"):
        if marker in stem:
            return stem.split(marker, 1)[0]
    return re.sub(r"_scored$", "", stem)


def pick_column(df: pd.DataFrame, candidates: list[str], required: bool = True) -> str | None:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    if required:
        raise ValueError(f"Missing required column. Tried candidates: {candidates}")
    return None


def normalize_scored_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        return pd.DataFrame(
            columns=[
                "corpus_id",
                "edit_type",
                "original_score",
                "edited_score",
                "delta",
                "source_file",
            ]
        )

    inferred_corpus = infer_corpus_id_from_filename(path)

    corpus_col = pick_column(df, ["corpus_id"], required=False)
    edit_col = pick_column(df, ["edit_type"])
    original_col = pick_column(df, ["speciteller_score_original", "original_score"])
    edited_col = pick_column(df, ["speciteller_score_edited", "edited_score"])
    delta_col = pick_column(df, ["delta"], required=False)

    out = pd.DataFrame()
    if corpus_col is None:
        out["corpus_id"] = inferred_corpus
    else:
        out["corpus_id"] = df[corpus_col].fillna(inferred_corpus).astype(str)

    out["edit_type"] = df[edit_col].astype(str)
    out["original_score"] = pd.to_numeric(df[original_col], errors="coerce")
    out["edited_score"] = pd.to_numeric(df[edited_col], errors="coerce")

    if delta_col is None:
        out["delta"] = out["edited_score"] - out["original_score"]
    else:
        out["delta"] = pd.to_numeric(df[delta_col], errors="coerce")

    # Keep optional appendix/debug fields when present.
    for optional_col in ["sent_id", "sentence_original", "sentence_edited"]:
        out[optional_col] = df[optional_col] if optional_col in df.columns else ""

    out["source_file"] = str(path)

    out = out.dropna(subset=["original_score", "edited_score", "delta", "edit_type", "corpus_id"])
    return out


def edit_type_sort_key(edit_type: str) -> tuple[int, str]:
    if edit_type in EDIT_TYPE_ORDER:
        return (EDIT_TYPE_ORDER.index(edit_type), edit_type)
    return (len(EDIT_TYPE_ORDER), edit_type)


def directional_consistency(series: pd.Series, edit_type: str, epsilon: float) -> float:
    if len(series) == 0:
        return float("nan")
    if edit_type == "add_specific":
        return float((series > 0).mean())
    if edit_type == "de_specify":
        return float((series < 0).mean())
    if edit_type == "irrelevant_rewrite":
        return float((series.abs() <= epsilon).mean())
    return float("nan")


def build_summary(df: pd.DataFrame, epsilon: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.groupby(["corpus_id", "edit_type"], sort=False)
    for (corpus_id, edit_type), group in grouped:
        deltas = group["delta"].astype(float)
        edit_type_str = str(edit_type)
        prop = directional_consistency(deltas, edit_type_str, epsilon)
        rows.append(
            {
                "corpus_id": corpus_id,
                "edit_type": edit_type_str,
                "n": int(len(group)),
                "mean_delta": float(deltas.mean()),
                "median_delta": float(deltas.median()),
                "mean_abs_delta": float(deltas.abs().mean()),
                "q25_delta": float(deltas.quantile(0.25)),
                "q75_delta": float(deltas.quantile(0.75)),
                "directional_consistency_prop": prop,
                "directional_consistency_pct": float(prop * 100.0) if not math.isnan(prop) else float("nan"),
                "directional_rule": EDIT_TYPE_TO_RULE.get(edit_type_str, "n/a"),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary

    summary["_edit_rank"] = summary["edit_type"].map(lambda x: edit_type_sort_key(str(x))[0])
    summary = summary.sort_values(
        by=["corpus_id", "_edit_rank", "edit_type"],
        kind="mergesort",
    ).drop(columns=["_edit_rank"]).reset_index(drop=True)
    return summary


def draw_corpus_boxplot(ax: Axes, corpus_df: pd.DataFrame, corpus_id: str) -> None:
    values = []
    labels = []
    for edit_type in EDIT_TYPE_ORDER:
        delta_vals = corpus_df.loc[corpus_df["edit_type"] == edit_type, "delta"].astype(float).tolist()
        values.append(delta_vals if len(delta_vals) else [float("nan")])
        labels.append(edit_type)

    ax.boxplot(values, tick_labels=labels)
    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_title(str(corpus_id))
    ax.set_xlabel("Edit Type")
    ax.set_ylabel("Δ score (Edited − Original)")


def write_combined_figure(df: pd.DataFrame, output_path: Path) -> None:
    corpora = sorted(df["corpus_id"].dropna().astype(str).unique().tolist())
    if not corpora:
        raise ValueError("No corpus data available to plot")

    fig, axes = plt.subplots(1, len(corpora), figsize=(6 * len(corpora), 5), sharey=True)
    if len(corpora) == 1:
        axes = [axes]

    for ax, corpus_id in zip(axes, corpora):
        corpus_df = df[df["corpus_id"] == corpus_id]
        draw_corpus_boxplot(ax, corpus_df, corpus_id)

    fig.suptitle("Controlled Edit Δ Distributions (Edited − Original)")
    fig.tight_layout()
    fig.savefig(str(output_path), format="svg")
    plt.close(fig)


def write_per_corpus_figures(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    written: list[Path] = []
    for corpus_id in sorted(df["corpus_id"].dropna().astype(str).unique().tolist()):
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        corpus_df = df[df["corpus_id"] == corpus_id]
        draw_corpus_boxplot(ax, corpus_df, corpus_id)
        fig.suptitle("Controlled Edit Δ Distributions (Edited − Original)")
        fig.tight_layout()
        out_path = output_dir / f"controlled_edit_delta_distribution_{corpus_id}.svg"
        fig.savefig(str(out_path), format="svg")
        plt.close(fig)
        written.append(out_path)
    return written


def _latex_escape(text: str) -> str:
    return text.replace("\\", "\\textbackslash ").replace("_", "\\_")


def write_latex_summary_table(summary_df: pd.DataFrame, output_path: Path) -> None:
    row_break = r"\\"
    lines: list[str] = []
    lines.append("% Auto-generated by scripts/package_controlled_edit_results.py")
    lines.append("\\begin{tabular}{llrrrrr}")
    lines.append(
        "corpus\\_id & edit\\_type & n & mean\\_delta & median\\_delta & "
        "mean\\_abs\\_delta & directional\\_consistency\\_pct "
        f"{row_break}"
    )
    lines.append("\\hline")

    current_corpus = None
    for row in summary_df.itertuples(index=False):
        corpus_id = str(row.corpus_id)
        if current_corpus != corpus_id:
            if current_corpus is not None:
                lines.append("\\hline")
            lines.append(f"\\multicolumn{{7}}{{l}}{{{_latex_escape(corpus_id)}}} {row_break}")
            current_corpus = corpus_id

        n_val = int(cast(Any, row.n))
        mean_delta = float(cast(Any, row.mean_delta))
        median_delta = float(cast(Any, row.median_delta))
        mean_abs_delta = float(cast(Any, row.mean_abs_delta))
        consistency_pct = float(cast(Any, row.directional_consistency_pct))

        lines.append(
            f" & {_latex_escape(str(row.edit_type))}"
            f" & {n_val}"
            f" & {mean_delta:.4f}"
            f" & {median_delta:.4f}"
            f" & {mean_abs_delta:.4f}"
            f" & {consistency_pct:.2f} {row_break}"
        )

    lines.append("\\end{tabular}")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    input_paths = sorted(Path().glob(args.inputs_glob))
    if not input_paths:
        raise FileNotFoundError(f"No input files matched --inputs-glob: {args.inputs_glob}")

    normalized_frames = [normalize_scored_file(path) for path in input_paths]
    data = pd.concat(normalized_frames, ignore_index=True)
    if data.empty:
        raise ValueError("Matched input files contained no valid rows after normalization")

    data["_edit_rank"] = data["edit_type"].map(lambda x: edit_type_sort_key(str(x))[0])
    data = data.sort_values(
        by=["corpus_id", "_edit_rank", "edit_type", "sent_id", "source_file"],
        kind="mergesort",
    ).drop(columns=["_edit_rank"]).reset_index(drop=True)

    summary_df = build_summary(data, epsilon=args.epsilon)
    summary_path = output_dir / "controlled_edit_summary_by_corpus.csv"
    summary_df.to_csv(summary_path, index=False)
    summary_tex_path = output_dir / "controlled_edit_summary_table.tex"
    write_latex_summary_table(summary_df, summary_tex_path)

    combined_fig_path = output_dir / "controlled_edit_delta_distributions.svg"
    write_combined_figure(data, combined_fig_path)

    outlier_path = output_dir / "controlled_edit_outlier_rows.csv"
    outliers = data.loc[data["delta"].abs() >= float(args.outlier_abs_delta)].copy()
    wrote_outliers = False
    if not outliers.empty:
        outlier_columns = [
            "corpus_id",
            "edit_type",
            "sent_id",
            "sentence_original",
            "sentence_edited",
            "original_score",
            "edited_score",
            "delta",
            "source_file",
        ]
        outliers = outliers.loc[:, outlier_columns].copy()
        outliers["_edit_rank"] = outliers["edit_type"].map(lambda x: edit_type_sort_key(str(x))[0])
        outliers = outliers.sort_values(
            by=["corpus_id", "_edit_rank", "edit_type", "source_file", "sent_id"],
        ).drop(columns=["_edit_rank"]).reset_index(drop=True)
        outliers.to_csv(outlier_path, index=False)
        wrote_outliers = True

    per_corpus_paths: list[Path] = []
    if args.write_per_corpus_figures:
        per_corpus_paths = write_per_corpus_figures(data, output_dir)

    corpora = sorted(data["corpus_id"].astype(str).unique().tolist())
    print("Packaging complete.")
    print(f"Input files: {len(input_paths)}")
    print(f"Input rows: {len(data)}")
    print(f"Corpora detected: {', '.join(corpora)}")
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {summary_tex_path}")
    print(f"Wrote: {combined_fig_path}")
    if wrote_outliers:
        print(f"Wrote: {outlier_path}")
    else:
        print(
            "No outlier rows met threshold "
            f"abs(delta) >= {float(args.outlier_abs_delta):.2f}; outlier CSV not written."
        )
    for path in per_corpus_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()
