r"""Pilot binned correspondence (appendix-style descriptive artifact).

This script computes a *binned correspondence* / *pilot-scale reliability view*
between predicted specificity scores and pooled pilot human labels.
It is intentionally descriptive and does NOT estimate formal calibration metrics.

Computation (deterministic):
1) pooled label per row: ``label_mean = mean(ann_a, ann_b, ann_c)``
2) normalized pooled label: ``label_norm = (label_mean - 1) / (K - 1)``
3) within each corpus, score quintiles via ``pandas.qcut(..., q=5, duplicates='drop')``
4) per bin: ``n``, ``mean_score``, ``mean_label_norm``
5) per corpus: ``mean_abs_binned_gap`` = weighted mean over bins of
   ``abs(mean_score - mean_label_norm)`` (weights = ``n``)

Input formats:
- Combined CSV via ``--input`` with columns:
  ``corpus, score, ann_a, ann_b, ann_c``
- Two per-corpus CSVs via ``--github`` and ``--ansible`` with columns:
  ``score, ann_a, ann_b, ann_c``

Outputs under ``--outdir`` (default: ``outputs``):
- ``pilot_binned_correspondence.csv``
- ``pilot_binned_gap.csv``
- ``pilot_binned_correspondence_rows.tex``  (rows only; no tabular wrapper)

Suggested appendix text:
"Scores are partitioned into quintiles within each corpus; pooled labels are
averaged across annotators and linearly normalized to [0,1] for comparability."

Suggested appendix LaTeX template:
    \begin{tabular}{llrr}
    Corpus & Quintile & Mean score & Mean normalized label \\
    \hline
    \input{outputs/pilot_binned_correspondence_rows.tex}
    \end{tabular}

Examples:
    python scripts/pilot_binned_correspondence.py --input data/pilot_labels.csv --label-scale 5 --outdir outputs
    python scripts/pilot_binned_correspondence.py --github data/github_pilot.csv --ansible data/ansible_pilot.csv --label-scale 5 --outdir outputs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COMBINED_COLUMNS = ["corpus", "score", "ann_a", "ann_b", "ann_c"]
REQUIRED_PER_CORPUS_COLUMNS = ["score", "ann_a", "ann_b", "ann_c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute pilot binned correspondence between pooled pilot labels and SpeciTeller scores."
    )
    parser.add_argument("--input", type=Path, default=None, help="Combined CSV path")
    parser.add_argument("--github", type=Path, default=None, help="GitHub-only CSV path")
    parser.add_argument("--ansible", type=Path, default=None, help="Ansible-only CSV path")
    parser.add_argument(
        "--label-scale",
        type=int,
        default=5,
        help="Ordinal label scale upper bound K for labels 1..K (default: 5)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("outputs"),
        help="Directory for output artifacts",
    )
    return parser.parse_args()


def _normalize_corpus_name(name: str) -> str:
    value = str(name).strip().lower()
    if "github" in value:
        return "github"
    if "ansible" in value:
        return "ansible"
    return value


def _display_corpus_name(name: str) -> str:
    if name == "github":
        return "GitHub"
    if name == "ansible":
        return "Ansible"
    return name


def _infer_corpus_from_filename(path: Path) -> str:
    return _normalize_corpus_name(path.stem)


def _validate_columns(df: pd.DataFrame, required: list[str], path: Path) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    use_combined = args.input is not None
    use_split = args.github is not None or args.ansible is not None

    if use_combined and use_split:
        raise ValueError("Use either --input OR (--github + --ansible), not both")
    if not use_combined and not use_split:
        raise ValueError("Provide --input or provide both --github and --ansible")

    if use_combined:
        df = pd.read_csv(args.input)
        _validate_columns(df, REQUIRED_COMBINED_COLUMNS, args.input)
        out = df[REQUIRED_COMBINED_COLUMNS].copy()
        out["corpus"] = out["corpus"].map(_normalize_corpus_name)
        return out

    if args.github is None or args.ansible is None:
        raise ValueError("When not using --input, provide both --github and --ansible")

    frames: list[pd.DataFrame] = []
    for path in [args.github, args.ansible]:
        df = pd.read_csv(path)
        _validate_columns(df, REQUIRED_PER_CORPUS_COLUMNS, path)
        out = df[REQUIRED_PER_CORPUS_COLUMNS].copy()
        out.insert(0, "corpus", _infer_corpus_from_filename(path))
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def _assign_quintile_labels(scores: pd.Series) -> pd.Series:
    numeric_scores = pd.to_numeric(scores, errors="coerce")
    try:
        bins = pd.qcut(numeric_scores, q=5, duplicates="drop")
    except ValueError:
        return pd.Series(["Q1"] * len(numeric_scores), index=numeric_scores.index, dtype="object")

    cat = bins.cat
    codes = cat.codes
    valid_codes = sorted({int(c) for c in codes.tolist() if int(c) >= 0})
    code_to_label = {code: f"Q{i + 1}" for i, code in enumerate(valid_codes)}

    labels: list[object] = []
    for code in codes.tolist():
        c = int(code)
        labels.append(code_to_label.get(c, np.nan))
    return pd.Series(labels, index=numeric_scores.index, dtype="object")


def compute_binned_correspondence(df: pd.DataFrame, label_scale: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if label_scale < 2:
        raise ValueError("--label-scale must be >= 2")

    work = df.copy()
    for col in ["score", "ann_a", "ann_b", "ann_c"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    work = work.dropna(subset=["corpus", "score", "ann_a", "ann_b", "ann_c"]).copy()
    work["label_mean"] = work[["ann_a", "ann_b", "ann_c"]].mean(axis=1)
    work["label_norm"] = (work["label_mean"] - 1.0) / float(label_scale - 1)

    summary_rows: list[dict[str, object]] = []
    gap_rows: list[dict[str, object]] = []

    for corpus_name, corpus_df in work.groupby("corpus", sort=True):
        corpus = str(corpus_name)
        cdf = corpus_df.copy()
        cdf["quintile"] = _assign_quintile_labels(cdf["score"])
        cdf = cdf.dropna(subset=["quintile"]).copy()

        grouped = (
            cdf.groupby("quintile", sort=False)
            .agg(
                n=("score", "size"),
                mean_score=("score", "mean"),
                mean_label_norm=("label_norm", "mean"),
            )
            .reset_index()
        )

        grouped["quintile_idx"] = grouped["quintile"].map(lambda q: int(str(q).replace("Q", "")))
        grouped = grouped.sort_values(by=["quintile_idx", "quintile"], kind="mergesort")

        for row in grouped.itertuples(index=False):
            summary_rows.append(
                {
                    "corpus": corpus,
                    "quintile": str(row.quintile),
                    "n": int(row.n),
                    "mean_score": float(row.mean_score),
                    "mean_label_norm": float(row.mean_label_norm),
                }
            )

        abs_gap = (grouped["mean_score"] - grouped["mean_label_norm"]).abs()
        weights = grouped["n"].astype(float)
        mean_abs_binned_gap = float(np.average(abs_gap.to_numpy(dtype=float), weights=weights.to_numpy(dtype=float)))
        gap_rows.append({"corpus": corpus, "mean_abs_binned_gap": mean_abs_binned_gap})

    summary_df = pd.DataFrame(summary_rows).sort_values(by=["corpus", "quintile"], kind="mergesort")
    gap_df = pd.DataFrame(gap_rows).sort_values(by=["corpus"], kind="mergesort")
    return summary_df, gap_df


def write_outputs(summary_df: pd.DataFrame, gap_df: pd.DataFrame, outdir: Path) -> tuple[Path, Path, Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    summary_path = outdir / "pilot_binned_correspondence.csv"
    gap_path = outdir / "pilot_binned_gap.csv"
    tex_path = outdir / "pilot_binned_correspondence_rows.tex"

    summary_df.to_csv(summary_path, index=False, float_format="%.6f")
    gap_df.to_csv(gap_path, index=False, float_format="%.6f")

    lines: list[str] = []
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"{_display_corpus_name(str(row.corpus))} & {row.quintile} & {float(row.mean_score):.3f} & {float(row.mean_label_norm):.3f} \\\\"
        )
    tex_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    return summary_path, gap_path, tex_path


def main() -> None:
    args = parse_args()
    input_df = load_input_dataframe(args)
    summary_df, gap_df = compute_binned_correspondence(input_df, label_scale=int(args.label_scale))
    summary_path, gap_path, tex_path = write_outputs(summary_df, gap_df, args.outdir)

    print("Pilot binned correspondence complete.")
    for row in gap_df.itertuples(index=False):
        bin_count = int((summary_df["corpus"] == row.corpus).sum())
        print(
            f"- corpus={row.corpus} bins={bin_count} "
            f"mean_abs_binned_gap={float(row.mean_abs_binned_gap):.6f}"
        )
    print(f"Wrote: {summary_path}")
    print(f"Wrote: {gap_path}")
    print(f"Wrote: {tex_path}")


if __name__ == "__main__":
    main()
