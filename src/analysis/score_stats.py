"""Corpus-level score summary statistics from canonical SpeciTeller score files."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


PREFERRED_SCORE_COLUMNS = ["score", "specificity", "speciteller_score"]
OUTPUT_COLUMNS = [
    "corpus_id",
    "score_count",
    "score_mean",
    "score_median",
    "score_std",
    "score_min",
    "score_max",
    "score_q1",
    "score_q3",
    "score_iqr",
]


@dataclass(frozen=True)
class ScoreStatsRow:
    corpus_id: str
    score_count: int
    score_mean: float
    score_median: float
    score_std: float
    score_min: float
    score_max: float
    score_q1: float
    score_q3: float
    score_iqr: float


def _looks_like_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _read_tsv_rows(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if all(not cell.strip() for cell in row):
                continue
            rows.append([cell.strip() for cell in row])
    if not rows:
        raise ValueError(f"Empty score file: {path}")
    return rows


def _rows_to_dataframe(path: Path) -> pd.DataFrame:
    rows = _read_tsv_rows(path)
    first = rows[0]
    first_lc = [cell.lower() for cell in first]
    has_named_header = "sent_id" in first_lc or any(
        name in first_lc for name in PREFERRED_SCORE_COLUMNS
    )
    header_like = has_named_header or (len(first) >= 2 and not _looks_like_float(first[1]))

    if header_like:
        header = first
        data_rows = rows[1:]
        if not data_rows:
            raise ValueError(f"Header-only score file (no data rows): {path}")
        max_len = max(len(header), max(len(r) for r in data_rows))
        header = header + [f"extra_{i}" for i in range(len(header), max_len)]
        normalized_rows = [r + [""] * (max_len - len(r)) for r in data_rows]
        return pd.DataFrame(normalized_rows, columns=header)

    # Canonical no-header TSV: sent_id<TAB>score
    if any(len(r) < 2 for r in rows):
        raise ValueError(f"Malformed no-header score TSV (expected >=2 columns): {path}")
    return pd.DataFrame(
        {
            "sent_id": [r[0] for r in rows],
            "score": [r[1] for r in rows],
        }
    )


def _pick_score_column(df: pd.DataFrame) -> str:
    cols_lc = {col.lower(): col for col in df.columns}
    for candidate in PREFERRED_SCORE_COLUMNS:
        if candidate in cols_lc:
            return cols_lc[candidate]

    if "sent_id" not in df.columns:
        raise ValueError("Could not find 'sent_id' column in score file")

    numeric_candidates: list[str] = []
    for col in df.columns:
        if col == "sent_id":
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().all():
            numeric_candidates.append(col)

    if len(numeric_candidates) == 1:
        return numeric_candidates[0]
    raise ValueError(
        "Could not infer unique score column. "
        f"Numeric candidates besides sent_id: {numeric_candidates}"
    )


def _write_qa_note(paths: list[Path], message: str) -> None:
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(message + "\n", encoding="utf-8")


def compute_score_stats_from_file(
    corpus_id: str,
    score_path: Path,
    qa_note_paths: list[Path] | None = None,
) -> ScoreStatsRow:
    qa_note_paths = qa_note_paths or []
    df = _rows_to_dataframe(score_path)

    if "sent_id" not in df.columns:
        msg = f"{corpus_id}: missing sent_id column in score file: {score_path}"
        _write_qa_note(qa_note_paths, msg)
        raise ValueError(msg)

    score_col = _pick_score_column(df)
    df = df[["sent_id", score_col]].copy()
    df["sent_id"] = df["sent_id"].astype(str)

    duplicate_mask = df["sent_id"].duplicated(keep=False)
    if duplicate_mask.any():
        dup_count = int(duplicate_mask.sum())
        uniq_dups = sorted(df.loc[duplicate_mask, "sent_id"].unique().tolist())
        sample = ", ".join(uniq_dups[:5])
        msg = (
            f"{corpus_id}: duplicate sent_id rows in score file ({dup_count} duplicate rows). "
            f"Example sent_ids: {sample}"
        )
        _write_qa_note(qa_note_paths, msg)
        raise ValueError(msg)

    scores = pd.to_numeric(df[score_col], errors="coerce")
    if scores.isna().any():
        msg = f"{corpus_id}: non-numeric score values detected in column '{score_col}'"
        _write_qa_note(qa_note_paths, msg)
        raise ValueError(msg)

    score_count = int(len(scores))
    unique_count = int(df["sent_id"].nunique())
    if score_count != unique_count:
        msg = (
            f"{corpus_id}: score_count ({score_count}) != unique sent_id count ({unique_count}) "
            f"in score file: {score_path}"
        )
        _write_qa_note(qa_note_paths, msg)
        raise ValueError(msg)

    q1 = float(scores.quantile(0.25))
    q3 = float(scores.quantile(0.75))
    return ScoreStatsRow(
        corpus_id=corpus_id,
        score_count=score_count,
        score_mean=float(scores.mean()),
        score_median=float(scores.median()),
        score_std=float(scores.std(ddof=0)),
        score_min=float(scores.min()),
        score_max=float(scores.max()),
        score_q1=q1,
        score_q3=q3,
        score_iqr=float(q3 - q1),
    )


def _row_to_csv_dict(row: ScoreStatsRow) -> dict[str, str | int]:
    return {
        "corpus_id": row.corpus_id,
        "score_count": row.score_count,
        "score_mean": f"{row.score_mean:.6f}",
        "score_median": f"{row.score_median:.6f}",
        "score_std": f"{row.score_std:.6f}",
        "score_min": f"{row.score_min:.6f}",
        "score_max": f"{row.score_max:.6f}",
        "score_q1": f"{row.score_q1:.6f}",
        "score_q3": f"{row.score_q3:.6f}",
        "score_iqr": f"{row.score_iqr:.6f}",
    }


def write_score_stats_csv(path: Path, rows: list[ScoreStatsRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r.corpus_id):
            writer.writerow(_row_to_csv_dict(row))


def compute_and_write_score_stats_for_corpora(
    corpus_ids: list[str],
    outputs_root: Path,
) -> list[ScoreStatsRow]:
    analysis_tables_dir = outputs_root / "analysis" / "tables"
    paper_pack_tables_dir = outputs_root / "paper_pack" / "tables"
    stats_rows: list[ScoreStatsRow] = []

    for corpus_id in sorted(corpus_ids):
        score_path = outputs_root / "speciteller" / f"{corpus_id}_scores.tsv"
        qa_analysis = analysis_tables_dir / f"{corpus_id}_score_stats_qa.txt"
        qa_paper = paper_pack_tables_dir / f"{corpus_id}_score_stats_qa.txt"
        stats = compute_score_stats_from_file(
            corpus_id=corpus_id,
            score_path=score_path,
            qa_note_paths=[qa_analysis, qa_paper],
        )
        stats_rows.append(stats)

        write_score_stats_csv(analysis_tables_dir / f"{corpus_id}_score_stats.csv", [stats])
        write_score_stats_csv(paper_pack_tables_dir / f"{corpus_id}_score_stats.csv", [stats])

    write_score_stats_csv(analysis_tables_dir / "corpus_score_stats_all.csv", stats_rows)
    write_score_stats_csv(paper_pack_tables_dir / "corpus_score_stats_all.csv", stats_rows)
    return stats_rows
