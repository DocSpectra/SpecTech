"""Phase 7.5: controlled-edit behavioral evaluation utilities."""
from __future__ import annotations

import csv
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Iterable

from src.analysis.phase07_analysis_outputs import AnalysisRow
from src.speciteller.config import SpeciTellerConfig
from src.speciteller.runner import preflight_speciteller
from src.speciteller.runner import run_speciteller
from src.speciteller.tokenize import tokenize_for_speciteller


EDIT_TYPES: tuple[str, str, str] = ("de_specify", "add_specific", "irrelevant_rewrite")


@dataclass(frozen=True)
class ControlledEditTemplateRow:
    sent_id: str
    corpus_id: str
    sentence_original: str
    speciteller_score_original: float
    token_count: int
    edit_type: str
    sentence_edited: str = ""


@dataclass(frozen=True)
class ScoredControlledEditRow:
    sent_id: str
    corpus_id: str
    sentence_original: str
    speciteller_score_original: float
    token_count: int
    edit_type: str
    sentence_edited: str
    speciteller_score_edited: float
    delta: float


@dataclass(frozen=True)
class ControlledEditStatRow:
    edit_type: str
    count: int
    mean_delta: float
    median_delta: float
    percent_directionally_correct: float | None
    mean_absolute_delta: float


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    q = min(1.0, max(0.0, q))
    idx = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[idx]


def _sample_from_pool(
    pool: list[AnalysisRow],
    count: int,
    rng: random.Random,
    selected_ids: set[str],
) -> list[AnalysisRow]:
    candidates = [row for row in pool if row.sent_id not in selected_ids]
    if count <= 0 or not candidates:
        return []
    ordered = sorted(candidates, key=lambda row: row.sent_id)
    rng.shuffle(ordered)
    sampled = sorted(ordered[: min(count, len(ordered))], key=lambda row: row.sent_id)
    selected_ids.update(row.sent_id for row in sampled)
    return sampled


def build_controlled_edit_template_rows(
    rows: list[AnalysisRow],
    *,
    bottom_count: int = 10,
    median_count: int = 10,
    top_count: int = 10,
    length_count: int = 0,
    seed: int = 13,
) -> list[ControlledEditTemplateRow]:
    """Deterministically sample and assign controlled-edit templates."""
    if not rows:
        return []

    rng = random.Random(seed)
    scores_sorted = sorted(row.score for row in rows)
    bottom_threshold = _quantile(scores_sorted, 0.10)
    top_threshold = _quantile(scores_sorted, 0.90)

    selected_ids: set[str] = set()
    selected: list[AnalysisRow] = []

    bottom_pool = [row for row in rows if row.score <= bottom_threshold]
    selected.extend(_sample_from_pool(bottom_pool, bottom_count, rng, selected_ids))

    top_pool = [row for row in rows if row.score >= top_threshold]
    selected.extend(_sample_from_pool(top_pool, top_count, rng, selected_ids))

    score_med = median([row.score for row in rows])
    median_pool = sorted(rows, key=lambda row: (abs(row.score - score_med), row.sent_id))
    selected.extend(_sample_from_pool(median_pool, median_count, rng, selected_ids))

    if length_count > 0:
        length_med = median([row.token_count for row in rows])
        length_pool = sorted(
            rows,
            key=lambda row: (
                abs(row.token_count - length_med),
                abs(row.score - score_med),
                row.sent_id,
            ),
        )
        selected.extend(_sample_from_pool(length_pool, length_count, rng, selected_ids))

    selected = sorted(selected, key=lambda row: row.sent_id)
    template_rows: list[ControlledEditTemplateRow] = []
    for idx, row in enumerate(selected):
        template_rows.append(
            ControlledEditTemplateRow(
                sent_id=row.sent_id,
                corpus_id=row.corpus_id,
                sentence_original=row.sent_text,
                speciteller_score_original=row.score,
                token_count=row.token_count,
                edit_type=EDIT_TYPES[idx % len(EDIT_TYPES)],
                sentence_edited="",
            )
        )
    return template_rows


def write_controlled_edit_template(path: Path, rows: Iterable[ControlledEditTemplateRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sent_id",
                "corpus_id",
                "sentence_original",
                "speciteller_score_original",
                "token_count",
                "edit_type",
                "sentence_edited",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sent_id,
                    row.corpus_id,
                    row.sentence_original,
                    f"{row.speciteller_score_original:.8f}",
                    row.token_count,
                    row.edit_type,
                    row.sentence_edited,
                ]
            )


def load_controlled_edit_template(path: Path) -> list[ControlledEditTemplateRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    out: list[ControlledEditTemplateRow] = []
    for row in rows:
        edit_type = row["edit_type"].strip()
        if edit_type not in EDIT_TYPES:
            raise ValueError(f"Unsupported edit_type: {edit_type}")
        out.append(
            ControlledEditTemplateRow(
                sent_id=row["sent_id"],
                corpus_id=row["corpus_id"],
                sentence_original=row["sentence_original"],
                speciteller_score_original=float(row["speciteller_score_original"]),
                token_count=int(row["token_count"]),
                edit_type=edit_type,
                sentence_edited=row.get("sentence_edited", ""),
            )
        )
    return out


def rescore_completed_controlled_edits(
    template_rows: list[ControlledEditTemplateRow],
    *,
    config: SpeciTellerConfig,
    outputs_dir: Path,
    preflight: bool = True,
) -> list[ScoredControlledEditRow]:
    completed_rows = [row for row in template_rows if row.sentence_edited.strip()]
    if not completed_rows:
        raise ValueError("No completed controlled edits found (sentence_edited is blank).")

    outputs_dir.mkdir(parents=True, exist_ok=True)
    input_path = outputs_dir / "controlled_edits_input.tsv"
    scored_path = outputs_dir / "controlled_edits_scores.tsv"

    with input_path.open("w", encoding="utf-8") as handle:
        for row in completed_rows:
            tokenized = tokenize_for_speciteller(row.sentence_edited)
            handle.write(f"{row.sent_id}\t{tokenized}\n")

    if preflight:
        preflight_speciteller(config)
    run_speciteller(config, input_path, scored_path)

    score_map: dict[str, float] = {}
    with scored_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for score_row in reader:
            if not score_row:
                continue
            score_map[score_row[0]] = float(score_row[1])

    scored_rows: list[ScoredControlledEditRow] = []
    for row in completed_rows:
        if row.sent_id not in score_map:
            raise ValueError(f"Missing edited score for sent_id={row.sent_id}")
        edited_score = score_map[row.sent_id]
        delta = edited_score - row.speciteller_score_original
        scored_rows.append(
            ScoredControlledEditRow(
                sent_id=row.sent_id,
                corpus_id=row.corpus_id,
                sentence_original=row.sentence_original,
                speciteller_score_original=row.speciteller_score_original,
                token_count=row.token_count,
                edit_type=row.edit_type,
                sentence_edited=row.sentence_edited,
                speciteller_score_edited=edited_score,
                delta=delta,
            )
        )
    return scored_rows


def compute_controlled_edit_stats(rows: list[ScoredControlledEditRow]) -> list[ControlledEditStatRow]:
    by_type: dict[str, list[ScoredControlledEditRow]] = {k: [] for k in EDIT_TYPES}
    for row in rows:
        by_type.setdefault(row.edit_type, []).append(row)

    stats: list[ControlledEditStatRow] = []
    for edit_type in EDIT_TYPES:
        group = by_type.get(edit_type, [])
        deltas = [row.delta for row in group]
        if not deltas:
            stats.append(
                ControlledEditStatRow(
                    edit_type=edit_type,
                    count=0,
                    mean_delta=0.0,
                    median_delta=0.0,
                    percent_directionally_correct=None,
                    mean_absolute_delta=0.0,
                )
            )
            continue

        directionally_correct: float | None
        if edit_type == "de_specify":
            directionally_correct = 100.0 * sum(1 for d in deltas if d < 0) / len(deltas)
        elif edit_type == "add_specific":
            directionally_correct = 100.0 * sum(1 for d in deltas if d > 0) / len(deltas)
        else:
            directionally_correct = None

        stats.append(
            ControlledEditStatRow(
                edit_type=edit_type,
                count=len(group),
                mean_delta=mean(deltas),
                median_delta=median(deltas),
                percent_directionally_correct=directionally_correct,
                mean_absolute_delta=mean(abs(d) for d in deltas),
            )
        )
    return stats


def write_scored_controlled_edits(path: Path, rows: Iterable[ScoredControlledEditRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "sent_id",
                "corpus_id",
                "sentence_original",
                "speciteller_score_original",
                "token_count",
                "edit_type",
                "sentence_edited",
                "speciteller_score_edited",
                "delta",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.sent_id,
                    row.corpus_id,
                    row.sentence_original,
                    f"{row.speciteller_score_original:.8f}",
                    row.token_count,
                    row.edit_type,
                    row.sentence_edited,
                    f"{row.speciteller_score_edited:.8f}",
                    f"{row.delta:.8f}",
                ]
            )


def write_controlled_edit_stats(path: Path, rows: Iterable[ControlledEditStatRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "edit_type",
                "count",
                "mean_delta",
                "median_delta",
                "percent_directionally_correct",
                "mean_absolute_delta",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.edit_type,
                    row.count,
                    f"{row.mean_delta:.8f}",
                    f"{row.median_delta:.8f}",
                    "" if row.percent_directionally_correct is None else f"{row.percent_directionally_correct:.4f}",
                    f"{row.mean_absolute_delta:.8f}",
                ]
            )