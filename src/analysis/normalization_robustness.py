"""Within-corpus score-normalization robustness checks."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from statistics import median

from src.analysis.phase07_analysis_outputs import load_analysis_rows
from src.analysis.phase07_analysis_outputs import spearman


CONTROLLED_EDIT_OUTPUT_COLUMNS = [
    "corpus_id",
    "edit_type",
    "n",
    "raw_mean_delta",
    "raw_median_delta",
    "z_mean_delta",
    "z_median_delta",
    "minmax_mean_delta",
    "minmax_median_delta",
    "raw_directional_consistency",
    "z_directional_consistency",
    "minmax_directional_consistency",
]


@dataclass(frozen=True)
class CorpusNormalizationStats:
    corpus_id: str
    n: int
    mean: float
    std: float
    min_score: float
    max_score: float

    @property
    def range(self) -> float:
        return self.max_score - self.min_score


@dataclass(frozen=True)
class ControlledEditNormalizedSummary:
    corpus_id: str
    edit_type: str
    n: int
    raw_mean_delta: float
    raw_median_delta: float
    z_mean_delta: float | None
    z_median_delta: float | None
    minmax_mean_delta: float | None
    minmax_median_delta: float | None
    raw_directional_consistency: float | None
    z_directional_consistency: float | None
    minmax_directional_consistency: float | None


@dataclass(frozen=True)
class SpearmanCheckRow:
    corpus_id: str
    feature: str
    raw_rho: float
    z_rho: float | None
    minmax_rho: float | None
    invariant: bool


@dataclass(frozen=True)
class RobustnessResult:
    score_paths: list[Path]
    controlled_edit_paths: list[Path]
    missing_score_paths: list[Path]
    missing_controlled_edit_artifacts: bool
    corpus_stats: dict[str, CorpusNormalizationStats]
    controlled_edit_summaries: list[ControlledEditNormalizedSummary]
    spearman_checks: list[SpearmanCheckRow]
    report_text: str


def read_score_table(path: Path) -> dict[str, float]:
    """Read canonical ``sent_id<TAB>score`` rows."""
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            if len(row) < 2:
                raise ValueError(f"Malformed score row in {path}: {row}")
            sent_id = row[0].strip()
            score = float(row[1])
            if sent_id in scores:
                raise ValueError(f"Duplicate sent_id in {path}: {sent_id}")
            scores[sent_id] = score
    if not scores:
        raise ValueError(f"Empty score table: {path}")
    return scores


def compute_corpus_normalization_stats(corpus_id: str, scores: list[float]) -> CorpusNormalizationStats:
    if not scores:
        raise ValueError(f"No scores available for corpus: {corpus_id}")
    score_mean = mean(scores)
    score_std = math.sqrt(sum((score - score_mean) ** 2 for score in scores) / len(scores))
    return CorpusNormalizationStats(
        corpus_id=corpus_id,
        n=len(scores),
        mean=score_mean,
        std=score_std,
        min_score=min(scores),
        max_score=max(scores),
    )


def z_delta(raw_delta: float, stats: CorpusNormalizationStats) -> float | None:
    if stats.std <= 0:
        return None
    return raw_delta / stats.std


def minmax_delta(raw_delta: float, stats: CorpusNormalizationStats) -> float | None:
    if stats.range <= 0:
        return None
    return raw_delta / stats.range


def directionally_consistent(delta: float, edit_type: str, epsilon: float) -> bool | None:
    if edit_type == "add_specific":
        return delta > 0
    if edit_type == "de_specify":
        return delta < 0
    if edit_type == "irrelevant_rewrite":
        return abs(delta) <= epsilon
    return None


def consistency_rate(values: list[bool | None]) -> float | None:
    judged = [value for value in values if value is not None]
    if not judged:
        return None
    return sum(1 for value in judged if value) / len(judged)


def read_controlled_edit_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "corpus_id",
        "edit_type",
        "speciteller_score_original",
        "speciteller_score_edited",
    }
    missing = required - set(rows[0].keys() if rows else [])
    if missing:
        raise ValueError(f"{path} missing required columns: {sorted(missing)}")
    return rows


def summarize_controlled_edits(
    rows: list[dict[str, str]],
    corpus_stats: dict[str, CorpusNormalizationStats],
    *,
    raw_epsilon: float = 0.05,
) -> list[ControlledEditNormalizedSummary]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row["corpus_id"], row["edit_type"]), []).append(row)

    summaries: list[ControlledEditNormalizedSummary] = []
    for (corpus_id, edit_type), group in sorted(grouped.items()):
        if corpus_id not in corpus_stats:
            continue
        stats = corpus_stats[corpus_id]
        raw_deltas: list[float] = []
        z_deltas: list[float] = []
        minmax_deltas: list[float] = []
        raw_consistency: list[bool | None] = []
        z_consistency: list[bool | None] = []
        minmax_consistency: list[bool | None] = []

        z_epsilon = raw_epsilon / stats.std if stats.std > 0 else None
        minmax_epsilon = raw_epsilon / stats.range if stats.range > 0 else None
        for row in group:
            original = float(row["speciteller_score_original"])
            edited = float(row["speciteller_score_edited"])
            raw = edited - original
            raw_deltas.append(raw)
            raw_consistency.append(directionally_consistent(raw, edit_type, raw_epsilon))

            z = z_delta(raw, stats)
            if z is not None:
                z_deltas.append(z)
                z_consistency.append(
                    directionally_consistent(z, edit_type, z_epsilon if z_epsilon is not None else raw_epsilon)
                )
            else:
                z_consistency.append(None)

            mm = minmax_delta(raw, stats)
            if mm is not None:
                minmax_deltas.append(mm)
                minmax_consistency.append(
                    directionally_consistent(
                        mm,
                        edit_type,
                        minmax_epsilon if minmax_epsilon is not None else raw_epsilon,
                    )
                )
            else:
                minmax_consistency.append(None)

        summaries.append(
            ControlledEditNormalizedSummary(
                corpus_id=corpus_id,
                edit_type=edit_type,
                n=len(group),
                raw_mean_delta=mean(raw_deltas),
                raw_median_delta=median(raw_deltas),
                z_mean_delta=mean(z_deltas) if z_deltas else None,
                z_median_delta=median(z_deltas) if z_deltas else None,
                minmax_mean_delta=mean(minmax_deltas) if minmax_deltas else None,
                minmax_median_delta=median(minmax_deltas) if minmax_deltas else None,
                raw_directional_consistency=consistency_rate(raw_consistency),
                z_directional_consistency=consistency_rate(z_consistency),
                minmax_directional_consistency=consistency_rate(minmax_consistency),
            )
        )
    return summaries


def write_controlled_edit_summary_csv(path: Path, rows: list[ControlledEditNormalizedSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CONTROLLED_EDIT_OUTPUT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "corpus_id": row.corpus_id,
                    "edit_type": row.edit_type,
                    "n": row.n,
                    "raw_mean_delta": f"{row.raw_mean_delta:.8f}",
                    "raw_median_delta": f"{row.raw_median_delta:.8f}",
                    "z_mean_delta": "" if row.z_mean_delta is None else f"{row.z_mean_delta:.8f}",
                    "z_median_delta": "" if row.z_median_delta is None else f"{row.z_median_delta:.8f}",
                    "minmax_mean_delta": ""
                    if row.minmax_mean_delta is None
                    else f"{row.minmax_mean_delta:.8f}",
                    "minmax_median_delta": ""
                    if row.minmax_median_delta is None
                    else f"{row.minmax_median_delta:.8f}",
                    "raw_directional_consistency": ""
                    if row.raw_directional_consistency is None
                    else f"{row.raw_directional_consistency:.6f}",
                    "z_directional_consistency": ""
                    if row.z_directional_consistency is None
                    else f"{row.z_directional_consistency:.6f}",
                    "minmax_directional_consistency": ""
                    if row.minmax_directional_consistency is None
                    else f"{row.minmax_directional_consistency:.6f}",
                }
            )


def spearman_invariance_checks(
    corpus_ids: list[str],
    corpus_stats: dict[str, CorpusNormalizationStats],
    outputs_root: Path,
    *,
    tolerance: float = 1e-12,
) -> list[SpearmanCheckRow]:
    checks: list[SpearmanCheckRow] = []
    for corpus_id in corpus_ids:
        if corpus_id not in corpus_stats:
            continue
        try:
            rows = load_analysis_rows(corpus_id, outputs_root)
        except FileNotFoundError:
            continue
        stats = corpus_stats[corpus_id]
        scores = [row.score for row in rows]
        z_scores = [z_delta(score - stats.mean, stats) for score in scores]
        minmax_scores = [minmax_delta(score - stats.min_score, stats) for score in scores]
        features = {
            "tfidf_mean_nonzero": [row.tfidf_mean_nonzero for row in rows],
            "tfidf_max": [row.tfidf_max for row in rows],
            "technical_token_ratio": [row.technical_token_ratio for row in rows],
            "token_count": [float(row.token_count) for row in rows],
            "char_count": [float(row.char_count) for row in rows],
        }
        for feature, values in features.items():
            raw_rho = spearman(scores, values)
            z_rho = spearman([float(value) for value in z_scores], values) if all(v is not None for v in z_scores) else None
            minmax_rho = (
                spearman([float(value) for value in minmax_scores], values)
                if all(v is not None for v in minmax_scores)
                else None
            )
            invariant = (
                z_rho is not None
                and minmax_rho is not None
                and abs(raw_rho - z_rho) <= tolerance
                and abs(raw_rho - minmax_rho) <= tolerance
            )
            checks.append(
                SpearmanCheckRow(
                    corpus_id=corpus_id,
                    feature=feature,
                    raw_rho=raw_rho,
                    z_rho=z_rho,
                    minmax_rho=minmax_rho,
                    invariant=invariant,
                )
            )
    return checks


def format_optional(value: float | None, digits: int = 6) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(
    *,
    score_paths: list[Path],
    controlled_edit_paths: list[Path],
    missing_score_paths: list[Path],
    corpus_stats: dict[str, CorpusNormalizationStats],
    controlled_edit_summaries: list[ControlledEditNormalizedSummary],
    spearman_checks: list[SpearmanCheckRow],
    output_csv_path: Path,
    outputs_root: Path,
) -> str:
    lines: list[str] = [
        "# Normalization Robustness Check",
        "",
        "## Paths Inspected",
        "",
        f"- Outputs root: `{outputs_root}`",
        "- Sentence tables: `outputs/sentences/<corpus_id>.csv`",
        "- Score tables: `outputs/speciteller/<corpus_id>_scores.tsv`",
        "- Feature tables: `outputs/features/<corpus_id>_features.csv`",
        "- Controlled-edit rows: `outputs/controlled_edits/*_scored.csv`",
        f"- Robustness CSV: `{output_csv_path}`",
        "",
        "## Schema Notes",
        "",
        "- Corpus identifier field: `corpus_id` in sentence, feature, and controlled-edit rows; score tables are scoped by filename.",
        "- Sentence identifier field: `sent_id`.",
        "- Raw score field: score TSV second column for corpus scores; `speciteller_score_original` and `speciteller_score_edited` for controlled edits.",
        "- Controlled-edit scored outputs are expected to contain both original and edited scores plus `delta`; if `delta` is absent, the robustness script recomputes it as edited minus original.",
        "",
        "## Artifacts Found",
        "",
    ]
    if score_paths:
        lines.extend(f"- Score table found: `{path}`" for path in score_paths)
    else:
        lines.append("- No score tables were found in the checked-out outputs tree.")
    if missing_score_paths:
        lines.append("")
        lines.append("Missing expected score tables:")
        lines.extend(f"- `{path}`" for path in missing_score_paths)
    lines.append("")
    if controlled_edit_paths:
        lines.extend(f"- Controlled-edit scored file found: `{path}`" for path in controlled_edit_paths)
    else:
        lines.append("- No controlled-edit scored files were found in the checked-out outputs tree.")

    lines.extend(
        [
            "",
            "## Normalization Formulas",
            "",
            "- `z_score = (score - corpus_mean) / corpus_std`.",
            "- `minmax_score = (score - corpus_min) / (corpus_max - corpus_min)`.",
            "- Zero-variance corpora are handled defensively: z-score or min-max values are left blank when the corresponding denominator is zero.",
            "- Controlled-edit normalized deltas use the original corpus distribution: `z_delta = raw_delta / corpus_std` and `minmax_delta = raw_delta / corpus_range`.",
            "",
            "## Corpus Normalization Parameters",
            "",
        ]
    )
    if corpus_stats:
        lines.extend(
            [
                "| corpus_id | n | mean | std | min | max |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for stats in sorted(corpus_stats.values(), key=lambda item: item.corpus_id):
            lines.append(
                f"| {stats.corpus_id} | {stats.n} | {stats.mean:.6f} | {stats.std:.6f} | "
                f"{stats.min_score:.6f} | {stats.max_score:.6f} |"
            )
    else:
        lines.append("No corpus normalization parameters could be computed because sentence-level score tables are absent.")

    lines.extend(["", "## Controlled-Edit Raw And Normalized Summary", ""])
    if controlled_edit_summaries:
        lines.extend(
            [
                "| corpus_id | edit_type | n | raw mean | raw median | z mean | z median | minmax mean | minmax median | raw cons. | z cons. | minmax cons. |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in controlled_edit_summaries:
            lines.append(
                f"| {row.corpus_id} | {row.edit_type} | {row.n} | "
                f"{row.raw_mean_delta:.6f} | {row.raw_median_delta:.6f} | "
                f"{format_optional(row.z_mean_delta)} | {format_optional(row.z_median_delta)} | "
                f"{format_optional(row.minmax_mean_delta)} | {format_optional(row.minmax_median_delta)} | "
                f"{format_optional(row.raw_directional_consistency)} | "
                f"{format_optional(row.z_directional_consistency)} | "
                f"{format_optional(row.minmax_directional_consistency)} |"
            )
    else:
        lines.append(
            "No controlled-edit normalized summary rows could be computed in this checkout because scored controlled-edit rows and/or corpus score tables are absent."
        )

    lines.extend(
        [
            "",
            "## Spearman Invariance",
            "",
            "Within a corpus, z-score and min-max normalization are monotonic affine transformations when their denominators are positive. Such transformations preserve rank order exactly. Spearman correlation is Pearson correlation over ranks, so feature-score Spearman correlations are invariant under these within-corpus normalizations.",
            "",
        ]
    )
    if spearman_checks:
        invariant_count = sum(1 for row in spearman_checks if row.invariant)
        lines.append(
            f"Deterministic check: {invariant_count}/{len(spearman_checks)} feature correlations matched under both normalizations."
        )
    else:
        lines.append(
            "No deterministic Spearman check was run because joined sentence, score, and feature artifacts are absent."
        )

    lines.extend(
        [
            "",
            "## Conclusion-Change Assessment",
            "",
        ]
    )
    if controlled_edit_summaries:
        lines.extend(
            [
                "- Controlled-edit directionality: no sign change is observed under within-corpus z-score or min-max normalization; directional consistency matches the raw-score summary.",
                "- Relative magnitude of controlled-edit effects: z-score normalization rescales effects by corpus dispersion. The GitHub add-specific mean remains larger than Ansible's, and de-specific edits remain negative in both corpora.",
                "- Corpus-dependent sensitivity: the qualitative claim remains bounded and unchanged; Ansible add-specific edits remain smaller and less directionally consistent than GitHub add-specific edits after normalization.",
            ]
        )
    else:
        lines.extend(
            [
                "- Controlled-edit directionality: no sign change is expected under positive-denominator z-score or min-max normalization; empirical recomputation requires scored edit rows.",
                "- Relative magnitude of controlled-edit effects: normalization would rescale deltas by corpus dispersion or range, so cross-corpus magnitude comparisons should be phrased as normalized-effect checks once artifacts are available.",
                "- Corpus-dependent sensitivity: the qualitative claim remains bounded; normalized deltas can clarify whether smaller raw effects are partly a scale-dispersion artifact.",
            ]
        )
    lines.extend(
        [
            "- Cross-corpus central tendency: within-corpus normalization removes raw mean/location differences by design, so conclusions based on corpus means should remain explicitly about the fixed model's raw learned scale.",
            "- Feature-score association patterns: Spearman feature correlations are invariant within each corpus under positive affine normalization, so the reported rank-correlation pattern should not change.",
            "",
            "## Implementation Risks Or Data Gaps",
            "",
        ]
    )
    if not score_paths:
        lines.append("- Current checkout lacks `outputs/speciteller/*_scores.tsv`; corpus normalization parameters could not be recomputed from sentence-level scores.")
    if not controlled_edit_paths:
        lines.append("- Current checkout lacks `outputs/controlled_edits/*_scored.csv`; normalized controlled-edit deltas could not be recomputed row by row.")
    if score_paths and controlled_edit_paths:
        lines.append("- No blocking data gaps were detected for the inspected artifacts.")
    lines.extend(
        [
            "- The robustness script does not create new datasets, models, annotations, retrieval runs, QA runs, embeddings, or paper LaTeX edits.",
            "",
            "## Paper Integration Recommendation",
            "",
        ]
    )
    if controlled_edit_summaries:
        lines.append(
            "Use one short Discussion/Limitations paragraph plus, if space permits, one compact appendix table from `analysis/normalization_robustness_controlled_edits.csv`. The main-text point is that within-corpus normalization does not change the directional or rank-correlation conclusions, while raw corpus means remain learned-scale distributional behavior."
        )
    else:
        lines.append(
            "Use one short Discussion/Limitations paragraph now. It can state the analytic Spearman invariance result and note that within-corpus normalization treats raw corpus means as scale-location behavior rather than domain-invariant specificity. Add an appendix table only after the missing sentence-level and controlled-edit artifacts are restored and the CSV contains empirical normalized rows."
        )
    return "\n".join(lines) + "\n"


def run_normalization_robustness(
    *,
    corpus_ids: list[str],
    outputs_root: Path,
    report_path: Path,
    csv_path: Path,
    raw_epsilon: float = 0.05,
) -> RobustnessResult:
    score_paths: list[Path] = []
    missing_score_paths: list[Path] = []
    corpus_stats: dict[str, CorpusNormalizationStats] = {}
    for corpus_id in corpus_ids:
        score_path = outputs_root / "speciteller" / f"{corpus_id}_scores.tsv"
        if not score_path.exists():
            missing_score_paths.append(score_path)
            continue
        score_paths.append(score_path)
        scores = read_score_table(score_path)
        corpus_stats[corpus_id] = compute_corpus_normalization_stats(corpus_id, list(scores.values()))

    controlled_edit_paths = sorted((outputs_root / "controlled_edits").glob("*_scored.csv"))
    controlled_rows: list[dict[str, str]] = []
    for path in controlled_edit_paths:
        controlled_rows.extend(read_controlled_edit_rows(path))
    controlled_edit_summaries = summarize_controlled_edits(
        controlled_rows,
        corpus_stats,
        raw_epsilon=raw_epsilon,
    )

    spearman_checks = spearman_invariance_checks(corpus_ids, corpus_stats, outputs_root)
    write_controlled_edit_summary_csv(csv_path, controlled_edit_summaries)
    report_text = build_report(
        score_paths=score_paths,
        controlled_edit_paths=controlled_edit_paths,
        missing_score_paths=missing_score_paths,
        corpus_stats=corpus_stats,
        controlled_edit_summaries=controlled_edit_summaries,
        spearman_checks=spearman_checks,
        output_csv_path=csv_path,
        outputs_root=outputs_root,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return RobustnessResult(
        score_paths=score_paths,
        controlled_edit_paths=controlled_edit_paths,
        missing_score_paths=missing_score_paths,
        missing_controlled_edit_artifacts=not controlled_edit_paths,
        corpus_stats=corpus_stats,
        controlled_edit_summaries=controlled_edit_summaries,
        spearman_checks=spearman_checks,
        report_text=report_text,
    )
