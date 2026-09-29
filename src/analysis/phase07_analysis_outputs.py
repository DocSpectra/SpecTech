"""Phase 7: analysis + visualization outputs (deterministic)."""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import sys


# Some normalized sentence rows (especially HTML-derived) can exceed Python's
# default CSV field-size limit on Windows. Raise limit deterministically for
# robust analysis table loading.
csv.field_size_limit(min(sys.maxsize, 2**31 - 1))


@dataclass(frozen=True)
class AnalysisRow:
    corpus_id: str
    doc_path: str
    sent_id: str
    sent_text: str
    score: float
    tfidf_mean_nonzero: float
    tfidf_max: float
    technical_token_ratio: float
    identifier_density: float
    command_path_flag_density: float
    version_numeric_density: float
    assignment_parameter_density: float
    token_shape_complexity_mean: float
    token_count: int
    char_count: int


def _read_sentences(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["sent_id"]: {
            "corpus_id": row["corpus_id"],
            "doc_path": row["doc_path"],
            "sent_text": row["sent_text"],
        }
        for row in rows
    }


def _read_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            scores[row[0]] = float(row[1])
    return scores


def _read_features(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row["sent_id"]: row for row in rows}


def load_analysis_rows(corpus_id: str, outputs_root: Path) -> list[AnalysisRow]:
    sentence_path = outputs_root / "sentences" / f"{corpus_id}.csv"
    score_path = outputs_root / "speciteller" / f"{corpus_id}_scores.tsv"
    feature_path = outputs_root / "features" / f"{corpus_id}_features.csv"

    sentence_rows = _read_sentences(sentence_path)
    score_rows = _read_scores(score_path)
    feature_rows = _read_features(feature_path)

    sentence_ids = set(sentence_rows)
    score_ids = set(score_rows)
    feature_ids = set(feature_rows)
    if sentence_ids != score_ids or sentence_ids != feature_ids:
        missing_scores = len(sentence_ids - score_ids)
        missing_features = len(sentence_ids - feature_ids)
        extra_scores = len(score_ids - sentence_ids)
        extra_features = len(feature_ids - sentence_ids)
        raise ValueError(
            "Phase 7 join mismatch "
            f"for {corpus_id}: missing_scores={missing_scores}, "
            f"missing_features={missing_features}, "
            f"extra_scores={extra_scores}, extra_features={extra_features}"
        )

    sent_ids = sorted(sentence_ids)
    merged: list[AnalysisRow] = []
    for sent_id in sent_ids:
        s = sentence_rows[sent_id]
        f = feature_rows[sent_id]
        merged.append(
            AnalysisRow(
                corpus_id=corpus_id,
                doc_path=s["doc_path"],
                sent_id=sent_id,
                sent_text=s["sent_text"],
                score=score_rows[sent_id],
                tfidf_mean_nonzero=float(f["tfidf_mean_nonzero"]),
                tfidf_max=float(f["tfidf_max"]),
                technical_token_ratio=float(f["technical_token_ratio"]),
                identifier_density=float(f.get("identifier_density", "0") or 0.0),
                command_path_flag_density=float(f.get("command_path_flag_density", "0") or 0.0),
                version_numeric_density=float(f.get("version_numeric_density", "0") or 0.0),
                assignment_parameter_density=float(f.get("assignment_parameter_density", "0") or 0.0),
                token_shape_complexity_mean=float(f.get("token_shape_complexity_mean", "0") or 0.0),
                token_count=int(f["token_count"]),
                char_count=int(f["char_count"]),
            )
        )
    return merged


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda x: (x[1], x[0]))
    out = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j + 2) / 2.0  # 1-based ranks
        for k in range(i, j + 1):
            out[indexed[k][0]] = avg_rank
        i = j + 1
    return out


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mx = mean(xs)
    my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return 0.0
    return num / den


def spearman(xs: list[float], ys: list[float]) -> float:
    return _pearson(_ranks(xs), _ranks(ys))


def write_corpus_stats_table(path: Path, rows: list[AnalysisRow]) -> None:
    doc_count = len({row.doc_path for row in rows})
    sentence_count = len(rows)
    tokens = [row.token_count for row in rows]
    chars = [row.char_count for row in rows]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "corpus_id",
                "doc_count",
                "sentence_count",
                "token_mean",
                "token_median",
                "token_min",
                "token_max",
                "char_mean",
                "char_median",
            ]
        )
        writer.writerow(
            [
                rows[0].corpus_id if rows else "",
                doc_count,
                sentence_count,
                f"{mean(tokens):.6f}" if tokens else "0.0",
                f"{median(tokens):.6f}" if tokens else "0.0",
                min(tokens) if tokens else 0,
                max(tokens) if tokens else 0,
                f"{mean(chars):.6f}" if chars else "0.0",
                f"{median(chars):.6f}" if chars else "0.0",
            ]
        )


def write_correlation_table(path: Path, rows: list[AnalysisRow]) -> None:
    scores = [row.score for row in rows]
    features = {
        "tfidf_mean_nonzero": [row.tfidf_mean_nonzero for row in rows],
        "tfidf_max": [row.tfidf_max for row in rows],
        "technical_token_ratio": [row.technical_token_ratio for row in rows],
        "identifier_density": [row.identifier_density for row in rows],
        "command_path_flag_density": [row.command_path_flag_density for row in rows],
        "version_numeric_density": [row.version_numeric_density for row in rows],
        "assignment_parameter_density": [row.assignment_parameter_density for row in rows],
        "token_shape_complexity_mean": [row.token_shape_complexity_mean for row in rows],
        "token_count": [float(row.token_count) for row in rows],
        "char_count": [float(row.char_count) for row in rows],
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["corpus_id", "feature", "spearman_rho"])
        for feature_name, values in features.items():
            writer.writerow(
                [
                    rows[0].corpus_id if rows else "",
                    feature_name,
                    f"{spearman(scores, values):.8f}",
                ]
            )


def _write_svg_histogram(
    path: Path,
    values: list[float],
    title: str,
    bins: int = 20,
    x_label: str = "Specificity score",
    y_label: str = "Sentence count",
) -> None:
    bins = max(1, bins)
    counts = [0] * bins
    for value in values:
        idx = min(bins - 1, max(0, int(value * bins)))
        counts[idx] += 1

    width = 840
    height = 420
    margin = 50
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    bar_w = plot_w / bins
    max_count = max(counts) if counts else 1
    x_min = 0.0
    x_max = 1.0

    bars: list[str] = []
    for i, count in enumerate(counts):
        h = 0 if max_count == 0 else (count / max_count) * plot_h
        x = margin + i * bar_w
        y = margin + (plot_h - h)
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1.0, bar_w - 1):.2f}" height="{h:.2f}" fill="#4c78a8"/>'
        )

    ticks: list[str] = []
    tick_n = 5
    for i in range(tick_n + 1):
        # x ticks
        tx = margin + (plot_w * i / tick_n)
        x_val = x_min + ((x_max - x_min) * i / tick_n)
        ticks.append(
            f'<line x1="{tx:.2f}" y1="{margin + plot_h:.2f}" x2="{tx:.2f}" y2="{margin + plot_h + 5:.2f}" stroke="black"/>'
        )
        ticks.append(
            f'<text x="{tx:.2f}" y="{margin + plot_h + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{x_val:.2f}</text>'
        )

        # y ticks
        ty = margin + plot_h - (plot_h * i / tick_n)
        y_val = (max_count * i / tick_n)
        ticks.append(
            f'<line x1="{margin - 5:.2f}" y1="{ty:.2f}" x2="{margin:.2f}" y2="{ty:.2f}" stroke="black"/>'
        )
        ticks.append(
            f'<text x="{margin - 8:.2f}" y="{ty + 3:.2f}" text-anchor="end" font-size="10" font-family="Arial">{int(round(y_val))}</text>'
        )

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{margin}" y="28" font-size="16" font-family="Arial">{title}</text>',
            f'<line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" stroke="black"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" stroke="black"/>',
            f'<text x="{margin + (plot_w / 2):.2f}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="Arial">{x_label}</text>',
            f'<text x="16" y="{margin + (plot_h / 2):.2f}" transform="rotate(-90 16 {margin + (plot_h / 2):.2f})" text-anchor="middle" font-size="12" font-family="Arial">{y_label}</text>',
            *ticks,
            *bars,
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _write_svg_binned_curve(
    path: Path,
    x_values: list[float],
    y_values: list[float],
    title: str,
    bins: int = 20,
    x_label: str = "Feature value",
    y_label: str = "Specificity score",
) -> None:
    bins = max(1, bins)
    pairs = sorted(zip(x_values, y_values), key=lambda p: (p[0], p[1]))
    if not pairs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        return

    grouped: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for idx, pair in enumerate(pairs):
        bucket = min(bins - 1, int(idx * bins / len(pairs)))
        grouped[bucket].append(pair)

    points: list[tuple[float, float]] = []
    for group in grouped:
        if not group:
            continue
        xs = [p[0] for p in group]
        ys = [p[1] for p in group]
        points.append((mean(xs), mean(ys)))

    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)
    x_span = (max_x - min_x) if max_x != min_x else 1.0
    y_span = (max_y - min_y) if max_y != min_y else 1.0

    width = 840
    height = 420
    margin = 50
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin

    def sx(v: float) -> float:
        return margin + ((v - min_x) / x_span) * plot_w

    def sy(v: float) -> float:
        return margin + (1.0 - ((v - min_y) / y_span)) * plot_h

    polyline = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
    circles = [f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="3" fill="#f58518"/>' for x, y in points]

    ticks: list[str] = []
    tick_n = 5
    for i in range(tick_n + 1):
        # x ticks
        tx = margin + (plot_w * i / tick_n)
        x_val = min_x + (x_span * i / tick_n)
        ticks.append(
            f'<line x1="{tx:.2f}" y1="{margin + plot_h:.2f}" x2="{tx:.2f}" y2="{margin + plot_h + 5:.2f}" stroke="black"/>'
        )
        ticks.append(
            f'<text x="{tx:.2f}" y="{margin + plot_h + 18:.2f}" text-anchor="middle" font-size="10" font-family="Arial">{x_val:.2f}</text>'
        )

        # y ticks
        ty = margin + plot_h - (plot_h * i / tick_n)
        y_val = min_y + (y_span * i / tick_n)
        ticks.append(
            f'<line x1="{margin - 5:.2f}" y1="{ty:.2f}" x2="{margin:.2f}" y2="{ty:.2f}" stroke="black"/>'
        )
        ticks.append(
            f'<text x="{margin - 8:.2f}" y="{ty + 3:.2f}" text-anchor="end" font-size="10" font-family="Arial">{y_val:.2f}</text>'
        )

    svg = "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="{margin}" y="28" font-size="16" font-family="Arial">{title}</text>',
            f'<line x1="{margin}" y1="{margin + plot_h}" x2="{margin + plot_w}" y2="{margin + plot_h}" stroke="black"/>',
            f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{margin + plot_h}" stroke="black"/>',
            f'<text x="{margin + (plot_w / 2):.2f}" y="{height - 12}" text-anchor="middle" font-size="12" font-family="Arial">{x_label}</text>',
            f'<text x="16" y="{margin + (plot_h / 2):.2f}" transform="rotate(-90 16 {margin + (plot_h / 2):.2f})" text-anchor="middle" font-size="12" font-family="Arial">{y_label}</text>',
            *ticks,
            f'<polyline points="{polyline}" fill="none" stroke="#f58518" stroke-width="2"/>',
            *circles,
            '</svg>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")


def _proxy_index(rows: list[AnalysisRow]) -> dict[str, float]:
    tfidf_ranks = _ranks([row.tfidf_mean_nonzero for row in rows])
    tech_ranks = _ranks([row.technical_token_ratio for row in rows])
    token_ranks = _ranks([float(row.token_count) for row in rows])
    n = max(1, len(rows))
    return {
        row.sent_id: ((tfidf_ranks[i] + tech_ranks[i] + token_ranks[i]) / 3.0) / n
        for i, row in enumerate(rows)
    }


def write_divergence_samples(path: Path, rows: list[AnalysisRow], per_bucket: int = 20) -> None:
    proxy = _proxy_index(rows)
    sorted_by_score = sorted(rows, key=lambda r: (r.score, r.sent_id))
    n = len(sorted_by_score)
    q = max(1, n // 4)
    low_score = sorted_by_score[:q]
    high_score = sorted_by_score[-q:]

    low_proxy_sorted = sorted(rows, key=lambda r: (proxy[r.sent_id], r.sent_id))
    low_proxy = set(r.sent_id for r in low_proxy_sorted[:q])
    high_proxy = set(r.sent_id for r in low_proxy_sorted[-q:])

    high_score_low_proxy = sorted(
        [r for r in high_score if r.sent_id in low_proxy], key=lambda r: (-r.score, r.sent_id)
    )[:per_bucket]
    low_score_high_proxy = sorted(
        [r for r in low_score if r.sent_id in high_proxy], key=lambda r: (r.score, r.sent_id)
    )[:per_bucket]
    extreme_low = sorted_by_score[:per_bucket]
    mid_center = n // 2
    mid_range = sorted(
        sorted_by_score[max(0, mid_center - per_bucket * 2): min(n, mid_center + per_bucket * 2)],
        key=lambda r: (abs(r.score - sorted_by_score[mid_center].score), r.sent_id),
    )[:per_bucket]

    rows_out: list[tuple[str, AnalysisRow]] = []
    rows_out.extend(("high_score_low_proxy", row) for row in high_score_low_proxy)
    rows_out.extend(("low_score_high_proxy", row) for row in low_score_high_proxy)
    rows_out.extend(("extreme_low", row) for row in extreme_low)
    rows_out.extend(("mid_range", row) for row in mid_range)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bucket",
                "corpus_id",
                "doc_path",
                "sent_id",
                "score",
                "proxy_index",
                "tfidf_mean_nonzero",
                "tfidf_max",
                "technical_token_ratio",
                "identifier_density",
                "command_path_flag_density",
                "version_numeric_density",
                "assignment_parameter_density",
                "token_shape_complexity_mean",
                "token_count",
                "char_count",
                "sent_text",
            ]
        )
        for bucket, row in rows_out:
            writer.writerow(
                [
                    bucket,
                    row.corpus_id,
                    row.doc_path,
                    row.sent_id,
                    f"{row.score:.8f}",
                    f"{proxy[row.sent_id]:.8f}",
                    f"{row.tfidf_mean_nonzero:.8f}",
                    f"{row.tfidf_max:.8f}",
                    f"{row.technical_token_ratio:.8f}",
                    f"{row.identifier_density:.8f}",
                    f"{row.command_path_flag_density:.8f}",
                    f"{row.version_numeric_density:.8f}",
                    f"{row.assignment_parameter_density:.8f}",
                    f"{row.token_shape_complexity_mean:.8f}",
                    row.token_count,
                    row.char_count,
                    row.sent_text,
                ]
            )


def run_phase7_for_corpus(corpus_id: str, outputs_root: Path = Path("outputs")) -> None:
    rows = load_analysis_rows(corpus_id, outputs_root)
    if not rows:
        return

    tables_dir = outputs_root / "analysis" / "tables"
    figures_dir = outputs_root / "analysis" / "figures"
    samples_dir = outputs_root / "analysis" / "samples"

    write_corpus_stats_table(tables_dir / f"{corpus_id}_corpus_stats.csv", rows)
    write_correlation_table(tables_dir / f"{corpus_id}_score_feature_spearman.csv", rows)

    _write_svg_histogram(
        figures_dir / f"{corpus_id}_score_distribution.svg",
        [row.score for row in rows],
        title=f"{corpus_id}: specificity score distribution",
        x_label="Specificity score",
        y_label="Sentence count",
    )
    _write_svg_binned_curve(
        figures_dir / f"{corpus_id}_score_vs_tfidf_mean.svg",
        [row.tfidf_mean_nonzero for row in rows],
        [row.score for row in rows],
        title=f"{corpus_id}: score vs tfidf_mean_nonzero (binned)",
        x_label="TF-IDF mean (non-zero)",
        y_label="Specificity score",
    )
    _write_svg_binned_curve(
        figures_dir / f"{corpus_id}_score_vs_tfidf_max.svg",
        [row.tfidf_max for row in rows],
        [row.score for row in rows],
        title=f"{corpus_id}: score vs tfidf_max (binned)",
        x_label="TF-IDF max",
        y_label="Specificity score",
    )
    _write_svg_binned_curve(
        figures_dir / f"{corpus_id}_score_vs_technical_ratio.svg",
        [row.technical_token_ratio for row in rows],
        [row.score for row in rows],
        title=f"{corpus_id}: score vs technical_token_ratio (binned)",
        x_label="Technical token ratio",
        y_label="Specificity score",
    )

    write_divergence_samples(samples_dir / f"{corpus_id}_divergence_samples.csv", rows)
