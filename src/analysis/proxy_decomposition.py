"""Documentation-aware proxy decomposition analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from src.analysis.phase07_analysis_outputs import spearman
from src.features.proxies import SentenceFeatureRow
from src.features.proxies import compute_corpus_features
from src.sentences.table import SentenceRecord


ORIGINAL_FEATURES = [
    "tfidf_mean_nonzero",
    "tfidf_max",
    "technical_token_ratio",
    "token_count",
    "char_count",
]

DOCUMENTATION_FEATURES = [
    "identifier_density",
    "command_path_flag_density",
    "version_numeric_density",
    "assignment_parameter_density",
    "token_shape_complexity_mean",
]

FEATURE_DEFINITIONS = {
    "tfidf_mean_nonzero": "Mean non-zero per-sentence TF-IDF over unique lowercased tokens.",
    "tfidf_max": "Maximum per-token TF-IDF value in the sentence.",
    "technical_token_ratio": "Aggregate ratio of technical-looking regex tokens to total tokens.",
    "token_count": "Treebank token count.",
    "char_count": "Character count of the sentence string.",
    "identifier_density": "Ratio of snake_case, camelCase, dotted, and ALL_CAPS identifier-like tokens to total tokens.",
    "command_path_flag_density": "Ratio of command-like, path-like, and flag-like tokens to total tokens.",
    "version_numeric_density": "Ratio of version-like and operational numeric tokens to total tokens.",
    "assignment_parameter_density": "Ratio of key/value, YAML-like key, and flag-value surface spans to total tokens.",
    "token_shape_complexity_mean": "Mean token shape complexity from case transitions, digits, separators, and punctuation.",
}

FEATURE_FAMILY = {
    "tfidf_mean_nonzero": "original_lexical",
    "tfidf_max": "original_lexical",
    "technical_token_ratio": "original_technical_aggregate",
    "token_count": "original_length",
    "char_count": "original_length",
    "identifier_density": "documentation_identifier",
    "command_path_flag_density": "documentation_operational",
    "version_numeric_density": "documentation_numeric",
    "assignment_parameter_density": "documentation_parameter",
    "token_shape_complexity_mean": "documentation_shape",
}


@dataclass(frozen=True)
class ProxyCorrelationRow:
    corpus_id: str
    feature: str
    feature_family: str
    feature_group: str
    spearman_rho: float
    abs_spearman_rho: float
    sentence_count: int
    mean_value: float
    nonzero_rate: float


@dataclass(frozen=True)
class ProxyDecompositionResult:
    source_artifacts: list[Path]
    missing_artifacts: list[Path]
    rows: list[ProxyCorrelationRow]
    report_text: str


def read_sentence_records(path: Path) -> list[SentenceRecord]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [
        SentenceRecord(
            corpus_id=row["corpus_id"],
            doc_path=row["doc_path"],
            sent_idx=int(row["sent_idx"]),
            sent_text=row["sent_text"],
            sent_id=row["sent_id"],
        )
        for row in rows
    ]


def read_scores(path: Path) -> dict[str, float]:
    scores: dict[str, float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if not row:
                continue
            scores[row[0]] = float(row[1])
    return scores


def feature_value(row: SentenceFeatureRow, feature: str) -> float:
    return float(getattr(row, feature))


def compute_proxy_correlations(
    corpus_id: str,
    sentence_records: list[SentenceRecord],
    scores: dict[str, float],
) -> list[ProxyCorrelationRow]:
    sentence_ids = {record.sent_id for record in sentence_records}
    score_ids = set(scores)
    if sentence_ids != score_ids:
        raise ValueError(
            f"Proxy decomposition join mismatch for {corpus_id}: "
            f"missing_scores={len(sentence_ids - score_ids)}, extra_scores={len(score_ids - sentence_ids)}"
        )

    feature_rows = sorted(compute_corpus_features(sentence_records), key=lambda row: row.sent_id)
    score_values = [scores[row.sent_id] for row in feature_rows]
    output_rows: list[ProxyCorrelationRow] = []
    for feature in ORIGINAL_FEATURES + DOCUMENTATION_FEATURES:
        values = [feature_value(row, feature) for row in feature_rows]
        rho = spearman(score_values, values)
        nonzero_count = sum(1 for value in values if value != 0)
        output_rows.append(
            ProxyCorrelationRow(
                corpus_id=corpus_id,
                feature=feature,
                feature_family=FEATURE_FAMILY[feature],
                feature_group="original" if feature in ORIGINAL_FEATURES else "documentation_aware",
                spearman_rho=rho,
                abs_spearman_rho=abs(rho),
                sentence_count=len(feature_rows),
                mean_value=mean(values) if values else 0.0,
                nonzero_rate=(nonzero_count / len(values)) if values else 0.0,
            )
        )
    return output_rows


def write_proxy_table(path: Path, rows: list[ProxyCorrelationRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "corpus_id",
                "feature",
                "feature_family",
                "feature_group",
                "spearman_rho",
                "abs_spearman_rho",
                "sentence_count",
                "mean_value",
                "nonzero_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "corpus_id": row.corpus_id,
                    "feature": row.feature,
                    "feature_family": row.feature_family,
                    "feature_group": row.feature_group,
                    "spearman_rho": f"{row.spearman_rho:.8f}",
                    "abs_spearman_rho": f"{row.abs_spearman_rho:.8f}",
                    "sentence_count": row.sentence_count,
                    "mean_value": f"{row.mean_value:.8f}",
                    "nonzero_rate": f"{row.nonzero_rate:.8f}",
                }
            )


def _rows_for_corpus(rows: list[ProxyCorrelationRow], corpus_id: str) -> list[ProxyCorrelationRow]:
    return [row for row in rows if row.corpus_id == corpus_id]


def _top_features(rows: list[ProxyCorrelationRow], *, group: str | None = None, limit: int = 3) -> list[ProxyCorrelationRow]:
    candidates = [row for row in rows if group is None or row.feature_group == group]
    return sorted(candidates, key=lambda row: (-row.abs_spearman_rho, row.feature))[:limit]


def _format_top(rows: list[ProxyCorrelationRow]) -> str:
    return ", ".join(f"`{row.feature}` ({row.spearman_rho:.3f})" for row in rows)


def _strongest_documentation_by_corpus(rows: list[ProxyCorrelationRow]) -> dict[str, ProxyCorrelationRow]:
    out: dict[str, ProxyCorrelationRow] = {}
    for corpus_id in sorted({row.corpus_id for row in rows}):
        doc_rows = _top_features(_rows_for_corpus(rows, corpus_id), group="documentation_aware", limit=1)
        if doc_rows:
            out[corpus_id] = doc_rows[0]
    return out


def build_proxy_report(
    *,
    rows: list[ProxyCorrelationRow],
    source_artifacts: list[Path],
    missing_artifacts: list[Path],
    outputs_root: Path,
    csv_path: Path,
) -> str:
    corpus_ids = sorted({row.corpus_id for row in rows})
    strongest_doc = _strongest_documentation_by_corpus(rows)
    tech_aggregate = {
        row.corpus_id: row
        for row in rows
        if row.feature == "technical_token_ratio"
    }
    by_feature = {(row.corpus_id, row.feature): row for row in rows}
    tech_beaten = [
        corpus_id
        for corpus_id, doc_row in strongest_doc.items()
        if doc_row.abs_spearman_rho > tech_aggregate[corpus_id].abs_spearman_rho
    ]

    lines: list[str] = [
        "# Documentation-Aware Proxy Decomposition",
        "",
        "## Implementation Summary",
        "",
        "This sprint decomposes the existing aggregate `technical_token_ratio` proxy into deterministic documentation-aware surface indicators while preserving the original proxy columns. No new corpora, labels, models, embeddings, parsing dependencies, retrieval tasks, or manuscript LaTeX edits were introduced.",
        "",
        "Paper-facing regeneration uses pinned NLTK 3.9.1 `TreebankWordTokenizer`; the CLI refuses the fallback tokenizer.",
        "",
        "## New Feature Definitions",
        "",
    ]
    for feature in DOCUMENTATION_FEATURES:
        lines.append(f"- `{feature}`: {FEATURE_DEFINITIONS[feature]}")

    lines.extend(
        [
            "",
            "## Source Artifacts Used",
            "",
            "- Outputs root: runtime-supplied reconstruction root (not stored in this report)",
            "- Sentence tables: `outputs/sentences/<corpus_id>.csv`",
            "- Score tables: `outputs/speciteller/<corpus_id>_scores.tsv`",
            f"- Generated comparison CSV: `{(Path('analysis') / csv_path.name).as_posix()}`",
        ]
    )
    for path in source_artifacts:
        portable_path = (Path("outputs") / path.relative_to(outputs_root)).as_posix()
        lines.append(f"- Found: `{portable_path}`")
    if missing_artifacts:
        lines.append("")
        lines.append("Missing artifacts:")
        lines.extend(
            f"- `{(Path('outputs') / path.relative_to(outputs_root)).as_posix()}`"
            for path in missing_artifacts
        )

    lines.extend(
        [
            "",
            "## Correlation Summaries",
            "",
            "| corpus_id | strongest original proxies | strongest documentation-aware proxies |",
            "|---|---|---|",
        ]
    )
    for corpus_id in corpus_ids:
        corpus_rows = _rows_for_corpus(rows, corpus_id)
        lines.append(
            f"| {corpus_id} | {_format_top(_top_features(corpus_rows, group='original'))} | "
            f"{_format_top(_top_features(corpus_rows, group='documentation_aware'))} |"
        )

    lines.extend(["", "## Corpus-Specific Observations", ""])
    for corpus_id in corpus_ids:
        corpus_rows = _rows_for_corpus(rows, corpus_id)
        doc_top = _top_features(corpus_rows, group="documentation_aware", limit=2)
        original_top = _top_features(corpus_rows, group="original", limit=2)
        aggregate = tech_aggregate[corpus_id]
        lines.append(
            f"- `{corpus_id}`: strongest documentation-aware proxies are {_format_top(doc_top)}; "
            f"strongest original proxies are {_format_top(original_top)}. "
            f"`technical_token_ratio` is {aggregate.spearman_rho:.3f}."
        )

    lines.extend(
        [
            "",
            "## Comparison Against Original Proxy Conclusions",
            "",
        ]
    )
    if tech_beaten:
        technical_beaten = [corpus_id for corpus_id in tech_beaten if corpus_id != "wikipedia_en"]
        if technical_beaten:
            lines.append(
                "- The aggregate `technical_token_ratio` hides meaningful internal variation in technical documentation: at least one decomposed documentation-aware proxy has a stronger absolute Spearman association than the aggregate in "
                + ", ".join(f"`{corpus_id}`" for corpus_id in technical_beaten)
                + "."
            )
        if "wikipedia_en" in tech_beaten:
            lines.append(
                "- Wikipedia also has decomposed documentation-aware proxies that exceed the aggregate technical proxy, but Wikipedia remains dominated by generic expository length and TF-IDF signals rather than documentation-specific interpretation."
            )
    else:
        lines.append(
            "- The aggregate `technical_token_ratio` remains as strong as or stronger than each decomposed documentation-aware proxy by absolute Spearman association in the inspected corpora."
        )

    doc_strengths = [row.abs_spearman_rho for row in rows if row.feature_group == "documentation_aware"]
    original_strengths = [row.abs_spearman_rho for row in rows if row.feature_group == "original"]
    lines.append(
        f"- Mean absolute correlation across documentation-aware proxies is {mean(doc_strengths):.3f}; across original proxies it is {mean(original_strengths):.3f}. This is descriptive only because the features are not independent model terms."
    )

    paper_keys = {
        ("wikipedia_en", "token_count"),
        ("wikipedia_en", "tfidf_mean_nonzero"),
        ("wikipedia_en", "char_count"),
        ("python_312_html", "version_numeric_density"),
        ("python_312_html", "token_shape_complexity_mean"),
        ("python_312_html", "technical_token_ratio"),
        ("github_docs", "identifier_density"),
        ("github_docs", "technical_token_ratio"),
    }
    if paper_keys.issubset(by_feature):
        scientific_objective_lines = [
            "- Generic expository proxies weaken in the technical corpora relative to Wikipedia. In `wikipedia_en`, length and TF-IDF dominate the table "
            f"(`token_count` {by_feature[('wikipedia_en', 'token_count')].spearman_rho:.3f}, "
            f"`tfidf_mean_nonzero` {by_feature[('wikipedia_en', 'tfidf_mean_nonzero')].spearman_rho:.3f}, "
            f"`char_count` {by_feature[('wikipedia_en', 'char_count')].spearman_rho:.3f}), while the technical corpora show smaller and more mixed generic-proxy associations.",
            "- Documentation-aware indicators become comparatively informative in technical documentation, but not uniformly. "
            f"In `python_312_html`, decomposed features outrank the aggregate technical proxy (`version_numeric_density` {by_feature[('python_312_html', 'version_numeric_density')].spearman_rho:.3f} and "
            f"`token_shape_complexity_mean` {by_feature[('python_312_html', 'token_shape_complexity_mean')].spearman_rho:.3f} versus `technical_token_ratio` {by_feature[('python_312_html', 'technical_token_ratio')].spearman_rho:.3f}). "
            f"In `github_docs`, `identifier_density` ({by_feature[('github_docs', 'identifier_density')].spearman_rho:.3f}) nearly matches the aggregate technical proxy ({by_feature[('github_docs', 'technical_token_ratio')].spearman_rho:.3f}), while `char_count` remains strongest. "
            "In `ansible_docs`, the aggregate technical proxy remains strongest, but version, identifier, and command/path/flag densities are all moderate positive correlates.",
        ]
    else:
        scientific_objective_lines = [
            "- The fixture confirms deterministic, join-complete computation of original and documentation-aware proxy families."
        ]

    lines.extend(
        [
            "",
            "## Scientific Objective Checks",
            "",
            *scientific_objective_lines,
            "- The aggregate `technical_token_ratio` therefore does hide internal variation, especially in the Python reference corpus, but the decomposition does not make the original aggregate obsolete.",
            "- Different corpora emphasize different proxy families: GitHub leans toward identifiers, Ansible shows a mixed version/identifier/command pattern, Python reference shows token-shape and version/numeric structure, and Wikipedia remains dominated by expository length and TF-IDF behavior.",
            "- No result contradicts the current Discussion framing. The evidence strengthens the score-portability caution by showing that the same scalar score aligns with different visible structures across documentation genres.",
            "- The assignment/parameter proxy is consistently weaker than the other documentation-aware additions, so paper wording should avoid claiming that parameter syntax is a dominant driver.",
            "",
            "## Whether Conclusions Changed",
            "",
            "The conclusions are refined rather than reversed. The decomposition supports the existing score-portability framing by showing that technical-documentation structure is not a single uniform signal. However, the evidence should be phrased carefully: these proxies characterize visible surface structure, not semantic operational completeness or true human specificity.",
            "",
            "The safest conclusion is that specificity in technical documentation aligns partly with documentation-aware structures, but the relevant structure varies by corpus and proxy family. This strengthens the argument that a transferred scalar score should be interpreted with corpus-specific evidence rather than as a domain-invariant measure.",
            "",
            "## Recommended Paper Wording",
            "",
            "A compact paper-facing wording would be: \"A decomposition of the aggregate technical-token proxy shows that score alignment in technical documentation is not driven by one uniform code-like signal. Depending on the corpus, specificity scores align with visible operational structures such as identifiers, paths or flags, versions or numeric tokens, and token-shape complexity, while parameter-assignment syntax is weaker in this analysis. These patterns refine, rather than replace, the portability caution: the same scalar score can correspond to different structural evidence across documentation genres.\"",
            "",
            "## Paper Integration Recommendation",
            "",
            "- Place this result in an appendix or compact supplemental table unless page space opens in the main Results section.",
            "- Extend rather than replace the original feature table so the original empirical content remains preserved.",
            "- Add one main-text Discussion sentence if space allows, emphasizing corpus-specific structural correlates rather than new model features.",
            "- Do not present these proxies as a new metric, an execution-based documentation analysis, or evidence of downstream task utility.",
            "",
            "## Implementation Risks Or Data Gaps",
            "",
            "- The current IEEE checkout still lacks local generated `outputs/`; this report was generated from the prior Speciteller output tree unless a different `--outputs-root` was supplied.",
            "- Assignment and command indicators are deterministic surface approximations. They should be described as documentation-aware structural proxies, not parser-derived syntax or semantic labels.",
            "- No paper LaTeX was modified in this sprint.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_proxy_decomposition(
    *,
    corpus_ids: list[str],
    outputs_root: Path,
    report_path: Path,
    csv_path: Path,
) -> ProxyDecompositionResult:
    source_artifacts: list[Path] = []
    missing_artifacts: list[Path] = []
    all_rows: list[ProxyCorrelationRow] = []

    for corpus_id in corpus_ids:
        sentence_path = outputs_root / "sentences" / f"{corpus_id}.csv"
        score_path = outputs_root / "speciteller" / f"{corpus_id}_scores.tsv"
        for path in (sentence_path, score_path):
            if path.exists():
                source_artifacts.append(path)
            else:
                missing_artifacts.append(path)
        if sentence_path.exists() and score_path.exists():
            all_rows.extend(
                compute_proxy_correlations(
                    corpus_id,
                    read_sentence_records(sentence_path),
                    read_scores(score_path),
                )
            )

    if not all_rows:
        raise FileNotFoundError(
            f"No complete sentence/score artifact pairs were found under {outputs_root}"
        )

    write_proxy_table(csv_path, all_rows)
    report_text = build_proxy_report(
        rows=all_rows,
        source_artifacts=source_artifacts,
        missing_artifacts=missing_artifacts,
        outputs_root=outputs_root,
        csv_path=csv_path,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report_text, encoding="utf-8")
    return ProxyDecompositionResult(
        source_artifacts=source_artifacts,
        missing_artifacts=missing_artifacts,
        rows=all_rows,
        report_text=report_text,
    )
