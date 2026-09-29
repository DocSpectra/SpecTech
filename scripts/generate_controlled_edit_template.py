"""Generate deterministic controlled-edit templates for Phase 7.5.

A ``controlled_edit_template`` is a CSV used to run the manual-edit behavioral
evaluation described in ``spec/SPEC.md`` and ``spec/TASK.md``.

How it is built:
- We load already-scored sentence rows from prior pipeline outputs.
- We deterministically sample sentences from score strata
  (bottom decile, median band, top decile, and optional length-controlled band).
- We assign each sampled row an ``edit_type`` in
  ``{de_specify, add_specific, irrelevant_rewrite}``.

What the template contains:
- ``sent_id`` and ``corpus_id``
- ``sentence_original`` and ``speciteller_score_original``
- ``token_count``
- ``edit_type``
- ``sentence_edited`` (left blank for humans to fill)

How it is used:
1. A human editor fills ``sentence_edited`` according to ``edit_type``:
   - ``de_specify``: rewrite to be less specific
   - ``add_specific``: rewrite to be more specific
   - ``irrelevant_rewrite``: rewrite while preserving rough specificity intent
2. A follow-up scoring step re-scores the edited sentences with SpeciTeller.
3. We compute per-edit-type deltas/directional statistics to describe model
   behavioral sensitivity (descriptive analysis only; not accuracy evaluation).

Usage example:
    python scripts/generate_controlled_edit_template.py --corpus-id github_docs
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.analysis.phase07_analysis_outputs import load_analysis_rows
from src.analysis.phase075_controlled_edit_evaluation import build_controlled_edit_template_rows
from src.analysis.phase075_controlled_edit_evaluation import write_controlled_edit_template
from src.analysis.phase07_analysis_outputs import AnalysisRow
from src.utils.text_filters import is_annotatable_sentence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate controlled-edit template CSV")
    parser.add_argument("--corpus-id", required=True, help="Corpus id, e.g. github_docs")
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root outputs directory containing phase-7 artifacts",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional explicit output path",
    )
    parser.add_argument(
        "--bottom-count",
        type=int,
        default=10,
        help="Samples from bottom decile score band",
    )
    parser.add_argument(
        "--median-count",
        type=int,
        default=10,
        help="Samples from median score band",
    )
    parser.add_argument(
        "--top-count",
        type=int,
        default=10,
        help="Samples from top decile score band",
    )
    parser.add_argument(
        "--length-count",
        type=int,
        default=0,
        help="Optional samples from length-controlled band",
    )
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=8,
        help="Minimum whitespace token count for annotatable sentence filtering",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Maximum whitespace token count for annotatable sentence filtering",
    )
    parser.add_argument(
        "--require-terminal-punct",
        action="store_true",
        help="Require sampled sentences to end with terminal punctuation (. ? !)",
    )
    parser.add_argument(
        "--exclude-bullets",
        action="store_true",
        help="Exclude bullet-style lines that start with '- '",
    )
    parser.add_argument(
        "--exclude-html-code",
        action="store_true",
        help="Exclude lines containing <code>...</code> markup",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help=(
            "Enable stricter filtering suitable for human annotation tasks "
            "(complete sentences, no bullets, no markup-heavy code lines)."
        ),
    )
    parser.add_argument("--seed", type=int, default=13, help="Deterministic sampling seed")
    return parser.parse_args()


def dedupe_rows_by_sent_text(rows: list[AnalysisRow]) -> list[AnalysisRow]:
    """Keep first occurrence per sent_text while preserving input order."""
    seen_texts: set[str] = set()
    out: list[AnalysisRow] = []
    for row in rows:
        if row.sent_text in seen_texts:
            continue
        seen_texts.add(row.sent_text)
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    if args.human:
        args.min_tokens = 8
        args.max_tokens = 40
        args.require_terminal_punct = True
        args.exclude_bullets = True
        args.exclude_html_code = True

    rows = [
        row
        for row in load_analysis_rows(args.corpus_id, args.outputs_root)
        if is_annotatable_sentence(
            row.sent_text,
            min_tokens=args.min_tokens,
            max_tokens=args.max_tokens,
            require_terminal_punct=args.require_terminal_punct,
            exclude_bullets=args.exclude_bullets,
            exclude_html_code=args.exclude_html_code,
        )
    ]
    rows = dedupe_rows_by_sent_text(rows)

    template_rows = build_controlled_edit_template_rows(
        rows,
        bottom_count=args.bottom_count,
        median_count=args.median_count,
        top_count=args.top_count,
        length_count=args.length_count,
        seed=args.seed,
    )

    expected_count = (
        max(0, args.bottom_count)
        + max(0, args.median_count)
        + max(0, args.top_count)
        + max(0, args.length_count)
    )
    if len(template_rows) < expected_count:
        raise ValueError(
            "Insufficient eligible unique sentences after filtering/deduplication: "
            f"requested={expected_count}, generated={len(template_rows)}. "
            "Adjust counts or relax filters."
        )

    output_path = args.output_path
    if output_path is None:
        output_path = (
            args.outputs_root
            / "controlled_edits"
            / f"{args.corpus_id}_controlled_edit_template.csv"
        )
    write_controlled_edit_template(output_path, template_rows)
    print(f"Wrote {len(template_rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
