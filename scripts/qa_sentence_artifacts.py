"""Standalone QA report for sentence-extraction artifacts.

Reads canonical sentence tables and emits prevalence metrics + deterministic examples
without changing any pipeline outputs.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.text_filters import alphabetic_ratio_non_space
from src.utils.text_filters import contains_code_fence_artifact
from src.utils.text_filters import contains_template_markup
from src.utils.text_filters import looks_like_markdown_table_row
from src.utils.text_filters import token_count_whitespace


@dataclass(frozen=True)
class SentenceRow:
    sent_id: str
    doc_path: str
    sent_idx: int
    sent_text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QA sentence artifacts from outputs/sentences")
    parser.add_argument("--corpus-id", required=True, help="Corpus id, e.g. github_docs")
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional explicit input sentence CSV path",
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Outputs root containing sentences/ and qa/",
    )
    return parser.parse_args()


def load_sentences(path: Path) -> list[SentenceRow]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        out: list[SentenceRow] = []
        for row in reader:
            out.append(
                SentenceRow(
                    sent_id=row["sent_id"],
                    doc_path=row["doc_path"],
                    sent_idx=int(row["sent_idx"]),
                    sent_text=row["sent_text"],
                )
            )
    return out


def _pct(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return (100.0 * count) / total


def write_examples(path: Path, rows: list[SentenceRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sent_id", "doc_path", "sent_idx", "sent_text"])
        for row in rows[:10]:
            writer.writerow([row.sent_id, row.doc_path, row.sent_idx, row.sent_text])


def main() -> None:
    args = parse_args()

    sentence_path = args.input_path
    if sentence_path is None:
        sentence_path = args.outputs_root / "sentences" / f"{args.corpus_id}.csv"

    rows = load_sentences(sentence_path)
    total = len(rows)

    short_rows: list[SentenceRow] = []
    template_rows: list[SentenceRow] = []
    table_rows: list[SentenceRow] = []
    code_fence_rows: list[SentenceRow] = []
    nonalpha_rows: list[SentenceRow] = []
    any_rows: list[SentenceRow] = []

    for row in rows:
        token_count = token_count_whitespace(row.sent_text)
        is_short = token_count < 5
        is_template = contains_template_markup(row.sent_text)
        is_table = looks_like_markdown_table_row(row.sent_text)
        is_code_fence = contains_code_fence_artifact(row.sent_text)
        is_nonalpha_heavy = alphabetic_ratio_non_space(row.sent_text) < 0.6

        if is_short:
            short_rows.append(row)
        if is_template:
            template_rows.append(row)
        if is_table:
            table_rows.append(row)
        if is_code_fence:
            code_fence_rows.append(row)
        if is_nonalpha_heavy:
            nonalpha_rows.append(row)

        if is_short or is_template or is_table or is_code_fence or is_nonalpha_heavy:
            any_rows.append(row)

    qa_dir = args.outputs_root / "qa"
    examples_dir = qa_dir / "examples"
    qa_dir.mkdir(parents=True, exist_ok=True)
    examples_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "corpus_id": args.corpus_id,
        "input_path": str(sentence_path).replace("\\", "/"),
        "total_sentences": total,
        "pct_short": _pct(len(short_rows), total),
        "pct_template": _pct(len(template_rows), total),
        "pct_table": _pct(len(table_rows), total),
        "pct_code_fence": _pct(len(code_fence_rows), total),
        "pct_nonalpha_heavy": _pct(len(nonalpha_rows), total),
        "pct_flagged_any": _pct(len(any_rows), total),
        "count_short": len(short_rows),
        "count_template": len(template_rows),
        "count_table": len(table_rows),
        "count_code_fence": len(code_fence_rows),
        "count_nonalpha_heavy": len(nonalpha_rows),
        "count_flagged_any": len(any_rows),
    }

    summary_path = qa_dir / f"{args.corpus_id}_qa_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    write_examples(examples_dir / f"{args.corpus_id}_short_examples.csv", short_rows)
    write_examples(examples_dir / f"{args.corpus_id}_template_examples.csv", template_rows)
    write_examples(examples_dir / f"{args.corpus_id}_table_examples.csv", table_rows)
    write_examples(examples_dir / f"{args.corpus_id}_code_fence_examples.csv", code_fence_rows)
    write_examples(examples_dir / f"{args.corpus_id}_nonalpha_heavy_examples.csv", nonalpha_rows)
    write_examples(examples_dir / f"{args.corpus_id}_flagged_any_examples.csv", any_rows)

    print(f"Wrote QA summary: {summary_path}")
    print(f"Wrote QA examples: {examples_dir}")


if __name__ == "__main__":
    main()
