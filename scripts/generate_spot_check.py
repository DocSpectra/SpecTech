"""Generate deterministic human spot-check samples from scored outputs.

Usage example:
    python scripts/generate_spot_check.py --corpus-id github_docs
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.utils.text_filters import is_annotatable_sentence


@dataclass(frozen=True)
class Row:
    corpus_id: str
    doc_path: str
    sent_idx: int
    sent_text: str
    sent_id: str
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate edge/average spot-check CSV")
    p.add_argument("--corpus-id", required=True, help="Corpus id, e.g. github_docs")
    p.add_argument("--edge-count", type=int, default=20, help="Total edge examples")
    p.add_argument(
        "--average-count", type=int, default=20, help="Total average examples"
    )
    p.add_argument(
        "--min-tokens",
        type=int,
        default=8,
        help="Minimum whitespace token count for annotatable sentence filtering",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=40,
        help="Maximum whitespace token count for annotatable sentence filtering",
    )
    p.add_argument(
        "--require-terminal-punct",
        action="store_true",
        help="Require sampled sentences to end with terminal punctuation (. ? !)",
    )
    p.add_argument(
        "--exclude-bullets",
        action="store_true",
        help="Exclude bullet-style lines that start with '- '",
    )
    p.add_argument(
        "--exclude-html-code",
        action="store_true",
        help="Exclude lines containing <code>...</code> markup",
    )
    p.add_argument(
        "--human",
        action="store_true",
        help=(
            "Enable stricter filtering suitable for human annotation tasks "
            "(complete sentences, no bullets, no markup-heavy code lines)."
        ),
    )
    p.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("outputs"),
        help="Root outputs directory",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional output CSV path",
    )
    return p.parse_args()


def load_scores(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sent_id, score = line.split("\t", 1)
            out[sent_id] = float(score)
    return out


def load_joined(
    sentence_csv: Path,
    scores_tsv: Path,
    *,
    min_tokens: int,
    max_tokens: int,
    require_terminal_punct: bool,
    exclude_bullets: bool,
    exclude_html_code: bool,
) -> list[Row]:
    scores = load_scores(scores_tsv)
    rows: list[Row] = []
    with sentence_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            sent_id = r["sent_id"]
            if sent_id not in scores:
                continue
            sent_text = r["sent_text"]
            if not is_annotatable_sentence(
                sent_text,
                min_tokens=min_tokens,
                max_tokens=max_tokens,
                require_terminal_punct=require_terminal_punct,
                exclude_bullets=exclude_bullets,
                exclude_html_code=exclude_html_code,
            ):
                continue
            rows.append(
                Row(
                    corpus_id=r["corpus_id"],
                    doc_path=r["doc_path"],
                    sent_idx=int(r["sent_idx"]),
                    sent_text=sent_text,
                    sent_id=sent_id,
                    score=scores[sent_id],
                )
            )
    return rows


def build_spot_check(rows: list[Row], edge_count: int, avg_count: int) -> list[tuple[str, Row]]:
    if not rows:
        return []

    sorted_rows = sorted(rows, key=lambda x: x.score)

    low_n = edge_count // 2
    high_n = edge_count - low_n
    low = sorted_rows[:low_n]
    high = sorted_rows[-high_n:] if high_n > 0 else []

    selected_ids = {r.sent_id for r in low + high}

    mid_score = sorted_rows[len(sorted_rows) // 2].score
    remaining = [r for r in sorted_rows if r.sent_id not in selected_ids]
    avg_sorted = sorted(remaining, key=lambda r: abs(r.score - mid_score))
    avg = avg_sorted[:avg_count]

    # First-pass deterministic sample (existing sampling logic/order).
    initial: list[tuple[str, Row]] = []
    initial.extend(("edge_low", r) for r in low)
    initial.extend(("edge_high", r) for r in high)
    initial.extend(("average", r) for r in avg)

    # Enforce uniqueness by sent_text while keeping first occurrence order.
    targets = {
        "edge_low": low_n,
        "edge_high": high_n,
        "average": avg_count,
    }
    by_bucket: dict[str, list[Row]] = {
        "edge_low": [],
        "edge_high": [],
        "average": [],
    }
    seen_texts: set[str] = set()
    used_ids: set[str] = set()

    for bucket, row in initial:
        if row.sent_text in seen_texts:
            continue
        by_bucket[bucket].append(row)
        seen_texts.add(row.sent_text)
        used_ids.add(row.sent_id)

    def backfill(bucket: str, candidates: list[Row]) -> None:
        target = targets[bucket]
        if len(by_bucket[bucket]) >= target:
            return
        for row in candidates:
            if len(by_bucket[bucket]) >= target:
                break
            if row.sent_id in used_ids:
                continue
            if row.sent_text in seen_texts:
                continue
            by_bucket[bucket].append(row)
            seen_texts.add(row.sent_text)
            used_ids.add(row.sent_id)

    # Backfill from deterministic, bucket-specific candidate order.
    backfill("edge_low", sorted_rows)
    backfill("edge_high", list(reversed(sorted_rows)))
    backfill("average", avg_sorted)

    result: list[tuple[str, Row]] = []
    result.extend(("edge_low", r) for r in by_bucket["edge_low"])
    result.extend(("edge_high", r) for r in by_bucket["edge_high"])
    result.extend(("average", r) for r in by_bucket["average"])
    return result


def write_output(path: Path, sampled: list[tuple[str, Row]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "bucket",
                "corpus_id",
                "doc_path",
                "sent_idx",
                "sent_id",
                "score",
                "sent_text",
                "human_label",
                "human_notes",
            ]
        )
        for bucket, row in sampled:
            writer.writerow(
                [
                    bucket,
                    row.corpus_id,
                    row.doc_path,
                    row.sent_idx,
                    row.sent_id,
                    f"{row.score:.6f}",
                    row.sent_text,
                    "",
                    "",
                ]
            )


def main() -> None:
    args = parse_args()
    if args.human:
        args.min_tokens = 8
        args.max_tokens = 40
        args.require_terminal_punct = True
        args.exclude_bullets = True
        args.exclude_html_code = True

    sentence_csv = args.outputs_root / "sentences" / f"{args.corpus_id}.csv"
    scores_tsv = args.outputs_root / "speciteller" / f"{args.corpus_id}_scores.tsv"

    if args.output_path is None:
        output_path = (
            args.outputs_root
            / "spot_checks"
            / f"{args.corpus_id}_spot_check_edge{args.edge_count}_avg{args.average_count}.csv"
        )
    else:
        output_path = args.output_path

    rows = load_joined(
        sentence_csv,
        scores_tsv,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        require_terminal_punct=args.require_terminal_punct,
        exclude_bullets=args.exclude_bullets,
        exclude_html_code=args.exclude_html_code,
    )
    sampled = build_spot_check(rows, args.edge_count, args.average_count)
    write_output(output_path, sampled)
    print(f"Wrote {len(sampled)} rows to {output_path}")


if __name__ == "__main__":
    main()
