"""Fetch deterministic Wikipedia plaintext samples into data/corpora/wikipedia_en.

This script uses the MediaWiki API in deterministic allpages order and writes
one UTF-8 .txt file per article.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import requests


WIKI_API = "https://en.wikipedia.org/w/api.php"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch deterministic Wikipedia plaintext")
    parser.add_argument("--target-count", type=int, default=2000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/corpora/wikipedia_en"),
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing .txt files in output dir before fetching",
    )
    parser.add_argument("--sleep-ms", type=int, default=50)
    parser.add_argument("--max-retries", type=int, default=8)
    parser.add_argument("--backoff-base-seconds", type=float, default=2.0)
    return parser.parse_args()


def _safe_name(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")
    return slug[:160] or "untitled"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.reset:
        for path in args.output_dir.glob("*.txt"):
            path.unlink()

    session = requests.Session()
    session.headers.update({"User-Agent": "SpecTechWikipediaBootstrap/1.0"})

    written = 0
    continuation: dict[str, str] = {}
    seen_titles: set[str] = set()

    while written < args.target_count:
        params = {
            "action": "query",
            "format": "json",
            "generator": "allpages",
            "gapnamespace": "0",
            "gapfilterredir": "nonredirects",
            # Keep this lower than max to reduce burstiness and avoid 429s.
            "gaplimit": "50",
            "prop": "extracts",
            "explaintext": "1",
            "exchars": "6000",
        }
        params.update(continuation)

        response = None
        for attempt in range(args.max_retries + 1):
            response = session.get(WIKI_API, params=params, timeout=60)
            if response.status_code != 429:
                break
            wait_s = args.backoff_base_seconds * (2**attempt)
            print(f"Received 429, backing off for {wait_s:.1f}s (attempt {attempt + 1})")
            time.sleep(wait_s)

        if response is None:
            raise RuntimeError("No response received from Wikipedia API")
        response.raise_for_status()
        payload = response.json()
        pages = payload.get("query", {}).get("pages", {})
        ordered_pages = sorted(
            pages.values(),
            key=lambda page: (str(page.get("title", "")), int(page.get("pageid", 0))),
        )

        for page in ordered_pages:
            if written >= args.target_count:
                break
            title = str(page.get("title", "")).strip()
            text = str(page.get("extract", "")).strip()
            if not title or not text or title in seen_titles:
                continue

            seen_titles.add(title)
            filename = f"{written + 1:04d}_{_safe_name(title)}.txt"
            (args.output_dir / filename).write_text(text + "\n", encoding="utf-8")
            written += 1

        if "continue" not in payload:
            break
        continuation = {
            key: value
            for key, value in payload["continue"].items()
            if key != "continue"
        }
        if args.sleep_ms > 0:
            time.sleep(args.sleep_ms / 1000.0)

    print(f"wrote {written} wikipedia plaintext files to {args.output_dir}")


if __name__ == "__main__":
    main()
