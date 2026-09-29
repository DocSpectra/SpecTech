"""Utilities for building plaintext wikipedia_en corpora from XML dump shards."""
from __future__ import annotations

import bz2
import re
from pathlib import Path
import xml.etree.ElementTree as ET


def _safe_title(title: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", title).strip("._")
    return slug[:160] or "untitled"


def _strip_tag_ns(tag: str) -> str:
    return tag.split("}", 1)[-1]


def extract_plaintext_from_wiki_dump(
    dump_bz2_paths: list[Path],
    output_dir: Path,
    target_count: int = 2000,
    reset: bool = False,
) -> int:
    """Extract deterministic plaintext pages from one or more dump shards.

    Selection rule is deterministic and stream-safe:
    - process dump files in lexicographic order
    - keep first ``target_count`` valid namespace-0, non-redirect pages
    - write one UTF-8 ``.txt`` file per article
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if reset:
        for existing in output_dir.glob("*.txt"):
            existing.unlink()

    written = 0
    for dump_path in sorted(dump_bz2_paths):
        if written >= target_count:
            break
        with bz2.open(dump_path, "rb") as handle:
            context = ET.iterparse(handle, events=("end",))
            for _event, elem in context:
                if _strip_tag_ns(elem.tag) != "page":
                    continue

                ns_text = elem.findtext("./{*}ns", default="")
                title = elem.findtext("./{*}title", default="").strip()
                redirect = elem.find("./{*}redirect")
                text = elem.findtext("./{*}revision/{*}text", default="")

                if ns_text != "0" or redirect is not None:
                    elem.clear()
                    continue

                text = (text or "").strip()
                if not title or not text:
                    elem.clear()
                    continue

                filename = f"{written + 1:04d}_{_safe_title(title)}.txt"
                (output_dir / filename).write_text(text + "\n", encoding="utf-8")
                written += 1

                elem.clear()
                if written >= target_count:
                    break

    return written
