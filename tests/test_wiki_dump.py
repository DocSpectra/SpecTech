"""Tests for deterministic Wikipedia dump extraction workflow."""
from __future__ import annotations

import bz2
from pathlib import Path

from src.ingestion.wiki_dump import extract_plaintext_from_wiki_dump


def _write_bz2(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with bz2.open(path, "wb") as handle:
        handle.write(content.encode("utf-8"))


def test_extract_plaintext_from_wiki_dump_filters_and_limits(tmp_path: Path) -> None:
    xml = """<?xml version='1.0'?>
<mediawiki>
  <page>
    <title>Alpha Topic</title>
    <ns>0</ns>
    <revision><text>Alpha body text.</text></revision>
  </page>
  <page>
    <title>Talk:Meta</title>
    <ns>1</ns>
    <revision><text>Should be skipped by ns.</text></revision>
  </page>
  <page>
    <title>Redirected</title>
    <ns>0</ns>
    <redirect title='Target'/>
    <revision><text>Should be skipped redirect.</text></revision>
  </page>
  <page>
    <title>Beta Topic</title>
    <ns>0</ns>
    <revision><text>Beta body text.</text></revision>
  </page>
</mediawiki>
"""
    dump = tmp_path / "dumps" / "part1.xml.bz2"
    out = tmp_path / "wiki_out"
    _write_bz2(dump, xml)

    written = extract_plaintext_from_wiki_dump([dump], out, target_count=1, reset=True)
    assert written == 1

    files = sorted(out.glob("*.txt"))
    assert len(files) == 1
    assert files[0].name.startswith("0001_Alpha_Topic")
    assert "Alpha body text." in files[0].read_text(encoding="utf-8")


def test_extract_plaintext_from_wiki_dump_is_deterministic(tmp_path: Path) -> None:
    xml = """<?xml version='1.0'?>
<mediawiki>
  <page><title>Zeta</title><ns>0</ns><revision><text>Z text.</text></revision></page>
  <page><title>Eta</title><ns>0</ns><revision><text>E text.</text></revision></page>
</mediawiki>
"""
    dump = tmp_path / "dumps" / "part1.xml.bz2"
    _write_bz2(dump, xml)

    out1 = tmp_path / "out1"
    out2 = tmp_path / "out2"
    extract_plaintext_from_wiki_dump([dump], out1, target_count=2, reset=True)
    extract_plaintext_from_wiki_dump([dump], out2, target_count=2, reset=True)

    names1 = [p.name for p in sorted(out1.glob("*.txt"))]
    names2 = [p.name for p in sorted(out2.glob("*.txt"))]
    assert names1 == names2
    texts1 = [p.read_text(encoding="utf-8") for p in sorted(out1.glob("*.txt"))]
    texts2 = [p.read_text(encoding="utf-8") for p in sorted(out2.glob("*.txt"))]
    assert texts1 == texts2
