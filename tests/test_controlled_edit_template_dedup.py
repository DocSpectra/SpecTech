"""Tests for controlled-edit template sent_text deduplication helper."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

from src.analysis.phase07_analysis_outputs import AnalysisRow


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "generate_controlled_edit_template.py"
    )
    module_name = "generate_controlled_edit_template_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _analysis_row(idx: int, text: str) -> AnalysisRow:
    return AnalysisRow(
        corpus_id="demo",
        doc_path="docs/example.md",
        sent_id=f"s{idx:03d}",
        sent_text=text,
        score=idx / 100.0,
        tfidf_mean_nonzero=0.1,
        tfidf_max=0.2,
        technical_token_ratio=0.05,
        token_count=10,
        char_count=len(text),
    )


def test_dedupe_rows_by_sent_text_keeps_first_and_order() -> None:
    module = _load_module()
    dedupe = module.dedupe_rows_by_sent_text

    rows = [
        _analysis_row(1, "Duplicate line."),
        _analysis_row(2, "Unique line A."),
        _analysis_row(3, "Duplicate line."),
        _analysis_row(4, "Unique line B."),
        _analysis_row(5, "Unique line A."),
    ]

    deduped = dedupe(rows)
    assert [r.sent_id for r in deduped] == ["s001", "s002", "s004"]
    assert [r.sent_text for r in deduped] == [
        "Duplicate line.",
        "Unique line A.",
        "Unique line B.",
    ]
