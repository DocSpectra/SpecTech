"""Tests for deterministic spot-check sampling behavior."""
from __future__ import annotations

from collections import Counter
import importlib.util
from pathlib import Path
import sys


def _load_spot_check_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "generate_spot_check.py"
    module_name = "generate_spot_check_for_tests"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_build_spot_check_enforces_unique_sent_text_with_backfill() -> None:
    module = _load_spot_check_module()
    Row = module.Row
    build_spot_check = module.build_spot_check

    rows = [
        Row("demo", "doc", 0, "Alpha duplicate sentence.", "s1", 0.10),
        Row("demo", "doc", 1, "Alpha duplicate sentence.", "s2", 0.20),
        Row("demo", "doc", 2, "Bravo unique sentence.", "s3", 0.30),
        Row("demo", "doc", 3, "Charlie unique sentence.", "s4", 0.40),
        Row("demo", "doc", 4, "Delta unique sentence.", "s5", 0.50),
        Row("demo", "doc", 5, "Echo unique sentence.", "s6", 0.60),
        Row("demo", "doc", 6, "Zulu duplicate sentence.", "s7", 0.90),
        Row("demo", "doc", 7, "Zulu duplicate sentence.", "s8", 1.00),
    ]

    sampled = build_spot_check(rows, edge_count=4, avg_count=2)

    assert len(sampled) == 6
    sent_texts = [row.sent_text for _, row in sampled]
    assert len(sent_texts) == len(set(sent_texts))

    bucket_counts = Counter(bucket for bucket, _ in sampled)
    assert bucket_counts["edge_low"] == 2
    assert bucket_counts["edge_high"] == 2
    assert bucket_counts["average"] == 2

    sampled_again = build_spot_check(rows, edge_count=4, avg_count=2)
    assert sampled == sampled_again
