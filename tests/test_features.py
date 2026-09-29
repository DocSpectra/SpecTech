"""Tests for deterministic proxy feature computation."""
from __future__ import annotations

import csv

from src.features.proxies import compute_corpus_features
from src.features.proxies import write_feature_table
from src.sentences.table import SentenceRecord


def _record(sent_id: str, text: str) -> SentenceRecord:
    return SentenceRecord(
        corpus_id="demo",
        doc_path="docs/example.md",
        sent_idx=0,
        sent_text=text,
        sent_id=sent_id,
    )


def test_compute_corpus_features_is_deterministic() -> None:
    records = [
        _record("s1", "Install with --force in module.func v3.12."),
        _record("s2", "This sentence is plain text."),
    ]

    first = compute_corpus_features(records)
    second = compute_corpus_features(records)

    assert first == second


def test_compute_corpus_features_has_expected_fields() -> None:
    records = [
        _record("s1", "Use --help in api_client for v3.12."),
        _record("s2", "Read the guide."),
    ]
    rows = compute_corpus_features(records)
    by_id = {row.sent_id: row for row in rows}

    assert by_id["s1"].token_count > 0
    assert by_id["s1"].char_count == len("Use --help in api_client for v3.12.")
    assert by_id["s1"].technical_token_ratio > 0
    assert by_id["s1"].identifier_density > 0
    assert by_id["s1"].command_path_flag_density > 0
    assert by_id["s1"].version_numeric_density > 0
    assert by_id["s1"].token_shape_complexity_mean > 0
    assert by_id["s2"].technical_token_ratio == 0
    assert by_id["s2"].identifier_density == 0
    assert by_id["s1"].tfidf_mean_nonzero >= 0
    assert by_id["s1"].tfidf_max >= by_id["s1"].tfidf_mean_nonzero


def test_documentation_aware_proxy_decomposition() -> None:
    records = [
        _record("s1", "Run git clone ./repo --branch=main with api_client v3.12 and timeout: 30."),
    ]

    row = compute_corpus_features(records)[0]

    assert row.identifier_density > 0
    assert row.command_path_flag_density > 0
    assert row.version_numeric_density > 0
    assert row.assignment_parameter_density > 0
    assert row.token_shape_complexity_mean > 0


def test_write_feature_table_writes_expected_columns(tmp_path) -> None:
    records = [_record("s1", "Check C:\\Tools\\app with --debug.")]
    rows = compute_corpus_features(records)
    output = tmp_path / "features.csv"

    write_feature_table(output, rows)

    with output.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        written = list(reader)

    assert reader.fieldnames == [
        "corpus_id",
        "sent_id",
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
    ]
    assert len(written) == 1
    assert written[0]["sent_id"] == "s1"
