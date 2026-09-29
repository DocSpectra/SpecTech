"""Tests for sentence splitting and repair."""
from __future__ import annotations

import pytest

from src.sentences.repair import repair_sentences
from src.sentences.table import build_sentence_records
from src.sentences.table import stable_sentence_id
from src.sentences.table import validate_sentence_records
from src.speciteller.tokenize import tokenize_for_speciteller


def test_repair_merges_artifact_only() -> None:
    sentences = ["Install the package.", ")", "Then restart."]
    repaired = list(repair_sentences(sentences))
    assert repaired[0] == "Install the package. )"
    assert repaired[1] == "Then restart."


def test_repair_merges_short_bullet_prefix() -> None:
    sentences = ["a.", "install dependencies.", "Next step."]
    repaired = list(repair_sentences(sentences))
    assert repaired[0] == "a. install dependencies."
    assert repaired[1] == "Next step."


def test_stable_sentence_id_is_deterministic() -> None:
    first = stable_sentence_id("c1", "doc", 0, "Sentence.")
    second = stable_sentence_id("c1", "doc", 0, "Sentence.")
    assert first == second


def test_sentence_record_validation_detects_duplicates() -> None:
    records = build_sentence_records("c1", "doc", ["A.", "B."])
    dup = records[0]
    with pytest.raises(ValueError):
        validate_sentence_records([dup, dup])


def test_speciteller_tokenization() -> None:
    text = "Tokenize this, please."
    tokenized = tokenize_for_speciteller(text)
    assert tokenized == "Tokenize this , please ."