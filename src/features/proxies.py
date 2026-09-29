"""Deterministic proxy feature computation for sentence specificity analysis."""
from __future__ import annotations

import csv
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

try:
    from nltk.tokenize import TreebankWordTokenizer
except ModuleNotFoundError:
    TreebankWordTokenizer = None

from src.sentences.table import SentenceRecord


_TOKENIZER = TreebankWordTokenizer() if TreebankWordTokenizer is not None else None
_FALLBACK_TOKEN_RE = re.compile(
    r"[A-Za-z]:\\[^\s]+|\.{0,2}/[^\s]+|/[^\s]+|--?[A-Za-z0-9][A-Za-z0-9_-]*(?:=[^\s]+)?|"
    r"[A-Za-z_][A-Za-z0-9_.-]*[:=][^\s]*|[A-Za-z0-9_./:-]+|[^\w\s]"
)

_FLAG_RE = re.compile(r"^--[a-zA-Z0-9][a-zA-Z0-9_-]*$")
_PATH_RE = re.compile(r"^(?:[A-Za-z]:\\|\.\.?/|/|[^\s]*[/\\][^\s]*)")
_VERSION_RE = re.compile(r"^v?\d+\.\d+(?:\.\d+)*(?:[a-zA-Z0-9-]+)?$")
_SNAKE_RE = re.compile(r"^[a-z]+(?:_[a-z0-9]+)+$")
_CAMEL_RE = re.compile(r"^(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+)$")
_DOTTED_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+$")
_ACRONYM_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,}$")
_COMMAND_RE = re.compile(
    r"^(?:ansible-playbook|bash|cargo|conda|curl|docker|git|helm|java|kubectl|make|mvn|npm|"
    r"pip|pip3|python|python3|pytest|ssh|sudo|terraform|uv|wget|yarn|[A-Za-z0-9_-]+\.exe)$",
    re.IGNORECASE,
)
_NUMERIC_RE = re.compile(r"^(?:\d+|\d+(?:[._:/-]\d+)+[A-Za-z0-9._:/%-]*|\d+[A-Za-z%]+)$")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*=.+$")
_YAML_KEY_TOKEN_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*:$")
_YAML_KEY_SPAN_RE = re.compile(r"(?<!\S)[A-Za-z_][A-Za-z0-9_.-]*\s*:\s*\S")
_FLAG_VALUE_SPAN_RE = re.compile(r"(?<!\S)-{1,2}[A-Za-z0-9][A-Za-z0-9_-]*(?:=|\s+)\S")


@dataclass(frozen=True)
class SentenceFeatureRow:
    corpus_id: str
    sent_id: str
    tfidf_mean_nonzero: float
    tfidf_max: float
    technical_token_ratio: float
    identifier_density: float
    command_path_flag_density: float
    version_numeric_density: float
    assignment_parameter_density: float
    token_shape_complexity_mean: float
    token_count: int
    char_count: int


def _tokenize(text: str) -> list[str]:
    if _TOKENIZER is not None:
        return _TOKENIZER.tokenize(text)
    return _FALLBACK_TOKEN_RE.findall(text)


def _is_technical_token(token: str) -> bool:
    core = token.strip("`'\"()[]{}<>,;:")
    if not core:
        return False
    return any(
        pattern.match(core)
        for pattern in (_FLAG_RE, _PATH_RE, _VERSION_RE, _SNAKE_RE, _CAMEL_RE, _DOTTED_RE)
    )


def _core_token(token: str) -> str:
    return token.strip("`'\"()[]{}<>,;.")


def _is_identifier_token(token: str) -> bool:
    core = _core_token(token).strip(":")
    if not core:
        return False
    return any(pattern.match(core) for pattern in (_SNAKE_RE, _CAMEL_RE, _DOTTED_RE, _ACRONYM_RE))


def _is_command_path_flag_token(token: str) -> bool:
    core = _core_token(token).strip(":")
    if not core:
        return False
    return any(pattern.match(core) for pattern in (_FLAG_RE, _PATH_RE, _COMMAND_RE))


def _is_version_numeric_token(token: str) -> bool:
    core = _core_token(token).strip(":")
    if not core:
        return False
    return bool(_VERSION_RE.match(core) or _NUMERIC_RE.match(core))


def _is_assignment_parameter_token(token: str) -> bool:
    core = _core_token(token)
    if not core:
        return False
    return bool(
        _ASSIGNMENT_RE.match(core)
        or _YAML_KEY_TOKEN_RE.match(core)
        or ("=" in core and _FLAG_RE.match(core.split("=", 1)[0]))
    )


def _assignment_parameter_count(tokens: Sequence[str], text: str) -> int:
    token_count = sum(1 for token in tokens if _is_assignment_parameter_token(token))
    span_count = len(_YAML_KEY_SPAN_RE.findall(text)) + len(_FLAG_VALUE_SPAN_RE.findall(text))
    return token_count + span_count


def _token_shape_complexity(token: str) -> int:
    core = _core_token(token)
    if not core:
        return 0
    score = 0
    if any(char.islower() for char in core) and any(char.isupper() for char in core):
        score += 1
    if any(char.isdigit() for char in core):
        score += 1
    if any(char in core for char in "_.:/-"):
        score += 1
    if any(not char.isalnum() and char not in "_.:/-" for char in core):
        score += 1
    if re.search(r"[a-z][A-Z]|[A-Z][a-z]", core):
        score += 1
    return score


def compute_corpus_features(records: Sequence[SentenceRecord]) -> list[SentenceFeatureRow]:
    """Compute per-sentence proxy features using per-corpus TF-IDF statistics."""
    tokenized: list[list[str]] = [_tokenize(record.sent_text) for record in records]
    lowered: list[list[str]] = [[tok.lower() for tok in sent_tokens] for sent_tokens in tokenized]

    document_frequency: Counter[str] = Counter()
    for sent_tokens in lowered:
        document_frequency.update(set(sent_tokens))

    sentence_count = len(records)
    idf: dict[str, float] = {
        token: math.log((1.0 + sentence_count) / (1.0 + freq)) + 1.0
        for token, freq in document_frequency.items()
    }

    feature_rows: list[SentenceFeatureRow] = []
    for record, sent_tokens, sent_tokens_lower in zip(records, tokenized, lowered):
        token_count = len(sent_tokens)
        char_count = len(record.sent_text)

        tfidf_values: list[float] = []
        if token_count:
            tf = Counter(sent_tokens_lower)
            tfidf_values = [
                (count / token_count) * idf[token]
                for token, count in tf.items()
            ]

        technical_count = sum(1 for token in sent_tokens if _is_technical_token(token))
        technical_ratio = (technical_count / token_count) if token_count else 0.0
        identifier_count = sum(1 for token in sent_tokens if _is_identifier_token(token))
        command_path_flag_count = sum(1 for token in sent_tokens if _is_command_path_flag_token(token))
        version_numeric_count = sum(1 for token in sent_tokens if _is_version_numeric_token(token))
        assignment_parameter_count = _assignment_parameter_count(sent_tokens, record.sent_text)
        shape_complexity_total = sum(_token_shape_complexity(token) for token in sent_tokens)

        tfidf_mean = (sum(tfidf_values) / len(tfidf_values)) if tfidf_values else 0.0
        tfidf_max = max(tfidf_values) if tfidf_values else 0.0

        feature_rows.append(
            SentenceFeatureRow(
                corpus_id=record.corpus_id,
                sent_id=record.sent_id,
                tfidf_mean_nonzero=tfidf_mean,
                tfidf_max=tfidf_max,
                technical_token_ratio=technical_ratio,
                identifier_density=(identifier_count / token_count) if token_count else 0.0,
                command_path_flag_density=(command_path_flag_count / token_count) if token_count else 0.0,
                version_numeric_density=(version_numeric_count / token_count) if token_count else 0.0,
                assignment_parameter_density=(assignment_parameter_count / token_count) if token_count else 0.0,
                token_shape_complexity_mean=(shape_complexity_total / token_count) if token_count else 0.0,
                token_count=token_count,
                char_count=char_count,
            )
        )

    return feature_rows


def write_feature_table(path: Path, rows: Sequence[SentenceFeatureRow]) -> None:
    """Persist computed feature rows to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
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
        )
        for row in rows:
            writer.writerow(
                [
                    row.corpus_id,
                    row.sent_id,
                    f"{row.tfidf_mean_nonzero:.8f}",
                    f"{row.tfidf_max:.8f}",
                    f"{row.technical_token_ratio:.8f}",
                    f"{row.identifier_density:.8f}",
                    f"{row.command_path_flag_density:.8f}",
                    f"{row.version_numeric_density:.8f}",
                    f"{row.assignment_parameter_density:.8f}",
                    f"{row.token_shape_complexity_mean:.8f}",
                    row.token_count,
                    row.char_count,
                ]
            )
