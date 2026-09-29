"""Shared text filters for annotation-facing sampling utilities.

These filters are intentionally applied only at sampling/template-generation time
to avoid changing core sentence extraction outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


STRICT_NATURAL_LANGUAGE_RULE_VERSION = "strict_natural_language_v1"
STRICT_NATURAL_LANGUAGE_REASON_CODES = (
    "template_markup",
    "code_markup",
    "table_markup",
    "url",
    "command_path_dominated",
    "heading_fragment",
    "list_fragment",
    "very_short_fragment",
    "low_alphabetic_content",
)

_URL_RE = re.compile(r"(?i)(?:\b(?:https?|ftp)://|\bwww\.|\bmailto:)[^\s]+")
_HTML_CODE_TAG_RE = re.compile(r"(?i)</?(?:code|pre|kbd)(?:\s[^>]*)?>")
_MARKDOWN_TABLE_SEPARATOR_RE = re.compile(
    r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$"
)
_WIKI_LINK_RE = re.compile(r"\[\[[^\]]+\]\]")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_LIST_RE = re.compile(r"^\s*(?:[-*+]\s+|\d{1,3}[.)]\s+)")
_WINDOWS_PATH_RE = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")
_ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=\S+$")
_SHORT_FLAG_RE = re.compile(r"^-[A-Za-z](?:\S*)?$")
_LONG_FLAG_RE = re.compile(r"^--[A-Za-z0-9][A-Za-z0-9_-]*(?:=\S+)?$")
_COMMAND_STARTERS = frozenset(
    {
        "ansible",
        "ansible-playbook",
        "cd",
        "cp",
        "curl",
        "docker",
        "export",
        "git",
        "kubectl",
        "make",
        "mkdir",
        "mv",
        "npm",
        "pip",
        "pip3",
        "python",
        "python3",
        "rm",
        "set",
        "sudo",
        "wget",
        "yarn",
    }
)


@dataclass(frozen=True)
class StrictNaturalLanguageDecision:
    """Auditable result from the frozen Round 2 row-selection predicate."""

    keep: bool
    reason_codes: tuple[str, ...]
    token_count: int
    alphabetic_ratio: float
    command_path_evidence_count: int


def _simple_token_count(text: str) -> int:
    return len(text.split())


def _alpha_ratio(text: str) -> float:
    non_space = [ch for ch in text if not ch.isspace()]
    if not non_space:
        return 0.0
    letters = sum(1 for ch in non_space if ch.isalpha())
    return letters / len(non_space)


def _looks_like_table_row(text: str) -> bool:
    # Wiki-link display separators (``[[target|label]]``) are not table cells.
    pipe_text = _WIKI_LINK_RE.sub("", text)
    if "|" not in pipe_text:
        return False
    stripped = pipe_text.strip()
    return bool(_MARKDOWN_TABLE_SEPARATOR_RE.fullmatch(stripped)) or pipe_text.count("|") >= 3


def _looks_like_inline_code_artifact(text: str) -> bool:
    stripped = text.strip()
    if "```" in stripped:
        return True
    # Common artifact pattern: an isolated inline-code span line
    return len(stripped) >= 3 and stripped.startswith("`") and stripped.endswith("`")


def _looks_like_code_markup(text: str) -> bool:
    return _looks_like_inline_code_artifact(text) or bool(_HTML_CODE_TAG_RE.search(text))


def _is_path_or_shell_syntax_token(token: str) -> bool:
    cleaned = token.strip("\"'`()[]{}.,:!")
    if not cleaned:
        return False
    if cleaned in {"$", ">", ">>>", "PS>", "&&", "||", "|", ";"}:
        return True
    if _LONG_FLAG_RE.fullmatch(cleaned) or _SHORT_FLAG_RE.fullmatch(cleaned):
        return True
    if _ASSIGNMENT_RE.fullmatch(cleaned):
        return True
    if _WINDOWS_PATH_RE.match(cleaned):
        return True
    if cleaned.startswith(("./", "../", "~/")):
        return True
    if cleaned.startswith("/") and "/" in cleaned[1:]:
        return True
    return cleaned.count("/") >= 2 or cleaned.count("\\") >= 2


def _command_path_dominated(text: str) -> tuple[bool, int]:
    tokens = text.split()
    if not tokens:
        return False, 0

    stripped = text.lstrip()
    prompt_led = stripped.startswith(("$ ", ">>> ", "PS> "))
    syntax_count = sum(_is_path_or_shell_syntax_token(token) for token in tokens)
    first = tokens[0].strip("\"'`()[]{}.,:!").lower()
    command_led = first in _COMMAND_STARTERS
    evidence_count = syntax_count + int(command_led)
    ratio = evidence_count / len(tokens)
    has_terminal_punct = text.rstrip().endswith((".", "?", "!"))

    dominated = prompt_led or (
        evidence_count >= 2
        and (
            ratio >= 0.50
            or (command_led and not has_terminal_punct and ratio >= 0.35)
        )
    )
    return dominated, evidence_count


def strict_natural_language_v1(text: str) -> StrictNaturalLanguageDecision:
    """Apply the frozen conservative Round 2 preprocessing-sensitivity rules.

    This function only selects existing rows. It never changes sentence text or
    scores, and its rules do not depend on corpus identity or model outcomes.
    Multiple reason codes are retained in a stable order for overlap audits.
    """

    value = text or ""
    stripped = value.strip()
    token_count = _simple_token_count(value)
    alpha_ratio = _alpha_ratio(value)
    command_path_dominated, evidence_count = _command_path_dominated(value)

    flags = {
        "template_markup": contains_template_markup(value),
        "code_markup": _looks_like_code_markup(value),
        "table_markup": looks_like_markdown_table_row(value),
        "url": bool(_URL_RE.search(value)),
        "command_path_dominated": command_path_dominated,
        "heading_fragment": bool(_HEADING_RE.match(value)),
        "list_fragment": bool(_LIST_RE.match(value))
        and not stripped.endswith((".", "?", "!")),
        "very_short_fragment": token_count < 5,
        "low_alphabetic_content": alpha_ratio < 0.60,
    }
    reasons = tuple(code for code in STRICT_NATURAL_LANGUAGE_REASON_CODES if flags[code])
    return StrictNaturalLanguageDecision(
        keep=not reasons,
        reason_codes=reasons,
        token_count=token_count,
        alphabetic_ratio=alpha_ratio,
        command_path_evidence_count=evidence_count,
    )


def is_annotatable_sentence(
    text: str,
    *,
    min_tokens: int = 8,
    max_tokens: int = 40,
    require_terminal_punct: bool = False,
    exclude_bullets: bool = False,
    exclude_html_code: bool = False,
) -> bool:
    """Return True when sentence text is suitable for human annotation.

    Rejection rules (deterministic):
    - Liquid/Jekyll template tags
    - Code fence / inline code artifact lines
    - Markdown table-like rows
    - token_count outside [min_tokens, max_tokens] using whitespace split
    - alphabetic character ratio < 0.6 over non-space characters
    - optionally require terminal punctuation (., ?, !)
    - optionally exclude bullet-like lines starting with '- '
    - optionally exclude HTML code tags (<code>...</code>)
    """
    if not text:
        return False

    if any(marker in text for marker in ("{%", "%}", "{{", "}}")):
        return False

    if _looks_like_inline_code_artifact(text):
        return False

    if _looks_like_table_row(text):
        return False

    count = _simple_token_count(text)
    if count < min_tokens or count > max_tokens:
        return False

    if _alpha_ratio(text) < 0.6:
        return False

    stripped = text.strip()
    if require_terminal_punct and (not stripped or stripped[-1] not in ".?!"):
        return False

    if exclude_bullets and text.lstrip().startswith("- "):
        return False

    if exclude_html_code and ("<code>" in text or "</code>" in text):
        return False

    return True


def token_count_whitespace(text: str) -> int:
    """Whitespace token count used by lightweight QA checks."""
    return _simple_token_count(text)


def alphabetic_ratio_non_space(text: str) -> float:
    """Alphabetic character ratio over non-space characters."""
    return _alpha_ratio(text)


def contains_template_markup(text: str) -> bool:
    """Detect Liquid/Jekyll-style template fragments."""
    return any(marker in text for marker in ("{%", "%}", "{{", "}}"))


def contains_code_fence_artifact(text: str) -> bool:
    """Detect code-fence or inline-code artifact fragments."""
    return _looks_like_inline_code_artifact(text)


def looks_like_markdown_table_row(text: str) -> bool:
    """Detect markdown table row fragments."""
    return _looks_like_table_row(text)
