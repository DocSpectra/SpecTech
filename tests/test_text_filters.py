"""Tests for annotation-facing text filtering used by sampling scripts."""
from __future__ import annotations

from src.utils.text_filters import is_annotatable_sentence
from src.utils.text_filters import strict_natural_language_v1


def test_is_annotatable_sentence_rejects_template_tags() -> None:
    assert not is_annotatable_sentence("{% if user %} show message {% endif %}")
    assert not is_annotatable_sentence("Hello {{ user.name }} welcome back")


def test_is_annotatable_sentence_rejects_code_fence_and_table_rows() -> None:
    assert not is_annotatable_sentence("``` bash")
    assert not is_annotatable_sentence("| Name | Value | Notes |")
    assert not is_annotatable_sentence("| --- | --- |")


def test_table_detector_does_not_treat_wiki_link_pipes_as_table_cells() -> None:
    text = (
        "This [[political philosophy|philosophy]] discusses [[social hierarchy|hierarchy]] "
        "and the [[state (polity)|state]] in natural prose."
    )
    assert is_annotatable_sentence(text)


def test_is_annotatable_sentence_rejects_short_or_low_alpha_ratio() -> None:
    assert not is_annotatable_sentence("too short now")
    assert not is_annotatable_sentence("1234 ==== -- ++ //")


def test_is_annotatable_sentence_rejects_long_by_default() -> None:
    long_text = " ".join(["token"] * 41)
    assert not is_annotatable_sentence(long_text)


def test_is_annotatable_sentence_supports_token_bound_overrides() -> None:
    short_text = "Install package now with pip"
    long_text = " ".join(["token"] * 50)
    assert is_annotatable_sentence(short_text, min_tokens=5, max_tokens=60)
    assert is_annotatable_sentence(long_text, min_tokens=5, max_tokens=60)


def test_is_annotatable_sentence_terminal_punctuation_toggle() -> None:
    text_no_punct = "This sentence has enough words but no terminal punctuation"
    text_with_punct = "This sentence has enough words and terminal punctuation."
    assert is_annotatable_sentence(text_no_punct)
    assert not is_annotatable_sentence(text_no_punct, require_terminal_punct=True)
    assert is_annotatable_sentence(text_with_punct, require_terminal_punct=True)


def test_is_annotatable_sentence_excludes_bullets_when_enabled() -> None:
    bullet = "- This bullet line has enough words for token filtering."
    plain = "This plain sentence has enough words for filtering checks."
    assert is_annotatable_sentence(bullet)
    assert not is_annotatable_sentence(bullet, exclude_bullets=True)
    assert is_annotatable_sentence(plain, exclude_bullets=True)


def test_is_annotatable_sentence_excludes_html_code_when_enabled() -> None:
    html_code = "This sentence includes <code>python -m pip install</code> inline markup."
    plain = "This sentence includes inline command references in prose form."
    assert is_annotatable_sentence(html_code)
    assert not is_annotatable_sentence(html_code, exclude_html_code=True)
    assert is_annotatable_sentence(plain, exclude_html_code=True)


def test_is_annotatable_sentence_accepts_natural_language_line() -> None:
    text = "Use this command to initialize the repository before deployment."
    assert is_annotatable_sentence(text)


def test_strict_natural_language_v1_accepts_natural_language() -> None:
    decision = strict_natural_language_v1(
        "This procedure initializes the repository before the deployment begins."
    )
    assert decision.keep
    assert decision.reason_codes == ()


def test_strict_natural_language_v1_marks_markup_reasons() -> None:
    assert strict_natural_language_v1(
        "{% if ready %} Render this documented sentence now. {% endif %}"
    ).reason_codes == ("template_markup",)
    assert strict_natural_language_v1(
        "<code>python -m pip install package</code>"
    ).reason_codes[:1] == ("code_markup",)
    assert strict_natural_language_v1(
        "| Name | Current value | Documented meaning |"
    ).reason_codes[:1] == ("table_markup",)


def test_strict_natural_language_v1_requires_real_html_code_tags() -> None:
    regex_prose = (
        "The named group (?P<prefix>[a-z]+) captures the documented prefix value."
    )
    assert "code_markup" not in strict_natural_language_v1(regex_prose).reason_codes


def test_strict_natural_language_v1_marks_urls() -> None:
    decision = strict_natural_language_v1(
        "Read the complete guide at https://example.org/docs before proceeding."
    )
    assert decision.reason_codes == ("url",)


def test_strict_natural_language_v1_command_path_rule_is_conservative() -> None:
    command = strict_natural_language_v1("python -m pip install package")
    prose = strict_natural_language_v1("Use /opt/app/config and --force when needed.")
    prompt = strict_natural_language_v1("$ run the documented deployment command now")
    assert "command_path_dominated" in command.reason_codes
    assert prose.keep
    assert "command_path_dominated" in prompt.reason_codes


def test_strict_natural_language_v1_heading_and_list_fragments() -> None:
    assert "heading_fragment" in strict_natural_language_v1(
        "## Configure the deployment environment"
    ).reason_codes
    assert "list_fragment" in strict_natural_language_v1(
        "- Configure the deployment environment now"
    ).reason_codes
    assert strict_natural_language_v1(
        "- Configure the deployment environment before continuing."
    ).keep


def test_strict_natural_language_v1_token_boundary() -> None:
    assert "very_short_fragment" in strict_natural_language_v1(
        "Exactly four natural words"
    ).reason_codes
    assert strict_natural_language_v1("Exactly five natural words remain").keep


def test_strict_natural_language_v1_alpha_ratio_boundary() -> None:
    exact = strict_natural_language_v1("abc12 abc12 abc12 abc12 abc12")
    below = strict_natural_language_v1("abc123 abc12 abc12 abc12 abc12")
    assert exact.alphabetic_ratio == 0.6
    assert "low_alphabetic_content" not in exact.reason_codes
    assert below.alphabetic_ratio < 0.6
    assert "low_alphabetic_content" in below.reason_codes


def test_strict_natural_language_v1_retains_stable_overlap_order() -> None:
    decision = strict_natural_language_v1("``` https://x/1 === +++")
    assert decision.reason_codes == (
        "code_markup",
        "url",
        "very_short_fragment",
        "low_alphabetic_content",
    )
