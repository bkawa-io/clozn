"""Model-free tests for clozn.behavior.compare -- the additive compare-view metrics for corrective
retries (repeated-phrase count, format-changed flag)."""
from __future__ import annotations

from clozn.behavior import compare


def test_repeated_phrases_counts_distinct_trigrams_that_recur():
    text = "one two three four one two three five"
    # "one two three" is the only 3-word window that recurs -- 1 distinct repeated trigram.
    assert compare._repeated_phrase_count(text) == 1
    assert compare._repeated_phrase_count("no repeats at all here") == 0


def test_repeated_phrases_is_zero_for_short_text():
    assert compare._repeated_phrase_count("one two") == 0    # shorter than the 3-word window
    assert compare._repeated_phrase_count("") == 0


def test_repeated_phrases_is_case_insensitive_and_punctuation_agnostic():
    text = "Stop, stop! Stop repeating yourself. Stop repeating yourself again."
    # "stop repeating yourself" appears twice -- 1 distinct repeated trigram.
    assert compare._repeated_phrase_count(text) >= 1


def test_compare_metrics_shape():
    out = compare.compare_metrics("a b c a b c", "a b c")
    assert set(out) == {"repeated_phrases", "format_changed"}
    assert out["repeated_phrases"] == [1, 0]
    assert out["format_changed"] is False


def test_format_changed_true_when_code_fence_appears():
    orig = "Here is the answer: 42."
    repl = "Here is the answer:\n```python\nprint(42)\n```"
    out = compare.compare_metrics(orig, repl)
    assert out["format_changed"] is True


def test_format_changed_true_when_bullets_appear():
    orig = "Do the thing, then the other thing."
    repl = "- Do the thing\n- Then the other thing"
    assert compare.compare_metrics(orig, repl)["format_changed"] is True


def test_format_changed_false_when_only_wording_differs():
    orig = "- Do the thing\n- Then the other thing"
    repl = "- Do the task\n- Then the second task"
    assert compare.compare_metrics(orig, repl)["format_changed"] is False


def test_compare_metrics_never_raises_on_none_or_empty():
    assert compare.compare_metrics(None, None) == {"repeated_phrases": [0, 0], "format_changed": False}
    assert compare.compare_metrics("", "") == {"repeated_phrases": [0, 0], "format_changed": False}


def test_compare_metrics_omits_source_coverage():
    """Source coverage depends on the not-yet-shipped Sources lens (feature 07). This module must
    never fabricate it -- SEAMS.md rule 2: omit, never null-pad."""
    assert "source_coverage" not in compare.compare_metrics("a", "b")
