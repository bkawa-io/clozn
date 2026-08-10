"""Tests for the ambient footer modes, shared signal policy, and contamination guard."""
from __future__ import annotations

from clozn.runs import receipt_footer


def _run(**kw):
    run = {
        "id": "run_x",
        "model": "llama-test",
        "response": "A normal answer.",
        "trace": {"tokens": ["A", " normal", " answer."]},
        "finish_reason": "stop",
    }
    run.update(kw)
    return run


def _context(*, pressure=False, omitted=False):
    prompt, window = (910, 1000) if pressure else (100, 1000)
    delivered = [{"segment_id": "seg_aaaaaaaaaaaaaaaa", "source_label": "system",
                  "included": not omitted}]
    return {
        "schema_version": "clozn.context-receipt.v1", "run_id": "run_x",
        "privacy": "metadata_only",
        "limits": {"prompt_tokens": prompt, "context_window_tokens": window, "generated_tokens": 3},
        "delivered": delivered, "assembled": [] if omitted else delivered,
        "omissions": ([{"segment_id": delivered[0]["segment_id"], "reason": "context_budget"}]
                       if omitted else []),
        "transformations": [], "termination": {"reason": "eos", "generated_tokens": 3},
    }


def test_off_never_appends_a_footer():
    assert receipt_footer.footer(_run(error="boom"), "http://h/r/x", mode="off") == ""


def test_exceptions_clean_run_is_silent():
    assert receipt_footer.footer(_run(), "http://h/r/x", mode="exceptions") == ""


def test_always_clean_run_is_receipt_link_only():
    footer = receipt_footer.footer(_run(), "http://h/r/x", mode="always")
    assert footer.count("receipt →") == 1
    assert receipt_footer.MARK in footer
    assert "·" not in footer.split(receipt_footer.MARK, 1)[1].replace("receipt →", "")


def test_one_attention_signal_uses_a_controlled_phrase():
    footer = receipt_footer.footer(_run(finish_reason="length"), "http://h/r/x", mode="exceptions")
    assert "cut off at token limit" in footer
    assert "http://h/r/x" in footer
    assert "mid-answer" not in footer


def test_three_attention_signals_are_limited_to_two_highest_priority_phrases():
    footer = receipt_footer.footer(
        _run(error="worker failed", finish_reason="length",
             context_receipt=_context(pressure=True)),
        "http://h/r/x", mode="exceptions",
    )
    line = footer.split("`", 2)[-1]
    phrases = [part.strip() for part in line.split(" · ") if part.strip() and not part.strip().startswith("receipt →")]
    assert phrases == ["run errored", "cut off at token limit"]
    assert "prompt is 91%" not in footer


def test_context_tension_uses_generic_controlled_phrase_and_no_source_text():
    run = _run(context_tension={
        "schema_version": "clozn.context-tension.v1", "measurement": {"state": "available"},
        "summary": {"answer_spans_with_tension": 1, "tension_pairs": 1},
    })
    footer = receipt_footer.footer(run, "http://h/r/x", mode="exceptions")
    assert "competing measured context effects" in footer
    assert "source" not in footer.lower()


def test_context_pressure_uses_literal_percentage_not_context_utilization_language():
    footer = receipt_footer.footer(_run(context_receipt=_context(pressure=True)), "http://h/r/x",
                                   mode="exceptions")
    assert "prompt is 91% of context window" in footer
    assert "% context used" not in footer


def test_influence_coverage_is_info_only_and_silent_in_exceptions_mode():
    run = _run(context_utilization={
        "schema_version": "clozn.context-utilization.v1", "measurement": {"state": "available"},
        "summary": {"prompt_sources": 14, "measured_sources": 8,
                     "sources_with_clear_measured_effect": 2,
                     "sources_below_measured_floor": 6, "sources_not_measured": 6},
        "sources": [],
    })
    assert receipt_footer.footer(run, "http://h/r/x", mode="exceptions") == ""


def test_echo_stripping_removes_exception_always_and_multisignal_shapes():
    runs = [
        _run(finish_reason="length"),
        _run(),
        _run(error="boom", finish_reason="length", context_receipt=_context(pressure=True)),
    ]
    modes = ["exceptions", "always", "exceptions"]
    footers = [receipt_footer.footer(run, "http://h/r/x", mode=mode) for run, mode in zip(runs, modes)]
    messages = [{"role": "assistant", "content": "answer" + footer} for footer in footers]
    out = receipt_footer.strip_footers(messages)
    assert [item["content"] for item in out] == ["answer", "answer", "answer"]
    assert all(receipt_footer.MARK in item["content"] for item in messages)


def test_user_and_system_pasted_footer_like_text_remain_untouched():
    footer = receipt_footer.footer(_run(finish_reason="length"), "http://h/r/x", mode="exceptions")
    messages = [
        {"role": "user", "content": "pasted " + footer},
        {"role": "system", "content": "policy " + footer},
    ]
    assert receipt_footer.strip_footers(messages) == messages


def test_multipart_assistant_footer_is_stripped_without_mutating_input():
    footer = receipt_footer.footer(_run(finish_reason="length"), "http://h/r/x", mode="exceptions")
    messages = [{"role": "assistant", "content": [
        {"type": "text", "text": "answer" + footer}, {"type": "text", "text": "kept"},
    ]}]
    out = receipt_footer.strip_footers(messages)
    assert out[0]["content"][0]["text"] == "answer"
    assert messages[0]["content"][0]["text"].endswith(footer)


def test_summary_legacy_helper_remains_defensive():
    for value in (None, {}, {"trace": "bad"}, {"trace": {"tokens": None}}):
        summary = receipt_footer.summary(value)
        assert summary["n_tokens"] == 0 and summary["mean_conf"] is None
