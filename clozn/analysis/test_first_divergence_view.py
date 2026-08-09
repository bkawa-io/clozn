"""test_first_divergence_view.py -- model-free unit tests for the `first_divergence_view` projection
nested in every `clozn.analysis.model_diff.diff_runs()` result (`clozn.first-divergence-view.v1`).

Co-located with test_model_diff.py by the same convention (clozn/analysis/test_*.py). Synthetic run
records only: no engine, no network, no live server. The view is a PROJECTION over `diff_runs()`'s
existing comparison -- these tests assert it agrees with `diff["first_divergence"]`/
`diff["common_prefix_len"]` exactly, never independently re-derives them.
"""
from __future__ import annotations

import copy

import pytest

from clozn import schemas
from clozn.analysis.model_diff import FIRST_DIVERGENCE_VIEW_SCHEMA, diff_runs


def _run(rid, prompt, tokens, *, confidence=None, alternatives=None, token_ids=None, response=None):
    """A minimal but store-shaped run record, mirroring test_model_diff.py's own `_run` helper.
    tokens=None -> a traceless (light-tier) run."""
    rec = {
        "id": rid,
        "model": "qwen2.5-7b",
        "substrate": "engine",
        "messages": [{"role": "user", "content": prompt}],
        "response": response if response is not None else "".join(tokens or []),
        "meta": {},
        "trace": {},
    }
    if tokens is not None:
        rec["trace"] = {"tokens": list(tokens)}
        if confidence is not None:
            rec["trace"]["confidence"] = list(confidence)
        if alternatives is not None:
            rec["trace"]["alternatives"] = list(alternatives)
        if token_ids is not None:
            rec["trace"]["token_ids"] = list(token_ids)
    return rec


_PARIS = ["The", " capital", " is", " Paris", "."]
_CONF_A = [0.99, 0.95, 0.9, 0.44, 0.97]


def _view(run_a, run_b, **kwargs):
    out = diff_runs(run_a, run_b, **kwargs)
    view = out["first_divergence_view"]
    schemas.validate(view, FIRST_DIVERGENCE_VIEW_SCHEMA)
    return out, view


# ======================================================================================================
# 1. Simple token mismatch
# ======================================================================================================

def test_simple_token_mismatch():
    a = _run("run_a", "capital?", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "capital?", ["The", " capital", " is", " Lyon", "."],
            confidence=[0.99, 0.95, 0.9, 0.39, 0.9])
    out, view = _view(a, b)
    assert view["common_prefix"] == {"token_count": 3, "last_shared_index": 2}
    assert view["divergence"]["index"] == 3
    assert view["divergence"]["kind"] == "token_mismatch"
    # agrees exactly with the canonical diff
    assert view["divergence"]["index"] == out["first_divergence"]["index"]
    assert view["common_prefix"]["token_count"] == out["common_prefix_len"]


# ======================================================================================================
# 2. Divergence at token zero
# ======================================================================================================

def test_divergence_at_token_zero():
    a = _run("run_a", "q", ["Yes", "."], confidence=[0.9, 0.9])
    b = _run("run_b", "q", ["No", " way", "."], confidence=[0.8, 0.7, 0.9])
    _out, view = _view(a, b)
    assert view["common_prefix"] == {"token_count": 0, "last_shared_index": None}
    assert view["divergence"]["index"] == 0


# ======================================================================================================
# 3 / 4. Length mismatch, either side exhausted first
# ======================================================================================================

def test_length_mismatch_a_ends_first():
    a = _run("run_a", "q", _PARIS[:3], confidence=_CONF_A[:3])
    b = _run("run_b", "q", _PARIS, confidence=_CONF_A)
    _out, view = _view(a, b)
    fd = view["divergence"]
    assert fd["kind"] == "length_mismatch"
    assert fd["a"] == {"piece": None, "token_id": None, "confidence": None}
    assert fd["b"]["piece"] == " Paris"
    assert fd["b"]["confidence"] == pytest.approx(0.44)


def test_length_mismatch_b_ends_first():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", _PARIS[:3], confidence=_CONF_A[:3])
    _out, view = _view(a, b)
    fd = view["divergence"]
    assert fd["kind"] == "length_mismatch"
    assert fd["b"] == {"piece": None, "token_id": None, "confidence": None}
    assert fd["a"]["piece"] == " Paris"


# ======================================================================================================
# 5. Identical traces
# ======================================================================================================

def test_identical_traces():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", list(_PARIS), confidence=list(_CONF_A))
    _out, view = _view(a, b)
    assert view["state"] == "identical"
    assert view["divergence"] is None
    assert view["common_prefix"] == {"token_count": 5, "last_shared_index": 4}


# ======================================================================================================
# 6 / 7 / 8. Missing trace(s)
# ======================================================================================================

def test_missing_a_trace():
    a = _run("run_a", "q", None, response="anything")
    b = _run("run_b", "q", _PARIS)
    _out, view = _view(a, b)
    assert view["state"] == "trace_unavailable"
    assert view["trace_missing"] == ["a"]
    assert view["divergence"] is None


def test_missing_b_trace():
    a = _run("run_a", "q", _PARIS)
    b = _run("run_b", "q", None, response="anything")
    _out, view = _view(a, b)
    assert view["state"] == "trace_unavailable"
    assert view["trace_missing"] == ["b"]


def test_both_traces_missing():
    a = _run("run_a", "q", None, response="x")
    b = _run("run_b", "q", None, response="y")
    _out, view = _view(a, b)
    assert view["state"] == "trace_unavailable"
    assert view["trace_missing"] == ["a", "b"]


# ======================================================================================================
# 9. Confidence copied exactly
# ======================================================================================================

def test_confidence_copied_exactly_from_recorded_traces():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."],
            confidence=[0.99, 0.95, 0.9, 0.39, 0.9])
    _out, view = _view(a, b)
    assert view["divergence"]["a"]["confidence"] == pytest.approx(0.44)
    assert view["divergence"]["b"]["confidence"] == pytest.approx(0.39)


# ======================================================================================================
# 10. Token IDs recorded when available, never invented
# ======================================================================================================

def test_token_ids_recorded_when_available_never_invented():
    a = _run("run_a", "q", _PARIS, token_ids=[1, 2, 3, 18273, 5])
    b_with_ids = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."], token_ids=[1, 2, 3, 39182, 5])
    _out, view = _view(a, b_with_ids)
    assert view["divergence"]["a"]["token_id"] == 18273
    assert view["divergence"]["b"]["token_id"] == 39182

    b_no_ids = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    _out2, view2 = _view(a, b_no_ids)
    assert view2["divergence"]["a"]["token_id"] == 18273  # a still has one
    assert view2["divergence"]["b"]["token_id"] is None   # never invented


# ======================================================================================================
# 11 / 12. Symmetric alternative lookup
# ======================================================================================================

def test_a_considered_b_found_with_rank_and_prob():
    alts_a = [[], [], [], [{"piece": " Rome", "prob": 0.22}, {"piece": " Lyon", "prob": 0.15}], []]
    a = _run("run_a", "q", _PARIS, alternatives=alts_a)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    _out, view = _view(a, b)
    lookup = view["alternatives"]["a_considered_b"]
    assert lookup["checked"] is True
    assert lookup["found"] is True
    assert lookup["rank"] == 1
    assert lookup["prob"] == pytest.approx(0.15)


def test_b_considered_a_is_the_symmetric_reverse_lookup():
    alts_b = [[], [], [], [{"piece": " Paris", "prob": 0.31}], []]
    a = _run("run_a", "q", _PARIS)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."], alternatives=alts_b)
    _out, view = _view(a, b)
    lookup = view["alternatives"]["b_considered_a"]
    assert lookup["checked"] is True
    assert lookup["found"] is True
    assert lookup["rank"] == 0
    assert lookup["prob"] == pytest.approx(0.31)
    # and the reverse-of-reverse: a's own alternatives were never consulted for this lookup
    assert view["alternatives"]["a_considered_b"]["checked"] is False


# ======================================================================================================
# 13. Alternative not in the recorded top-k
# ======================================================================================================

def test_alternative_not_captured_is_checked_true_found_false():
    alts_a = [[], [], [], [{"piece": " Rome", "prob": 0.22}], []]
    a = _run("run_a", "q", _PARIS, alternatives=alts_a)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    _out, view = _view(a, b)
    lookup = view["alternatives"]["a_considered_b"]
    assert lookup["checked"] is True
    assert lookup["found"] is False
    assert lookup["rank"] is None


# ======================================================================================================
# 14. No alternatives recorded -- unknown, never collapsed to false
# ======================================================================================================

def test_no_alternatives_recorded_is_unknown_not_false():
    a = _run("run_a", "q", _PARIS)  # no alternatives array at all
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    _out, view = _view(a, b)
    lookup = view["alternatives"]["a_considered_b"]
    assert lookup["checked"] is False
    assert lookup["found"] is None


# ======================================================================================================
# 15. Compact token window
# ======================================================================================================

def test_window_returns_exactly_the_requested_counts_where_available():
    toks_a = [f"a{i}" for i in range(20)]
    toks_b = list(toks_a[:10]) + ["DIFF"] + list(toks_a[11:])
    a = _run("run_a", "q", toks_a)
    b = _run("run_b", "q", toks_b)
    _out, view = _view(a, b, context_before=4, context_after=8)
    window = view["window"]
    assert window["divergence_index"] == 10
    assert window["start_index"] == 6
    assert [t["index"] for t in window["common_prefix"]] == [6, 7, 8, 9]
    assert [t["index"] for t in window["a"]] == list(range(10, 18))
    assert [t["index"] for t in window["b"]] == list(range(10, 18))
    assert window["a"][0]["piece"] == toks_a[10]
    assert window["b"][0]["piece"] == "DIFF"


def test_window_clips_near_the_beginning_and_end_of_the_trace():
    toks_a = ["a0", "a1", "MISMATCH"]
    toks_b = ["a0", "a1", "other"]
    a = _run("run_a", "q", toks_a)
    b = _run("run_b", "q", toks_b)
    _out, view = _view(a, b, context_before=4, context_after=8)
    window = view["window"]
    assert window["start_index"] == 0  # clipped: common_prefix_len(2) - context_before(4) < 0
    assert len(window["common_prefix"]) == 2
    assert len(window["a"]) == 1  # only one token left from the divergence index onward
    assert len(window["b"]) == 1


# ======================================================================================================
# 16. Exact recorded-answer offsets, including Unicode code points
# ======================================================================================================

def test_exact_recorded_answer_offsets_are_unicode_code_points_not_utf8_bytes():
    toks_a = ["café ", "is", " nice"]      # "café is nice"
    toks_b = ["café ", "is", " \U0001F642"]  # "café is 🙂" -- an astral (surrogate-pair-in-UTF-16) char
    a = _run("run_a", "q", toks_a)
    b = _run("run_b", "q", toks_b)
    _out, view = _view(a, b)
    fd = view["divergence"]
    assert fd["index"] == 2
    loc = view["recorded_answer_location"]
    # "café " (5 code points) + "is" (2 code points) = 7 code points in, BYTE offset would differ (café's
    # é is 2 UTF-8 bytes, and 🙂 is a single Unicode code point but a UTF-16 surrogate pair / 4 UTF-8 bytes).
    assert loc["a"] == {"state": "exact", "unit": "unicode_code_points", "interval": "half_open",
                        "start": 7, "end": 12}      # " nice" = 5 code points
    assert loc["b"] == {"state": "exact", "unit": "unicode_code_points", "interval": "half_open",
                        "start": 7, "end": 9}        # "🙂" = 2 code points (space + one emoji code point)


# ======================================================================================================
# 17. Trace/response mismatch -- location unavailable, divergence still available
# ======================================================================================================

def test_trace_response_mismatch_makes_location_unavailable_not_divergence():
    a = _run("run_a", "q", _PARIS, response="a completely different recorded response")
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    _out, view = _view(a, b)
    assert view["state"] == "available"
    assert view["divergence"]["index"] == 3   # token-level divergence still valid
    assert view["recorded_answer_location"]["a"] == {
        "state": "unavailable", "reason": "trace_text_does_not_match_recorded_response",
    }
    # b's own trace DID reconstruct its own response exactly -- independent per side
    assert view["recorded_answer_location"]["b"]["state"] == "exact"


# ======================================================================================================
# 18. Divergence beyond the 200-position display cap
# ======================================================================================================

def test_divergence_beyond_200_positions_still_fully_resolved():
    long_a = ["tok"] * 350
    long_b = ["tok"] * 349 + ["OTHER"]
    a = _run("run_a", "q", long_a)
    b = _run("run_b", "q", long_b)
    out, view = _view(a, b)
    assert len(out["positions"]) == 200
    assert out["positions_truncated"] is True
    assert view["divergence"]["index"] == 349
    assert view["common_prefix"]["token_count"] == 349
    assert view["window"]["a"][0]["index"] == 349


# ======================================================================================================
# 19. max_positions=0 does not disable the view
# ======================================================================================================

def test_max_positions_zero_does_not_disable_the_view():
    long_a = ["tok"] * 350
    long_b = ["tok"] * 349 + ["OTHER"]
    a = _run("run_a", "q", long_a)
    b = _run("run_b", "q", long_b)
    out, view = _view(a, b, max_positions=0)
    assert out["positions"] == []
    assert view["state"] == "available"
    assert view["divergence"]["index"] == 349


def test_max_positions_zero_with_identical_long_traces_is_still_identical():
    long_a = ["tok"] * 350
    long_b = ["tok"] * 350
    _out, view = _view(a := _run("run_a", "q", long_a), _run("run_b", "q", long_b), max_positions=0)
    assert view["state"] == "identical"
    assert view["common_prefix"]["token_count"] == 350


# ======================================================================================================
# 20. Prompt mismatch is explicitly surfaced, divergence remains available
# ======================================================================================================

def test_prompt_mismatch_surfaced_but_divergence_still_available():
    a = _run("run_a", "capital of France?", _PARIS)
    b = _run("run_b", "capital of Italy?", ["The", " capital", " is", " Rome", "."])
    _out, view = _view(a, b)
    assert view["comparison"] == {"prompts_match": False, "controlled": False}
    assert view["state"] == "available"
    assert view["divergence"] is not None


# ======================================================================================================
# 21. Run immutability
# ======================================================================================================

def test_runs_are_never_mutated():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    before_a, before_b = copy.deepcopy(a), copy.deepcopy(b)
    diff_runs(a, b)
    assert a == before_a
    assert b == before_b


# ======================================================================================================
# 22. Determinism
# ======================================================================================================

def test_deterministic_output():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."], confidence=[0.99, 0.95, 0.9, 0.39, 0.9])
    first = diff_runs(copy.deepcopy(a), copy.deepcopy(b))["first_divergence_view"]
    second = diff_runs(copy.deepcopy(a), copy.deepcopy(b))["first_divergence_view"]
    assert first == second


# ======================================================================================================
# 23. No engine/model/worker access
# ======================================================================================================

def test_no_engine_or_worker_access(monkeypatch):
    def _explode(*_a, **_kw):
        raise AssertionError("first-divergence view touched an engine/model/worker seam")

    from clozn.server import app as ctx
    monkeypatch.setattr(ctx, "active_engine", _explode, raising=False)
    monkeypatch.setattr(ctx, "ENGINE", None, raising=False)
    import clozn.server.model_routing as model_routing
    monkeypatch.setattr(model_routing, "select_control_model_for_run", _explode, raising=False)

    a = _run("run_a", "q", _PARIS)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."])
    out = diff_runs(a, b)
    assert out["first_divergence_view"]["state"] == "available"


# ======================================================================================================
# Backward compatibility: existing top-level diff fields are unaffected by the new nested view
# ======================================================================================================

def test_existing_diff_fields_unaffected_by_the_new_view():
    a = _run("run_a", "q", _PARIS, confidence=_CONF_A)
    b = _run("run_b", "q", ["The", " capital", " is", " Lyon", "."], confidence=[0.99, 0.95, 0.9, 0.39, 0.9])
    out = diff_runs(a, b)
    assert out["common_prefix_len"] == 3
    assert out["first_divergence"]["index"] == 3
    assert out["first_divergence"]["kind"] == "token_mismatch"
    assert len(out["positions"]) == 5
    assert "summary" in out
    assert out["summary"]["b_was_alternative_in_a"]["checked"] is False
    # and the view agrees exactly with the canonical fields it was built from
    assert out["first_divergence_view"]["divergence"]["index"] == out["first_divergence"]["index"]
