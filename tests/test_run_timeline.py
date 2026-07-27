"""test_run_timeline -- model-free tests for research/run_timeline.py (the RunEvent timeline: reshape a
stored run into an ordered list of typed events the Studio can render as a timeline strip).

Drives run_timeline.timeline() directly against fixture run dicts, mostly built through the REAL
runlog.record() + get_run() round trip (mirrors test_explain.py's `store` fixture) so the trace/memory/
behavior shapes are byte-for-byte what the real logging paths persist, not a hand-rolled guess at the
schema. A couple of cases are hand-built on purpose to exercise shapes runlog itself can't currently
produce (garbage input, a maximally malformed but dict-shaped run) -- run_timeline must degrade cleanly on
both, per the module's own "never raises" contract.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.runs import timeline as run_timeline      # noqa: E402
import clozn.runs.store as runlog             # noqa: E402


@pytest.fixture
def store(tmp_path, monkeypatch):
    """Redirect runlog's run store to a tmp dir for the duration of one test (mirrors test_runlog.py)."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return runlog


# ------------------------------------------------------------------------------------------- fixture: full run

def test_full_run_ordered_events_with_pluralization_and_hesitations(store):
    tokens = ["The", " sky", " is", " blue", "."]
    confidence = [0.95, 0.30, 0.92, 0.41, 0.99]
    alternatives = [[], [{"piece": " sea", "prob": 0.22}], [],
                    [{"piece": " grey", "prob": 0.31}, {"piece": " green", "prob": 0.10}], []]
    rid = store.record(
        source="engine_chat", client="studio", model="clozn-qwen",
        messages=[{"role": "user", "content": "what color is the sky?"}],
        response="The sky is blue.",
        trace={"tokens": tokens, "confidence": confidence, "alternatives": alternatives},
        memory={"cards_applied": ["Keep it brief.", "Be nice"], "applied_ids": ["c1", "c2"],
                "relevance": [0.81, None], "gate": 0.77, "mode": "prompt"},
        behavior={"active_dials": {"concise": 0.5, "warm": -0.2}},
        finish_reason="length",
    )
    run = store.get_run(rid)

    events = run_timeline.timeline(run)
    types = [e["type"] for e in events]
    assert types == ["run_started", "memory_applied", "dials_applied", "generation",
                      "hesitation", "hesitation", "finished"]

    started = events[0]
    assert started["source"] == "engine_chat" and started["client"] == "studio"
    assert started["model"] == "clozn-qwen" and started["at"] == run["created_at"]
    assert started["label"] == "Run started"

    mem_ev = events[1]
    assert mem_ev["label"] == "2 memory cards applied"        # pluralized (N != 1)
    assert mem_ev["count"] == 2
    assert mem_ev["gate"] == 0.77 and mem_ev["mode"] == "prompt"
    assert mem_ev["cards"] == [
        {"text": "Keep it brief.", "id": "c1", "relevance": 0.81},
        {"text": "Be nice", "id": "c2", "relevance": None},
    ]

    dial_ev = events[2]
    assert dial_ev["label"] == "2 behavior dials"              # pluralized
    assert dial_ev["dials"] == {"concise": 0.5, "warm": -0.2}

    gen_ev = events[3]
    assert gen_ev["label"] == "Generated 5 tokens"
    assert gen_ev["n_tokens"] == 5
    assert gen_ev["duration_ms"] == run["timing"]["duration_ms"]

    h1, h2 = events[4], events[5]
    assert h1["index"] == 1 and h1["token"] == " sky" and h1["confidence"] == 0.30
    assert h1["prob"] == 0.30 and h1["logprob"] == -1.203973
    assert h1["alternatives"] == [{"piece": " sea", "text": " sea", "prob": 0.22, "logprob": -1.514128}]
    assert h1["label"] == 'Unsure at " sky"'
    assert h2["index"] == 3 and h2["token"] == " blue" and h2["confidence"] == 0.41
    assert h2["alternatives"] == [{"piece": " grey", "text": " grey", "prob": 0.31, "logprob": -1.171183},
                                  {"piece": " green", "text": " green", "prob": 0.10, "logprob": -2.302585}]
    # the two mid-confidence tokens (0.92, 0.95, 0.99) never cross LOW_CONF -- no hesitation for them
    assert {h1["index"], h2["index"]} == {1, 3}

    finished = events[6]
    assert finished == {"type": "finished", "label": "Finished (length)",
                        "finish_reason": "length", "truncated": True}


def test_hesitation_events_carry_token_id_logprob_and_topk_entropy_when_present(store):
    """Backlog #2: the new v2 parallel arrays (token_ids/logprobs/topk_entropy -- see runlog.py's
    TRACE_KEYS docstring) must ride the hesitation event alongside confidence/alternatives, when the run's
    trace actually carries them -- even when the trace has no `steps` list of its own (record() derives
    one from the parallel arrays, topk_entropy included)."""
    rid = store.record(
        source="engine_chat", client="studio", model="clozn-qwen",
        messages=[{"role": "user", "content": "what color is the sky?"}],
        response="The sky",
        trace={"tokens": ["The", " sky"], "confidence": [0.95, 0.30],
              "token_ids": [101, 202], "topk_entropy": [None, 0.87]},
    )
    run = store.get_run(rid)
    events = run_timeline.timeline(run)
    hesitation = next(e for e in events if e["type"] == "hesitation")
    assert hesitation["token_id"] == 202
    assert hesitation["logprob"] is not None            # derived from confidence 0.30, never fabricated
    assert hesitation["topk_entropy"] == 0.87


def test_hesitation_events_omit_new_fields_on_a_legacy_trace(store):
    """A trace shaped exactly like every run persisted before v2 (tokens/confidence/alternatives only)
    must still produce a hesitation event -- just without token_id/topk_entropy, never a KeyError."""
    rid = store.record(
        source="cli", messages=[{"role": "user", "content": "q"}], response="um",
        trace={"tokens": ["um"], "confidence": [0.1], "alternatives": [[]]},
    )
    events = run_timeline.timeline(store.get_run(rid))
    hesitation = next(e for e in events if e["type"] == "hesitation")
    assert "token_id" not in hesitation and "topk_entropy" not in hesitation
    # runlog.record() itself always derives a per-step logprob from confidence -- that part of the v2
    # schema isn't optional infrastructure, it's math on a number the run already had.
    assert hesitation.get("logprob") is not None


def test_singular_card_and_dial_labels_are_not_pluralized(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a",
                       memory={"cards_applied": ["only one"], "mode": "prompt"},
                       behavior={"active_dials": {"warm": 0.2}})
    events = run_timeline.timeline(store.get_run(rid))
    mem_ev = next(e for e in events if e["type"] == "memory_applied")
    dial_ev = next(e for e in events if e["type"] == "dials_applied")
    assert mem_ev["label"] == "1 memory card applied"
    assert mem_ev["count"] == 1
    assert dial_ev["label"] == "1 behavior dial"


# ---------------------------------------------------------------------------------------- fixture: minimal run

def test_minimal_run_has_started_generation_and_an_honest_finished_with_no_reason(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "hi"}],
                       response="hey there friend")
    events = run_timeline.timeline(store.get_run(rid))
    types = [e["type"] for e in events]
    assert types == ["run_started", "generation", "finished"]
    gen = events[1]
    assert gen["n_tokens"] == 3                   # word count of the response -- no trace to count instead
    assert gen["label"] == "Generated 3 tokens"
    assert events[2] == {"type": "finished", "label": "Finished", "finish_reason": None, "truncated": False}


# ---------------------------------------------------------------------------------------- fixture: errored run

def test_errored_run_has_an_error_event_and_no_finished(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="", error="boom")
    events = run_timeline.timeline(store.get_run(rid))
    types = [e["type"] for e in events]
    assert types == ["run_started", "error"]       # no response/trace -> no generation event either
    assert events[1] == {"type": "error", "label": "Error", "message": "boom"}
    assert "finished" not in types


def test_errored_run_with_a_trace_still_skips_finished(store):
    """An error can fire after some tokens were already generated -- hesitations/generation still show,
    but `finished` must never appear alongside `error` (mutually exclusive stop states)."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="uh",
                       trace={"tokens": ["uh"], "confidence": [0.2]}, error="model crashed mid-stream")
    events = run_timeline.timeline(store.get_run(rid))
    types = [e["type"] for e in events]
    assert "generation" in types and "hesitation" in types
    assert "finished" not in types
    assert types[-1] == "error"


# --------------------------------------------------------------------------------------------- fixture: replay

def test_branched_run_reports_its_parent_right_after_run_started(store):
    parent_rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    child_rid = store.record(source="replay", client="studio",
                             messages=[{"role": "user", "content": "q"}], response="b",
                             parent_run_id=parent_rid)
    events = run_timeline.timeline(store.get_run(child_rid))
    assert events[0]["type"] == "run_started"
    assert events[1] == {"type": "branched_from", "label": "Branched from an earlier run",
                         "parent_run_id": parent_rid}


def test_a_run_with_no_parent_has_no_branched_from_event(store):
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="a")
    types = [e["type"] for e in run_timeline.timeline(store.get_run(rid))]
    assert "branched_from" not in types


# ------------------------------------------------------------------------------------------ fixture: no-trace

def test_no_trace_run_has_no_hesitations_and_a_word_count_not_a_fabricated_token_count(store):
    rid = store.record(source="studio_chat", messages=[{"role": "user", "content": "hi"}],
                       response="hey there friend",
                       memory={"cards_applied": ["be concise"], "gate": 0.9, "mode": "prompt"},
                       behavior={"active_dials": {"concise": 0.4}})
    run = store.get_run(rid)
    assert run["trace"] == {}                                     # confirms runlog really stored nothing

    events = run_timeline.timeline(run)
    types = [e["type"] for e in events]
    assert "hesitation" not in types
    gen = next(e for e in events if e["type"] == "generation")
    assert gen["n_tokens"] == 3                # word-count fallback, never a fabricated trace token count
    mem_ev = next(e for e in events if e["type"] == "memory_applied")
    assert mem_ev["cards"][0]["id"] is None                       # no applied_ids logged -> None, no guess
    assert mem_ev["cards"][0]["relevance"] is None


def test_no_response_and_no_trace_means_no_generation_event(store):
    """A run that produced literally nothing generative (no response, no trace) must not fabricate a
    'Generated 0 tokens' event out of thin air."""
    rid = store.record(source="cli", messages=[{"role": "user", "content": "q"}], response="")
    types = [e["type"] for e in run_timeline.timeline(store.get_run(rid))]
    assert "generation" not in types


# ------------------------------------------------------------------------------------- fixture: empty/non-dict

@pytest.mark.parametrize("garbage", [None, "not a dict", 42, [], ["also", "not", "a", "dict"], {}])
def test_non_dict_or_empty_run_returns_empty_list(garbage):
    assert run_timeline.timeline(garbage) == []


def test_never_raises_on_a_maximally_malformed_but_dict_shaped_run():
    """Every sub-field is the WRONG type (string where a list is expected, list where a dict is expected,
    mismatched-length lists, ...) -- timeline() must degrade field-by-field, never raise."""
    run = {
        "id": "run_weird",
        "trace": {"tokens": ["a", "b", "c"], "confidence": "not-a-list", "alternatives": {"nope": True}},
        "memory": {"cards_applied": ["x", "y"], "applied_ids": "not-a-list", "relevance": "not-a-list",
                  "gate": "n/a", "mode": 123},
        "behavior": {"active_dials": ["not", "a", "dict"]},
        "timing": "not-a-dict either",
        "response": 12345,
    }
    events = run_timeline.timeline(run)                 # must not raise
    types = [e["type"] for e in events]
    assert types[0] == "run_started"
    # tokens present but confidence unusable -> no token ever clears the LOW_CONF check -> zero hesitations
    assert "hesitation" not in types
    mem_ev = next(e for e in events if e["type"] == "memory_applied")
    assert len(mem_ev["cards"]) == 2
    assert all(c["id"] is None and c["relevance"] is None for c in mem_ev["cards"])
    assert "dials_applied" not in types                 # a list isn't a dial dict -> no dials, no crash
    gen_ev = next(e for e in events if e["type"] == "generation")
    assert gen_ev["n_tokens"] == 3                      # tokens list still usable even though confidence isn't
    assert gen_ev["duration_ms"] is None                # timing wasn't a dict -> absent, not a crash
    assert types[-1] == "finished"                      # no error key -> an honest finished with no reason


def test_timeline_never_raises_on_a_run_missing_every_optional_field():
    """A dict with only an id -- the sparsest possible non-empty run -- still degrades to a clean
    run_started + finished pair rather than raising or returning something half-built."""
    events = run_timeline.timeline({"id": "run_bare"})
    assert [e["type"] for e in events] == ["run_started", "finished"]
    assert events[0] == {"type": "run_started", "label": "Run started", "source": None,
                         "client": None, "model": None, "at": None}
    assert events[1] == {"type": "finished", "label": "Finished", "finish_reason": None, "truncated": False}
