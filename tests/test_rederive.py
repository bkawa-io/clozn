"""test_rederive -- clozn/rederive.py: deterministic
teacher-forced re-derivation of a stored run's exact answer, built entirely from the run record and
`sub.score_tokens` -- NO generation anywhere.

Model-free throughout: a FakeScoreSub stands in for EngineSubstrate, exposing only `.score_tokens` (the
seam rederive.py duck-types against, mirroring test_engine_score.py's own conventions) -- no C++ engine,
no GPU, no real socket.

Covers:
  * with_arm_conditions(run) -- the assembled/raw message split (assembled_messages preferred when
    present, block=None there since it's already folded in; raw_messages/raw_block ALWAYS the un-folded
    pair, for receipts.py's arm-swapping), the trace-token-ids-vs-retokenize-fallback decision, and dials.
  * score_arm(sub, conditions, ...) -- continuation-ids-primary / continuation-text-fallback dispatch,
    kwargs passthrough, and the "never raise, return ([], False)" failure contract.
  * rederive(run, sub) -- the S3 deliverable: {"text","steps","meta"} assembled from score_tokens's
    token list, or None on any failure.
"""
from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.receipts.rederive as rederive  # noqa: E402


# ==================================================================================== fakes

class FakeScoreSub:
    """Exposes exactly `.score_tokens` -- the only surface rederive.py touches on a substrate."""

    def __init__(self, tokens=None, raises=False):
        self.calls = []
        self._tokens = tokens if tokens is not None else []
        self._raises = raises

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        if self._raises:
            raise RuntimeError("boom")
        self.calls.append({"messages": messages, "continuation_ids": continuation_ids,
                           "continuation": continuation, "block": block,
                           "steer_strengths": steer_strengths, "steer_vec": steer_vec, "topk": topk})
        return self._tokens


TOKENS = [{"id": 11, "piece": "Hello", "logprob": -0.1},
          {"id": 22, "piece": " there", "logprob": -0.2}]


# ==================================================================================== with_arm_conditions

def test_with_arm_conditions_prefers_assembled_messages_and_omits_block():
    run = {
        "messages": [{"role": "user", "content": "hi"}],
        "assembled_messages": [{"role": "system", "content": "BLOCK"},
                               {"role": "user", "content": "hi"}],
        "memory": {"prompt_block": "BLOCK"},
        "response": "hey",
    }
    c = rederive.with_arm_conditions(run)
    assert c["messages"] == run["assembled_messages"]
    assert c["block"] is None                          # already folded in -- do not re-inject
    assert c["block_source"] == "assembled_messages"
    assert c["raw_messages"] == run["messages"]         # ALWAYS the un-folded pair
    assert c["raw_block"] == "BLOCK"


def test_with_arm_conditions_falls_back_to_messages_plus_prompt_block():
    run = {"messages": [{"role": "user", "content": "hi"}], "memory": {"prompt_block": "BLOCK"},
          "response": "hey"}
    c = rederive.with_arm_conditions(run)
    assert c["messages"] == run["messages"]
    assert c["block"] == "BLOCK"
    assert c["block_source"] == "prompt_block"
    assert c["raw_messages"] == run["messages"]
    assert c["raw_block"] == "BLOCK"


def test_with_arm_conditions_no_block_at_all():
    run = {"messages": [{"role": "user", "content": "hi"}], "response": "hey"}
    c = rederive.with_arm_conditions(run)
    assert c["block"] is None
    assert c["block_source"] == "none"
    assert c["raw_block"] is None



def test_with_arm_conditions_continuation_ids_from_v1_token_ids():
    run = {"messages": [], "response": "x", "trace": {"token_ids": [11, 22]}}
    c = rederive.with_arm_conditions(run)
    assert c["continuation_ids"] == [11, 22]
    assert c["retokenized"] is False


def test_with_arm_conditions_falls_back_to_v2_steps_token_id():
    run = {"messages": [], "response": "x",
          "trace": {"steps": [{"piece": "a", "token_id": 5}, {"piece": "b", "token_id": 6}]}}
    c = rederive.with_arm_conditions(run)
    assert c["continuation_ids"] == [5, 6]
    assert c["retokenized"] is False


def test_with_arm_conditions_retokenizes_when_no_ids_anywhere():
    run = {"messages": [], "response": "hello", "trace": {"tokens": ["hello"], "confidence": [0.9]}}
    c = rederive.with_arm_conditions(run)
    assert c["continuation_ids"] is None
    assert c["retokenized"] is True
    assert c["response"] == "hello"


def test_with_arm_conditions_a_partial_id_list_falls_back_to_retokenize():
    """One position missing a real id -> the WHOLE list is untrustworthy (never patch with a fabricated
    id, never feed score_tokens a None it would choke on)."""
    run = {"messages": [], "response": "x", "trace": {"token_ids": [11, None, 22]}}
    c = rederive.with_arm_conditions(run)
    assert c["continuation_ids"] is None
    assert c["retokenized"] is True


# ==================================================================================== score_arm

def test_score_arm_returns_false_when_substrate_has_no_score_tokens():
    class NoScore:
        pass
    tokens, ok = rederive.score_arm(NoScore(), {"messages": [], "continuation_ids": [1]})
    assert (tokens, ok) == ([], False)


def test_score_arm_uses_continuation_ids_when_present():
    sub = FakeScoreSub(tokens=TOKENS)
    conditions = {"messages": [{"role": "user", "content": "hi"}], "continuation_ids": [11, 22],
                 "response": "Hello there"}
    tokens, ok = rederive.score_arm(sub, conditions, block="B", steer_strengths={"warm": 1.0})
    assert ok is True
    assert tokens == TOKENS
    assert sub.calls[-1]["continuation_ids"] == [11, 22]
    assert sub.calls[-1]["block"] == "B"
    assert sub.calls[-1]["steer_strengths"] == {"warm": 1.0}
    assert "continuation" not in sub.calls[-1] or sub.calls[-1]["continuation"] is None


def test_score_arm_falls_back_to_continuation_text_when_ids_are_none():
    sub = FakeScoreSub(tokens=TOKENS)
    conditions = {"messages": [], "continuation_ids": None, "response": "Hello there"}
    tokens, ok = rederive.score_arm(sub, conditions, block=None)
    assert ok is True
    assert sub.calls[-1]["continuation_ids"] is None
    assert sub.calls[-1]["continuation"] == "Hello there"


def test_score_arm_fails_cleanly_when_ids_none_and_response_empty():
    sub = FakeScoreSub(tokens=TOKENS)
    conditions = {"messages": [], "continuation_ids": None, "response": ""}
    assert rederive.score_arm(sub, conditions) == ([], False)
    assert sub.calls == []                              # never even called score_tokens


def test_score_arm_messages_override_takes_precedence_over_conditions():
    sub = FakeScoreSub(tokens=TOKENS)
    conditions = {"messages": ["from-conditions"], "continuation_ids": [1]}
    rederive.score_arm(sub, conditions, messages=["override"])
    assert sub.calls[-1]["messages"] == ["override"]


def test_score_arm_passes_steer_vec_through():
    sub = FakeScoreSub(tokens=TOKENS)
    conditions = {"messages": [], "continuation_ids": [1]}
    rederive.score_arm(sub, conditions, steer_vec=[0.1, 0.2])
    assert sub.calls[-1]["steer_vec"] == [0.1, 0.2]


def test_score_arm_never_raises_on_a_scoring_exception():
    sub = FakeScoreSub(raises=True)
    conditions = {"messages": [], "continuation_ids": [1]}
    assert rederive.score_arm(sub, conditions) == ([], False)


def test_score_arm_tolerates_a_non_list_reply():
    class WeirdSub:
        def score_tokens(self, *a, **k):
            return {"not": "a list"}
    tokens, ok = rederive.score_arm(WeirdSub(), {"messages": [], "continuation_ids": [1]})
    assert tokens == [] and ok is True                  # scoring "succeeded" but returned nothing usable


# ==================================================================================== rederive()

RUN = {
    "id": "run_1", "messages": [{"role": "user", "content": "hi"}],
    "assembled_messages": [{"role": "user", "content": "hi"}],
    "response": "Hello there",
    "behavior": {"active_dials": {"warm": 0.5}},
    "trace": {"token_ids": [11, 22]},
}


def test_rederive_returns_none_for_bad_run():
    assert rederive.rederive(None, FakeScoreSub()) is None
    assert rederive.rederive({}, FakeScoreSub()) is None
    assert rederive.rederive("not a dict", FakeScoreSub()) is None


def test_rederive_returns_none_when_substrate_cannot_score():
    class NoScore:
        pass
    assert rederive.rederive(RUN, NoScore()) is None


def test_rederive_happy_path_builds_text_steps_and_meta():
    sub = FakeScoreSub(tokens=TOKENS)
    out = rederive.rederive(RUN, sub)
    assert out is not None
    assert out["text"] == "Hello there"
    assert out["steps"] == [
        {"piece": "Hello", "token_id": 11, "logprob": -0.1, "conf": math.exp(-0.1)},
        {"piece": " there", "token_id": 22, "logprob": -0.2, "conf": math.exp(-0.2)},
    ]
    assert out["meta"]["retokenized"] is False
    assert out["meta"]["n_tokens"] == 2
    # the WITH arm scores under the run's own recorded conditions -- same messages/ids/dials
    assert sub.calls[-1]["continuation_ids"] == [11, 22]


def test_rederive_flags_retokenized_when_trace_lacks_ids():
    run = {"id": "run_2", "messages": [{"role": "user", "content": "hi"}], "response": "Hello there"}
    sub = FakeScoreSub(tokens=TOKENS)
    out = rederive.rederive(run, sub)
    assert out is not None
    assert out["meta"]["retokenized"] is True
    assert sub.calls[-1]["continuation_ids"] is None
    assert sub.calls[-1]["continuation"] == "Hello there"


def test_rederive_missing_logprob_yields_none_confidence():
    sub = FakeScoreSub(tokens=[{"id": 1, "piece": "x"}])          # no logprob key at all
    out = rederive.rederive(RUN, sub)
    assert out["steps"][0]["conf"] is None
    assert out["steps"][0]["logprob"] is None


def test_rederive_returns_none_when_scoring_yields_no_tokens():
    sub = FakeScoreSub(tokens=[])
    assert rederive.rederive(RUN, sub) is None


def test_rederive_is_deterministic_across_repeated_calls():
    sub = FakeScoreSub(tokens=TOKENS)
    a = rederive.rederive(RUN, sub)
    b = rederive.rederive(RUN, sub)
    assert a == b
