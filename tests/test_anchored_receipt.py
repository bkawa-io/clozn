"""test_anchored_receipt.py -- clozn/receipts/anchored_receipt.py: the causal receipt for anchored
memory ("verify with a causal receipt" for the fit -> whatlearned -> recall -> delete_term demo).

Model-free and GPU-free (mirrors tests/test_swap_receipt.py's own style, this module's template): the
engine is a FakeEngineClient (no real clozn-server, no socket) wired through a FakeSub exposing
`.engine` -- the same duck-typed shape clozn.server.app.EngineSubstrate presents. Anchored bags are
hand-built dicts written straight into an isolated `anchored.BAGS_PATH` (no fit_bag/DirProvider needed --
the receipt only ever READS the store; fitting is covered by tests/test_anchored_routes.py).
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.memory.anchored as anchored               # noqa: E402
import clozn.receipts.anchored_receipt as ar            # noqa: E402


# ==================================================================================== fake engine + substrate

class FakeEngineClient:
    """Mirrors test_swap_receipt.py's FakeEngineClient pattern exactly: `.score()` serves TWO distinct
    call shapes (token RESOLUTION via `continuation=`, and the quantitative logprob measure via
    `continuation_ids=`)."""

    def __init__(self, *, vocab=None, unresolvable=(), apply_template_fails=False, complete_fails=False,
                 intervene_fails_on=(), score_fails=False,
                 baseline_text="the weather today is calm and mild",
                 anchored_text="kyoto kyoto gardens in kyoto are lovely in autumn",
                 null_text="xk garble zzzz repeated repeated repeated repeated",
                 score_values=(-2.0, -0.3, -1.9)):
        self.vocab = dict(vocab or {})
        self.unresolvable = set(unresolvable)
        self.apply_template_fails = apply_template_fails
        self.complete_fails = complete_fails
        self.intervene_fails_on = set(intervene_fails_on)
        self.score_fails = score_fails
        self.baseline_text, self.anchored_text, self.null_text = baseline_text, anchored_text, null_text
        self.score_values = list(score_values)
        self.intervene_calls = []
        self.score_calls = []
        self.complete_calls = []
        self._next_id = 1000

    def apply_template(self, messages, add_assistant=True):
        if self.apply_template_fails:
            raise RuntimeError("no embedded chat template")
        return "PROMPT::" + " | ".join(str(m.get("content", "")) for m in messages)

    def complete(self, prompt, max_tokens=64):
        self.complete_calls.append(prompt)
        if self.complete_fails:
            raise RuntimeError("engine unreachable")
        return {"choices": [{"text": self.baseline_text}]}

    def intervene(self, prompt, vector=None, coef=None, layer=None, max_tokens=64):
        which = "anchored" if len(self.intervene_calls) == 0 else "null"
        self.intervene_calls.append({"vector": vector, "coef": coef, "layer": layer, "which": which})
        if which in self.intervene_fails_on:
            raise RuntimeError(f"{which} generation failed")
        return {"choices": [{"text": self.anchored_text if which == "anchored" else self.null_text}]}

    def score(self, prompt=None, continuation_ids=None, continuation=None, topk=0, steer=None, steer_vec=None):
        self.score_calls.append({"continuation": continuation, "continuation_ids": continuation_ids,
                                 "steer_vec": steer_vec, "steer": steer})
        if self.score_fails:
            raise RuntimeError("score unreachable")
        if continuation is not None:                     # token-resolution path
            word = continuation.strip()
            if word in self.unresolvable:
                return {"tokens": [{"id": 1, "piece": word[: len(word) // 2]},
                                   {"id": 2, "piece": word[len(word) // 2:]}]}
            ids = self.vocab.get(continuation)
            if ids is None:
                ids = [self._next_id]
                self._next_id += 1
                self.vocab[continuation] = ids
            return {"tokens": [{"id": i, "piece": continuation} for i in ids]}
        tid = continuation_ids[0] if continuation_ids else 0
        if steer_vec is None:
            lp = self.score_values[0]
        else:
            n_steered = sum(1 for c in self.score_calls if c["steer_vec"] is not None)
            lp = self.score_values[1] if n_steered == 1 else self.score_values[2]
        return {"tokens": [{"id": tid, "piece": "x", "logprob": lp}]}


class FakeSub:
    def __init__(self, engine):
        self.engine = engine


class FakeSubNoEngine:
    pass


RUN = {"id": "run_1", "messages": [{"role": "user", "content": "Tell me about your day."}],
      "response": "It was fine.", "behavior": {"active_dials": {}}, "trace": {}}


# ==================================================================================== bag fixtures

def _unit(i: int, n: int = 4) -> list:
    v = [0.0] * n
    v[i] = 1.0
    return v


def _make_bag(card_id="card_1", terms=None, vector=None, on=True,
             card_text="likes kyoto gardens and tea"):
    if terms is None:
        terms = [{"token": "kyoto", "alpha": 0.62}, {"token": "gardens", "alpha": 0.31},
                 {"token": "tea", "alpha": 0.18}]
    if vector is None:
        vector = _unit(0)
    return {
        "card_id": card_id, "card_text": card_text, "terms": terms,
        "k": len(terms), "k_requested": len(terms), "reconstruction_cos": 0.87, "residual_norm": 0.12,
        "vector": vector, "candidate_bank": [t["token"] for t in terms],
        "layer": anchored.LAYER, "scale": anchored.SCALE, "on": on, "envelope": anchored.ENVELOPE,
        "fitted_at": "2026-01-01T00:00:00", "lens_manifest_hash": None,
    }


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(anchored, "BAGS_PATH", str(tmp_path / "bags.json"))
    return tmp_path


# ==================================================================================== graceful degrade paths

def test_no_run_is_graceful_not_a_raise():
    out = ar.anchored_receipt(None, None, FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert out["mode"] == "anchored_receipt"
    assert out["blocked"] is None
    assert "no run" in out["note"]


def test_run_not_a_dict_is_graceful():
    out = ar.anchored_receipt(["not", "a", "dict"], None, FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False


def test_no_engine_on_substrate_is_blocked_cleanly(iso):
    anchored.put_bag(_make_bag())
    out = ar.anchored_receipt(RUN, "card_1", FakeSubNoEngine())
    assert out["causal_verified"] is False
    assert out["blocked"] == "no_engine"


def test_card_id_with_no_stored_bag_is_blocked_no_bag(iso):
    out = ar.anchored_receipt(RUN, "does_not_exist", FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert out["blocked"] == "no_bag"


def test_no_card_id_and_no_active_bags_is_blocked_no_bag(iso):
    out = ar.anchored_receipt(RUN, None, FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert out["blocked"] == "no_bag"


def test_explicit_card_id_toggled_off_is_blocked_no_bag(iso):
    """compile_steer() itself skips a bag with on=False -- requesting that exact card_id honestly
    reports 'nothing composes', not a fabricated injection."""
    anchored.put_bag(_make_bag(on=False))
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert out["blocked"] == "no_bag"


def test_bag_with_no_terms_is_blocked_no_bag(iso):
    bag = _make_bag(terms=[])
    anchored.put_bag(bag)
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert out["blocked"] == "no_bag"


def test_token_resolution_failure_blocks_before_any_generation(iso):
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient(unresolvable={"kyoto"})
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))
    assert out["causal_verified"] is False
    assert out["blocked"] == "token_resolution"
    assert out["baseline_reply"] is None       # never got to generation
    assert ec.complete_calls == []
    assert ec.intervene_calls == []
    # the composition itself is still recorded -- the WRITE succeeded, only the auxiliary measure's
    # token lookup failed.
    assert out["injected"]["coef"] is not None


def test_run_with_no_messages_is_graceful(iso):
    anchored.put_bag(_make_bag())
    out = ar.anchored_receipt({"id": "r2", "messages": []}, "card_1", FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert out["blocked"] is None
    assert "no messages" in out["note"]


def test_apply_template_failure_is_graceful_not_blocked(iso):
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient(apply_template_fails=True)
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))
    assert out["causal_verified"] is False
    assert out["blocked"] is None
    assert "could not render" in out["note"]


# ==================================================================================== generation failures

def test_baseline_generation_failure_is_blocked_cleanly(iso):
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient(complete_fails=True)
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))
    assert out["causal_verified"] is False
    assert out["blocked"] == "generation_failed"
    assert out["injected"]["target_token_id"] is not None   # already resolved before the failure


def test_anchored_generation_failure_is_blocked_after_baseline_succeeds(iso):
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient(intervene_fails_on={"anchored"})
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))
    assert out["causal_verified"] is False
    assert out["blocked"] == "generation_failed"
    assert out["baseline_reply"] == ec.baseline_text


def test_null_generation_failure_degrades_only_the_null_control(iso):
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient(intervene_fails_on={"null"})
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))
    assert out["causal_verified"] is True              # the receipt itself still completes
    assert out["null_control_available"] is False
    assert out["null_reply"] is None
    assert out["lexicon_hits"]["null"] is None


# ==================================================================================== happy path end to end

def test_happy_path_end_to_end_single_card(iso):
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient()
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))

    assert out["causal_verified"] is True
    assert out["blocked"] is None
    assert out["run_id"] == "run_1"
    assert out["card_id"] == "card_1"

    # injected metadata: the composed bag's own coef/layer/s_total + the picked target term.
    assert out["injected"]["layer"] == anchored.LAYER
    assert out["injected"]["coef"] == pytest.approx(anchored.SCALE * anchored.BASE_NORM)
    assert out["injected"]["target_term"] == "kyoto"      # highest |alpha| term
    assert out["injected"]["target_token_id"] == ec.vocab[" kyoto"][0]
    assert out["injected"]["bags"][0]["card_id"] == "card_1"

    # WHATLEARNED: the alpha table for exactly the bag(s) injected -- a pure lookup, never a generation.
    assert out["whatlearned"]["bags"][0]["card_id"] == "card_1"
    assert "kyoto" in out["whatlearned"]["bags"][0]["table"]
    assert out["whatlearned"]["note"] == anchored.WHATLEARNED_NOTE

    # generations
    assert out["baseline_reply"] == ec.baseline_text
    assert out["anchored_reply"] == ec.anchored_text
    assert out["null_reply"] == ec.null_text
    assert out["null_control_available"] is True

    # measure A: lexicon hits ("kyoto" appears 3x in anchored_text, 0x elsewhere)
    assert out["lexicon_hits"] == {"baseline": 0, "anchored": 3, "null": 0}

    # measure B: quantitative shift
    assert out["logprob_shift"]["baseline"] == -2.0
    assert out["logprob_shift"]["anchored"] == -0.3
    assert out["logprob_shift"]["null"] == -1.9
    assert out["logprob_shift"]["anchored_over_baseline_nat"] == pytest.approx(1.7)
    assert out["logprob_shift"]["anchored_over_null_nat"] == pytest.approx(1.6)

    assert out["has_effect"] is True
    assert out["targeted_shift"] is True
    assert out["coherent"] is True    # "kyoto" repeats 3x among 9 words -- fluent, not degenerate looping
    assert isinstance(out["coherence_score"], float)
    assert out["null_note"] == ar._NULL_NOTE
    assert out["lexicon_note"] == ar._LEXICON_CAVEAT

    # the engine calls carried the UNIT-ish composed vector + a separate coef, exactly like swap_receipt.
    anchored_call = ec.intervene_calls[0]
    assert anchored_call["which"] == "anchored"
    assert anchored_call["coef"] == pytest.approx(anchored.SCALE * anchored.BASE_NORM)
    null_call = ec.intervene_calls[1]
    assert null_call["which"] == "null"
    assert null_call["vector"] != anchored_call["vector"]
    assert null_call["coef"] == anchored_call["coef"]


def test_happy_path_active_bags_composes_every_card_without_a_card_id(iso):
    anchored.put_bag(_make_bag(card_id="card_1", terms=[{"token": "kyoto", "alpha": 0.9}], vector=_unit(0)))
    anchored.put_bag(_make_bag(card_id="card_2", terms=[{"token": "sushi", "alpha": 0.4}], vector=_unit(1),
                               card_text="likes sushi"))
    ec = FakeEngineClient()
    out = ar.anchored_receipt(RUN, None, FakeSub(ec))

    assert out["causal_verified"] is True
    assert out["card_id"] is None
    assert {b["card_id"] for b in out["injected"]["bags"]} == {"card_1", "card_2"}
    assert out["injected"]["target_term"] == "kyoto"   # the single highest |alpha| across both bags
    assert len(out["whatlearned"]["bags"]) == 2


def test_has_effect_is_false_when_anchored_does_not_beat_the_null(iso):
    """Honesty: a text/logprob shift that the null ALSO produces must not be claimed as an effect."""
    anchored.put_bag(_make_bag())
    ec = FakeEngineClient(anchored_text="a plain unrelated sentence about weather",
                          null_text="a plain unrelated sentence about weather",
                          score_values=(-2.0, -0.3, -0.3))   # anchored == null quantitatively too
    out = ar.anchored_receipt(RUN, "card_1", FakeSub(ec))
    assert out["causal_verified"] is True
    assert out["has_effect"] is False
    assert out["targeted_shift"] is False


def test_null_vector_is_deterministic_for_the_same_run_and_card(iso):
    anchored.put_bag(_make_bag())
    ec1 = FakeEngineClient()
    ec2 = FakeEngineClient()
    out1 = ar.anchored_receipt(RUN, "card_1", FakeSub(ec1))
    out2 = ar.anchored_receipt(RUN, "card_1", FakeSub(ec2))
    assert out1["causal_verified"] is True and out2["causal_verified"] is True
    assert ec1.intervene_calls[1]["vector"] == ec2.intervene_calls[1]["vector"]


def test_never_raises_on_a_totally_malformed_run(iso):
    anchored.put_bag(_make_bag())
    out = ar.anchored_receipt(object(), "card_1", FakeSub(FakeEngineClient()))
    assert isinstance(out, dict)
    assert out["causal_verified"] is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
