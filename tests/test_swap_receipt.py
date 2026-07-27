"""test_swap_receipt.py -- clozn/receipts/swap_receipt.py (SWAP-RECEIPTS): read a run's disposition,
inject dir(to_concept) at L21 during regeneration, diff vs
baseline + a random-equal-norm null.

Model-free and GPU-free throughout (a GPU experiment is running elsewhere; this suite boots no
engine): the J-lens sidecar + unembed export are tiny on-disk FIXTURES this suite writes itself
(same construction as test_concept_dir.py's -- an orthogonal J_l + orthonormal W_U rows, so
dir(c) is exact, not merely probable), and the engine itself is a FakeEngineClient (mirrors
test_engine_add_custom.py's _FakeEC pattern) wired through a FakeSub exposing `.engine` + `.jlens`
-- the same duck-typed shape clozn.server.app.EngineSubstrate presents.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.behavior.steering.concept_dir as concept_dir  # noqa: E402
import clozn.receipts.swap_receipt as sr                    # noqa: E402


# ==================================================================================== fixtures (mirror test_concept_dir.py)

def _orthogonal(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return q


def _write_jlens_fixture(tmp_path, *, d_model=32, layers=(21,), seed=1):
    jdir = tmp_path / "jlens"
    jdir.mkdir()
    manifest = {"model": "fixture", "d_model": d_model, "vocab": d_model, "layers": list(layers),
                "engine_default_tap_layer": layers[0]}
    (jdir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    for i, layer in enumerate(layers):
        J = _orthogonal(seed + i, d_model).astype(np.float32)
        J.astype("<f2").tofile(str(jdir / f"J_layer{layer}.f16"))
    return str(jdir)


def _write_unembed_fixture(tmp_path, *, d_model=32, vocab=32, seed=2):
    udir = tmp_path / "unembed"
    udir.mkdir()
    q = _orthogonal(seed, d_model)[:vocab].astype(np.float32)
    np.save(str(udir / "norm_weight.npy"), np.ones(d_model, dtype=np.float32))
    np.save(str(udir / "lm_head_weight.npy"), q)
    (udir / "unembed_meta.json").write_text(json.dumps({"rms_norm_eps": 1e-6}), encoding="utf-8")
    return str(udir)


def _make_source(tmp_path, *, with_unembed=True, d_model=32, vocab=32, layer=21):
    jdir = _write_jlens_fixture(tmp_path, d_model=d_model, layers=(layer,))
    udir = _write_unembed_fixture(tmp_path, d_model=d_model, vocab=vocab) if with_unembed else None
    return concept_dir.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)


# ==================================================================================== fake engine + substrate

class FakeEngineClient:
    """Mirrors test_engine_add_custom.py's _FakeEC pattern: no real clozn-server, no socket.
    `.score()` serves TWO distinct call shapes the real EngineClient.score also serves: token
    RESOLUTION (continuation= text, used by ConceptSteer.resolve_token_id) and the quantitative
    logprob measure (continuation_ids=, used by swap_receipt itself)."""

    def __init__(self, *, vocab=None, apply_template_fails=False, complete_fails=False,
                 intervene_fails_on=(), score_fails=False,
                 baseline_text="the sky is calm and blue today",
                 swap_text="a vast ocean ocean wave of deep ocean water",
                 null_text="xk garble zzzz repeated repeated repeated repeated",
                 score_values=(-2.0, -0.3, -1.9)):
        self.vocab = dict(vocab or {})
        self.apply_template_fails = apply_template_fails
        self.complete_fails = complete_fails
        self.intervene_fails_on = set(intervene_fails_on)
        self.score_fails = score_fails
        self.baseline_text, self.swap_text, self.null_text = baseline_text, swap_text, null_text
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
        which = "swap" if len(self.intervene_calls) == 0 else "null"
        self.intervene_calls.append({"vector": vector, "coef": coef, "layer": layer, "which": which})
        if which in self.intervene_fails_on:
            raise RuntimeError(f"{which} generation failed")
        return {"choices": [{"text": self.swap_text if which == "swap" else self.null_text}]}

    def score(self, prompt=None, continuation_ids=None, continuation=None, topk=0, steer=None, steer_vec=None):
        self.score_calls.append({"continuation": continuation, "continuation_ids": continuation_ids,
                                 "steer_vec": steer_vec, "steer": steer})
        if self.score_fails:
            raise RuntimeError("score unreachable")
        if continuation is not None:                     # token-resolution path
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
    def __init__(self, engine, jlens_result=None, jlens_raises=False):
        self.engine = engine
        self._jlens_result = (jlens_result if jlens_result is not None
                              else {"available": False, "reason": "no jlens sidecar loaded"})
        self._jlens_raises = jlens_raises

    def jlens(self, text, layer=None, topk=5):
        if self._jlens_raises:
            raise RuntimeError("jlens boom")
        return self._jlens_result


class FakeSubNoJlens:
    """A minimal substrate with no .jlens method at all."""

    def __init__(self, engine):
        self.engine = engine


class FakeSubNoEngine:
    pass


RUN = {"id": "run_1", "messages": [{"role": "user", "content": "Tell me a fact."}],
      "response": "Here is a fact.", "behavior": {"active_dials": {}}, "trace": {}}

JLENS_OK = {"available": True, "layer": 21, "n_tokens": 3, "tokens": ["a", "b", "c"],
           "readouts": [
               [{"id": 1, "piece": "cat", "score": 1.0}],
               [{"id": 2, "piece": "dog", "score": 1.0}],
               [{"id": 3, "piece": "king", "score": 5.0}, {"id": 4, "piece": "queen", "score": 4.0},
                {"id": 5, "piece": "royal", "score": 3.0}, {"id": 6, "piece": "throne", "score": 2.0},
                {"id": 7, "piece": "crown", "score": 1.0}],
           ]}


# ==================================================================================== graceful degrade paths

def test_no_run_is_graceful_not_a_raise():
    ec = FakeEngineClient()
    out = sr.swap_receipt(None, None, "ocean", FakeSub(ec))
    assert out["causal_verified"] is False
    assert out["mode"] == "swap_receipt"
    assert "no run" in out["note"]


def test_run_not_a_dict_is_graceful():
    out = sr.swap_receipt(["not", "a", "dict"], None, "ocean", FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False


def test_no_engine_on_substrate_is_blocked_cleanly():
    out = sr.swap_receipt(RUN, None, "ocean", FakeSubNoEngine())
    assert out["causal_verified"] is False
    assert out["blocked"] == "no_engine"


def test_run_with_no_messages_is_graceful():
    out = sr.swap_receipt({"id": "r2", "messages": []}, None, "ocean", FakeSub(FakeEngineClient()))
    assert out["causal_verified"] is False
    assert "no messages" in out["note"]


def test_apply_template_failure_is_blocked_cleanly():
    ec = FakeEngineClient(apply_template_fails=True)
    out = sr.swap_receipt(RUN, None, "ocean", FakeSub(ec))
    assert out["causal_verified"] is False
    assert out["blocked"] == "template_render"


def test_swap_receipt_never_raises_on_a_totally_malformed_run():
    out = sr.swap_receipt(object(), None, "ocean", FakeSub(FakeEngineClient()))
    assert isinstance(out, dict)
    assert out["causal_verified"] is False


# ==================================================================================== the BLOCKER: unembed unavailable

def test_unembed_unavailable_blocks_but_disposition_is_still_read(tmp_path, monkeypatch):
    monkeypatch.delenv("CLOZN_DIRC_UNEMBED_DIR", raising=False)
    source = _make_source(tmp_path, with_unembed=False)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    sub = FakeSub(ec, jlens_result=JLENS_OK)
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)

    out = sr.swap_receipt(RUN, "cats", "ocean", sub, concept_steer=steer)

    assert out["causal_verified"] is False
    assert out["blocked"] == "unembed_unavailable"
    # the READ already happened before the WRITE failed -- disposed is populated, not blank.
    assert out["disposed"]["jlens_available"] is True
    assert out["disposed"]["jlens_top1"] == "king"
    assert out["disposed"]["hint"] == "cats"
    assert out["baseline_reply"] is None   # never got this far


def test_token_resolution_failure_is_blocked_cleanly(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" antidisestablishmentarianism": [1, 2, 3]})
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, None, "antidisestablishmentarianism", FakeSub(ec), concept_steer=steer)
    assert out["causal_verified"] is False
    assert out["blocked"] == "token_resolution"


# ==================================================================================== generation failures

def test_baseline_generation_failure_is_blocked_cleanly(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]}, complete_fails=True)
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, None, "ocean", FakeSub(ec), concept_steer=steer)
    assert out["causal_verified"] is False
    assert out["blocked"] == "generation_failed"
    assert out["swapped_to"]["token_id"] == 3   # already resolved before the failure


def test_swap_generation_failure_is_blocked_after_baseline_succeeds(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]}, intervene_fails_on={"swap"})
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, None, "ocean", FakeSub(ec), concept_steer=steer)
    assert out["causal_verified"] is False
    assert out["blocked"] == "generation_failed"
    assert out["baseline_reply"] == ec.baseline_text   # baseline DID complete


def test_null_generation_failure_degrades_only_the_null_control(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]}, intervene_fails_on={"null"})
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, None, "ocean", FakeSub(ec), concept_steer=steer)
    assert out["causal_verified"] is True          # the receipt itself still completes
    assert out["null_control_available"] is False
    assert out["null_reply"] is None
    assert out["lexicon_hits"]["null"] is None


# ==================================================================================== happy path end to end

def test_happy_path_end_to_end(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    sub = FakeSub(ec, jlens_result=JLENS_OK)
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)

    out = sr.swap_receipt(RUN, "cats", "ocean", sub, concept_steer=steer, strength=0.4)

    assert out["causal_verified"] is True
    assert out["blocked"] is None
    assert out["run_id"] == "run_1"

    # disposition: both the hint and the independent J-lens read are present, never merged.
    assert out["disposed"]["hint"] == "cats"
    assert out["disposed"]["jlens_top1"] == "king"
    assert out["disposed"]["jlens_top5"] == ["king", "queen", "royal", "throne", "crown"]
    assert out["disposed"]["baseline_lean"] == "sky"   # first salient content word of baseline_text

    # swap metadata
    assert out["swapped_to"]["concept"] == "ocean"
    assert out["swapped_to"]["token_id"] == 3
    assert out["swapped_to"]["layer"] == 21
    assert out["swapped_to"]["coef"] == pytest.approx(0.4 * concept_dir.VALIDATED_MEDIAN_RESID_NORM[21])

    # generations
    assert out["baseline_reply"] == ec.baseline_text
    assert out["swapped_reply"] == ec.swap_text
    assert out["null_reply"] == ec.null_text
    assert out["null_control_available"] is True

    # measure A: lexicon hits ("ocean" appears 3x in swap_text, 0x elsewhere)
    assert out["lexicon_hits"] == {"baseline": 0, "swap": 3, "null": 0}

    # measure B: quantitative shift (swap beats both baseline and null by > 1 nat)
    assert out["logprob_shift"]["baseline"] == -2.0
    assert out["logprob_shift"]["swap"] == -0.3
    assert out["logprob_shift"]["null"] == -1.9
    assert out["logprob_shift"]["swap_over_baseline_nat"] == pytest.approx(1.7)
    assert out["logprob_shift"]["swap_over_null_nat"] == pytest.approx(1.6)

    assert out["targeted_shift"] is True
    assert out["coherent"] is True
    assert isinstance(out["coherence_score"], float)
    assert out["null_note"] == sr._NULL_NOTE
    assert out["lexicon_note"] == sr._LEXICON_CAVEAT

    # the engine calls actually carried the UNIT vector + a separate coef (server multiplies
    # coef*vector itself -- see engine/core/serve/routes_state.cpp's `coef * raw_vec[i]`).
    swap_call = ec.intervene_calls[0]
    assert swap_call["which"] == "swap"
    assert abs(float(np.linalg.norm(swap_call["vector"])) - 1.0) < 1e-4
    assert swap_call["coef"] == pytest.approx(0.4 * concept_dir.VALIDATED_MEDIAN_RESID_NORM[21])
    null_call = ec.intervene_calls[1]
    assert null_call["which"] == "null"
    assert null_call["vector"] != swap_call["vector"]        # a DIFFERENT random direction
    assert abs(float(np.linalg.norm(null_call["vector"])) - 1.0) < 1e-4   # but the SAME magnitude


def test_from_hint_equal_to_to_concept_is_flagged_as_a_no_op(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, "Ocean", "ocean", FakeSub(ec), concept_steer=steer)
    assert out["causal_verified"] is True
    assert "no-op" in out["note"]


def test_no_jlens_method_on_substrate_degrades_disposition_only(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, None, "ocean", FakeSubNoJlens(ec), concept_steer=steer)
    assert out["causal_verified"] is True
    assert out["disposed"]["jlens_available"] is False
    assert out["disposed"]["jlens_reason"] == "substrate has no .jlens method"


def test_jlens_raising_is_handled_gracefully(tmp_path):
    source = _make_source(tmp_path)
    ec = FakeEngineClient(vocab={" ocean": [3]})
    sub = FakeSub(ec, jlens_raises=True)
    steer = concept_dir.ConceptSteer(ec, source=source, layer=21)
    out = sr.swap_receipt(RUN, None, "ocean", sub, concept_steer=steer)
    assert out["causal_verified"] is True
    assert out["disposed"]["jlens_available"] is False
    assert "jlens boom" in out["disposed"]["jlens_reason"]


def test_null_vector_is_deterministic_for_the_same_run_and_concept(tmp_path):
    """Reproducibility: the SAME (run, concept, layer) must draw the SAME null direction across
    calls (uses a stable sha256-derived seed, not Python's randomized str hash())."""
    source = _make_source(tmp_path)
    ec1 = FakeEngineClient(vocab={" ocean": [3]})
    ec2 = FakeEngineClient(vocab={" ocean": [3]})
    out1 = sr.swap_receipt(RUN, None, "ocean", FakeSub(ec1),
                           concept_steer=concept_dir.ConceptSteer(ec1, source=source, layer=21))
    out2 = sr.swap_receipt(RUN, None, "ocean", FakeSub(ec2),
                           concept_steer=concept_dir.ConceptSteer(ec2, source=source, layer=21))
    assert ec1.intervene_calls[1]["vector"] == ec2.intervene_calls[1]["vector"]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
