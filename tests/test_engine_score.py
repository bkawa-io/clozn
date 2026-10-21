"""/score seam: teacher-forced per-token logprob scoring,
reachable from Python and shaped for the receipts stack (rederive.py, forced receipts -- not yet
built here). Two layers under test:

  * clozn_engine.EngineClient.score -- the thin SDK wrapper: request-body construction (prompt vs
    prompt_ids precedence and continuation_ids vs continuation precedence).
  * clozn_server.EngineSubstrate.score_tokens -- assembles the prompt EXACTLY like chat() does
    (inject_block + the worker template) from explicit inputs, which makes a stored run's
    with/without receipt arms reconstructable from the run record alone.

Model-free throughout: EngineClient._post and EngineSubstrate.engine/.steer are faked (no C++ server, no
GPU), mirroring test_engine_layers.py / test_engine_substrate.py's own conventions. The C++ /score route
itself (engine/core/serve/clozn_server.cpp) and its self-consistency-vs-generation acceptance are
validated separately against a live clozn-server (not exercised by this offline suite).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn_engine import EngineClient          # noqa: E402
from clozn.server import app as cs           # noqa: E402
from tests.chat_templates import render_qwen  # noqa: E402


# ==================================================================================== EngineClient.score

def test_score_sends_prompt_text_and_continuation_ids(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {"ok": True})
    ec.score(prompt="hello", continuation_ids=[1, 2, 3])
    assert seen["path"] == "/score"
    assert seen["body"] == {
        "topk": 0,
        "use_prefix_cache": True,
        "prompt": "hello",
        "continuation_ids": [1, 2, 3],
    }


def test_score_prompt_ids_take_precedence_over_prompt_text(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.score(prompt="ignored", prompt_ids=[9, 8, 7], continuation_ids=[1])
    assert "prompt" not in seen["body"]
    assert seen["body"]["prompt_ids"] == [9, 8, 7]


def test_score_continuation_ids_take_precedence_over_continuation_text(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.score(prompt="p", continuation_ids=[1, 2], continuation="ignored text")
    assert "continuation" not in seen["body"]
    assert seen["body"]["continuation_ids"] == [1, 2]


def test_score_continuation_text_fallback_when_no_ids(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.score(prompt="p", continuation="hello world")
    assert seen["body"]["continuation"] == "hello world"
    assert "continuation_ids" not in seen["body"]


def test_score_passes_topk(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.score(prompt="p", continuation_ids=[1], topk=5)
    assert seen["body"]["topk"] == 5


def test_score_can_force_the_cold_reference_path(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.score(prompt="p", continuation_ids=[1], use_prefix_cache=False)
    assert seen["body"]["use_prefix_cache"] is False


def test_score_omits_steer_fields_when_not_given(monkeypatch):
    seen = {}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: seen.update(path=path, body=body) or {})
    ec.score(prompt="p", continuation_ids=[1])
    assert "steer" not in seen["body"]
    assert "steer_vec" not in seen["body"]


def test_score_returns_the_raw_engine_reply(monkeypatch):
    canned = {"n_prompt": 3, "n_cont": 2, "tokens": [{"id": 1, "piece": "a", "logprob": -0.1}],
              "sum_logprob": -0.1}
    ec = EngineClient(port=1)
    monkeypatch.setattr(ec, "_post", lambda path, body: canned)
    assert ec.score(prompt="p", continuation_ids=[1, 2]) == canned


# ==================================================================================== EngineSubstrate.score_tokens

class _FakeScoreEngine:
    """Stands in for clozn_engine.EngineClient inside EngineSubstrate.score_tokens: .score() plus
    .apply_template() (score_tokens now templates the prompt via the engine's per-model chat template,
    not a hardcoded Qwen string). This fake mimics a ChatML model, so the rendered prompt carries ChatML
    markers here; on a real engine the FORMAT follows the loaded GGUF (see the live cross-model proof)."""

    def __init__(self, reply=None):
        self.calls = []
        self.template_calls = []
        self._reply = reply if reply is not None else {"tokens": []}

    def apply_template(self, messages, add_assistant=True):
        self.template_calls.append([dict(m) for m in messages])
        return render_qwen(messages)

    def score(self, **kw):
        self.calls.append(kw)
        return self._reply


def _bare_engine_substrate(engine, steer):
    """EngineSubstrate via object.__new__ (mirrors test_engine_substrate.py's own helper) -- exercises
    score_tokens's prompt assembly without a real worker."""
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    sub.steer = steer
    return sub


def test_score_tokens_assembles_prompt_from_explicit_block_not_live_memory(monkeypatch):
    """block is an EXPLICIT input, never read from anywhere else."""
    fe = _FakeScoreEngine()
    sub = _bare_engine_substrate(fe, steer=None)
    def _boom(*a, **k):
        raise AssertionError("score_tokens must not read live memory")

    block = "Here is what you know about them:\n- loves rock climbing"
    sub.score_tokens([{"role": "user", "content": "hi"}], [1, 2, 3], block=block)

    # the block was folded into the system message and handed to the ENGINE to template (per-model);
    # the fake mimics a ChatML engine, so ChatML markers appear here.
    assert fe.template_calls[-1][0] == {"role": "system", "content": block}
    sent_prompt = fe.calls[-1]["prompt"]
    assert block in sent_prompt
    assert "<|im_start|>system" in sent_prompt


def test_score_tokens_omits_the_block_when_none():
    fe = _FakeScoreEngine()
    sub = _bare_engine_substrate(fe, steer=None)
    sub.score_tokens([{"role": "user", "content": "hi"}], [1], block=None)
    assert "rock climbing" not in fe.calls[-1]["prompt"]


def test_score_tokens_forwards_continuation_ids_as_ints_and_topk():
    fe = _FakeScoreEngine()
    sub = _bare_engine_substrate(fe, steer=None)
    sub.score_tokens([{"role": "user", "content": "hi"}], [1.0, 2.0, 3], block=None, topk=5)
    assert fe.calls[-1]["continuation_ids"] == [1, 2, 3]
    assert fe.calls[-1]["topk"] == 5


def test_score_tokens_many_reports_aggregate_prefix_cache_work(monkeypatch):
    monkeypatch.setenv("CLOZN_MINIMAL_CONTEXT_BATCH_WORKERS", "1")
    fe = _FakeScoreEngine(reply={
        "tokens": [{"id": 1, "piece": "x", "logprob": -0.1}],
        "cache": {"matched_tokens": 90, "cached_tokens": 80, "evaluated_tokens": 20},
        "numerical_regime": "scalar_prefix_cached_v1",
    })
    sub = _bare_engine_substrate(fe, steer=None)

    result = sub.score_tokens_many([
        {"messages": [{"role": "user", "content": "a"}], "continuation_ids": [1]},
        {"messages": [{"role": "user", "content": "b"}], "continuation_ids": [1]},
    ])

    assert len(result["results"]) == 2
    assert result["prompt_prefix_cache"] == {
        "request_count": 2,
        "cache_hit_count": 2,
        "matched_tokens": 180,
        "cached_tokens": 160,
        "evaluated_tokens": 40,
    }



def test_score_tokens_omits_steering_when_no_raw_vector_is_given():
    fe = _FakeScoreEngine()
    sub = _bare_engine_substrate(fe, None)
    sub.score_tokens([{"role": "user", "content": "hi"}], [1], block=None)
    assert "steer_vec" not in fe.calls[-1]


def test_score_tokens_returns_the_tokens_list():
    reply = {"n_prompt": 5, "n_cont": 2,
             "tokens": [{"id": 1, "piece": "a", "logprob": -0.1}, {"id": 2, "piece": "b", "logprob": -0.2}],
             "sum_logprob": -0.3}
    fe = _FakeScoreEngine(reply=reply)
    sub = _bare_engine_substrate(fe, steer=None)
    out = sub.score_tokens([{"role": "user", "content": "hi"}], [1, 2], block=None)
    assert out == reply["tokens"]


def test_score_tokens_tolerates_a_degraded_reply():
    fe = _FakeScoreEngine(reply={})
    sub = _bare_engine_substrate(fe, steer=None)
    assert sub.score_tokens([{"role": "user", "content": "hi"}], [1], block=None) == []


# ================================================================= continuation text fallback

def test_score_tokens_continuation_text_fallback_when_no_ids():
    fe = _FakeScoreEngine()
    sub = _bare_engine_substrate(fe, steer=None)
    sub.score_tokens([{"role": "user", "content": "hi"}], None, continuation="hello world", block=None)
    assert fe.calls[-1]["continuation"] == "hello world"
    assert "continuation_ids" not in fe.calls[-1]


def test_score_tokens_continuation_ids_take_precedence_over_continuation_text():
    fe = _FakeScoreEngine()
    sub = _bare_engine_substrate(fe, steer=None)
    sub.score_tokens([{"role": "user", "content": "hi"}], [1, 2], continuation="ignored", block=None)
    assert fe.calls[-1]["continuation_ids"] == [1, 2]
    assert "continuation" not in fe.calls[-1]
