"""test_engine_stream -- EngineSubstrate.chat_stream: the streaming twin of EngineSubstrate.chat (see
test_engine_substrate.py for chat()'s own coverage). Before this existed, `getattr(SUB, "chat_stream",
None)` was None on the pure-engine substrate, so /v1/chat/completions's SSE branch (_sse_chat) never
fired there -- a `stream: true` request silently fell through to one blocking chat() reply. chat_stream
gives the engine substrate the same live-token UX QwenSubstrate.chat_stream already has.

Model-free throughout -- no C++ engine process, no GPU, no real socket. urllib.request.urlopen is
monkeypatched directly to a fake response object iterable over canned SSE byte-lines, exercising
chat_stream's OWN streaming parse -- unlike test_engine_substrate.py's FakeEngine, which deliberately
points .base at a closed port so _engine_complete_traced's streaming attempt fails over to its
non-streaming .complete() fallback. chat_stream has no such fallback; it IS the stream.

Covers:
  * yields exactly the pieces from tokens_committed frames, in order, skipping empty pieces
  * fills mem_out via the SAME _prompt_block_for() call chat() makes
  * after the generator is exhausted, last_stream_trace() returns the accumulated per-token steps
  * the engine connection is closed whether the stream runs to [DONE] or the caller stops early
    (GeneratorExit) -- and GeneratorExit is never swallowed
  * the request body mirrors _engine_complete_traced's (stream, temperature=0.0, prompt, max_tokens)
  * active tone dials forward a steer_vec into the streaming request, exactly like chat()
"""
from __future__ import annotations

import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs          # noqa: E402
import clozn.memory.cards as memory_cards                # noqa: E402
import clozn.memory.mode as memory_mode                 # noqa: E402
import urllib.request               # noqa: E402


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite might touch so nothing reads or writes the real ~/.clozn on this
    machine (mirrors test_engine_substrate.py's own iso fixture)."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


def _no_block(mem, last_user, strength=None):
    return None, [], 0.0


class FakeEngine:
    """Just enough of clozn_engine.EngineClient's surface for chat_stream: .base (the URL it POSTs to),
    .timeout (the urlopen timeout), and .apply_template (chat_stream now templates the prompt via the
    engine's per-model chat template, not a hardcoded Qwen string). chat_stream has no .complete()
    fallback, so unlike test_engine_substrate.py's FakeEngine, that method is never exercised here. This
    fake mimics a ChatML model, so the rendered prompt carries ChatML markers here; on a real engine the
    FORMAT follows the loaded GGUF (see the live cross-model proof, not this model-free unit test)."""

    def __init__(self):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2

    def apply_template(self, messages, add_assistant=True):
        return cs._qwen_tmpl(messages)


def _bare_engine_substrate(engine, steer=None, mem=None):
    """EngineSubstrate via object.__new__ (mirrors test_engine_substrate.py's helper of the same name)
    -- exercises chat_stream's logic directly, without constructing a real EngineSteer/_EngineMemory."""
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    sub.steer = steer
    sub._mem = mem if mem is not None else cs._EngineMemory()
    sub.memory = sub._mem
    return sub


# --- canned engine SSE frames + a fake urlopen ---------------------------------------------------------

def _sse_line(obj):
    return ("data: " + json.dumps(obj) + "\n").encode("utf-8")


def canned_lines():
    """The engine's SSE frames for a 3-token completion (" Par" + "is" + "." -> "Paris."): gen_started, one
    tokens_committed per token (each needs its OWN `pos` -- accumulate_ar_events keys steps by position,
    so a missing/shared pos would collapse distinct tokens into one step), gen_finished, the final choices
    frame (the assembled text, no per-token data), then [DONE]. A blank keep-alive line is thrown in too --
    non-`data:` lines must be silently skipped, same as _engine_complete_traced's own parsing."""
    return [
        _sse_line({"type": "gen_started"}),
        b"\n",
        _sse_line({"type": "tokens_committed", "items": [{"piece": " Par", "conf": 0.9, "pos": 0}]}),
        _sse_line({"type": "tokens_committed", "items": [{"piece": "is", "conf": 0.8, "pos": 1}]}),
        _sse_line({"type": "tokens_committed", "items": [{"piece": ".", "conf": 0.7, "pos": 2}]}),
        _sse_line({"type": "gen_finished"}),
        _sse_line({"id": "cmpl-x", "object": "text_completion",
                   "choices": [{"text": " Paris.", "index": 0, "finish_reason": "stop"}]}),
        b"data: [DONE]\n",
    ]


class FakeSSEResponse:
    """Stand-in for urllib.request.urlopen's return value: iterable over canned SSE byte-lines (mirrors
    how http.client.HTTPResponse iterates -- undecoded bytes, one line per item), plus the .close() the
    generator's `finally` must always call. `closed` records whether that happened, so a test can assert
    the engine connection was actually released -- on a clean run AND on an early caller disconnect."""

    def __init__(self, lines):
        self._lines = list(lines)
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def close(self):
        self.closed = True


def _patch_urlopen(monkeypatch, lines):
    """Monkeypatch urllib.request.urlopen (module-level -- clozn_server's `import urllib.request` inside
    chat_stream binds the SAME module object, so patching it here reaches that call too) to hand back a
    fresh FakeSSEResponse over `lines` on every call. Returns the calls list ({req, timeout, resp} per
    call) so a test can inspect exactly what chat_stream sent and whether the response was closed after."""
    calls = []

    def _fake(req, timeout=None):
        resp = FakeSSEResponse(lines)
        calls.append({"req": req, "timeout": timeout, "resp": resp})
        return resp

    monkeypatch.setattr(urllib.request, "urlopen", _fake)
    return calls


@pytest.fixture
def fake_urlopen(monkeypatch):
    return _patch_urlopen(monkeypatch, canned_lines())


# ==================================================================================== yields + mem_out

def test_chat_stream_yields_pieces_in_order(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())
    mem_out = {}

    pieces = list(sub.chat_stream([{"role": "user", "content": "capital of France?"}], mem_out=mem_out))

    assert pieces == [" Par", "is", "."]
    assert {k: mem_out[k] for k in ("mode", "applied", "gate")} == {
        "mode": "prompt", "applied": [], "gate": 0.0}
    assert mem_out["prompt_block"] is None
    assert mem_out["assembled_messages"] == [{"role": "user", "content": "capital of France?"}]


def test_chat_stream_records_the_rendered_final_prompt(iso, monkeypatch, fake_urlopen):
    """backlog #5: chat_stream fills mem_out.final_prompt with the EXACT rendered string it POSTed to
    /v1/completions -- kept in lockstep with chat()'s own final_prompt capture."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())
    mem_out = {}
    list(sub.chat_stream([{"role": "user", "content": "hi"}], mem_out=mem_out))
    body = json.loads(fake_urlopen[-1]["req"].data.decode("utf-8"))
    assert mem_out["final_prompt"] == body["prompt"]       # the rendered string that reached generation
    assert mem_out["final_prompt"]                         # non-empty even with no memory block


def test_chat_stream_skips_empty_pieces(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    lines = [
        _sse_line({"type": "tokens_committed", "items": [{"piece": "", "conf": 0.5, "pos": 0}]}),
        _sse_line({"type": "tokens_committed", "items": [{"piece": "ok", "conf": 0.5, "pos": 1}]}),
        b"data: [DONE]\n",
    ]
    _patch_urlopen(monkeypatch, lines)
    sub = _bare_engine_substrate(FakeEngine())

    pieces = list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    assert pieces == ["ok"]                          # the empty piece at pos 0 is never yielded


def test_chat_stream_omits_the_block_when_prompt_block_for_returns_none(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    assert "Paris" not in fake_urlopen[-1]["req"].data.decode("utf-8")  # no block text leaked into the prompt


# ==================================================================================== last_stream_trace

def test_chat_stream_last_stream_trace_after_exhausting_the_generator(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    list(sub.chat_stream([{"role": "user", "content": "hi"}]))          # drain it fully

    steps = sub.last_stream_trace()
    assert len(steps) == 3
    assert [s["piece"] for s in steps] == [" Par", "is", "."]
    assert [s["conf"] for s in steps] == [0.9, 0.8, 0.7]


def test_last_stream_trace_is_empty_before_any_stream_ran(iso):
    sub = _bare_engine_substrate(FakeEngine())
    assert sub.last_stream_trace() == []


def test_last_stream_trace_returns_a_copy_not_a_live_reference(iso, monkeypatch, fake_urlopen):
    """Mirrors QwenSubstrate.last_stream_trace's contract: callers get list(...) of the stored steps, so
    mutating the returned list can't corrupt the substrate's own record."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())
    list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    got = sub.last_stream_trace()
    got.append({"piece": "tampered"})
    assert len(sub.last_stream_trace()) == 3


# ==================================================================================== the engine connection is always closed

def test_chat_stream_closes_the_connection_after_a_normal_stream(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    assert fake_urlopen[-1]["resp"].closed is True


def test_chat_stream_closes_the_connection_on_early_generator_close(iso, monkeypatch, fake_urlopen):
    """The studio's SSE handler stops consuming early on a client disconnect; the generator's own
    .close() sends GeneratorExit in at the `yield` -- it must not be swallowed (close() must not raise
    RuntimeError: generator ignored GeneratorExit), and the engine connection must still be released."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    gen = sub.chat_stream([{"role": "user", "content": "hi"}])
    first = next(gen)
    assert first == " Par"
    gen.close()                                       # must not raise -- GeneratorExit propagates cleanly

    assert fake_urlopen[-1]["resp"].closed is True


# ==================================================================================== cancellation (backlog #2): RequestContext.cancel()
# sse.py calls sub._request.cancel() the instant it detects the CLIENT is gone (a failed write), then
# gen.close() for an immediate stop. This is the belt-and-suspenders half: the read loop ALSO checks
# is_cancelled() between worker frames, so a caller that (for whatever reason) keeps pulling from the
# generator after marking it cancelled -- rather than closing it outright -- still gets a prompt stop
# instead of draining the rest of a reply nobody wants.

def test_chat_stream_stops_pulling_from_the_worker_once_cancelled(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    gen = sub.chat_stream([{"role": "user", "content": "capital of France?"}])
    first = next(gen)                     # the context now exists (chat_stream's top-of-body ran)
    assert first == " Par"
    sub._request.cancel()

    remaining = list(gen)                 # keep pulling -- the cancellation flag, not a close(), must stop it

    assert remaining == []                # "is." never arrives -- the loop broke on the NEXT frame boundary
    assert sub._request.is_cancelled() is True
    assert fake_urlopen[-1]["resp"].closed is True   # the worker connection was still released via `finally`


def test_chat_stream_cancellation_flag_is_false_by_default(iso, monkeypatch, fake_urlopen):
    """A normal, uncancelled stream's context reports is_cancelled() False throughout -- cancellation is
    opt-in, never a side effect of ordinary completion."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    pieces = list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    assert pieces == [" Par", "is", "."]
    assert sub._request.is_cancelled() is False


# ==================================================================================== the request mirrors _engine_complete_traced

def test_chat_stream_request_body_mirrors_engine_complete_traced(iso, monkeypatch, fake_urlopen):
    """With S5's "sampling" setting off, chat_stream's body is byte-identical to pre-S5: temperature 0,
    rep_penalty 1, seed 0 -- exactly what _engine_complete_traced's greedy fallback sends too."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    memory_mode.set_setting("sampling", False)
    sub = _bare_engine_substrate(FakeEngine())

    list(sub.chat_stream([{"role": "user", "content": "capital of France?"}], max_new=64))

    req = fake_urlopen[-1]["req"]
    assert req.full_url == "http://127.0.0.1:1/v1/completions"
    body = json.loads(req.data.decode("utf-8"))
    assert body["stream"] is True
    assert body["temperature"] == 0.0
    assert body["rep_penalty"] == 1.0
    assert body["seed"] == 0
    assert body["max_tokens"] == 64
    assert "capital of France?" in body["prompt"]
    assert "<|im_start|>assistant" in body["prompt"]        # rendered via the engine's apply_template (fake mimics ChatML)
    assert fake_urlopen[-1]["timeout"] == 0.2                # FakeEngine.timeout, via getattr fallback


# ==================================================================================== S5: interactive sampling

def test_chat_stream_samples_by_default(iso, monkeypatch, fake_urlopen):
    """The default-True `sample` arg inherits the persisted Ollama/llama.cpp params + a real seed:
    temperature, rep_penalty, AND the top_k/top_p nucleus."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = _bare_engine_substrate(FakeEngine())

    list(sub.chat_stream([{"role": "user", "content": "capital of France?"}], max_new=64))

    body = json.loads(fake_urlopen[-1]["req"].data.decode("utf-8"))
    assert body["temperature"] == 0.8
    assert body["rep_penalty"] == 1.1
    assert isinstance(body["seed"], int) and body["seed"] != 0
    assert body["top_k"] == 40 and body["top_p"] == 0.9

    meta = sub._last_generation_meta
    assert meta["sampler_mode"] == "sample"
    assert meta["decode"]["top_p"] == 0.9 and meta["decode"]["top_k"] == 40


def test_chat_stream_honors_per_request_sampling_override(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    memory_mode.set_setting("sampling", False)  # explicit HTTP fields must still win
    sub = _bare_engine_substrate(FakeEngine())

    sample = {"temperature": 0.4, "top_p": 0.65, "top_k": 12, "repeat_penalty": 1.03, "seed": 77}
    list(sub.chat_stream([{"role": "user", "content": "hi"}], sample=sample))

    body = json.loads(fake_urlopen[-1]["req"].data.decode("utf-8"))
    assert {key: body[key] for key in ("temperature", "top_p", "top_k", "rep_penalty", "seed")} == {
        "temperature": 0.4, "top_p": 0.65, "top_k": 12, "rep_penalty": 1.03, "seed": 77}


def test_chat_stream_sampling_off_is_byte_identical_to_pre_s5(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    memory_mode.set_setting("sampling", False)
    sub = _bare_engine_substrate(FakeEngine())

    list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    body = json.loads(fake_urlopen[-1]["req"].data.decode("utf-8"))
    assert body["temperature"] == 0.0 and body["rep_penalty"] == 1.0 and body["seed"] == 0
    assert sub._last_generation_meta["decode"] == {"mode": "greedy", "temperature": 0.0, "seed": 0}


# ==================================================================================== dial forwarding parity with chat()

class FakeSteer:
    """A minimal SteeringControl-compatible double (mirrors test_engine_substrate.py's FakeSteer): just
    chat_stream's TONE branch needs (.strength, .layer, .steer_vector())."""

    def __init__(self, strength, vec, layer=14):
        self.strength = dict(strength)
        self._vec = vec
        self.layer = layer
        self.vector_calls = []

    def steer_vector(self, strength):
        self.vector_calls.append(dict(strength))
        return self._vec


def test_chat_stream_forwards_the_active_dials_steer_vec(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    steer = FakeSteer(strength={"warm": 1.0}, vec=[0.1, 0.2, 0.3], layer=14)
    sub = _bare_engine_substrate(FakeEngine(), steer=steer)

    list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    assert steer.vector_calls == [{"warm": 1.0}]
    body = json.loads(fake_urlopen[-1]["req"].data.decode("utf-8"))
    assert body["steer_vec"] == [0.1, 0.2, 0.3]
    assert body["steer"] == {"coef": 1.0, "layer": 14}


def test_chat_stream_skips_steer_vec_when_no_dial_is_active(iso, monkeypatch, fake_urlopen):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    steer = FakeSteer(strength={"warm": 0.0}, vec=None)      # present, but every value is falsy
    sub = _bare_engine_substrate(FakeEngine(), steer=steer)

    list(sub.chat_stream([{"role": "user", "content": "hi"}]))

    assert steer.vector_calls == []                          # any(st.values()) is False -> never even asked
    body = json.loads(fake_urlopen[-1]["req"].data.decode("utf-8"))
    assert "steer_vec" not in body


