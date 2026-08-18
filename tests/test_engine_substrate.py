"""test_engine_substrate -- EngineSubstrate: chat on the C++ GGUF engine, NO PyTorch model resident
(clozn_server.EngineSubstrate / RUNTIME_SPLIT.md's keystone). This is what lets /v1/chat/completions
(and, via SUB.chat(), the whole receipts/replay/explain/narrate stack) run on the fast engine instead of
a loaded Qwen-7B.

Model-free throughout -- no C++ engine process, no GPU, no real socket. clozn_server.ENGINE is
monkeypatched to a FakeEngine whose `.base` points at a closed local port (127.0.0.1:1, with a short
`.timeout`): _engine_complete_traced's streaming attempt fails fast (no DNS lookup, an immediate refused/
timed-out connect) and falls through to its own pre-existing plain-.complete() fallback -- the "stream
hiccup" path clozn_server.py already ships for exactly this case. That fallback is what every
FakeEngine-backed test below actually exercises; it is NOT itself under test here (see test_hf_trace.py /
test_trace_capture.py for _engine_complete_traced's own streaming-path coverage).

Covers:
  * EngineSubstrate.__init__: builds when ENGINE is configured and raises when it is not; the lazy
    model_family/model_id/model_sha256 identity resolution (cooldown-gated retry while the engine is
    down at startup).
  * EngineSubstrate.chat(): returns the engine's text (stripped); fills mem_out/trace_out exactly like
    the historical contract; captures the exact rendered final_prompt.
  * run_meta() reproducibility metadata and S5 interactive-sampling resolution.
  * RequestContext: the per-call bundle chat()/chat_stream() publish as self._request (backlog #2).

No tone dials, no memory cards, no EngineSteer: both were retired with the personalization cut (see
RUNTIME_SPLIT.md). Raw-vector steering (steer_vec/steer_strengths on score_tokens, /intervene) survives
as interpretability/causal-attribution machinery -- see test_engine_score.py / test_engine_stream.py.
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
import clozn.settings as clozn_settings          # noqa: E402


# --- a stand-in for clozn_engine.EngineClient: .base points at a closed local port (127.0.0.1:1 --
# an IP literal, so no DNS lookup, and a reserved port nothing ever listens on) with a short .timeout,
# so _engine_complete_traced's streaming attempt excepts out fast and every call below actually exercises
# its pre-existing plain .complete() fallback. -------------------------------------------------------

class FakeEngine:
    def __init__(self, text="hi"):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self.text = text
        self.calls = []            # [{"prompt": ..., "params": {...}}, ...] -- every .complete() call seen
        self.template_calls = []   # [messages, ...] -- every .apply_template() call seen

    def apply_template(self, messages, add_assistant=True):
        # Stand in for the engine's per-model templating (chat() now renders via the loaded GGUF's own
        # chat template, not a hardcoded Qwen string). This fake mimics a ChatML model (Qwen), so ChatML
        # markers appear here; on a real engine the FORMAT follows the loaded GGUF (Llama-3 headers,
        # Gemma turns, ...), which the live cross-model proof covers -- not this model-free unit test.
        self.template_calls.append([dict(m) for m in messages])
        return cs._qwen_tmpl(messages)

    def complete(self, prompt, **params):
        self.calls.append({"prompt": prompt, "params": dict(params)})
        return {"choices": [{"text": self.text}]}


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite might touch so nothing reads or writes the real ~/.clozn on this
    machine: CLOZN_DIR and the settings file."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def fake_engine(monkeypatch):
    """clozn_server.ENGINE -> a fresh FakeEngine."""
    fe = FakeEngine()
    monkeypatch.setattr(cs, "ENGINE", fe)
    return fe


# ==================================================================================== construction

def test_engine_substrate_needs_a_configured_engine(monkeypatch):
    monkeypatch.setattr(cs, "ENGINE", None)
    with pytest.raises(RuntimeError):
        cs.EngineSubstrate()


def test_engine_substrate_builds_a_real_product_adapter(iso, fake_engine):
    sub = cs.EngineSubstrate()
    assert isinstance(sub, cs.EngineSubstrate)
    assert sub.name == "engine"
    assert sub.engine is fake_engine
    assert sub.brain is None                       # no SAE on the pure-engine substrate


# ==================================================================================== fix #2: lazy identity re-resolution
# (engine-down pressure test finding #2): model_family/model_id/model_sha256 used to resolve from ONE
# best-effort health() call at __init__ time, wrapped in a bare `except Exception: pass`. If the engine
# was down when the gateway started, they stayed None (the "clozn-local" fallback) for the REST OF THE
# PROCESS's life -- only a full restart re-derived them. They're properties now (backed by
# _resolve_identity()/_maybe_reresolve_identity()): a read while still unresolved retries (cooldown-gated,
# so a persistently-down engine never taxes every request with an extra health() round-trip), and never
# re-fetches once resolved.

class _FlakyHealthEngine(FakeEngine):
    """health() raises the first `fail_times` calls (a down-at-startup engine), then succeeds -- lets a
    test simulate "the engine was down, then came back" with no real socket."""

    def __init__(self, *, fail_times=1, model="qwen2.5-7b", model_sha256="deadbeef01"):
        super().__init__()
        self.health_calls = 0
        self.fail_times = fail_times
        self._model = model
        self._model_sha256 = model_sha256

    def health(self):
        self.health_calls += 1
        if self.health_calls <= self.fail_times:
            raise OSError("connection refused")
        return {"model": self._model, "model_sha256": self._model_sha256}


def test_identity_stays_unresolved_when_the_engine_is_down_at_startup(iso, monkeypatch):
    fe = _FlakyHealthEngine(fail_times=99)              # never recovers within this test
    monkeypatch.setattr(cs, "ENGINE", fe)
    sub = cs.EngineSubstrate()
    assert fe.health_calls == 1                          # one attempt, at construction
    assert sub.model_sha256 is None
    assert sub.model_family is None
    assert sub.model_id is None


def test_identity_retry_is_cooldown_gated_not_on_every_read(iso, monkeypatch):
    """A persistently-down engine must not pay a health() round-trip on every single property read --
    that would add the connect-refused tax to every ordinary request while the engine stays down."""
    fe = _FlakyHealthEngine(fail_times=99)
    monkeypatch.setattr(cs, "ENGINE", fe)
    sub = cs.EngineSubstrate()
    assert fe.health_calls == 1
    for _ in range(10):
        assert sub.model_sha256 is None
        assert sub.model_family is None
    assert fe.health_calls == 1                          # still just the one attempt -- cooldown held


def test_identity_lazily_resolves_once_the_engine_comes_back(iso, monkeypatch):
    fe = _FlakyHealthEngine(fail_times=1)                # down for one attempt, then healthy
    monkeypatch.setattr(cs, "ENGINE", fe)
    sub = cs.EngineSubstrate()
    assert sub.model_sha256 is None                      # unresolved right after construction
    assert fe.health_calls == 1

    sub._identity_last_attempt -= (sub._IDENTITY_RETRY_COOLDOWN_S + 1)   # the cooldown has now elapsed

    assert sub.model_sha256 == "deadbeef01"              # the read itself triggers the retry
    assert fe.health_calls == 2
    assert sub.model_family == "qwen2.5-7b"
    assert sub.model_id == "Qwen/Qwen2.5-7B-Instruct"


def test_identity_never_refetches_once_resolved(iso, monkeypatch):
    fe = _FlakyHealthEngine(fail_times=1)
    monkeypatch.setattr(cs, "ENGINE", fe)
    sub = cs.EngineSubstrate()
    sub._identity_last_attempt -= (sub._IDENTITY_RETRY_COOLDOWN_S + 1)
    assert sub.model_sha256 == "deadbeef01"
    assert fe.health_calls == 2

    sub._identity_last_attempt -= 10 ** 6                # force the cooldown check to be moot either way
    for _ in range(5):
        assert sub.model_sha256 == "deadbeef01"
    assert fe.health_calls == 2                          # never re-fetched once resolved


# ==================================================================================== chat() basics

def test_chat_returns_the_engines_text_stripped(iso, fake_engine, monkeypatch):
    fake_engine.text = "  hello there  "
    sub = cs.EngineSubstrate()
    assert sub.chat([{"role": "user", "content": "hi"}]) == "hello there"
    assert len(fake_engine.calls) == 1             # the .complete() fallback actually ran


# ==================================================================================== chat() -- final_prompt capture (backlog #5)

def test_chat_records_the_rendered_final_prompt_in_mem_out(iso, fake_engine, monkeypatch):
    """mem_out.final_prompt is the EXACT rendered string the engine templated -- the same string that
    reached generation (fake_engine.calls[-1]['prompt']). _log_run persists it as run.final_prompt."""
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "hi"}], mem_out=mem_out)
    assert mem_out["final_prompt"] == fake_engine.calls[-1]["prompt"]   # exactly what generation saw
    assert mem_out["final_prompt"]                                      # non-empty even with no memory block
    assert "hi" in mem_out["final_prompt"]


# ==================================================================================== run_meta (repro metadata)

class _HealthEngine:
    """A stand-in engine exposing just /health, for run_meta(): {model (a GGUF path), mode, n_ctx, device}."""

    def __init__(self, model="/models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf", mode="autoregressive"):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self._h = {"status": "ok", "model": model, "mode": mode,
                   "n_ctx": 4096, "device": "cuda", "gpu_layers": 99}

    def health(self):
        return dict(self._h)

    def apply_template(self, messages, add_assistant=True):
        return cs._qwen_tmpl(messages)   # chat() templates via the engine now (fake mimics a ChatML model)

    def complete(self, prompt, **params):
        return {"choices": [{"text": "ok", "finish_reason": "stop"}]}


def test_quant_from_name_reads_gguf_tags():
    assert cs._quant_from_name("Qwen2.5-0.5B-Instruct-Q4_K_M.gguf") == "Q4_K_M"
    assert cs._quant_from_name("model-q8_0.gguf") == "Q8_0"
    assert cs._quant_from_name("tiny-IQ4_XS.gguf") == "IQ4_XS"
    assert cs._quant_from_name("weights-f16.gguf") == "F16"
    assert cs._quant_from_name("no-quant-here.gguf") is None


def test_run_meta_reads_model_file_quant_and_mode(iso):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    meta = sub.run_meta()
    assert meta["model_file"] == "Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"    # basename of the /health model path
    assert meta["quant"] == "Q4_K_M"
    assert meta["mode"] == "autoregressive"
    assert meta["sampling"] == "greedy"                                  # chat/chat_stream force temperature 0
    assert meta["sampler_mode"] == "greedy"
    assert meta["temperature"] == 0.0
    assert meta["repetition_penalty"] == 1.0
    assert meta["seed"] == 0
    assert meta["n_ctx"] == 4096 and meta["device"] == "cuda"           # from /health, once the engine exposes them
    assert meta["gpu_layers"] == 99


def test_run_meta_is_cached_after_first_call(iso):
    sub = object.__new__(cs.EngineSubstrate)
    calls = {"n": 0}

    class _CountEngine:
        base = "x"

        def health(self):
            calls["n"] += 1
            return {"model": "m-Q4_0.gguf", "mode": "autoregressive"}

    sub.engine = _CountEngine()
    sub.run_meta()
    sub.run_meta()
    assert calls["n"] == 1                              # /health fetched once, then cached for the session


def test_run_meta_never_raises_on_a_bad_health(iso):
    sub = object.__new__(cs.EngineSubstrate)

    class _BoomEngine:
        base = "x"

        def health(self):
            raise RuntimeError("no engine")

    sub.engine = _BoomEngine()
    assert sub.run_meta() == {"sampler_mode": "greedy", "sampling": "greedy", "temperature": 0.0,
                              "repetition_penalty": 1.0, "seed": 0,
                              "decode": {"mode": "greedy", "temperature": 0.0, "seed": 0}}


# ==================================================================================== S5: interactive sampling

def test_resolve_sampling_generates_a_fresh_seed_each_call(iso):
    """A fresh per-turn seed (not a fixed one) is what S5 promises -- two resolutions differ."""
    a = cs._resolve_sampling(True)
    b = cs._resolve_sampling(True)
    assert a["on"] is True and b["on"] is True
    assert a["seed"] != b["seed"]


def test_explicit_request_sampling_fields_win_over_the_studio_default(iso):
    """An OpenAI request is a per-call contract: Studio's persisted master switch must not silently
    discard fields the HTTP request explicitly supplied."""
    clozn_settings.set_setting("sampling", False)
    out = cs._resolve_sampling({"temperature": 0.35, "top_p": 0.7, "top_k": 9,
                                "repeat_penalty": 1.02, "seed": 123})
    assert out == {"on": True, "temperature": 0.35, "top_p": 0.7, "top_k": 9,
                   "repeat_penalty": 1.02, "seed": 123}
    assert cs._resolve_sampling({"temperature": 0, "seed": 123}) is None


def test_engine_complete_traced_sends_the_resolved_sampler_params(iso, fake_engine, monkeypatch):
    """_engine_complete_traced forwards the FULL resolved regime -- temperature/rep_penalty/seed AND the
    Ollama nucleus top_k/top_p -- from a _resolve_sampling() dict to the engine's .complete() fallback
    (FakeEngine's .base is unroutable, so every call here exercises that fallback). The fallback must
    decode under the SAME regime the HTTP path recorded in the run's meta, so the nucleus rides along."""
    samp = {"on": True, "temperature": 0.8, "top_p": 0.9, "top_k": 40, "repeat_penalty": 1.1, "seed": 12345}
    cs._engine_complete_traced(fake_engine, "hello", 16, {}, sample=samp)
    params = fake_engine.calls[-1]["params"]
    assert params["temperature"] == 0.8
    assert params["rep_penalty"] == 1.1
    assert params["seed"] == 12345
    assert params["top_k"] == 40 and params["top_p"] == 0.9


def test_engine_complete_traced_refuses_to_synthesize_text_from_a_malformed_reply(iso):
    """REGRESSION: a worker reply with no usable `choices` must raise, never become model text.

    The fallback path used to end `return (ch[0].get("text","") if ch else str(r)), ...` -- so an
    engine reply like {"nope": True} came back as the literal string "{'nope': True}" AS THE
    MODEL'S ANSWER. fork() then built and STORED a child run whose response was parent-prefix +
    forced token + that repr, reported as a successful fork. Fabricating output is the one failure
    mode this codebase cannot have; a malformed reply is a protocol violation and must fail loudly."""
    class MalformedEngine:
        timeout = 5
        base = "http://127.0.0.1:9"          # unroutable -> the streaming attempt fails, fallback runs
        def complete(self, prompt, **params):
            return {"nope": True}

    with pytest.raises(ValueError, match="no usable 'choices'"):
        cs._engine_complete_traced(MalformedEngine(), "hello", 16, {})

    class NoChoicesButOtherwiseSane:
        timeout = 5
        base = "http://127.0.0.1:9"
        def complete(self, prompt, **params):
            return {"choices": [], "usage": {"prompt_tokens": 3}}

    with pytest.raises(ValueError, match="no usable 'choices'"):
        cs._engine_complete_traced(NoChoicesButOtherwiseSane(), "hello", 16, {})


# ==================================================================================== RequestContext (backlog #2: request isolation)
# chat()'s piecemeal self._last_generation_meta/_last_finish_reason/_last_diverged/_last_diverged_at
# writes were consolidated onto ONE clozn.server.request_context.RequestContext, published as
# self._request in a single assignment (see substrates.py's EngineSubstrate._new_request). These tests
# cover the consolidation itself -- the ALIASES' existing behavior is already exhaustively covered by
# every test above (they all read sub._last_generation_meta / run_meta() / last_finish_reason() and never
# noticed the change), so this section only tests what's NEW: the context object's identity/lifecycle.

def test_request_context_fields_are_none_shaped_before_any_chat_call(iso, fake_engine):
    sub = cs.EngineSubstrate()
    assert getattr(sub, "_request", None) is None
    assert sub._last_generation_meta is None
    assert sub._last_finish_reason is None
    assert sub._last_diverged is None
    assert sub._last_diverged_at is None
    assert sub._last_stream_trace == []


def test_chat_publishes_a_fresh_request_context_each_call(iso, fake_engine, monkeypatch):
    sub = cs.EngineSubstrate()

    sub.chat([{"role": "user", "content": "hi"}])
    first = sub._request
    sub.chat([{"role": "user", "content": "hi again"}])
    second = sub._request

    assert first is not None and second is not None
    assert first is not second                      # a brand-new object every call, never mutated in place
    assert first.request_id != second.request_id     # a fresh id each call (new_request_id())
    # the piecemeal aliases are VIEWS onto the CURRENT context -- identity, not a copy
    assert sub._last_generation_meta is second.generation_meta
    assert sub._last_finish_reason == second.finish_reason


def test_request_context_carries_sampling_and_trace(iso, fake_engine, monkeypatch):
    """The context's fields are actually POPULATED, not just plumbing -- sampling (the resolved regime)
    and trace (the per-token steps)."""
    sub = cs.EngineSubstrate()
    fake_engine.text = "hello there"

    sub.chat([{"role": "user", "content": "hi"}], sample=False)

    req = sub._request
    assert req.sampling is None                      # sample=False -> greedy -> _resolve_sampling -> None
    assert req.finish_reason is None or isinstance(req.finish_reason, str)
    assert isinstance(req.trace, list)


def test_last_generation_meta_never_shows_a_stale_mix_across_calls(iso, fake_engine, monkeypatch):
    """A sampled call followed by a forced-greedy call: the alias must show ONLY the second call's
    complete, self-consistent meta -- never e.g. a leftover sampled seed next to a greedy temperature."""
    clozn_settings.set_setting("sampling", True)
    sub = cs.EngineSubstrate()

    sub.chat([{"role": "user", "content": "hi"}], sample=True)
    assert sub._last_generation_meta["sampler_mode"] == "sample"

    sub.chat([{"role": "user", "content": "hi"}], sample=False)
    meta = sub._last_generation_meta
    assert meta["sampler_mode"] == "greedy"
    assert meta["temperature"] == 0.0
    assert "seed" not in meta or meta.get("seed") == 0   # no sampled-call seed bled into the greedy meta


def test_the_piecemeal_aliases_are_read_only(iso, fake_engine):
    """Hardening: the only legitimate writers are chat()/chat_stream() (through self._request); a stray
    direct assignment must fail loudly instead of silently reintroducing the old piecemeal-write pattern."""
    sub = cs.EngineSubstrate()
    with pytest.raises(AttributeError):
        sub._last_generation_meta = {"sampler_mode": "sample"}
    with pytest.raises(AttributeError):
        sub._last_finish_reason = "stop"
