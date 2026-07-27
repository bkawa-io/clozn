"""test_engine_substrate -- EngineSubstrate: chat + prompt-mode memory + tone dials on the C++ GGUF
engine, NO PyTorch model resident (clozn_server.EngineSubstrate / RUNTIME_SPLIT.md's keystone). This is
what lets /v1/chat/completions (and, via SUB.chat(), the whole receipts/replay/explain/narrate/
counterfactual stack) run on the fast engine instead of a loaded Qwen-7B.

Model-free throughout -- no C++ engine process, no GPU, no real socket. clozn_server.ENGINE is
monkeypatched to a FakeEngine whose `.base` points at a closed local port (127.0.0.1:1, with a short
`.timeout`): _engine_complete_traced's streaming attempt fails fast (no DNS lookup, an immediate refused/
timed-out connect) and falls through to its own pre-existing plain-.complete() fallback -- the "stream
hiccup" path clozn_server.py already ships for exactly this case. That fallback is what every
FakeEngine-backed test below actually exercises; it is NOT itself under test here (see test_hf_trace.py /
test_trace_capture.py for _engine_complete_traced's own streaming-path coverage).

Covers:
  * EngineSubstrate.__init__: builds when ENGINE is configured and raises when it is not.
  * EngineSubstrate.chat(): returns the engine's text (stripped); fills mem_out/trace_out exactly like
    QwenSubstrate.chat; folds the prompt-mode memory block into the rendered chat-template prompt (and
    omits it when there is none); forwards the active dials' steer_vec (falling back to disk when the
    live steer carries no strength, mirroring the pre-existing /engine/chat hybrid endpoint).
  * _EngineMemory: card-store-backed .rules (active cards only), .prefix always None, no-op
    consolidate()/reset(), .state() shape.
  * steering.EngineSteer's new SteeringControl-compatible surface: clear/engage/disengage/active/
    save_state/load_state, and generate()'s engage-gated default-dial path (the clean A/B baseline
    Substrate._steer's /steer/check needs).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs          # noqa: E402
import clozn.memory.cards as memory_cards                # noqa: E402
import clozn.memory.mode as memory_mode                 # noqa: E402
import clozn.memory.anchored as anchored_memory          # noqa: E402
import clozn.memory.topic_gate as topic_gate             # noqa: E402
from clozn.behavior.steering import EngineSteer   # noqa: E402


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
    machine: CLOZN_DIR (_pers()/_disk_dials()/EngineSteer.save_state/load_state), the card store
    (_EngineMemory.rules/.state), and the settings file (_EngineMemory.__init__'s memory_strength read).
    Mirrors test_dial_library_server.py / test_memory_mode.py's own iso fixtures."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    # the anchored-bag store too: a REAL bag in ~/.clozn/anchored_bags.json would honestly steer any
    # apply_anchored=True chat here and leak machine state into these unit tests
    monkeypatch.setattr(anchored_memory, "BAGS_PATH", str(tmp_path / "anchored_bags.json"))
    return tmp_path


@pytest.fixture
def fake_engine(monkeypatch):
    """clozn_server.ENGINE -> a fresh FakeEngine; ENGINE_STEER reset to None so _engine_steer()
    builds a real steering.EngineSteer on it (construction itself makes no network call -- .compute()
    would, but nothing here has a reason to call it)."""
    fe = FakeEngine()
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    return fe


def _no_block(mem, last_user, strength=None):
    return None, [], 0.0


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
    assert isinstance(sub.steer, EngineSteer)
    assert sub.memory is sub._mem


# ==================================================================================== J-transport auto-wiring
# (engine_adapter.EngineSteer.enable_j_transport / jlens_transport.py -- see their docstrings.
# EngineSubstrate is the ONE real production call site: it auto-enables J-transport using the
# running engine's OWN reported model_sha256, the strongest model identity this substrate ever
# actually has (no local GGUF file path to re-derive full contracts.gguf_identity() metadata
# from). This is always safe to attempt -- see jlens_transport's HONESTY CONTRACT -- because
# "no compact-eligible artifact claims this GGUF" degrades to an exact no-op, never a guess.

def test_engine_substrate_auto_enables_j_transport_when_engine_reports_model_sha256(iso, monkeypatch):
    class _FakeEngineWithHealth(FakeEngine):
        def health(self):
            return {"model": "qwen2.5-7b", "model_sha256": "abc123abcd"}

    fe = _FakeEngineWithHealth()
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    assert sub.model_sha256 == "abc123abcd"
    assert sub.steer._j_transport is True
    assert sub.steer._jlens_model_sha256 == "abc123abcd"


def test_engine_substrate_leaves_j_transport_off_without_a_model_sha256(iso, fake_engine):
    """FakeEngine (no .health() at all -- this suite's default) -> model_sha256 stays None -> the
    J-transport wiring is skipped entirely, so every OTHER test in this file (none of which ever
    supply a model_sha256) is byte-for-byte unaffected by its existence."""
    sub = cs.EngineSubstrate()
    assert sub.model_sha256 is None
    assert sub.steer._j_transport is False
    assert sub.steer.last_j_transport is None


# ==================================================================================== fix #2: lazy identity re-resolution
# (engine-down pressure test finding #2): model_family/model_id/model_sha256 used to resolve from ONE
# best-effort health() call at __init__ time, wrapped in a bare `except Exception: pass`. If the engine
# was down when the gateway started, they stayed None (the "clozn-local" fallback) for the REST OF THE
# PROCESS's life -- only a full restart re-derived them -- which silently and permanently disabled
# per-model dial persistence (self._pers_steer, set once from model_sha256) even after the engine came
# back up. They're properties now (backed by _resolve_identity()/_maybe_reresolve_identity()): a read
# while still unresolved retries (cooldown-gated, so a persistently-down engine never taxes every
# request with an extra health() round-trip), and never re-fetches once resolved.

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
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    assert fe.health_calls == 1                          # one attempt, at construction
    assert sub.model_sha256 is None
    assert sub.model_family is None
    assert sub.model_id is None
    assert sub._pers_steer is None


def test_identity_retry_is_cooldown_gated_not_on_every_read(iso, monkeypatch):
    """A persistently-down engine must not pay a health() round-trip on every single property read --
    that would add the connect-refused tax to every ordinary request while the engine stays down."""
    fe = _FlakyHealthEngine(fail_times=99)
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    assert fe.health_calls == 1
    for _ in range(10):
        assert sub.model_sha256 is None
        assert sub.model_family is None
        assert sub._pers_steer is None
    assert fe.health_calls == 1                          # still just the one attempt -- cooldown held


def test_identity_lazily_resolves_once_the_engine_comes_back(iso, monkeypatch):
    fe = _FlakyHealthEngine(fail_times=1)                # down for one attempt, then healthy
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    assert sub.model_sha256 is None                      # unresolved right after construction
    assert fe.health_calls == 1

    sub._identity_last_attempt -= (sub._IDENTITY_RETRY_COOLDOWN_S + 1)   # the cooldown has now elapsed

    assert sub.model_sha256 == "deadbeef01"              # the read itself triggers the retry
    assert fe.health_calls == 2
    assert sub.model_family == "qwen2.5-7b"
    assert sub.model_id == "Qwen/Qwen2.5-7B-Instruct"
    assert sub._pers_steer is not None
    assert "deadbeef01" in sub._pers_steer


def test_identity_never_refetches_once_resolved(iso, monkeypatch):
    fe = _FlakyHealthEngine(fail_times=1)
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    sub._identity_last_attempt -= (sub._IDENTITY_RETRY_COOLDOWN_S + 1)
    assert sub.model_sha256 == "deadbeef01"
    assert fe.health_calls == 2

    sub._identity_last_attempt -= 10 ** 6                # force the cooldown check to be moot either way
    for _ in range(5):
        assert sub.model_sha256 == "deadbeef01"
    assert fe.health_calls == 2                          # never re-fetched once resolved


def test_dial_persistence_activates_once_identity_resolves(iso, monkeypatch):
    """The functional payoff, not just cosmetics: once model_sha256 resolves, self._pers_steer becomes a
    real per-model path and saving a dial value actually persists to disk -- silently and permanently
    broken before this fix, for the rest of a process that started with the engine down."""
    fe = _FlakyHealthEngine(fail_times=1)
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    assert sub._pers_steer is None                       # dial persistence disabled while unresolved

    sub._identity_last_attempt -= (sub._IDENTITY_RETRY_COOLDOWN_S + 1)
    path = sub._pers_steer
    assert path is not None

    sub.steer.set("warm", 0.7)
    sub.steer.save_state(sub._pers_steer)                # exactly what /steer/set's route code does
    assert os.path.isfile(path)
    with open(path, encoding="utf-8") as f:
        assert json.load(f)["warm"] == 0.7


# ==================================================================================== chat() basics

def test_chat_returns_the_engines_text_stripped(iso, fake_engine, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    fake_engine.text = "  hello there  "
    sub = cs.EngineSubstrate()
    assert sub.chat([{"role": "user", "content": "hi"}]) == "hello there"
    assert len(fake_engine.calls) == 1             # the .complete() fallback actually ran


def test_chat_fills_mem_out_and_trace_out_without_raising(iso, fake_engine, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = cs.EngineSubstrate()
    trace_out, mem_out = [], {}
    reply = sub.chat([{"role": "user", "content": "hi"}], trace_out=trace_out, mem_out=mem_out)
    assert reply == "hi"
    assert {k: mem_out[k] for k in ("mode", "applied", "gate")} == {
        "mode": "prompt", "applied": [], "gate": 0.0}
    assert mem_out["prompt_block"] is None
    assert mem_out["assembled_messages"] == [{"role": "user", "content": "hi"}]
    assert trace_out == []                         # the fallback path traces empty, but never raises


# ==================================================================================== chat() -- prompt-mode memory folds into the prompt

def test_chat_folds_the_memory_block_into_the_rendered_prompt(iso, fake_engine, monkeypatch):
    block = "Here is what you know about them:\n- loves rock climbing"
    monkeypatch.setattr(cs, "_prompt_block_for",
                        lambda mem, last_user, strength=None: (block, [{"id": "c1", "text": "x"}], 1.0))
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "what should I do this weekend?"}], mem_out=mem_out)

    assert {k: mem_out[k] for k in ("mode", "applied", "gate")} == {
        "mode": "prompt", "applied": [{"id": "c1", "text": "x"}], "gate": 1.0}
    assert mem_out["prompt_block"] == block
    assert mem_out["assembled_messages"] == [
        {"role": "system", "content": block},
        {"role": "user", "content": "what should I do this weekend?"},
    ]
    # the block-bearing assembled messages were handed to the ENGINE to template (per-model), NOT
    # pre-rendered as Qwen ChatML in Python -- the model-agnostic seam.
    assert fake_engine.template_calls[-1] == mem_out["assembled_messages"]
    sent_prompt = fake_engine.calls[-1]["prompt"]
    assert block in sent_prompt                    # the block actually reached the engine's prompt
    assert "<|im_start|>system" in sent_prompt      # fake mimics a ChatML engine (a Llama engine would emit headers)
    assert "what should I do this weekend?" in sent_prompt


def test_chat_omits_the_block_when_prompt_block_for_returns_none(iso, fake_engine, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = cs.EngineSubstrate()
    sub.chat([{"role": "user", "content": "hi"}])
    assert "loves rock climbing" not in fake_engine.calls[-1]["prompt"]


# ==================================================================================== chat() -- final_prompt capture (backlog #5)

def test_chat_records_the_rendered_final_prompt_in_mem_out(iso, fake_engine, monkeypatch):
    """mem_out.final_prompt is the EXACT rendered string the engine templated -- the same string that
    reached generation (fake_engine.calls[-1]['prompt']). _log_run persists it as run.final_prompt."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "hi"}], mem_out=mem_out)
    assert mem_out["final_prompt"] == fake_engine.calls[-1]["prompt"]   # exactly what generation saw
    assert mem_out["final_prompt"]                                      # non-empty even with no memory block
    assert "hi" in mem_out["final_prompt"]


def test_chat_final_prompt_contains_the_memory_block(iso, fake_engine, monkeypatch):
    block = "Here is what you know about them:\n- loves rock climbing"
    monkeypatch.setattr(cs, "_prompt_block_for",
                        lambda mem, last_user, strength=None: (block, [{"id": "c1", "text": "x"}], 1.0))
    sub = cs.EngineSubstrate()
    mem_out = {}
    sub.chat([{"role": "user", "content": "plans?"}], mem_out=mem_out)
    # the rendered final_prompt is the post-template form; assembled_messages is its pre-template form.
    assert block in mem_out["final_prompt"]
    assert mem_out["final_prompt"] == fake_engine.calls[-1]["prompt"]


# ==================================================================================== chat() -- dials forward a steer_vec

class FakeSteer:
    """A minimal SteeringControl-compatible double: just chat()'s TONE branch needs (.strength,
    .layer, .steer_vector()) -- unlike a real steering.EngineSteer, no .compute()/harvest() (which
    would need a live engine to derive axis directions from) is involved."""

    def __init__(self, strength, vec, layer=14):
        self.strength = dict(strength)
        self._vec = vec
        self.layer = layer
        self.vector_calls = []

    def steer_vector(self, strength):
        self.vector_calls.append(dict(strength))
        return self._vec


def _bare_engine_substrate(engine, steer, mem=None):
    """EngineSubstrate via object.__new__ (mirrors test_dial_library_server.py's _bare_substrate) --
    exercises chat()'s dial-forwarding logic directly against a hand-picked FakeSteer, without needing
    a real EngineSteer (whose .compute() would need a live engine to harvest axis vectors from)."""
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    sub.steer = steer
    sub._mem = mem if mem is not None else cs._EngineMemory()
    sub.memory = sub._mem
    return sub


def test_chat_forwards_the_active_dials_steer_vec_to_the_engine(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    fe = FakeEngine()
    steer = FakeSteer(strength={"warm": 1.0}, vec=[0.1, 0.2, 0.3], layer=14)
    sub = _bare_engine_substrate(fe, steer)

    sub.chat([{"role": "user", "content": "hi"}])

    assert steer.vector_calls == [{"warm": 1.0}]
    params = fe.calls[-1]["params"]
    assert params["steer_vec"] == [0.1, 0.2, 0.3]
    assert params["steer"] == {"coef": 1.0, "layer": 14}


def test_chat_skips_steer_vec_when_no_dial_is_active(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    fe = FakeEngine()
    steer = FakeSteer(strength={"warm": 0.0}, vec=None)      # present, but every value is falsy
    sub = _bare_engine_substrate(fe, steer)

    sub.chat([{"role": "user", "content": "hi"}])

    assert steer.vector_calls == []                # any(st.values()) is False -> never even asked
    assert "steer_vec" not in fe.calls[-1]["params"]


def test_chat_falls_back_to_disk_dials_when_the_live_steer_has_no_strength(iso, monkeypatch):
    """self.steer.strength == {} (e.g. nothing dialed yet this process) -> chat() reads the persisted
    personality.json dials instead, exactly like the pre-existing /engine/chat hybrid endpoint does."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    with open(os.path.join(str(iso), "studio_personality.json"), "w", encoding="utf-8") as f:
        json.dump({"warm": 0.8}, f)
    fe = FakeEngine()
    steer = FakeSteer(strength={}, vec=[0.4, 0.5])
    sub = _bare_engine_substrate(fe, steer)

    sub.chat([{"role": "user", "content": "hi"}])

    assert steer.vector_calls == [{"warm": 0.8}]
    assert fe.calls[-1]["params"]["steer_vec"] == [0.4, 0.5]


def test_chat_can_apply_anchored_memory_when_live_path_opts_in(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    bag = {"card_id": "mem_tea", "card_text": "likes tea gardens",
           "vector": [1.0, 0.0], "on": True,
           "terms": [{"token": "tea", "alpha": 0.7}, {"token": "gardens", "alpha": 0.2}]}
    monkeypatch.setattr(anchored_memory, "active_bags", lambda: [bag])

    class Gate:
        def scalar(self, prompt, texts):
            return 0.5

    monkeypatch.setattr(topic_gate, "get_gate", lambda: Gate())
    fe = FakeEngine()
    sub = _bare_engine_substrate(fe, FakeSteer(strength={}, vec=None))
    mem_out = {}

    sub.chat([{"role": "user", "content": "tell me about tea"}],
             mem_out=mem_out, apply_anchored=True)

    params = fe.calls[-1]["params"]
    assert params["steer"] == {"coef": 1.0, "layer": anchored_memory.LAYER}
    assert params["steer_vec"][0] == pytest.approx(anchored_memory.SCALE * 0.5 * anchored_memory.BASE_NORM)
    assert mem_out["anchored"][0]["card_id"] == "mem_tea"
    assert mem_out["anchored"][0]["gate"] == pytest.approx(0.5)
    assert mem_out["anchored_layer"] == anchored_memory.LAYER


def test_chat_leaves_anchored_memory_off_by_default_for_replay_receipt_paths(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    bag = {"card_id": "mem_tea", "card_text": "likes tea",
           "vector": [1.0, 0.0], "on": True, "terms": [{"token": "tea", "alpha": 1.0}]}
    monkeypatch.setattr(anchored_memory, "active_bags", lambda: [bag])
    fe = FakeEngine()
    sub = _bare_engine_substrate(fe, FakeSteer(strength={}, vec=None))

    sub.chat([{"role": "user", "content": "hi"}])

    assert "steer_vec" not in fe.calls[-1]["params"]


def test_chat_records_anchor_skip_when_tone_dials_use_raw_steer_slot(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    bag = {"card_id": "mem_tea", "card_text": "likes tea",
           "vector": [1.0, 0.0], "on": True, "terms": [{"token": "tea", "alpha": 1.0}]}
    monkeypatch.setattr(anchored_memory, "active_bags", lambda: [bag])
    monkeypatch.setattr(topic_gate, "get_gate", lambda: type("Gate", (), {"scalar": lambda self, p, t: 1.0})())
    fe = FakeEngine()
    sub = _bare_engine_substrate(fe, FakeSteer(strength={"warm": 1.0}, vec=[0.1, 0.2], layer=14))
    mem_out = {}

    sub.chat([{"role": "user", "content": "hi"}], mem_out=mem_out, apply_anchored=True)

    assert fe.calls[-1]["params"]["steer_vec"] == [0.1, 0.2]
    assert mem_out["anchored_skipped"] == "tone dials held the raw-steer channel this turn"


# ==================================================================================== the anchored-memory loop guard
# The loop-guard policy's substrate wiring: chat()'s non-streaming path only ever generates
# through the module-level cs._engine_complete_traced (FakeEngine's own .complete()/.apply_template are
# irrelevant here) -- monkeypatching THAT one seam to a canned call-by-call responder lets these tests
# drive the guard's retry/zero/flag behavior without any real per-token engine trace.

LOOP_PIECES = ["the", "cake"] * 4                      # period-2 cycle, 8 pieces -- detect_loop fires
CLEAN_PIECES = ["The", "quiet", "temple", "gardens", "of", "Kyoto", "draw", "visitors"]  # no cycle


class _FakeTraced:
    """Stand-in for cs._engine_complete_traced: the Nth call returns the Nth canned
    (text, pieces, finish) response as (text, steps, finish, (None, None)), and records every call's kw
    so a test can inspect exactly what steer rode each regeneration attempt."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, engine, prompt, max_tokens, kw, sample=None):
        self.calls.append({"engine": engine, "prompt": prompt, "max_tokens": max_tokens, "kw": dict(kw),
                           "sample": sample})
        text, pieces, finish = self.responses[len(self.calls) - 1]
        steps = [{"piece": p} for p in pieces]
        return text, steps, finish, (None, None)


def _anchored_chat_setup(monkeypatch):
    """One active bag + a fail-open gate (mirrors test_chat_can_apply_anchored_memory_when_live_path_opts_in),
    so apply_anchored=True actually composes and injects a steer -- the guard's precondition."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    bag = {"card_id": "mem_tea", "card_text": "likes tea gardens", "vector": [1.0, 0.0], "on": True,
           "terms": [{"token": "tea", "alpha": 0.7}, {"token": "gardens", "alpha": 0.2}]}
    monkeypatch.setattr(anchored_memory, "active_bags", lambda: [bag])

    class Gate:
        def scalar(self, prompt, texts):
            return 1.0

    monkeypatch.setattr(topic_gate, "get_gate", lambda: Gate())
    fe = FakeEngine()
    return _bare_engine_substrate(fe, FakeSteer(strength={}, vec=None))


def test_loop_guard_no_loop_is_byte_identical_to_today(iso, monkeypatch):
    """No loop on the first generation -> exactly one _engine_complete_traced call, the reply/trace pass
    through untouched, and mem_out never even gains an anchored_loop_guard key."""
    sub = _anchored_chat_setup(monkeypatch)
    fake = _FakeTraced([("a clean reply", CLEAN_PIECES, "stop")])
    monkeypatch.setattr(cs, "_engine_complete_traced", fake)
    mem_out, trace_out = {}, []

    reply = sub.chat([{"role": "user", "content": "tell me about tea"}],
                     mem_out=mem_out, trace_out=trace_out, apply_anchored=True)

    assert reply == "a clean reply"
    assert len(fake.calls) == 1
    assert trace_out == [{"piece": p} for p in CLEAN_PIECES]
    assert "anchored_loop_guard" not in mem_out
    assert mem_out["anchored"][0]["card_id"] == "mem_tea"    # the original injection record, untouched


def test_loop_guard_fires_retries_at_half_strength_and_resolves(iso, monkeypatch):
    """First generation loops under the FULL-strength anchored steer -> retry ONCE at s_total/2 -> clean
    -> the retried reply wins, mem_out records the retry honestly, and anchored_s_total is corrected to
    the halved value that actually shaped the final reply."""
    sub = _anchored_chat_setup(monkeypatch)
    fake = _FakeTraced([("the cake the cake...", LOOP_PIECES, "stop"),
                        ("a clean retried reply", CLEAN_PIECES, "stop")])
    monkeypatch.setattr(cs, "_engine_complete_traced", fake)
    mem_out, trace_out = {}, []

    reply = sub.chat([{"role": "user", "content": "tell me about tea"}],
                     mem_out=mem_out, trace_out=trace_out, apply_anchored=True)

    assert reply == "a clean retried reply"
    assert len(fake.calls) == 2
    assert trace_out == [{"piece": p} for p in CLEAN_PIECES]     # the RETRIED trace, not the looping one
    full_steer_vec = fake.calls[0]["kw"]["steer_vec"]
    half_steer_vec = fake.calls[1]["kw"]["steer_vec"]
    assert half_steer_vec == pytest.approx([x * 0.5 for x in full_steer_vec])
    assert fake.calls[1]["kw"]["steer"] == {"coef": 1.0, "layer": anchored_memory.LAYER}
    assert mem_out["anchored_loop_guard"] == {"fired": True, "action": "retried@s/2", "resolved": True}
    full_s_total = anchored_memory.SCALE * 1.0
    assert mem_out["anchored_s_total"] == pytest.approx(full_s_total / 2.0)
    # never a claim the memory "worked" -- the composed-bag record is factual (what was injected first),
    # the guard block is the honest correction on top of it
    assert mem_out["anchored"][0]["card_id"] == "mem_tea"


def test_loop_guard_still_loops_falls_back_to_zeroed_steer(iso, monkeypatch):
    """Still loops at half strength -> a THIRD pass with the anchored steer entirely absent from kw (not
    just re-zeroed -- genuinely unsteered, since the raw-steer slot was free before anchored claimed it).
    Flags "disabled", not "retried@s/2" -- the halved retry did NOT resolve it."""
    sub = _anchored_chat_setup(monkeypatch)
    fake = _FakeTraced([("loop 1", LOOP_PIECES, "stop"),
                        ("loop 2", LOOP_PIECES, "stop"),
                        ("finally clean", CLEAN_PIECES, "stop")])
    monkeypatch.setattr(cs, "_engine_complete_traced", fake)
    mem_out, trace_out = {}, []

    reply = sub.chat([{"role": "user", "content": "tell me about tea"}],
                     mem_out=mem_out, trace_out=trace_out, apply_anchored=True)

    assert reply == "finally clean"
    assert len(fake.calls) == 3
    zeroed_kw = fake.calls[2]["kw"]
    assert "steer_vec" not in zeroed_kw and "steer" not in zeroed_kw
    assert mem_out["anchored_loop_guard"] == {"fired": True, "action": "disabled", "resolved": True}
    assert mem_out["anchored_s_total"] == 0.0


def test_loop_guard_never_engages_without_an_actual_anchored_injection(iso, monkeypatch):
    """apply_anchored=True but no active bags -> _apply_anchored_memory injects nothing (comp is None) ->
    the guard must not even LOOK at the trace, even though these pieces would trip detect_loop on their
    own -- a looping reply with no anchored memory involved is not this guard's problem."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    monkeypatch.setattr(anchored_memory, "active_bags", lambda: [])
    fe = FakeEngine()
    sub = _bare_engine_substrate(fe, FakeSteer(strength={}, vec=None))
    fake = _FakeTraced([("the cake the cake...", LOOP_PIECES, "stop")])
    monkeypatch.setattr(cs, "_engine_complete_traced", fake)
    mem_out = {}

    reply = sub.chat([{"role": "user", "content": "hi"}], mem_out=mem_out, apply_anchored=True)

    assert reply == "the cake the cake..."
    assert len(fake.calls) == 1                      # no retry
    assert "anchored_loop_guard" not in mem_out


def test_loop_guard_never_engages_when_apply_anchored_is_false(iso, monkeypatch):
    """Even with an active bag composing cleanly, apply_anchored=False (the receipts/replay default)
    never calls _apply_anchored_memory at all -- comp is always None, so a looping reply here is simply
    returned as-is, exactly like today."""
    sub = _anchored_chat_setup(monkeypatch)
    fake = _FakeTraced([("the cake the cake...", LOOP_PIECES, "stop")])
    monkeypatch.setattr(cs, "_engine_complete_traced", fake)
    mem_out = {}

    reply = sub.chat([{"role": "user", "content": "hi"}], mem_out=mem_out)   # apply_anchored defaults False

    assert reply == "the cake the cake..."
    assert len(fake.calls) == 1
    assert "anchored_loop_guard" not in mem_out
    assert "anchored" not in mem_out


# ==================================================================================== _EngineMemory

def test_engine_memory_rules_reads_active_cards_only(iso, monkeypatch):
    cards = [{"id": "1", "text": "likes tea", "status": "active"},
             {"id": "2", "text": "pending thing", "status": "pending"},
             {"id": "3", "text": "likes coffee", "status": "active"},
             {"id": "4", "text": "old thing", "status": "disabled"}]
    monkeypatch.setattr(memory_cards, "list_cards", lambda: cards)
    assert cs._EngineMemory().rules == ["likes tea", "likes coffee"]


def test_engine_memory_prefix_is_always_none(iso):
    assert cs._EngineMemory().prefix is None


def test_engine_memory_consolidate_and_reset_are_noops(iso):
    mem = cs._EngineMemory()
    assert mem.consolidate(["some rule"]) == {"ok": True, "mode": "prompt"}
    assert mem.reset() is None                      # a no-op that doesn't raise


def test_engine_memory_state_shape(iso, monkeypatch):
    cards = [{"id": "1", "text": "likes tea", "status": "active"}]
    monkeypatch.setattr(memory_cards, "list_cards", lambda: cards)
    mem = cs._EngineMemory()
    assert mem.state() == {"mode": "prompt", "has_prefix": False, "cards": 1, "rules": ["likes tea"]}


# ==================================================================================== EngineSteer's new SteeringControl-compatible surface

class _FakeEC:
    """A stand-in engine client for steering.EngineSteer ITSELF (distinct from FakeEngine above, which
    stands in for clozn_server's module-level ENGINE) -- just enough for generate()'s no-dial path
    (.complete) vs its dialed path (.intervene), so engage()'s gating is observable without a real
    engine or a harvested axis vector."""

    def __init__(self):
        self.complete_calls = []
        self.intervene_calls = []

    def complete(self, prompt, **params):
        self.complete_calls.append(prompt)
        return {"choices": [{"text": "baseline"}]}

    def intervene(self, prompt, **params):
        self.intervene_calls.append(prompt)
        return {"choices": [{"text": "steered"}]}


def test_engine_steer_save_and_load_state_round_trip(tmp_path):
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 0.7, "concise": -0.3}
    path = str(tmp_path / "nested" / "personality.json")   # save_state must create the parent dir too
    es.save_state(path)

    es2 = EngineSteer(_FakeEC())
    es2.load_state(path)
    assert es2.strength == {"warm": 0.7, "concise": -0.3}


def test_engine_steer_load_state_missing_file_is_a_noop(tmp_path):
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 0.5}
    es.load_state(str(tmp_path / "nope.json"))
    assert es.strength == {"warm": 0.5}             # untouched


def test_engine_steer_load_state_corrupt_file_is_a_noop_never_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 0.5}
    es.load_state(str(path))
    assert es.strength == {"warm": 0.5}             # untouched -- a corrupt file must never crash a boot


def test_engine_steer_clear_empties_strength():
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 1.0}
    es.clear()
    assert es.strength == {}


def test_engine_steer_active_filters_zeros_keeps_negatives():
    es = EngineSteer(_FakeEC())
    es.strength = {"warm": 1.0, "concise": 0.0, "formal": -0.5}
    assert es.active() == {"warm": 1.0, "formal": -0.5}


def test_engine_steer_generate_gates_dials_on_engage_disengage():
    ec = _FakeEC()
    es = EngineSteer(ec)
    es.ready = True                                 # skip .compute() -- no live engine to harvest from
    es.vecs = {"warm": np.array([1.0, 0.0])}
    es.strength = {"warm": 1.0}

    es.generate("hello")                            # unengaged default path -> the clean, unsteered baseline
    assert ec.complete_calls == ["hello"]
    assert ec.intervene_calls == []

    es.engage()
    es.generate("hello")                            # engaged -> the active dial actually applies
    assert ec.intervene_calls == ["hello"]
    assert ec.complete_calls == ["hello"]            # (unchanged from the call above)

    es.disengage()
    es.generate("hello")                            # back to the clean baseline
    assert ec.complete_calls == ["hello", "hello"]


def test_engine_steer_generate_explicit_strength_is_unaffected_by_engage():
    """An explicit `strength=` kwarg (how /engine/steer/check already calls generate()) bypasses the
    engage gate entirely -- CHANGE 1 only touches the strength=None DEFAULT path."""
    ec = _FakeEC()
    es = EngineSteer(ec)
    es.ready = True
    es.vecs = {"warm": np.array([1.0, 0.0])}
    es.generate("hello", strength={"warm": 1.0})    # never engaged, but strength is explicit
    assert ec.intervene_calls == ["hello"]
    assert ec.complete_calls == []


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


def test_run_meta_includes_request_specific_generation_fields_after_chat(iso, monkeypatch):
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    sub.steer = None
    sub._mem = cs._EngineMemory()
    sub.memory = sub._mem

    # sample=False (the receipt/replay contract) -- this test is about request-specific fields
    # (max_tokens/stream) riding into run_meta(), not about S5 sampling; see the dedicated
    # test_run_meta_reflects_a_sampled_chat_call below for that.
    sub.chat([{"role": "user", "content": "hi"}], max_new=17, sample=False)

    meta = sub.run_meta()
    assert meta["max_tokens"] == 17
    assert meta["stream"] is False
    assert meta["temperature"] == 0.0
    assert meta["seed"] == 0


# ==================================================================================== S5: interactive sampling

def test_run_meta_reflects_a_sampled_chat_call(iso, monkeypatch):
    """sample=True (interactive chat's default) + the "sampling" setting ON (the S5 default) -> run_meta
    reports the REAL regime this call used: the Ollama/llama.cpp canonical params, sampler_mode "sample",
    and a decode block with a concrete (non-zero) seed -- not the greedy baseline."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    sub.steer = None
    sub._mem = cs._EngineMemory()
    sub.memory = sub._mem

    sub.chat([{"role": "user", "content": "hi"}], max_new=17, sample=True)

    meta = sub.run_meta()
    assert meta["sampler_mode"] == "sample" and meta["sampling"] == "sample"
    assert meta["temperature"] == 0.8
    assert meta["repetition_penalty"] == 1.1
    assert isinstance(meta["seed"], int) and meta["seed"] != 0
    decode = meta["decode"]
    assert decode["mode"] == "sample"
    assert decode["top_p"] == 0.9 and decode["top_k"] == 40
    assert "note" not in decode                   # top-p/k are enforced by engine/core/src/sample.cpp


def test_sampling_setting_off_forces_greedy_even_when_sample_true(iso, monkeypatch):
    """The persisted "sampling" setting is the master off switch -- OFF means every sample=True caller
    still gets exactly today's greedy behavior (temperature 0, seed 0), byte-identical to pre-S5."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    memory_mode.set_setting("sampling", False)
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    sub.steer = None
    sub._mem = cs._EngineMemory()
    sub.memory = sub._mem

    sub.chat([{"role": "user", "content": "hi"}], max_new=17, sample=True)

    meta = sub.run_meta()
    assert meta["sampler_mode"] == "greedy" and meta["sampling"] == "greedy"
    assert meta["temperature"] == 0.0
    assert meta["seed"] == 0
    assert meta["decode"] == {"mode": "greedy", "temperature": 0.0, "seed": 0}


def test_sample_false_stays_greedy_regardless_of_the_sampling_setting(iso, monkeypatch):
    """The receipt/replay contract: sample=False ALWAYS decodes greedy, even with "sampling" ON (the
    default) -- the setting is read only AFTER want_sample is checked (_resolve_sampling), so it can
    never turn a forced-greedy call into a sampled one."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    memory_mode.set_setting("sampling", True)     # explicit: still must not matter
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = _HealthEngine()
    sub.steer = None
    sub._mem = cs._EngineMemory()
    sub.memory = sub._mem

    sub.chat([{"role": "user", "content": "hi"}], max_new=17, sample=False)

    meta = sub.run_meta()
    assert meta["sampler_mode"] == "greedy"
    assert meta["temperature"] == 0.0 and meta["seed"] == 0


def test_resolve_sampling_generates_a_fresh_seed_each_call(iso):
    """A fresh per-turn seed (not a fixed one) is what S5 promises -- two resolutions differ."""
    a = cs._resolve_sampling(True)
    b = cs._resolve_sampling(True)
    assert a["on"] is True and b["on"] is True
    assert a["seed"] != b["seed"]


def test_explicit_request_sampling_fields_win_over_the_studio_default(iso):
    """An OpenAI request is a per-call contract: Studio's persisted master switch must not silently
    discard fields the HTTP request explicitly supplied."""
    memory_mode.set_setting("sampling", False)
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
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
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


def test_request_context_carries_sampling_steering_and_trace(iso, fake_engine, monkeypatch):
    """The context's fields are actually POPULATED, not just plumbing -- sampling (the resolved regime),
    steering_snapshot (a COPY of the dial strengths this call used), and trace (the per-token steps)."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    steer = FakeSteer(strength={"warm": 0.6}, vec=[0.1, 0.2], layer=14)
    sub = cs.EngineSubstrate()
    sub.steer = steer
    fake_engine.text = "hello there"

    sub.chat([{"role": "user", "content": "hi"}], sample=False)

    req = sub._request
    assert req.sampling is None                      # sample=False -> greedy -> _resolve_sampling -> None
    assert req.steering_snapshot == {"warm": 0.6}
    assert req.steering_snapshot is not steer.strength   # a COPY -- a later live mutation must not retro-edit it
    assert req.finish_reason is None or isinstance(req.finish_reason, str)
    assert isinstance(req.trace, list)


def test_last_generation_meta_never_shows_a_stale_mix_across_calls(iso, fake_engine, monkeypatch):
    """A sampled call followed by a forced-greedy call: the alias must show ONLY the second call's
    complete, self-consistent meta -- never e.g. a leftover sampled seed next to a greedy temperature."""
    monkeypatch.setattr(cs, "_prompt_block_for", _no_block)
    memory_mode.set_setting("sampling", True)
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
