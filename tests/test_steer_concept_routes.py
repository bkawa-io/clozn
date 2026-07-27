"""test_steer_concept_routes.py -- POST /steer/concept/set + /steer/concept/check
(clozn.server.app.Substrate._steer), the last-mile studio wiring for
the any-concept dial (dir(c), clozn/behavior/steering/concept_dir.py's ConceptSteer) dropped in
ALONGSIDE the existing tone-dial routes (/steer/set, /steer/check, /steer/custom), so the studio UI can
"type a word -> steer" the same way it already drives the pole-pair tone dials.

  /steer/concept/set   {concept, strength?} -> ConceptSteer.steer_toward(concept, strength): a built
                        dir(c) payload (ok/vector/coef/layer/token_id) PLUS the persisted `active` dial
                        map (mirrors /steer/set's {"active": ...} shape).
  /steer/concept/check {concept, strength?, prompt, max_new?} -> an A/B: baseline vs dir(concept)
                        injected via the engine's raw /intervene (mirrors /steer/check's A/B shape, but
                        ConceptSteer has no engage()/disengage() persistent-patch mechanism -- it rides
                        the SAME raw-vector wire /intervene already serves, see concept_dir.py's demo).

Model-free and GPU-free: clozn.server.app.ENGINE is monkeypatched to a FakeEngineClient (harvest/
complete/intervene/score); ENGINE_CONCEPT_STEER is a REAL ConceptSteer wired to a tiny on-disk J-lens +
unembed FIXTURE (orthogonal J_l + orthonormal W_U rows, so dir(c) is exact -- mirrors
test_concept_dir.py / test_swap_receipt.py's own construction), so dir(c) actually resolves without any
engine round trip for W_U (the lab-export path wins, see concept_dir.py's BLOCKER_NOTE). Drives the REAL
do_POST handler via the object.__new__(H) no-socket trick (test_receipts_server.py's _dispatch).

.harvest exists on FakeEngineClient purely so a MISBEHAVING test (or a regression that re-gates these
routes) would fail loudly rather than hang -- /steer/concept/* (ConceptSteer/dir(c)) is a mechanism
entirely separate from Substrate._ensure_steer()'s diff-of-means EngineSteer tone-dial calibration, and
(engine-down pressure test finding #1b) must never call it at all. The "harvest never called" section
near the bottom of this file proves exactly that, against an engine whose harvest() would raise.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from clozn.server import app as cs                          # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.behavior.steering.concept_dir as concept_dir    # noqa: E402




# ==================================================================================== J-lens/unembed fixtures (mirror test_swap_receipt.py)

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


# ==================================================================================== fake engine

class FakeEngineClient:
    """.harvest -- so the shared _ensure_steer() tone-dial harvest (unconditional, before any /steer/*
    branch) succeeds, mirroring test_engine_add_custom.py's _FakeEC. .score -- token resolution for
    ConceptSteer.resolve_token_id (continuation= text -> one token, unless the word is registered as
    MULTI-token). .complete/.intervene -- canned text, mirroring test_swap_receipt.py's FakeEngineClient."""

    N_EMBD = 8
    N_TOKENS = 5

    def __init__(self, multi_token_words=()):
        self.multi_token_words = set(multi_token_words)
        self.vocab = {}
        self._next_id = 0     # fixture vocab is d_model=32 rows -- stay well inside range
        self.harvest_calls = []
        self.complete_calls = []
        self.intervene_calls = []
        self.score_calls = []

    def harvest(self, text, layer=None):
        self.harvest_calls.append(text)
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        activations = rng.randn(self.N_TOKENS, self.N_EMBD).astype(np.float32)

        class _H:
            pass
        h = _H()
        h.activations = activations
        return h

    def score(self, prompt=None, continuation=None, continuation_ids=None, topk=0, **_kw):
        self.score_calls.append({"continuation": continuation, "continuation_ids": continuation_ids})
        if continuation is not None:
            word = continuation.strip()
            if word in self.multi_token_words:
                return {"tokens": [{"id": 1, "piece": word[:2]}, {"id": 2, "piece": word[2:]}]}
            tid = self.vocab.get(word)
            if tid is None:
                tid = self._next_id
                self._next_id += 1
                self.vocab[word] = tid
            return {"tokens": [{"id": tid, "piece": word}]}
        return {"tokens": [{"id": 0, "piece": "x", "logprob": -1.0}]}

    def complete(self, prompt, max_tokens=64, **_kw):
        self.complete_calls.append(prompt)
        return {"choices": [{"text": "a plain baseline reply"}]}

    def intervene(self, prompt, vector=None, coef=None, layer=None, max_tokens=64, **_kw):
        self.intervene_calls.append({"vector": vector, "coef": coef, "layer": layer})
        return {"choices": [{"text": "a reply steered toward the concept"}]}


# ==================================================================================== HTTP dispatch (mirrors test_receipts_server.py)

def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite might touch, mirroring test_engine_add_custom.py's own iso fixture."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def fake_concept_engine(iso, monkeypatch):
    """clozn.server.app.ENGINE -> a fresh FakeEngineClient; ENGINE_STEER/ENGINE_CONCEPT_STEER reset
    so _engine_steer()/_engine_concept_steer() build fresh on it. SUB -> a real EngineSubstrate on top,
    so /steer/concept/* is dispatched exactly the way a live studio request would be."""
    ec = FakeEngineClient()
    monkeypatch.setattr(cs, "ENGINE", ec)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", None)
    sub = cs.EngineSubstrate()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return ec


@pytest.fixture
def fixture_source(iso):
    """A tiny on-disk J-lens + unembed export -- dir(c) resolves exactly, no engine round trip needed
    for W_U (the lab-export path wins)."""
    jdir = _write_jlens_fixture(iso)
    udir = _write_unembed_fixture(iso)
    return concept_dir.ConceptDirSource(jlens_dir=jdir, unembed_dir=udir)


@pytest.fixture
def wired_concept_steer(fake_concept_engine, fixture_source, monkeypatch):
    """Swap the lazily-built ENGINE_CONCEPT_STEER for one pointed at the fixture source (so dir(c)
    resolves) -- mirrors _engine_concept_steer()'s own cache slot, just pre-filled for the test."""
    steer = concept_dir.ConceptSteer(fake_concept_engine, source=fixture_source, layer=21, median_norm=10.0)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", steer)
    return steer


# ==================================================================================== _engine_concept_steer() -- mirrors _engine_steer()

def test_engine_concept_steer_is_none_when_engine_unconfigured(monkeypatch):
    monkeypatch.setattr(cs, "ENGINE", None)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", None)
    assert cs._engine_concept_steer() is None


def test_engine_concept_steer_builds_once_and_caches(monkeypatch):
    ec = FakeEngineClient()
    monkeypatch.setattr(cs, "ENGINE", ec)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", None)
    first = cs._engine_concept_steer()
    assert isinstance(first, concept_dir.ConceptSteer)
    assert first.ec is ec
    second = cs._engine_concept_steer()
    assert second is first                          # cached, not rebuilt


# ==================================================================================== /steer/concept/set

def test_steer_concept_set_needs_a_running_engine(fake_concept_engine, monkeypatch):
    monkeypatch.setattr(cs, "_engine_concept_steer", lambda: None)
    out = _post("/steer/concept/set", {"concept": "ocean"})
    assert out == {"error": "concept dials need the product model worker (CLOZN_ENGINE_PORT)"}


def test_steer_concept_set_needs_a_concept_word(wired_concept_steer):
    out = _post("/steer/concept/set", {})
    assert out == {"error": "need a concept word"}


def test_steer_concept_set_happy_path_builds_dir_c(wired_concept_steer):
    out = _post("/steer/concept/set", {"concept": "ocean", "strength": 0.4})
    assert out["ok"] is True
    assert out["concept"] == "ocean"
    assert out["layer"] == 21
    assert out["strength"] == 0.4
    assert isinstance(out["token_id"], int)
    assert isinstance(out["vector"], list) and len(out["vector"]) == 32
    assert out["coef"] == pytest.approx(0.4 * 10.0)
    assert out["active"] == {"ocean": 0.4}           # persisted on the ConceptSteer, mirrors /steer/set


def test_steer_concept_set_default_strength_is_the_validated_midpoint(wired_concept_steer):
    out = _post("/steer/concept/set", {"concept": "ocean"})
    assert out["strength"] == concept_dir.DEFAULT_STRENGTH


def test_steer_concept_set_accumulates_active_dials_across_calls(wired_concept_steer):
    _post("/steer/concept/set", {"concept": "ocean", "strength": 0.3})
    out = _post("/steer/concept/set", {"concept": "paris", "strength": 0.5})
    assert out["active"] == {"ocean": 0.3, "paris": 0.5}


def test_steer_concept_set_blocked_on_a_multi_token_concept(fake_concept_engine, fixture_source, monkeypatch):
    fake_concept_engine.multi_token_words.add("unresolvable")
    steer = concept_dir.ConceptSteer(fake_concept_engine, source=fixture_source, layer=21, median_norm=10.0)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", steer)
    out = _post("/steer/concept/set", {"concept": "unresolvable"})
    assert out["ok"] is False
    assert out["blocked"] == "token_resolution"
    assert "note" in out


# ==================================================================================== /steer/concept/check

def test_steer_concept_check_needs_a_concept_word(wired_concept_steer):
    out = _post("/steer/concept/check", {"prompt": "The capital of France is"})
    assert out == {"error": "need a concept word"}


def test_steer_concept_check_happy_path_ab(wired_concept_steer):
    out = _post("/steer/concept/check", {"concept": "ocean", "strength": 0.4,
                                         "prompt": "The capital of France is"})
    assert out["concept"] == "ocean"
    assert out["strength"] == 0.4
    assert out["baseline"] == "a plain baseline reply"
    assert out["steered"] == "a reply steered toward the concept"
    assert out["layer"] == 21
    assert isinstance(out["token_id"], int)
    assert out["coef"] == pytest.approx(0.4 * 10.0)


def test_steer_concept_check_reports_blocked_when_concept_unresolvable(fake_concept_engine, fixture_source,
                                                                       monkeypatch):
    fake_concept_engine.multi_token_words.add("unresolvable")
    steer = concept_dir.ConceptSteer(fake_concept_engine, source=fixture_source, layer=21, median_norm=10.0)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", steer)
    out = _post("/steer/concept/check", {"concept": "unresolvable", "prompt": "hi"})
    assert out["baseline"] == "a plain baseline reply"    # baseline still generated
    assert out["steered"] is None
    assert out["blocked"] == "token_resolution"


def test_steer_concept_check_passes_the_built_vector_to_intervene(wired_concept_steer, fake_concept_engine):
    _post("/steer/concept/check", {"concept": "ocean", "strength": 0.4, "prompt": "hi"})
    assert len(fake_concept_engine.intervene_calls) == 1
    call = fake_concept_engine.intervene_calls[0]
    assert call["layer"] == 21
    assert call["coef"] == pytest.approx(4.0)
    assert isinstance(call["vector"], list) and len(call["vector"]) == 32


# ==================================================================================== fix 1b: no tone-dial calibration
# Engine-down pressure test finding #1b: /steer/concept/* used to sit behind Substrate._steer's
# unconditional _ensure_steer() call (the SAME diff-of-means EngineSteer harvest /steer/set etc. need),
# even though ConceptSteer/dir(c) is a completely separate mechanism that never touches it. Proven here
# against an engine whose harvest() raises -- if a regression ever re-gated these routes behind
# _ensure_steer(), these tests would fail with a raised/propagated error instead of the clean result below.

class _HarvestRefusingEngineClient(FakeEngineClient):
    """Identical to FakeEngineClient, except .harvest() always raises -- standing in for a fully
    unreachable engine from _ensure_steer()'s point of view (mirrors clozn_engine.EngineClient's own
    behavior on a connection refusal: a raw urllib.error.URLError, never caught by _request())."""

    def harvest(self, text, layer=None):
        import urllib.error
        self.harvest_calls.append(text)
        raise urllib.error.URLError("refused")


@pytest.fixture
def fake_concept_engine_harvest_down(iso, monkeypatch):
    ec = _HarvestRefusingEngineClient()
    monkeypatch.setattr(cs, "ENGINE", ec)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", None)
    sub = cs.EngineSubstrate()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return ec


@pytest.fixture
def wired_concept_steer_harvest_down(fake_concept_engine_harvest_down, fixture_source, monkeypatch):
    steer = concept_dir.ConceptSteer(fake_concept_engine_harvest_down, source=fixture_source,
                                     layer=21, median_norm=10.0)
    monkeypatch.setattr(cs, "ENGINE_CONCEPT_STEER", steer)
    return steer


def test_steer_concept_set_never_triggers_the_tone_dial_calibration_harvest(
        wired_concept_steer_harvest_down, fake_concept_engine_harvest_down):
    out = _post("/steer/concept/set", {"concept": "ocean", "strength": 0.4})
    assert out["ok"] is True                                    # succeeds despite harvest() being lethal
    assert fake_concept_engine_harvest_down.harvest_calls == []  # ...because it was never called


def test_steer_concept_check_never_triggers_the_tone_dial_calibration_harvest(
        wired_concept_steer_harvest_down, fake_concept_engine_harvest_down):
    out = _post("/steer/concept/check", {"concept": "ocean", "strength": 0.4, "prompt": "hi"})
    assert out["steered"] == "a reply steered toward the concept"
    assert fake_concept_engine_harvest_down.harvest_calls == []
