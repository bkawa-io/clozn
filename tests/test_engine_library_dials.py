"""test_engine_library_dials.py -- Phase 1 of RUNTIME_SPLIT.md: make the 27 shipped LIBRARY tone dials
(research/deploy_dial_library.py's ~/.clozn/studio_library.json) work on the C++ engine substrate
(steering.EngineSteer), the way the 6 built-in AXES dials already do.

On the HF backbone, a library dial is a SteeringControl.custom entry: metadata (pos/neg/max/poles) loaded
by load_custom -> add_custom, which immediately harvests a diff-of-means direction into .vecs, exactly like
a built-in AXES entry. EngineSteer had no .custom dict at all, so a library dial (i) never showed up in
/steer/axes on the engine substrate (clozn_server.Substrate._steer's "/steer/axes" reads
getattr(self.steer, "custom", {})) and (ii) had no direction vector to steer with. This suite covers the
new wiring end to end:

  load_library(path)      -- metadata (pos/neg/max/poles) into .custom, NO harvest call (cheap at studio
                              boot, before the engine may even be warmed up)
  compute()                -- AFTER the built-in AXES loop, harvests a direction for each .custom entry
                              too, using the SAME diff-of-means recipe, landing in .vecs exactly like a
                              built-in axis
  set() / steer_vector()   -- a library dial is capped by ITS OWN "max" (not the generic 1.5), and once in
                              .vecs, sums into steer_vector() like any other active dial

Model-free throughout: no live engine, no GPU. Mirrors test_engine_substrate.py's fake-engine pattern
(_FakeEC there covers .complete/.intervene for EngineSteer.generate; here it also grows a .harvest that
returns a fake Harvest-like object -- a numpy .activations of shape [n_tokens, 8] -- so compute() can run
its real diff-of-means recipe against canned, deterministic activations instead of a real model).
"""
from __future__ import annotations

import hashlib
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
from clozn.behavior.steering import AXES, EngineSteer   # noqa: E402


# --- a stand-in for clozn_engine.EngineClient, extended with .harvest ------------------------------

class _FakeHarvest:
    """Stands in for engine.client.clozn_engine.Harvest: compute() only reads `.activations`
    ([n_tokens, n_embd]) via `.activations.mean(0)`."""

    def __init__(self, activations):
        self.activations = activations


class _FakeEC:
    """A stand-in engine client: .harvest returns a DETERMINISTIC, text-dependent [n_tokens, 8] activation
    matrix (seeded off a stable hash of the text) -- so the pos/neg poles of the SAME axis/dial reliably
    produce a different, reproducible, non-degenerate diff-of-means direction, without needing any real
    model. Also carries .complete/.intervene (EngineSteer.generate's two paths)."""

    N_EMBD = 8
    N_TOKENS = 5

    def __init__(self):
        self.harvest_calls = []
        self.complete_calls = []
        self.intervene_calls = []

    def health(self):
        return {"status": "ok", "n_layer": 24, "n_embd": self.N_EMBD}

    def harvest(self, text, layer=None):
        self.harvest_calls.append(text)
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.RandomState(seed)
        activations = rng.randn(self.N_TOKENS, self.N_EMBD).astype(np.float32)
        return _FakeHarvest(activations)

    def complete(self, prompt, **params):
        self.complete_calls.append(prompt)
        return {"choices": [{"text": "baseline"}]}

    def intervene(self, prompt, **params):
        self.intervene_calls.append(prompt)
        return {"choices": [{"text": "steered"}]}


_ONE_DIAL = {"eli5": {"pos": "Respond simply.", "neg": "Respond expertly.", "max": 1.5,
                      "source": "library", "category": "audience_level"}}


# ==================================================================================== load_library()

def test_load_library_populates_custom_metadata_only(tmp_path):
    lib = dict(_ONE_DIAL, wry={"pos": "Respond wryly.", "neg": "Respond earnestly.", "max": 0.5,
                                "source": "library", "category": "affect_tone"})
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(lib), encoding="utf-8")

    ec = _FakeEC()
    es = EngineSteer(ec)
    es.load_library(str(path))

    assert set(es.custom) == {"eli5", "wry"}
    assert es.custom["eli5"] == {"pos": "Respond simply.", "neg": "Respond expertly.",
                                  "max": 1.5, "poles": ["eli5", "neutral"]}
    assert es.custom["wry"]["max"] == 0.5
    assert ec.harvest_calls == []            # metadata only -- no harvest yet
    assert es.vecs == {}                     # no direction computed yet either


def test_load_library_missing_file_is_a_noop(tmp_path):
    es = EngineSteer(_FakeEC())
    es.load_library(str(tmp_path / "nope.json"))
    assert es.custom == {}


def test_load_library_corrupt_file_is_a_noop_never_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.load_library(str(path))
    assert es.custom == {}


def test_load_library_is_idempotent(tmp_path):
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(_ONE_DIAL), encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.load_library(str(path))
    es.load_library(str(path))               # calling twice must not raise or duplicate/corrupt anything
    assert set(es.custom) == {"eli5"}


# ==================================================================================== compute()

def test_compute_derives_vecs_for_library_dials_too(tmp_path):
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(_ONE_DIAL), encoding="utf-8")
    ec = _FakeEC()
    es = EngineSteer(ec)
    es.load_library(str(path))

    info = es.compute(seeds=["hello", "tell me something"])

    assert set(AXES) <= set(es.vecs)                     # every built-in axis still computed
    assert "eli5" in es.vecs                             # ... and now the library dial too
    vec = es.vecs["eli5"]
    assert vec.shape == (8,)
    assert np.isfinite(vec).all()
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5  # unit direction
    assert "eli5" in info["axes"]
    assert es.ready is True
    assert es.layer == 12                         # derived from this loaded model, never Qwen's hardcoded 14


def test_compute_skips_a_custom_name_already_in_vecs(tmp_path):
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(_ONE_DIAL), encoding="utf-8")
    ec = _FakeEC()
    es = EngineSteer(ec)
    es.load_library(str(path))
    es.vecs["eli5"] = np.ones(8, dtype=np.float32)       # pretend it's already computed

    es.compute(seeds=["hello"])

    assert list(es.vecs["eli5"]) == [1.0] * 8            # untouched
    assert not any("Respond simply." in c for c in ec.harvest_calls)   # never re-harvested


# ==================================================================================== set()

def test_set_caps_a_library_dial_by_its_own_max_not_the_generic_default(tmp_path):
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(
        {"detailed": {"pos": "Respond with detail.", "neg": "Respond minimally.", "max": 0.25}}),
        encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.load_library(str(path))

    es.set("detailed", 5.0)          # way above both its own 0.25 max AND the generic 1.5
    assert es.strength["detailed"] == 0.25

    es.set("detailed", -5.0)
    assert es.strength["detailed"] == -0.25


def test_set_still_caps_a_builtin_axis_by_its_own_axes_max():
    es = EngineSteer(_FakeEC())
    es.set("candid", 99.0)           # AXES["candid"]["max"] == 0.45 -- unchanged built-in behavior
    assert es.strength["candid"] == 0.45


def test_set_still_falls_back_to_1_5_for_a_builtin_axis_with_no_explicit_max():
    es = EngineSteer(_FakeEC())
    es.set("warm", 99.0)             # AXES["warm"] has no "max" key -> generic default
    assert es.strength["warm"] == 1.5


# ==================================================================================== steer_vector()

def test_steer_vector_returns_a_vector_for_an_active_library_dial(tmp_path):
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(_ONE_DIAL), encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.load_library(str(path))

    vec = es.steer_vector({"eli5": 1.0})     # not yet .ready -> triggers compute() lazily

    assert vec is not None
    assert es.ready is True
    assert len(vec) == 8
    assert any(x != 0.0 for x in vec)


def test_steer_vector_composes_a_library_dial_with_a_builtin_axis(tmp_path):
    path = tmp_path / "studio_library.json"
    path.write_text(json.dumps(_ONE_DIAL), encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.load_library(str(path))

    vec = es.steer_vector({"eli5": 1.0, "warm": 0.5})

    assert vec is not None
    assert len(vec) == 8
    assert {"eli5", "warm"} <= set(es.vecs)


# ==================================================================================== clozn_server integration: EngineSubstrate + /steer/axes + /steer/set

@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite's server-level tests might touch, mirroring
    test_engine_substrate.py's own `iso` fixture exactly."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(memory_mode, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def fake_engine(monkeypatch):
    """clozn_server.ENGINE -> a fresh _FakeEC; ENGINE_STEER reset so _engine_steer() builds a real
    steering.EngineSteer on it (construction + load_library make no harvest call; only compute() would)."""
    fe = _FakeEC()
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    return fe


def _write_library(iso_dir, dials=_ONE_DIAL):
    with open(os.path.join(str(iso_dir), "studio_library.json"), "w", encoding="utf-8") as f:
        json.dump(dials, f)


def test_engine_substrate_loads_the_library_at_construction(iso, fake_engine):
    _write_library(iso)

    sub = cs.EngineSubstrate()

    assert "eli5" in sub.steer.custom
    assert sub.steer.custom["eli5"]["max"] == 1.5
    assert fake_engine.harvest_calls == []          # metadata only at boot -- no harvest yet


def test_engine_substrate_steer_axes_lists_the_library_dial_as_library_not_custom(iso, fake_engine):
    _write_library(iso)
    sub = cs.EngineSubstrate()

    result = sub.handle("/steer/axes", {})

    names = {a["name"]: a for a in result["axes"]}
    assert "eli5" in names
    assert names["eli5"].get("library") is True
    assert "custom" not in names["eli5"]
    assert result["substrate"] == "engine"
    # every built-in axis is still listed too -- the library addition doesn't crowd them out
    assert set(AXES) <= set(names)


def test_engine_substrate_steer_set_caps_a_library_dial_end_to_end(iso, fake_engine):
    _write_library(iso, dials={"detailed": {"pos": "Respond with detail.", "neg": "Respond minimally.",
                                             "max": 0.25, "source": "library", "category": "verbosity"}})
    sub = cs.EngineSubstrate()

    result = sub.handle("/steer/set", {"name": "detailed", "value": 9.0})

    assert result == {"active": {"detailed": 0.25}}   # capped by the LIBRARY dial's own max, not 1.5
    assert "detailed" in sub.steer.vecs               # compute() ran and derived its direction too


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
