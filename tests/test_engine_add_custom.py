"""test_engine_add_custom.py -- USER-CREATED tone dials ("create your own tone dial") on the C++ engine
substrate (steering.EngineSteer). test_engine_library_dials.py ported the 27 SHIPPED library dials to the
engine; this suite ports the OTHER path -- a studio user typing two pole sentences of their own and getting
back a brand-new, steerable dial, exactly as SteeringControl.add_custom already does on the PyTorch backbone
(SteeringControl.add_custom on the PyTorch/HF adapter).

Before this, EngineSteer had no add_custom/remove_custom/save_custom/load_custom at all, so the generic
handler at clozn_server.Substrate._steer's "/steer/custom" route
(`if not hasattr(self.steer, "add_custom"): return {"error": "custom dials are not supported on this
substrate yet"}`) always short-circuited on the engine substrate. Once EngineSteer grows that surface, the
route works with NO handler change -- this suite proves it end to end.

  add_custom(name, pos, neg, mx)  -- the SAME diff-of-means recipe compute() uses for a built-in AXES
                                     entry, harvested live over one arbitrary pole pair; lands in .vecs AND
                                     .custom, the latter tagged "source": "user"
  remove_custom(name)             -- pops name from .custom / .vecs / .strength (mirrors
                                     SteeringControl.remove_custom)
  save_custom(path)               -- writes ONLY the "source": "user" entries. A load_library entry never
                                     carries a "source" key at all -- its dict shape is a small, separately-
                                     tested contract (see test_engine_library_dials.py's exact-equality
                                     check on `es.custom["eli5"]`, which a literal "source": "library" tag
                                     there would break for no functional gain). So the filter here is
                                     inverted from what you might expect: "include source=='user'", not
                                     "exclude source=='library'" -- a library dial is excluded by the
                                     ABSENCE of the tag. This is why load_library itself is untouched by
                                     this feature.
  load_custom(path)               -- round-trips a saved file back through add_custom (re-harvesting each
                                     dial), missing/broken file -> no-op, never raises

Model-free throughout: no live engine, no GPU. Mirrors test_engine_library_dials.py's fake-engine pattern
(_FakeEC/_FakeHarvest: a deterministic, text-dependent [n_tokens, 8] activation matrix per harvest call, so
two distinct pole prompts reliably produce a different, reproducible, non-degenerate direction).
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
import clozn.settings as clozn_settings          # noqa: E402


from clozn.behavior.steering import EngineSteer   # noqa: E402


# --- a stand-in for clozn_engine.EngineClient (identical to test_engine_library_dials.py's) ---------

class _FakeHarvest:
    """Stands in for engine.client.clozn_engine.Harvest: add_custom/compute() only read `.activations`
    ([n_tokens, n_embd]) via `.activations.mean(0)`."""

    def __init__(self, activations):
        self.activations = activations


class _FakeEC:
    """A stand-in engine client: .harvest returns a DETERMINISTIC, text-dependent [n_tokens, 8] activation
    matrix (seeded off a stable hash of the text) -- so the pos/neg poles of the SAME dial reliably produce
    a different, reproducible, non-degenerate diff-of-means direction, with no real model."""

    N_EMBD = 8
    N_TOKENS = 5

    def __init__(self):
        self.harvest_calls = []
        self.complete_calls = []
        self.intervene_calls = []

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


_SEEDS = ["hello", "tell me something", "what's up"]
_LIB_DIAL = {"eli5": {"pos": "Respond simply.", "neg": "Respond expertly.", "max": 1.5,
                      "source": "library", "category": "audience_level"}}


def _write_library(dirpath, dials=_LIB_DIAL):
    with open(os.path.join(str(dirpath), "studio_library.json"), "w", encoding="utf-8") as f:
        json.dump(dials, f)


# ==================================================================================== add_custom()

def test_add_custom_populates_vecs_and_custom_tagged_source_user():
    ec = _FakeEC()
    es = EngineSteer(ec)

    info = es.add_custom("upbeat", "Respond in an upbeat, energetic tone.",
                          "Respond in a flat, low-energy tone.", 0.7, seeds=_SEEDS)

    assert "upbeat" in es.vecs
    vec = es.vecs["upbeat"]
    assert vec.shape == (8,)
    assert np.isfinite(vec).all()
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5        # unit direction
    assert es.custom["upbeat"] == {"pos": "Respond in an upbeat, energetic tone.",
                                    "neg": "Respond in a flat, low-energy tone.",
                                    "max": 0.7, "poles": ["upbeat", "neutral"], "source": "user"}
    assert info is es.custom["upbeat"]                          # returns the stored entry
    assert len(ec.harvest_calls) == 2 * len(_SEEDS)             # pos + neg, once per seed -- nothing extra


def test_add_custom_name_is_stripped_and_truncated_to_24_chars():
    es = EngineSteer(_FakeEC())
    es.add_custom("  a-very-long-custom-dial-name-indeed  ", "pos text", "neg text", seeds=["hi"])
    (name,) = es.custom.keys()
    assert name == "a-very-long-custom-dial-name-indeed".strip()[:24]
    assert len(name) == 24


def test_add_custom_default_max_is_half():
    es = EngineSteer(_FakeEC())
    info = es.add_custom("plain", "pos text", "neg text", seeds=["hi"])
    assert info["max"] == 0.5


# ==================================================================================== remove_custom()

def test_remove_custom_pops_from_custom_vecs_and_strength():
    es = EngineSteer(_FakeEC())
    es.add_custom("mine", "pos text", "neg text", seeds=["hi"])
    es.set("mine", 0.3)
    assert "mine" in es.custom and "mine" in es.vecs and "mine" in es.strength

    es.remove_custom("mine")

    assert "mine" not in es.custom
    assert "mine" not in es.vecs
    assert "mine" not in es.strength


def test_remove_custom_on_an_unknown_name_is_a_noop():
    es = EngineSteer(_FakeEC())
    es.remove_custom("never-existed")          # must not raise
    assert es.custom == {}


# ==================================================================================== save_custom()

def test_save_custom_writes_only_the_user_dial_not_a_library_one(tmp_path):
    es = EngineSteer(_FakeEC())
    lib_path = tmp_path / "studio_library.json"
    lib_path.write_text(json.dumps(_LIB_DIAL), encoding="utf-8")
    es.load_library(str(lib_path))                          # a LIBRARY dial: metadata only, no "source" tag
    es.add_custom("mine", "Respond my way.", "Respond the other way.", 0.6, seeds=_SEEDS)   # a USER dial

    out_path = tmp_path / "studio_custom_engine.json"
    es.save_custom(str(out_path))

    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert set(saved) == {"mine"}                            # "eli5" (library) never leaked in
    assert saved["mine"] == {"pos": "Respond my way.", "neg": "Respond the other way.", "max": 0.6}


def test_save_custom_with_only_library_dials_writes_an_empty_file(tmp_path):
    es = EngineSteer(_FakeEC())
    lib_path = tmp_path / "studio_library.json"
    lib_path.write_text(json.dumps(_LIB_DIAL), encoding="utf-8")
    es.load_library(str(lib_path))

    out_path = tmp_path / "studio_custom_engine.json"
    es.save_custom(str(out_path))

    assert json.loads(out_path.read_text(encoding="utf-8")) == {}


# ==================================================================================== load_custom()

def test_load_custom_round_trips_a_saved_user_dial(tmp_path):
    saver = EngineSteer(_FakeEC())
    saver.add_custom("mine", "Respond my way.", "Respond the other way.", 0.6, seeds=_SEEDS)
    path = tmp_path / "studio_custom_engine.json"
    saver.save_custom(str(path))

    loader = EngineSteer(_FakeEC())                          # a FRESH instance, its own fake engine client
    loader.load_custom(str(path))

    assert "mine" in loader.custom
    assert loader.custom["mine"]["source"] == "user"          # round-trips through add_custom again
    assert loader.custom["mine"]["max"] == 0.6
    assert "mine" in loader.vecs
    assert loader.vecs["mine"].shape == (8,)


def test_load_custom_missing_file_is_a_noop(tmp_path):
    es = EngineSteer(_FakeEC())
    es.load_custom(str(tmp_path / "nope.json"))
    assert es.custom == {}


def test_load_custom_corrupt_file_is_a_noop_never_raises(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    es = EngineSteer(_FakeEC())
    es.load_custom(str(path))
    assert es.custom == {}


# ============================================================ clozn_server integration: /steer/custom(_delete)

@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite's server-level tests might touch, mirroring
    test_engine_library_dials.py's own `iso` fixture exactly."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def fake_engine(monkeypatch):
    """clozn_server.ENGINE -> a fresh _FakeEC; ENGINE_STEER reset so _engine_steer() builds a real
    steering.EngineSteer on it (construction + load_library make no harvest call; only compute() would)."""
    fe = _FakeEC()
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    return fe


def test_steer_custom_no_longer_errors_on_the_engine_substrate(iso, fake_engine):
    sub = cs.EngineSubstrate()

    result = sub.handle("/steer/custom", {"name": "mine", "pos": "Respond my way.",
                                          "neg": "Respond the other way.", "max": 0.6})

    assert "error" not in result
    assert result["name"] == "mine"
    assert result["max"] == 0.6
    assert "mine" in result["custom"]
    assert "mine" in sub.steer.custom
    assert sub.steer.custom["mine"]["source"] == "user"


def test_steer_custom_persisted_file_excludes_a_coexisting_library_dial(iso, fake_engine):
    _write_library(iso)
    sub = cs.EngineSubstrate()                   # loads the library dial's metadata at construction
    assert "eli5" in sub.steer.custom

    sub.handle("/steer/custom", {"name": "mine", "pos": "Respond my way.",
                                 "neg": "Respond the other way.", "max": 0.6})

    saved = json.loads((iso / "studio_custom_engine.json").read_text(encoding="utf-8"))
    assert set(saved) == {"mine"}                 # "eli5" excluded even though it's right there in .custom


def test_steer_custom_survives_a_restart(iso, fake_engine, monkeypatch):
    sub1 = cs.EngineSubstrate()
    sub1.handle("/steer/custom", {"name": "mine", "pos": "Respond my way.",
                                  "neg": "Respond the other way.", "max": 0.6})

    monkeypatch.setattr(cs, "ENGINE_STEER", None)   # force a FRESH EngineSteer, simulating a process restart
    sub2 = cs.EngineSubstrate()

    assert sub2.steer is not sub1.steer             # a genuinely new instance, not the cached one
    assert "mine" in sub2.steer.custom
    assert sub2.steer.custom["mine"]["source"] == "user"
    assert "mine" in sub2.steer.vecs                # load_custom re-harvests, not just metadata


def test_steer_custom_delete_removes_it_and_updates_the_persisted_file(iso, fake_engine):
    sub = cs.EngineSubstrate()
    sub.handle("/steer/custom", {"name": "mine", "pos": "Respond my way.",
                                 "neg": "Respond the other way.", "max": 0.6})

    result = sub.handle("/steer/custom_delete", {"name": "mine"})

    assert result == {"custom": []}
    assert "mine" not in sub.steer.custom
    saved = json.loads((iso / "studio_custom_engine.json").read_text(encoding="utf-8"))
    assert saved == {}


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
