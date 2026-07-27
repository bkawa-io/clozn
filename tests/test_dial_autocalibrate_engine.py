"""test_dial_autocalibrate_engine.py -- model-free tests for research/dial_autocalibrate_engine.py (the
ENGINE-side dial usable-range calibration -- see that module's own docstring for why it exists separately
from dial_autocalibrate.py's PyTorch rig).

No engine, no model, no GPU, no torch: every test drives the module through a FAKE engine client (_FakeEC,
a stand-in for clozn_engine.EngineClient) and a FAKE steer object (_FakeSteer, a stand-in for
steering.EngineSteer) -- NEITHER imports steering.py or clozn_engine.py, so this file exercises exactly the
same "numpy + a harvest/generate duck-type, no torch" surface dial_autocalibrate_engine.py itself promises
to run on.

_FakeEC.harvest is DETERMINISTIC and TEXT-DEPENDENT: it parses a "dial=<name> strength=<value>" marker
that _FakeSteer.generate embeds in its own canned replies, and returns activations whose mean-pooled vector
projects entirely onto e_0 -- scaled by the dose -- WHEN AND ONLY WHEN the text says
dial=<_FakeEC.RESPONSIVE_DIAL>; any OTHER dial name (a shuffled null's "_shuf_tmp" key, or a deliberately
inert "flat" test dial) gets an all-zero vector, regardless of dose. This is what lets engine_alignment's
real projection math be exercised end to end (never monkeypatched away, unlike dial_autocalibrate.py's own
directional_alignment in its sibling suite) while staying fully deterministic and fast. Test dial names
deliberately avoid steering.AXES' own built-in keys (warm/concise/formal/playful/curious/poetic/technical/
candid/confident/concrete) EXCEPT in the one test that specifically exercises the shadow-collision fix
(_resync_shadowed_directions), which always passes an explicit fake `axes=`/`seeds=` override anyway -- so
no test's outcome depends on whether a real `steering` (and therefore torch) happens to be importable in
whatever environment runs this file.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.join(os.path.dirname(HERE), "scripts", "calibration")   # engine_autocalibrate.py lives in scripts/calibration/
sys.path.insert(0, SCRIPTS)

import engine_autocalibrate as dae   # noqa: E402


# ================================================================================================
# fakes
# ================================================================================================
class _FakeHarvest:
    """Stands in for clozn_engine.Harvest: dial_autocalibrate_engine only ever reads `.activations`."""
    def __init__(self, activations):
        self.activations = activations


class _FakeEC:
    """Deterministic fake engine client. RESPONSIVE_DIAL is the ONLY dial name whose steered text moves the
    fake activation space at all -- any other name (a shuffled-null "_shuf_tmp", or a deliberately inert
    "flat" test dial) projects to an all-zero vector, so engine_alignment/engine_directional_effect see NO
    movement for it regardless of dose. `scale` is picked so the raw projection lands well clear of the
    small, explicitly-passed effect_eps values the tests below use."""
    N_TOKENS = 4
    RESPONSIVE_DIAL = "trdial"

    _STRENGTH_RE = re.compile(r"strength=(-?[\d.]+)")
    _DIAL_RE = re.compile(r"dial=(\S+)")

    def __init__(self, n_embd: int = 8, scale: float = 3.0):
        self.n_embd = n_embd
        self.scale = scale
        self.harvest_calls: list = []

    def harvest(self, text, layer=None):
        self.harvest_calls.append((text, layer))
        text = text or ""
        m_s, m_d = self._STRENGTH_RE.search(text), self._DIAL_RE.search(text)
        val = float(m_s.group(1)) if m_s else 0.0
        name = m_d.group(1) if m_d else None
        vec = np.zeros(self.n_embd, dtype=np.float64)
        if name == self.RESPONSIVE_DIAL:
            vec[0] = val * self.scale
        acts = np.tile(vec, (self.N_TOKENS, 1)).astype(np.float32)
        return _FakeHarvest(acts)


class _FakeSteer:
    """Stands in for steering.EngineSteer -- just the surface dial_autocalibrate_engine actually drives:
    .layer/.vecs/.custom/.resid_norm/.base/.ready/.ec, .compute(), .generate(prompt, strength=None,
    max_new=70). generate()'s active-dial filter (`v and k in self.vecs`) mirrors the real EngineSteer
    exactly -- a strength entry for a name with no computed direction is silently inert, same as the real
    thing."""
    def __init__(self, ec, layer: int = 5, resid_norm: float = 40.0):
        self.ec = ec
        self.layer = layer
        self.vecs: dict = {}
        self.custom: dict = {}
        self.strength: dict = {}
        self.resid_norm = resid_norm
        self.base = 1.0
        self.ready = False

    def compute(self, seeds=None):
        self.ready = True
        return {"resid_norm": self.resid_norm, "base": self.base, "axes": list(self.vecs)}

    def generate(self, prompt, strength=None, max_new=70):
        active = {k: v for k, v in (strength or {}).items() if v and k in self.vecs}
        if not active:
            return "baseline reply strength=0.0000 dial=none"
        name, value = next(iter(active.items()))
        return f"steered reply dial={name} strength={value:.4f} prompt={prompt[:16]}"


def _unit(v) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (float(np.linalg.norm(v)) + 1e-8)


def _seeded_steer(dial_names, ec=None, resid_norm: float = 40.0):
    """A steer whose .vecs/.custom are already populated for `dial_names` (steer.compute() need not run).
    Every name gets unit(e_0) as its direction -- correct for _FakeEC.RESPONSIVE_DIAL (whose harvested
    pooled vector really does land on e_0, scaled by dose) AND harmless for any other name (its pooled
    vector is always all-zero regardless of dose, so which axis it's projected onto is irrelevant -- a
    zero vector dotted with anything is zero)."""
    ec = ec or _FakeEC()
    steer = _FakeSteer(ec, resid_norm=resid_norm)
    e0 = [1.0] + [0.0] * 7
    for name in dial_names:
        steer.vecs[name] = _unit(e0)
        steer.custom[name] = {"max": 1.5, "pos": f"{name} pos", "neg": f"{name} neg",
                              "poles": [name, "neutral"]}
    steer.ready = True
    return ec, steer


# ================================================================================================
# engine_alignment / engine_directional_effect
# ================================================================================================
def test_engine_alignment_rises_with_dose():
    ec = _FakeEC()
    steer = _FakeSteer(ec)
    steer.vecs["trdial"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])

    doses = [0.0, 0.5, 1.0, 1.5]
    texts = [steer.generate("hi", strength={"trdial": d}) for d in doses]
    aligns = [dae.engine_alignment(ec, steer, "trdial", t) for t in texts]

    assert aligns == sorted(aligns)
    assert aligns[0] == pytest.approx(0.0, abs=1e-9)
    assert aligns[-1] > aligns[0]


def test_engine_alignment_only_responds_to_its_own_dial():
    """A push recorded under a DIFFERENT name (the shuffled null's usual key) must not register on
    "trdial"'s axis -- the property engine_directional_effect's shuffled-null comparison depends on."""
    ec = _FakeEC()
    steer = _FakeSteer(ec)
    steer.vecs["trdial"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    text = "steered reply dial=_shuf_tmp strength=1.5000 prompt=hi"
    assert dae.engine_alignment(ec, steer, "trdial", text) == pytest.approx(0.0, abs=1e-9)


def test_engine_alignment_empty_text_is_zero_and_never_calls_the_engine():
    ec = _FakeEC()
    steer = _FakeSteer(ec)
    steer.vecs["trdial"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    assert dae.engine_alignment(ec, steer, "trdial", "") == 0.0
    assert dae.engine_alignment(ec, steer, "trdial", "   ") == 0.0
    assert ec.harvest_calls == []


def test_engine_directional_effect_positive_for_the_responsive_dial():
    ec = _FakeEC()
    steer = _FakeSteer(ec)
    steer.vecs["trdial"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    baseline = [steer.generate("hi", strength={"trdial": 0.0})]
    steered = [steer.generate("hi", strength={"trdial": 1.0})]
    assert dae.engine_directional_effect(ec, steer, "trdial", baseline, steered) > 0


def test_engine_directional_effect_mismatched_lengths_is_zero():
    ec = _FakeEC()
    steer = _FakeSteer(ec)
    steer.vecs["trdial"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    assert dae.engine_directional_effect(ec, steer, "trdial", ["a"], ["a", "b"]) == 0.0


# ================================================================================================
# engine_effect_eps
# ================================================================================================
def test_engine_effect_eps_scales_with_resid_norm():
    small = _FakeSteer(_FakeEC(), resid_norm=10.0)
    big = _FakeSteer(_FakeEC(), resid_norm=100.0)
    small.vecs["x"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    big.vecs["x"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    assert dae.engine_effect_eps(big) > dae.engine_effect_eps(small) > 0


def test_engine_effect_eps_zero_when_uncalibrated():
    steer = _FakeSteer(_FakeEC(), resid_norm=0.0)
    steer.vecs["x"] = _unit([1, 0, 0, 0, 0, 0, 0, 0])
    assert dae.engine_effect_eps(steer) == 0.0


# ================================================================================================
# _dial_seed / _make_shuffle_unit_vector
# ================================================================================================
def test_dial_seed_deterministic():
    assert dae._dial_seed(0, "trdial") == dae._dial_seed(0, "trdial")


def test_dial_seed_distinct_per_name():
    assert dae._dial_seed(0, "trdial") != dae._dial_seed(0, "other")


def test_make_shuffle_unit_vector_unit_norm_and_shape():
    ref = np.zeros(8, dtype=np.float64)
    v = dae._make_shuffle_unit_vector(ref, 7)
    assert v.shape == ref.shape
    assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-6)


def test_make_shuffle_unit_vector_deterministic_given_same_seed():
    ref = np.zeros(8, dtype=np.float64)
    assert np.array_equal(dae._make_shuffle_unit_vector(ref, 5), dae._make_shuffle_unit_vector(ref, 5))


def test_make_shuffle_unit_vector_differs_across_seeds():
    ref = np.zeros(8, dtype=np.float64)
    assert not np.array_equal(dae._make_shuffle_unit_vector(ref, 1), dae._make_shuffle_unit_vector(ref, 2))


# ================================================================================================
# _axis_max
# ================================================================================================
def test_axis_max_custom_first():
    steer = _FakeSteer(_FakeEC())
    steer.custom["trdial"] = {"max": 0.75}
    assert dae._axis_max(steer, "trdial") == 0.75


def test_axis_max_default_when_nowhere_declared():
    steer = _FakeSteer(_FakeEC())
    assert dae._axis_max(steer, "totally-unknown-dial-name-xyz") == 1.5


# ================================================================================================
# _compute_calibration -- pure math, explicit (degen_threshold, effect_eps) rather than module constants
# ================================================================================================
def test_compute_calibration_normal_case():
    eps = 1.0
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 10, "shuffled_effect": eps * 1.5},
        {"frac": 1.0, "real_degenerate_rate": 0.1, "effect": eps * 20, "shuffled_effect": eps * 6},
        {"frac": 1.5, "real_degenerate_rate": 0.8, "effect": eps * 30, "shuffled_effect": eps * 10},
    ]
    out = dae._compute_calibration(curve, 0.34, eps)
    assert out["derail_point"] == 1.5
    assert out["dead_below"] == 0.5
    assert out["usable_max"] == 1.0
    assert out["usable_range"] == [0.5, 1.0]
    assert out["range_valid"] is True


def test_compute_calibration_flat_dial_never_works():
    curve = [{"frac": f, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0}
             for f in (0.0, 0.5, 1.0, 1.5)]
    out = dae._compute_calibration(curve, 0.34, 1.0)
    assert out["dead_below"] is None
    assert out["usable_max"] is None
    assert out["usable_range"] is None      # a bare None -- unlike dial_autocalibrate's [None, None]
    assert out["range_valid"] is False


def test_compute_calibration_effect_never_beats_shuffled_null():
    eps = 1.0
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 6, "shuffled_effect": eps * 8},
    ]
    out = dae._compute_calibration(curve, 0.34, eps)
    assert out["dead_below"] == 0.5      # effect alone is "real"...
    assert out["usable_max"] is None     # ...but never attributable to the dial's own direction
    assert out["usable_range"] is None
    assert out["range_valid"] is False


def test_compute_calibration_derail_excludes_the_top_dose():
    eps = 1.0
    curve = [
        {"frac": 0.0, "real_degenerate_rate": 0.0, "effect": 0.0, "shuffled_effect": 0.0},
        {"frac": 0.5, "real_degenerate_rate": 0.0, "effect": eps * 10, "shuffled_effect": eps * 1},
        {"frac": 1.0, "real_degenerate_rate": 1.0, "effect": eps * 25, "shuffled_effect": eps * 1},
    ]
    out = dae._compute_calibration(curve, 0.34, eps)
    assert out["usable_max"] == 0.5
    assert out["derail_point"] == 1.0


# ================================================================================================
# calibrate_engine_dial
# ================================================================================================
def test_calibrate_engine_dial_works_dial_gets_a_plausible_range():
    ec, steer = _seeded_steer(["trdial"])      # "trdial" IS _FakeEC.RESPONSIVE_DIAL
    prompts = ["hello there", "what is up"]
    report = dae.calibrate_engine_dial(ec, steer, "trdial", prompts, effect_eps=1.0)

    assert report["dial"] == "trdial"
    assert len(report["per_dose"]) == len(dae._SWEEP_FRACS)
    assert len(report["sample_replies"]) == len(dae._SWEEP_FRACS)
    assert report["works"] is True
    assert report["usable_range"] is not None
    lo, hi = report["usable_range"]
    assert lo is not None and hi is not None and lo <= hi
    # output shape matches dial_calibration.json's {usable_max, usable_range, works}:
    assert {"usable_max", "usable_range", "works"} <= set(report)


def test_calibrate_engine_dial_flat_dial_gets_works_false():
    ec, steer = _seeded_steer(["flat"])         # NOT _FakeEC.RESPONSIVE_DIAL -> never moves the fake activations
    prompts = ["hello there", "what is up"]
    report = dae.calibrate_engine_dial(ec, steer, "flat", prompts, effect_eps=1.0)

    assert report["works"] is False
    assert report["usable_range"] is None
    assert report["usable_max"] is None
    assert {"usable_max", "usable_range", "works"} <= set(report)


def test_calibrate_engine_dial_missing_direction_raises():
    ec, steer = _seeded_steer([])
    with pytest.raises(KeyError):
        dae.calibrate_engine_dial(ec, steer, "nope", ["hi"], effect_eps=1.0)


def test_calibrate_engine_dial_baseline_reused_at_dose_zero():
    ec, steer = _seeded_steer(["trdial"])
    report = dae.calibrate_engine_dial(ec, steer, "trdial", ["hi"], effect_eps=1.0)
    c0 = report["per_dose"][0]
    assert c0["frac"] == 0.0
    assert c0["effect"] == 0.0
    s0 = report["sample_replies"][0]
    assert s0["baseline_reply"] == s0["steered_reply"]


def test_calibrate_engine_dial_cleans_up_shuf_tmp():
    ec, steer = _seeded_steer(["trdial"])
    dae.calibrate_engine_dial(ec, steer, "trdial", ["hi"], effect_eps=1.0)
    assert "_shuf_tmp" not in steer.vecs


def test_calibrate_engine_dial_per_dose_field_names_match_pytorch_curve_rows():
    """So a PyTorch curve and an engine curve can be diffed side by side (see the module docstring)."""
    ec, steer = _seeded_steer(["trdial"])
    report = dae.calibrate_engine_dial(ec, steer, "trdial", ["hi"], effect_eps=1.0)
    expected = {"frac", "strength", "real_degenerate_rate", "shuffled_degenerate_rate", "effect",
               "shuffled_effect"}
    assert set(report["per_dose"][0]) == expected


# ================================================================================================
# load_shipped_library
# ================================================================================================
def test_load_shipped_library_valid_file(tmp_path):
    data = {"dials": [{"name": "a", "category": "c", "pos": "p", "neg": "n", "ship_range": [0.5, 1.0]}]}
    path = tmp_path / "shipped.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    dials = dae.load_shipped_library(str(path))
    assert len(dials) == 1 and dials[0]["name"] == "a"


def test_load_shipped_library_missing_field_raises(tmp_path):
    data = {"dials": [{"name": "a", "category": "c", "pos": "p"}]}    # no "neg"
    path = tmp_path / "shipped.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        dae.load_shipped_library(str(path))


def test_load_shipped_library_duplicate_name_raises(tmp_path):
    entry = {"name": "a", "category": "c", "pos": "p", "neg": "n"}
    path = tmp_path / "shipped.json"
    path.write_text(json.dumps({"dials": [entry, dict(entry)]}), encoding="utf-8")
    with pytest.raises(ValueError):
        dae.load_shipped_library(str(path))


def test_load_shipped_library_real_shipped_file_loads_cleanly():
    real_path = os.path.join(os.path.dirname(HERE), "clozn", "data", "dial_library_shipped.json")
    dials = dae.load_shipped_library(real_path)
    assert len(dials) == 33
    assert all({"name", "category", "pos", "neg"} <= set(d) for d in dials)


# ================================================================================================
# register_shipped_dials
# ================================================================================================
def test_register_shipped_dials_sets_a_fresh_max_not_ship_range():
    steer = _FakeSteer(_FakeEC())
    library = [{"name": "warm", "category": "affect_tone", "pos": "p", "neg": "n", "ship_range": [0.5, 0.5]}]
    meta = dae.register_shipped_dials(steer, library)
    assert steer.custom["warm"]["max"] == dae._ENGINE_SWEEP_MAX
    assert steer.custom["warm"]["max"] != 0.5
    assert meta["warm"]["pytorch_ship_range"] == [0.5, 0.5]


# ================================================================================================
# _resync_shadowed_directions -- the EngineSteer.compute() collision workaround
# ================================================================================================
def test_resync_shadowed_directions_forces_the_shipped_pair():
    ec, steer = _seeded_steer([])
    # Simulate a stale BUILT-IN AXES direction already sitting in .vecs for "warm" -- exactly what
    # EngineSteer.compute()'s unconditional built-in loop would have put there before its custom loop
    # ever got a chance to look (see the module docstring's A REAL ENGINESTEER GOTCHA section).
    steer.vecs["warm"] = _unit([0, 1, 0, 0, 0, 0, 0, 0])
    lib_meta = {"warm": {"pos": "warm pos", "neg": "warm neg", "category": "affect_tone",
                        "pytorch_ship_range": [0.5, 0.5]}}
    fake_axes = {"warm": {"pos": "builtin warm pos", "neg": "builtin warm neg"}}

    resynced = dae._resync_shadowed_directions(steer, lib_meta, axes=fake_axes,
                                               seeds=["seed one", "seed two"])

    assert resynced == ["warm"]
    assert not np.allclose(steer.vecs["warm"], _unit([0, 1, 0, 0, 0, 0, 0, 0]))


def test_resync_shadowed_directions_noop_when_nothing_shadows():
    ec, steer = _seeded_steer([])
    lib_meta = {"totally_novel_dial": {"pos": "p", "neg": "n", "category": "c", "pytorch_ship_range": None}}
    resynced = dae._resync_shadowed_directions(steer, lib_meta, axes={"warm": {}}, seeds=["s"])
    assert resynced == []


# ================================================================================================
# calibrate_library_engine
# ================================================================================================
def test_calibrate_library_engine_output_shape_matches_dial_calibration_json(tmp_path):
    library = [
        {"name": "trdial", "category": "affect_tone", "pos": "p1", "neg": "n1", "ship_range": [0.5, 0.5]},
        {"name": "flat", "category": "other", "pos": "p2", "neg": "n2", "ship_range": [0.25, 1.0]},
    ]
    ec = _FakeEC()

    class _ComputingFakeSteer(_FakeSteer):
        """.compute() populates unit(e_0) for any registered custom name not yet in .vecs -- standing in for
        EngineSteer's real harvest recipe (already covered directly by the engine_alignment tests above);
        this test is about calibrate_library_engine's own orchestration, not re-proving the harvest math."""
        def compute(self, seeds=None):
            e0 = [1.0] + [0.0] * 7
            for name in self.custom:
                if name not in self.vecs:
                    self.vecs[name] = _unit(e0)
            self.ready = True
            return super().compute(seeds)

    # resid_norm picked small on purpose: engine_effect_eps derives eps from it, and this fake's dose scale
    # (see _FakeEC.scale) needs eps comfortably below its lowest nonzero-dose effect for "trdial" to clear it.
    steer = _ComputingFakeSteer(ec, resid_norm=1.0)

    shipped_path = tmp_path / "shipped.json"
    shipped_path.write_text(json.dumps({"dials": library}), encoding="utf-8")
    checkpoint = tmp_path / "calib.json"

    calib = dae.calibrate_library_engine(ec, steer, str(shipped_path), ["hello", "hi there"],
                                         checkpoint_path=str(checkpoint))

    assert set(calib) == {"trdial", "flat"}
    for name, entry in calib.items():
        assert {"usable_max", "usable_range", "works"} <= set(entry)
        assert "pytorch_ship_range" in entry and "category" in entry
    assert calib["trdial"]["works"] is True
    assert calib["flat"]["works"] is False

    assert checkpoint.is_file()          # checkpointed after every dial
    on_disk = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert set(on_disk) == {"trdial", "flat"}


def test_calibrate_library_engine_dial_names_filters_the_library(tmp_path):
    library = [
        {"name": "trdial", "category": "affect_tone", "pos": "p1", "neg": "n1", "ship_range": [0.5, 0.5]},
        {"name": "flat", "category": "other", "pos": "p2", "neg": "n2", "ship_range": [0.25, 1.0]},
    ]
    ec, steer = _seeded_steer(["trdial", "flat"])
    shipped_path = tmp_path / "shipped.json"
    shipped_path.write_text(json.dumps({"dials": library}), encoding="utf-8")

    calib = dae.calibrate_library_engine(ec, steer, str(shipped_path), ["hi"], dial_names=["trdial"])
    assert set(calib) == {"trdial"}


def test_calibrate_library_engine_empty_selection_raises(tmp_path):
    library = [{"name": "trdial", "category": "affect_tone", "pos": "p1", "neg": "n1",
               "ship_range": [0.5, 0.5]}]
    ec, steer = _seeded_steer(["trdial"])
    shipped_path = tmp_path / "shipped.json"
    shipped_path.write_text(json.dumps({"dials": library}), encoding="utf-8")

    with pytest.raises(ValueError):
        dae.calibrate_library_engine(ec, steer, str(shipped_path), ["hi"], dial_names=["nope"])
