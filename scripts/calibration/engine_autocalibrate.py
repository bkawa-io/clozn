"""dial_autocalibrate_engine.py -- per-dial usable-range calibration ON THE C++ ENGINE substrate
(clozn.behavior.steering.engine_adapter.EngineSteer), not the PyTorch/HF backbone dial_autocalibrate.py calibrates.

WHY THIS FILE EXISTS, SEPARATELY: the shipped calibration (research/dial_library_shipped.json's
`ship_range`, distilled by gen_dial_calibration.py into ~/.clozn/dial_calibration.json) was measured on
Qwen2.5-7B-Instruct nf4 THROUGH PYTORCH -- dial_autocalibrate.py's own docstring is explicit that a raw
number there ("SINGLE MODEL ONLY") does not transfer even to a different PyTorch quantization/layer. It
transfers even less to the C++ engine substrate: a llama.cpp-style engine commonly runs a DIFFERENT
quantization format from bitsandbytes nf4, and steering.EngineSteer's own base-scale calibration
(`base = 0.08*resid_norm`) is a different constant from SteeringControl's (`base = 0.85*resid_norm`),
confirming the two substrates do not share a residual-activation scale even nominally serving "the same"
checkpoint. So a dial's usable dose RANGE has to be re-measured, on the engine, from scratch -- this module
is that measurement, built to run with no torch and no HF model at all (only numpy + the engine's own
/harvest + /intervene HTTP surface via EngineSteer), so it can run on a deployment that never
installs PyTorch.

WHAT'S REUSED FROM dial_autocalibrate.py, AND WHY IT'S COPIED HERE INSTEAD OF IMPORTED: that module does
`import torch` / `from transformers import ...` at MODULE SCOPE, so a bare `import dial_autocalibrate`
already requires a working torch+transformers install -- exactly what an engine-only deployment doesn't
have, and exactly what this module's own import must stay free of (`python -c "import
dial_autocalibrate_engine"` must succeed with no torch/model on the path). Python has no way to import a
single name out of a module without first executing that module's top-level code, so anything reused here
that LIVES in dial_autocalibrate.py (or in the HF steering adapter, which imports torch) is
COPIED -- verbatim where the number/logic matters -- not imported. This mirrors this codebase's own
precedent (dial_autocalibrate.wants_four_bit: "copied verbatim from parliament.py ... not imported: this
codebase's own precedent is that each experiment script owns its small ... helpers rather than importing a
sibling script"). `clozn.behavior.steering.axes.AXES`/`SEED_PROMPTS` are still read live where truly needed (the
built-in/shadow-collision handling below), via a LAZY import inside the one function that needs them --
deferred to call time, never to bare-import time; see `_resync_shadowed_directions`. What IS imported at
module scope: `counterfactual._coherence` and `runlog` -- both confirmed torch-free by their own docstrings
("no torch import, no model, no GPU") -- so the mandatory coherence gate is reused for real, never
reinvented.

THE EFFECT MEASURE, PORTED: dial_autocalibrate.directional_alignment projects a re-encoded reply's
mean-pooled hidden state (one extra forward pass) onto the dial's OWN unit diff-of-means direction
(sc.vecs[dial]) -- a raw, UN-normalized dot product (only the DIRECTION is unit-normalized; the pooled
reply vector itself never is -- see that module's `_project_onto_unit`). `engine_alignment` below is the
identical recipe with one HTTP /harvest round-trip standing in for the forward pass: mean-pool
`engine_client.harvest(reply_text, layer=steer.layer).activations` over token positions, then the SAME raw
dot product against unit(steer.vecs[dial]). Deliberately NOT a cosine similarity (dividing by the pooled
vector's own norm too, which would be a materially different, scale-INVARIANT metric) -- see
`engine_alignment`'s own docstring.

WHY THE THRESHOLD CANNOT BE THE SAME NUMBER: dial_autocalibrate._EFFECT_EPS=2.0 is explicitly, loudly NOT a
portable constant even across two PyTorch runs of a different model/layer/quantization (see that module's
THRESHOLDS section) -- it was picked as roughly 1.74x a rough noise-floor estimate
(resid_norm/sqrt(hidden_size)) for ONE specific run (Qwen2.5-7B nf4, layer 14: resid_norm 68.7, hidden_size
3584). The engine's own resid_norm/embedding-size are essentially certain to differ (a different backbone
or quantization is the whole point of the engine substrate, AND EngineSteer.compute() computes resid_norm
from a differently-shaped quantity than SteeringControl.compute() does -- see `engine_effect_eps`'s own
docstring), so reusing 2.0 verbatim would silently over- or under-gate every dial here. `engine_effect_eps`
reruns the SAME derivation (noise floor x the SAME ~1.74 ratio) against THIS steer's own calibrated
resid_norm/embedding size, never hand-waves the old absolute number in. Still an eyeballed heuristic, not a
law -- see that function's docstring for the honest caveat.

THE SWEEP CEILING IS ALSO NOT THE OLD SHIPPED RANGE: `_ENGINE_SWEEP_MAX` (a fresh, generous per-dial ceiling
for the engine sweep) is used instead of dial_library_shipped.json's own curated `ship_range` top --
re-using `ship_range` as the ceiling would silently bias the "fresh" engine measurement right back toward
the PyTorch finding this module exists to stop assuming transfers. `ship_range` is still carried forward
into every report, under `pytorch_ship_range`, purely so a human can compare the two side by side.

A REAL ENGINESTEER GOTCHA THIS MODULE WORKS AROUND (read before trusting a "shadowed" dial's number): 6 of
the 33 shipped dials (warm, playful, formal, concise, poetic, concrete) share a NAME with a steering.AXES
built-in, but the shipped library's own (pos, neg) wording for that name usually differs from the built-in's
(compare dial_library_shipped.json's "warm" pair against steering.AXES["warm"]'s). On the PyTorch side this
is harmless: SteeringControl.add_custom unconditionally overwrites sc.vecs[name], so "the library's own
pos/neg wins" for free (see dial_autocalibrate.register_library_dials's own docstring). EngineSteer's
two-phase load/compute split does NOT give this for free: EngineSteer.compute() always harvests every
built-in AXES direction FIRST (unconditionally, every call), and its custom-dial loop then SKIPS any name
ALREADY in `.vecs` -- so a shadowing shipped dial's own pos/neg would otherwise never get harvested at all,
and `.vecs[name]` would silently stay the BUILT-IN direction even though `.custom[name]` correctly holds the
shipped text. `_resync_shadowed_directions` (called by `calibrate_library_engine`) forces a fresh harvest
from the shipped pair for exactly these names, immediately after `steer.compute()` -- restoring the same
"library's own pair wins" contract the PyTorch side gets for free. Discovered by reading EngineSteer.compute
's exact loop order, not observed on a real run; --smoke on a live engine is what would actually confirm it.

Cost: one steer.compute() (built-ins + every non-shadowed custom: ~(10 + n_custom) x 18 seeds x 2 poles
harvest calls), plus ~2 x 18 extra harvest calls per shadowed name, THEN O(n_dials x n_doses x n_prompts x 2)
engine generations (real dose + shuffled-null dose; dose 0 is shared, one generation), PLUS one /harvest
call per generated reply for engine_alignment. --smoke (1-2 dials, 3 doses, 2 prompts) proves the wiring in
a handful of HTTP round-trips -- NOT a finding.

Run (needs a live clozn-server; see engine_steer_spike.py for the same connection convention):
    PY=/c/Users/brigi/src/clozn/.venv/Scripts/python.exe   (any numpy-having interpreter works too)
    $PY research/dial_autocalibrate_engine.py --smoke --port 8092 --layer 14 \\
        --out research/runs/dial_autocalibrate_engine_smoke.json
Full shipped-library sweep:
    $PY research/dial_autocalibrate_engine.py --port 8092 --layer 14 \\
        --out research/runs/dial_autocalibrate_engine.json
A subset:
    $PY research/dial_autocalibrate_engine.py --dials warm tender wry --port 8092
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # research/ on path -- so any other research-local sibling still resolves
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))   # repo root -- so `from clozn import ...` resolves (counterfactual/
#                            runlog/steering all moved into the clozn/ package).

from clozn.replay.counterfactual import _coherence   # noqa: E402  -- {"degenerate": bool, "reason": str}; torch-free.
import clozn.runs.store as runlog                           # noqa: E402  -- torch-free (see its own docstring).


# ================================================================================================ constants
# Copied verbatim from dial_autocalibrate.py (not imported -- see the module docstring for why): the sweep
# doses are a property of "how we probe a dial", not of which substrate reads the activations, so the SAME
# fractions-of-axis-max convention is used here for a curve a human can compare side by side against a
# PyTorch one.
_SWEEP_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
_SWEEP_FRACS_SMOKE = [0.0, 0.75, 1.5]

# Coherence-rate gate -- copied verbatim from dial_autocalibrate._DEGEN_THRESHOLD. UNCHANGED on purpose: this
# is a *rate* (fraction of sampled prompts that came back degenerate), dimensionless and substrate-agnostic --
# unlike the effect epsilon below, nothing about "how many replies collapsed into a repeat loop" depends on
# which backbone produced them, so this constant needed no re-derivation for the engine.
_DEGEN_THRESHOLD = 0.34

# A FRESH exploration ceiling for every shipped-library dial on the engine sweep -- deliberately NOT
# dial_library_shipped.json's own `ship_range` top (see the module docstring's THE SWEEP CEILING section).
# 1.5 mirrors dial_autocalibrate.py's OWN identical choice for its --library candidate sweep
# (_LIBRARY_DEFAULT_MAX) -- same value, same reasoning (explore the regime dials actually showed effect/
# derail in, don't let a prior finding cap the search before it starts), copied not imported.
_ENGINE_SWEEP_MAX = 1.5

# The ratio dial_autocalibrate.py's own _EFFECT_EPS=2.0 bears to ITS rough noise-floor estimate
# (resid_norm/sqrt(hidden_size)) for the ONE run it was eyeballed against (Qwen2.5-7B nf4, layer 14:
# resid_norm=68.7, hidden_size=3584 -- both numbers straight from that module's own THRESHOLDS section).
# Reused as a MULTIPLIER of the ratio, never as the raw 2.0 -- see engine_effect_eps for why re-deriving
# against THIS steer's own resid_norm/embedding size is the point, not an afterthought.
_PYTORCH_RESID_NORM = 68.7
_PYTORCH_HIDDEN = 3584
_PYTORCH_EFFECT_EPS = 2.0
_EFFECT_EPS_MULTIPLIER = _PYTORCH_EFFECT_EPS / (_PYTORCH_RESID_NORM / math.sqrt(_PYTORCH_HIDDEN))


# ================================================================================================ helpers
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def degenerate_rate(texts: list[str]) -> float:
    """Copied verbatim from dial_autocalibrate.py: the coherence gate is reused for real (via the module-
    level `_coherence` import), only the tiny averaging wrapper is duplicated."""
    return round(_mean(_coherence(t)["degenerate"] for t in texts), 3) if texts else 0.0


def _dial_seed(base_seed: int, name: str) -> int:
    """Copied verbatim from dial_autocalibrate.py -- pure integer arithmetic, zero deps even in the
    original, so this is a straight duplicate, not a port. Deterministic per-(run-seed, dial-name) seed for
    the shuffled-direction generator (position-weighted so an anagram-like name pair can't collide)."""
    name_val = sum((i + 1) * ord(c) for i, c in enumerate(name))
    return (int(base_seed) * 1_000_003 + name_val * 97 + 13) & 0xFFFFFFFF


def _make_shuffle_unit_vector(ref: np.ndarray, seed: int) -> np.ndarray:
    """numpy PORT of dial_autocalibrate.make_shuffle_unit_vector (that one builds a torch.Tensor -- this
    module never imports torch, so the RNG is numpy's RandomState instead of torch.Generator). Same recipe:
    a fresh random UNIT direction, same shape/dtype as `ref`, seeded reproducibly so a given --seed
    reproduces the same shuffled directions run to run. NOT bit-identical to the PyTorch module's own
    shuffled vectors even given "the same" seed (different RNG algorithms/bit-streams) -- that was never a
    goal: the two modules calibrate different backbones, so cross-module reproducibility of the NULL
    direction specifically buys nothing; determinism WITHIN one engine run is what matters here."""
    ref = np.asarray(ref)
    rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)
    v = rng.standard_normal(size=ref.shape).astype(ref.dtype, copy=False)
    n = float(np.linalg.norm(v))
    return v / (n + 1e-8)


def _axis_max(steer, dial: str) -> float:
    """Sweep ceiling for `dial`, in the SAME slider-unit convention EngineSteer.generate()/set() use --
    custom-first (a registered dial's own "max"), else a built-in steering.AXES max, else the generic 1.5.
    The IDENTICAL precedence chain dial_autocalibrate.axis_max_of() uses on the PyTorch rig (copied logic,
    not the number -- see that function's own docstring for why custom must win on a name collision: this
    module's register_shipped_dials always sets a fresh "max" on `.custom`, so that branch wins for every
    shipped dial; the AXES fallback only matters for a plain built-in calibrated directly, never via the
    shipped library). `steering.AXES` is imported LAZILY, inside this function, not at module top -- steering
    .py itself `import torch`s at module scope, and this module's whole point is staying importable with no
    torch on the path. By the time any caller actually HAS a live `steer` (an EngineSteer instance) to pass
    in here, `steering` (and torch) is already resident in the process anyway, so deferring costs nothing
    real."""
    custom = getattr(steer, "custom", {}) or {}
    entry = custom.get(dial)
    if entry and entry.get("max") is not None:
        return float(entry["max"])
    try:
        from clozn.behavior.steering.axes import AXES as _AXES
    except Exception:
        _AXES = {}
    return float((_AXES.get(dial) or {}).get("max", 1.5))


# ================================================================================================ the effect measure
def engine_alignment(engine_client, steer, dial: str, reply_text: str) -> float:
    """The engine analog of dial_autocalibrate.directional_alignment: does `reply_text`'s OWN
    representation, read back off the engine, sit further toward `dial`'s pole -- not just "does it look
    different" (a white-box projection, no LLM judge).

    RECIPE, MIRRORING _project_onto_unit EXACTLY: `acts = engine_client.harvest(reply_text,
    layer=steer.layer).activations` ([n_tokens, n_embd]); mean-pool over token positions (`v = acts.mean(
    axis=0)` -- identical to hs[0].mean(dim=0) on the PyTorch side); project onto unit(steer.vecs[dial])
    via a plain dot product. Deliberately, NOT ALSO dividing by ||v|| (which would make this a cosine
    similarity): dial_autocalibrate._project_onto_unit only ever unit-normalizes the DIRECTION, never the
    pooled vector being projected -- re-reading that function was the whole point of "mirror dial_
    autocalibrate's exact normalization" (see the module docstring), and a cosine similarity is a materially
    different, bounded, scale-invariant metric, not a mirror of it. The upshot: like the PyTorch measure,
    this returns a raw, UNBOUNDED dot product in the engine's own residual-activation units -- meaningless
    read in isolation, and NOT comparable in absolute magnitude to a PyTorch run's numbers (different
    backbone, different quantization, near-certainly a different resid_norm) -- always read as a DELTA
    against that same prompt's own dose-0 baseline alignment (see engine_directional_effect), exactly like
    the PyTorch measure.

    Empty/whitespace-only text, or an engine response with zero harvested tokens, -> 0.0 (nothing to
    project) -- mirrors directional_alignment's own empty-input guard."""
    text = (reply_text or "").strip()
    if not text:
        return 0.0
    acts = np.asarray(engine_client.harvest(text, layer=steer.layer).activations, dtype=np.float64)
    if acts.ndim != 2 or acts.shape[0] == 0:
        return 0.0
    pooled = acts.mean(axis=0)                                     # [n_embd] -- mean over token positions
    direction = np.asarray(steer.vecs[dial], dtype=np.float64)
    unit = direction / (float(np.linalg.norm(direction)) + 1e-8)   # defensive re-normalize (already unit)
    return float(np.dot(pooled, unit))


def engine_directional_effect(engine_client, steer, dial: str, baseline_texts: list[str],
                              steered_texts: list[str]) -> float:
    """The engine analog of dial_autocalibrate.directional_effect: mean over the prompt sample of
    [engine_alignment(steered) - engine_alignment(that SAME prompt's own dose-0 baseline)]. Positive = the
    steered reply sits further toward the pole than that prompt's own unsteered reply did; ~0 = no net
    directional movement; negative = moved toward the OPPOSITE pole (a real, reportable finding, never
    clamped away). Called with the SAME `dial` name for both the real-direction arm and the shuffled-
    direction null -- only `steered_texts` differs between the two calls a sweep makes per dose (see
    calibrate_engine_dial and dial_autocalibrate's own THE SHUFFLED NULL section for why)."""
    if not baseline_texts or not steered_texts or len(baseline_texts) != len(steered_texts):
        return 0.0
    vals = [engine_alignment(engine_client, steer, dial, s) - engine_alignment(engine_client, steer, dial, b)
            for b, s in zip(baseline_texts, steered_texts)]
    return round(_mean(vals), 4)


def engine_effect_eps(steer) -> float:
    """Re-derive an effect epsilon FOR THIS STEER, rather than reusing dial_autocalibrate._EFFECT_EPS=2.0
    verbatim (see the module docstring's WHY THE THRESHOLD CANNOT BE THE SAME NUMBER). Same rough-
    noise-floor-estimate approach that module's own THRESHOLDS section describes (an uncorrelated vector of
    norm `resid_norm` projected onto an arbitrary fixed unit direction in an `n_embd`-dim space lands, very
    roughly, around resid_norm/sqrt(n_embd)), scaled by the SAME ~1.74x margin that module picked between
    ITS eyeballed 2.0 and ITS OWN noise-floor estimate (_EFFECT_EPS_MULTIPLIER, derived from its own
    documented numbers, not re-guessed here).

    TWO STACKED HEURISTICS, STATED LOUD: (1) the noise-floor approximation itself assumes roughly isotropic
    high-dimensional geometry -- real activations are anisotropic, so this is "a rough sanity check and NOT
    a rigorous bound" on the engine exactly as much as it was on the PyTorch rig; (2) EngineSteer.resid_norm
    is computed from a DIFFERENTLY-shaped quantity than SteeringControl.resid_norm -- SteeringControl
    averages the norm of each individual (per-seed, per-token) residual BEFORE any pooling across seeds;
    EngineSteer averages activations over seeds first (and over tokens, via .mean(0) inside compute()'s own
    harvest calls) and only THEN takes ONE norm per pole per axis. Averaging first, then norming, tends to
    produce a SMALLER number than norming many individual samples then averaging (noise cancels before the
    norm is taken) -- so this function's noise-floor estimate could plausibly run systematically LOW versus
    what the PyTorch-side derivation intended, not just "differently scaled". Net: treat the returned value
    as a reasonable STARTING epsilon, not a settled one -- exactly like _EFFECT_EPS itself, re-eyeball it
    against a real engine run's own printed curve (main() prints it precisely so a human can do that).

    steer.resid_norm/steer.vecs must already be populated (i.e. steer.compute() already ran) -- an
    unconfigured steer (resid_norm==0.0) returns 0.0 rather than raising, so a caller sees an obviously-
    degenerate epsilon (everything nonzero "clears" it) rather than a crash; calibrate_engine_dial's own
    precondition check (dial in steer.vecs) is what actually guards the real call path against this."""
    vecs = list(steer.vecs.values())
    n_embd = int(np.asarray(vecs[0]).shape[0]) if vecs else 1
    resid_norm = float(getattr(steer, "resid_norm", 0.0) or 0.0)
    noise_floor = resid_norm / math.sqrt(max(n_embd, 1))
    return float(noise_floor * _EFFECT_EPS_MULTIPLIER)


# ================================================================================================ pure calibration math
def _compute_calibration(curve: list[dict], degen_threshold: float, effect_eps: float) -> dict:
    """Pure function, no engine, no I/O -- the engine-side mirror of dial_autocalibrate._compute_calibration
    (identical comparison logic; `degen_threshold`/`effect_eps` are passed in explicitly rather than read off
    module constants, since the effect epsilon is run-specific here, not a fixed law -- see
    engine_effect_eps). derail_point/dead_below/usable_max/usable_range/range_valid have the exact same
    meanings as the PyTorch function's docstring describes.

    ONE DELIBERATE SHAPE DIFFERENCE from the PyTorch original: when range_valid is False, this returns
    usable_max=None and usable_range=None (a bare None, not dial_autocalibrate's [dead_below, usable_max]
    list-with-Nones-inside) -- matching the "works:False -> range None" contract this module's callers were
    built to, and matching how ~/.clozn/dial_calibration.json's own consumer (clozn_server._with_calibration)
    already treats a None usable_max: "the axis's own already-declared max when usable_max itself is None (a
    dial swept but never found usable)". `dead_below` is still reported even when usable_max isn't (the
    dial-moved-but-never-beat-the-null case stays visible as its own field), just not folded into
    usable_range once the pair isn't jointly valid."""
    derail_point = next((c["frac"] for c in curve if c["real_degenerate_rate"] > degen_threshold), None)
    dead_below = next((c["frac"] for c in curve if c["frac"] > 0 and c["effect"] > effect_eps), None)
    usable_fracs = [c["frac"] for c in curve
                    if c["frac"] > 0
                    and c["real_degenerate_rate"] <= degen_threshold
                    and c["effect"] > effect_eps
                    and c["effect"] > c["shuffled_effect"]]
    usable_max = max(usable_fracs) if usable_fracs else None
    range_valid = dead_below is not None and usable_max is not None and dead_below <= usable_max
    return {
        "derail_point": derail_point,
        "dead_below": dead_below,
        "usable_max": usable_max if range_valid else None,
        "usable_range": [dead_below, usable_max] if range_valid else None,
        "range_valid": range_valid,
    }


# ================================================================================================ per-dial sweep
def calibrate_engine_dial(engine_client, steer, dial: str, prompts: list[str],
                          fracs: list[float] = _SWEEP_FRACS, seed: int = 0, max_new: int = 60,
                          degen_threshold: float = _DEGEN_THRESHOLD,
                          effect_eps: float | None = None) -> dict:
    """Sweep `dial` over `fracs` (each a fraction of _axis_max(steer, dial)) on `prompts`, against a
    matched-norm SHUFFLED-direction null at the IDENTICAL magnitude, at every dose -- the engine-side mirror
    of dial_autocalibrate.calibrate_dial, driven through steer.generate(prompt, strength={...}) instead of a
    forward hook. `steer` must already have a computed direction for `dial` (steer.compute() already ran,
    directly or via calibrate_library_engine) -- raises KeyError immediately, before any engine call,
    otherwise (fail fast and legibly rather than KeyError deep inside the sweep).

    STRENGTH IS PASSED DIRECTLY to steer.generate(..., strength={...}), never through steer.set(): .set()
    clamps into [-max, max], exactly the ceiling this sweep needs to go PAST (fracs run up to 1.5x axis_max
    by design, to find where a dial derails beyond its "safe" max) -- generate()'s own strength= argument,
    when given a dict, is used as-is with no clamping, so this works without any bypass hack the PyTorch
    module needed (SteeringControl.generate always reads self.strength, which .set() DOES clamp -- see that
    module's own calibrate_dial docstring for the direct-write workaround it needs instead).

    At frac=0.0: `strength={dial: 0.0}` makes EngineSteer.generate()'s own `active = {k: v for k, v in
    s.items() if v and ...}` filter drop it (0.0 is falsy) -- so this transparently takes the PLAIN
    completion path (no /intervene call at all), exactly "steering off", used as both the real and shuffled
    arm's shared baseline (reused as the fixed reference for engine_directional_effect at every other dose).

    At each nonzero frac: `steer.vecs["_shuf_tmp"]` is set to a shuffled unit direction (drawn once per
    dial, reused at every dose -- see _make_shuffle_unit_vector/_dial_seed) and cleaned up in a `finally`
    right after that dose's null-arm generations, so a raised exception mid-sweep can never leave a stray
    "_shuf_tmp" entry sitting in steer.vecs.

    Returns {dial, axis_max, effect_eps, usable_max, usable_range, derail_point, dead_below, works, per_dose,
    sample_replies, note}. `per_dose` rows share dial_autocalibrate's OWN curve field names
    (frac/strength/real_degenerate_rate/shuffled_degenerate_rate/effect/shuffled_effect) so a PyTorch curve
    and an engine curve can be diffed side by side by a human or a script. `works`/`usable_range` come
    straight from _compute_calibration (see its docstring for the "range is a bare None when not working"
    convention). `note` is a short, human-legible one-line verdict."""
    if dial not in steer.vecs:
        raise KeyError(f"{dial!r} has no computed direction on this steer yet -- call steer.compute() "
                       f"(or calibrate_library_engine, which does this for every shipped dial) first")
    eps = float(effect_eps) if effect_eps is not None else engine_effect_eps(steer)
    axis_max = _axis_max(steer, dial)
    shuffle_vec = _make_shuffle_unit_vector(steer.vecs[dial], _dial_seed(seed, dial))

    baseline_texts: list[str] | None = None
    curve: list[dict] = []
    sample_replies: list[dict] = []
    for frac in fracs:
        strength = round(float(frac) * axis_max, 4)
        if frac == 0.0:
            real_texts = [steer.generate(p, strength={dial: 0.0}, max_new=max_new) for p in prompts]
            baseline_texts = real_texts
            shuf_texts = real_texts     # steering off either way at frac=0 -- identical by construction
        else:
            real_texts = [steer.generate(p, strength={dial: strength}, max_new=max_new) for p in prompts]
            steer.vecs["_shuf_tmp"] = shuffle_vec
            try:
                shuf_texts = [steer.generate(p, strength={"_shuf_tmp": strength}, max_new=max_new)
                             for p in prompts]
            finally:
                steer.vecs.pop("_shuf_tmp", None)

        curve.append({
            "frac": frac, "strength": strength,
            "real_degenerate_rate": degenerate_rate(real_texts),
            "shuffled_degenerate_rate": degenerate_rate(shuf_texts),
            "effect": engine_directional_effect(engine_client, steer, dial, baseline_texts, real_texts),
            "shuffled_effect": engine_directional_effect(engine_client, steer, dial, baseline_texts,
                                                         shuf_texts),
        })
        sample_replies.append({
            "frac": frac, "prompt": prompts[0] if prompts else "",
            "baseline_reply": baseline_texts[0] if baseline_texts else "",
            "steered_reply": real_texts[0] if real_texts else "",
        })

    calib = _compute_calibration(curve, degen_threshold, eps)
    works = bool(calib["range_valid"])
    note = (f"usable_range={calib['usable_range']} on {len(prompts)} prompt(s): clears effect_eps="
            f"{eps:.3g}, beats the shuffled null, stays coherent (<= {degen_threshold} degenerate rate)"
            if works else
            f"no clearing band on this sample (degen_threshold={degen_threshold}, effect_eps={eps:.3g}) -- "
            f"never simultaneously coherent + above-eps + beating the shuffled null")
    return {
        "dial": dial, "axis_max": axis_max, "effect_eps": round(eps, 4),
        "usable_max": calib["usable_max"], "usable_range": calib["usable_range"],
        "derail_point": calib["derail_point"], "dead_below": calib["dead_below"], "works": works,
        "per_dose": curve, "sample_replies": sample_replies, "note": note,
    }


# ================================================================================================ the shipped library
def load_shipped_library(path: str) -> list[dict]:
    """Pure I/O, no engine/model: load research/dial_library_shipped.json's {"dials": [{"name","category",
    "pos","neg","ship_range","note"}, ...]} shape. NOT dial_library_candidates.json's shape (which also
    carries a "predict" field this file never has) and NOT studio_library.json's flat {name: {pos,neg,max,
    source,category}} shape that EngineSteer.load_library reads -- three different shapes live in this
    codebase; easy to conflate, so this loader is deliberately its own function rather than a reuse of
    either. Raises ValueError, before any engine call, on a structurally broken file or a duplicate dial
    name -- the same fail-fast discipline dial_autocalibrate.load_dial_library uses (logic mirrored, not
    imported; required-field set adjusted since "predict" is a candidates-only field)."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dials = data.get("dials") if isinstance(data, dict) else None
    if not isinstance(dials, list) or not dials:
        got = type(dials).__name__ if dials is not None else type(data).__name__
        raise ValueError(f"{path}: expected a top-level {{'dials': [...]}} non-empty list, got {got}")
    required = ("name", "category", "pos", "neg")
    for i, d in enumerate(dials):
        if not isinstance(d, dict):
            raise ValueError(f"{path}: dials[{i}] is not an object: {d!r}")
        missing = [k for k in required if k not in d]
        if missing:
            raise ValueError(f"{path}: dials[{i}] missing required field(s) {missing}: {d}")
    names = [d["name"] for d in dials]
    dupes = sorted({n for n in names if names.count(n) > 1})
    if dupes:
        raise ValueError(f"{path}: duplicate dial name(s) in library: {dupes}")
    return dials


def register_shipped_dials(steer, library: list[dict]) -> dict:
    """Register every shipped dial spec into steer.custom -- the SAME {pos,neg,max,poles} shape
    EngineSteer.load_library/add_custom produce, so /steer/axes and steer.generate() treat these exactly
    like any other custom dial -- at a FRESH _ENGINE_SWEEP_MAX ceiling, never the old PyTorch `ship_range`
    (see the module docstring's THE SWEEP CEILING section). Returns {name: {"category", "pos", "neg",
    "pytorch_ship_range"}}, carried forward onto that dial's calibration report so a human can compare the
    fresh engine-measured range against the old PyTorch one without re-reading the library file again.
    Idempotent: re-registering a name just overwrites the same fields with the same values. Does NOT harvest
    directions itself (metadata only, matching EngineSteer.load_library's own two-phase split) -- the
    caller (calibrate_library_engine) still has to get steer.compute() (and, for a shadowing name,
    _resync_shadowed_directions) to run afterward."""
    out = {}
    for spec in library:
        name = spec["name"]
        steer.custom[name] = {"pos": spec["pos"], "neg": spec["neg"], "max": _ENGINE_SWEEP_MAX,
                              "poles": [name, "neutral"]}
        out[name] = {"category": spec.get("category", "?"), "pos": spec["pos"], "neg": spec["neg"],
                    "pytorch_ship_range": spec.get("ship_range")}
    return out


def _harvest_direction(steer, pos_instruction: str, neg_instruction: str, seeds) -> np.ndarray:
    """Manual diff-of-means harvest for one (pos, neg) pair -- the IDENTICAL recipe EngineSteer.compute()'s
    own loops use (both the built-in AXES loop and the custom-dial loop), just callable directly rather than
    reading pos/neg off self.custom[name]. The one caller that needs this (_resync_shadowed_directions)
    explains why steer.compute() alone can't be trusted to harvest a shadowing shipped dial's OWN pair."""
    pos = np.mean([steer.ec.harvest(pos_instruction + "\n\n" + s, layer=steer.layer).activations.mean(0)
                   for s in seeds], axis=0)
    neg = np.mean([steer.ec.harvest(neg_instruction + "\n\n" + s, layer=steer.layer).activations.mean(0)
                   for s in seeds], axis=0)
    d = pos - neg
    return d / (float(np.linalg.norm(d)) + 1e-8)


def _resync_shadowed_directions(steer, lib_meta: dict, axes: dict | None = None,
                                seeds: list | None = None) -> list:
    """For every registered shipped-dial NAME that also exists as a steering.AXES built-in (warm, playful,
    formal, concise, poetic, concrete -- see the module docstring's A REAL ENGINESTEER GOTCHA section), force
    a fresh diff-of-means harvest from the SHIPPED library's OWN (pos, neg) pair and overwrite
    steer.vecs[name] with it. Needed because EngineSteer.compute()'s built-in-AXES loop runs UNCONDITIONALLY
    before its custom-dial loop on every call (always (re)writing steer.vecs[name] from the built-in AXES for
    every built-in name), and its custom loop then SKIPS any name already in .vecs -- so a shadowing shipped
    dial's own pair would otherwise never get harvested, and .vecs[name] would silently stay the BUILT-IN
    direction. This restores the same "the library's own pair wins the collision, unconditionally" contract
    dial_autocalibrate.register_library_dials documents for the PyTorch side (there, add_custom's
    unconditional overwrite already gives it for free).

    `axes`/`seeds` default to clozn.behavior.steering.axes.AXES/SEED_PROMPTS, imported LAZILY inside this
    function so this module stays importable without torch on the path -- both are still overridable
    directly, which is what makes this unit-testable without touching real steering state.

    Returns the sorted list of names actually resynced (empty if this library has no built-in collisions, or
    if the axes module couldn't be imported at all -- silently a no-op in that case, mirroring _axis_max's own
    defensive fallback, never a raise: a missing steering install should not break a shipped-library sweep
    over dials that don't happen to collide with anything)."""
    if axes is None or seeds is None:
        try:
            import clozn.behavior.steering.axes as _steering_mod
        except Exception:
            return []
        axes = _steering_mod.AXES if axes is None else axes
        seeds = _steering_mod.SEED_PROMPTS if seeds is None else seeds
    shadowed = sorted(n for n in lib_meta if n in axes)
    for name in shadowed:
        meta = lib_meta[name]
        steer.vecs[name] = _harvest_direction(steer, meta["pos"], meta["neg"], seeds)
    return shadowed


def calibrate_library_engine(engine_client, steer, shipped_json_path: str, prompts: list[str],
                             fracs: list[float] = _SWEEP_FRACS, seed: int = 0, max_new: int = 60,
                             dial_names: list | None = None, checkpoint_path: str | None = None) -> dict:
    """Load dial_library_shipped.json, register every entry (or just `dial_names`, if given) onto `steer`,
    ensure `steer` has a computed direction for each (steer.compute() if any are missing, plus a forced
    resync for any that shadow a steering.AXES built-in -- see _resync_shadowed_directions), calibrate every
    one via calibrate_engine_dial, and return {name: {usable_max, usable_range, derail_point, works,
    category, pytorch_ship_range, note}} -- a superset of the SAME 4 keys (usable_max, usable_range,
    derail_point, works) clozn_server._dial_calibration() reads out of ~/.clozn/dial_calibration.json (extra
    keys are ignored by that reader, never a problem -- see gen_dial_calibration.py's own build_calibration
    for the same shape). A drop-in, engine-specific replacement for that file's contents.

    `checkpoint_path`, if given, is (re)written after EVERY dial finishes (json.dump, not append) -- so a
    kill/OOM/engine-disconnect partway through a big sweep keeps every already-finished dial's result on
    disk, mirroring dial_autocalibrate.run_library's own checkpoint discipline for its much bigger sweep."""
    library = load_shipped_library(shipped_json_path)
    if dial_names is not None:
        wanted = set(dial_names)
        library = [d for d in library if d["name"] in wanted]
        unknown = wanted - {d["name"] for d in library}
        if unknown:
            print(f"[warn] unknown shipped dial name(s) ignored: {sorted(unknown)}", flush=True)
    if not library:
        raise ValueError("no dials left to calibrate (empty library, or --dials matched nothing in it)")

    lib_meta = register_shipped_dials(steer, library)
    names = list(lib_meta)
    missing = [n for n in names if n not in steer.vecs]
    if missing or not getattr(steer, "ready", False):
        steer.compute()
    shadowed = _resync_shadowed_directions(steer, lib_meta)
    if shadowed:
        print(f"[warn] {len(shadowed)} shipped dial(s) shadow a steering.AXES built-in -- re-harvested from "
              f"the shipped library's OWN pos/neg pair (not the built-in's): {shadowed}", flush=True)
    eps = engine_effect_eps(steer)

    out: dict = {}
    for name in names:
        report = calibrate_engine_dial(engine_client, steer, name, prompts, fracs=fracs, seed=seed,
                                       max_new=max_new, effect_eps=eps)
        out[name] = {
            "usable_max": report["usable_max"], "usable_range": report["usable_range"],
            "derail_point": report["derail_point"], "works": report["works"],
            "category": lib_meta[name]["category"], "pytorch_ship_range": lib_meta[name]["pytorch_ship_range"],
            "note": report["note"],
        }
        if checkpoint_path:
            _save(checkpoint_path, out)
    return out


# ================================================================================================ prompt sample
# Copied verbatim from dial_autocalibrate.py (not imported -- see the module docstring for why): deliberately
# original text, disjoint from steering.SEED_PROMPTS (used only to compute the diff-of-means directions
# themselves), so evaluating on it is never circular.
NEUTRAL_PROMPTS = [
    "What's a good way to spend a rainy afternoon?",
    "Can you help me plan a small dinner party?",
    "I'm not sure what to do about a noisy neighbor.",
    "What should I keep in mind before starting a garden?",
    "Tell me about a topic you find interesting.",
    "I'm nervous about an upcoming presentation at work.",
    "What's the best way to organize a messy closet?",
    "How do I get better at sticking to a morning routine?",
    "My phone battery drains really fast lately, any ideas?",
    "What are some good conversation starters for a first date?",
]


def sample_prompts(n: int, seed: int = 0) -> tuple:
    """Copied verbatim from dial_autocalibrate.py (runlog is already imported torch-free at module scope
    here, so this is a straight duplicate, not a port): (prompts, source), pulling the n most recent DISTINCT
    user turns from runlog, else NEUTRAL_PROMPTS[:n] when the runlog is empty/unavailable. `seed` is accepted
    for interface symmetry but unused (sampling takes "the N most recent", nothing to seed)."""
    del seed  # unused -- see docstring
    try:
        rows = runlog.list_runs(limit=max(200, n * 20))
    except Exception:
        rows = []
    seen: set = set()
    prompts: list = []
    for row in rows:
        rid = row.get("id") if isinstance(row, dict) else None
        if not rid:
            continue
        rec = runlog.get_run(rid)
        if not rec:
            continue
        msgs = rec.get("messages") or []
        user_text = next((m.get("content", "") for m in reversed(msgs) if m.get("role") == "user"), "")
        user_text = (user_text or "").strip()
        if not user_text or user_text in seen:
            continue
        seen.add(user_text)
        prompts.append(user_text)
        if len(prompts) >= n:
            break
    if prompts:
        return prompts, "runlog"
    return list(NEUTRAL_PROMPTS[:n]), "neutral-fallback"


# ================================================================================================ CLI
def _save(path: str, res) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8092")))
    ap.add_argument("--layer", type=int, default=int(os.environ.get("LAYER", "14")))
    ap.add_argument("--shipped", default=os.path.join(HERE, "..", "..", "clozn", "data", "dial_library_shipped.json"),
                    help="path to the shipped dial library (default: clozn/data/dial_library_shipped.json)")
    ap.add_argument("--dials", nargs="+", default=None,
                    help="calibrate only these shipped dial names (default: all of them)")
    ap.add_argument("--n-prompts", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(HERE, "runs", "dial_autocalibrate_engine.json"))
    ap.add_argument("--smoke", action="store_true",
                    help="1-2 dials, 3 doses, 2 prompts -- prove the wiring cheaply, not a finding")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    sys.path.insert(0, os.path.join(HERE, "..", "..", "engine", "client"))
    from clozn_engine import EngineClient   # noqa: E402 -- deferred: keeps a bare module import client-free
    from clozn.behavior.steering.engine_adapter import EngineSteer        # noqa: E402 -- deferred: keeps bare import engine-client-free

    ec = EngineClient(host=args.host, port=args.port, timeout=240)
    print(f"[engine] {ec.health()}", flush=True)
    steer = EngineSteer(ec, layer=args.layer)

    dial_names = list(args.dials) if args.dials else None
    if args.smoke:
        if dial_names is None:
            dial_names = [d["name"] for d in load_shipped_library(args.shipped)]
        dial_names = dial_names[:2]
    n_eff = 2 if args.smoke else args.n_prompts
    fracs = _SWEEP_FRACS_SMOKE if args.smoke else _SWEEP_FRACS

    prompts, prompt_source = sample_prompts(n_eff, seed=args.seed)
    print(f"[prompts] {len(prompts)} prompt(s) from {prompt_source}", flush=True)
    print(f"[dials] calibrating: {dial_names or 'all shipped dials'}", flush=True)

    t0 = time.time()
    calib = calibrate_library_engine(ec, steer, args.shipped, prompts, fracs=fracs, seed=args.seed,
                                     max_new=args.max_new, dial_names=dial_names, checkpoint_path=args.out)
    dt = round(time.time() - t0, 1)
    _save(args.out, calib)

    print(f"\n[calib] resid_norm={steer.resid_norm:.1f} base={steer.base:.3f} "
          f"effect_eps={engine_effect_eps(steer):.4f} (engine-derived -- NOT dial_autocalibrate.py's 2.0; "
          f"re-eyeball against the curves in {args.out})", flush=True)
    print(f"\n{'dial':20} {'usable_range':16} {'works':6} pytorch_ship_range", flush=True)
    for name, v in calib.items():
        print(f"{name:20} {str(v['usable_range']):16} {str(v['works']):6} {v['pytorch_ship_range']}",
              flush=True)
    n_works = sum(1 for v in calib.values() if v["works"])
    print(f"\n{n_works}/{len(calib)} dial(s) got a usable range on THIS engine + prompt sample ({dt}s). "
          f"Compare usable_range against pytorch_ship_range above -- Law #6 predicts these will often "
          f"differ.\nsaved -> {args.out}", flush=True)
    return calib


if __name__ == "__main__":
    main()
