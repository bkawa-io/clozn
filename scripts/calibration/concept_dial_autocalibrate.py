"""concept_dial_autocalibrate.py -- per-model calibration for the any-concept dial (dir(c),
clozn/behavior/steering/concept_dir.py), against a LIVE clozn-server.

WHY THIS EXISTS: concept_dir.py's VALIDATED_MEDIAN_RESID_NORM ({16: 40.71, 21: 146.68, 25: 343.14}) and
VALIDATED_SCALE_RANGE (0.25, 0.5) are pinned to ONE exact model (Qwen2.5-7B-Instruct Q4_K_M) -- the same
"a raw number does not transfer even to a different quantization/layer" lesson
research/dial_autocalibrate_engine.py's own docstring documents for the diff-of-means TONE dials. This
script is the dir(c) equivalent of that module: it measures, on the model actually loaded on a running
engine, (a) this model's own median residual-row L2 norm at each fitted J-lens layer (the realistic `coef`
scale a UNIT dir(c) should be multiplied by), and (b) a usable dir(c) STRENGTH range at each layer -- the
scale band where logprob(c) rises cleanly over baseline AND generation stays coherent -- then writes both
under the SAME per-exact-GGUF ~/.clozn/models/<model_sha256>/ convention
clozn.server.app._model_scoped_path already uses for the tone-dial calibration file (dial_calibration.json),
via concept_dir.save_concept_dial_calibration.

WHAT'S MEASURED, PRECISELY:
  * median_resid_norm(layer): harvest a small sample of NEUTRAL prompts at `layer`
    (engine_client.harvest(text, layer=layer).activations, one causal forward per prompt -- see
    clozn_engine.EngineClient.harvest), take the L2 norm of every token row across every harvested
    prompt, and report the MEDIAN of that pooled sample -- the exact quantity
    ../clozn-jlens-work/scripts/run_j5a_swap.py's MEDIAN_NORM measured for the one model
    VALIDATED_MEDIAN_RESID_NORM is pinned to (see concept_dir.py's own docstring).
  * usable_scale_range(layer, concept): a small sweep over STRENGTH values (the same dimensionless
    "fraction of median_resid_norm" unit concept_dir.ConceptSteer.steer_toward already uses -- `coef =
    strength * median_norm`), each dose scored TWO ways on a sample of prompts:
      (1) logprob(c) rise: engine_client.score(prompt, continuation=" "+concept, steer_vec=dir(c),
          steer={"coef", "layer"}) MINUS the same call with no steer -- the forced logprob the model
          assigns the concept token itself, steered vs baseline (mirrors the live-checked "+6..9 nat over
          baseline logprob" finding concept_dir.py's own docstring cites for L21 scale 0.25-0.5).
      (2) coherence: engine_client.intervene(prompt, vector=dir(c), coef=coef, layer=layer,
          max_tokens=...) generates real continuation text, scored by
          clozn.replay.counterfactual._coherence (the SAME mandatory degenerate-output gate
          research/dial_autocalibrate_engine.py's own sweep uses -- copied import, not reinvented).
    `_compute_scale_calibration` (pure, no I/O) turns one dial's (scale, logprob_delta,
    degenerate_rate) curve into {usable_scale_range, derail_point, works} -- the ONLY part of this module
    unit-tested directly (tests/test_concept_dial_autocalibrate.py); everything that actually talks to an
    engine is deliberately DEFERRED from the test suite (needs a live clozn-server + a loaded J-lens
    sidecar + a real GPU-resident model), the same "live path is deferred, the pure math/schema is
    model-free tested" split this codebase already uses for quant-check and
    research/dial_autocalibrate_engine.py's own _compute_calibration.

Run (needs a live clozn-server with a J-lens sidecar loaded -- see engine_steer_spike.py for the same
connection convention concept_dir.py's own --demo path uses):
    PY=/c/Users/brigi/src/clozn/.venv/Scripts/python.exe   (any numpy-having interpreter works too)
    $PY scripts/calibration/concept_dial_autocalibrate.py --port 8095 --concept ocean
A subset of layers, or a quick smoke:
    $PY scripts/calibration/concept_dial_autocalibrate.py --port 8095 --layers 21 --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO_ROOT)   # so `from clozn import ...` resolves regardless of cwd

from clozn.behavior.steering.concept_dir import (   # noqa: E402
    ConceptDirSource, ConceptSteer, save_concept_dial_calibration,
)
from clozn.replay.counterfactual import _coherence   # noqa: E402 -- {"degenerate": bool, "reason": str}; torch-free.


# ================================================================================================ constants
# A dimensionless STRENGTH sweep in the SAME "fraction of median_resid_norm" unit
# concept_dir.ConceptSteer.steer_toward already uses (`coef = strength * median_norm`) -- deliberately NOT
# capped at VALIDATED_SCALE_RANGE's top (0.5): this sweep exists precisely to find THIS model's own usable
# band, which may sit somewhere else entirely (Law #6 -- never let a prior model's finding cap the search
# before it starts, mirroring research/dial_autocalibrate_engine.py's own _ENGINE_SWEEP_MAX rationale).
_SWEEP_SCALES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5]
_SWEEP_SCALES_SMOKE = [0.0, 0.35, 1.0]

# Coherence-rate gate -- copied verbatim from research/dial_autocalibrate_engine.py's _DEGEN_THRESHOLD
# (itself copied from dial_autocalibrate.py's _DEGEN_THRESHOLD): a dimensionless RATE, substrate-agnostic,
# so it needs no per-model re-derivation the way an absolute logprob/norm threshold would.
_DEGEN_THRESHOLD = 0.34

# The minimum steered-minus-baseline forced logprob(c) rise (nats) counted as "rising cleanly", loosely
# anchored to the LIVE-CHECKED "+6..9 nat over baseline" finding concept_dir.py's own docstring cites for
# the one validated model/layer/scale-range -- but treated here as a conservative FLOOR for THIS model's own
# sweep, not a reproduction target (a different model's own baseline/ceiling logprob geometry near-certainly
# differs). Re-eyeball against the printed curve for a specific model before trusting a borderline range.
_LOGPROB_RISE_MIN = 2.0

# Neutral prompt sample for the median-residual-norm harvest -- deliberately generic (not concept-specific):
# this measures the model's OWN typical residual scale at a layer, independent of which concept is later
# steered toward, mirroring VALIDATED_MEDIAN_RESID_NORM's own "typical residual magnitude" framing.
NEUTRAL_PROMPTS = [
    "The weather today is",
    "My favorite way to spend an afternoon is",
    "The most important thing to remember is",
    "When I think about the future, I",
    "A good story usually starts with",
    "One thing I learned recently is",
]


# ================================================================================================ pure math
def _mean(xs) -> float:
    xs = list(xs)
    return sum(xs) / len(xs) if xs else 0.0


def median_resid_norm_from_harvests(activations_list: list) -> float:
    """The MEDIAN L2 row-norm pooled across every harvested prompt's token activations -- pure numpy, no
    I/O, so this is unit-testable with plain synthetic arrays (tests/test_concept_dial_autocalibrate.py).
    `activations_list` is a list of [n_tokens_i, n_embd] arrays (one per harvested prompt, as returned by
    clozn_engine.Harvest.activations); rows across every prompt are pooled into ONE sample before taking the
    median -- exactly what VALIDATED_MEDIAN_RESID_NORM's own "measured over cached hf_hidden activations"
    provenance describes (concept_dir.py's module docstring). Empty input -> 0.0 (nothing measured), never
    a NaN/crash."""
    rows = []
    for acts in activations_list:
        arr = np.asarray(acts, dtype=np.float64)
        if arr.ndim == 2 and arr.shape[0] > 0:
            rows.extend(np.linalg.norm(arr, axis=1).tolist())
    return float(np.median(rows)) if rows else 0.0


def degenerate_rate(texts: list) -> float:
    """Copied convention from research/dial_autocalibrate_engine.py's own degenerate_rate: the coherence
    gate is reused for real (via the module-level `_coherence` import), only the averaging wrapper is
    duplicated."""
    return round(_mean(_coherence(t)["degenerate"] for t in texts), 3) if texts else 0.0


def _compute_scale_calibration(curve: list, degen_threshold: float, logprob_rise_min: float) -> dict:
    """Pure function, no engine, no I/O -- turns one (layer, concept)'s (scale, logprob_delta,
    degenerate_rate) sweep curve into {usable_scale_range, derail_point, works}. Mirrors research/
    dial_autocalibrate_engine.py's own _compute_calibration shape/semantics (same field names, same
    range-is-a-bare-None-when-not-working convention), adapted to a single-arm sweep (dir(c) already has a
    strong, separately-validated self-consistency proof -- concept_dir.py's own dir_c/read_through_lens
    tests -- so this sweep does not also need a shuffled-direction null the way a diff-of-means tone dial
    does; it is checking WHERE the effect/coherence trade-off lands for THIS model, not WHETHER dir(c) has
    directional content at all).

    `curve` rows: {"scale", "logprob_delta", "degenerate_rate"} (scale=0.0 is the baseline row and is
    never itself a candidate). derail_point: the first scale>0 whose degenerate_rate exceeds
    `degen_threshold`. usable_scales: every scale>0 that is BOTH below any derail point in coherence AND
    clears `logprob_rise_min`. usable_scale_range: [min, max] of usable_scales, or None when empty (never
    a bare number pretending to be a range)."""
    derail_point = next((c["scale"] for c in curve if c["scale"] > 0 and c["degenerate_rate"] > degen_threshold),
                        None)
    usable_scales = [
        c["scale"] for c in curve
        if c["scale"] > 0
        and c["degenerate_rate"] <= degen_threshold
        and c["logprob_delta"] >= logprob_rise_min
    ]
    usable_range = [min(usable_scales), max(usable_scales)] if usable_scales else None
    works = usable_range is not None
    return {"usable_scale_range": usable_range, "derail_point": derail_point, "works": works}


# ================================================================================================ live measurement
# Everything below talks to a real engine_client -- DEFERRED from this module's own test suite (needs a
# running clozn-server with a J-lens sidecar loaded and a real GPU-resident model); only the pure functions
# above are unit-tested.

def measure_median_resid_norm(engine_client, layer: int, prompts: list = NEUTRAL_PROMPTS) -> float:
    """LIVE: median_resid_norm_from_harvests over one /harvest round trip per prompt at `layer`."""
    acts = [engine_client.harvest(p, layer=layer).activations for p in prompts]
    return median_resid_norm_from_harvests(acts)


def measure_concept_scale_sweep(engine_client, steer: ConceptSteer, concept: str, layer: int,
                                scales: list, prompts: list, max_new: int = 40) -> list:
    """LIVE: one sweep curve for `concept` at `layer` on `steer` (a ConceptSteer already constructed with
    this model's OWN freshly-measured median_norm for `layer` -- see main()). At scale 0.0, logprob_delta
    and degenerate_rate are trivially 0.0/computed-from-unsteered-text (the shared baseline every other
    dose is compared against, mirroring research/dial_autocalibrate_engine.py's own dose-0 handling)."""
    curve = []
    for scale in scales:
        if scale == 0.0:
            baseline_texts = [
                _text_of(engine_client.complete(p, max_tokens=max_new)) for p in prompts
            ]
            curve.append({"scale": 0.0, "logprob_delta": 0.0,
                         "degenerate_rate": degenerate_rate(baseline_texts)})
            continue
        built = steer.steer_toward(concept, scale, layer=layer)
        if not built.get("ok"):
            curve.append({"scale": scale, "logprob_delta": 0.0, "degenerate_rate": 1.0,
                         "note": f"dir(c) unavailable at scale={scale}: {built.get('note')}"})
            continue
        vector, coef = built["vector"], built["coef"]
        deltas = []
        steered_texts = []
        for prompt in prompts:
            base = engine_client.score(prompt=prompt, continuation=" " + concept, topk=0)
            steered = engine_client.score(prompt=prompt, continuation=" " + concept, topk=0,
                                         steer_vec=vector, steer={"coef": coef, "layer": layer})
            base_lp = (base.get("tokens") or [{}])[0].get("logprob", 0.0)
            steered_lp = (steered.get("tokens") or [{}])[0].get("logprob", 0.0)
            deltas.append(float(steered_lp) - float(base_lp))
            resp = engine_client.intervene(prompt, vector=vector, coef=coef, layer=layer, max_tokens=max_new)
            steered_texts.append(_text_of(resp))
        curve.append({"scale": scale, "logprob_delta": round(_mean(deltas), 4),
                     "degenerate_rate": degenerate_rate(steered_texts)})
    return curve


def _text_of(resp) -> str:
    """Mirrors concept_dir._text_of / engine_adapter.EngineSteer._text -- extract generated text from an
    EngineClient .complete()/.intervene() response."""
    ch = resp.get("choices") if isinstance(resp, dict) else None
    if ch:
        return ch[0].get("text") or (ch[0].get("message") or {}).get("content") or ""
    return (resp.get("text") or "") if isinstance(resp, dict) else str(resp)


def calibrate_layer(engine_client, source: ConceptDirSource, layer: int, concept: str,
                    scales: list = _SWEEP_SCALES, prompts: list = NEUTRAL_PROMPTS, max_new: int = 40,
                    degen_threshold: float = _DEGEN_THRESHOLD,
                    logprob_rise_min: float = _LOGPROB_RISE_MIN) -> dict:
    """LIVE: the full per-layer measurement -- median_resid_norm, then a scale sweep using THAT freshly
    measured norm (an explicit `median_norm=` override on a throwaway ConceptSteer, never the global
    VALIDATED_MEDIAN_RESID_NORM table this whole script exists to stop assuming transfers). Returns
    {"layer", "median_resid_norm", "usable_scale_range", "derail_point", "works", "n_samples", "per_scale",
    "note"} -- a superset of the 4 keys concept_dir.save_concept_dial_calibration's own `layers` entries
    need."""
    median_norm = measure_median_resid_norm(engine_client, layer, prompts)
    steer = ConceptSteer(engine_client, source=source, layer=layer, median_norm=median_norm)
    curve = measure_concept_scale_sweep(engine_client, steer, concept, layer, scales, prompts, max_new=max_new)
    calib = _compute_scale_calibration(curve, degen_threshold, logprob_rise_min)
    note = (f"usable_scale_range={calib['usable_scale_range']} for concept={concept!r} at L{layer} on "
            f"{len(prompts)} prompt(s): clears logprob_rise_min={logprob_rise_min}, stays coherent "
            f"(<= {degen_threshold} degenerate rate)"
            if calib["works"] else
            f"no clearing band on this sample (concept={concept!r}, L{layer}) -- never simultaneously "
            f"coherent + above logprob_rise_min={logprob_rise_min}")
    return {"layer": layer, "median_resid_norm": round(median_norm, 4),
           "usable_scale_range": calib["usable_scale_range"], "derail_point": calib["derail_point"],
           "works": calib["works"], "n_samples": len(prompts), "per_scale": curve, "note": note}


# ================================================================================================ CLI
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8095")))
    ap.add_argument("--jlens-dir", default=None, help="default: ~/.clozn/jlens or CLOZN_JLENS_DIR")
    ap.add_argument("--unembed-dir", default=None, help="see concept_dir.BLOCKER_NOTE; usually unneeded "
                                                        "(the engine's /jlens/unembed_row is the default)")
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="fitted J-lens layers to calibrate (default: every layer the sidecar has)")
    ap.add_argument("--concept", default="ocean", help="probe concept word (must resolve to one token)")
    ap.add_argument("--n-prompts", type=int, default=len(NEUTRAL_PROMPTS))
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out", default=None, help="override the written path (default: the per-model "
                                                "~/.clozn/models/<sha256>/concept_dial_calibration.json)")
    ap.add_argument("--smoke", action="store_true",
                    help="1 layer (or the first requested), 3 scales, 2 prompts -- prove the wiring "
                        "cheaply, not a finding")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))
    from clozn_engine import EngineClient   # noqa: E402 -- deferred: keeps a bare module import client-free

    ec = EngineClient(host=args.host, port=args.port, timeout=240)
    health = ec.health()
    model_sha256 = health.get("model_sha256")
    print(f"[engine] model={health.get('model')} sha256={model_sha256}", flush=True)

    source = ConceptDirSource(jlens_dir=args.jlens_dir, unembed_dir=args.unembed_dir)
    layers = args.layers or source.available_layers()
    if not layers:
        print("[error] no fitted J-lens layers reported by the manifest -- nothing to calibrate",
              flush=True)
        return 1

    prompts = NEUTRAL_PROMPTS[:args.n_prompts]
    scales = _SWEEP_SCALES
    if args.smoke:
        layers = layers[:1]
        prompts = NEUTRAL_PROMPTS[:2]
        scales = _SWEEP_SCALES_SMOKE
    print(f"[layers] calibrating: {layers}", flush=True)

    t0 = time.time()
    out_layers = {}
    for layer in layers:
        report = calibrate_layer(ec, source, layer, args.concept, scales=scales, prompts=prompts,
                                 max_new=args.max_new)
        out_layers[layer] = report
        print(f"  L{layer}: median_resid_norm={report['median_resid_norm']} "
              f"usable_scale_range={report['usable_scale_range']} works={report['works']}", flush=True)
    dt = round(time.time() - t0, 1)

    dest = save_concept_dial_calibration(
        model_sha256, out_layers, path=args.out,
        note=f"concept_dial_autocalibrate.py: concept={args.concept!r}, {len(prompts)} prompt(s), "
             f"{len(scales)} scale(s), {dt}s",
    )
    n_works = sum(1 for v in out_layers.values() if v["works"])
    print(f"\n{n_works}/{len(out_layers)} layer(s) got a usable dir(c) scale range on THIS engine + prompt "
          f"sample ({dt}s).\nsaved -> {dest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
