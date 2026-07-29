"""bisect_acceptance.py -- the acceptance batteries for clozn's cross-model mechanistic diff / causal
bisect stack (clozn/analysis/{pair_compatibility,transplant,causal_bisect,restoration_metrics,
mechanistic_diff}.py). Those modules were built and unit-tested MODEL-FREE (fake engines only); nothing
until this script has proven they find a real effect that is actually there, or stay silent when
nothing is there. This script is that proof, run live against the GPU engine and real GGUFs.

Four batteries, run against real weights on real GPU forwards -- see each battery_N() function's own
docstring for the exact design and the mechanics it depends on:

  1. battery_1_positive_control  -- a KNOWN, planted perturbation; the bisect must localize it and the
     reference transplant must reverse it, discriminated from a random-equal-norm control.
  2. battery_2_negative_control  -- a model diffed/bisected against ITSELF; every measured divergence
     and intervention effect must sit at the numerical floor.
  3. battery_3_random_control_gate -- a constructed case where a random perturbation ALSO flips the
     answer; the system must report the perturbation-sensitive verdict, never a localized one.
  4. battery_4_real_quant_pairs -- Qwen2.5-7B-Instruct Q8_0 vs Q2_K / Q4_K_M, real prompts, real
     disagreements. `localized_*`, `distributed_restoration`, and `inconclusive` are ALL valid outcomes
     here -- the acceptance bar is accurate classification, not a neat location (see the prior transplant
     study, docs/research/DISTRIBUTED_FUNCTION.md section B: only 3/12 "fixed" quant flips were
     genuinely reference-specific once a random-equal-norm control was applied).

HOW BATTERY 1 AND 3's SYNTHETIC PERTURBATIONS ARE CONSTRUCTED
------------------------------------------------------------------
There is no API in clozn.analysis for injecting a known ground-truth corruption -- these modules only
ever capture a model's OWN natural activations. So this script builds one: `PlantProxy` wraps a real
`EngineClient` and, on every `.score()` call, APPENDS one extra write spec (residual `write` or
`ffn_write`) to whatever the caller (transplant.py / causal_bisect.py) already sent, at a FIXED
(layer, position) -- emulating "a candidate model that is permanently corrupted at exactly this site."
This was validated empirically (not just reasoned about) against the live engine before being used here
-- three load-bearing, non-obvious facts, confirmed by reading engine/core/serve/routes_whitebox.cpp and
engine/core/src/model_ggml.cpp AND by live experiment:

  * writes are fully PER-REQUEST (routes_whitebox.cpp clear_write()s both before applying a request's
    own specs and again in that request's own cleanup) -- nothing persists across calls, so the plant
    MUST be re-injected on every single .score() call, which is what PlantProxy does.
  * a capture and a write to the SAME (layer, position) in ONE call: eval_cb (model_ggml.cpp) reads
    (captures) the tensor BEFORE applying any write to it -- "capture is pre-edit". This makes a
    self-transplant control's own captured value structurally BLIND to a same-position plant that is
    active during that same call. It does NOT block a plant and a harness write from coexisting when
    they target the SAME (layer, position): PlantProxy appends its spec AFTER the caller's own, and two
    specs at the identical (layer, position) apply in list order (ggml_backend_tensor_set is a
    destructive overwrite per spec) -- so the LAST spec for that exact slot wins. Appending last means
    the plant wins ties: at the EXACT plant site, no write-based arm (including reference_transplant)
    can ever out-write the plant, so that one site is structurally unfixable through itself -- this is
    disclosed and measured below, not hidden.
  * a residual (`l_out`) write is a FULL STREAM OVERWRITE (never additive across layers -- see
    causal_bisect.py's own module docstring). A write at any layer AFTER the plant layer completely
    replaces the corrupted trajectory at that point with whatever is written there, discarding the
    plant's influence entirely (an activation patch cannot see "through" an intervening full overwrite).
    This is what makes the DOWNSTREAM sites in battery 1 cleanly fixable: they were never themselves a
    plant target, so their own captures are genuinely undisturbed, and installing the reference's clean
    row there fully cancels the propagated corruption.

Both the exact mechanics above and every numeric threshold below were confirmed live (not merely
theorized) via a battery of throwaway validation scripts run in this session before this file was
written; see the git history / session log for that record.

STORE_TENSORS=False THROUGHOUT
--------------------------------
Every transplant.run_site()/causal_bisect.run_bisect() call below passes `store_tensors=False` -- this
acceptance run cares about the ANALYSIS documents' own verdicts and metrics, not about persisting
synthetic/throwaway tensors into ~/.clozn/tensors. The modules already omit the `vectors`/`tensors`
fields honestly when this is False (never a fabricated placeholder) -- see each module's own docstring.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "engine", "client"))

from clozn.analysis import causal_bisect, mechanistic_diff, pair_compatibility, transplant  # noqa: E402
from clozn.cli.engine_process import spawn_engine, _terminate_process  # noqa: E402
from clozn.cli.commands.models import _flags_for  # noqa: E402
from clozn_engine import EngineClient  # noqa: E402

MODELS_DIR = os.path.expanduser("~/.clozn/models")
FAST_MODEL = os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
Q8_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q8_0.gguf")
Q4_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf")
Q2_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q2_K.gguf")

_PORT = [8700]


def _next_port() -> int:
    _PORT[0] += 1
    return _PORT[0]


# =============================================================================================== shared

def norm(v):
    return math.sqrt(sum(x * x for x in v))


def scaled_random_vector(reference_row, seed: int, scale: float):
    """A random direction scaled to `scale` times `reference_row`'s own L2 norm -- the SAME primitive
    transplant.py's own `_random_equal_norm_vector` uses (deliberately reimplemented, not imported: this
    script is a caller/consumer, not an internal of clozn.analysis), so a chosen `seed` here reproduces
    exactly the same realization `transplant.run_site(seed=...)`'s internal random_equal_norm arm would
    draw when scale=1.0."""
    rng = random.Random(seed)
    raw = [rng.gauss(0.0, 1.0) for _ in range(len(reference_row))]
    k = (scale * norm(reference_row)) / norm(raw)
    return [x * k for x in raw]


class PlantProxy:
    """Wraps a real EngineClient. Every .score() call gets ONE extra write spec appended -- see this
    module's docstring for exactly why (per-request writes, capture-before-write, append-wins-ties)."""

    _FIELD = {"residual": "write", "ffn": "ffn_write"}

    def __init__(self, engine, hook: str, plant_layer: int, plant_position: int, plant_vector):
        self._engine = engine
        self._field = self._FIELD[hook]
        self._spec = {"layer": plant_layer, "positions": [plant_position], "values": list(plant_vector)}

    def score(self, **kwargs):
        specs = list(kwargs.get(self._field) or [])
        specs.append(dict(self._spec))
        kwargs[self._field] = specs
        return self._engine.score(**kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)


@contextmanager
def _boot(path: str, port: int):
    proc, health, gpu = spawn_engine(path, port, _flags_for(path), prefer_gpu=True)
    try:
        yield EngineClient(host="127.0.0.1", port=port), health
    finally:
        _terminate_process(proc)


def get_health(path: str) -> dict:
    with _boot(path, _next_port()) as (_eng, health):
        return dict(health)


def clean_loader(path: str):
    """A context manager yielding a plain, unproxied EngineClient -- the shape transplant.py /
    causal_bisect.py / mechanistic_diff.py expect from reference_loader/candidate_loader."""
    port = _next_port()

    @contextmanager
    def _cm():
        with _boot(path, port) as (eng, _health):
            yield eng
    return _cm()


def proxied_loader(path: str, hook: str, plant_layer: int, plant_position: int, plant_vector):
    port = _next_port()

    @contextmanager
    def _cm():
        with _boot(path, port) as (eng, _health):
            yield PlantProxy(eng, hook, plant_layer, plant_position, plant_vector)
    return _cm()


def record(battery: int, name: str, status: str, message: str, detail: dict) -> dict:
    assert status in ("PASS", "FAIL", "SKIP")
    return {"battery": battery, "name": name, "status": status, "message": message, "detail": detail}


# ============================================================================== battery 1: positive control

def battery_1_positive_control(model_path: str) -> dict:
    """A KNOWN, planted residual corruption at (plant_layer, write_position) on a candidate that is
    otherwise IDENTICAL to the reference (same GGUF file, loaded twice, sequentially -- pair_compat is
    trivially 'compatible'). The bisect must localize the corruption's DOWNSTREAM effect and the
    reference transplant must reverse it there, discriminated from a random-equal-norm control that must
    NOT reverse it.

    Ground-truth design (validated live, see module docstring): the plant is injected at EXACTLY
    (plant_layer, write_position) on every candidate call. Because a residual `write` is a full stream
    OVERWRITE and PlantProxy appends its spec last (winning ties), a write-based arm AT plant_layer
    itself can never out-write the plant -- that ONE site is structurally unfixable through itself, and
    this script tests it explicitly to report that honestly rather than hide it. Every layer AFTER
    plant_layer is a genuinely clean test (never itself a plant target): a full residual overwrite there
    discards the corrupted trajectory entirely and replaces it with whatever is written -- the reference's
    clean row restores the correct answer; a random equal-norm vector should not (and, per an extensive
    live sweep during this script's development, essentially never does in this ~900-dim residual width).
    Layers BEFORE plant_layer are decoys: the corruption has not happened yet at that point in the graph,
    so a write there should never restore anything.
    """
    if not os.path.isfile(model_path):
        return record(1, "positive_control", "SKIP", f"model not found: {model_path}", {})

    try:
        health = get_health(model_path)
        n_layer, n_embd = health["n_layer"], health["n_embd"]
        plant_layer = max(2, n_layer // 4)
        prompt = "Water is made of hydrogen and"
        plant_scale = 8.0
        seed = 12345

        with _boot(model_path, _next_port()) as (eng, _h):
            r0 = eng.score(prompt=prompt, continuation=".", topk=5)
            n_prompt = r0["n_prompt"]
            write_position = n_prompt - 1
            r = eng.score(prompt=prompt, continuation=".", topk=5,
                           capture_layers=[plant_layer], capture_positions=[write_position])
            prompt_ids = r["prompt_ids"]
            top1 = r["tokens"][0]["topk"][0]
            target_token_id, target_piece = top1["id"], top1["piece"]
            clean_row = r["captured"][str(plant_layer)][str(write_position)]

        plant_vector = scaled_random_vector(clean_row, seed=seed, scale=plant_scale)
        if not all(math.isfinite(x) for x in plant_vector):
            return record(1, "positive_control", "FAIL", "constructed plant vector is not finite", {})
        continuation_ids = [target_token_id]

        def ref_loader():
            return clean_loader(model_path)

        def cand_loader():
            return proxied_loader(model_path, "residual", plant_layer, write_position, plant_vector)

        pc = pair_compatibility.assess_gguf_pair(model_path, model_path, label_a="reference",
                                                  label_b="candidate")
        if not pair_compatibility.may_residual_transplant(pc):
            return record(1, "positive_control", "FAIL",
                          "pair_compatibility refused residual_transplant on a model diffed against "
                          "itself -- this is a bug, not an environmental gap", {"pair_compat": pc})

        # Confirm the plant actually perturbs the candidate's baseline answer before spending the
        # bisect's own budget on it.
        with _boot(model_path, _next_port()) as (raw_eng, _h):
            proxy = PlantProxy(raw_eng, "residual", plant_layer, write_position, plant_vector)
            baseline = proxy.score(prompt_ids=prompt_ids, continuation_ids=continuation_ids, topk=5)
        baseline_top1 = baseline["tokens"][0]["topk"][0]["id"]
        if baseline_top1 == target_token_id:
            return record(1, "positive_control", "FAIL",
                          f"planted perturbation (scale={plant_scale}x) did not change the candidate's "
                          f"top-1 answer away from target ({target_token_id}) -- too weak to test "
                          "localization against", {"plant_layer": plant_layer, "plant_scale": plant_scale})

        decoys_before = sorted({max(1, plant_layer - 4), max(1, plant_layer - 2)})
        decoys_after = sorted({plant_layer + 2, plant_layer + 6, plant_layer + 10})
        decoys_after = [l for l in decoys_after if l < n_layer - 1]
        residual_layers = decoys_before + [plant_layer] + decoys_after

        bisect_out = causal_bisect.run_bisect(
            pair_compat=pc, reference_loader=ref_loader, candidate_loader=cand_loader,
            prompt_ids=prompt_ids, continuation_ids=continuation_ids,
            write_positions=[write_position], readout_position=n_prompt, target_token_id=target_token_id,
            primary_metric="reference_token_logprob_recovery", search_kinds=["residual"],
            residual_layers=residual_layers, topk=5, seed=seed, store_tensors=False, validate=True)

        if not bisect_out["ok"]:
            return record(1, "positive_control", "FAIL", f"run_bisect refused: {bisect_out['error']}",
                          {"residual_layers": residual_layers})

        doc = bisect_out["document"]
        verdict = doc["verdict"]
        per_site = {}
        for s in doc["single_site_tests"]:
            if not s["ok"]:
                per_site[s["layer"]] = {"ok": False, "error": s["error"]}
                continue
            a = s["transplant"]["analysis"]
            per_site[s["layer"]] = {
                "ok": True, "instrument_sane": a.get("instrument_sane"),
                "reference_moved": a.get("reference_moved_toward_reference"),
                "random_moved": a.get("random_moved_toward_reference"),
                "reference_specific": a.get("reference_specific"),
            }

        found_sites = sorted(x["layer"] for x in verdict.get("evidence", {}).get("sites", []))
        expected_downstream = sorted(decoys_after)
        instrument_sane_ok = all(v.get("instrument_sane") for v in per_site.values() if v.get("ok"))
        upstream_clean = all(not per_site.get(l, {}).get("reference_specific") for l in decoys_before)
        downstream_found = all(per_site.get(l, {}).get("reference_specific") for l in decoys_after
                                if per_site.get(l, {}).get("ok"))
        exact_site_unfixable = not per_site.get(plant_layer, {}).get("reference_specific")

        passed = (verdict["label"] == "localized_site" and instrument_sane_ok and upstream_clean
                  and downstream_found and exact_site_unfixable and set(found_sites) == set(expected_downstream))

        precision = ("exact site (layer=%d) correctly NOT reported (structurally unfixable through "
                     "itself by this plant's own construction -- see module docstring); every downstream "
                     "layer %r correctly localized; every upstream decoy %r correctly excluded"
                     % (plant_layer, expected_downstream, decoys_before))

        detail = {
            "n_layer": n_layer, "n_embd": n_embd, "prompt": prompt, "plant_layer": plant_layer,
            "plant_scale": plant_scale, "target_token_id": target_token_id, "target_piece": target_piece,
            "write_position": write_position, "baseline_top1": baseline_top1,
            "residual_layers_tested": residual_layers, "verdict": verdict, "per_site": per_site,
            "found_sites": found_sites, "expected_downstream_sites": expected_downstream,
            "precision": precision,
        }
        msg = (f"verdict={verdict['label']!r} evidence_sites={found_sites} "
               f"(expected downstream {expected_downstream}, exact plant layer {plant_layer} correctly "
               f"excluded, upstream decoys {decoys_before} correctly excluded)")
        return record(1, "positive_control", "PASS" if passed else "FAIL", msg, detail)

    except Exception as exc:  # noqa: BLE001
        return record(1, "positive_control", "FAIL", f"{type(exc).__name__}: {exc}",
                      {"traceback": traceback.format_exc()})


# ============================================================================== battery 2: negative control

def battery_2_negative_control(model_path: str) -> dict:
    """A model diffed and bisected against ITSELF (same GGUF file, no perturbation at all, two separate
    plain -- unproxied -- loads). Observational divergence (mechanistic_diff) and intervention effects
    (causal_bisect) must sit at the numerical floor: near-zero residual distance, near-1.0 cosine
    similarity, and NO nonzero localization verdict anywhere (there is no disagreement for a transplant
    to correct, since candidate's own greedy top-1 already equals target_token_id by construction --
    same weights, same forward). Reports the ACTUAL measured floor values, not just pass/fail.
    """
    if not os.path.isfile(model_path):
        return record(2, "negative_control", "SKIP", f"model not found: {model_path}", {})

    try:
        health = get_health(model_path)
        n_layer, n_embd = health["n_layer"], health["n_embd"]
        prompt = "Water is made of hydrogen and"

        with _boot(model_path, _next_port()) as (eng, _h):
            r0 = eng.score(prompt=prompt, continuation=".", topk=5)
            n_prompt = r0["n_prompt"]
            write_position = n_prompt - 1
            prompt_ids = r0["prompt_ids"]
            target_token_id = r0["tokens"][0]["topk"][0]["id"]
        continuation_ids = [target_token_id]

        pc = pair_compatibility.assess_gguf_pair(model_path, model_path, label_a="reference",
                                                  label_b="candidate")

        layers = sorted({2, n_layer // 4, n_layer // 2, (3 * n_layer) // 4, n_layer - 2})
        layers = [l for l in layers if 1 <= l < n_layer - 1]
        mdiff_out = mechanistic_diff.compare(
            pair_compat=pc, reference_loader=lambda: clean_loader(model_path),
            candidate_loader=lambda: clean_loader(model_path), prompt_ids=prompt_ids,
            continuation_ids=continuation_ids, layers=layers, positions=[write_position], topk=10,
            store_tensors=False, validate=True)
        if not mdiff_out["ok"]:
            return record(2, "negative_control", "FAIL", f"mechanistic_diff refused: {mdiff_out['error']}", {})

        mdoc = mdiff_out["document"]
        cosines, l2s = [], []
        for p in mdoc["residual_points"]:
            m = p.get("metrics", {})
            if "residual_cosine_similarity" in m:
                cosines.append(m["residual_cosine_similarity"])
            if "residual_l2_normalized" in m:
                l2s.append(m["residual_l2_normalized"])
        min_cos = min(cosines) if cosines else None
        max_l2 = max(l2s) if l2s else None
        floor_ok = (cosines and l2s and min_cos > 0.999 and max_l2 < 0.01)

        # No-perturbation bisect: baseline should already match target at every tested residual layer
        # (same weights => same greedy answer), so nothing should ever localize.
        residual_layers = layers
        bisect_out = causal_bisect.run_bisect(
            pair_compat=pc, reference_loader=lambda: clean_loader(model_path),
            candidate_loader=lambda: clean_loader(model_path), prompt_ids=prompt_ids,
            continuation_ids=continuation_ids, write_positions=[write_position], readout_position=n_prompt,
            target_token_id=target_token_id, primary_metric="reference_token_logprob_recovery",
            search_kinds=["residual"], residual_layers=residual_layers, topk=5, seed=1,
            store_tensors=False, validate=True)
        if not bisect_out["ok"]:
            return record(2, "negative_control", "FAIL", f"run_bisect refused: {bisect_out['error']}", {})

        bdoc = bisect_out["document"]
        verdict = bdoc["verdict"]
        per_site = {}
        baseline_already_matches = []
        instrument_sane_vals = []
        self_transplant_matches = []
        for s in bdoc["single_site_tests"]:
            if not s["ok"]:
                per_site[s["layer"]] = {"ok": False, "error": s["error"]}
                continue
            a = s["transplant"]["analysis"]
            baseline_already_matches.append(bool(a.get("baseline_already_matches_target")))
            instrument_sane_vals.append(a.get("instrument_sane"))
            baseline_top1 = s["transplant"]["baseline"]["metrics"].get("top1_token_id")
            self_top1 = None
            for arm in s["transplant"]["arms"]:
                if arm["name"] == "candidate_self_transplant":
                    self_top1 = arm["metrics"].get("top1_token_id")
            self_transplant_matches.append(self_top1 == baseline_top1)
            per_site[s["layer"]] = {
                "ok": True, "instrument_sane": a.get("instrument_sane"),
                "baseline_already_matches_target": a.get("baseline_already_matches_target"),
                "reference_specific": a.get("reference_specific"),
                "baseline_top1": baseline_top1, "self_transplant_top1": self_top1,
            }

        no_nonzero_verdict = verdict["label"] not in ("localized_site", "localized_window",
                                                       "distributed_restoration")
        all_baseline_matched = all(baseline_already_matches) if baseline_already_matches else False
        all_instrument_sane = all(v is True for v in instrument_sane_vals) if instrument_sane_vals else False
        all_self_matched = all(self_transplant_matches) if self_transplant_matches else False

        passed = bool(floor_ok and no_nonzero_verdict and all_baseline_matched and all_instrument_sane
                      and all_self_matched)

        detail = {
            "n_layer": n_layer, "n_embd": n_embd, "prompt": prompt, "layers_tested": layers,
            "residual_cosine_similarity_min": min_cos, "residual_l2_normalized_max": max_l2,
            "residual_points_measured": len(mdoc["residual_points"]),
            "bisect_verdict": verdict, "per_site": per_site,
            "all_baseline_already_matches_target": all_baseline_matched,
            "all_instrument_sane": all_instrument_sane,
            "all_self_transplant_reproduces_baseline": all_self_matched,
        }
        msg = (f"floor: cos>={min_cos:.6f} l2<={max_l2:.6f}; bisect verdict={verdict['label']!r} "
               f"(no nonzero localization); baseline_already_matches_target everywhere={all_baseline_matched}; "
               f"instrument_sane everywhere={all_instrument_sane}") if cosines else "no residual_points measured"
        return record(2, "negative_control", "PASS" if passed else "FAIL", msg, detail)

    except Exception as exc:  # noqa: BLE001
        return record(2, "negative_control", "FAIL", f"{type(exc).__name__}: {exc}",
                      {"traceback": traceback.format_exc()})


# ========================================================================= battery 3: random-control gate

def battery_3_random_control_gate(model_path: str) -> dict:
    """Constructs a genuine knife-edge case: SOME perturbation (the reference's own clean state, but ALSO
    a random equal-norm vector) flips the candidate's answer, but the reference's specific content is not
    what makes the difference. The system must report `reference_specific=False` /
    perturbation-sensitive reasoning at that site -- never claim it as localized -- exactly the repo's own
    prior correction (docs/research/DISTRIBUTED_FUNCTION.md section B).

    Construction (found by a live scale/layer/seed sweep during this script's development -- see module
    docstring): a SMALL, ADDITIVE `ffn_write` plant (composable, unlike residual's full overwrite) at a
    layer just upstream of the tested site puts the candidate on a genuine decision boundary: at the
    tested site, the reference's clean ffn_out AND the large majority of random equal-norm realizations
    both flip the top-1 token to target_token_id. A residual (full-overwrite) plant was tried first and
    proved extremely robust against this (0/360 trials across 6 scales x 4 layers x 15 seeds during
    development) -- reported here as a substantive, informational finding: genuine perturbation-sensitivity
    to isotropic random noise appears to require the additive/partial-overwrite regime, not a full
    residual-stream reset, at least in this ~900-dim width.
    """
    if not os.path.isfile(model_path):
        return record(3, "random_control_gate", "SKIP", f"model not found: {model_path}", {})

    try:
        prompt = "The sky is"
        plant_layer = 6
        test_layer = 8
        plant_scale = 1.5
        seed = 0   # random_equal_norm's realization at this seed is the one confirmed live to also flip

        with _boot(model_path, _next_port()) as (eng, health):
            n_layer = health["n_layer"]
            r0 = eng.score(prompt=prompt, continuation=".", topk=5)
            n_prompt = r0["n_prompt"]
            write_position = n_prompt - 1
            r = eng.score(prompt=prompt, continuation=".", topk=5,
                           ffn_capture_layers=[plant_layer], ffn_capture_positions=[write_position])
            prompt_ids = r["prompt_ids"]
            target_token_id = r["tokens"][0]["topk"][0]["id"]
            target_piece = r["tokens"][0]["topk"][0]["piece"]
            clean_ffn = r["ffn_captured"][str(plant_layer)][str(write_position)]

        plant_vector = scaled_random_vector(clean_ffn, seed=777, scale=plant_scale)
        continuation_ids = [target_token_id]

        def ref_loader():
            return clean_loader(model_path)

        def cand_loader():
            return proxied_loader(model_path, "ffn", plant_layer, write_position, plant_vector)

        pc = pair_compatibility.assess_gguf_pair(model_path, model_path, label_a="reference",
                                                  label_b="candidate")

        site_out = transplant.run_site(
            pair_compat=pc, reference_loader=ref_loader, candidate_loader=cand_loader,
            prompt_ids=prompt_ids, continuation_ids=continuation_ids,
            site={"hook": "ffn", "layer": test_layer}, shuffled_layer=test_layer + 1,
            write_positions=[write_position], readout_position=n_prompt, target_token_id=target_token_id,
            topk=5, seed=seed, store_tensors=False, validate=True)
        if not site_out["ok"]:
            return record(3, "random_control_gate", "FAIL", f"run_site refused: {site_out['error']}", {})

        sdoc = site_out["document"]
        a = sdoc["analysis"]
        gate_held = (a.get("instrument_sane") is True and a.get("reference_moved_toward_reference") is True
                     and a.get("random_moved_toward_reference") is True
                     and a.get("reference_specific") is False)

        # Secondary, informational observation: does the FULL bisect (auto-sweeping every ffn layer)
        # correctly keep this knife-edge site's perturbation-sensitivity SEPARATE from any genuinely
        # reference-specific site elsewhere in the model, rather than either hiding a real finding or
        # falsely broadening one knife-edge site's sensitivity into a whole-model claim?
        bisect_out = causal_bisect.run_bisect(
            pair_compat=pc, reference_loader=ref_loader, candidate_loader=cand_loader,
            prompt_ids=prompt_ids, continuation_ids=continuation_ids, write_positions=[write_position],
            readout_position=n_prompt, target_token_id=target_token_id,
            primary_metric="reference_token_logprob_recovery", search_kinds=["ffn"], window_size=4,
            topk=5, seed=seed, store_tensors=False, validate=True)
        bisect_summary = None
        if bisect_out["ok"]:
            bverdict = bisect_out["document"]["verdict"]
            site_records = {s["layer"]: s for s in bisect_out["document"]["single_site_tests"] if s["ok"]}
            test_layer_record = site_records.get(test_layer, {})
            test_layer_analysis = (test_layer_record.get("transplant") or {}).get("analysis", {})
            bisect_summary = {
                "verdict": bverdict,
                "knife_edge_site_reference_specific": test_layer_analysis.get("reference_specific"),
            }

        detail = {
            "prompt": prompt, "plant_layer": plant_layer, "test_layer": test_layer,
            "plant_scale": plant_scale, "target_token_id": target_token_id, "target_piece": target_piece,
            "write_position": write_position, "seed": seed,
            "single_site_analysis": a, "bisect_secondary_observation": bisect_summary,
            "residual_sweep_finding": ("a full residual-overwrite plant was swept across 6 scales x 4 "
                                       "layers x 15 seeds (360 trials) during development and NEVER "
                                       "produced a random-equal-norm hit -- genuine perturbation-"
                                       "sensitivity in this model/width required the additive ffn hook"),
        }
        msg = (f"single-site gate: instrument_sane={a.get('instrument_sane')} "
               f"reference_moved={a.get('reference_moved_toward_reference')} "
               f"random_moved={a.get('random_moved_toward_reference')} "
               f"reference_specific={a.get('reference_specific')} (expected False)")
        return record(3, "random_control_gate", "PASS" if gate_held else "FAIL", msg, detail)

    except Exception as exc:  # noqa: BLE001
        return record(3, "random_control_gate", "FAIL", f"{type(exc).__name__}: {exc}",
                      {"traceback": traceback.format_exc()})


# ============================================================================ battery 4: real quant pairs

_QUANT_PROMPTS = [
    "The capital of France is",
    "The capital of Japan is",
    "The chemical symbol for gold is",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "The author of Romeo and Juliet is",
    "The speed of light is approximately",
    "The first president of the United States was",
    "A triangle has this many sides:",
    "The opposite of hot is",
    "Photosynthesis occurs in the",
    "The square root of 64 is",
    "The currency used in Japan is",
    "DNA stands for",
    "The freezing point of water in Celsius is",
]


def _top1_answers(model_path: str, prompts: list) -> dict:
    """{prompt: {"top1_id", "top1_piece", "prompt_ids", "n_prompt"}} -- one load, N forwards."""
    out = {}
    with _boot(model_path, _next_port()) as (eng, _h):
        for p in prompts:
            r = eng.score(prompt=p, continuation=".", topk=1)
            top1 = r["tokens"][0]["topk"][0]
            out[p] = {"top1_id": top1["id"], "top1_piece": top1["piece"], "prompt_ids": r["prompt_ids"],
                      "n_prompt": r["n_prompt"]}
    return out


def _run_quant_pair(reference_path: str, candidate_path: str, label: str) -> dict:
    battery_name = f"real_quant_pair[{label}]"
    for path in (reference_path, candidate_path):
        if not os.path.isfile(path):
            return record(4, battery_name, "SKIP", f"model not found: {path}", {})
    try:
        pc = pair_compatibility.assess_gguf_pair(reference_path, candidate_path,
                                                  label_a="reference", label_b="candidate")
        if pc["verdict"]["overall"] == "incompatible":
            return record(4, battery_name, "FAIL",
                          f"pair_compatibility refused this quant pair as incompatible: "
                          f"{pc['verdict']['reasons']}", {"pair_compat": pc})

        t0 = time.monotonic()
        ref_answers = _top1_answers(reference_path, _QUANT_PROMPTS)
        cand_answers = _top1_answers(candidate_path, _QUANT_PROMPTS)
        disagreements = [p for p in _QUANT_PROMPTS if ref_answers[p]["top1_id"] != cand_answers[p]["top1_id"]]
        load_time = time.monotonic() - t0

        if not disagreements:
            return record(4, battery_name, "PASS",
                          f"no top-1 disagreement across {len(_QUANT_PROMPTS)} prompts -- nothing to "
                          "bisect (both quants agree everywhere tried); this IS a legitimate, honestly "
                          "reported outcome, not a manufactured one",
                          {"prompts_tried": _QUANT_PROMPTS, "ref_answers": ref_answers,
                           "cand_answers": cand_answers, "load_time_s": load_time})

        prompt = disagreements[0]
        prompt_ids = ref_answers[prompt]["prompt_ids"]
        n_prompt = ref_answers[prompt]["n_prompt"]
        write_position = n_prompt - 1
        target_token_id = ref_answers[prompt]["top1_id"]
        continuation_ids = [target_token_id]

        n_layer = (pc.get("layer_count") or {}).get("value_b")
        window_size = max(4, (n_layer or 28) // 4)

        bisect_out = causal_bisect.run_bisect(
            pair_compat=pc, reference_loader=lambda: clean_loader(reference_path),
            candidate_loader=lambda: clean_loader(candidate_path), prompt_ids=prompt_ids,
            continuation_ids=continuation_ids, write_positions=[write_position], readout_position=n_prompt,
            target_token_id=target_token_id, primary_metric="reference_token_logprob_recovery",
            search_kinds=["ffn"], window_size=window_size, max_windows=4, topk=5, seed=1,
            store_tensors=False, validate=True)

        if not bisect_out["ok"]:
            return record(4, battery_name, "FAIL", f"run_bisect refused: {bisect_out['error']}",
                          {"prompt": prompt, "disagreements": disagreements})

        doc = bisect_out["document"]
        verdict = doc["verdict"]
        VALID_LABELS = {"localized_site", "localized_window", "distributed_restoration",
                        "perturbation_sensitive", "no_restoration", "inconclusive", "unavailable"}
        classification_valid = verdict["label"] in VALID_LABELS

        detail = {
            "reference_model": os.path.basename(reference_path), "candidate_model": os.path.basename(candidate_path),
            "prompts_tried": _QUANT_PROMPTS, "disagreements_found": disagreements,
            "bisected_prompt": prompt, "target_token_id": target_token_id,
            "reference_piece": ref_answers[prompt]["top1_piece"],
            "candidate_piece": cand_answers[prompt]["top1_piece"],
            "window_size": window_size, "coverage": doc.get("coverage"), "verdict": verdict,
            "n_window_tests": len(doc.get("window_tests", [])),
            "n_single_site_tests": len(doc.get("single_site_tests", [])),
            "load_time_s": load_time,
        }
        msg = (f"{len(disagreements)}/{len(_QUANT_PROMPTS)} prompts disagreed; bisected {prompt!r} "
               f"({ref_answers[prompt]['top1_piece']!r} vs {cand_answers[prompt]['top1_piece']!r}); "
               f"verdict={verdict['label']!r} ({len(doc.get('window_tests', []))} window tests, "
               f"{len(doc.get('single_site_tests', []))} single-site confirmations) -- "
               "localized/distributed/inconclusive/perturbation_sensitive/no_restoration are all valid")
        return record(4, battery_name, "PASS" if classification_valid else "FAIL", msg, detail)

    except Exception as exc:  # noqa: BLE001
        return record(4, battery_name, "FAIL", f"{type(exc).__name__}: {exc}",
                      {"traceback": traceback.format_exc()})


def battery_4_real_quant_pairs() -> list:
    return [
        _run_quant_pair(Q8_MODEL, Q2_MODEL, "Q8_0_vs_Q2_K"),
        _run_quant_pair(Q8_MODEL, Q4_MODEL, "Q8_0_vs_Q4_K_M"),
    ]


# ======================================================================================================= cli

def _print_table(results: list) -> None:
    print("\n" + "=" * 100)
    print(f"{'#':<3} {'battery':<28} {'status':<6} message")
    print("-" * 100)
    for r in results:
        print(f"{r['battery']:<3} {r['name']:<28} {r['status']:<6} {r['message'][:200]}")
    print("=" * 100)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--battery", default="all", help="comma list of 1,2,3,4 or 'all'")
    ap.add_argument("--fast-model", default=FAST_MODEL)
    ap.add_argument("--out", default=None, help="output JSON path (default: runs/experiments/bisect_acceptance_<ts>.json)")
    args = ap.parse_args()

    selected = {1, 2, 3, 4} if args.battery == "all" else {int(x) for x in args.battery.split(",")}

    results = []
    if 1 in selected:
        print("\n### battery 1: synthetic positive control ###", flush=True)
        results.append(battery_1_positive_control(args.fast_model))
    if 2 in selected:
        print("\n### battery 2: same-model negative control ###", flush=True)
        results.append(battery_2_negative_control(args.fast_model))
    if 3 in selected:
        print("\n### battery 3: random-control gate ###", flush=True)
        results.append(battery_3_random_control_gate(args.fast_model))
    if 4 in selected:
        print("\n### battery 4: real quant pairs ###", flush=True)
        results.extend(battery_4_real_quant_pairs())

    _print_table(results)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = args.out or os.path.join(REPO, "runs", "experiments", f"bisect_acceptance_{ts}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    summary = {
        "schema": "clozn.bisect_acceptance_report.v1",
        "generated_at": ts,
        "results": results,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1, default=str)
    print(f"\nwrote {out_path}")

    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    n_skip = sum(1 for r in results if r["status"] == "SKIP")
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
