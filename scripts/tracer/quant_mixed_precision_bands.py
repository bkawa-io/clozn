"""quant_mixed_precision_bands.py -- the MIXED-PRECISION counterfactual check
(research/quant-regression-population's follow-on): tests the engineering claim that falls out of
docs/research/QUANT_REGRESSION_POPULATION.md's depth gradient (ffn restoration-beats-control share early
20.6% / mid 33.2% / late 53.1%, recurring restoring layers 24/25/26 task-independent across both Q2_K and
Q4_K_M): "quantization damage concentrates in the last few MLP layers, so mixed-precision quantization
should protect late-MLP precision."

WHAT THIS DOES NOT DO
------------------------
It does NOT build a mixed-precision GGUF (a separate llama.cpp/gguf toolchain problem, out of scope here).
It SIMULATES the counterfactual "what if the candidate's late-MLP layers were reference (Q8_0) precision"
by BAND-LEVEL BROAD TRANSPLANT: writing the reference's captured ffn_out state at EVERY site in a whole
depth band (early/mid/late thirds) into the candidate, jointly, in ONE forward. This is an UPPER BOUND on
what real mixed-precision quantization could recover (an exact reference activation is strictly better
information than any quantized-but-higher-bit-width approximation of it would produce) -- see "LIMITS" in
the appended docs section for the full statement of what this design cannot show.

WHY `causal_bisect._run_window` DIRECTLY, NOT THE PUBLIC `run_bisect()` ENTRYPOINT
----------------------------------------------------------------------------------------
`run_bisect()` is the right instrument for a coarse-to-fine SEARCH: it always tiles the ffn hook's FULL
writable range starting at layer 0 (module docstring: "ffn sweeps its ENTIRE writable layer range
automatically") and, when a tile beats control, recursively bisects it in HALF. It has no parameter to
request an arbitrary, non-first, non-half-of-a-half layer subset as a single window -- there is no "test
layers 19..27 as one unit, and layers 9..17 as another, without also auto-splitting a retained one into
halves" mode, because bisecting is the whole point of that module. This experiment's question is the
opposite of a search: three (four, counting the natural/matched late split) SPECIFIC, PRE-CHOSEN bands,
each tested ONCE as a whole, never subdivided. `causal_bisect._run_window` is the exact composable primitive
underneath `run_bisect()`'s own tiling loop -- the SAME reference_transplant / candidate_self_transplant /
random_equal_norm / shuffled_window joint multi-site arm battery, with the SAME instrument_sane gate and
the SAME `beat_control` rule (`transplant.py`'s five-arm reference_specific rule, generalized to N sites) --
called here directly against caller-chosen site lists instead of through the tiler. This is composition of
`clozn.analysis.causal_bisect`'s own machinery, not a parallel transplant path: every write, every control
arm, and every gate below is still `_run_window`'s code, unmodified. The capture step (reference ffn
capture, candidate baseline+self capture) mirrors `run_bisect()`'s own internal ffn setup byte-for-byte in
shape (same `_call_score`/`_read_captured_multi`/`_read_arm_metrics` helpers), reused here rather than
reimplemented, because `run_bisect()` has no way to hand back its own internally-captured vectors for reuse
outside its own tiling loop.

BANDS
-------
Early/mid/late thirds use the SAME depth-bucket convention `docs/research/QUANT_REGRESSION_POPULATION.md`'s
own depth-gradient table and `scripts/tracer/quant_regression_bisect.py`'s `_depth_bucket` already use:
`frac = layer / (n_layer - 1)`; `frac < 1/3` early, `frac < 2/3` mid, else late. For Qwen2.5-7B-Instruct
(n_layer=28, confirmed live via `pair_compatibility.assess_gguf_pair` and an `ffn_capture` probe across all
28 layers on all three GGUF files -- every layer captures on every file, no known-gap layers here) that is
early=[0..8] (9), mid=[9..17] (9), late=[18..27] (10).

MATCHED SITE COUNT: early/mid are naturally 9 already; late is naturally 10. `late_matched` drops the
SHALLOWEST layer of the natural late band (18) to reach the same count (9) as early/mid, keeping the
deepest 9 (19..27) -- the policy applied uniformly is "keep the deepest N of the band", which for early/mid
here is a no-op (they are already exactly N) and for late trims from the shallow end. `late_natural` (all
10, 18..27) is ALSO tested, unbisected, as a robustness check that the matched-count restriction doesn't
change the qualitative picture -- reported alongside, never substituted for the matched comparison.

THE FIVE ARMS (the experiment brief)
---------------------------------------
  1. LATE band reference transplant   (natural, n=10, AND matched, n=9 -- both reported)
  2. EARLY band reference transplant  (matched site count, n=9)
  3. MID band reference transplant    (matched site count, n=9)
  4. RANDOM equal-norm control at the LATE band
  5. No-write replay (execution stability baseline)
Arms 2/3's OWN random_equal_norm and shuffled_window controls come bundled for free (`_run_window` always
runs its full arm battery for whatever site set it is given) -- reported as bonus context, not required by
the brief, never hidden. Arm 5 (no-write replay) is a bare repeat of the candidate's own baseline call with
no capture, no write -- same execution-stability role as `transplant.py`'s own `no_write_replay` arm,
duplicated here as a single extra `_call_score` per disagreement rather than imported, since it is a
zero-logic repeat of a call this script already makes, not a control that needs `transplant.py`'s gate
logic.

SEED / RANDOM-CONTROL INDEPENDENCE
--------------------------------------
`seed = args.seed + sample_index` (matches `quant_regression_bisect.py`'s fix for the retracted seed
confound -- docs/research/QUANT_REGRESSION_POPULATION.md's "The seed confound" section). ONE
`random.Random(seed)` is constructed per disagreement and threaded, BY REFERENCE, across that
disagreement's four band `_run_window()` calls (early -> mid -> late_matched -> late_natural, in that
order) -- mirroring `run_bisect()`'s OWN discipline of threading one continuously-advancing `rng` across
all of a single hook search's tiles (see `run_bisect`'s own `rng = random.Random(seed)` constructed once,
reused for every tile). Each band's `random_equal_norm` draw therefore comes from a DIFFERENT point in that
one advancing stream, never a frozen, reused direction -- the exact confound the population study's own
retraction was about. `--seed` defaults to 9001, deliberately distinct from `quant_regression_bisect.py`'s
own default base seed (1), so no reader mistakes these as sharing random state (they wouldn't even if the
integer matched -- totally different draw-consumption pattern -- but a visibly different default removes
the question).

SEQUENTIAL VRAM DISCIPLINE, BATCHED ACROSS DISAGREEMENTS (not per-disagreement, unlike
quant_regression_bisect.py)
-----------------------------------------------------------------------------------------------------------
`quant_regression_bisect.py` boots a fresh reference+candidate process PER DISAGREEMENT (via
`run_bisect()`'s own `reference_loader`/`candidate_loader` contract, called once per `run_bisect()`
invocation). Calling `_run_window` directly frees this script from that contract, so it follows
`quant_regression_mine.py`'s OWN, cheaper discipline instead: the reference (Q8_0) is booted ONCE and walks
EVERY selected disagreement's capture forward before being torn down completely; each candidate (Q2_K,
Q4_K_M) is then booted ONCE and walks every disagreement assigned to it (baseline + 4 band windows + 1
no-write-replay each) before being torn down. The two 7B-class models are never resident together at any
point. Reference and candidate are still never concurrent -- same rule, cheaper batching of the SAME rule
already established by this repo's Phase-1 mining script.

CHECKPOINTING
----------------
runs/experiments/_quant_mixed_precision_bands_checkpoint.json accumulates {"ref_capture": {id: {...}},
"candidate_results": {id: {...}}}, atomic write-then-replace after each disagreement in each phase.
Re-running resumes from whatever is already there.

OUTPUT
--------
runs/experiments/quant_mixed_precision_bands.json -- `clozn.quant_mixed_precision_bands.v1`.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "engine", "client"))
sys.path.insert(0, os.path.join(REPO, "scripts", "tracer"))

from clozn.analysis import causal_bisect, pair_compatibility  # noqa: E402
from clozn.cli.engine_process import spawn_engine, _terminate_process  # noqa: E402
from clozn.cli.commands.models import _flags_for  # noqa: E402
from clozn_engine import EngineClient  # noqa: E402

# scripts/tracer has no __init__.py (not a package) -- loaded by path so this script can reuse
# quant_regression_bisect.py's pool-building/stratified-sampling functions verbatim rather than
# re-deriving the SAME deterministic sampling logic a second time.
_qrb_spec = importlib.util.spec_from_file_location(
    "quant_regression_bisect", os.path.join(REPO, "scripts", "tracer", "quant_regression_bisect.py"))
qrb = importlib.util.module_from_spec(_qrb_spec)
_qrb_spec.loader.exec_module(qrb)

MODELS_DIR = os.path.expanduser("~/.clozn/models")
REFERENCE_MODEL = os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q8_0.gguf")
CANDIDATES = {
    "Q2_K": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q2_K.gguf"),
    "Q4_K_M": os.path.join(MODELS_DIR, "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
}
MINE_REPORT_PATH = os.path.join(REPO, "runs", "experiments", "quant_regression_mine.json")
OUT_PATH = os.path.join(REPO, "runs", "experiments", "quant_mixed_precision_bands.json")
CHECKPOINT_PATH = os.path.join(REPO, "runs", "experiments", "_quant_mixed_precision_bands_checkpoint.json")

BAND_ORDER = ("early", "mid", "late_matched", "late_natural")
_PORT = [8701]


def _next_port() -> int:
    _PORT[0] += 1
    return _PORT[0]


@contextmanager
def _boot(path: str, port: int):
    proc, health, gpu = spawn_engine(path, port, _flags_for(path), prefer_gpu=True)
    try:
        yield EngineClient(host="127.0.0.1", port=port), health
    finally:
        _terminate_process(proc)


def _atomic_write(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_checkpoint() -> dict:
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# =========================================================================== depth banding (matches
# quant_regression_bisect.py's _depth_bucket / docs/research/QUANT_REGRESSION_POPULATION.md's own table)

def _depth_bucket(layer: int, n_layer: int) -> str:
    if not n_layer or n_layer <= 1:
        return "unknown"
    frac = layer / max(1, n_layer - 1)
    if frac < 1 / 3:
        return "early"
    if frac < 2 / 3:
        return "mid"
    return "late"


def compute_bands(usable_layers: list, n_layer: int) -> dict:
    early = sorted(l for l in usable_layers if _depth_bucket(l, n_layer) == "early")
    mid = sorted(l for l in usable_layers if _depth_bucket(l, n_layer) == "mid")
    late = sorted(l for l in usable_layers if _depth_bucket(l, n_layer) == "late")
    natural_sizes = {"early": len(early), "mid": len(mid), "late": len(late)}
    matched_n = min(len(early), len(mid), len(late)) if all([early, mid, late]) else 0
    return {
        "early": early[-matched_n:] if matched_n else [],
        "mid": mid[-matched_n:] if matched_n else [],
        "late_matched": late[-matched_n:] if matched_n else [],
        "late_natural": late,
        "matched_n": matched_n,
        "natural_sizes": natural_sizes,
        "natural_layers": {"early": early, "mid": mid, "late": late},
    }


# ==================================================================================== Wilson score CI

def _wilson_ci(k: int, n: int, z: float = 1.96) -> "list | None":
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))) / denom
    return [round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)]


# ==================================================================================== extended context
# (mirrors quant_regression_bisect.py's run_one_disagreement's own prompt/position construction exactly --
# pure data plumbing, not transplant machinery, so replicated inline rather than imported)

def _extended_context(entry: dict) -> dict:
    k = entry["first_disagree_index"]
    prompt_ids_ext = list(entry["prompt_ids"]) + list(entry["continuation_ids"][:k])
    continuation_ids_ext = [entry["continuation_ids"][k]]
    write_position = len(prompt_ids_ext) - 1
    readout_position = len(prompt_ids_ext)
    return {
        "prompt_ids_ext": prompt_ids_ext, "continuation_ids_ext": continuation_ids_ext,
        "write_position": write_position, "readout_position": readout_position,
        "target_token_id": entry["target_token_id"],
        "reference_target_logprob": entry.get("reference_target_logprob"),
    }


# ============================================================================================ Phase A

def reference_capture_all(sample: list, n_layer: int, topk_unused: int, done: dict) -> dict:
    """Boot the reference ONCE, capture ffn_out at every usable layer for every disagreement's own
    write_position. Returns {id: {"ok": True, "usable_layers": [...], "vectors_by_layer": {...}}
    | {"ok": False, "error": ...}}."""
    out = dict(done)
    to_do = [e for e in sample if e["id"] not in out]
    if not to_do:
        return out
    print(f"[reference] booting {os.path.basename(REFERENCE_MODEL)} for {len(to_do)} disagreements "
          f"({len(out)} resumed from checkpoint) ...", flush=True)
    with _boot(REFERENCE_MODEL, _next_port()) as (eng, _h):
        for i, entry in enumerate(to_do):
            ctx = _extended_context(entry)
            candidate_layers = list(range(n_layer))
            call = causal_bisect._call_score(
                eng, "reference ffn capture", prompt_ids=ctx["prompt_ids_ext"],
                continuation_ids=ctx["continuation_ids_ext"], topk=0,
                ffn_capture_layers=candidate_layers, ffn_capture_positions=[ctx["write_position"]])
            if not call["ok"]:
                out[entry["id"]] = {"ok": False, "error": call["error"], "ctx": ctx}
            else:
                captured = causal_bisect._read_captured_multi(
                    call["response"], "ffn_captured", candidate_layers, [ctx["write_position"]])
                usable = [l for l in candidate_layers if captured[l][ctx["write_position"]] is not None]
                vectors_by_layer = {l: captured[l][ctx["write_position"]] for l in usable}
                out[entry["id"]] = {"ok": True, "usable_layers": usable,
                                    "vectors_by_layer": vectors_by_layer, "ctx": ctx}
            if (i + 1) % 5 == 0 or (i + 1) == len(to_do):
                _atomic_write(CHECKPOINT_PATH, {"ref_capture": out,
                                                "candidate_results": done.get("_candidate_results", {})})
                print(f"  [reference] {i + 1}/{len(to_do)}", flush=True)
    return out


# ============================================================================================ Phase B

def run_candidate_disagreement(engine, entry: dict, ref_capture: dict, n_layer: int, entry_seed: int,
                               topk: int, primary_metric: str) -> dict:
    ctx = ref_capture["ctx"]
    if not ref_capture.get("ok"):
        return {"ok": False, "error": f"reference capture failed for this disagreement: "
                                      f"{ref_capture.get('error')}"}
    ref_usable = ref_capture["usable_layers"]
    if not ref_usable:
        return {"ok": False, "error": "reference ffn capture produced no usable layer for this "
                                      "disagreement -- nothing to transplant"}

    baseline_call = causal_bisect._call_score(
        engine, "candidate ffn baseline", prompt_ids=ctx["prompt_ids_ext"],
        continuation_ids=ctx["continuation_ids_ext"], topk=topk,
        ffn_capture_layers=ref_usable, ffn_capture_positions=[ctx["write_position"]])
    if not baseline_call["ok"]:
        return {"ok": False, "error": baseline_call["error"]}
    self_captured = causal_bisect._read_captured_multi(
        baseline_call["response"], "ffn_captured", ref_usable, [ctx["write_position"]])
    usable_layers = [l for l in ref_usable if self_captured[l][ctx["write_position"]] is not None]
    if not usable_layers:
        return {"ok": False, "error": "candidate ffn capture produced no usable layer at any of the "
                                      "reference's usable layers for this disagreement"}
    baseline_metrics = causal_bisect._read_arm_metrics(
        baseline_call["response"], n_prompt=len(ctx["prompt_ids_ext"]), n_cont=1,
        readout_position=ctx["readout_position"], target_token_id=ctx["target_token_id"])["metrics"]

    ref_vectors_by_layer = {l: {ctx["write_position"]: ref_capture["vectors_by_layer"][l]} for l in usable_layers}
    self_vectors_by_layer = {l: {ctx["write_position"]: self_captured[l][ctx["write_position"]]}
                             for l in usable_layers}

    bands = compute_bands(usable_layers, n_layer)
    rng = random.Random(entry_seed)

    band_results: dict = {}
    for band_name in BAND_ORDER:
        sites = bands[band_name]
        if band_name == "late_natural" and sites == bands["late_matched"]:
            band_results[band_name] = {"skipped": True,
                                       "reason": "late_natural equals late_matched for this disagreement "
                                                "(natural late band was already exactly the matched "
                                                "count) -- not re-tested, would be an identical call"}
            continue
        if len(sites) < 2:
            band_results[band_name] = {"skipped": True,
                                       "reason": f"band has fewer than 2 usable sites ({len(sites)}) -- "
                                                f"_run_window requires a genuine multi-site window"}
            continue
        result = causal_bisect._run_window(
            candidate_engine=engine, hook="ffn", sites=sites, depth=0,
            ref_vectors_by_site=ref_vectors_by_layer, self_vectors_by_site=self_vectors_by_layer,
            usable_sites=usable_layers, baseline_metrics=baseline_metrics, positions=[ctx["write_position"]],
            prompt_ids=ctx["prompt_ids_ext"], continuation_ids=ctx["continuation_ids_ext"],
            n_prompt=len(ctx["prompt_ids_ext"]), n_cont=1, readout_position=ctx["readout_position"],
            target_token_id=ctx["target_token_id"], topk=topk, rng=rng,
            reference_target_logprob=ctx.get("reference_target_logprob"), primary_metric=primary_metric)
        random_arm = (result.get("arms") or {}).get("random_equal_norm")
        result["random_moved"] = (causal_bisect._flipped_to_target(baseline_metrics, random_arm)
                                  if random_arm is not None else None)
        result["n_sites"] = len(sites)
        band_results[band_name] = result

    no_write_call = causal_bisect._call_score(
        engine, "no_write_replay", prompt_ids=ctx["prompt_ids_ext"],
        continuation_ids=ctx["continuation_ids_ext"], topk=topk)
    if no_write_call["ok"]:
        replay_metrics = causal_bisect._read_arm_metrics(
            no_write_call["response"], n_prompt=len(ctx["prompt_ids_ext"]), n_cont=1,
            readout_position=ctx["readout_position"], target_token_id=ctx["target_token_id"])["metrics"]
        logprob_a = baseline_metrics.get("target_token_logprob")
        logprob_b = replay_metrics.get("target_token_logprob")
        logprob_diff = (abs(logprob_a - logprob_b) if isinstance(logprob_a, (int, float))
                        and isinstance(logprob_b, (int, float)) else None)
        stable = (baseline_metrics.get("top1_token_id") == replay_metrics.get("top1_token_id")
                 and (logprob_diff is None or logprob_diff < 1e-6))
        no_write_replay = {"ok": True, "stable": stable, "baseline_top1": baseline_metrics.get("top1_token_id"),
                           "replay_top1": replay_metrics.get("top1_token_id"), "logprob_diff": logprob_diff}
    else:
        no_write_replay = {"ok": False, "error": no_write_call["error"]}

    return {"ok": True, "usable_layers": usable_layers, "bands": {k: v for k, v in bands.items()
                                                                  if k != "natural_layers" or True},
           "baseline_metrics": baseline_metrics, "band_results": band_results,
           "no_write_replay": no_write_replay, "seed": entry_seed}


# =================================================================================== Phase C: characterize

def _band_stats(records: list, band_name: str) -> dict:
    """`tested`/`restoration_rate` count only disagreements where there was something LIVE to restore at
    this band (baseline top-1 != target at the moment this band was tested) -- audit finding, 2026-07-30:
    an earlier version of this function counted 'the candidate's baseline top-1 already equals
    target_token_id' cases (see `_run_window`'s own early-return reason string) in the SAME denominator
    as genuine restoration failures, silently treating 'nothing needed fixing' as if it were 'the
    transplant failed to fix it'. Those cases are real and disclosed (`n_already_correct_no_disagreement`)
    but excluded from `restoration_rate`'s denominator -- a band that had nothing to restore cannot
    honestly count against (or for) restoration. Separately, a distinct and rarer failure mode --
    instrument_sane True but movement still not evaluable (missing top-1/target-hit data on an arm) --
    is tracked as `n_movement_not_evaluable` and also excluded, since it is a data-quality gap, not a
    disagreement that existed and failed to restore."""
    tested = 0
    instrument_sane_true = 0
    instrument_sane_false = 0
    already_correct = 0
    movement_not_evaluable = 0
    moved = 0
    beat_control = 0
    random_moved = 0
    shuffled_available = 0
    skipped = 0
    for rec in records:
        cr = rec.get("candidate_result")
        if not cr or not cr.get("ok"):
            continue
        br = (cr.get("band_results") or {}).get(band_name)
        if br is None:
            continue
        if br.get("skipped"):
            skipped += 1
            continue
        if not br.get("instrument_sane"):
            instrument_sane_false += 1
            continue
        instrument_sane_true += 1
        if br.get("moved") is None:
            reasons = br.get("reasons") or []
            if any("already equals target_token_id" in r for r in reasons):
                already_correct += 1
            else:
                movement_not_evaluable += 1
            continue
        tested += 1
        if br.get("moved"):
            moved += 1
        if br.get("beat_control"):
            beat_control += 1
        if br.get("random_moved"):
            random_moved += 1
        if "shuffled_window" in (br.get("arms") or {}):
            shuffled_available += 1
    rate = round(beat_control / tested, 4) if tested else None
    return {
        "n_disagreements_with_band_tested": tested + already_correct + movement_not_evaluable,
        "n_skipped": skipped,
        "instrument_sane_true": instrument_sane_true, "instrument_sane_false": instrument_sane_false,
        "n_already_correct_no_disagreement": already_correct,
        "n_movement_not_evaluable": movement_not_evaluable,
        "n_tested_sane": tested,
        "n_moved_reference_arm": moved,
        "n_random_control_also_moved": random_moved,
        "n_beat_control": beat_control,
        "restoration_rate": rate,
        "restoration_rate_wilson_ci95": _wilson_ci(beat_control, tested) if tested else None,
        "n_shuffled_window_available": shuffled_available,
    }


def characterize(records: list) -> dict:
    by_band = {b: _band_stats(records, b) for b in BAND_ORDER}

    late = by_band["late_matched"]
    early = by_band["early"]
    mid = by_band["mid"]

    # paired comparison: for the SAME disagreements, how often did late_matched beat_control while
    # early/mid did not, and vice versa -- more robust than a raw rate difference at this sample size.
    paired_late_vs_early = Counter()
    paired_late_vs_mid = Counter()
    for rec in records:
        cr = rec.get("candidate_result")
        if not cr or not cr.get("ok"):
            continue
        brs = cr.get("band_results") or {}
        l = brs.get("late_matched") or {}
        e = brs.get("early") or {}
        m = brs.get("mid") or {}
        if l.get("instrument_sane") and e.get("instrument_sane"):
            lb, eb = bool(l.get("beat_control")), bool(e.get("beat_control"))
            paired_late_vs_early[(lb, eb)] += 1
        if l.get("instrument_sane") and m.get("instrument_sane"):
            lb, mb = bool(l.get("beat_control")), bool(m.get("beat_control"))
            paired_late_vs_mid[(lb, mb)] += 1

    no_write_stable = sum(1 for r in records
                          if (r.get("candidate_result") or {}).get("no_write_replay", {}).get("stable") is True)
    no_write_ok = sum(1 for r in records
                      if (r.get("candidate_result") or {}).get("no_write_replay", {}).get("ok"))

    errors = [{"id": r["id"], "error": (r.get("candidate_result") or {}).get("error")}
             for r in records if not (r.get("candidate_result") or {}).get("ok")]

    return {
        "by_band": by_band,
        "late_matched_vs_early_mid": {
            "late_beats_early_by_rate": (late["restoration_rate"] is not None and
                                        early["restoration_rate"] is not None and
                                        late["restoration_rate"] > early["restoration_rate"]),
            "late_beats_mid_by_rate": (late["restoration_rate"] is not None and
                                      mid["restoration_rate"] is not None and
                                      late["restoration_rate"] > mid["restoration_rate"]),
            "paired_late_vs_early_counts": {
                "late_yes_early_yes": paired_late_vs_early[(True, True)],
                "late_yes_early_no": paired_late_vs_early[(True, False)],
                "late_no_early_yes": paired_late_vs_early[(False, True)],
                "late_no_early_no": paired_late_vs_early[(False, False)],
            },
            "paired_late_vs_mid_counts": {
                "late_yes_mid_yes": paired_late_vs_mid[(True, True)],
                "late_yes_mid_no": paired_late_vs_mid[(True, False)],
                "late_no_mid_yes": paired_late_vs_mid[(False, True)],
                "late_no_mid_no": paired_late_vs_mid[(False, False)],
            },
        },
        "late_natural_vs_late_matched": {
            "late_natural_rate": by_band["late_natural"]["restoration_rate"],
            "late_matched_rate": by_band["late_matched"]["restoration_rate"],
        },
        "no_write_replay_stability": {"ok": no_write_ok, "stable": no_write_stable, "total": len(records)},
        "errors": errors,
    }


# ======================================================================================================= cli

def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mine-report", default=MINE_REPORT_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--sample-size", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--primary-metric", default="reference_token_logprob_recovery")
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test knob: only run the first N selected disagreements")
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()

    with open(args.mine_report, encoding="utf-8") as f:
        mine_report = json.load(f)

    pool = qrb.build_pool(mine_report)
    print("pool sizes by category:", {c: len(v) for c, v in pool.items()}, flush=True)
    sample = qrb.select_stratified_sample(pool, args.sample_size)
    if args.limit is not None:
        sample = sample[:args.limit]
    print(f"selected {len(sample)} disagreements (target {args.sample_size}):", flush=True)
    cat_counts = Counter(e["category"] for e in sample)
    cand_counts = Counter(e["candidate"] for e in sample)
    print(f"  by category: {dict(cat_counts)}", flush=True)
    print(f"  by candidate: {dict(cand_counts)}", flush=True)
    for i, e in enumerate(sample):
        print(f"    [{i}] [{e['id']}] {e['category']:<22} {e['candidate']:<7} pos={e['first_disagree_index']} "
              f"{e['prompt'][:50]!r} ref={e['target_piece']!r} cand={e['candidate_top1_piece']!r}", flush=True)
    if args.select_only:
        return 0

    dims_by_candidate = {name: qrb._pair_and_dims(path) for name, path in CANDIDATES.items()
                        if any(e["candidate"] == name for e in sample)}
    n_layers = {name: d["n_layer"] for name, d in dims_by_candidate.items()}
    distinct_n_layer = set(n_layers.values())
    if len(distinct_n_layer) != 1:
        print(f"WARNING: candidates report different layer counts: {n_layers} -- bands computed per "
              f"candidate's own n_layer, not assumed shared", flush=True)
    n_layer_for_ref = next(iter(distinct_n_layer)) if len(distinct_n_layer) == 1 else max(distinct_n_layer)

    checkpoint = _load_checkpoint()
    ref_done = dict(checkpoint.get("ref_capture") or {})
    candidate_done = dict(checkpoint.get("candidate_results") or {})
    if ref_done or candidate_done:
        print(f"resuming: {len(ref_done)} reference captures, {len(candidate_done)} candidate results "
              f"already in checkpoint", flush=True)

    t0 = time.monotonic()
    ref_capture = reference_capture_all(sample, n_layer_for_ref, args.topk, ref_done)
    _atomic_write(CHECKPOINT_PATH, {"ref_capture": ref_capture, "candidate_results": candidate_done})
    print(f"[reference] done in {time.monotonic() - t0:.1f}s", flush=True)

    by_candidate: dict = {}
    for e in sample:
        by_candidate.setdefault(e["candidate"], []).append(e)

    records: list = []
    for cand_name, entries in by_candidate.items():
        cand_path = CANDIDATES[cand_name]
        n_layer = n_layers[cand_name]
        to_do = [e for e in entries if e["id"] not in candidate_done]
        print(f"\n[{cand_name}] booting {os.path.basename(cand_path)} for {len(to_do)} disagreements "
              f"({len(entries) - len(to_do)} resumed from checkpoint) ...", flush=True)
        if to_do:
            with _boot(cand_path, _next_port()) as (eng, _h):
                for entry in to_do:
                    sample_index = sample.index(entry)
                    entry_seed = args.seed + sample_index
                    t1 = time.monotonic()
                    result = run_candidate_disagreement(eng, entry, ref_capture[entry["id"]], n_layer,
                                                        entry_seed, args.topk, args.primary_metric)
                    elapsed = time.monotonic() - t1
                    if result.get("ok"):
                        summary = {b: (result["band_results"].get(b) or {}).get("beat_control")
                                  for b in BAND_ORDER}
                        print(f"  [{entry['id']}] beat_control by band: {summary} "
                              f"no_write_stable={result['no_write_replay'].get('stable')} "
                              f"({elapsed:.1f}s)", flush=True)
                    else:
                        print(f"  [{entry['id']}] ERROR: {result.get('error')} ({elapsed:.1f}s)", flush=True)
                    candidate_done[entry["id"]] = result
                    _atomic_write(CHECKPOINT_PATH, {"ref_capture": ref_capture,
                                                    "candidate_results": candidate_done})

    for e in sample:
        records.append({"id": e["id"], "candidate": e["candidate"], "category": e["category"],
                        "prompt": e["prompt"], "first_disagree_index": e["first_disagree_index"],
                        "target_piece": e["target_piece"], "candidate_top1_piece": e["candidate_top1_piece"],
                        "candidate_result": candidate_done.get(e["id"])})

    characterization = characterize(records)

    report = {
        "schema": "clozn.quant_mixed_precision_bands.v1",
        "generated_at": _now_iso(),
        "reference_model": os.path.basename(REFERENCE_MODEL),
        "candidates": {k: os.path.basename(v) for k, v in CANDIDATES.items()},
        "mine_report_used": os.path.relpath(args.mine_report, REPO),
        "pool_sizes_by_category": {c: len(v) for c, v in pool.items()},
        "bounds": {
            "sample_size_target": args.sample_size, "sample_size_actual": len(sample),
            "n_layer_by_candidate": n_layers,
            "band_definition": "frac = layer / (n_layer - 1); <1/3 early, <2/3 mid, else late -- same "
                              "convention as quant_regression_bisect.py's _depth_bucket and "
                              "docs/research/QUANT_REGRESSION_POPULATION.md's own depth-gradient table",
            "matched_count_policy": "matched_n = min(len(early), len(mid), len(late)); each band trimmed "
                                   "to its OWN deepest matched_n layers (early/mid are no-ops here since "
                                   "already exactly matched_n; late drops its shallowest layer(s))",
            "late_natural_also_tested": "the FULL, untrimmed late band is also tested unbisected as a "
                                       "robustness check on the matched-count restriction",
            "seed_base": args.seed,
            "seed_scheme": "seed = seed_base + sample_index (index into the deterministic stratified "
                          "sample); ONE random.Random(seed) threaded across a disagreement's 4 band calls "
                          "(early, mid, late_matched, late_natural in that order), never reconstructed "
                          "per band -- each band's random-control draw comes from a different point in "
                          "that one advancing stream. See module docstring.",
            "topk": args.topk, "primary_metric": args.primary_metric,
            "write_target": "first disagreement position only, per prompt (matches population bisect "
                           "study's own convention)",
            "store_tensors": False,
        },
        "sample": [{"id": e["id"], "candidate": e["candidate"], "category": e["category"],
                    "prompt": e["prompt"], "first_disagree_index": e["first_disagree_index"],
                    "target_piece": e["target_piece"], "candidate_top1_piece": e["candidate_top1_piece"]}
                   for e in sample],
        "characterization": characterization,
        "records": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    _atomic_write(args.out, report)
    print(f"\nwrote {args.out}", flush=True)

    print("\n=== restoration rate by band ===")
    for b in BAND_ORDER:
        s = characterization["by_band"][b]
        print(f"  {b:<14} tested={s['n_tested_sane']:<4} beat_control={s['n_beat_control']:<4} "
              f"rate={s['restoration_rate']} ci95={s['restoration_rate_wilson_ci95']}")
    print("\n=== late_matched vs early/mid (paired) ===")
    print(json.dumps(characterization["late_matched_vs_early_mid"], indent=2))
    print("\n=== no-write replay stability ===")
    print(characterization["no_write_replay_stability"])
    if characterization["errors"]:
        print(f"\n=== {len(characterization['errors'])} errors ===")
        for e in characterization["errors"][:10]:
            print(f"  [{e['id']}] {e['error']}")

    try:
        os.remove(CHECKPOINT_PATH)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
