"""causal_bisect.py -- slice 3.5: the CAUSAL BISECT, a coarse-to-fine search over
`clozn.analysis.transplant`'s single-site five-arm primitive for WHERE a reference model's behavior
localizes in a candidate.

THE CENTRAL DESIGN FACT (measured live, see hook_vocabulary.py / scripts/spike/additive_writes_probe.py
and scripts/spike/ffn_hook_probe.py)
--------------------------------------------------------------------------------------------------------
`residual` (`l_out`) writes OVERWRITE the stream: writing at layer L1 and L2 is bit-identical to writing
at L2 alone. A residual "window" spanning several layers is therefore degenerate -- it silently collapses
to the deepest write and proves nothing about the shallower ones. Residual sites are SINGLE-SITE ONLY in
this module: they are tested individually (via `transplant.run_site()` directly, never here), and never
combined into a multi-site window.

`ffn` (`ffn_out`) and `head` (`kqv_out`) CONTRIBUTE/ADD into the stream and genuinely compose across
layers (MEASURED: engine/core/tests/test_ffn_hook.cpp, scripts/spike/ffn_hook_probe.py -- an L1-only
write, an L2-only write, and BOTH together produce three mutually different results). Multi-site windows
are built ONLY from these two "composable" kinds -- `COMPOSABLE_HOOKS`.

SCOPE OF THIS SLICE: WINDOWING IS ffn-ONLY
--------------------------------------------
`head` (`kqv_out`) is composable in principle, but its wire contract is per-Q-head
(`head_write: {layer, head, positions, values}`, `values` width = `positions.size() * d_head`, and
`d_head` is only knowable from a runtime `head_capture` probe -- see hook_vocabulary.py's `d_head_probe`).
Building a correct multi-layer, multi-head WINDOW write plan is real, separate work this slice does not
attempt (and `clozn.analysis.transplant.run_site()` -- which this module composes and does not modify --
does not support `hook="head"` at the time this module was written; a concurrent change may add it).
`_WINDOW_CAPABLE_HOOKS = ("ffn",)` names this precisely: `head` may appear in `search_kinds` and get
tested at the SINGLE-SITE step (delegated straight to `transplant.run_site()`, which degrades honestly --
refuses cleanly, never raises -- if the hook is not yet supported there), but it never participates in
window construction here. This is a disclosed scope limit, recorded in every artifact's
`search.window_capable_kinds` / `search.hooks_unavailable`, not a silent gap.

THE SEARCH
------------
1. Tile the candidate's writable range for each kind in `search_kinds ∩ _WINDOW_CAPABLE_HOOKS` (today:
   ffn only) into coarse windows of `window_size` layers (the LAST tile may be smaller; sizes are
   recorded, never silently rounded).
2. Test each coarse window: write the reference's captured state at EVERY layer in the window, jointly,
   in ONE forward per arm (`reference_transplant`, `candidate_self_transplant`, `random_equal_norm`, and
   `shuffled_window` when a disjoint same-size layer set exists) -- the same instrument-sanity +
   reference-vs-random-control structure `clozn.analysis.transplant._derive_analysis` uses at one site,
   generalized to N sites. A window is `retained` only when `instrument_sane` AND the reference arm beat
   the random equal-norm control (`beat_control`).
3. Recursively bisect every RETAINED window in half. A half of size 1 is a SITE, not a window -- it is
   handed to the single-site confirmation step, never tested by the window harness.
4. Single-site confirmation: every bisection leaf (plus any explicitly requested `residual_layers` /
   `head_layers`) is re-tested with `clozn.analysis.transplant.run_site()` DIRECTLY. Its
   `analysis.reference_specific` / `analysis.instrument_sane` are embedded and read verbatim -- this
   module never bypasses or recomputes them (see transplant.py's own docstring on why that field may only
   be set one way).
5. If no window/site anywhere beat control, the search reports `perturbation_sensitive` (something moved
   but a random perturbation moved it just as well) or `no_restoration` (nothing moved), never a
   localization claim -- see `_derive_verdict`.

WHY THE VERDICT NEVER CONFUSES "WINDOW REQUIRED" WITH "NOTHING WORKED"
--------------------------------------------------------------------------
`distributed_restoration` means a BROAD intervention restores while no narrower subset does -- the search
found real evidence, at a genuinely broad granularity, that resisted every attempt to narrow it. This
module reports it ONLY when a coarse (unbisected, depth-0) window beat control while every window/site
inside it that was actually tested did not; if the search ever DID find a real, narrower-than-coarse
window or site that beats control, that narrower result is reported instead (`localized_window` /
`localized_site`) -- `distributed_restoration` is reserved for the case where narrowing was attempted and
failed at every level, not merely "we didn't try." Because this verdict can only be built from
`window_tests` records, and `window_tests` is populated ONLY when a composable-kind window search
actually ran, a residual-only (or head-only, in this slice) search can never produce it -- enforced by a
hard assertion in `_derive_verdict` in addition to being structurally unreachable from empty data.

THE FIVE-ARM RULE, GENERALIZED, NEVER WEAKENED
--------------------------------------------------
A window/site is retained/localizing ONLY if its reference arm moved the candidate's answer toward the
target AND the random-equal-norm control did NOT -- exactly `transplant.py`'s own rule (see that module's
docstring and docs/research/DISTRIBUTED_FUNCTION.md section B: the prior transplant-localization study's
first pass overclaimed 5/12 "fixes" until the equal-norm control showed 3/12 were just perturbation-
sensitive, leaving 3/12 genuinely reference-specific). A perturbation that flips the answer without
beating that control is NEVER reported as localizing here, at any granularity.

RESTORATION_METRICS IS INFORMATIONAL HERE, NOT THE GATE
------------------------------------------------------------
`restoration_metrics.select_primary()` / `beat_control()` ARE used (per the caller-declared
`primary_metric`) to report a continuous movement comparison between the reference and random-control
arms (`movement_metrics` on every window). This needs a reference-side logprob for `target_token_id`
(`reference_target_logprob`, optional, caller-supplied) to produce a `gap_closed_fraction` at all; when it
is absent the metrics still report raw `movement`/`movement_sign`, just no `arm_beat_control`. The
STRUCTURAL gate that decides `retained`/the verdict is always the discrete top-1 rule above (always
computable from the arms this module already runs) -- `movement_metrics` never overrides it.

BATCHED `arms` SCREENING: HONESTLY UNAVAILABLE FOR WHAT THIS MODULE SEARCHES
----------------------------------------------------------------------------------
The engine's batched multi-sequence `arms` field (routes_whitebox.cpp, `numerical_regime:
"batched_approximate"`, measured up to ~0.19 nats drift vs sequential `/score` -- screening only, never a
receipt) parses each arm's `write` as a RESIDUAL write (`parse_write_specs`, `w.layer >= 1` required); it
has no `ffn_write`/`head_write` shape today. Since this module's windows are built from `ffn` (never
residual, per the central design fact above), there is no sound way to route a window candidate through
`arms` without silently misapplying an ffn-intended vector as a residual overwrite -- exactly the kind of
silent misapplication this codebase refuses to ship. `use_batched_screen=True` is accepted and reported
(`search.screening`) but is honestly marked `used=False` with the reason above; every window/site test in
this module is a normal sequential `/score` call. The parameter and the `screening` document field exist
so a future slice that DOES wire `ffn_write`/`head_write` into `arms` can flip this on without a schema
change.

STDLIB ONLY, OMIT NEVER NULL-PAD, SEQUENTIAL MODEL ORCHESTRATION
----------------------------------------------------------------------
Same three rules as every sibling module in `clozn.analysis`: no imports beyond the standard library
(`pyproject.toml` declares `dependencies = []`); a value that cannot be honestly computed is an absent key
plus a reason, never a fabricated zero or null; and the reference model is loaded for exactly one forward
per composable kind searched (capturing every candidate layer's row across ALL its windows/sites in that
ONE call), torn down, and only then is the candidate loaded and kept resident for the whole window search
-- the two 16GB-VRAM-worthy models are never resident together. Single-site confirmation calls
`transplant.run_site()`, which manages its OWN reference/candidate lifecycle per call (unavoidable
reloads, one pair per confirmed site -- correctness over throughput).
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from clozn import schemas
from clozn.analysis import pair_compatibility, restoration_metrics, transplant

SCHEMA_VERSION = "clozn.causal-bisect.v1"

_ALL_HOOKS = ("residual", "ffn", "head")
COMPOSABLE_HOOKS = ("ffn", "head")
_WINDOW_CAPABLE_HOOKS = ("ffn",)

_WINDOW_WRITE_FIELD = {"ffn": "ffn_write", "head": "head_write"}
_WINDOW_APPLIED_FIELD = {"ffn": "ffn_write_applied", "head": "head_write_applied"}
_WINDOW_CAPTURED_FIELD = {"ffn": "ffn_captured", "head": "head_rows"}

_SINGLE_SITE_SOURCE = {"residual": "explicit_residual", "ffn": "bisection_leaf", "head": "explicit_head"}

_ZERO_TOL = 1e-12


# =========================================================================================== tiny math
# Deliberately duplicated from clozn.analysis.transplant rather than importing its underscore-prefixed
# internals -- the same choice transplant.py itself made for pair_compatibility's writable-range logic
# ("duplicated here ... rather than asking pair_compatibility to grow a gate it was never asked to
# model"). Pure stdlib math over single ffn/head rows (n_embd floats), not a hot loop.

def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in values))


def _random_equal_norm_vector(reference_row: Sequence[float], rng: "random.Random") -> list:
    n = len(reference_row)
    ref_norm = _norm(reference_row)
    if ref_norm < _ZERO_TOL:
        return [0.0] * n
    raw = [rng.gauss(0.0, 1.0) for _ in range(n)]
    raw_norm = _norm(raw)
    if raw_norm < _ZERO_TOL:
        raw = [1.0] + [0.0] * (n - 1)
        raw_norm = 1.0
    scale = ref_norm / raw_norm
    return [x * scale for x in raw]


def _flatten(vectors_by_position: Mapping[int, Sequence[float]], positions: Sequence[int]) -> list:
    out: list = []
    for position in positions:
        out.extend(float(x) for x in vectors_by_position[position])
    return out


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _writable_range(hook: str, layer_count: int) -> tuple:
    if hook == "residual":
        return (1, layer_count)
    return (0, layer_count)


def _pick_any_other_layer(layer: int, lo: int, hi: int) -> "int | None":
    """A single OTHER writable layer for a single-site `shuffled_layer` control -- None when the writable
    range is too small (fewer than 2 layers) to construct one."""
    if hi - lo < 2:
        return None
    return lo if layer != lo else lo + 1


def _pick_shuffled_layers(layers: Sequence[int], usable_layers: Sequence[int]) -> "list | None":
    """`len(layers)` DIFFERENT layers, disjoint from `layers`, drawn from `usable_layers` -- the
    multi-site analogue of transplant.py's single `shuffled_layer` control. None when no disjoint set of
    the right size exists (e.g. the window already spans the entire usable range) -- omitted honestly,
    never padded with an overlapping or partial set."""
    pool = [l for l in usable_layers if l not in layers]
    if len(pool) < len(layers):
        return None
    return pool[: len(layers)]


def _tile(usable_layers: Sequence[int], window_size: int) -> list:
    return [list(usable_layers[i:i + window_size]) for i in range(0, len(usable_layers), window_size)]


# ================================================================================== engine call plumbing
# Mirrors transplant.py's own `_call_score` exactly (never raises; a failure is reported and attributable).

def _call_score(engine, label: str, **kwargs) -> dict:
    try:
        response = engine.score(**kwargs)
    except Exception as exc:      # noqa: BLE001 -- reported, never propagated
        return {"ok": False, "error": f"{label} failed: {type(exc).__name__}: {exc}"}
    if not isinstance(response, dict):
        return {"ok": False,
                "error": f"{label} failed: engine.score returned {type(response).__name__}, expected an object"}
    return {"ok": True, "response": response}


def _read_captured_multi(response: dict, field: str, layers: Sequence[int], positions: Sequence[int]) -> dict:
    """{layer: {position: [float,...] | None}} -- None marks a (layer, position) that did not land (a
    capture-armed request can still yield nothing at a given layer; see hook_vocabulary's
    known_gap_last_layer / architecture_coverage)."""
    captured = response.get(field)
    out: dict = {}
    for layer in layers:
        layer_rows = captured.get(str(layer)) if isinstance(captured, dict) else None
        row_by_position: dict = {}
        for position in positions:
            row = None
            if isinstance(layer_rows, dict):
                candidate_row = layer_rows.get(str(position))
                if isinstance(candidate_row, list):
                    row = candidate_row
            row_by_position[position] = row
        out[layer] = row_by_position
    return out


def _read_arm_metrics(response: dict, *, n_prompt: int, n_cont: int, readout_position: int,
                      target_token_id: int) -> dict:
    """What ONE /score response says about `target_token_id` at `readout_position` -- structurally
    identical to transplant.py's `_target_metrics` (duplicated for the same reason as the tiny math
    helpers above): every metric that could not be honestly read off the wire is OMITTED with a reason,
    never a guessed value."""
    metrics: dict = {}
    omitted: list = []

    sum_logprob = response.get("sum_logprob")
    if isinstance(sum_logprob, (int, float)) and not isinstance(sum_logprob, bool):
        metrics["sum_logprob"] = float(sum_logprob)
    else:
        omitted.append({"metric": "sum_logprob", "reason": "engine response carried no sum_logprob"})

    index = readout_position - n_prompt
    tokens = response.get("tokens") or []
    if not (0 <= index < n_cont) or not (0 <= index < len(tokens)):
        reason = (f"readout_position {readout_position} is outside the scored continuation range "
                 f"[{n_prompt}, {n_prompt + n_cont}), or the engine's tokens[] was shorter than expected")
        for name in ("target_token_logprob", "target_token_rank", "top1_token_id", "top1_token_piece",
                    "top1_is_target"):
            omitted.append({"metric": name, "reason": reason})
        return {"metrics": metrics, "omitted": omitted}

    entry = tokens[index] if isinstance(tokens[index], dict) else {}
    topk_list = entry.get("topk")

    if entry.get("id") == target_token_id and isinstance(entry.get("logprob"), (int, float)):
        metrics["target_token_logprob"] = float(entry["logprob"])
        if isinstance(entry.get("piece"), str):
            metrics["target_token_piece"] = entry["piece"]
    else:
        found = None
        if isinstance(topk_list, list):
            for item in topk_list:
                if isinstance(item, dict) and item.get("id") == target_token_id:
                    found = item
                    break
        if found is not None and isinstance(found.get("logprob"), (int, float)):
            metrics["target_token_logprob"] = float(found["logprob"])
            if isinstance(found.get("piece"), str):
                metrics["target_token_piece"] = found["piece"]
        else:
            omitted.append({"metric": "target_token_logprob",
                            "reason": f"target_token_id={target_token_id} is neither the forced "
                                     f"continuation token here nor present in the returned top-k"})

    rank = None
    if isinstance(topk_list, list):
        for index2, item in enumerate(topk_list):
            if isinstance(item, dict) and item.get("id") == target_token_id:
                rank = index2
                break
    if rank is not None:
        metrics["target_token_rank"] = rank
    else:
        omitted.append({"metric": "target_token_rank",
                        "reason": f"target_token_id={target_token_id} does not appear in the returned top-k"})

    top1 = topk_list[0] if isinstance(topk_list, list) and topk_list and isinstance(topk_list[0], dict) else None
    top1_id = top1.get("id") if top1 else None
    if isinstance(top1_id, int) and not isinstance(top1_id, bool):
        metrics["top1_token_id"] = top1_id
        if isinstance(top1.get("piece"), str):
            metrics["top1_token_piece"] = top1["piece"]
        metrics["top1_is_target"] = (top1_id == target_token_id)
    else:
        reason = "topk was not requested or returned empty"
        for name in ("top1_token_id", "top1_token_piece", "top1_is_target"):
            omitted.append({"metric": name, "reason": reason})

    return {"metrics": metrics, "omitted": omitted}


def _flipped_to_target(baseline_metrics: dict, arm_metrics: dict) -> "bool | None":
    baseline_hit = baseline_metrics.get("top1_is_target")
    arm_hit = arm_metrics.get("top1_is_target")
    if baseline_hit is None or arm_hit is None:
        return None
    return (not baseline_hit) and arm_hit


def _movement_results(*, baseline_metrics: dict, arm_metrics: dict,
                      reference_target_logprob: "float | None") -> dict:
    baseline_logprob = baseline_metrics.get("target_token_logprob")
    treated_logprob = arm_metrics.get("target_token_logprob")
    return {
        "reference_token_logprob_recovery": restoration_metrics.reference_token_logprob_recovery(
            reference_logprob=reference_target_logprob, baseline_logprob=baseline_logprob,
            treated_logprob=treated_logprob),
        "candidate_token_suppression": restoration_metrics.candidate_token_suppression(
            baseline_logprob=baseline_logprob, treated_logprob=treated_logprob,
            reference_logprob=reference_target_logprob),
    }


# =================================================================================== the window harness

def _run_window(*, candidate_engine, hook: str, layers: Sequence[int], depth: int,
                ref_vectors_by_layer: Mapping[int, Mapping[int, Sequence[float]]],
                self_vectors_by_layer: Mapping[int, Mapping[int, Sequence[float]]],
                usable_layers: Sequence[int], baseline_metrics: dict, positions: Sequence[int],
                prompt_ids: Sequence[int], continuation_ids: Sequence[int], n_prompt: int, n_cont: int,
                readout_position: int, target_token_id: int, topk: int, rng: "random.Random",
                reference_target_logprob: "float | None", primary_metric: str) -> dict:
    """One multi-site window test: `reference_transplant` / `candidate_self_transplant` /
    `random_equal_norm` (+ `shuffled_window` when possible) written JOINTLY across every layer in
    `layers`, in ONE forward per arm -- the composable-kind analogue of `transplant.run_site()`'s five-arm
    harness, compared against the ONE candidate baseline captured once before the whole window search
    began (not a fresh no_write_replay per window -- an efficiency choice this module makes at N-window
    granularity that `transplant.run_site()` does not need to make at 1-site granularity; disclosed here,
    not hidden)."""
    write_field = _WINDOW_WRITE_FIELD[hook]
    applied_field = _WINDOW_APPLIED_FIELD[hook]

    def _specs(vectors_by_layer, use_layers):
        return [{"layer": l, "positions": list(positions), "values": _flatten(vectors_by_layer[l], positions)}
                for l in use_layers]

    random_vectors_by_layer = {
        l: {p: _random_equal_norm_vector(ref_vectors_by_layer[l][p], rng) for p in positions}
        for l in layers
    }
    shuffled_layers = _pick_shuffled_layers(layers, usable_layers)

    arm_plan = [
        ("reference_transplant", _specs(ref_vectors_by_layer, layers)),
        ("candidate_self_transplant", _specs(self_vectors_by_layer, layers)),
        ("random_equal_norm", _specs(random_vectors_by_layer, layers)),
    ]
    if shuffled_layers is not None:
        shuffled_specs = [{"layer": dst, "positions": list(positions),
                           "values": _flatten(ref_vectors_by_layer[src], positions)}
                          for src, dst in zip(layers, shuffled_layers)]
        arm_plan.append(("shuffled_window", shuffled_specs))

    arms: dict = {}
    for name, specs in arm_plan:
        call = _call_score(candidate_engine, f"{name} window arm", prompt_ids=list(prompt_ids),
                           continuation_ids=list(continuation_ids), topk=topk, **{write_field: specs})
        if not call["ok"]:
            return {"hook": hook, "layers": list(layers), "depth": depth, "instrument_sane": False,
                   "retained": False, "reasons": [f"{name} window arm failed: {call['error']}"]}
        read = _read_arm_metrics(call["response"], n_prompt=n_prompt, n_cont=n_cont,
                                 readout_position=readout_position, target_token_id=target_token_id)
        metrics = dict(read["metrics"])
        applied = call["response"].get(applied_field)
        if isinstance(applied, bool):
            metrics["write_applied"] = applied
        arms[name] = metrics

    reasons: list = []
    self_metrics = arms["candidate_self_transplant"]
    self_applied = self_metrics.get("write_applied")
    self_top1 = self_metrics.get("top1_token_id")
    baseline_top1 = baseline_metrics.get("top1_token_id")

    if self_applied is not True:
        instrument_sane = False
        reasons.append("candidate_self_transplant's write_applied was not confirmed true for this window "
                       "-- the write path itself is not confirmed to have run.")
    elif self_top1 is None or baseline_top1 is None:
        instrument_sane = False
        reasons.append("instrument sanity could not be evaluated for this window (top-1 token missing "
                       "from the baseline or self-transplant response).")
    elif self_top1 != baseline_top1:
        instrument_sane = False
        reasons.append("candidate_self_transplant changed the top-1 token for this window -- the write "
                       "mechanism itself is not a no-op here, so no other arm's result is interpretable.")
    else:
        instrument_sane = True

    result: dict = {"hook": hook, "layers": list(layers), "depth": depth,
                    "instrument_sane": instrument_sane, "arms": arms}

    if not instrument_sane:
        result["retained"] = False
        result["reasons"] = reasons
        return result

    if baseline_top1 is not None and baseline_metrics.get("top1_is_target") is True:
        result["retained"] = False
        result["reasons"] = ["the candidate's own baseline top-1 already equals target_token_id -- there "
                             "is no disagreement for this window's transplant to correct."]
        return result

    reference_moved = _flipped_to_target(baseline_metrics, arms["reference_transplant"])
    random_moved = _flipped_to_target(baseline_metrics, arms["random_equal_norm"])
    if reference_moved is not None:
        result["moved"] = reference_moved

    if reference_moved is not None and random_moved is not None:
        beat_control = bool(reference_moved and not random_moved)
        result["beat_control"] = beat_control
        result["retained"] = beat_control
        if reference_moved and random_moved:
            reasons.append("the random equal-norm control ALSO flipped the top-1 token to target_token_id "
                           "for this window -- not reference-specific (perturbation-sensitive, not "
                           "localizing evidence).")
        elif not reference_moved:
            reasons.append("the reference transplant did not flip the top-1 token to target_token_id for "
                           "this window.")
        else:
            reasons.append("the reference transplant flipped the top-1 token to target_token_id and the "
                           "random equal-norm control did not.")
    else:
        result["retained"] = False
        reasons.append("movement could not be evaluated for this window (top-1/target-hit missing on the "
                       "reference or random arm).")

    result["reasons"] = reasons

    ref_results = _movement_results(baseline_metrics=baseline_metrics, arm_metrics=arms["reference_transplant"],
                                    reference_target_logprob=reference_target_logprob)
    rand_results = _movement_results(baseline_metrics=baseline_metrics, arm_metrics=arms["random_equal_norm"],
                                     reference_target_logprob=reference_target_logprob)
    ref_primary = restoration_metrics.select_primary(ref_results, primary_metric=primary_metric)
    rand_primary = restoration_metrics.select_primary(rand_results, primary_metric=primary_metric)
    movement_metrics: dict = {"reference_transplant": ref_primary, "random_equal_norm": rand_primary}
    if ref_primary.get("state") == "selected" and rand_primary.get("state") == "selected":
        movement_metrics["beat_control"] = restoration_metrics.beat_control(ref_primary["result"],
                                                                             rand_primary["result"])
    result["movement_metrics"] = movement_metrics

    return result


def _bisect_window(*, candidate_engine, hook: str, layers: Sequence[int], depth: int,
                   ref_vectors_by_layer, self_vectors_by_layer, usable_layers, baseline_metrics,
                   positions, prompt_ids, continuation_ids, n_prompt, n_cont, readout_position,
                   target_token_id, topk, rng, reference_target_logprob, primary_metric,
                   window_tests_out: list, leaf_layers_out: list) -> None:
    """A window of size 1 is a SITE, never tested by the window harness -- it goes straight to
    `leaf_layers_out` for single-site confirmation (see module docstring, step 4). A window of size > 1 is
    tested; if retained, it is split in half and each half recurses. `window_tests_out` therefore only
    ever contains records with `len(layers) >= 2` -- the invariant `_derive_verdict` relies on to
    distinguish a genuine window-level localization from a single-site one."""
    if len(layers) == 1:
        leaf_layers_out.append(layers[0])
        return
    result = _run_window(candidate_engine=candidate_engine, hook=hook, layers=layers, depth=depth,
                         ref_vectors_by_layer=ref_vectors_by_layer, self_vectors_by_layer=self_vectors_by_layer,
                         usable_layers=usable_layers, baseline_metrics=baseline_metrics, positions=positions,
                         prompt_ids=prompt_ids, continuation_ids=continuation_ids, n_prompt=n_prompt,
                         n_cont=n_cont, readout_position=readout_position, target_token_id=target_token_id,
                         topk=topk, rng=rng, reference_target_logprob=reference_target_logprob,
                         primary_metric=primary_metric)
    window_tests_out.append(result)
    if not result.get("retained"):
        return
    mid = len(layers) // 2
    left, right = list(layers[:mid]), list(layers[mid:])
    for half in (left, right):
        _bisect_window(candidate_engine=candidate_engine, hook=hook, layers=half, depth=depth + 1,
                       ref_vectors_by_layer=ref_vectors_by_layer, self_vectors_by_layer=self_vectors_by_layer,
                       usable_layers=usable_layers, baseline_metrics=baseline_metrics, positions=positions,
                       prompt_ids=prompt_ids, continuation_ids=continuation_ids, n_prompt=n_prompt,
                       n_cont=n_cont, readout_position=readout_position, target_token_id=target_token_id,
                       topk=topk, rng=rng, reference_target_logprob=reference_target_logprob,
                       primary_metric=primary_metric, window_tests_out=window_tests_out,
                       leaf_layers_out=leaf_layers_out)


# =================================================================================== the verdict rule

def _derive_verdict(*, window_tests: Sequence[dict], single_site_tests: Sequence[dict],
                    composable_kinds_searched: set, search_kinds: Sequence[str],
                    hooks_unavailable: Sequence[dict]) -> dict:
    """THE structural gate this module exists to make unskippable. `distributed_restoration` can only be
    built from `window_tests` entries (never `single_site_tests`), and `window_tests` is populated only
    when a composable-kind window search actually ran -- so a residual-only (or, in this slice, head-only)
    search structurally cannot reach it; the assertion below is defense-in-depth on top of that data-level
    guarantee, not the only thing preventing it."""
    observations: list = []
    for w in window_tests:
        observations.append({"kind": "window", "hook": w["hook"], "layers": w["layers"], "depth": w["depth"],
                             "instrument_sane": w["instrument_sane"], "moved": w.get("moved"),
                             "beat_control": w.get("beat_control")})
    for s in single_site_tests:
        if not s.get("ok"):
            continue
        analysis = (s.get("transplant") or {}).get("analysis", {})
        observations.append({"kind": "site", "hook": s["hook"], "layer": s["layer"],
                             "instrument_sane": analysis.get("instrument_sane"),
                             "moved": analysis.get("reference_moved_toward_reference"),
                             "beat_control": analysis.get("reference_specific")})

    unavailable_kinds = {h["hook"] for h in hooks_unavailable}
    if not observations:
        if unavailable_kinds and unavailable_kinds.issuperset(search_kinds):
            return {"label": "unavailable",
                   "reasons": [f"every requested hook kind was unavailable: "
                              f"{', '.join(sorted(unavailable_kinds))}"],
                   "evidence": {"hooks_unavailable": list(hooks_unavailable)}}
        return {"label": "inconclusive",
               "reasons": ["no window or site could be tested (no candidate layers were supplied/usable "
                          "for any requested hook kind)"],
               "evidence": {}}

    sane = [o for o in observations if o["instrument_sane"] is True]
    if not sane:
        return {"label": "inconclusive",
               "reasons": ["every tested window/site had instrument_sane=False (candidate_self_transplant "
                          "did not confirm the write mechanism is a no-op on itself) -- nothing observed "
                          "here is trustworthy enough to support a substantive verdict"],
               "evidence": {}}

    beaten = [o for o in sane if o["beat_control"] is True]
    if beaten:
        sites = [o for o in beaten if o["kind"] == "site"]
        if sites:
            return {"label": "localized_site",
                   "reasons": ["the reference arm beat the random equal-norm control at an individual "
                              "site (reference_specific=True on clozn.transplant.v1)"],
                   "evidence": {"sites": [{"hook": o["hook"], "layer": o["layer"]} for o in sites]}}

        deeper = [o for o in beaten if o["depth"] > 0]
        if deeper:
            return {"label": "localized_window",
                   "reasons": ["a window narrower than the original coarse tiling beat the random "
                              "equal-norm control; no single site within it independently did"],
                   "evidence": {"windows": [{"hook": o["hook"], "layers": o["layers"]} for o in deeper]}}

        assert composable_kinds_searched, (
            "internal invariant violated: distributed_restoration requires a composable-kind window "
            "search to have actually run (window_tests must be non-empty for this branch to be reached)")
        return {"label": "distributed_restoration",
               "reasons": ["a broad, coarse (unbisected) multi-site window beat the random equal-norm "
                          "control, and every narrower window or individual site tested inside it did not"],
               "evidence": {"windows": [{"hook": o["hook"], "layers": o["layers"]} for o in beaten]}}

    moved_only = [o for o in sane if o["moved"] is True]
    if moved_only:
        return {"label": "perturbation_sensitive",
               "reasons": ["the reference arm moved the candidate's answer toward the target at least "
                          "once, but the random equal-norm control moved it just as well every time -- "
                          "this is knife-edge sensitivity to ANY perturbation, not evidence the reference "
                          "state was uniquely correct (see docs/research/DISTRIBUTED_FUNCTION.md section "
                          "B: the prior transplant study's own overclaim, caught by this exact control)"],
               "evidence": {}}

    return {"label": "no_restoration",
           "reasons": ["instrument_sane held wherever it could be evaluated, but the reference transplant "
                      "never moved the candidate's answer toward the target at any tested window or site"],
           "evidence": {}}


# =========================================================================================== public API

def run_bisect(*, pair_compat: Mapping[str, Any], reference_loader: Callable[[], Any],
              candidate_loader: Callable[[], Any], prompt_ids: Sequence[int],
              continuation_ids: Sequence[int], write_positions: Sequence[int], readout_position: int,
              target_token_id: int, primary_metric: str, search_kinds: Sequence[str] = ("ffn",),
              window_size: int = 4, max_windows: "int | None" = None,
              residual_layers: "Sequence[int] | None" = None, head_layers: "Sequence[int] | None" = None,
              reference_target_logprob: "float | None" = None, topk: int = 5, seed: int = 0,
              store_tensors: bool = True, use_batched_screen: bool = False,
              generated_at: "str | None" = None, validate: bool = True) -> dict:
    """Run the coarse-to-fine causal bisect and build a `clozn.causal-bisect.v1` document. Returns
    `{"ok": True, "document": {...}}` on success, `{"ok": False, "error": ...}` on a preflight refusal or
    a hard engine failure during reference/baseline capture (mirrors `transplant.run_site()`'s own
    contract) -- never raises for those. Individual window/site test FAILURES during the search itself
    (an engine hiccup on one of many windows) are non-fatal: that window/site is recorded as untestable
    and the search continues, since this is a multi-call search, not a single experiment.

    `search_kinds` (subset of `residual`, `ffn`, `head`) selects which hook kinds are in play.
    `residual_layers` / `head_layers` are the ONLY candidate layers ever tested for those two kinds
    (never an implicit full-range sweep -- see the module docstring's coverage discipline); `ffn`'s
    candidate layers are always the hook's whole writable range, tiled per `window_size` and (optionally)
    capped by `max_windows` -- both bounds are recorded in the returned document's `coverage`, never
    silent. `primary_metric` must name one of `clozn.analysis.restoration_metrics.METRIC_KINDS`.
    """
    if not isinstance(pair_compat, dict):
        return {"ok": False, "error": "pair_compat must be a clozn.pair-compatibility.v1 document (dict)"}
    if not pair_compatibility.may_residual_transplant(pair_compat):
        reason = (pair_compat.get("verdict", {}).get("operations", {})
                 .get("residual_transplant", {}).get("reason") or "residual transplant is not permitted")
        return {"ok": False, "error": f"causal bisect refused: {reason}"}

    layer_count = (pair_compat.get("layer_count") or {}).get("value_b")
    if not isinstance(layer_count, int) or isinstance(layer_count, bool):
        return {"ok": False, "error": "causal bisect refused: the candidate's layer_count is unknown"}

    search_kinds = tuple(dict.fromkeys(search_kinds))
    if not search_kinds:
        return {"ok": False, "error": "search_kinds must not be empty"}
    for kind in search_kinds:
        if kind not in _ALL_HOOKS:
            return {"ok": False, "error": f"search_kinds must be a subset of {_ALL_HOOKS}, got {kind!r}"}

    positions = sorted({int(p) for p in write_positions})
    if not positions:
        return {"ok": False, "error": "write_positions must not be empty"}
    readout_position = int(readout_position)
    target_token_id = int(target_token_id)
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
        return {"ok": False, "error": "causal bisect needs topk >= 1 to read each arm's top-1 token"}
    if not isinstance(window_size, int) or isinstance(window_size, bool) or window_size < 1:
        return {"ok": False, "error": "window_size must be an integer >= 1"}
    if max_windows is not None and (not isinstance(max_windows, int) or isinstance(max_windows, bool)
                                    or max_windows < 1):
        return {"ok": False, "error": "max_windows must be a positive integer when given"}
    if not isinstance(primary_metric, str) or not primary_metric:
        return {"ok": False, "error": "primary_metric must be a non-empty string (see "
                                      "clozn.analysis.restoration_metrics.METRIC_KINDS)"}

    prompt_id_list = [int(x) for x in prompt_ids]
    continuation_id_list = [int(x) for x in continuation_ids]
    if not continuation_id_list:
        return {"ok": False, "error": "run_bisect() needs a non-empty continuation"}
    n_prompt, n_cont = len(prompt_id_list), len(continuation_id_list)

    residual_layers = sorted({int(x) for x in (residual_layers or [])})
    head_layers = sorted({int(x) for x in (head_layers or [])})

    lo_ffn, hi_ffn = _writable_range("ffn", layer_count)
    lo_res, hi_res = _writable_range("residual", layer_count)
    lo_head, hi_head = _writable_range("head", layer_count)

    hooks_unavailable: list = []
    composable_kinds_searched: set = set()
    window_tests: list = []
    leaf_layers: list = []
    single_site_tests: list = []
    coverage_layer_range: dict = {}
    bounds_applied: list = [
        f"ffn windows are tiled at window_size={window_size} layers per coarse window before any "
        f"bisection -- no window larger than this was ever tested as a single unit"
    ]

    usable_ffn_layers: list = []
    ref_vectors_by_layer: dict = {}
    self_vectors_by_layer: dict = {}
    baseline_metrics: dict = {}

    if "ffn" in search_kinds:
        candidate_ffn_layers = list(range(lo_ffn, hi_ffn))
        if not candidate_ffn_layers:
            hooks_unavailable.append({"hook": "ffn",
                                      "reason": "the writable ffn range is empty for this candidate "
                                               "(layer_count too small)"})
        else:
            with reference_loader() as reference_engine:
                ref_call = _call_score(reference_engine, "reference ffn capture", prompt_ids=prompt_id_list,
                                       continuation_ids=continuation_id_list, topk=0,
                                       ffn_capture_layers=candidate_ffn_layers, ffn_capture_positions=positions)
            if not ref_call["ok"]:
                return {"ok": False, "error": ref_call["error"]}
            ref_captured = _read_captured_multi(ref_call["response"], "ffn_captured", candidate_ffn_layers,
                                                positions)
            ref_usable = [l for l in candidate_ffn_layers
                         if all(ref_captured[l][p] is not None for p in positions)]
            if not ref_usable:
                hooks_unavailable.append({"hook": "ffn",
                                          "reason": f"ffn_out capture produced no row at any of "
                                                   f"{len(candidate_ffn_layers)} candidate layers on the "
                                                   f"REFERENCE model -- likely absent on this architecture "
                                                   f"(see hook_vocabulary's ffn_out architecture_coverage: "
                                                   f"known absent for e.g. mamba/rwkv and several MoE "
                                                   f"variants)"})
            else:
                usable_ffn_layers = ref_usable
                ref_vectors_by_layer = {l: ref_captured[l] for l in ref_usable}

    have_ffn_work = bool(usable_ffn_layers)
    have_explicit_work = bool(residual_layers) or bool(head_layers)

    if have_ffn_work or have_explicit_work:
        with candidate_loader() as candidate_engine:
            if have_ffn_work:
                baseline_call = _call_score(candidate_engine, "candidate ffn baseline",
                                            prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
                                            topk=topk, ffn_capture_layers=usable_ffn_layers,
                                            ffn_capture_positions=positions)
                if not baseline_call["ok"]:
                    return {"ok": False, "error": baseline_call["error"]}
                self_captured = _read_captured_multi(baseline_call["response"], "ffn_captured",
                                                     usable_ffn_layers, positions)
                cand_usable = [l for l in usable_ffn_layers
                              if all(self_captured[l][p] is not None for p in positions)]
                usable_ffn_layers = cand_usable
                if not usable_ffn_layers:
                    hooks_unavailable.append({"hook": "ffn",
                                              "reason": "ffn_out capture produced no row on the CANDIDATE "
                                                       "model at any layer the reference could supply"})
                else:
                    self_vectors_by_layer = {l: self_captured[l] for l in usable_ffn_layers}
                    ref_vectors_by_layer = {l: ref_vectors_by_layer[l] for l in usable_ffn_layers}
                    baseline_read = _read_arm_metrics(baseline_call["response"], n_prompt=n_prompt,
                                                      n_cont=n_cont, readout_position=readout_position,
                                                      target_token_id=target_token_id)
                    baseline_metrics = baseline_read["metrics"]
                    composable_kinds_searched.add("ffn")

                    coverage_layer_range["ffn"] = {
                        "writable_min": lo_ffn, "writable_max_exclusive": hi_ffn,
                        "usable_layers_count": len(usable_ffn_layers), "usable_layers": usable_ffn_layers,
                    }
                    tiles = _tile(usable_ffn_layers, window_size)
                    windows_before_cap = len(tiles)
                    if max_windows is not None and windows_before_cap > max_windows:
                        tiles = tiles[:max_windows]
                    windows_after_cap = len(tiles)
                    if max_windows is not None:
                        bounds_applied.append(
                            f"max_windows={max_windows}: {windows_after_cap} of {windows_before_cap} "
                            f"candidate coarse ffn windows were tested; the remaining "
                            f"{windows_before_cap - windows_after_cap} were never examined")

                    rng = random.Random(seed)
                    for tile in tiles:
                        _bisect_window(candidate_engine=candidate_engine, hook="ffn", layers=tile, depth=0,
                                       ref_vectors_by_layer=ref_vectors_by_layer,
                                       self_vectors_by_layer=self_vectors_by_layer,
                                       usable_layers=usable_ffn_layers, baseline_metrics=baseline_metrics,
                                       positions=positions, prompt_ids=prompt_id_list,
                                       continuation_ids=continuation_id_list, n_prompt=n_prompt, n_cont=n_cont,
                                       readout_position=readout_position, target_token_id=target_token_id,
                                       topk=topk, rng=rng, reference_target_logprob=reference_target_logprob,
                                       primary_metric=primary_metric, window_tests_out=window_tests,
                                       leaf_layers_out=leaf_layers)

    if "ffn" in search_kinds:
        bounds_applied.append(
            f"ffn: exactly the writable range [{lo_ffn}, {hi_ffn}) intersected with what both models "
            f"could actually capture was searched -- {len(usable_ffn_layers)} layers usable")

    for layer in sorted(set(leaf_layers)):
        shuffled = _pick_any_other_layer(layer, lo_ffn, hi_ffn)
        if shuffled is None:
            single_site_tests.append({"hook": "ffn", "layer": layer, "source": "bisection_leaf",
                                      "ok": False,
                                      "error": "the writable ffn range is too small to construct a "
                                              "shuffled_layer control"})
            continue
        site_result = transplant.run_site(
            pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
            prompt_ids=prompt_id_list, continuation_ids=continuation_id_list, site={"hook": "ffn", "layer": layer},
            shuffled_layer=shuffled, write_positions=positions, readout_position=readout_position,
            target_token_id=target_token_id, topk=topk, seed=seed, store_tensors=store_tensors,
            generated_at=generated_at, validate=False)
        if site_result["ok"]:
            single_site_tests.append({"hook": "ffn", "layer": layer, "source": "bisection_leaf", "ok": True,
                                      "transplant": site_result["document"]})
        else:
            single_site_tests.append({"hook": "ffn", "layer": layer, "source": "bisection_leaf", "ok": False,
                                      "error": site_result["error"]})

    if "residual" in search_kinds:
        coverage_layer_range["residual"] = {"writable_min": lo_res, "writable_max_exclusive": hi_res,
                                            "layers_tested": residual_layers}
        bounds_applied.append(
            f"residual sites are single-site only (never windowed -- see module docstring); exactly the "
            f"{len(residual_layers)} caller-supplied residual_layers were tested out of the writable "
            f"range [{lo_res}, {hi_res}), not an implicit full-range sweep")
        for layer in residual_layers:
            shuffled = _pick_any_other_layer(layer, lo_res, hi_res)
            if shuffled is None:
                single_site_tests.append({"hook": "residual", "layer": layer, "source": "explicit_residual",
                                          "ok": False,
                                          "error": "the writable residual range is too small to construct "
                                                  "a shuffled_layer control"})
                continue
            site_result = transplant.run_site(
                pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
                prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
                site={"hook": "residual", "layer": layer}, shuffled_layer=shuffled, write_positions=positions,
                readout_position=readout_position, target_token_id=target_token_id, topk=topk, seed=seed,
                store_tensors=store_tensors, generated_at=generated_at, validate=False)
            if site_result["ok"]:
                single_site_tests.append({"hook": "residual", "layer": layer, "source": "explicit_residual",
                                          "ok": True, "transplant": site_result["document"]})
            else:
                single_site_tests.append({"hook": "residual", "layer": layer, "source": "explicit_residual",
                                          "ok": False, "error": site_result["error"]})

    if "head" in search_kinds:
        coverage_layer_range["head"] = {"writable_min": lo_head, "writable_max_exclusive": hi_head,
                                        "layers_tested": head_layers}
        bounds_applied.append(
            f"head: exactly the {len(head_layers)} caller-supplied head_layers were attempted at the "
            f"single-site step (per-head, multi-layer window construction is not implemented in this "
            f"slice -- see module docstring)")
        for layer in head_layers:
            shuffled = _pick_any_other_layer(layer, lo_head, hi_head)
            if shuffled is None:
                single_site_tests.append({"hook": "head", "layer": layer, "source": "explicit_head",
                                          "ok": False,
                                          "error": "the writable head layer range is too small to "
                                                  "construct a shuffled_layer control"})
                continue
            site_result = transplant.run_site(
                pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
                prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
                site={"hook": "head", "layer": layer}, shuffled_layer=shuffled, write_positions=positions,
                readout_position=readout_position, target_token_id=target_token_id, topk=topk, seed=seed,
                store_tensors=store_tensors, generated_at=generated_at, validate=False)
            if site_result["ok"]:
                single_site_tests.append({"hook": "head", "layer": layer, "source": "explicit_head",
                                          "ok": True, "transplant": site_result["document"]})
            else:
                single_site_tests.append({"hook": "head", "layer": layer, "source": "explicit_head",
                                          "ok": False, "error": site_result["error"]})

    for hook in ("ffn", "residual", "head"):
        attempts = [s for s in single_site_tests if s["hook"] == hook]
        if attempts and all(not a["ok"] and "site.hook must be one of" in (a.get("error") or "")
                            for a in attempts):
            hooks_unavailable.append({"hook": hook, "reason": attempts[0]["error"]})

    screening: dict = {"requested": bool(use_batched_screen), "used": False}
    if use_batched_screen:
        screening["reason"] = (
            "the engine's batched `arms` field (routes_whitebox.cpp) only accepts RESIDUAL `write` specs "
            "today -- ffn_write/head_write cannot be screened via arms on this engine version, and this "
            "module's window search only builds ffn/head (composable) windows. Every window and site in "
            "this search was scored with a normal, sequential, non-batched /score call; nothing here came "
            "from the batched_approximate regime to separately confirm.")

    coverage = {
        "window_size": window_size,
        "layer_range_searched": coverage_layer_range,
        "bounds_applied": bounds_applied,
    }
    if max_windows is not None:
        coverage["max_windows"] = max_windows

    verdict = _derive_verdict(window_tests=window_tests, single_site_tests=single_site_tests,
                              composable_kinds_searched=composable_kinds_searched, search_kinds=search_kinds,
                              hooks_unavailable=hooks_unavailable)

    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "reference_model": dict(pair_compat.get("model_a") or {}),
        "candidate_model": dict(pair_compat.get("model_b") or {}),
        "pair_compatibility": dict(pair_compat),
        "target_token_id": target_token_id,
        "readout_position": readout_position,
        "write_positions": positions,
        "continuation": {"n_prompt": n_prompt, "n_cont": n_cont},
        "primary_metric": primary_metric,
        "seed": seed,
        "search": {
            "kinds_requested": list(search_kinds),
            "composable_kinds_searched": sorted(composable_kinds_searched),
            "window_capable_kinds": list(_WINDOW_CAPABLE_HOOKS),
            "hooks_unavailable": hooks_unavailable,
            "screening": screening,
        },
        "coverage": coverage,
        "window_tests": window_tests,
        "single_site_tests": single_site_tests,
        "verdict": verdict,
    }
    if reference_target_logprob is not None:
        document["reference_target_logprob"] = float(reference_target_logprob)
    if validate:
        schemas.validate(document)
    return {"ok": True, "document": document}
