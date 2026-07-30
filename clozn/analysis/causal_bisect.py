"""causal_bisect.py -- slice 3.5 (+ head-window addendum): the CAUSAL BISECT, a coarse-to-fine search over
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
are built from these two "composable" kinds -- `COMPOSABLE_HOOKS`.

HEAD WINDOWS: WHAT COMPOSES NOW, AND WHAT'S DELIBERATELY STILL OUT OF SCOPE
-------------------------------------------------------------------------------
`head` (`kqv_out`) was previously scoped OUT of window construction here because a correct multi-layer,
multi-head write plan needed `d_head` (only knowable from a runtime probe) and `clozn.analysis.transplant`
did not yet support `hook="head"` at all. Both blockers are gone: `transplant.py`'s `_read_head_vectors()`
reads `d_head`/`n_head` from a live `head_capture` response's own `head_dims` field -- engine ground
truth, never computed from `hidden_size // head_count` -- and `pair_compatibility.may_head_transplant()`
statically gates WHICH model pairs may even attempt a head transplant (on `head_count` matching exactly,
independent of `residual_transplant`'s `hidden_size` gate). This module mirrors that same `head_dims`
discipline for MULTI-site head capture: `_read_head_dims`/`_read_head_rows_multi`/`_slice_head` read one
`head_capture` response (covering every requested layer AND, since a captured row spans the full `[ne0]`
merged tensor, every head at that layer, client-side-sliced with zero extra engine calls) and slice each
requested `(layer, head)` site out of it -- so a SINGLE reference forward and a SINGLE candidate baseline
forward supply vectors for the WHOLE `head_layers x head_indices` grid, exactly like `ffn`'s one-forward
capture already does for its whole layer range.

Because per-head slices of `kqv_out` occupy DISJOINT ranges of the same `[ne0]` row (`head_write` lays out
head `h` at `[h*d_head, (h+1)*d_head)`, per hook_vocabulary's measured tensor layout), writing several
heads at the SAME layer in one forward is mechanically the same kind of non-colliding, independently-
applied write as writing several DIFFERENT layers -- so a head window may legitimately span multiple
layers, multiple heads within a layer, or both. `_WINDOW_CAPABLE_HOOKS` now names `("ffn", "head")`.

WHAT THIS SLICE DELIBERATELY DOES NOT DO: MIX `ffn` AND `head` SITES IN ONE JOINT WINDOW WRITE. The engine
mechanically supports it (hook_vocabulary, measured: an `ffn_write(L1) + head_write(L2)` combination in
ONE forward produced a logprob distinct from both single-surface arms, with both `*_applied` flags true),
so a mixed window is not mechanically wrong. It is left out here because `window_test.hook` (this schema)
is a single string naming ONE kind for the whole window -- representing a genuinely mixed window would
mean replacing `layers`/`heads` with a `sites: [{hook, layer, head?}, ...]` list and widening `hook` to
something like `"mixed"`, a bigger schema commitment (this schema's fields are additive-only once
released) better made deliberately in its own slice than as a side effect of this one. `ffn` and `head`
each get their OWN independent tile-and-bisect search in this slice, both populating the SAME
`window_tests` array (discriminated by `hook`) -- which already closes the stated gap ("distributed
across attention" is now measurable) without that added representational cost. Disclosed, not hidden.

COMBINATORICS: `head_layers x head_indices` IS BOUNDED, NEVER AN IMPLICIT SWEEP
------------------------------------------------------------------------------------
`ffn` sweeps its ENTIRE writable layer range automatically (a single int per layer is cheap to enumerate
exhaustively). `head` multiplies layers by attention heads -- e.g. 28 layers x 28 heads is 784 candidate
sites, and tiling+bisecting that many sites is many multiples of ffn's own call count. So `head`, like
`residual`, is NEVER implicitly swept: the caller must supply BOTH `head_layers` AND `head_indices`
(non-empty) or head is reported `hooks_unavailable` with that reason, never silently skipped. When the
Cartesian product of the two is more than one site, it is genuinely searched as a window (tiled at
`window_size` sites per coarse window, exactly like `ffn`'s layer tiles, and bisected the same way). When
it is EXACTLY one site (`len(head_layers) == len(head_indices) == 1`), no window search is meaningful --
it is tested directly via `transplant.run_site()`, `source="explicit_head"`, mirroring `residual`'s own
always-single-site path.

An OPTIONAL `max_head_sites` caps the grid further: when the usable grid exceeds it, this module keeps the
top `max_head_sites` sites by OBSERVATIONAL divergence -- the L2 distance between the reference's and the
candidate's OWN captured vectors at that site, computed from the two capture forwards already made (zero
extra engine calls) -- and drops the rest. The exact grid size, the cap (if any), and how many sites
survived are ALWAYS recorded in `coverage.bounds_applied` (and `coverage.max_head_sites` when the cap was
given) -- a truncated search that reads as exhaustive is exactly the failure mode this module exists to
avoid (see the "no silent caps" discipline already established for `max_windows`, generalized here).

THE SEARCH
------------
1. For each composable kind actually in play (`ffn` always via its full writable range; `head` only when
   the caller supplies `head_layers`/`head_indices` with more than one total site), tile the candidate
   sites into coarse windows of `window_size` sites (the LAST tile may be smaller; sizes are recorded,
   never silently rounded).
2. Test each coarse window: write the reference's captured state at EVERY site in the window, jointly, in
   ONE forward per arm (`reference_transplant`, `candidate_self_transplant`, `random_equal_norm`, and
   `shuffled_window` when a disjoint same-size site set exists) -- the same instrument-sanity +
   reference-vs-random-control structure `clozn.analysis.transplant._derive_analysis` uses at one site,
   generalized to N sites. A window is `retained` only when `instrument_sane` AND the reference arm beat
   the random equal-norm control (`beat_control`).
3. Recursively bisect every RETAINED window in half. A half of size 1 is a SITE, not a window -- it is
   handed to the single-site confirmation step, never tested by the window harness.
4. Single-site confirmation: every bisection leaf (plus any explicitly requested `residual_layers` or a
   trivial single `head_layers`/`head_indices` pair) is re-tested with `clozn.analysis.transplant.run_site()`
   DIRECTLY. Its `analysis.reference_specific` / `analysis.instrument_sane` are embedded and read verbatim
   -- this module never bypasses or recomputes them (see transplant.py's own docstring on why that field
   may only be set one way).
5. If no window/site anywhere beat control, the search reports `perturbation_sensitive` (something moved
   but a random perturbation moved it just as well) or `no_restoration` (nothing moved), never a
   localization claim -- see `_derive_verdict`.

INDEPENDENT, REPRODUCIBLE RANDOM CONTROLS AT SINGLE-SITE CONFIRMATION
------------------------------------------------------------------------
`run_bisect(seed=N)` treats N as a BASE seed, not the literal seed passed unchanged to every confirmed
site. Every `transplant.run_site()` confirmation derives its own uint64 seed from SHA-256 over canonical
JSON containing `{base_seed, source, hook, layer, head?}` (sorted keys, compact separators, UTF-8; `head`
is omitted for non-head sites; the first eight digest bytes are read unsigned big-endian). This makes
the control direction deterministic for one named site yet independent of traversal order and distinct
across different leaves/sources. Passing one seed to every leaf would make `transplant.run_site()` create
the same fresh `random.Random(seed)` at every equal-width site, reusing one frozen raw random direction
merely rescaled to each site's norm -- the second-order confound documented in
docs/research/QUANT_REGRESSION_POPULATION.md. The artifact records the derivation strategy, keeps `seed`
as the caller's base seed, and embeds each `clozn.transplant.v1` document unchanged, including its actual
derived `random_seed`.

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
actually ran, a residual-only (or a head-only-single-site) search can never produce it -- enforced by a
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
has no `ffn_write`/`head_write` shape today. Since this module's windows are built from `ffn`/`head`
(never residual, per the central design fact above), there is no sound way to route a window candidate
through `arms` without silently misapplying an ffn/head-intended vector as a residual overwrite -- exactly
the kind of silent misapplication this codebase refuses to ship. `use_batched_screen=True` is accepted and
reported (`search.screening`) but is honestly marked `used=False` with the reason above; every window/site
test in this module is a normal sequential `/score` call. The parameter and the `screening` document field
exist so a future slice that DOES wire `ffn_write`/`head_write` into `arms` can flip this on without a
schema change.

STDLIB ONLY, OMIT NEVER NULL-PAD, SEQUENTIAL MODEL ORCHESTRATION
----------------------------------------------------------------------
Same three rules as every sibling module in `clozn.analysis`: no imports beyond the standard library
(`pyproject.toml` declares `dependencies = []`); a value that cannot be honestly computed is an absent key
plus a reason, never a fabricated zero or null; and the reference model is loaded for exactly one forward
per composable kind ACTUALLY searched (`ffn` gets its own forward covering its whole layer range; `head`,
when its grid has more than one site, gets its OWN separate forward covering its whole `head_layers x
head_indices` grid -- two forwards, never combined into one, kept independent per kind), torn down, and
only then is the candidate loaded and kept resident for every composable kind's window search in one
residency (never reloaded between `ffn`'s and `head`'s window searches) -- the two 16GB-VRAM-worthy models
are never resident together. Single-site confirmation (bisection leaves, `residual_layers`, and a trivial
single `head_layers`/`head_indices` pair) calls `transplant.run_site()`, which manages its OWN
reference/candidate lifecycle per call (unavoidable reloads, one pair per confirmed site -- correctness
over throughput).
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from clozn import schemas
from clozn.analysis import pair_compatibility, restoration_metrics, transplant

SCHEMA_VERSION = "clozn.causal-bisect.v1"

_ALL_HOOKS = ("residual", "ffn", "head")
COMPOSABLE_HOOKS = ("ffn", "head")
_WINDOW_CAPABLE_HOOKS = ("ffn", "head")

_WINDOW_WRITE_FIELD = {"ffn": "ffn_write", "head": "head_write"}
_WINDOW_APPLIED_FIELD = {"ffn": "ffn_write_applied", "head": "head_write_applied"}

_ZERO_TOL = 1e-12
_SINGLE_SITE_SEED_STRATEGY = "sha256_canonical_json_uint64_be_v1"
_SINGLE_SITE_SEED_DERIVATION = {
    "strategy": _SINGLE_SITE_SEED_STRATEGY,
    "base_seed_field": "seed",
    "site_key_fields": ["source", "hook", "layer", "head_if_present"],
}


# =========================================================================================== tiny math
# Deliberately duplicated from clozn.analysis.transplant rather than importing its underscore-prefixed
# internals -- the same choice transplant.py itself made for pair_compatibility's writable-range logic
# ("duplicated here ... rather than asking pair_compatibility to grow a gate it was never asked to
# model"). Pure stdlib math over single ffn/head rows (n_embd or d_head floats), not a hot loop.

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


def _site_divergence(reference_flat: Sequence[float], candidate_flat: Sequence[float]) -> "float | None":
    """L2 distance between the reference's and the candidate's OWN captured vectors at one site --
    the observational ranking signal `max_head_sites` uses to keep the most-different sites first when
    the full `head_layers x head_indices` grid must be narrowed (see module docstring's combinatorics
    section). None when the two vectors do not even have matching lengths (never guessed)."""
    if len(reference_flat) != len(candidate_flat):
        return None
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(reference_flat, candidate_flat)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _derive_single_site_seed(base_seed: int, *, source: str, hook: str, layer: int,
                             head: "int | None" = None) -> int:
    """Derive one order-independent uint64 random-control seed for a named confirmation site.

    Do not use Python's process-randomized ``hash()`` here. The exact canonicalization and digest
    truncation are part of ``_SINGLE_SITE_SEED_STRATEGY``'s persisted contract; changing either requires
    a new strategy name so an old artifact remains reproducible.
    """
    key = {
        "base_seed": base_seed,
        "hook": hook,
        "layer": int(layer),
        "source": source,
    }
    if head is not None:
        key["head"] = int(head)
    canonical = json.dumps(key, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return int.from_bytes(hashlib.sha256(canonical).digest()[:8], byteorder="big", signed=False)


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


def _pick_shuffled_sites(sites: Sequence, usable_sites: Sequence) -> "list | None":
    """`len(sites)` DIFFERENT sites, disjoint from `sites`, drawn from `usable_sites` -- the multi-site
    analogue of transplant.py's single `shuffled_layer` control. A "site" is an int (ffn: a layer) or an
    `(layer, head)` tuple (head) -- this function only needs sites to be hashable and comparable for
    equality, so the same logic serves both kinds. None when no disjoint set of the right size exists
    (e.g. the window already spans the entire usable range) -- omitted honestly, never padded with an
    overlapping or partial set."""
    pool = [s for s in usable_sites if s not in sites]
    if len(pool) < len(sites):
        return None
    return pool[: len(sites)]


def _tile(usable_sites: Sequence, window_size: int) -> list:
    return [list(usable_sites[i:i + window_size]) for i in range(0, len(usable_sites), window_size)]


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
    known_gap_last_layer / architecture_coverage). ffn only -- see the head-specific readers below for
    the "head" hook's own (structurally different) capture-response shape."""
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


def _read_head_dims(response: dict) -> "tuple | None":
    """(d_head, n_head), read from a head_capture response's own `head_dims` -- engine ground truth,
    never computed from hidden_size // head_count (mirrors transplant.py's `_read_head_vectors`, which
    makes the identical choice for a single head). None when the probe reports d_head<=0 (the "division
    did not divide evenly, applies nothing" case) or head_dims is absent/malformed -- never guessed."""
    head_dims = response.get("head_dims")
    d_head = head_dims.get("d_head") if isinstance(head_dims, dict) else None
    n_head = head_dims.get("n_head") if isinstance(head_dims, dict) else None
    if not isinstance(d_head, int) or isinstance(d_head, bool) or d_head <= 0:
        return None
    if not isinstance(n_head, int) or isinstance(n_head, bool) or n_head <= 0:
        return None
    return (d_head, n_head)


def _read_head_rows_multi(response: dict, layers: Sequence[int], positions: Sequence[int]) -> dict:
    """{layer: {position: [float,...] (length ne0) | None}} -- the FULL per-position merged row across
    every head, straight off `head_rows` (requires `head_capture_rows=True`). Head slicing happens later,
    client-side, once per (layer, head) site actually needed via `_slice_head` -- one capture response,
    covering every requested layer, already carries every head's data (mirrors ffn's one-response, every-
    layer capture; generalizes transplant.py's own single-head `_read_head_vectors` to many layers)."""
    head_rows = response.get("head_rows")
    out: dict = {}
    for layer in layers:
        layer_rows = head_rows.get(str(layer)) if isinstance(head_rows, dict) else None
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


def _slice_head(full_row: "list | None", head: int, d_head: int, n_head: int) -> "list | None":
    """`full_row`'s `[head*d_head, (head+1)*d_head)` slice -- exactly mirroring transplant.py's
    `_read_head_vectors`, generalized to be called once per (layer, head) site against an already-fetched
    multi-layer row. None when `full_row` is missing, `head` is outside `[0, n_head)`, or the row is
    shorter than the slice needs -- never a guessed partial slice."""
    if full_row is None or not (0 <= head < n_head):
        return None
    lo, hi = head * d_head, (head + 1) * d_head
    if len(full_row) < hi:
        return None
    return [float(x) for x in full_row[lo:hi]]


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

def _write_spec_for_site(hook: str, site, vectors_by_site: Mapping, positions: Sequence[int]) -> dict:
    """One write spec dict for one site. `site` is an int (ffn: a layer) for hook=="ffn", or an
    `(layer, head)` tuple for hook=="head" -- the "head" key is only present in the latter case, matching
    transplant.py's own `_write_kwargs` (a spec carries "head" only when hook=="head")."""
    values = _flatten(vectors_by_site[site], positions)
    if hook == "head":
        layer, head = site
        return {"layer": layer, "head": head, "positions": list(positions), "values": values}
    return {"layer": site, "positions": list(positions), "values": values}


def _site_layer(hook: str, site) -> int:
    return site[0] if hook == "head" else site


def _run_window(*, candidate_engine, hook: str, sites: Sequence, depth: int,
                ref_vectors_by_site: Mapping, self_vectors_by_site: Mapping,
                usable_sites: Sequence, baseline_metrics: dict, positions: Sequence[int],
                prompt_ids: Sequence[int], continuation_ids: Sequence[int], n_prompt: int, n_cont: int,
                readout_position: int, target_token_id: int, topk: int, rng: "random.Random",
                reference_target_logprob: "float | None", primary_metric: str) -> dict:
    """One multi-site window test: `reference_transplant` / `candidate_self_transplant` /
    `random_equal_norm` (+ `shuffled_window` when possible) written JOINTLY across every site in `sites`,
    in ONE forward per arm -- the composable-kind analogue of `transplant.run_site()`'s five-arm harness,
    compared against the ONE candidate baseline captured once before the whole window search began (not a
    fresh no_write_replay per window -- an efficiency choice this module makes at N-site granularity that
    `transplant.run_site()` does not need to make at 1-site granularity; disclosed here, not hidden). A
    "site" is an int (ffn: a layer) or an `(layer, head)` tuple (head); this function is generic over
    both via `_write_spec_for_site`/`_site_layer`."""
    write_field = _WINDOW_WRITE_FIELD[hook]
    applied_field = _WINDOW_APPLIED_FIELD[hook]

    def _specs(vectors_by_site, use_sites):
        return [_write_spec_for_site(hook, s, vectors_by_site, positions) for s in use_sites]

    random_vectors_by_site = {
        s: {p: _random_equal_norm_vector(ref_vectors_by_site[s][p], rng) for p in positions}
        for s in sites
    }
    shuffled_sites = _pick_shuffled_sites(sites, usable_sites)

    arm_plan = [
        ("reference_transplant", _specs(ref_vectors_by_site, sites)),
        ("candidate_self_transplant", _specs(self_vectors_by_site, sites)),
        ("random_equal_norm", _specs(random_vectors_by_site, sites)),
    ]
    if shuffled_sites is not None:
        shuffled_vectors_by_dst = {dst: ref_vectors_by_site[src] for src, dst in zip(sites, shuffled_sites)}
        shuffled_specs = _specs(shuffled_vectors_by_dst, shuffled_sites)
        arm_plan.append(("shuffled_window", shuffled_specs))

    arms: dict = {}
    for name, specs in arm_plan:
        call = _call_score(candidate_engine, f"{name} window arm", prompt_ids=list(prompt_ids),
                           continuation_ids=list(continuation_ids), topk=topk, **{write_field: specs})
        if not call["ok"]:
            return {"hook": hook, "layers": [_site_layer(hook, s) for s in sites], "depth": depth,
                   "instrument_sane": False, "retained": False,
                   "reasons": [f"{name} window arm failed: {call['error']}"]}
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

    result: dict = {"hook": hook, "layers": [_site_layer(hook, s) for s in sites], "depth": depth,
                    "instrument_sane": instrument_sane, "arms": arms}
    if hook == "head":
        result["heads"] = [s[1] for s in sites]

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


def _bisect_window(*, candidate_engine, hook: str, sites: Sequence, depth: int,
                   ref_vectors_by_site, self_vectors_by_site, usable_sites, baseline_metrics,
                   positions, prompt_ids, continuation_ids, n_prompt, n_cont, readout_position,
                   target_token_id, topk, rng, reference_target_logprob, primary_metric,
                   window_tests_out: list, leaf_sites_out: list) -> None:
    """A window of size 1 is a SITE, never tested by the window harness -- it goes straight to
    `leaf_sites_out` for single-site confirmation (see module docstring, step 4). A window of size > 1 is
    tested; if retained, it is split in half and each half recurses. `window_tests_out` therefore only
    ever contains records with `len(sites) >= 2` -- the invariant `_derive_verdict` relies on to
    distinguish a genuine window-level localization from a single-site one."""
    if len(sites) == 1:
        leaf_sites_out.append(sites[0])
        return
    result = _run_window(candidate_engine=candidate_engine, hook=hook, sites=sites, depth=depth,
                         ref_vectors_by_site=ref_vectors_by_site, self_vectors_by_site=self_vectors_by_site,
                         usable_sites=usable_sites, baseline_metrics=baseline_metrics, positions=positions,
                         prompt_ids=prompt_ids, continuation_ids=continuation_ids, n_prompt=n_prompt,
                         n_cont=n_cont, readout_position=readout_position, target_token_id=target_token_id,
                         topk=topk, rng=rng, reference_target_logprob=reference_target_logprob,
                         primary_metric=primary_metric)
    window_tests_out.append(result)
    if not result.get("retained"):
        return
    mid = len(sites) // 2
    left, right = list(sites[:mid]), list(sites[mid:])
    for half in (left, right):
        _bisect_window(candidate_engine=candidate_engine, hook=hook, sites=half, depth=depth + 1,
                       ref_vectors_by_site=ref_vectors_by_site, self_vectors_by_site=self_vectors_by_site,
                       usable_sites=usable_sites, baseline_metrics=baseline_metrics, positions=positions,
                       prompt_ids=prompt_ids, continuation_ids=continuation_ids, n_prompt=n_prompt,
                       n_cont=n_cont, readout_position=readout_position, target_token_id=target_token_id,
                       topk=topk, rng=rng, reference_target_logprob=reference_target_logprob,
                       primary_metric=primary_metric, window_tests_out=window_tests_out,
                       leaf_sites_out=leaf_sites_out)


# =================================================================================== the verdict rule

def _derive_verdict(*, window_tests: Sequence[dict], single_site_tests: Sequence[dict],
                    composable_kinds_searched: set, search_kinds: Sequence[str],
                    hooks_unavailable: Sequence[dict]) -> dict:
    """THE structural gate this module exists to make unskippable. `distributed_restoration` can only be
    built from `window_tests` entries (never `single_site_tests`), and `window_tests` is populated only
    when a composable-kind window search actually ran -- so a residual-only (or a head grid that reduced
    to a single explicit site) search structurally cannot reach it; the assertion below is defense-in-
    depth on top of that data-level guarantee, not the only thing preventing it."""
    observations: list = []
    for w in window_tests:
        obs = {"kind": "window", "hook": w["hook"], "layers": w["layers"], "depth": w["depth"],
              "instrument_sane": w["instrument_sane"], "moved": w.get("moved"),
              "beat_control": w.get("beat_control")}
        if w.get("heads") is not None:
            obs["heads"] = w["heads"]
        observations.append(obs)
    for s in single_site_tests:
        if not s.get("ok"):
            continue
        analysis = (s.get("transplant") or {}).get("analysis", {})
        obs = {"kind": "site", "hook": s["hook"], "layer": s["layer"],
              "instrument_sane": analysis.get("instrument_sane"),
              "moved": analysis.get("reference_moved_toward_reference"),
              "beat_control": analysis.get("reference_specific")}
        if "head" in s:
            obs["head"] = s["head"]
        observations.append(obs)

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
                   "evidence": {"sites": [
                       {"hook": o["hook"], "layer": o["layer"],
                        **({"head": o["head"]} if o.get("head") is not None else {})}
                       for o in sites]}}

        deeper = [o for o in beaten if o["depth"] > 0]
        if deeper:
            return {"label": "localized_window",
                   "reasons": ["a window narrower than the original coarse tiling beat the random "
                              "equal-norm control; no single site within it independently did"],
                   "evidence": {"windows": [
                       {"hook": o["hook"], "layers": o["layers"],
                        **({"heads": o["heads"]} if o.get("heads") is not None else {})}
                       for o in deeper]}}

        assert composable_kinds_searched, (
            "internal invariant violated: distributed_restoration requires a composable-kind window "
            "search to have actually run (window_tests must be non-empty for this branch to be reached)")
        return {"label": "distributed_restoration",
               "reasons": ["a broad, coarse (unbisected) multi-site window beat the random equal-norm "
                          "control, and every narrower window or individual site tested inside it did not"],
               "evidence": {"windows": [
                   {"hook": o["hook"], "layers": o["layers"],
                    **({"heads": o["heads"]} if o.get("heads") is not None else {})}
                   for o in beaten]}}

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
              head_indices: "Sequence[int] | None" = None, max_head_sites: "int | None" = None,
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
    `residual_layers` are the ONLY candidate layers ever tested for `residual` (never an implicit
    full-range sweep -- see the module docstring's coverage discipline). `head_layers`/`head_indices`
    together are the ONLY candidate (layer, head) sites ever tested for `head` -- BOTH must be supplied
    (non-empty) or `head` is reported unavailable; their Cartesian product forms the candidate grid,
    optionally narrowed by `max_head_sites` (top-N by observational divergence) before being tiled and
    bisected exactly like `ffn`'s layer windows, UNLESS the grid is exactly one site, in which case it is
    tested directly with no window search. `ffn`'s candidate layers are always the hook's whole writable
    range, tiled per `window_size` and (optionally) capped by `max_windows` -- every bound applied to
    either kind is recorded in the returned document's `coverage`, never silent. `primary_metric` must
    name one of `clozn.analysis.restoration_metrics.METRIC_KINDS`.
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
    if max_head_sites is not None and (not isinstance(max_head_sites, int) or isinstance(max_head_sites, bool)
                                       or max_head_sites < 1):
        return {"ok": False, "error": "max_head_sites must be a positive integer when given"}
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
    head_indices = sorted({int(x) for x in (head_indices or [])})

    lo_ffn, hi_ffn = _writable_range("ffn", layer_count)
    lo_res, hi_res = _writable_range("residual", layer_count)
    lo_head, hi_head = _writable_range("head", layer_count)

    hooks_unavailable: list = []
    composable_kinds_searched: set = set()
    window_tests: list = []
    leaf_layers: list = []
    leaf_head_sites: list = []
    single_site_tests: list = []
    coverage_layer_range: dict = {}
    bounds_applied: list = [
        f"kinds_requested={list(search_kinds)}: only these hook kind(s) were searched this run (never an "
        f"implicit sweep of every kind clozn knows about)"
    ]
    if "ffn" in search_kinds:
        bounds_applied.append(
            f"ffn windows are tiled at window_size={window_size} layers per coarse window before any "
            f"bisection -- no window larger than this was ever tested as a single unit")
    if "head" in search_kinds:
        bounds_applied.append(
            f"head windows are tiled at window_size={window_size} (layer, head) sites per coarse window "
            f"before any bisection -- no window larger than this was ever tested as a single unit")

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

    # ---- head: preflight gate (permission + caller-supplied grid), then (when the grid has more than one
    # site) a SEPARATE reference forward capturing the whole head_layers x head_indices grid in one call.
    explicit_head_site: "tuple | None" = None
    head_grid: list = []
    ref_vectors_by_head_site: dict = {}
    self_vectors_by_head_site: dict = {}
    head_baseline_metrics: dict = {}
    search_head_sites: list = []

    if "head" in search_kinds:
        if not pair_compatibility.may_head_transplant(pair_compat):
            reason = (pair_compat.get("verdict", {}).get("operations", {})
                     .get("head_transplant", {}).get("reason") or "head transplant is not permitted "
                                                                   "for this pair")
            hooks_unavailable.append({"hook": "head", "reason": reason})
        elif not head_layers or not head_indices:
            hooks_unavailable.append({"hook": "head",
                                      "reason": "head_layers and head_indices must both be supplied "
                                               "(non-empty) to search head sites -- a head site needs "
                                               "both a layer and a head index, and (like residual) head "
                                               "is never implicitly swept across the full architecture"})
        elif len(head_layers) == 1 and len(head_indices) == 1:
            explicit_head_site = (head_layers[0], head_indices[0])
        else:
            full_grid = [(l, h) for l in head_layers for h in head_indices]
            with reference_loader() as reference_engine:
                ref_call = _call_score(reference_engine, "reference head capture", prompt_ids=prompt_id_list,
                                       continuation_ids=continuation_id_list, topk=0,
                                       head_capture_layers=head_layers, head_capture_positions=positions,
                                       head_capture_rows=True)
            if not ref_call["ok"]:
                return {"ok": False, "error": ref_call["error"]}
            dims = _read_head_dims(ref_call["response"])
            if dims is None:
                hooks_unavailable.append({"hook": "head",
                                          "reason": "head_capture produced no usable head_dims on the "
                                                   "REFERENCE model (d_head could not be probed -- see "
                                                   "hook_vocabulary's d_head_probe)"})
            else:
                d_head, n_head = dims
                ref_rows = _read_head_rows_multi(ref_call["response"], head_layers, positions)
                grid_vectors: dict = {}
                for site in full_grid:
                    layer, head = site
                    vectors = {p: _slice_head(ref_rows[layer][p], head, d_head, n_head) for p in positions}
                    if all(v is not None for v in vectors.values()):
                        grid_vectors[site] = vectors
                head_grid = [site for site in full_grid if site in grid_vectors]
                if not head_grid:
                    hooks_unavailable.append({"hook": "head",
                                              "reason": f"head_capture produced no row at any of "
                                                       f"{len(full_grid)} candidate (layer, head) sites "
                                                       f"on the REFERENCE model"})
                else:
                    ref_vectors_by_head_site = grid_vectors

    have_ffn_work = bool(usable_ffn_layers)
    have_head_window_work = bool(head_grid)

    if have_ffn_work or have_head_window_work:
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
                        _bisect_window(candidate_engine=candidate_engine, hook="ffn", sites=tile, depth=0,
                                       ref_vectors_by_site=ref_vectors_by_layer,
                                       self_vectors_by_site=self_vectors_by_layer,
                                       usable_sites=usable_ffn_layers, baseline_metrics=baseline_metrics,
                                       positions=positions, prompt_ids=prompt_id_list,
                                       continuation_ids=continuation_id_list, n_prompt=n_prompt, n_cont=n_cont,
                                       readout_position=readout_position, target_token_id=target_token_id,
                                       topk=topk, rng=rng, reference_target_logprob=reference_target_logprob,
                                       primary_metric=primary_metric, window_tests_out=window_tests,
                                       leaf_sites_out=leaf_layers)

            if have_head_window_work:
                baseline_call = _call_score(candidate_engine, "candidate head baseline",
                                            prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
                                            topk=topk, head_capture_layers=head_layers,
                                            head_capture_positions=positions, head_capture_rows=True)
                if not baseline_call["ok"]:
                    return {"ok": False, "error": baseline_call["error"]}
                self_dims = _read_head_dims(baseline_call["response"])
                if self_dims is None:
                    hooks_unavailable.append({"hook": "head",
                                              "reason": "head_capture produced no usable head_dims on the "
                                                       "CANDIDATE model (d_head could not be probed)"})
                else:
                    self_d_head, self_n_head = self_dims
                    self_rows = _read_head_rows_multi(baseline_call["response"], head_layers, positions)
                    cand_usable_sites = []
                    self_vectors: dict = {}
                    for site in head_grid:
                        layer, head = site
                        vectors = {p: _slice_head(self_rows[layer][p], head, self_d_head, self_n_head)
                                  for p in positions}
                        if all(v is not None for v in vectors.values()):
                            self_vectors[site] = vectors
                            cand_usable_sites.append(site)
                    if not cand_usable_sites:
                        hooks_unavailable.append({"hook": "head",
                                                  "reason": "head_capture produced no row on the "
                                                           "CANDIDATE model at any (layer, head) site the "
                                                           "reference could supply"})
                    else:
                        self_vectors_by_head_site = self_vectors
                        ref_vectors_by_head_site = {s: ref_vectors_by_head_site[s] for s in cand_usable_sites}
                        baseline_read = _read_arm_metrics(baseline_call["response"], n_prompt=n_prompt,
                                                          n_cont=n_cont, readout_position=readout_position,
                                                          target_token_id=target_token_id)
                        head_baseline_metrics = baseline_read["metrics"]
                        composable_kinds_searched.add("head")

                        coverage_layer_range["head"] = {
                            "writable_min": lo_head, "writable_max_exclusive": hi_head,
                            "head_layers_requested": head_layers, "head_indices_requested": head_indices,
                            "usable_sites_count": len(cand_usable_sites),
                        }

                        search_head_sites = list(cand_usable_sites)
                        if max_head_sites is not None and len(search_head_sites) > max_head_sites:
                            def _rank_key(site):
                                divergence = _site_divergence(
                                    _flatten(ref_vectors_by_head_site[site], positions),
                                    _flatten(self_vectors_by_head_site[site], positions))
                                d = divergence if divergence is not None else -1.0
                                return (-d, site[0], site[1])

                            ranked = sorted(search_head_sites, key=_rank_key)
                            kept = set(ranked[:max_head_sites])
                            bounds_applied.append(
                                f"max_head_sites={max_head_sites}: kept the top {max_head_sites} of "
                                f"{len(search_head_sites)} usable head sites by observational "
                                f"reference-vs-candidate L2 divergence; the remaining "
                                f"{len(search_head_sites) - max_head_sites} were never tested")
                            search_head_sites = [s for s in cand_usable_sites if s in kept]

                        tiles = _tile(search_head_sites, window_size)
                        windows_before_cap = len(tiles)
                        if max_windows is not None and windows_before_cap > max_windows:
                            tiles = tiles[:max_windows]
                        windows_after_cap = len(tiles)
                        if max_windows is not None:
                            bounds_applied.append(
                                f"max_windows={max_windows}: {windows_after_cap} of {windows_before_cap} "
                                f"candidate coarse head windows were tested; the remaining "
                                f"{windows_before_cap - windows_after_cap} were never examined")

                        head_rng = random.Random(seed)
                        for tile in tiles:
                            _bisect_window(candidate_engine=candidate_engine, hook="head", sites=tile, depth=0,
                                           ref_vectors_by_site=ref_vectors_by_head_site,
                                           self_vectors_by_site=self_vectors_by_head_site,
                                           usable_sites=search_head_sites, baseline_metrics=head_baseline_metrics,
                                           positions=positions, prompt_ids=prompt_id_list,
                                           continuation_ids=continuation_id_list, n_prompt=n_prompt,
                                           n_cont=n_cont, readout_position=readout_position,
                                           target_token_id=target_token_id, topk=topk, rng=head_rng,
                                           reference_target_logprob=reference_target_logprob,
                                           primary_metric=primary_metric, window_tests_out=window_tests,
                                           leaf_sites_out=leaf_head_sites)

    if "ffn" in search_kinds:
        bounds_applied.append(
            f"ffn: exactly the writable range [{lo_ffn}, {hi_ffn}) intersected with what both models "
            f"could actually capture was searched -- {len(usable_ffn_layers)} layers usable")

    for layer in sorted(set(leaf_layers)):
        source = "bisection_leaf"
        shuffled = _pick_any_other_layer(layer, lo_ffn, hi_ffn)
        if shuffled is None:
            single_site_tests.append({"hook": "ffn", "layer": layer, "source": source,
                                      "ok": False,
                                      "error": "the writable ffn range is too small to construct a "
                                              "shuffled_layer control"})
            continue
        leaf_seed = _derive_single_site_seed(
            seed, source=source, hook="ffn", layer=layer)
        site_result = transplant.run_site(
            pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
            prompt_ids=prompt_id_list, continuation_ids=continuation_id_list, site={"hook": "ffn", "layer": layer},
            shuffled_layer=shuffled, write_positions=positions, readout_position=readout_position,
            target_token_id=target_token_id, topk=topk, seed=leaf_seed, store_tensors=store_tensors,
            generated_at=generated_at, validate=False)
        if site_result["ok"]:
            single_site_tests.append({"hook": "ffn", "layer": layer, "source": source, "ok": True,
                                      "transplant": site_result["document"]})
        else:
            single_site_tests.append({"hook": "ffn", "layer": layer, "source": source, "ok": False,
                                      "error": site_result["error"]})

    for layer, head in sorted(set(leaf_head_sites)):
        source = "bisection_leaf"
        shuffled = _pick_any_other_layer(layer, lo_head, hi_head)
        if shuffled is None:
            single_site_tests.append({"hook": "head", "layer": layer, "head": head, "source": source,
                                      "ok": False,
                                      "error": "the writable head layer range is too small to construct "
                                              "a shuffled_layer control"})
            continue
        leaf_seed = _derive_single_site_seed(
            seed, source=source, hook="head", layer=layer, head=head)
        site_result = transplant.run_site(
            pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
            prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
            site={"hook": "head", "layer": layer, "head": head}, shuffled_layer=shuffled,
            write_positions=positions, readout_position=readout_position, target_token_id=target_token_id,
            topk=topk, seed=leaf_seed, store_tensors=store_tensors, generated_at=generated_at,
            validate=False)
        if site_result["ok"]:
            single_site_tests.append({"hook": "head", "layer": layer, "head": head, "source": source,
                                      "ok": True, "transplant": site_result["document"]})
        else:
            single_site_tests.append({"hook": "head", "layer": layer, "head": head, "source": source,
                                      "ok": False, "error": site_result["error"]})

    if "residual" in search_kinds:
        coverage_layer_range["residual"] = {"writable_min": lo_res, "writable_max_exclusive": hi_res,
                                            "layers_tested": residual_layers}
        bounds_applied.append(
            f"residual sites are single-site only (never windowed -- see module docstring); exactly the "
            f"{len(residual_layers)} caller-supplied residual_layers were tested out of the writable "
            f"range [{lo_res}, {hi_res}), not an implicit full-range sweep")
        for layer in residual_layers:
            source = "explicit_residual"
            shuffled = _pick_any_other_layer(layer, lo_res, hi_res)
            if shuffled is None:
                single_site_tests.append({"hook": "residual", "layer": layer, "source": source,
                                          "ok": False,
                                          "error": "the writable residual range is too small to construct "
                                                  "a shuffled_layer control"})
                continue
            leaf_seed = _derive_single_site_seed(
                seed, source=source, hook="residual", layer=layer)
            site_result = transplant.run_site(
                pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
                prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
                site={"hook": "residual", "layer": layer}, shuffled_layer=shuffled, write_positions=positions,
                readout_position=readout_position, target_token_id=target_token_id, topk=topk, seed=leaf_seed,
                store_tensors=store_tensors, generated_at=generated_at, validate=False)
            if site_result["ok"]:
                single_site_tests.append({"hook": "residual", "layer": layer, "source": source,
                                          "ok": True, "transplant": site_result["document"]})
            else:
                single_site_tests.append({"hook": "residual", "layer": layer, "source": source,
                                          "ok": False, "error": site_result["error"]})

    if explicit_head_site is not None:
        layer, head = explicit_head_site
        source = "explicit_head"
        shuffled = _pick_any_other_layer(layer, lo_head, hi_head)
        if shuffled is None:
            single_site_tests.append({"hook": "head", "layer": layer, "head": head, "source": source,
                                      "ok": False,
                                      "error": "the writable head layer range is too small to construct "
                                              "a shuffled_layer control"})
        else:
            leaf_seed = _derive_single_site_seed(
                seed, source=source, hook="head", layer=layer, head=head)
            site_result = transplant.run_site(
                pair_compat=pair_compat, reference_loader=reference_loader, candidate_loader=candidate_loader,
                prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
                site={"hook": "head", "layer": layer, "head": head}, shuffled_layer=shuffled,
                write_positions=positions, readout_position=readout_position, target_token_id=target_token_id,
                topk=topk, seed=leaf_seed, store_tensors=store_tensors, generated_at=generated_at,
                validate=False)
            if site_result["ok"]:
                single_site_tests.append({"hook": "head", "layer": layer, "head": head,
                                          "source": source, "ok": True,
                                          "transplant": site_result["document"]})
            else:
                single_site_tests.append({"hook": "head", "layer": layer, "head": head,
                                          "source": source, "ok": False,
                                          "error": site_result["error"]})

    if "head" in search_kinds:
        head_unavailable_entry = next((h for h in hooks_unavailable if h["hook"] == "head"), None)
        if explicit_head_site is not None:
            coverage_layer_range["head"] = {
                "writable_min": lo_head, "writable_max_exclusive": hi_head,
                "head_layers_requested": head_layers, "head_indices_requested": head_indices,
                "mode": "explicit_single_site",
            }
            bounds_applied.append(
                "head: head_layers/head_indices each had exactly one entry -- a single (layer, head) "
                "site was tested directly (source=explicit_head), no window search")
        elif head_unavailable_entry is not None:
            bounds_applied.append(f"head: not searched this run -- {head_unavailable_entry['reason']}")
        else:
            total_grid = len(head_layers) * len(head_indices)
            bounds_applied.append(
                f"head: candidate grid was {len(head_layers)} head_layers x {len(head_indices)} "
                f"head_indices = {total_grid} (layer, head) sites (caller-supplied, never an implicit "
                f"full-architecture sweep); {len(search_head_sites)} were usable on both models and "
                f"searched")

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
    if max_head_sites is not None:
        coverage["max_head_sites"] = max_head_sites

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
        "single_site_seed_derivation": dict(_SINGLE_SITE_SEED_DERIVATION),
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
