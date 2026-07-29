"""transplant.py -- slice 3.4: the CONTROLLED TRANSPLANT primitive.

Take the reference model's internal state at ONE site (a residual `l_out-<layer>` row or an
`ffn_out-<layer>` MLP-contribution row) and write it into the candidate model, then observe whether the
candidate's own answer moves toward the reference's -- alongside the four controls that make a "yes"
mean something instead of an overclaim. This is the foundation the causal bisect (a LATER slice) will
search over; this module runs and reports ONE site's five arms, nothing more.

THE FIVE CONTROLS
------------------
For every candidate site, `run_site()` runs all five arms, always, in this order:

  1. reference_transplant     -- candidate receives the reference's captured state at the site.
  2. candidate_self_transplant -- candidate receives its OWN captured state at the site. Proves the
     write path itself does not change the answer. If THIS arm moves the output, the instrument is
     broken and every other arm is uninterpretable -- see `_derive_analysis`'s `instrument_sane`.
  3. random_equal_norm        -- a random direction, scaled to the reference vector's own L2 norm,
     written at the SAME site. Detects perturbation-sensitive knife edges.
  4. shuffled_layer           -- the SAME reference vector, written at a DIFFERENT, dimensionally-
     compatible layer.
  5. no_write_replay          -- a plain repeat of the baseline call, no write at all. Establishes
     ordinary execution stability.

THE RULE, MADE STRUCTURAL
---------------------------
A site is NOT reference-specific merely because SOME perturbation flips the candidate's top-1 answer to
the reference's. It is reference-specific only if the reference arm flips the answer toward the
reference AND the random-equal-norm control does NOT. This project's own prior transplant study
(scripts/tracer/transplant_localize.py, docs/research/DISTRIBUTED_FUNCTION.md section B) found FP
transplant "fixed" 5/12 quantization flips -- until the random-equal-norm control showed it also fixed
3/12, leaving only 3/12 genuinely reference-specific: "the control corrected the first run's overclaim."
`_derive_analysis()` computes `reference_specific` from exactly this rule and is the ONLY place that
field is set -- a caller cannot report a site as reference-specific by skipping the control, because the
control's result is a required input to the computation, not an optional add-on.

`instrument_sane` is checked FIRST and gates everything else: if candidate_self_transplant does not
reproduce baseline (or any write-bearing arm's write did not confirm `*_applied: true`), the document
reports `instrument_sane: false` and OMITS `reference_specific` entirely -- never a default of False or
True standing in for "we couldn't tell."

This module never uses the words "localize"/"localized"/"localizing" or "distributed" anywhere --
that vocabulary belongs to the LATER slice that actually searches across sites for a causal account.
Everything here is scoped to ONE named site's five measured arms.

WHY pair_compatibility's `residual_transplant` GATES BOTH HOOK KINDS
-----------------------------------------------------------------------
`clozn.analysis.pair_compatibility` (read-only reused here, never modified) models exactly ONE transplant
operation, named `residual_transplant`, gated on `hidden_size` matching exactly -- because the engine's
write validation (`values.size() == positions.size() * n_embd` of the TARGET engine, no projection layer
anywhere) is the same mechanical fact regardless of WHICH n_embd-wide hook is being written. `ffn_write`
(this module's other supported site kind) carries the IDENTICAL n_embd-width, position-major contract as
residual `write` (see clozn/receipts/hook_vocabulary.py's `_FFN_OUT.ffn_write`) -- the only thing that
differs between the two hooks is the writable LAYER RANGE (`[1, n_layer)` for residual, reserving layer 0
as the read tap's sentinel; `[0, n_layer)` for ffn_out, which has no such reservation), which this module
computes directly in `_writable_range` rather than asking pair_compatibility to grow a third gate it was
never asked to model. Per-head `kqv_out` sites are OUT OF SCOPE for this module: `head_write`'s value
width is `d_head`, only known from a runtime probe (see hook_vocabulary's `d_head_probe`), not statically
the way `write`/`ffn_write`'s n_embd width is -- a real gap, not a silent omission, left for whichever
slice next needs per-head transplant.

SEQUENCING -- ONE MODEL RESIDENT AT A TIME
--------------------------------------------
Exactly like `clozn.analysis.mechanistic_diff.compare()`: `reference_loader`/`candidate_loader` are
zero-argument callables returning a context manager yielding something with `.score(...)`. The reference
is loaded for ONE forward (captures the site's vector at `write_positions`), then torn down completely
BEFORE the candidate is ever loaded. The candidate then stays resident for baseline + all five arms (six
/score calls total) -- this box has 16GB VRAM and the two GGUFs are never assumed to fit together.

WRITE SPECS REFERENCE TENSORS BY CONTENT ADDRESS
----------------------------------------------------
Every vector this module writes (the reference's captured state, the candidate's own captured state, the
random-equal-norm direction) is persisted through `clozn.analysis.tensor_store` (reused, never modified)
and referenced by its sha256 in the returned document's `arms[i].write.vectors`, never inlined as a raw
float array -- the same discipline `mechanistic_diff` already established for `residual_points`.

ON NOT EXTENDING `clozn.intervention_manifest.v1`
------------------------------------------------------
`clozn/receipts/intervention_manifest.py` is left untouched by this slice, deliberately -- see this
repo's commit history for the reasoning: that manifest is a general open-ended arms list (steer /
steer_vec / attention_knockout) that is cross-hash-checked byte-for-byte against the `clozn-client` pip
package, and clozn-client has no concept of a content-addressed tensor write or a fixed five-arm control
structure. This module's artifact (`clozn.transplant.v1`) is a different SHAPE of thing: not an
open-ended arms list a caller assembles, but a closed, always-five-arm harness whose whole point is that
the caller CANNOT omit a control. Bolting that onto intervention_manifest as a v2 would either weaken the
"exactly these five, always" guarantee to just another optional arm type, or fork the manifest's
semantics for no shared benefit -- clozn-client would gain nothing from a schema it cannot produce.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from clozn import schemas
from clozn.analysis import pair_compatibility, tensor_store
from clozn.experiments.stats import replay_class_for_meta

SCHEMA_VERSION = "clozn.transplant.v1"

_ARM_ORDER = ("reference_transplant", "candidate_self_transplant", "random_equal_norm", "shuffled_layer")
_ALL_ARM_NAMES = _ARM_ORDER + ("no_write_replay",)

# hook -> (capture kwarg names on EngineClient.score, write kwarg name, response field names)
_HOOK_WRITE_FIELD = {"residual": "write", "ffn": "ffn_write"}
_HOOK_APPLIED_FIELD = {"residual": "write_applied", "ffn": "ffn_write_applied"}
_HOOK_CAPTURED_FIELD = {"residual": "captured", "ffn": "ffn_captured"}

_ZERO_TOL = 1e-12


# =========================================================================================== tiny math
# Deliberately stdlib-only (docs/SEAMS.md rule 1) -- mirrors clozn.analysis.mechanistic_diff's own
# discipline: no numpy, these are single residual/ffn rows (n_embd floats), not a hot loop.

def _norm(values: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in values))


def _random_equal_norm_vector(reference_row: Sequence[float], rng: "random.Random") -> list:
    """A random direction scaled to `reference_row`'s own L2 norm -- the random-equal-norm control's
    write value. Deterministic given `rng`'s seed, so the exact arm is reproducible from `random_seed`
    alone, not just described after the fact."""
    n = len(reference_row)
    ref_norm = _norm(reference_row)
    if ref_norm < _ZERO_TOL:
        return [0.0] * n          # a zero reference vector has no norm to match; the honest match is zero
    raw = [rng.gauss(0.0, 1.0) for _ in range(n)]
    raw_norm = _norm(raw)
    if raw_norm < _ZERO_TOL:      # astronomically unlikely for n_embd-scale n, but never divide by ~0
        raw = [1.0] + [0.0] * (n - 1)
        raw_norm = 1.0
    scale = ref_norm / raw_norm
    return [x * scale for x in raw]


def _find_in_topk(topk: Any, token_id) -> "dict | None":
    if not isinstance(topk, list):
        return None
    for item in topk:
        if isinstance(item, dict) and item.get("id") == token_id:
            return item
    return None


def _rank_of(topk: Any, token_id) -> "int | None":
    if not isinstance(topk, list):
        return None
    for index, item in enumerate(topk):
        if isinstance(item, dict) and item.get("id") == token_id:
            return index
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _flatten_position_major(vectors_by_position: Mapping[int, Sequence[float]], positions: Sequence[int]) -> list:
    out: list = []
    for position in positions:
        out.extend(float(x) for x in vectors_by_position[position])
    return out


# ================================================================================== engine call plumbing

def _call_score(engine, label: str, **kwargs) -> dict:
    """One /score call, never raising: an engine exception or a malformed response is reported as
    `{"ok": False, "error": ...}` (mirrors mechanistic_diff's `_score_with_capture`) so a failure on any
    one of this module's six calls can be attributed cleanly instead of crashing the whole run."""
    try:
        response = engine.score(**kwargs)
    except Exception as exc:      # noqa: BLE001 -- reported, never propagated
        return {"ok": False, "error": f"{label} failed: {type(exc).__name__}: {exc}"}
    if not isinstance(response, dict):
        return {"ok": False,
                "error": f"{label} failed: engine.score returned {type(response).__name__}, expected an object"}
    return {"ok": True, "response": response}


def _capture_kwargs(hook: str, layer: int, positions: Sequence[int]) -> dict:
    if hook == "residual":
        return {"capture_layers": [layer], "capture_positions": list(positions)}
    return {"ffn_capture_layers": [layer], "ffn_capture_positions": list(positions)}


def _write_kwargs(hook: str, *, layer: int, positions: Sequence[int], values: Sequence[float]) -> dict:
    spec = {"layer": layer, "positions": list(positions), "values": list(values)}
    return {_HOOK_WRITE_FIELD[hook]: [spec]}


def _read_captured_vectors(response: dict, hook: str, layer: int, positions: Sequence[int]) -> dict:
    """position -> [float, ...] or None (this position did not land -- see hook_vocabulary's
    known_gap_last_layer / architecture_coverage for why a request can be armed and still yield nothing)."""
    field = _HOOK_CAPTURED_FIELD[hook]
    captured = response.get(field)
    layer_rows = captured.get(str(layer)) if isinstance(captured, dict) else None
    out: dict = {}
    for position in positions:
        row = None
        if isinstance(layer_rows, dict):
            candidate_row = layer_rows.get(str(position))
            if isinstance(candidate_row, list):
                row = candidate_row
        out[position] = row
    return out


# =================================================================================== per-arm metrics

def _target_metrics(response: dict, *, n_prompt: int, n_cont: int, readout_position: int,
                    target_token_id: int) -> dict:
    """What ONE /score response says about `target_token_id` at `readout_position`: the candidate's own
    sum_logprob, the target's logprob/rank if it is observable, and the discrete top-1 signal
    `_derive_analysis`'s reference-specific rule is built on. Every metric that could not be honestly
    read off the wire is OMITTED and named with a reason -- never a zero or a guessed value."""
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
        found = _find_in_topk(topk_list, target_token_id)
        if found is not None and isinstance(found.get("logprob"), (int, float)):
            metrics["target_token_logprob"] = float(found["logprob"])
            if isinstance(found.get("piece"), str):
                metrics["target_token_piece"] = found["piece"]
        else:
            omitted.append({"metric": "target_token_logprob",
                            "reason": f"target_token_id={target_token_id} is neither the forced "
                                     f"continuation token here nor present in the returned top-k"})

    rank = _rank_of(topk_list, target_token_id)
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


def _build_arm(*, name: str, hook: str, response: dict, write_layer: "int | None",
              positions: Sequence[int], vectors_by_position: "dict | None", n_prompt: int, n_cont: int,
              readout_position: int, target_token_id: int, store_tensors: bool) -> dict:
    target = _target_metrics(response, n_prompt=n_prompt, n_cont=n_cont, readout_position=readout_position,
                             target_token_id=target_token_id)
    metrics = dict(target["metrics"])
    omitted = list(target["omitted"])

    if write_layer is None:       # no_write_replay: nothing was written, so there is nothing to confirm landed
        omitted.append({"metric": "write_applied",
                        "reason": f"{name} issues no write; this arm has nothing to confirm landed"})
        return {"name": name, "metrics": metrics, "omitted": omitted}

    applied_field = _HOOK_APPLIED_FIELD[hook]
    applied = response.get(applied_field)
    if isinstance(applied, bool):
        metrics["write_applied"] = applied
    else:
        omitted.append({"metric": "write_applied", "reason": f"engine response carried no {applied_field!r} flag"})

    write_block: dict = {"layer": write_layer, "positions": list(positions)}
    if store_tensors:
        vectors = []
        for position in positions:
            row = vectors_by_position[position]
            ref = tensor_store.store_tensor(
                row, shape=[len(row)],
                provenance={"role": f"{hook}_transplant_write", "arm": name, "layer": write_layer,
                           "position": position})
            vectors.append({"position": position, "tensor": ref})
        write_block["vectors"] = vectors
    return {"name": name, "write": write_block, "metrics": metrics, "omitted": omitted}


# =================================================================================== the structural rule

def _flipped_to_target(baseline_metrics: dict, arm_metrics: dict) -> "bool | None":
    """The discrete 'moved toward reference' signal: the candidate's own top-1 answer was NOT the target
    at baseline and IS the target under this arm. None (not True, not False) when either side's top-1 is
    unknown -- never guessed."""
    baseline_hit = baseline_metrics.get("top1_is_target")
    arm_hit = arm_metrics.get("top1_is_target")
    if baseline_hit is None or arm_hit is None:
        return None
    return (not baseline_hit) and arm_hit


def _derive_analysis(baseline_metrics: dict, arms_by_name: Mapping[str, dict]) -> dict:
    """THE structural gate this module exists to make unskippable -- see the module docstring's "THE
    RULE, MADE STRUCTURAL". `reference_specific` is computed here and ONLY here, from the reference and
    random-equal-norm arms' own measured movement; there is no path that sets it without both."""
    reasons: list = []
    analysis: dict = {}

    self_metrics = arms_by_name["candidate_self_transplant"]["metrics"]
    self_applied = self_metrics.get("write_applied")
    self_top1 = self_metrics.get("top1_token_id")
    baseline_top1 = baseline_metrics.get("top1_token_id")

    if self_applied is not True:
        instrument_sane = False
        reasons.append(
            "candidate_self_transplant's write_applied was not confirmed true -- the write path itself "
            "is not confirmed to have run, so no other write-bearing arm's result is interpretable.")
    elif self_top1 is None or baseline_top1 is None:
        instrument_sane = False
        reasons.append(
            "candidate_self_transplant sanity check could not be evaluated (top-1 token missing from "
            "baseline or self-transplant response) -- treated as NOT sane; no silent fallback.")
    elif self_top1 != baseline_top1:
        instrument_sane = False
        reasons.append(
            "candidate_self_transplant changed the top-1 token (from writing the candidate's own "
            "captured state back into itself) -- the write path itself is not a no-op here, so no "
            "other arm's result is interpretable until this is fixed.")
    else:
        instrument_sane = True
    analysis["instrument_sane"] = instrument_sane

    if not instrument_sane:
        analysis["reasons"] = reasons
        return analysis

    if baseline_top1 is not None and baseline_metrics.get("top1_is_target") is True:
        analysis["baseline_already_matches_target"] = True
        reasons.append(
            "the candidate's own baseline top-1 already equals target_token_id -- there is no "
            "disagreement for a transplant to correct, so 'moved toward reference' is not evaluated.")
        analysis["reasons"] = reasons
        return analysis

    reference_metrics = arms_by_name["reference_transplant"]["metrics"]
    random_metrics = arms_by_name["random_equal_norm"]["metrics"]
    reference_moved = _flipped_to_target(baseline_metrics, reference_metrics)
    random_moved = _flipped_to_target(baseline_metrics, random_metrics)

    if reference_moved is None:
        reasons.append("reference_transplant movement could not be evaluated (top-1/target-hit missing).")
    else:
        analysis["reference_moved_toward_reference"] = reference_moved
    if random_moved is None:
        reasons.append("random_equal_norm movement could not be evaluated (top-1/target-hit missing).")
    else:
        analysis["random_moved_toward_reference"] = random_moved

    if reference_moved is not None and random_moved is not None:
        reference_specific = bool(reference_moved and not random_moved)
        analysis["reference_specific"] = reference_specific
        if reference_moved and random_moved:
            reasons.append(
                "the random equal-norm control ALSO flipped the top-1 token to target_token_id -- the "
                "reference arm's flip is not reference-specific; this looks like perturbation "
                "sensitivity, not evidence the reference's state was uniquely correct here (see "
                "docs/research/DISTRIBUTED_FUNCTION.md's own prior transplant-study control finding).")
        elif not reference_moved:
            reasons.append("the reference transplant did not flip the top-1 token to target_token_id.")
        else:
            reasons.append(
                "the reference transplant flipped the top-1 token to target_token_id and the random "
                "equal-norm control did not -- reference-specific by this harness's rule.")

    analysis["reasons"] = reasons
    return analysis


# =================================================================================== preflight helpers

def _writable_range(hook: str, layer_count: int) -> tuple:
    """[min, max_exclusive) writable layers for `hook` on a model with `layer_count` layers -- mirrors
    GgmlAdapter::add_write_state's own gate for residual `write` (pair_compatibility.writable_layer_range
    computes the identical range; duplicated here as a plain tuple so this module does not need to build
    a fake pair_compatibility document just to reuse that one helper) and hook_vocabulary's stated
    `ffn_write` range (every layer has its own FFN block, so layer 0 IS valid there)."""
    if hook == "residual":
        return (1, layer_count)
    return (0, layer_count)


# =========================================================================================== public API

def run_site(*, pair_compat: Mapping[str, Any], reference_loader: Callable[[], Any],
            candidate_loader: Callable[[], Any], prompt_ids: Sequence[int],
            continuation_ids: Sequence[int], site: Mapping[str, Any], shuffled_layer: int,
            write_positions: Sequence[int], readout_position: int, target_token_id: int, topk: int = 5,
            seed: int = 0, store_tensors: bool = True, generated_at: "str | None" = None,
            validate: bool = True) -> dict:
    """Run the five-arm controlled transplant at ONE site. Capture the reference (one forward, then
    torn down completely), then run baseline + all five arms on the candidate (six forwards, candidate
    resident throughout, never concurrent with the reference) and build a `clozn.transplant.v1` document.

    `site` is `{"hook": "residual"|"ffn", "layer": int}`. `shuffled_layer` must be a DIFFERENT layer
    within the same writable range -- it is the shuffled-layer control's write target. `write_positions`
    is where the transplant writes; `readout_position` is where each arm's top-1/target-token metrics are
    read (commonly the same position, but kept independent since a write can propagate to a LATER
    position -- see hook_vocabulary's `propagation` note). `target_token_id` is the token whose recovery
    in the candidate is being tested -- typically the reference model's own top-1 answer at
    `readout_position`, per the prior transplant study's own method (scripts/tracer/transplant_localize.py).

    Returns `{"ok": True, "document": {...}}` on success, or `{"ok": False, "error": ...}` on a preflight
    refusal or a capture/arm failure on either engine. Never raises for an ordinary refusal or engine-side
    failure -- mirrors `clozn.analysis.mechanistic_diff.compare()`'s own contract exactly.
    """
    if not isinstance(pair_compat, dict):
        return {"ok": False, "error": "pair_compat must be a clozn.pair-compatibility.v1 document (dict)"}
    if not isinstance(site, Mapping):
        return {"ok": False, "error": "site must be an object {hook, layer}"}
    hook = site.get("hook")
    if hook not in _HOOK_WRITE_FIELD:
        return {"ok": False, "error": f"site.hook must be one of {sorted(_HOOK_WRITE_FIELD)}, got {hook!r}"}
    layer = site.get("layer")
    if not isinstance(layer, int) or isinstance(layer, bool):
        return {"ok": False, "error": "site.layer must be an integer"}

    if not pair_compatibility.may_residual_transplant(pair_compat):
        reason = (pair_compat.get("verdict", {}).get("operations", {})
                 .get("residual_transplant", {}).get("reason") or "residual transplant is not permitted")
        return {"ok": False, "error": f"transplant refused: {reason}"}

    layer_count = (pair_compat.get("layer_count") or {}).get("value_b")
    if not isinstance(layer_count, int) or isinstance(layer_count, bool):
        return {"ok": False, "error": "transplant refused: the candidate's layer_count is unknown"}
    lo, hi = _writable_range(hook, layer_count)
    if not (lo <= layer < hi):
        return {"ok": False,
                "error": f"site.layer {layer} is outside the writable range [{lo}, {hi}) for hook {hook!r}"}
    if not isinstance(shuffled_layer, int) or isinstance(shuffled_layer, bool) or not (lo <= shuffled_layer < hi):
        return {"ok": False,
                "error": f"shuffled_layer must be an integer in [{lo}, {hi}) for hook {hook!r}"}
    if shuffled_layer == layer:
        return {"ok": False,
                "error": "shuffled_layer must differ from site.layer (it is the SHUFFLED-layer control)"}

    positions = sorted({int(p) for p in write_positions})
    if not positions:
        return {"ok": False, "error": "write_positions must not be empty"}
    readout_position = int(readout_position)
    target_token_id = int(target_token_id)
    if not isinstance(topk, int) or isinstance(topk, bool) or topk < 1:
        return {"ok": False, "error": "transplant needs topk >= 1 to read each arm's top-1 token"}
    prompt_id_list = [int(x) for x in prompt_ids]
    continuation_id_list = [int(x) for x in continuation_ids]
    if not continuation_id_list:
        return {"ok": False, "error": "run_site() needs a non-empty continuation"}
    n_prompt = len(prompt_id_list)
    n_cont = len(continuation_id_list)

    # ---- reference: ONE forward, capture the site's vector at write_positions, then torn down. No
    # topk requested here -- target_token_id is interpreted purely under the CANDIDATE's own vocabulary
    # (every top1/target-hit metric this module computes reads a CANDIDATE response's tokens/topk; the
    # reference's own token distribution is never consulted), so tokenizer compatibility between the two
    # models is not a requirement for this primitive's mechanics the way it is for mechanistic_diff's
    # per-token comparison -- only hidden_size (the write's mechanical constraint) is gated below.
    with reference_loader() as reference_engine:
        ref_call = _call_score(reference_engine, "reference capture", prompt_ids=prompt_id_list,
                               continuation_ids=continuation_id_list, topk=0,
                               **_capture_kwargs(hook, layer, positions))
    if not ref_call["ok"]:
        return {"ok": False, "error": ref_call["error"]}
    ref_vectors = _read_captured_vectors(ref_call["response"], hook, layer, positions)
    missing = [p for p, row in ref_vectors.items() if row is None]
    if missing:
        return {"ok": False,
                "error": f"reference capture produced no row at layer={layer}, positions={missing}"}

    # ---- candidate: resident for baseline + all five arms, never concurrent with the reference.
    rng = random.Random(seed)
    with candidate_loader() as candidate_engine:
        baseline_call = _call_score(candidate_engine, "baseline capture", prompt_ids=prompt_id_list,
                                    continuation_ids=continuation_id_list, topk=topk,
                                    **_capture_kwargs(hook, layer, positions))
        if not baseline_call["ok"]:
            return {"ok": False, "error": baseline_call["error"]}
        baseline_response = baseline_call["response"]
        self_vectors = _read_captured_vectors(baseline_response, hook, layer, positions)
        self_missing = [p for p, row in self_vectors.items() if row is None]
        if self_missing:
            return {"ok": False,
                    "error": f"candidate self-capture produced no row at layer={layer}, positions={self_missing}"}

        random_vectors = {p: _random_equal_norm_vector(ref_vectors[p], rng) for p in positions}

        arm_write_plan = (
            ("reference_transplant", layer, ref_vectors),
            ("candidate_self_transplant", layer, self_vectors),
            ("random_equal_norm", layer, random_vectors),
            ("shuffled_layer", shuffled_layer, ref_vectors),
        )
        arm_results = []
        for name, write_layer, vectors_by_position in arm_write_plan:
            values = _flatten_position_major(vectors_by_position, positions)
            call = _call_score(candidate_engine, f"{name} arm", prompt_ids=prompt_id_list,
                               continuation_ids=continuation_id_list, topk=topk,
                               **_write_kwargs(hook, layer=write_layer, positions=positions, values=values))
            if not call["ok"]:
                return {"ok": False, "error": call["error"]}
            arm_results.append(_build_arm(
                name=name, hook=hook, response=call["response"], write_layer=write_layer,
                positions=positions, vectors_by_position=vectors_by_position, n_prompt=n_prompt,
                n_cont=n_cont, readout_position=readout_position, target_token_id=target_token_id,
                store_tensors=store_tensors))

        no_write_call = _call_score(candidate_engine, "no_write_replay arm", prompt_ids=prompt_id_list,
                                    continuation_ids=continuation_id_list, topk=topk)
        if not no_write_call["ok"]:
            return {"ok": False, "error": no_write_call["error"]}
        arm_results.append(_build_arm(
            name="no_write_replay", hook=hook, response=no_write_call["response"], write_layer=None,
            positions=positions, vectors_by_position=None, n_prompt=n_prompt, n_cont=n_cont,
            readout_position=readout_position, target_token_id=target_token_id, store_tensors=store_tensors))

    baseline_metrics = _target_metrics(baseline_response, n_prompt=n_prompt, n_cont=n_cont,
                                       readout_position=readout_position, target_token_id=target_token_id)
    arms_by_name = {arm["name"]: arm for arm in arm_results}
    analysis = _derive_analysis(baseline_metrics["metrics"], arms_by_name)

    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "reference_model": dict(pair_compat.get("model_a") or {}),
        "candidate_model": dict(pair_compat.get("model_b") or {}),
        "pair_compatibility": dict(pair_compat),
        "site": {"hook": hook, "layer": layer},
        "shuffled_layer": shuffled_layer,
        "write_positions": positions,
        "readout_position": readout_position,
        "target_token_id": target_token_id,
        "continuation": {"n_prompt": n_prompt, "n_cont": n_cont},
        "replay_class": replay_class_for_meta({"forced_rescore": True}),
        "random_seed": seed,
        "baseline": {"metrics": baseline_metrics["metrics"], "omitted": baseline_metrics["omitted"]},
        "arms": arm_results,
        "analysis": analysis,
    }
    target_piece = baseline_metrics["metrics"].get("target_token_piece")
    if target_piece:
        document["target_token_piece"] = target_piece
    if validate:
        schemas.validate(document)
    return {"ok": True, "document": document}
