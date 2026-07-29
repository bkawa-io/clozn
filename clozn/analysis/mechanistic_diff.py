"""mechanistic_diff.py -- slice 3.2: the cross-model mechanistic DIVERGENCE MAP.

Given a reference model and a candidate model that clozn.analysis.pair_compatibility has already found
compatible for per-token comparison, run the SAME teacher-forced continuation through both and report
WHERE their internals differ -- residual-stream direction and magnitude at a configurable layer/position
grid, and how each model's own ranking of the forced continuation token (and of the candidate's own
top-1 alternative) differs between the two. Produces `clozn.mechanistic-diff.v1`
(clozn/schemas/defs/clozn.mechanistic-diff.v1.json).

THIS SLICE IS PURELY OBSERVATIONAL
-----------------------------------
Nothing here runs an intervention, nothing here compares against a control, and nothing here is
entitled to say a difference in one place EXPLAINS a difference anywhere else. This module and every
string it produces must stay free of causal vocabulary ("caused", "because", "responsible for",
"localized") -- that vocabulary belongs to a later slice that runs controlled interventions with
controls and can actually earn it (docs/SEAMS.md rule 4: evidence before narration). What this module
is entitled to say is only ever "at layer L, position P, the two models' residuals point cos=X apart"
or "the forced token's rank moved by N places" -- a measured difference, not an explanation of it.

WHERE THE NUMBERS COME FROM
-----------------------------
One POST /score call per model, using this slice's EngineClient.score() extension
(engine/client/clozn_engine.py): `capture_layers`/`capture_positions` request the residual-stream rows
at the grid, and `topk` (already existed) requests, for every teacher-forced continuation token, the
model's own top-k alternatives at that position with their log-softmax values. Both pieces of evidence
come out of the SAME forward pass -- there is no second call, no extra latency budget spent.

  * residual_points  (per LAYER x POSITION): cosine similarity and normalized L2 distance between the
    two models' captured residual rows. Genuinely varies by layer -- the residual stream is a different
    vector at every depth.
  * position_metrics (per POSITION only): logit-delta and rank-movement of (a) the actual forced
    continuation token and (b) the candidate model's own top-1 alternative at that position, read off
    each model's /score `tokens[i].logprob`/`tokens[i].topk`. This is layer-INVARIANT by construction --
    /score's topk is the model's final output distribution, not a per-layer readout (a per-layer
    logit-lens projection through a J-lens sidecar is a plausible future extension, not implemented
    here: it would add a heavy precondition -- a fitted sidecar per layer -- this slice does not need
    and Task 1 was not scoped to wire).
  * layer_change (per MODEL x POSITION, across consecutive grid layers): the same residual metrics,
    applied WITHIN one model across depth instead of across models -- how fast that model's own residual
    evolves through the grid.

Every metric that could not be honestly read off the wire (a layer the engine could not capture, a
position outside the scored continuation, a token absent from a returned top-k list) is OMITTED from
the point's `metrics` dict and named in that point's `omitted` list with a plain-English reason -- see
`_METRIC_NAMES`. A `0.0` or a missing key never means the same thing here; only the former is a claim.

SEQUENCING -- ONE MODEL RESIDENT AT A TIME
--------------------------------------------
`compare()` takes `reference_loader`/`candidate_loader`: zero-argument callables each returning a
context manager that, on `__enter__`, yields something with `.score(...)` (a real EngineClient or a
test double) and on `__exit__` tears it down. The reference is loaded, captured, and torn down BEFORE
the candidate is ever loaded -- this box has 16GB VRAM and the two GGUFs are not assumed to fit
together. Neither loader is required to keep anything resident once its `with` block exits.

PREFLIGHT
---------
`compare()` takes an ALREADY-COMPUTED `clozn.pair-compatibility.v1` document (see
clozn/analysis/pair_compatibility.py, which this module reuses and never modifies) and refuses to
proceed unless it permits per-token comparison (tokenizer-exact) AND its `hidden_size` dimension is
"same". The tokenizer gate is pair_compatibility's own (comparing token id N across two different
vocabularies is meaningless); the hidden_size gate is this module's OWN addition on top of that
contract -- pair_compatibility only gates hidden_size for a residual TRANSPLANT (writing one model's
vector into the other), but computing a cosine similarity between two residual vectors needs them to be
the same length for exactly the same mechanical reason, so this module checks it directly rather than
asking pair_compatibility to grow a third operation it was never asked to model.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from clozn import schemas
from clozn.analysis import pair_compatibility, tensor_store

SCHEMA_VERSION = "clozn.mechanistic-diff.v1"

_POSITION_METRIC_NAMES = (
    "reference_token_logit_delta", "reference_token_rank_reference", "reference_token_rank_candidate",
    "reference_token_rank_movement", "candidate_token_logit_delta", "candidate_token_rank_reference",
    "candidate_token_rank_candidate", "candidate_token_rank_movement",
)
_RESIDUAL_METRIC_NAMES = ("residual_cosine_similarity", "residual_l2_normalized")

_ZERO_TOL = 1e-12


# =========================================================================================== tiny math
# Deliberately stdlib-only (docs/SEAMS.md rule 1) -- these vectors are single residual rows (n_embd
# floats), and this is not a hot loop, so plain Python is fast enough and keeps this module import-safe
# with nothing but the standard library, matching every other clozn.analysis module's own discipline.

def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _norm(a: Sequence[float]) -> float:
    return math.sqrt(sum(x * x for x in a))


def _cosine_similarity(a: Sequence[float], b: Sequence[float]):
    """None (never 0.0) when either vector is all-zero -- a zero vector has no direction, so cosine
    similarity is undefined, not "maximally similar" or "maximally different"."""
    na, nb = _norm(a), _norm(b)
    if na < _ZERO_TOL or nb < _ZERO_TOL:
        return None
    cos = _dot(a, b) / (na * nb)
    return max(-1.0, min(1.0, cos))          # clip float noise back into the schema's [-1, 1] bound


def _l2_normalized(a: Sequence[float], b: Sequence[float]):
    """||a - b|| / ((||a|| + ||b||) / 2) -- the raw L2 distance scaled by the pair's average magnitude,
    so the number stays comparable across layers whose typical residual norm differs wildly (early
    layers are small, late layers are often huge). None when both vectors are (near-)zero."""
    na, nb = _norm(a), _norm(b)
    denom = (na + nb) / 2.0
    if denom < _ZERO_TOL:
        return None
    diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return diff / denom


def _rank_of(topk: Any, token_id):
    """0-based index of `token_id` in a /score `topk` list (already sorted by logprob descending), or
    None if it is not there -- meaning only "rank >= len(topk)" is known, not the exact rank."""
    if not isinstance(topk, list):
        return None
    for index, item in enumerate(topk):
        if isinstance(item, dict) and item.get("id") == token_id:
            return index
    return None


def _logprob_of(topk: Any, token_id):
    if not isinstance(topk, list):
        return None
    for item in topk:
        if isinstance(item, dict) and item.get("id") == token_id:
            return item.get("logprob")
    return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# =================================================================================== single-model capture

def _score_with_capture(engine, *, prompt_ids: list, continuation_ids: list, layers: list,
                        positions: list, topk: int) -> dict:
    """One POST /score call carrying both `topk` and `capture`. Never raises: any exception from the
    engine (a real EngineError, or a test double's own failure) is caught and reported as
    `{"ok": False, "error": ...}` so a capture failure on one side can be attributed cleanly rather than
    crashing the whole comparison."""
    try:
        response = engine.score(prompt_ids=prompt_ids, continuation_ids=continuation_ids, topk=int(topk),
                                 capture_layers=list(layers), capture_positions=list(positions))
    except Exception as exc:      # noqa: BLE001 -- reported, never propagated (see docstring)
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if not isinstance(response, dict):
        return {"ok": False, "error": f"engine.score returned {type(response).__name__}, expected an object"}
    return {"ok": True, "response": response}


# ======================================================================================= layer_capture

def _capture_note(layer: int, missing: set, layer_count) -> str:
    if layer in missing:
        return ("layer armed for capture but produced no rows (commonly the last layer -- llama.cpp's "
                "inp_out_ids optimization only materializes logit rows there, so a whole-sequence "
                "capture request for it is always empty)")
    if isinstance(layer_count, int) and not (0 < layer < layer_count):
        return f"layer {layer} is outside this model's capturable range [1, {layer_count})"
    return "layer was not armed for capture (outside the model's capturable range)"


def _layer_capture_entries(layers: list, ref_resp: dict, cand_resp: dict, pair_compat: dict) -> list:
    ref_captured = ref_resp.get("captured") or {}
    cand_captured = cand_resp.get("captured") or {}
    ref_missing = set(ref_resp.get("capture_missing") or [])
    cand_missing = set(cand_resp.get("capture_missing") or [])
    ref_layer_count = (pair_compat.get("layer_count") or {}).get("value_a")
    cand_layer_count = (pair_compat.get("layer_count") or {}).get("value_b")

    out = []
    for layer in layers:
        key = str(layer)
        ref_ok = key in ref_captured
        cand_ok = key in cand_captured
        entry = {"layer": layer, "reference_captured": ref_ok, "candidate_captured": cand_ok}
        notes = []
        if not ref_ok:
            notes.append("reference: " + _capture_note(layer, ref_missing, ref_layer_count))
        if not cand_ok:
            notes.append("candidate: " + _capture_note(layer, cand_missing, cand_layer_count))
        if notes:
            entry["note"] = "; ".join(notes)
        out.append(entry)
    return out


# ===================================================================================== position_metrics

def _empty_position_entry(position: int, reason: str) -> dict:
    return {"position": position, "metrics": {},
            "omitted": [{"metric": name, "reason": reason} for name in _POSITION_METRIC_NAMES]}


def _position_metrics_entry(position: int, n_prompt: int, n_cont: int, ref_resp: dict, cand_resp: dict,
                            *, topk: int) -> dict:
    index = position - n_prompt
    if not (0 <= index < n_cont):
        return _empty_position_entry(
            position,
            f"position {position} is outside the scored continuation range "
            f"[{n_prompt}, {n_prompt + n_cont}) -- no forced/candidate token exists there to compare")

    ref_tokens = ref_resp.get("tokens") or []
    cand_tokens = cand_resp.get("tokens") or []
    if not (0 <= index < len(ref_tokens)) or not (0 <= index < len(cand_tokens)):
        return _empty_position_entry(
            position, "the engine's own tokens[] response was shorter than n_cont implied -- "
                     "token-level data is unavailable at this position")

    ref_entry = ref_tokens[index] if isinstance(ref_tokens[index], dict) else {}
    cand_entry = cand_tokens[index] if isinstance(cand_tokens[index], dict) else {}
    ref_id = ref_entry.get("id")
    ref_logprob_ref = ref_entry.get("logprob")
    ref_logprob_cand = cand_entry.get("logprob")

    out: dict = {"position": position}
    if isinstance(ref_id, int):
        out["reference_token_id"] = ref_id
        if isinstance(ref_entry.get("piece"), str):
            out["reference_token_piece"] = ref_entry["piece"]

    metrics: dict = {}
    omitted: list = []

    if isinstance(ref_logprob_ref, (int, float)) and isinstance(ref_logprob_cand, (int, float)):
        metrics["reference_token_logit_delta"] = float(ref_logprob_cand) - float(ref_logprob_ref)
    else:
        omitted.append({"metric": "reference_token_logit_delta",
                        "reason": "logprob for the forced token was missing from one side's /score response"})

    ref_rank_ref = _rank_of(ref_entry.get("topk"), ref_id)
    ref_rank_cand = _rank_of(cand_entry.get("topk"), ref_id)
    if ref_rank_ref is not None:
        metrics["reference_token_rank_reference"] = ref_rank_ref
    else:
        omitted.append({"metric": "reference_token_rank_reference",
                        "reason": f"the forced token does not appear in the reference model's own "
                                 f"returned top-{topk}"})
    if ref_rank_cand is not None:
        metrics["reference_token_rank_candidate"] = ref_rank_cand
    else:
        omitted.append({"metric": "reference_token_rank_candidate",
                        "reason": f"the forced token does not appear in the candidate model's returned "
                                 f"top-{topk}"})
    if ref_rank_ref is not None and ref_rank_cand is not None:
        metrics["reference_token_rank_movement"] = ref_rank_cand - ref_rank_ref
    else:
        omitted.append({"metric": "reference_token_rank_movement",
                        "reason": "requires both reference_token_rank_reference and "
                                 "reference_token_rank_candidate"})

    cand_topk = cand_entry.get("topk") or []
    top1 = cand_topk[0] if cand_topk and isinstance(cand_topk[0], dict) else None
    cand_id = top1.get("id") if top1 else None
    if isinstance(cand_id, int):
        out["candidate_token_id"] = cand_id
        if isinstance(top1.get("piece"), str):
            out["candidate_token_piece"] = top1["piece"]
        cand_logprob_cand = top1.get("logprob")
        if cand_id == ref_id:
            cand_logprob_ref = ref_logprob_ref
            cand_rank_ref = ref_rank_ref
        else:
            cand_logprob_ref = _logprob_of(ref_entry.get("topk"), cand_id)
            cand_rank_ref = _rank_of(ref_entry.get("topk"), cand_id)
        metrics["candidate_token_rank_candidate"] = 0    # trivial: it IS the candidate's own top-1

        if isinstance(cand_logprob_cand, (int, float)) and isinstance(cand_logprob_ref, (int, float)):
            metrics["candidate_token_logit_delta"] = float(cand_logprob_cand) - float(cand_logprob_ref)
        else:
            omitted.append({"metric": "candidate_token_logit_delta",
                            "reason": f"the candidate's top token (id={cand_id}) does not appear in the "
                                     f"reference model's returned top-{topk} at this position, and is not "
                                     f"the reference's forced token either -- its value under the "
                                     f"reference model is not observable from a single /score call"})
        if cand_rank_ref is not None:
            metrics["candidate_token_rank_reference"] = cand_rank_ref
            metrics["candidate_token_rank_movement"] = 0 - cand_rank_ref
        else:
            omitted.append({"metric": "candidate_token_rank_reference",
                            "reason": f"the candidate's top token (id={cand_id}) does not appear in the "
                                     f"reference model's returned top-{topk} at this position"})
            omitted.append({"metric": "candidate_token_rank_movement",
                            "reason": "requires candidate_token_rank_reference"})
    else:
        reason = f"topk was not requested or returned empty (topk={topk}); the candidate's own top-1 " \
                 f"token could not be determined"
        omitted.extend({"metric": name, "reason": reason} for name in
                       ("candidate_token_logit_delta", "candidate_token_rank_reference",
                        "candidate_token_rank_candidate", "candidate_token_rank_movement"))

    out["metrics"] = metrics
    out["omitted"] = omitted
    return out


# ===================================================================================== residual_points

def _residual_row(resp: dict, layer: int, position: int):
    captured = resp.get("captured") or {}
    layer_rows = captured.get(str(layer))
    if not isinstance(layer_rows, dict):
        return None
    row = layer_rows.get(str(position))
    return row if isinstance(row, list) else None


def _residual_point(layer: int, position: int, ref_resp: dict, cand_resp: dict, *, store_tensors: bool):
    ref_row = _residual_row(ref_resp, layer, position)
    cand_row = _residual_row(cand_resp, layer, position)
    if ref_row is None and cand_row is None:
        return None      # nothing at all here -- already explained at the layer_capture level

    metrics: dict = {}
    omitted: list = []
    if ref_row is None:
        reason = "reference residual was not captured at this layer (see layer_capture)"
        omitted.extend({"metric": name, "reason": reason} for name in _RESIDUAL_METRIC_NAMES)
    elif cand_row is None:
        reason = "candidate residual was not captured at this layer (see layer_capture)"
        omitted.extend({"metric": name, "reason": reason} for name in _RESIDUAL_METRIC_NAMES)
    else:
        cos = _cosine_similarity(ref_row, cand_row)
        if cos is None:
            omitted.append({"metric": "residual_cosine_similarity",
                            "reason": "a captured residual vector was all-zero (direction undefined)"})
        else:
            metrics["residual_cosine_similarity"] = cos
        l2 = _l2_normalized(ref_row, cand_row)
        if l2 is None:
            omitted.append({"metric": "residual_l2_normalized",
                            "reason": "both captured residual vectors were (near-)zero "
                                     "(normalization undefined)"})
        else:
            metrics["residual_l2_normalized"] = l2

    point: dict = {"layer": layer, "position": position, "metrics": metrics, "omitted": omitted}
    if store_tensors:
        tensors: dict = {}
        if ref_row is not None:
            tensors["reference"] = tensor_store.store_tensor(
                ref_row, shape=[len(ref_row)],
                provenance={"role": "residual_stream", "model": "reference", "layer": layer,
                           "position": position})
        if cand_row is not None:
            tensors["candidate"] = tensor_store.store_tensor(
                cand_row, shape=[len(cand_row)],
                provenance={"role": "residual_stream", "model": "candidate", "layer": layer,
                           "position": position})
        if tensors:
            point["tensors"] = tensors
    return point


# ======================================================================================== layer_change

def _layer_change_entries(layers: list, positions: list, ref_resp: dict, cand_resp: dict) -> list:
    out = []
    for model_name, resp in (("reference", ref_resp), ("candidate", cand_resp)):
        for position in positions:
            available = [L for L in layers if _residual_row(resp, L, position) is not None]
            for a, b in zip(available, available[1:]):
                row_a = _residual_row(resp, a, position)
                row_b = _residual_row(resp, b, position)
                metrics: dict = {}
                omitted: list = []
                cos = _cosine_similarity(row_a, row_b)
                if cos is None:
                    omitted.append({"metric": "residual_cosine_similarity",
                                    "reason": "a captured residual vector was all-zero "
                                             "(direction undefined)"})
                else:
                    metrics["residual_cosine_similarity"] = cos
                l2 = _l2_normalized(row_a, row_b)
                if l2 is None:
                    omitted.append({"metric": "residual_l2_normalized",
                                    "reason": "both captured residual vectors were (near-)zero "
                                             "(normalization undefined)"})
                else:
                    metrics["residual_l2_normalized"] = l2
                out.append({"model": model_name, "position": position, "from_layer": a, "to_layer": b,
                           "metrics": metrics, "omitted": omitted})
    return out


# =========================================================================================== the document

def _build_document(*, pair_compat: Mapping[str, Any], prompt_ids: list, continuation_ids: list,
                    layers: list, positions: list, topk: int, ref_resp: dict, cand_resp: dict,
                    store_tensors: bool, generated_at) -> dict:
    n_prompt = len(prompt_ids)
    n_cont = len(continuation_ids)

    residual_points = []
    for layer in layers:
        for position in positions:
            point = _residual_point(layer, position, ref_resp, cand_resp, store_tensors=store_tensors)
            if point is not None:
                residual_points.append(point)

    doc: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "reference_model": dict(pair_compat.get("model_a") or {}),
        "candidate_model": dict(pair_compat.get("model_b") or {}),
        "pair_compatibility": dict(pair_compat),
        "continuation": {"n_prompt": n_prompt, "n_cont": n_cont},
        "layers_requested": list(layers),
        "positions_requested": list(positions),
        "layer_capture": _layer_capture_entries(layers, ref_resp, cand_resp, pair_compat),
        "position_metrics": [_position_metrics_entry(p, n_prompt, n_cont, ref_resp, cand_resp, topk=topk)
                             for p in positions],
        "residual_points": residual_points,
    }
    layer_change = _layer_change_entries(layers, positions, ref_resp, cand_resp)
    if layer_change:
        doc["layer_change"] = layer_change
    return doc


# =========================================================================================== public API

def compare(*, pair_compat: Mapping[str, Any], reference_loader: Callable[[], Any],
           candidate_loader: Callable[[], Any], prompt_ids: Sequence[int],
           continuation_ids: Sequence[int], layers: Sequence[int], positions: Sequence[int],
           topk: int = 10, store_tensors: bool = True, generated_at: "str | None" = None,
           validate: bool = True) -> dict:
    """Capture the reference, then the candidate (never both resident at once -- see module docstring),
    and build a `clozn.mechanistic-diff.v1` document. Pure orchestration otherwise: `reference_loader`/
    `candidate_loader` own how a model is actually loaded/unloaded (a real engine spawn, or a test
    double), and this function never touches the filesystem or a subprocess directly.

    Returns `{"ok": True, "document": {...}}` on success, or `{"ok": False, "error": ...}` -- on a
    preflight refusal (mirrors clozn.analysis.run_diff.compare_runs()'s own refusal shape, deliberately
    NOT clozn.mechanistic-diff.v1-shaped, since it carries no captured data at all) or a capture failure
    on either side. Never raises for an ordinary refusal or engine-side failure; a malformed `pair_compat`
    (not even a dict) is likewise reported, not raised, per docs/SEAMS.md rule 3 (no silent fallback --
    but also no crash on a caller's honest mistake).
    """
    if not isinstance(pair_compat, dict):
        return {"ok": False, "error": "pair_compat must be a clozn.pair-compatibility.v1 document (dict)"}
    if not pair_compatibility.may_per_token_compare(pair_compat):
        reason = (pair_compat.get("verdict", {}).get("operations", {})
                 .get("per_token_comparison", {}).get("reason") or "per-token comparison is not permitted")
        return {"ok": False, "error": f"mechanistic diff refused: {reason}"}
    hidden_state = (pair_compat.get("hidden_size") or {}).get("state")
    if hidden_state != "same":
        return {"ok": False, "error": (
            f"mechanistic diff refused: hidden_size is {hidden_state!r}, not 'same' -- residual vectors "
            f"of different width cannot be compared directly (the same mechanical fact that gates "
            f"pair_compatibility's residual_transplant operation, checked here directly since this "
            f"module's own operation -- observational residual comparison -- is a third case "
            f"pair_compatibility does not model)")}

    layer_list = sorted({int(x) for x in layers})
    position_list = sorted({int(x) for x in positions})
    if not layer_list or not position_list:
        return {"ok": False, "error": "compare() needs at least one layer and one position"}
    prompt_id_list = [int(x) for x in prompt_ids]
    continuation_id_list = [int(x) for x in continuation_ids]
    if not continuation_id_list:
        return {"ok": False, "error": "compare() needs a non-empty continuation"}

    with reference_loader() as reference_engine:
        reference_capture = _score_with_capture(
            reference_engine, prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
            layers=layer_list, positions=position_list, topk=topk)
    if not reference_capture["ok"]:
        return {"ok": False, "error": f"reference capture failed: {reference_capture['error']}"}

    with candidate_loader() as candidate_engine:
        candidate_capture = _score_with_capture(
            candidate_engine, prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
            layers=layer_list, positions=position_list, topk=topk)
    if not candidate_capture["ok"]:
        return {"ok": False, "error": f"candidate capture failed: {candidate_capture['error']}"}

    document = _build_document(
        pair_compat=pair_compat, prompt_ids=prompt_id_list, continuation_ids=continuation_id_list,
        layers=layer_list, positions=position_list, topk=topk,
        ref_resp=reference_capture["response"], cand_resp=candidate_capture["response"],
        store_tensors=store_tensors, generated_at=generated_at)
    if validate:
        schemas.validate(document)
    return {"ok": True, "document": document}
