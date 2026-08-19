"""Neutral exact-resume worker protocol and unchanged-control proof.

The worker's ``execution_fork`` RPC is retained as a low-level resume
primitive.  These helpers validate its evidence for the new Generate adapter;
they do not persist a result or construct a Run.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
import math


class ExactExecutionError(RuntimeError):
    """The exact-resume worker did not satisfy its planned protocol."""


def _sha(value) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def worker_generation_steps(reply: Mapping) -> list[dict] | None:
    """Extract worker-supplied token evidence without retokenizing reply text."""
    from clozn.runs.trace import accumulate_ar_events, normalize_trace

    candidates = []
    if isinstance(reply.get("steps"), list):
        candidates.append(reply["steps"])
    raw_trace = reply.get("trace")
    if isinstance(raw_trace, (list, Mapping)):
        candidates.append(raw_trace)
    for key in ("events", "generation_events", "frames"):
        if isinstance(reply.get(key), list):
            candidates.append(accumulate_ar_events(reply[key]))
    for raw in candidates:
        normalized = normalize_trace(raw)
        steps = normalized.get("steps")
        if isinstance(steps, list) and steps:
            return [deepcopy(step) for step in steps if isinstance(step, Mapping)]
    return None


def _expected_worker(plan: Mapping) -> Mapping:
    identity = plan.get("identity")
    worker = identity.get("selected_worker") if isinstance(identity, Mapping) else None
    if not isinstance(worker, Mapping):
        checkpoint = plan.get("checkpoint_reference")
        generation = checkpoint.get("worker_generation_id") if isinstance(checkpoint, Mapping) else None
        if isinstance(generation, str) and generation:
            return {"worker_generation_id": generation}
        return {}
    return worker


def validate_worker_receipt(reply: Mapping, plan: Mapping,
                            expected_intervention: Mapping) -> dict:
    """Validate the exact-resume receipt and return semantic provenance."""
    if not isinstance(reply, Mapping):
        raise ExactExecutionError("worker returned no exact-resume object")
    checkpoint = plan.get("checkpoint_reference")
    exactness = plan.get("exactness")
    if not isinstance(checkpoint, Mapping) or not isinstance(exactness, Mapping):
        raise ExactExecutionError("exact-resume plan is missing checkpoint or exactness facts")
    expected_generation = checkpoint.get("worker_generation_id")
    selected_worker = _expected_worker(plan)
    if reply.get("worker_generation_id") != expected_generation:
        raise ExactExecutionError("worker generation changed after planning")
    expected_restore = "live_kv_truncated" if exactness.get("regime") == "generated_token_live_kv" else "reprefill"
    if reply.get("restore_mode") != expected_restore:
        raise ExactExecutionError(f"worker restore_mode {reply.get('restore_mode')!r} did not match {expected_restore!r}")
    receipt = reply.get("exactness")
    if not isinstance(receipt, Mapping):
        raise ExactExecutionError("worker omitted exactness receipt")
    if receipt.get("source") != exactness.get("source") or receipt.get("boundary_shape_true") is not True:
        raise ExactExecutionError("worker did not confirm the planned exactness source and boundary shape")
    if reply.get("n_past_restored") != exactness.get("truncate_to"):
        raise ExactExecutionError("worker restored a different token boundary")
    applied = reply.get("intervention_applied")
    expected_type = expected_intervention.get("type")
    if not isinstance(applied, Mapping) or applied.get("type") != expected_type:
        raise ExactExecutionError("worker applied a different intervention type")
    for name, value in expected_intervention.items():
        if name in {"type", "steer_vec", "steer_layer"}:
            continue
        echoed_name = "cleared" if name == "clear" else name
        if echoed_name in applied and applied[echoed_name] != value:
            raise ExactExecutionError(f"worker applied a different intervention field {name}")
    if expected_type == "force_token" and applied.get("token_id") != expected_intervention.get("token_id"):
        raise ExactExecutionError("worker intervention receipt did not match the requested token")
    tokens = reply.get("tokens")
    if not isinstance(tokens, list) or any(
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
            for token_id in tokens):
        raise ExactExecutionError("worker returned invalid generated token ids")
    text = reply.get("text")
    if not isinstance(text, str):
        raise ExactExecutionError("worker returned invalid generated text")
    out = {
        "worker_generation_id": expected_generation,
        "restore_mode": expected_restore,
        "exactness_source": exactness.get("source"),
        "boundary_shape_true": True,
        "n_past_restored": exactness.get("truncate_to"),
        "token_count": len(tokens),
        "token_ids_sha256": _sha(tokens),
        "text_sha256": _text_sha(text),
        "intervention_type": expected_type,
        "intervention_sha256": _sha(dict(expected_intervention)),
    }
    for name in ("finish_reason", "sampler_source", "steer_source", "sampler_state_preserved",
                 "rng_state_preserved", "sampler_state_sha256"):
        value = reply.get(name)
        if isinstance(value, (str, bool)) and value:
            out[name] = value
    sampler = reply.get("sampler")
    if isinstance(sampler, Mapping) and (
            sampler.get("sampler_state_preserved") is True
            or sampler.get("state_preserved") is True):
        out["sampler_state_preserved"] = True
    return out


SAMPLER_FIELDS = ("temperature", "top_p", "top_k", "rep_penalty", "seed")


def resolved_sampler_receipt(reply: Mapping) -> dict:
    """Return the fully resolved sampler the worker reports it actually applied.

    A sampler override request names only the fields it changes, so the request alone never says
    what regime produced the continuation.  The worker echoes all five resolved values in its
    ``intervention_applied`` receipt, and that echo -- not a child Run's journal -- is the evidence
    a sampled observation carries.  Raises rather than guessing when the echo is absent or invalid.
    """
    applied = reply.get("intervention_applied") if isinstance(reply, Mapping) else None
    if not isinstance(applied, Mapping):
        raise ExactExecutionError("worker omitted the resolved sampling receipt")
    if any(name not in applied for name in SAMPLER_FIELDS):
        raise ExactExecutionError("worker omitted resolved sampler fields")
    temperature, top_p, top_k = applied["temperature"], applied["top_p"], applied["top_k"]
    rep_penalty, seed = applied["rep_penalty"], applied["seed"]
    if (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature)) or temperature < 0
            or isinstance(top_p, bool) or not isinstance(top_p, (int, float))
            or not math.isfinite(float(top_p)) or not 0 <= top_p <= 1
            or isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 0
            or isinstance(rep_penalty, bool) or not isinstance(rep_penalty, (int, float))
            or not math.isfinite(float(rep_penalty)) or rep_penalty <= 0
            or isinstance(seed, bool) or not isinstance(seed, int) or seed < 0):
        raise ExactExecutionError("worker returned invalid resolved sampler fields")
    return {
        "temperature": float(temperature), "top_p": float(top_p), "top_k": top_k,
        "rep_penalty": float(rep_penalty), "seed": seed,
        "mode": "sample" if float(temperature) > 0 else "greedy",
    }


def _boundary_stop_token_exempt(parent_run: Mapping, reply: Mapping,
                                expected_tokens: list, expected_text: str,
                                actual_tokens: list, actual_text: str) -> bool:
    if parent_run.get("finish_reason") != "stop" or reply.get("finish_reason") != "stop":
        return False
    return (len(expected_tokens) == len(actual_tokens) + 1
            and expected_tokens[:-1] == actual_tokens
            and expected_text == actual_text)


def prove_unchanged_control(parent_run: Mapping, plan: Mapping, engine) -> dict:
    """Run the mandatory unchanged control and return inspectable proof only."""
    if not isinstance(parent_run, Mapping) or not isinstance(plan, Mapping):
        raise ExactExecutionError("unchanged control requires a recorded run and exact plan")
    if plan.get("classification") != "exact_execution_fork":
        raise ExactExecutionError("unchanged control requires an exact execution state")
    if plan.get("checkpoint_reference", {}).get("parent_run_id") != parent_run.get("id"):
        raise ExactExecutionError("unchanged control parent does not match the supplied run")
    position = plan.get("position")
    trace = parent_run.get("trace")
    if not isinstance(position, int) or not isinstance(trace, Mapping):
        raise ExactExecutionError("unchanged control token boundary is unavailable")
    expected_tokens = list(trace.get("token_ids", [])[position:])
    expected_text = "".join(trace.get("tokens", [])[position:])
    checkpoint = plan["checkpoint_reference"]
    exactness = plan["exactness"]
    reply = engine.execution_fork(
        checkpoint_id=checkpoint["checkpoint_id"],
        worker_generation_id=checkpoint["worker_generation_id"],
        truncate_to=exactness["truncate_to"],
        max_tokens=len(expected_tokens), intervention={"type": "none"},
    )
    receipt = validate_worker_receipt(reply, plan, {"type": "none"})
    actual_tokens = list(reply["tokens"])
    actual_text = reply["text"]
    matched = actual_tokens == expected_tokens and actual_text == expected_text
    boundary_exempt = False
    if not matched and _boundary_stop_token_exempt(
            parent_run, reply, expected_tokens, expected_text, actual_tokens, actual_text):
        matched = True
        boundary_exempt = True
    if not matched:
        note = "unchanged control differed in token ids or decoded text"
    elif boundary_exempt:
        note = "parent suffix matched except the recorded trailing stop token"
    else:
        note = "parent suffix token ids and text matched exactly"
    return {
        "status": "matched" if matched else "diverged",
        "worker_receipt": receipt,
        "result": {
            "status": "matched" if matched else "diverged",
            "exact_match": matched,
            "parent_suffix_sha256": _sha({"token_ids_sha256": _sha(expected_tokens),
                                            "text_sha256": _text_sha(expected_text)}),
            "control_suffix_sha256": _sha({"token_ids_sha256": _sha(actual_tokens),
                                             "text_sha256": _text_sha(actual_text)}),
            "note": note,
        },
    }


def wire_intervention(intervention: Mapping) -> dict:
    """Remove planning-only token text from the low-level worker wire request."""
    result = deepcopy(dict(intervention))
    if result.get("type") == "force_token":
        result.pop("token_piece", None)
    return result


__all__ = [
    "ExactExecutionError", "SAMPLER_FIELDS", "prove_unchanged_control", "resolved_sampler_receipt",
    "validate_worker_receipt", "wire_intervention", "worker_generation_steps",
]
