"""Execute a preflighted exact execution fork and persist its terminal evidence."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math
import time
import uuid

from clozn import schemas
from clozn.replay.execution_fork import (
    parent_execution_fingerprint,
    plan_execution_fork,
)
from clozn.replay import execution_fork_results


class ExecutionForkExecutionError(RuntimeError):
    """The supplied object is not an executable FORK-00 exact plan."""


class ExecutionForkCancelled(RuntimeError):
    """Cooperative cancellation observed between the two synchronous worker calls."""


def _sha(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _worker_generation_steps(reply: Mapping) -> list[dict] | None:
    """Extract worker-supplied token evidence through the canonical trace normalizer.

    Exact execution must never fabricate a child timeline by tokenizing its returned text.  Workers
    that expose generation-time ``steps`` or folded autoregressive ``events`` can therefore hand us
    an honest trace; older workers simply leave the comparison trace unavailable.
    """
    from clozn.experiments.exact_execution import worker_generation_steps
    return worker_generation_steps(reply)


def _recorded_forced_step(parent_steps: list[dict], position: int,
                          execution_change: Mapping) -> dict | None:
    """Build the forced boundary step from recorded evidence, never the committed probability."""
    if position < 0 or position >= len(parent_steps):
        return None
    forced_piece = execution_change.get("token_piece")
    forced_id = execution_change.get("token_id")
    source = parent_steps[position]
    if not isinstance(source, Mapping):
        return None
    chosen = {"index": position, "piece": forced_piece if isinstance(forced_piece, str) else ""}
    if forced_id is not None:
        chosen["token_id"] = forced_id
    # The distribution before the intervention is unchanged.  Only copy distribution-level
    # alternatives/entropy; the committed token's own probability is deliberately not copied.
    if isinstance(source.get("alternatives"), list):
        chosen["alternatives"] = deepcopy(source["alternatives"])
    for key in ("topk_entropy", "entropy"):
        if source.get(key) is not None:
            chosen[key] = source[key]
    for alt in source.get("alternatives") or []:
        if not isinstance(alt, Mapping):
            continue
        alt_piece = alt.get("piece", alt.get("text"))
        alt_id = alt.get("token_id", alt.get("id"))
        same_piece = forced_piece is not None and alt_piece == forced_piece
        same_id = forced_id is not None and alt_id == forced_id
        if not (same_piece or same_id):
            continue
        if forced_id is None and isinstance(alt_id, int) and not isinstance(alt_id, bool):
            chosen["token_id"] = alt_id
        probability = alt.get("prob", alt.get("confidence", alt.get("conf")))
        if isinstance(probability, (int, float)) and not isinstance(probability, bool):
            if math.isfinite(float(probability)):
                chosen["prob"] = probability
        break
    return chosen


def _exact_child_trace(parent: Mapping, plan: Mapping, reply: Mapping) -> dict | None:
    """Assemble an exact child's trace from parent + worker generation evidence only.

    The worker may return the forced token as the first generated step, or may expose only the fresh
    continuation after the forced intervention.  Both forms are joined using the recorded forced
    piece.  Any mismatch is treated as unavailable comparison evidence rather than repaired by
    retokenizing text.
    """
    from clozn.runs.trace import normalize_trace, steps_to_trace

    worker_steps = _worker_generation_steps(reply)
    if not worker_steps:
        return None
    parent_trace = normalize_trace(parent.get("trace") or {})
    parent_steps = parent_trace.get("steps")
    if not isinstance(parent_steps, list):
        return None
    position = plan["request"]["position"]
    change = plan["request"].get("execution_change") or plan["request"].get("change") or {}
    reply_text = reply.get("text")
    if not isinstance(reply_text, str):
        return None

    worker_text = "".join(str(step.get("piece", "")) for step in worker_steps)
    parent_prefix_text = "".join(
        str(piece) for piece in (parent.get("trace", {}).get("tokens") or [])[:position]
    )

    # Sampling interventions do not have a forced boundary piece.  The worker's generation-time
    # steps are therefore the complete changed suffix (or, on newer workers, the full generated
    # child timeline).  Join that evidence with the immutable parent prefix without retokenizing
    # the returned text.  This keeps exact sampler children comparable while preserving the same
    # fail-closed behavior as the force-token path.
    if change.get("type") == "sampling":
        if worker_text == parent_prefix_text + reply_text:
            worker_prefix_pieces = [str(step.get("piece", "")) for step in worker_steps[:position]]
            parent_prefix_pieces = list((parent.get("trace", {}).get("tokens") or [])[:position])
            if worker_prefix_pieces == parent_prefix_pieces:
                worker_steps = worker_steps[position:]
                worker_text = "".join(str(step.get("piece", "")) for step in worker_steps)
        if worker_text != reply_text:
            return None
        combined = [deepcopy(step) for step in parent_steps[:position]] + [
            deepcopy(step) for step in worker_steps
        ]
        for index, step in enumerate(combined):
            step["index"] = index
        trace = steps_to_trace(combined)
        tokens = trace.get("tokens")
        expected = parent_prefix_text + reply_text
        if not isinstance(tokens, list) or "".join(tokens) != expected:
            return None
        return trace

    forced_piece = change.get("token_piece")
    if not isinstance(forced_piece, str) or not forced_piece:
        return None
    # A newer private worker may return the complete generated child timeline rather than only the
    # post-boundary suffix.  Strip the already-recorded immutable prefix by token evidence, never by
    # searching/fuzzy-matching response text.
    if worker_text == parent_prefix_text + reply_text:
        parent_prefix_pieces = list((parent.get("trace", {}).get("tokens") or [])[:position])
        worker_prefix_pieces = [str(step.get("piece", "")) for step in worker_steps[:position]]
        if worker_prefix_pieces == parent_prefix_pieces:
            worker_steps = worker_steps[position:]
            worker_text = "".join(str(step.get("piece", "")) for step in worker_steps)
    prefix_steps = [deepcopy(step) for step in parent_steps[:position]]
    suffix_steps = worker_steps
    forced_step = _recorded_forced_step(parent_steps, position, change)

    # Some execution workers report the whole child suffix, including the forced token.  If the
    # worker omitted that intervention step, prepend only its recorded boundary evidence.
    if worker_text == reply_text:
        if worker_steps and worker_steps[0].get("piece") == forced_piece and forced_step is not None:
            first = dict(worker_steps[0])
            for key in ("token_id", "prob"):
                if key not in first and key in forced_step:
                    first[key] = forced_step[key]
            suffix_steps = [first, *worker_steps[1:]]
    elif forced_step is not None and reply_text.startswith(forced_piece) \
            and worker_text == reply_text[len(forced_piece):]:
        suffix_steps = [forced_step, *worker_steps]
    else:
        return None

    # A full-suffix worker response is expected here.  Prefix it with the immutable parent's exact
    # steps and reindex through the shared normalizer; no token IDs or pieces are guessed.
    combined = prefix_steps + [deepcopy(step) for step in suffix_steps]
    for index, step in enumerate(combined):
        step["index"] = index
    trace = steps_to_trace(combined)
    tokens = trace.get("tokens")
    expected = "".join(str(piece) for piece in parent.get("trace", {}).get("tokens", [])[:position]) \
        + reply_text
    if not isinstance(tokens, list) or "".join(tokens) != expected:
        return None
    return trace


def _error(stage: str, code: str, message: str) -> dict:
    return {"stage": stage, "code": code, "message": message}


def _terminal(
    plan: Mapping,
    *,
    phase: str,
    status: str,
    reason_code: str,
    reason_message: str,
    execution_id: str,
    started: float,
    ended: float,
    control: dict | None = None,
    intervention: dict | None = None,
    error: dict | None = None,
    control_status: str = "failed",
    control_result: dict | None = None,
) -> dict:
    receipt = deepcopy(dict(plan))
    receipt["execution_id"] = execution_id
    receipt["phase"] = phase
    receipt["reasons"] = [{"code": reason_code, "message": reason_message}]
    receipt["execution"] = {
        "status": status,
        "started_ts": started,
        "ended_ts": ended,
    }
    if control is not None:
        receipt["execution"]["control"] = control
    if intervention is not None:
        receipt["execution"]["intervention"] = intervention
    if error is not None:
        receipt["execution"]["error"] = error
    receipt["unchanged_control"] = {"required": True, "status": control_status}
    if control_result is not None:
        receipt["unchanged_control"]["result"] = control_result
    receipt["child_lineage"].pop("child_run_id", None)
    receipt["child_lineage"]["receipt_status"] = (
        "cancelled" if phase == "cancelled"
        else "failed" if phase == "failed"
        else "not_created"
    )
    receipt["exactness"]["proof_status"] = "confirmed" if phase == "completed" else "failed"
    schemas.validate(receipt, "clozn.execution-fork.v1")
    return receipt


def _save_terminal(receipt: dict, save_result: Callable[[dict], dict]) -> dict:
    return save_result(receipt)


def _cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        return False


def _wire_change(change: Mapping) -> dict:
    """Strip planning-only token text before calling the closed worker intervention wire shape."""
    out = deepcopy(dict(change))
    if out.get("type") == "force_token":
        out.pop("token_piece", None)
    return out


def _worker_receipt(reply, plan: Mapping, expected_intervention: Mapping) -> dict:
    if not isinstance(reply, Mapping):
        raise ExecutionForkExecutionError("worker returned no execution-fork object")
    expected_generation = plan["checkpoint_reference"]["worker_generation_id"]
    expected_restore = (
        "live_kv_truncated"
        if plan["exactness"]["regime"] == "generated_token_live_kv"
        else "reprefill"
    )
    expected_source = plan["exactness"]["source"]
    exactness = reply.get("exactness")
    applied = reply.get("intervention_applied")
    tokens = reply.get("tokens")
    if reply.get("worker_generation_id") != expected_generation:
        raise ExecutionForkExecutionError(
            "worker generation changed after planning")
    if reply.get("restore_mode") != expected_restore:
        raise ExecutionForkExecutionError(
            f"worker restore_mode {reply.get('restore_mode')!r} did not match {expected_restore!r}")
    if not isinstance(exactness, Mapping):
        raise ExecutionForkExecutionError("worker omitted exactness receipt")
    if exactness.get("source") != expected_source or exactness.get("boundary_shape_true") is not True:
        raise ExecutionForkExecutionError(
            "worker did not confirm the planned exactness source and boundary shape")
    if reply.get("n_past_restored") != plan["exactness"]["truncate_to"]:
        raise ExecutionForkExecutionError("worker restored a different token boundary")
    intervention_type = expected_intervention["type"]
    if not isinstance(applied, Mapping) or applied.get("type") != intervention_type:
        raise ExecutionForkExecutionError("worker applied a different intervention type")
    for name, value in expected_intervention.items():
        if name in {"type", "steer_vec", "steer_layer"}:
            continue
        echoed_name = "cleared" if name == "clear" else name
        if echoed_name in applied and applied[echoed_name] != value:
            raise ExecutionForkExecutionError(
                f"worker applied a different intervention field {name}")
    if intervention_type in {"force_token", "residual_write"}:
        required_echoes = (
            ("token_id",) if intervention_type == "force_token" else ("layer", "position"))
        if any(applied.get(name) != expected_intervention.get(name) for name in required_echoes):
            raise ExecutionForkExecutionError(
                "worker intervention receipt did not match the requested intervention")
    if intervention_type == "sampling":
        # The request may override only one sampler field, but the child journal needs the fully
        # resolved checkpoint-plus-override regime. Production workers echo all five exact values.
        _child_decode({}, reply)
    if not isinstance(tokens, list) or any(
        not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0
        for token_id in tokens
    ):
        raise ExecutionForkExecutionError("worker returned invalid generated token ids")
    text = reply.get("text")
    if not isinstance(text, str):
        raise ExecutionForkExecutionError("worker returned invalid generated text")
    out = {
        "worker_generation_id": expected_generation,
        "restore_mode": expected_restore,
        "exactness_source": expected_source,
        "boundary_shape_true": True,
        "n_past_restored": plan["exactness"]["truncate_to"],
        "token_count": len(tokens),
        "token_ids_sha256": _sha(tokens),
        "text_sha256": _text_sha(text),
        "intervention_type": intervention_type,
        "intervention_sha256": _sha(expected_intervention),
    }
    for name in ("finish_reason", "sampler_source", "steer_source", "sampler_state_preserved",
                 "rng_state_preserved", "sampler_state_sha256"):
        value = reply.get(name)
        if isinstance(value, (str, bool)) and value:
            out[name] = value
    sampler = reply.get("sampler")
    if isinstance(sampler, Mapping):
        if sampler.get("sampler_state_preserved") is True or sampler.get("state_preserved") is True:
            out["sampler_state_preserved"] = True
    return out


def _failed_control_result(code: str, message: str, *, status: str = "failed") -> dict:
    return {
        "status": status,
        "exact_match": False,
        "note": f"{code}: {message}",
    }


def _boundary_stop_token_exempt(parent_run: Mapping, reply: Mapping,
                                expected_tokens: list, expected_text: str,
                                actual_tokens: list, actual_text: str) -> bool:
    """True iff the ONLY difference between the parent's recorded suffix and the worker's replayed
    suffix is that the parent's own trace carries one extra trailing entry -- the chat turn's
    stop/EOS-class token -- that the raw engine's generation loop structurally never returns as part
    of `tokens` (see generate_ar / finish_reason(): sampling that token TERMINATES the loop; it is a
    stop signal there, never committed output). clozn.runs.store records the FULL native-chat-io
    transcript, which DOES include it. Two different, independently correct conventions for "what
    counts as generated" collide exactly at the one position where they can ever disagree: the very
    last token of an EOS-terminated response. Both conventions predate this function; this reconciles
    them rather than favoring one silently.

    Narrow by construction -- ALL of the following must hold, not just finish_reason:
      - the parent's own recorded finish_reason is "stop" (it terminated via EOS/stop-sequence, not
        length) AND the worker's reconstructed reply ALSO reports finish_reason "stop" -- i.e., both
        the original run and the replay independently agree generation ended by hitting a stop
        condition, not that one ran out of budget while the other didn't;
      - the parent's suffix is EXACTLY one token longer than the worker's;
      - every token before that final one matches EXACTLY (a real earlier divergence still fails);
      - the decoded TEXT matches exactly (the exemption never overrides a text mismatch -- a
        stop-class token that decoded to visible text, on a model where that's true, would fail this
        and correctly report diverged).
    """
    if parent_run.get("finish_reason") != "stop" or reply.get("finish_reason") != "stop":
        return False
    if len(expected_tokens) != len(actual_tokens) + 1:
        return False
    if expected_tokens[:-1] != actual_tokens:
        return False
    return expected_text == actual_text


def prove_unchanged_control(parent_run: Mapping, plan: Mapping, engine) -> dict:
    """Run only the exact fork's mandatory unchanged control.

    This is the shared proof seam for FORK-01 execution and FORK-CKPT-01 checkpoint capture.  It
    deliberately does not persist a terminal execution receipt and does not create a child run:
    checkpoint preparation needs to establish that a reconstructed worker checkpoint can reproduce
    its immutable parent before that reference is exposed as eligible.

    Worker/protocol drift raises :class:`ExecutionForkExecutionError`. A well-formed worker result
    that differs from the parent returns ``status == "diverged"`` with hashes, never an exception
    containing private generated text.
    """
    try:
        schemas.validate(plan, "clozn.execution-fork.v1")
    except (schemas.ValidationError, schemas.SchemaError) as exc:
        raise ExecutionForkExecutionError(
            f"invalid execution-fork plan: {exc}") from None
    if plan.get("phase") != "planned" or plan.get("classification") != "exact_execution_fork":
        raise ExecutionForkExecutionError(
            "unchanged control requires a planned exact_execution_fork artifact")
    if plan.get("parent_run_id") != parent_run.get("id"):
        raise ExecutionForkExecutionError(
            "unchanged control plan parent does not match the supplied run")

    position = plan["request"]["position"]
    expected_tokens = list(parent_run["trace"]["token_ids"][position:])
    expected_text = "".join(parent_run["trace"]["tokens"][position:])
    reply = engine.execution_fork(
        checkpoint_id=plan["checkpoint_reference"]["checkpoint_id"],
        worker_generation_id=plan["checkpoint_reference"]["worker_generation_id"],
        truncate_to=plan["exactness"]["truncate_to"],
        max_tokens=len(expected_tokens),
        intervention={"type": "none"},
    )
    worker_receipt = _worker_receipt(reply, plan, {"type": "none"})
    actual_tokens = list(reply["tokens"])
    actual_text = reply["text"]
    parent_token_hash = _sha(expected_tokens)
    control_token_hash = _sha(actual_tokens)
    parent_text_hash = _text_sha(expected_text)
    control_text_hash = _text_sha(actual_text)
    matched = actual_tokens == expected_tokens and actual_text == expected_text
    boundary_exempt = False
    if not matched and _boundary_stop_token_exempt(
        parent_run, reply, expected_tokens, expected_text, actual_tokens, actual_text
    ):
        matched = True
        boundary_exempt = True
    if matched:
        note = (
            "parent suffix token ids and text matched exactly" if not boundary_exempt else
            "parent suffix matched exactly except the parent's recorded trailing chat-turn stop "
            "token, which the raw engine's own generation loop never returns as generated output "
            "(both the parent and the replay independently finished via finish_reason='stop' at the "
            "same content boundary; exempted, not silently dropped -- see prove_unchanged_control's "
            "_boundary_stop_token_exempt)"
        )
    else:
        note = "unchanged control differed in token ids or decoded text"
    return {
        "status": "matched" if matched else "diverged",
        "worker_receipt": worker_receipt,
        "result": {
            "status": "matched" if matched else "diverged",
            "exact_match": matched,
            "parent_suffix_sha256": _sha({
                "token_ids_sha256": parent_token_hash, "text_sha256": parent_text_hash}),
            "control_suffix_sha256": _sha({
                "token_ids_sha256": control_token_hash, "text_sha256": control_text_hash}),
            "note": note,
        },
    }


def _routing_runtime(selected_runtime: Mapping) -> dict:
    return {
        "key_sha256": selected_runtime["runtime_key_sha256"],
        "gguf_artifact_sha256": selected_runtime["model_sha256"],
        "context_size": selected_runtime["context_size"],
        "backend": selected_runtime["backend"],
        "adapter": deepcopy(selected_runtime["adapter"]),
        "template_fingerprint": selected_runtime["template_fingerprint"],
        "engine_build": selected_runtime["engine_build"],
        "white_box_flags": deepcopy(selected_runtime["white_box_flags"]),
    }


def _child_model_routing(
    parent: Mapping,
    receipt: Mapping,
    selected_runtime: Mapping,
    selected_worker: Mapping,
    current_worker: Mapping,
) -> dict | None:
    """Rebind a managed parent's routing receipt to this child control operation.

    A copied generation receipt is worse than no receipt after a restart. The exact-fork plan carries
    the selected immutable runtime and the executor receives the currently qualified worker binding.
    If the binding lacks the model-routing generation counter, omit the incompatible parent receipt
    instead of leaving a stale worker generation on the child.
    """
    parent_meta = parent.get("meta")
    parent_meta = parent_meta if isinstance(parent_meta, Mapping) else {}
    routing = parent_meta.get("model_routing")
    if not isinstance(routing, Mapping):
        return None
    try:
        schemas.validate(dict(routing), "clozn.model-routing.v1")
    except (schemas.ValidationError, schemas.SchemaError):
        return None
    generation = current_worker.get("worker_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        old_receipt = routing.get("result", {}).get("receipt", {})
        old_worker = old_receipt.get("worker_identity", {})
        if (
            isinstance(old_worker, Mapping)
            and old_worker.get("worker_id") == selected_worker["worker_id"]
        ):
            generation = old_worker.get("worker_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        return None

    updated = deepcopy(dict(routing))
    updated["protocol"] = {
        "surface": "native",
        "route": "/internal/exact-resume",
    }
    updated["request"] = {
        "request_id": receipt["execution_id"],
        "requested_model": parent.get("model"),
        "selection_source": "explicit",
        "load_policy": "wait",
    }
    routed = updated["result"]
    routed["status"] = "routed"
    routed["lifecycle_state"] = "ready"
    child_receipt = routed["receipt"]
    runtime_key = _routing_runtime(selected_runtime)
    child_receipt.update({
        "requested_model": parent.get("model"),
        "selection_source": "explicit",
        "resolved_model_id": parent.get("model"),
        "runtime_key": runtime_key,
        "worker_identity": {
            "worker_id": selected_worker["worker_id"],
            "worker_generation": generation,
            "runtime_key_sha256": selected_runtime["runtime_key_sha256"],
            "protocol_version": selected_worker["protocol_version"],
            "engine_build": selected_runtime["engine_build"],
            "backend": selected_runtime["backend"],
        },
        "adapter": deepcopy(selected_runtime["adapter"]),
        "load_event": {
            "event_id": None,
            "kind": "not_required",
            "outcome": "already_ready",
            "state_before": "ready",
            "state_after": "ready",
            "coalesced": False,
            "wait_ms": 0,
        },
    })
    schemas.validate(updated, "clozn.model-routing.v1")
    return updated


def _child_decode(parent_meta: Mapping, reply: Mapping) -> dict:
    applied = reply.get("intervention_applied")
    if not isinstance(applied, Mapping):
        raise ExecutionForkExecutionError(
            "worker omitted the resolved sampling intervention receipt")
    required = ("temperature", "top_p", "top_k", "rep_penalty", "seed")
    if any(name not in applied for name in required):
        raise ExecutionForkExecutionError(
            "worker omitted resolved sampling fields needed for child journaling")
    temperature = applied["temperature"]
    top_p = applied["top_p"]
    top_k = applied["top_k"]
    rep_penalty = applied["rep_penalty"]
    seed = applied["seed"]
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not math.isfinite(float(temperature))
        or temperature < 0
        or isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not math.isfinite(float(top_p))
        or not 0 <= top_p <= 1
        or isinstance(top_k, bool)
        or not isinstance(top_k, int)
        or top_k < 0
        or isinstance(rep_penalty, bool)
        or not isinstance(rep_penalty, (int, float))
        or not math.isfinite(float(rep_penalty))
        or rep_penalty <= 0
        or isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed < 0
    ):
        raise ExecutionForkExecutionError(
            "worker returned invalid resolved sampling fields")
    mode = "sample" if temperature > 0 else "greedy"
    return {
        "mode": mode,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": top_k,
        "repeat_penalty": float(rep_penalty),
        "seed": seed,
    }


def _child_journal(
    parent: Mapping,
    plan: Mapping,
    reply: Mapping,
    receipt: Mapping,
    *,
    current_worker: Mapping,
) -> tuple[dict, dict]:
    """Return truthful child meta and reproduction identity."""
    meta = deepcopy(parent.get("meta") or {})
    identity = deepcopy(parent.get("identity") or {})
    selected_runtime = plan["identity"]["selected_runtime"]
    selected_worker = plan["identity"]["selected_worker"]

    identity.update({
        "model_sha256": selected_runtime["model_sha256"],
        "template_fingerprint": selected_runtime["template_fingerprint"],
        "engine_build": selected_runtime["engine_build"],
        "white_box_flags": deepcopy(selected_runtime["white_box_flags"]),
    })
    meta.update({
        "n_ctx": selected_runtime["context_size"],
        "device": selected_runtime["backend"],
        "white_box_flags": deepcopy(selected_runtime["white_box_flags"]),
        "execution_fork_plan_id": plan["plan_id"],
        "execution_fork_execution_id": receipt["execution_id"],
        "execution_fork_worker_generation_id": selected_worker["worker_generation_id"],
    })
    routing = _child_model_routing(
        parent, receipt, selected_runtime, selected_worker, current_worker)
    if routing is None:
        meta.pop("model_routing", None)
    else:
        meta["model_routing"] = routing

    change = plan["request"]["execution_change"]
    if change["type"] == "sampling":
        decode = _child_decode(meta, reply)
        meta["decode"] = decode
        meta["sampler_mode"] = decode["mode"]
        meta["sampling"] = decode["mode"]
        meta["temperature"] = decode["temperature"]
        meta["repetition_penalty"] = decode["repeat_penalty"]
        meta["seed"] = decode["seed"]
    elif change["type"] == "steer":
        if change.get("clear") is True:
            meta.pop("execution_fork_steering", None)
        else:
            vector = list(change["steer_vec"])
            meta["execution_fork_steering"] = {
                "source": "recorded_raw_vector",
                "steer_vec": vector,
                "steer_layer": change.get("steer_layer", 0),
                "steer_coef": float(change.get("steer_coef", 1.0)),
                "intervention_sha256": _sha(change),
            }
    return meta, identity


def _record_child(parent: Mapping, plan: Mapping, reply: Mapping, receipt: dict,
                  *, started: float, current_worker: Mapping) -> dict | None:
    import clozn.runs.store as runlog

    position = plan["request"]["position"]
    pieces = parent["trace"]["tokens"]
    prefix = "".join(pieces[:position])
    response = prefix + reply["text"]
    meta, identity = _child_journal(
        parent, plan, reply, receipt, current_worker=current_worker)
    changes = {
        "execution_fork": {
            "plan_id": plan["plan_id"],
            "execution_id": receipt["execution_id"],
            "position": position,
            "change_sha256": plan["request"]["change_sha256"],
            "intervention": deepcopy(plan["request"]["execution_change"]),
        }
    }
    # Trace evidence is additive comparison support.  Its absence must not downgrade the exact
    # execution proof or prevent the child from being persisted.
    child_trace = _exact_child_trace(parent, plan, reply)
    run_id = runlog.record(
        source="fork",
        client="studio",
        model=str(parent.get("model") or ""),
        substrate=str(parent.get("substrate") or ""),
        messages=deepcopy(parent.get("messages") or []),
        assembled_messages=deepcopy(parent.get("assembled_messages")),
        response=response,
        trace=child_trace,
        started=started,
        parent_run_id=parent["id"],
        changes_applied=changes,
        finish_reason=reply.get("finish_reason"),
        meta=meta,
        final_prompt=parent.get("final_prompt"),
        identity=identity,
        session_key=parent.get("session_key"),
        client_key=parent.get("client_key"),
        client_key_source=parent.get("client_key_source"),
        project_key=parent.get("project_key"),
        output_contract=deepcopy(parent.get("output_contract") or {}),
        execution_fork_receipt=receipt,
    )
    return runlog.get_run(run_id) if run_id else None


def execute_exact_fork(
    parent_run: Mapping,
    plan: Mapping,
    engine,
    *,
    runtime_identity: Mapping,
    worker_identity: Mapping,
    reload_parent: Callable[[str], Mapping | None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
    record_child: Callable[..., dict | None] = _record_child,
    save_result: Callable[[dict], dict] = execution_fork_results.save,
    clock: Callable[[], float] = time.time,
) -> dict:
    """Run control then intervention for one still-current exact plan.

    Returns ``{"receipt": ..., "child": dict|None}``. Every accepted attempt produces one immutable
    terminal result receipt. Only a matched control permits the intervention call and only a successful
    intervention becomes a run.
    """
    try:
        schemas.validate(plan, "clozn.execution-fork.v1")
    except (schemas.ValidationError, schemas.SchemaError) as exc:
        raise ExecutionForkExecutionError(f"invalid execution-fork plan: {exc}") from None
    if plan.get("phase") != "planned" or plan.get("classification") != "exact_execution_fork":
        raise ExecutionForkExecutionError(
            "only a planned exact_execution_fork artifact can execute")
    if plan.get("parent_run_id") != parent_run.get("id"):
        raise ExecutionForkExecutionError("plan parent does not match the requested run")
    if not isinstance(plan.get("parent_fingerprint_sha256"), str):
        raise ExecutionForkExecutionError(
            "plan predates the execution-safe parent fingerprint; create a new plan")

    started = clock()
    execution_id = "fork_exec_" + uuid.uuid4().hex[:20]
    current_parent = reload_parent(parent_run["id"]) if reload_parent else parent_run
    stale_message = None
    if not isinstance(current_parent, Mapping):
        stale_message = "parent run disappeared after planning"
    elif parent_execution_fingerprint(current_parent) != plan["parent_fingerprint_sha256"]:
        stale_message = "parent execution evidence changed after planning"
    else:
        recomputed = plan_execution_fork(
            current_parent,
            {
                "position": plan["request"]["position"],
                "change": plan["request"]["change"],
            },
            checkpoint=plan.get("checkpoint_reference"),
            worker_identity=worker_identity,
            runtime_identity=runtime_identity,
        )
        if (
            recomputed.get("classification") != "exact_execution_fork"
            or recomputed.get("plan_id") != plan.get("plan_id")
            or recomputed.get("identity") != plan.get("identity")
            or recomputed.get("exactness") != plan.get("exactness")
        ):
            stale_message = (
                "worker, runtime, checkpoint, or exactness preconditions changed after planning")
    if stale_message is not None:
        ended = clock()
        receipt = _terminal(
            plan,
            phase="failed",
            status="control_failed",
            reason_code="stale_plan",
            reason_message=stale_message,
            execution_id=execution_id,
            started=started,
            ended=ended,
            error=_error("precondition", "stale_plan", stale_message),
            control_result=_failed_control_result("stale_plan", stale_message),
        )
        return {"receipt": _save_terminal(receipt, save_result), "child": None}
    parent_run = current_parent

    if _cancelled(cancel_check):
        ended = clock()
        message = "execution was cancelled before the unchanged control"
        receipt = _terminal(
            plan, phase="cancelled", status="cancelled",
            reason_code="execution_cancelled", reason_message=message,
            execution_id=execution_id,
            started=started, ended=ended,
            error=_error("control", "cancelled", message),
            control_status="cancelled",
            control_result=_failed_control_result("cancelled", message, status="cancelled"),
        )
        return {"receipt": _save_terminal(receipt, save_result), "child": None}

    try:
        control_evidence = prove_unchanged_control(parent_run, plan, engine)
        control_receipt = control_evidence["worker_receipt"]
        control_result = control_evidence["result"]
    except Exception as exc:
        ended = clock()
        if _cancelled(cancel_check) or isinstance(exc, ExecutionForkCancelled):
            phase, status, reason = "cancelled", "cancelled", "execution_cancelled"
            control_status = "cancelled"
        else:
            phase, status, reason = "failed", "control_failed", "control_failed"
            control_status = "failed"
        message = str(exc) or type(exc).__name__
        receipt = _terminal(
            plan, phase=phase, status=status,
            reason_code=reason, reason_message=message,
            execution_id=execution_id,
            started=started, ended=ended,
            error=_error("control", type(exc).__name__, message),
            control_status=control_status,
            control_result=_failed_control_result(
                type(exc).__name__, message, status=control_status),
        )
        return {"receipt": _save_terminal(receipt, save_result), "child": None}

    if control_evidence["status"] != "matched":
        ended = clock()
        receipt = _terminal(
            plan, phase="failed", status="control_diverged",
            reason_code="control_diverged",
            reason_message="unchanged control did not reproduce the parent suffix exactly",
            execution_id=execution_id,
            started=started, ended=ended,
            control=control_receipt,
            error=_error(
                "control", "control_diverged",
                "unchanged control token ids or decoded text differed"),
            control_status="diverged",
            control_result=control_result,
        )
        return {"receipt": _save_terminal(receipt, save_result), "child": None}

    if _cancelled(cancel_check):
        ended = clock()
        message = "execution was cancelled after the unchanged control"
        receipt = _terminal(
            plan, phase="cancelled", status="cancelled",
            reason_code="execution_cancelled", reason_message=message,
            execution_id=execution_id,
            started=started, ended=ended,
            control=control_receipt,
            error=_error("intervention", "cancelled", message),
            control_status="matched",
            control_result=control_result,
        )
        return {"receipt": _save_terminal(receipt, save_result), "child": None}

    position = plan["request"]["position"]
    expected_tokens = list(parent_run["trace"]["token_ids"][position:])
    common = {
        "checkpoint_id": plan["checkpoint_reference"]["checkpoint_id"],
        "worker_generation_id": plan["checkpoint_reference"]["worker_generation_id"],
        "truncate_to": plan["exactness"]["truncate_to"],
        "max_tokens": len(expected_tokens),
    }
    intervention = _wire_change(plan["request"]["execution_change"])
    try:
        child_reply = engine.execution_fork(**common, intervention=intervention)
        child_worker_receipt = _worker_receipt(
            child_reply, plan, intervention)
    except Exception as exc:
        ended = clock()
        if _cancelled(cancel_check) or isinstance(exc, ExecutionForkCancelled):
            phase, status, reason = "cancelled", "cancelled", "execution_cancelled"
        else:
            phase, status, reason = "failed", "intervention_failed", "intervention_failed"
        message = str(exc) or type(exc).__name__
        receipt = _terminal(
            plan, phase=phase, status=status,
            reason_code=reason, reason_message=message,
            execution_id=execution_id,
            started=started, ended=ended,
            control=control_receipt,
            error=_error("intervention", type(exc).__name__, message),
            control_status="matched",
            control_result=control_result,
        )
        return {"receipt": _save_terminal(receipt, save_result), "child": None}

    ended = clock()
    receipt = _terminal(
        plan, phase="completed", status="succeeded",
        reason_code="execution_succeeded",
        reason_message="unchanged control matched and the exact intervention completed",
        execution_id=execution_id,
        started=started, ended=ended,
        control=control_receipt,
        intervention=child_worker_receipt,
        control_status="matched",
        control_result=control_result,
    )
    child = record_child(
        parent_run,
        plan,
        child_reply,
        receipt,
        started=started,
        current_worker=worker_identity,
    )
    if child is None:
        failed = _terminal(
            plan, phase="failed", status="persistence_failed",
            reason_code="persistence_failed",
            reason_message="the intervention completed but its immutable child could not be stored",
            execution_id=execution_id,
            started=started, ended=clock(),
            control=control_receipt,
            intervention=child_worker_receipt,
            error=_error(
                "persistence", "child_store_failed",
                "run store returned no completed child"),
            control_status="matched",
            control_result=control_result,
        )
        return {"receipt": _save_terminal(failed, save_result), "child": None}

    stored_receipt = child.get("execution_fork")
    if not isinstance(stored_receipt, dict):
        raise ExecutionForkExecutionError(
            "child store omitted its immutable execution-fork receipt")
    return {
        "receipt": _save_terminal(stored_receipt, save_result),
        "child": child,
    }
