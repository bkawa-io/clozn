"""Fail-closed recorded-parent capture for exact execution forks.

The private worker's checkpoint store is bounded process memory.  This module turns one eligible,
immutable run into that worker state and returns a versioned public receipt.  The default path creates
an ephemeral checkpoint; an explicit ``checkpoint_envelope`` may instead hydrate a previously pinned
export into the selected worker before running the same exact unchanged-control proof.  The public
receipt remains ephemeral after hydration -- the durable bytes stay in the pin store and never enter
the receipt.

Prompt IDs are obtained from the matching worker's existing ``/score`` contract.  Prompt text and
the run's recorded continuation IDs are sent as separate fields, so no prompt/continuation BPE
boundary is retokenized.  The returned prompt count must match the count recorded on the original
run.  Merely retokenizing ``final_prompt`` without these identity/count checks is not considered
exact.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import hashlib
import json
import math
import time
import uuid

from clozn import schemas
from clozn.replay.controlled import recorded_sampling_config
from clozn.replay.execution_fork import (
    _runtime_projection,
    _worker_projection,
    parent_execution_fingerprint,
    parent_runtime_projection,
    plan_execution_fork,
)
from clozn.replay.execution_fork_execute import prove_unchanged_control


SCHEMA_VERSION = "clozn.checkpoint-reference.v1"
_EXPIRES_WHEN = ["worker_restart", "fifo_eviction", "gateway_shutdown"]


class CheckpointCaptureError(RuntimeError):
    """The caller supplied an object that cannot identify a parent/worker at all."""


def _sha(value) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _lifecycle(observed_state: str, *, size_bytes: int | None = None) -> dict:
    out = {
        "storage": "worker_memory",
        "durability": "ephemeral",
        "pinned": False,
        "eviction_policy": "bounded_fifo",
        "validity_scope": "worker_process_generation",
        "observed_state": observed_state,
        "expires_when": list(_EXPIRES_WHEN),
    }
    if isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0:
        out["size_bytes"] = size_bytes
    return out


def _finish(artifact: dict, *, status: str, code: str, message: str,
            proof: dict | None = None) -> dict:
    artifact["status"] = status
    artifact["reasons"] = [_reason(code, message)]
    artifact["proof"] = proof or {"status": "not_run"}
    schemas.validate(artifact, SCHEMA_VERSION)
    return artifact


def _trace_history(parent: Mapping) -> tuple[list[str], list[int]] | None:
    trace = parent.get("trace")
    pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
    token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    if not (
        isinstance(pieces, list)
        and pieces
        and all(isinstance(piece, str) for piece in pieces)
        and isinstance(token_ids, list)
        and len(token_ids) == len(pieces)
        and all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
                for token_id in token_ids)
    ):
        return None
    return list(pieces), list(token_ids)


def _sampler(parent: Mapping, output_count: int) -> tuple[dict, dict | None] | None:
    config = recorded_sampling_config(dict(parent))
    if config is None:
        return None
    if config is False:
        subject = {"mode": "greedy"}
        return {
            "mode": "greedy",
            "provenance": "recorded_run_decode",
            "rng_draws": 0,
            "config_sha256": _sha(subject),
        }, None

    temperature = config.get("temperature")
    top_k = config.get("top_k")
    top_p = config.get("top_p")
    rep_penalty = config.get("repeat_penalty")
    seed = config.get("seed")
    if not (
        _finite(temperature) and float(temperature) > 0
        and isinstance(top_k, int) and not isinstance(top_k, bool) and top_k >= 0
        and _finite(top_p) and 0 <= float(top_p) <= 1
        and _finite(rep_penalty) and float(rep_penalty) > 0
        and isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0
    ):
        return None
    wire = {
        "seed": seed,
        "rng_draws": output_count,
        "temperature": float(temperature),
        "top_k": top_k,
        "top_p": float(top_p),
        "rep_penalty": float(rep_penalty),
    }
    artifact = {
        "mode": "sample",
        "provenance": "recorded_run_decode",
        **wire,
        "config_sha256": _sha({"mode": "sample", **wire}),
    }
    return artifact, wire


def _steering(parent: Mapping) -> tuple[dict, dict] | None:
    behavior = parent.get("behavior")
    behavior = behavior if isinstance(behavior, Mapping) else {}
    active_dials = behavior.get("active_dials", {})
    if not isinstance(active_dials, Mapping):
        return None
    active_dials = dict(active_dials)
    meta = parent.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    raw = meta.get("execution_fork_steering")

    if not active_dials and raw is None:
        return {
            "mode": "none",
            "provenance": "recorded_none",
        }, {}
    if not isinstance(raw, Mapping) or raw.get("source") != "recorded_raw_vector":
        return None

    vector = raw.get("steer_vec")
    layer = raw.get("steer_layer")
    coef = raw.get("steer_coef")
    dials_sha = raw.get("active_dials_sha256")
    expected_dials_sha = _sha(active_dials)
    if not (
        isinstance(vector, list)
        and vector
        and all(_finite(value) for value in vector)
        and isinstance(layer, int)
        and not isinstance(layer, bool)
        and layer >= 0
        and _finite(coef)
        and isinstance(dials_sha, str)
        and dials_sha == expected_dials_sha
    ):
        return None
    artifact = {
        "mode": "raw_vector",
        "provenance": "recorded_raw_vector",
        "vector_sha256": _sha(vector),
        "vector_elements": len(vector),
        "steer_layer": layer,
        "steer_coef": float(coef),
        "active_dials_sha256": expected_dials_sha,
    }
    return artifact, {
        "steer_vec": list(vector),
        "steer_layer": layer,
        "steer_coef": float(coef),
    }


def _score_prompt_ids(engine, final_prompt: str, output_ids: list[int],
                      expected_prompt_tokens: int) -> list[int]:
    response = engine.score(
        prompt=final_prompt,
        continuation_ids=output_ids,
        topk=0,
    )
    if not isinstance(response, Mapping) or response.get("boundary_approximate") is True:
        raise CheckpointCaptureError(
            "worker score did not preserve the explicit prompt/continuation boundary")
    prompt_ids = response.get("prompt_ids")
    if not (
        isinstance(prompt_ids, list)
        and prompt_ids
        and all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
                for token_id in prompt_ids)
    ):
        raise CheckpointCaptureError(
            "worker score did not return exact prompt token IDs")
    if response.get("n_prompt") != len(prompt_ids):
        raise CheckpointCaptureError(
            "worker score prompt count did not match its prompt token IDs")
    if len(prompt_ids) != expected_prompt_tokens:
        raise CheckpointCaptureError(
            "matching worker tokenization did not match the parent prompt-token count")
    if response.get("n_cont") != len(output_ids):
        raise CheckpointCaptureError(
            "worker score continuation count did not match the recorded output")
    scored_tokens = response.get("tokens")
    if not (
        isinstance(scored_tokens, list)
        and [item.get("id") for item in scored_tokens if isinstance(item, Mapping)] == output_ids
        and len(scored_tokens) == len(output_ids)
    ):
        raise CheckpointCaptureError(
            "worker score did not echo the exact recorded continuation token IDs")
    return list(prompt_ids)


def _checkpoint_reference(parent_id: str, response: Mapping,
                          expected_generation: str, prompt_tokens: int,
                          total_tokens: int, *, allow_missing_n_tokens: bool = False) -> dict:
    checkpoint_id = response.get("checkpoint_id")
    generation = response.get("worker_generation_id")
    n_past = response.get("n_past")
    n_tokens = response.get("n_tokens")
    size_bytes = response.get("size_bytes")
    n_tokens_ok = n_tokens == total_tokens or (allow_missing_n_tokens and n_tokens is None)
    if not (
        isinstance(checkpoint_id, str) and checkpoint_id
        and generation == expected_generation
        and n_past == total_tokens
        and n_tokens_ok
        and isinstance(size_bytes, int) and not isinstance(size_bytes, bool) and size_bytes >= 0
    ):
        raise CheckpointCaptureError(
            "worker returned an incomplete or inconsistent checkpoint receipt")
    return {
        "checkpoint_id": checkpoint_id,
        "worker_generation_id": generation,
        "state": "available",
        "parent_run_id": parent_id,
        "prompt_tokens": prompt_tokens,
        "n_past": n_past,
        "size_bytes": size_bytes,
    }


def capture_parent_checkpoint(
    parent_run: Mapping,
    engine,
    *,
    runtime_identity: Mapping,
    worker_identity: Mapping,
    checkpoint_envelope: Mapping | None = None,
    material_out: dict | None = None,
    clock: Callable[[], float] = time.time,
) -> dict:
    """Capture and prove one ephemeral checkpoint, returning a v1 lifecycle artifact.

    Expected eligibility failures are returned as ``status == "unavailable"``. Worker calls that
    fail, return inconsistent evidence, or produce a divergent unchanged control return
    ``status == "failed"``. Only ``status == "available"`` may be handed to the exact fork planner.
    ``checkpoint_envelope`` is an internal, already-resolved durable pin. It is deliberately not
    accepted from the public request body: the gateway resolves the run-scoped pin and passes the
    envelope here only after the pin store has re-verified its blob and sidecar digests.
    """
    if not isinstance(parent_run, Mapping):
        raise CheckpointCaptureError("parent_run must be an object")
    if material_out is not None and not isinstance(material_out, dict):
        raise CheckpointCaptureError("material_out must be a dict when supplied")
    parent_id = parent_run.get("id")
    if not isinstance(parent_id, str) or not parent_id:
        raise CheckpointCaptureError("parent_run.id must be a non-empty string")
    worker = _worker_projection(worker_identity)
    if worker is None:
        raise CheckpointCaptureError(
            "worker identity needs id, process-generation id, and protocol version")

    meta = parent_run.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    parent_runtime = parent_runtime_projection(parent_run)
    selected_runtime = _runtime_projection(runtime_identity)
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_reference_id": "checkpoint_ref_" + uuid.uuid4().hex[:20],
        "parent_run_id": parent_id,
        "parent_fingerprint_sha256": parent_execution_fingerprint(parent_run),
        "captured_ts": float(clock()),
        "status": "unavailable",
        "lifecycle": _lifecycle("not_created"),
        "identity": {
            "parent_runtime_key_sha256": (
                parent_runtime.get("runtime_key_sha256") if parent_runtime else None),
            "selected_runtime_key_sha256": (
                selected_runtime.get("runtime_key_sha256") if selected_runtime else None),
            "worker_id": worker["worker_id"],
            "worker_generation_id": worker["worker_generation_id"],
            "protocol_version": worker["protocol_version"],
        },
        "reasons": [],
    }

    history = _trace_history(parent_run)
    if history is None:
        return _finish(
            artifact, status="unavailable",
            code="missing_output_token_history",
            message="the parent needs complete, aligned response token pieces and token IDs")
    _pieces, output_ids = history

    final_prompt = parent_run.get("final_prompt")
    if not isinstance(final_prompt, str) or not final_prompt:
        return _finish(
            artifact, status="unavailable",
            code="missing_final_prompt",
            message="the parent has no exact rendered prompt text for identity-qualified tokenization")
    prompt_tokens = meta.get("prompt_tokens")
    if not (
        isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and prompt_tokens > 0
    ):
        return _finish(
            artifact, status="unavailable",
            code="missing_prompt_boundary",
            message="the parent has no positive recorded prompt-token boundary")

    if (
        parent_run.get("substrate") != "engine"
        or not isinstance(meta.get("stream"), bool)
        or parent_run.get("parent_run_id") is not None
        or bool(parent_run.get("changes_applied"))
        or bool(parent_run.get("reasoning"))
        or bool(parent_run.get("output_contract"))
        or "clozn_guard_receipt" in meta
    ):
        return _finish(
            artifact, status="unavailable",
            code="unsupported_execution_shape",
            message=(
                "v1 capture requires an organic engine run with recorded stream shape and no "
                "structured grammar, hidden reasoning, generation guard, or prior intervention"))

    sampler_pair = _sampler(parent_run, len(output_ids))
    if sampler_pair is None:
        return _finish(
            artifact, status="unavailable",
            code="sampler_provenance_missing",
            message="the parent sampler mode, parameters, and fixed seed are not exactly recoverable")
    sampler_artifact, sampler_wire = sampler_pair

    steering_pair = _steering(parent_run)
    if steering_pair is None:
        return _finish(
            artifact, status="unavailable",
            code="steering_provenance_missing",
            message=(
                "steered parents require their exact recorded raw vector, layer, coefficient, "
                "and active-dial digest; dial names alone are insufficient"))
    steering_artifact, steering_wire = steering_pair

    placeholder = {
        "checkpoint_id": "checkpoint-preflight",
        "worker_generation_id": worker["worker_generation_id"],
        "state": "available",
        "parent_run_id": parent_id,
        "prompt_tokens": prompt_tokens,
        "n_past": prompt_tokens + len(output_ids),
        "size_bytes": 0,
    }
    preflight = plan_execution_fork(
        parent_run,
        {"position": 0, "change": {"type": "none"}},
        checkpoint=placeholder,
        runtime_identity=runtime_identity,
        worker_identity=worker_identity,
    )
    if preflight.get("classification") != "exact_execution_fork":
        reason = (preflight.get("reasons") or [_reason(
            "exact_preconditions_unavailable",
            "the execution-fork planner rejected the parent and selected worker")])[0]
        return _finish(
            artifact, status="unavailable",
            code=str(reason.get("code") or "exact_preconditions_unavailable"),
            message=str(reason.get("message") or "exact preconditions are unavailable"))

    try:
        prompt_ids = _score_prompt_ids(
            engine, final_prompt, output_ids, prompt_tokens)
    except Exception as exc:
        return _finish(
            artifact, status="failed",
            code="prompt_tokenization_failed",
            message=f"matching-worker prompt tokenization failed: {type(exc).__name__}: {exc}")

    full_ids = prompt_ids + output_ids
    token_history = {
        "prompt_source": "identity_qualified_worker_score",
        "continuation_source": "recorded_trace_token_ids",
        "boundary_handling": "prompt_text_and_continuation_ids_separate",
        "prompt_tokens": len(prompt_ids),
        "output_tokens": len(output_ids),
        "total_tokens": len(full_ids),
        "original_prefill_boundary": len(prompt_ids),
        "prompt_token_ids_sha256": _sha(prompt_ids),
        "output_token_ids_sha256": _sha(output_ids),
        "full_token_ids_sha256": _sha(full_ids),
        "execution_shape": "prompt_batch_then_single_token_decode",
    }

    if checkpoint_envelope is not None:
        if not isinstance(checkpoint_envelope, Mapping):
            return _finish(
                artifact, status="unavailable",
                code="pinned_checkpoint_envelope_invalid",
                message="the resolved pinned checkpoint envelope is not an object")
        state = checkpoint_envelope.get("state")
        state = state if isinstance(state, Mapping) else {}
        pinned_tokens = state.get("tokens")
        if (
            not isinstance(pinned_tokens, list)
            or any(not isinstance(token, int) or isinstance(token, bool) for token in pinned_tokens)
            or pinned_tokens != full_ids
            or state.get("n_tokens") != len(full_ids)
            or state.get("n_past") != len(full_ids)
            or state.get("prompt_tokens") != len(prompt_ids)
        ):
            return _finish(
                artifact, status="unavailable",
                code="pinned_checkpoint_parent_mismatch",
                message=(
                    "the pinned checkpoint token history does not match the immutable parent "
                    "under the selected worker"))
        checkpoint_kwargs = None
    else:
        checkpoint_kwargs = {
            "n_past": len(full_ids),
            "prefill_to": len(prompt_ids),
            "worker_generation_id": worker["worker_generation_id"],
        }
        if sampler_wire is not None:
            checkpoint_kwargs["sampler"] = sampler_wire
        checkpoint_kwargs.update(steering_wire)
    try:
        response = (
            engine.import_checkpoint(dict(checkpoint_envelope))
            if checkpoint_envelope is not None
            else engine.create_checkpoint(full_ids, **checkpoint_kwargs)
        )
        reference = _checkpoint_reference(
            parent_id, response, worker["worker_generation_id"],
            len(prompt_ids), len(full_ids),
            allow_missing_n_tokens=checkpoint_envelope is not None)
    except Exception as exc:
        return _finish(
            artifact, status="failed",
            code="checkpoint_capture_failed",
            message=f"worker checkpoint capture failed: {type(exc).__name__}: {exc}")

    artifact["checkpoint_reference"] = reference
    artifact["token_history"] = token_history
    artifact["sampler"] = sampler_artifact
    artifact["steering"] = steering_artifact
    artifact["lifecycle"] = _lifecycle(
        "unusable", size_bytes=reference["size_bytes"])

    plan = plan_execution_fork(
        parent_run,
        {"position": 0, "change": {"type": "none"}},
        checkpoint=reference,
        runtime_identity=runtime_identity,
        worker_identity=worker_identity,
    )
    if plan.get("classification") != "exact_execution_fork":
        reason = (plan.get("reasons") or [{}])[0]
        return _finish(
            artifact, status="failed",
            code="checkpoint_plan_failed",
            message=str(reason.get("message") or "captured checkpoint failed exact planning"))

    try:
        evidence = prove_unchanged_control(parent_run, plan, engine)
    except Exception as exc:
        return _finish(
            artifact, status="failed",
            code="unchanged_control_failed",
            message=f"unchanged exact-fork control failed: {type(exc).__name__}: {exc}",
            proof={
                "status": "failed",
                "execution_fork_plan_id": plan["plan_id"],
                "exactness_regime": "prompt_boundary_reprefill",
                "error_code": type(exc).__name__,
            })

    proof = {
        "status": evidence["status"],
        "execution_fork_plan_id": plan["plan_id"],
        "exactness_regime": "prompt_boundary_reprefill",
        "control_result": deepcopy(evidence["result"]),
        "worker_receipt": deepcopy(evidence["worker_receipt"]),
    }
    if evidence["status"] != "matched":
        return _finish(
            artifact, status="failed",
            code="unchanged_control_diverged",
            message="the reconstructed checkpoint did not reproduce the parent token IDs and text",
            proof=proof)

    # Private integration seam for append-only continuation.  The public checkpoint-reference
    # artifact intentionally keeps prompt token IDs out of its closed schema, but the gateway needs
    # the exact in-memory history to prove the newly rendered conversation is a strict token suffix.
    # Publish it only after the unchanged control matched, and never mutate the immutable source run.
    if material_out is not None:
        material_out.update({
            "historical_token_ids": list(full_ids),
            "checkpoint_response": deepcopy(dict(response)),
            "capture_regime": "verified_prompt_boundary_reprefill",
            "checkpoint_provenance": (
                "durable_pin_import" if checkpoint_envelope is not None
                else "live_worker_checkpoint"
            ),
            "sampler": deepcopy(sampler_artifact),
            "sampler_wire": deepcopy(sampler_wire),
            "steering": deepcopy(steering_artifact),
        })

    artifact["lifecycle"] = _lifecycle(
        "available", size_bytes=reference["size_bytes"])
    return _finish(
        artifact, status="available",
        code="exact_checkpoint_captured",
        message=(
            (
                "the hydrated pinned checkpoint matched its unchanged exact-fork control and is "
                "eligible until worker restart, FIFO eviction, or gateway shutdown"
                if checkpoint_envelope is not None
                else
                "the ephemeral checkpoint matched its unchanged exact-fork control and is eligible "
                "until worker restart, FIFO eviction, or gateway shutdown")),
        proof=proof)
