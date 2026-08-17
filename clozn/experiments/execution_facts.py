"""Neutral, model-free facts about a recorded execution.

This module is the kernel owner of the immutable facts shared by state
addressing, exact resume, diagnostics, and read-only fidelity projections.
It deliberately does not define a fork request, a fork plan, or any product
orchestration.  The legacy replay planner may consume these helpers, but the
experimental kernel does not depend on that planner.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

from clozn import schemas


RECONSTRUCTION_DIFFERENCES = [
    "kv_state_not_restored",
    "sampler_state_reinitialized",
    "prompt_prefix_retokenized",
    "batch_shape_not_preserved",
]
KNOWN_CHANGES = frozenset({"none", "force_token", "sampling", "steer", "residual_write"})
RECONSTRUCTED_CHANGES = frozenset({"none", "force_token"})


def json_copy(value: Any) -> Any:
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                             ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"execution facts must be JSON-serializable: {exc}") from None


def sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_hex(value: Any, length: int) -> bool:
    return isinstance(value, str) and len(value) == length and all(
        char in "0123456789abcdef" for char in value
    )


def finite_number(value: Any, *, minimum: float | None = None,
                  maximum: float | None = None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if not math.isfinite(float(value)):
        return False
    if minimum is not None and value < minimum:
        return False
    if maximum is not None and value > maximum:
        return False
    return True


def parent_execution_fingerprint(parent_run: Mapping[str, Any]) -> str:
    """Digest the immutable recorded facts consumed by execution state."""
    if not isinstance(parent_run, Mapping):
        raise ValueError("parent_run must be an object")
    trace = parent_run.get("trace")
    trace = trace if isinstance(trace, Mapping) else {}
    return sha256({
        "id": parent_run.get("id"),
        "identity": parent_run.get("identity"),
        "meta": parent_run.get("meta"),
        "final_prompt": parent_run.get("final_prompt"),
        "response": parent_run.get("response"),
        "trace": {"tokens": trace.get("tokens"), "token_ids": trace.get("token_ids")},
    })


def _adapter_projection(value: Any) -> dict | None:
    if value is None or value == {}:
        return {"present": False, "identity_sha256": None,
                "artifact_sha256": None, "scale": None}
    if not isinstance(value, Mapping):
        return None
    raw = json_copy(dict(value))
    if raw.get("present") is False:
        if any(raw.get(name) is not None for name in
               ("identity_sha256", "artifact_sha256", "scale")):
            return None
        return {"present": False, "identity_sha256": None,
                "artifact_sha256": None, "scale": None}
    supplied_hash = raw.get("identity_sha256")
    artifact_hash = raw.get("artifact_sha256")
    scale = raw.get("scale")
    if not (_is_hex(artifact_hash, 64) and finite_number(scale)):
        return None
    if not _is_hex(supplied_hash, 64):
        identity_source = dict(raw)
        identity_source.pop("present", None)
        identity_source.pop("identity_sha256", None)
        supplied_hash = sha256(identity_source)
    return {"present": True, "identity_sha256": supplied_hash,
            "artifact_sha256": artifact_hash, "scale": scale}


def runtime_projection(value: Any, *, run_meta: Mapping | None = None) -> dict | None:
    """Normalize a runtime identity to the canonical exact-execution key."""
    if not isinstance(value, Mapping):
        return None
    raw = dict(value)
    meta = dict(run_meta or {})
    model_sha = raw.get("model_sha256", raw.get("gguf_artifact_sha256"))
    template = raw.get("template_fingerprint")
    engine_build = raw.get("engine_build")
    if engine_build is None:
        ext = raw.get("ext")
        artifact = ext.get("engine_artifact") if isinstance(ext, Mapping) else None
        artifact_sha = artifact.get("artifact_sha256") if isinstance(artifact, Mapping) else None
        if _is_hex(artifact_sha, 64):
            engine_build = f"sha256:{artifact_sha}"
    ext = raw.get("ext")
    engine_artifact = ext.get("engine_artifact") if isinstance(ext, Mapping) else None
    engine_artifact_sha = engine_artifact.get("artifact_sha256") if isinstance(engine_artifact, Mapping) else None
    if engine_artifact_sha is not None:
        if not _is_hex(engine_artifact_sha, 64):
            return None
        observed = f"sha256:{engine_artifact_sha}"
        if engine_build is None:
            engine_build = observed
        elif engine_build != observed:
            return None
    context_size = raw.get("context_size", raw.get("n_ctx", meta.get("n_ctx")))
    backend = raw.get("backend", raw.get("device", meta.get("device")))
    white_box_flags = raw.get("white_box_flags", meta.get("white_box_flags"))
    adapter_raw = raw.get("adapter")
    if adapter_raw is None:
        ext = raw.get("ext")
        adapter_raw = ext.get("adapter") if isinstance(ext, Mapping) else None
    adapter = _adapter_projection(adapter_raw)
    if not (_is_hex(model_sha, 64) and isinstance(template, str)
            and 16 <= len(template) <= 64
            and all(c in "0123456789abcdef" for c in template)
            and isinstance(engine_build, str) and bool(engine_build)
            and is_int(context_size, minimum=1) and isinstance(backend, str) and bool(backend)
            and adapter is not None and isinstance(white_box_flags, Mapping)
            and all(isinstance(name, str) and isinstance(enabled, bool)
                    for name, enabled in white_box_flags.items())):
        return None
    facets = {
        "gguf_artifact_sha256": model_sha,
        "template_fingerprint": template,
        "engine_build": engine_build,
        "context_size": context_size,
        "backend": backend,
        "adapter": adapter,
        "white_box_flags": dict(sorted(white_box_flags.items())),
    }
    calculated = sha256(facets)
    supplied = raw.get("key_sha256", raw.get("runtime_key_sha256"))
    if supplied is not None and supplied != calculated:
        return None
    return {
        "runtime_key_sha256": calculated,
        "model_sha256": model_sha,
        "template_fingerprint": template,
        "engine_build": engine_build,
        "context_size": context_size,
        "backend": backend,
        "adapter": adapter,
        "white_box_flags": dict(sorted(white_box_flags.items())),
    }


def parent_runtime_projection(parent_run: Mapping[str, Any]) -> dict | None:
    """Resolve recorded runtime identity, failing closed on contradictions."""
    if not isinstance(parent_run, Mapping):
        return None
    meta = parent_run.get("meta") if isinstance(parent_run.get("meta"), Mapping) else {}
    identity = parent_run.get("identity") if isinstance(parent_run.get("identity"), Mapping) else {}
    routing = meta.get("model_routing")
    if routing is None:
        return runtime_projection(identity, run_meta=meta)
    if not isinstance(routing, Mapping):
        return None
    try:
        schemas.validate(dict(routing), "clozn.model-routing.v1")
    except (schemas.ValidationError, schemas.SchemaError):
        return None
    result = routing.get("result")
    receipt = result.get("receipt") if isinstance(result, Mapping) else None
    if (not isinstance(result, Mapping) or result.get("status") != "routed"
            or not isinstance(receipt, Mapping)):
        return None
    authoritative = runtime_projection(receipt.get("runtime_key"))
    resolved_model = receipt.get("resolved_model_id")
    if authoritative is None or not (isinstance(resolved_model, str) and resolved_model
                                     and parent_run.get("model") == resolved_model):
        return None
    complete_legacy = runtime_projection(identity, run_meta=meta)
    if complete_legacy is not None and complete_legacy != authoritative:
        return None
    expected = {name: authoritative[name] for name in
                ("model_sha256", "template_fingerprint", "engine_build", "context_size", "backend")}
    partial = {
        "model_sha256": identity.get("model_sha256", identity.get("gguf_artifact_sha256")),
        "template_fingerprint": identity.get("template_fingerprint"),
        "engine_build": identity.get("engine_build"),
        "context_size": identity.get("context_size", identity.get("n_ctx", meta.get("n_ctx"))),
    }
    if any(observed is not None and observed != expected[name]
           for name, observed in partial.items()):
        return None
    observed_backend = identity.get("backend", identity.get("device", meta.get("device")))
    if observed_backend is not None:
        authoritative_backend = authoritative["backend"]
        if authoritative_backend in {"cpu", "cuda"} and observed_backend != authoritative_backend:
            return None
        if authoritative_backend == "gpu" and (
                not isinstance(observed_backend, str)
                or observed_backend.strip().lower() not in {"gpu", "cuda", "metal"}):
            return None
        if authoritative_backend not in {"cpu", "cuda", "gpu"} and observed_backend != authoritative_backend:
            return None
    adapter_raw = identity.get("adapter")
    if adapter_raw is None:
        ext = identity.get("ext")
        adapter_raw = ext.get("adapter") if isinstance(ext, Mapping) else None
    if adapter_raw is not None:
        if not isinstance(adapter_raw, Mapping):
            return None
        expected_adapter = authoritative["adapter"]
        presence = adapter_raw.get("present")
        presence = bool(adapter_raw) if presence is None else presence
        if not isinstance(presence, bool) or presence is not expected_adapter["present"]:
            return None
        if any(adapter_raw.get(name) is not None and adapter_raw.get(name) != expected_adapter[name]
               for name in ("identity_sha256", "artifact_sha256", "scale")):
            return None
    for source in (identity.get("white_box_flags"), meta.get("white_box_flags")):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            return None
        if any(not isinstance(name, str) or not isinstance(enabled, bool)
               or authoritative["white_box_flags"].get(name) is not enabled
               for name, enabled in source.items()):
            return None
    if meta.get("model_sha256") is not None and meta.get("model_sha256") != authoritative["model_sha256"]:
        return None
    return authoritative


def recorded_execution_prerequisites(parent_run: Mapping[str, Any]) -> dict[str, Any]:
    """Return model-free readiness facts for a recorded answer-token trajectory."""
    parent_run = parent_run if isinstance(parent_run, Mapping) else {}
    trace = parent_run.get("trace")
    pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
    token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    pieces_available = isinstance(pieces, list) and bool(pieces) and all(isinstance(piece, str) for piece in pieces)
    ids_available = isinstance(token_ids, list) and bool(token_ids) and all(is_int(tid) for tid in token_ids)
    aligned = pieces_available and ids_available and len(pieces) == len(token_ids)
    final_prompt = parent_run.get("final_prompt")
    return {
        "token_pieces_available": pieces_available,
        "token_ids_available": ids_available,
        "token_alignment_available": aligned,
        "recorded_token_count": len(pieces) if pieces_available else None,
        "final_prompt_available": isinstance(final_prompt, str) and bool(final_prompt),
        "parent_runtime_identity_available": parent_runtime_projection(parent_run) is not None,
    }


def normalize_checkpoint_reference(value: Any) -> dict | None:
    """Normalize immutable checkpoint evidence without consulting a checkpoint store."""
    if not isinstance(value, Mapping):
        return None
    raw = json_copy(dict(value))
    required = ("checkpoint_id", "worker_generation_id", "state", "parent_run_id")
    if any(not isinstance(raw.get(name), str) or not raw[name] for name in required):
        return None
    if raw["state"] not in {"available", "missing", "expired"}:
        return None
    allowed = required + ("prompt_tokens", "n_past", "size_bytes")
    result = {name: raw[name] for name in allowed if name in raw}
    if any(name in result and not is_int(result[name]) for name in
           ("prompt_tokens", "n_past", "size_bytes")):
        return None
    return result


def worker_identity_projection(value: Any) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    worker_id = value.get("worker_id")
    generation = value.get("worker_generation_id")
    protocol = value.get("protocol_version")
    if not all(isinstance(item, str) and item for item in (worker_id, generation, protocol)):
        return None
    return {"worker_id": worker_id, "worker_generation_id": generation,
            "protocol_version": protocol}


def resolve_exact_resume_facts(parent_run: Mapping[str, Any], *, position: int,
                               checkpoint: Mapping[str, Any],
                               runtime_identity: Mapping[str, Any] | None,
                               worker_identity: Mapping[str, Any] | None) -> tuple[dict | None, dict]:
    """Validate a checkpoint-backed state address for exact resume.

    The returned artifact is an execution-state artifact, not an experiment or fork plan.  It is
    intentionally sufficient for the low-level ``execution_fork`` RPC and the unchanged-control
    proof, while carrying no child-lineage or legacy request concepts.
    """
    prerequisites = recorded_execution_prerequisites(parent_run)
    if not prerequisites["token_alignment_available"]:
        return None, {"code": "recorded_token_history_unavailable", "message": "recorded token IDs and pieces are not aligned"}
    count = prerequisites["recorded_token_count"]
    if not is_int(position) or position >= count:
        return None, {"code": "position_out_of_range", "message": f"position {position} is outside the recorded token history"}
    reference = normalize_checkpoint_reference(checkpoint)
    if reference is None:
        return None, {"code": "checkpoint_missing", "message": "the checkpoint reference is malformed or incomplete"}
    if reference["state"] == "missing":
        return None, {"code": "checkpoint_missing", "message": "the referenced checkpoint is not present"}
    if reference["state"] == "expired":
        return None, {"code": "checkpoint_expired", "message": "the referenced checkpoint has expired or been evicted"}
    parent_id = parent_run.get("id")
    if reference["parent_run_id"] != parent_id:
        return None, {"code": "checkpoint_parent_mismatch", "message": "the checkpoint was not captured for this parent run"}
    selected_worker = worker_identity_projection(worker_identity)
    if selected_worker is None:
        return None, {"code": "worker_identity_unavailable", "message": "the selected worker identity is unavailable"}
    if reference["worker_generation_id"] != selected_worker["worker_generation_id"]:
        return None, {"code": "stale_worker_generation", "message": "the checkpoint belongs to a different worker process generation"}
    selected_runtime = runtime_projection(runtime_identity)
    recorded_runtime = parent_runtime_projection(parent_run)
    if recorded_runtime is None or selected_runtime is None:
        return None, {"code": "runtime_identity_unavailable", "message": "parent and selected runtime identity are unavailable"}
    if recorded_runtime != selected_runtime:
        return None, {"code": "runtime_identity_mismatch", "message": "the selected runtime identity does not match the recorded runtime"}
    prompt_tokens = reference.get("prompt_tokens")
    if not is_int(prompt_tokens, minimum=1):
        return None, {"code": "missing_prompt_boundary", "message": "the checkpoint has no positive prompt-token boundary"}
    n_past = reference.get("n_past")
    truncate_to = prompt_tokens + position
    if not is_int(n_past, minimum=1) or truncate_to > n_past:
        return None, {"code": "checkpoint_range_mismatch", "message": "the requested boundary is outside checkpoint token history"}
    regime = "generated_token_live_kv" if truncate_to > prompt_tokens else "prompt_boundary_reprefill"
    source = "live_kv" if regime == "generated_token_live_kv" else "reprefill"
    artifact = {
        "schema_version": "clozn.execution-state-resolution.v1",
        "classification": "exact_execution_fork",
        "position": position,
        "checkpoint_reference": reference,
        "exactness": {"regime": regime, "source": source, "proof_status": "planned",
                      "truncate_to": truncate_to, "boundary_shape_true": True},
        "identity": {"parent_runtime": recorded_runtime, "selected_runtime": selected_runtime,
                      "selected_worker": selected_worker},
    }
    return artifact, {"code": "exact_preconditions_met", "message": "exact execution state is eligible; unchanged control must still confirm fidelity"}


def selection_identity_facts(selection: Any) -> tuple[dict | None, dict | None, object | None]:
    """Return runtime/worker/engine facts from a resolved model selection.

    This is deliberately independent of HTTP routes so run-scoped kernel routes can share the
    exact same identity normalization without importing a legacy product route.
    """
    if selection is None:
        return None, None, None
    if getattr(selection, "runtime_key", None) is not None:
        runtime = getattr(selection, "runtime_key")
        worker = getattr(selection, "worker_identity", None)
        return (dict(runtime) if isinstance(runtime, Mapping) else None,
                dict(worker) if isinstance(worker, Mapping) else None,
                getattr(selection, "engine", None))
    sub = getattr(selection, "sub", None)
    engine = getattr(selection, "engine", None) or getattr(sub, "engine", None)
    if engine is None:
        return None, None, None
    try:
        health = engine.health()
        health = health if isinstance(health, Mapping) else {}
    except Exception:
        health = {}
    runtime = getattr(sub, "runtime_identity", None)
    try:
        runtime = runtime() if callable(runtime) else runtime
    except Exception:
        runtime = None
    if not isinstance(runtime, Mapping):
        runtime_key = getattr(sub, "runtime_key", None)
        runtime = runtime_key.as_dict() if hasattr(runtime_key, "as_dict") else runtime_key
    if not isinstance(runtime, Mapping):
        try:
            identity = sub.identity_meta() if callable(getattr(sub, "identity_meta", None)) else {}
        except Exception:
            identity = {}
        try:
            meta = sub.run_meta() if callable(getattr(sub, "run_meta", None)) else {}
        except Exception:
            meta = {}
        runtime = dict(identity or {})
        runtime["context_size"] = runtime.get("context_size", (meta or {}).get("n_ctx", health.get("n_ctx")))
        runtime["backend"] = runtime.get("backend", (meta or {}).get("device", health.get("device")))
        if "white_box_flags" not in runtime:
            flags = (meta or {}).get("white_box_flags")
            if not isinstance(flags, Mapping):
                capabilities = health.get("capabilities") if isinstance(health.get("capabilities"), Mapping) else {}
                flags = {name: capabilities[name] for name in ("sae", "jlens", "attn_knockout")
                         if isinstance(capabilities.get(name), bool)}
            runtime["white_box_flags"] = dict(flags)
    worker = getattr(sub, "worker_identity", None)
    try:
        worker = worker() if callable(worker) else worker
    except Exception:
        worker = None
    if not isinstance(worker, Mapping):
        generation = health.get("worker_generation_id")
        protocol = health.get("protocol_version")
        if isinstance(generation, str) and generation and protocol is not None:
            worker = {"worker_id": generation, "worker_generation_id": generation,
                      "protocol_version": str(protocol)}
    return (dict(runtime) if isinstance(runtime, Mapping) else None,
            dict(worker) if isinstance(worker, Mapping) else None, engine)


__all__ = [
    "KNOWN_CHANGES", "RECONSTRUCTED_CHANGES", "RECONSTRUCTION_DIFFERENCES",
    "finite_number", "is_int", "json_copy", "normalize_checkpoint_reference",
    "parent_execution_fingerprint", "parent_runtime_projection", "recorded_execution_prerequisites",
    "resolve_exact_resume_facts", "runtime_projection", "selection_identity_facts", "sha256",
    "worker_identity_projection",
]
