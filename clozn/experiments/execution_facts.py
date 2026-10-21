"""Neutral, model-free facts about a recorded execution.

This module is the kernel owner of the immutable facts shared by state
addressing, exact resume, diagnostics, and read-only fidelity projections.
It deliberately does not define a fork request, a fork plan, or any product
orchestration.
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


def runtime_projection(value: Any, *, run_meta: Mapping | None = None) -> dict | None:
    """Normalize a runtime identity to the canonical exact-execution key."""
    if not isinstance(value, Mapping):
        return None
    raw = dict(value)
    meta = dict(run_meta or {})
    model_sha = raw.get("model_sha256", raw.get("gguf_artifact_sha256"))
    template = raw.get("template_fingerprint")
    engine_build = raw.get("engine_build")
    ext = raw.get("ext")
    engine_artifact = ext.get("engine_artifact") if isinstance(ext, Mapping) else None
    engine_artifact_sha = engine_artifact.get("artifact_sha256") if isinstance(engine_artifact, Mapping) else None
    if engine_artifact_sha is not None:
        if not _is_hex(engine_artifact_sha, 64):
            return None
        observed = f"sha256:{engine_artifact_sha}"
        # runtime_identity() records the worker's human-readable build_id at
        # the top level while the engine-artifact provider records the exact
        # executable digest in ext.engine_artifact.  The digest is the
        # behavior-bearing runtime-key value; a non-digest build label is
        # useful provenance, but is not a contradiction.  Preserve the old
        # fallback for records that only contain the provider facet and still
        # reject two explicit digest identities that disagree.
        if (not isinstance(engine_build, str)
                or not engine_build.startswith("sha256:")):
            engine_build = observed
        elif engine_build != observed:
            return None
    context_size = raw.get("context_size", raw.get("n_ctx", meta.get("n_ctx")))
    backend = raw.get("backend", raw.get("device", meta.get("device")))
    if not (_is_hex(model_sha, 64) and isinstance(template, str)
            and 16 <= len(template) <= 64
            and all(c in "0123456789abcdef" for c in template)
            and isinstance(engine_build, str) and bool(engine_build)
            and is_int(context_size, minimum=1)
            and isinstance(backend, str) and bool(backend)):
        return None
    facets = {
        "gguf_artifact_sha256": model_sha,
        "template_fingerprint": template,
        "engine_build": engine_build,
        "context_size": context_size,
        "backend": backend,
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
    }


def runtime_projections_equivalent(left: Any, right: Any) -> bool:
    """Compare runtime projections while honoring the routed GPU abstraction.

    Model routing deliberately uses ``gpu`` as the portable backend identity,
    while a live worker reports the concrete implementation (``metal`` or
    ``cuda``).  Those spellings describe the same routed backend; every other
    behavior-bearing runtime facet must still match exactly.
    """
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        return False
    for key in ("model_sha256", "template_fingerprint", "engine_build", "context_size"):
        if left.get(key) != right.get(key):
            return False
    left_backend = left.get("backend")
    right_backend = right.get("backend")
    if left_backend == right_backend:
        return isinstance(left_backend, str) and bool(left_backend)
    gpu_backends = {"gpu", "cuda", "metal"}
    return (
        isinstance(left_backend, str)
        and isinstance(right_backend, str)
        and left_backend.strip().lower() in gpu_backends
        and right_backend.strip().lower() in gpu_backends
        and "gpu" in {left_backend.strip().lower(), right_backend.strip().lower()}
    )


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
    recorded_runtime = runtime_projection(identity, run_meta=meta)
    if recorded_runtime is not None and not runtime_projections_equivalent(
        recorded_runtime, authoritative,
    ):
        return None
    expected = {name: authoritative[name] for name in
                ("model_sha256", "template_fingerprint", "engine_build", "context_size", "backend")}
    partial = {
        "model_sha256": identity.get("model_sha256", identity.get("gguf_artifact_sha256")),
        "template_fingerprint": identity.get("template_fingerprint"),
        # When the complete recorded identity normalized successfully, compare
        # its canonical executable digest rather than the worker's friendly
        # build label.  runtime_projection() already proved that the label and
        # ext.engine_artifact facet form one non-contradictory identity.
        "engine_build": (
            recorded_runtime["engine_build"]
            if recorded_runtime is not None else identity.get("engine_build")
        ),
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
    if meta.get("model_sha256") is not None and meta.get("model_sha256") != authoritative["model_sha256"]:
        return None
    return authoritative


__all__ = [
    "KNOWN_CHANGES", "RECONSTRUCTED_CHANGES", "RECONSTRUCTION_DIFFERENCES",
    "finite_number", "is_int", "json_copy", "parent_execution_fingerprint",
    "parent_runtime_projection", "runtime_projection", "runtime_projections_equivalent", "sha256",
]
