"""Model-free planning for exact execution forks.

The private worker already knows how to truncate a saved KV checkpoint and continue it.  This module
does not call that worker.  It decides whether one requested child is eligible for that exact path,
must use the older reconstructed-text path, or cannot run honestly at all, and emits the versioned
``clozn.execution-fork.v1`` artifact that a later executor will consume.

The distinction around an exact checkpoint is intentionally strict:

* no checkpoint supplied -> reconstructed replay may be planned when its own prerequisites hold;
* checkpoint supplied but missing, expired, stale, or incompatible -> unavailable.

The second case never falls through to reconstruction.  A caller that believed it selected a saved
execution state must see that state fail, not receive a different experiment under the same button.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

from clozn import schemas

SCHEMA_VERSION = "clozn.execution-fork.v1"
CLASSIFICATIONS = (
    "exact_execution_fork",
    "reconstructed_replay",
    "unavailable",
)
# Public (no leading underscore, unlike the module's other internal projection helpers) because
# clozn.replay.rewind_fidelity's read-only fidelity projection reuses these exact three definitions --
# "expose those capabilities by importing/reusing the same definitions... never copy arrays into a
# second module where they can drift" is the whole point of naming them for reuse rather than keeping
# them a private implementation detail of the planner alone.
KNOWN_CHANGES = frozenset({"none", "force_token", "sampling", "steer", "residual_write"})
RECONSTRUCTED_CHANGES = frozenset({"none", "force_token"})
RECONSTRUCTION_DIFFERENCES = [
    "kv_state_not_restored",
    "sampler_state_reinitialized",
    "prompt_prefix_retokenized",
    "batch_shape_not_preserved",
]


def parent_execution_fingerprint(parent_run: Mapping) -> str:
    """Digest the immutable parent facts that an execution plan actually consumes."""
    if not isinstance(parent_run, Mapping):
        raise ValueError("parent_run must be an object")
    trace = parent_run.get("trace")
    trace = trace if isinstance(trace, Mapping) else {}
    return _sha256({
        "id": parent_run.get("id"),
        "identity": parent_run.get("identity"),
        "meta": parent_run.get("meta"),
        "final_prompt": parent_run.get("final_prompt"),
        "response": parent_run.get("response"),
        "trace": {
            "tokens": trace.get("tokens"),
            "token_ids": trace.get("token_ids"),
        },
    })


def _json_copy(value: Any) -> Any:
    """Return a JSON-safe detached copy, rejecting NaN and non-wire values up front."""
    try:
        encoded = json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
        return json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"execution-fork input must be JSON-serializable: {exc}") from None


def _sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def _finite_number(value: Any, *, minimum: float | None = None,
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


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _adapter_projection(value: Any) -> dict | None:
    """Normalize either ADR-004's adapter shape or a run identity's raw adapter facet."""
    if value is None or value == {}:
        return {
            "present": False,
            "identity_sha256": None,
            "artifact_sha256": None,
            "scale": None,
        }
    if not isinstance(value, Mapping):
        return None
    raw = _json_copy(dict(value))
    if raw.get("present") is False:
        if any(raw.get(name) is not None for name in ("identity_sha256", "artifact_sha256", "scale")):
            return None
        return {
            "present": False,
            "identity_sha256": None,
            "artifact_sha256": None,
            "scale": None,
        }
    supplied_hash = raw.get("identity_sha256")
    artifact_hash = raw.get("artifact_sha256")
    scale = raw.get("scale")
    # A raw run facet may not have promoted identity_sha256 yet. It is safe to derive identity from
    # the complete facet, but artifact_sha256 and scale are irreducible facts and must be recorded.
    if not (_is_hex(artifact_hash, 64)
            and _finite_number(scale)):
        return None
    if not _is_hex(supplied_hash, 64):
        identity_source = dict(raw)
        identity_source.pop("present", None)
        identity_source.pop("identity_sha256", None)
        supplied_hash = _sha256(identity_source)
    return {
        "present": True,
        "identity_sha256": supplied_hash,
        "artifact_sha256": artifact_hash,
        "scale": scale,
    }


def _runtime_projection(value: Any, *, run_meta: Mapping | None = None) -> dict | None:
    """Normalize current or parent run identity into the exact runtime key needed by this planner."""
    if not isinstance(value, Mapping):
        return None
    raw = dict(value)
    meta = dict(run_meta or {})
    model_sha = raw.get("model_sha256", raw.get("gguf_artifact_sha256"))
    template = raw.get("template_fingerprint")
    engine_build = raw.get("engine_build")
    if engine_build is None:
        ext = raw.get("ext")
        engine_artifact = (
            ext.get("engine_artifact") if isinstance(ext, Mapping) else None
        )
        artifact_sha = (
            engine_artifact.get("artifact_sha256")
            if isinstance(engine_artifact, Mapping) else None
        )
        if _is_hex(artifact_sha, 64):
            # The supervisor hashes the selected executable bytes before
            # worker launch and passes that measured digest to the gateway.
            engine_build = f"sha256:{artifact_sha}"
    ext = raw.get("ext")
    engine_artifact = ext.get("engine_artifact") if isinstance(ext, Mapping) else None
    engine_artifact_sha = (
        engine_artifact.get("artifact_sha256")
        if isinstance(engine_artifact, Mapping)
        else None
    )
    if engine_artifact_sha is not None:
        if not _is_hex(engine_artifact_sha, 64):
            return None
        observed_engine_build = f"sha256:{engine_artifact_sha}"
        if engine_build is None:
            engine_build = observed_engine_build
        elif engine_build != observed_engine_build:
            # These are two purported exact identities for the same selected executable. Never pick
            # one silently: an old named build plus a different observed artifact is not replay-safe.
            return None
    context_size = raw.get("context_size", raw.get("n_ctx", meta.get("n_ctx")))
    backend = raw.get("backend", raw.get("device", meta.get("device")))
    white_box_flags = raw.get("white_box_flags", meta.get("white_box_flags"))

    adapter_raw = raw.get("adapter")
    if adapter_raw is None:
        ext = raw.get("ext")
        adapter_raw = ext.get("adapter") if isinstance(ext, Mapping) else None
    adapter = _adapter_projection(adapter_raw)

    if not (
        _is_hex(model_sha, 64)
        and isinstance(template, str)
        and 16 <= len(template) <= 64
        and all(c in "0123456789abcdef" for c in template)
        and isinstance(engine_build, str)
        and bool(engine_build)
        and _is_int(context_size, minimum=1)
        and isinstance(backend, str)
        and bool(backend)
        and adapter is not None
        and isinstance(white_box_flags, Mapping)
        and all(isinstance(name, str) and isinstance(enabled, bool)
                for name, enabled in white_box_flags.items())
    ):
        return None
    # Use ADR 004's exact canonical field names so this digest is byte-for-byte compatible with
    # clozn.cli.worker_registry.RuntimeKey.key_sha256, not a fork-specific lookalike hash.
    key_facets = {
        "gguf_artifact_sha256": model_sha,
        "template_fingerprint": template,
        "engine_build": engine_build,
        "context_size": context_size,
        "backend": backend,
        "adapter": adapter,
        "white_box_flags": dict(sorted(white_box_flags.items())),
    }
    calculated_key = _sha256(key_facets)
    supplied_key = raw.get("key_sha256", raw.get("runtime_key_sha256"))
    if supplied_key is not None and supplied_key != calculated_key:
        return None
    return {
        "runtime_key_sha256": calculated_key,
        "model_sha256": model_sha,
        "template_fingerprint": template,
        "engine_build": engine_build,
        "context_size": context_size,
        "backend": backend,
        "adapter": adapter,
        "white_box_flags": dict(sorted(white_box_flags.items())),
    }


def parent_runtime_projection(parent_run: Mapping) -> dict | None:
    """Resolve the immutable parent's exact runtime without masking contradictions.

    Managed multi-model runs persist the authoritative ADR-004 key in their model-routing receipt.
    That receipt may be more complete than the older top-level ``identity`` block. When present and
    schema-valid it is authoritative, but every overlapping top-level identity/meta fact must agree;
    malformed routing evidence or any disagreement fails closed instead of falling back.
    """
    if not isinstance(parent_run, Mapping):
        return None
    meta = parent_run.get("meta")
    meta = meta if isinstance(meta, Mapping) else {}
    identity = parent_run.get("identity")
    identity = identity if isinstance(identity, Mapping) else {}
    routing = meta.get("model_routing")
    if routing is None:
        return _runtime_projection(identity, run_meta=meta)
    if not isinstance(routing, Mapping):
        return None
    try:
        schemas.validate(dict(routing), "clozn.model-routing.v1")
    except (schemas.ValidationError, schemas.SchemaError):
        return None
    result = routing.get("result")
    receipt = result.get("receipt") if isinstance(result, Mapping) else None
    if (
        not isinstance(result, Mapping)
        or result.get("status") != "routed"
        or not isinstance(receipt, Mapping)
    ):
        return None
    authoritative = _runtime_projection(receipt.get("runtime_key"))
    if authoritative is None:
        return None
    resolved_model = receipt.get("resolved_model_id")
    if not (
        isinstance(resolved_model, str)
        and resolved_model
        and parent_run.get("model") == resolved_model
    ):
        return None

    complete_legacy = _runtime_projection(identity, run_meta=meta)
    if complete_legacy is not None and complete_legacy != authoritative:
        return None
    expected = {
        "model_sha256": authoritative["model_sha256"],
        "template_fingerprint": authoritative["template_fingerprint"],
        "engine_build": authoritative["engine_build"],
        "context_size": authoritative["context_size"],
        "backend": authoritative["backend"],
    }
    partial = {
        "model_sha256": identity.get(
            "model_sha256", identity.get("gguf_artifact_sha256")),
        "template_fingerprint": identity.get("template_fingerprint"),
        "engine_build": identity.get("engine_build"),
        "context_size": identity.get(
            "context_size", identity.get("n_ctx", meta.get("n_ctx"))),
    }
    for name, observed in partial.items():
        if observed is not None and observed != expected[name]:
            return None

    observed_backend = identity.get(
        "backend", identity.get("device", meta.get("device")))
    if observed_backend is not None:
        authoritative_backend = authoritative["backend"]
        if authoritative_backend in {"cpu", "cuda"}:
            if observed_backend != authoritative_backend:
                return None
        elif authoritative_backend == "gpu":
            if (
                not isinstance(observed_backend, str)
                or observed_backend.strip().lower() not in {"gpu", "cuda", "metal"}
            ):
                return None
        elif observed_backend != authoritative_backend:
            return None

    adapter_raw = identity.get("adapter")
    if adapter_raw is None:
        ext = identity.get("ext")
        adapter_raw = ext.get("adapter") if isinstance(ext, Mapping) else None
    if adapter_raw is not None:
        if not isinstance(adapter_raw, Mapping):
            return None
        adapter_raw = dict(adapter_raw)
        expected_adapter = authoritative["adapter"]
        recorded_presence = adapter_raw.get("present")
        if recorded_presence is None:
            recorded_presence = bool(adapter_raw)
        if (
            not isinstance(recorded_presence, bool)
            or recorded_presence is not expected_adapter["present"]
        ):
            return None
        for name in ("identity_sha256", "artifact_sha256", "scale"):
            observed = adapter_raw.get(name)
            if observed is not None and observed != expected_adapter[name]:
                return None

    for source in (identity.get("white_box_flags"), meta.get("white_box_flags")):
        if source is None:
            continue
        if not isinstance(source, Mapping):
            return None
        for name, enabled in source.items():
            if (
                not isinstance(name, str)
                or not isinstance(enabled, bool)
                or authoritative["white_box_flags"].get(name) is not enabled
            ):
                return None
    # EngineSubstrate meta.model_id is the worker/upstream friendly identity. It is intentionally
    # not compared with the gateway's canonical resolved_model_id namespace; parent_run.model is
    # the canonical selected ID after model-routing journaling and was checked above.
    meta_sha = meta.get("model_sha256")
    if meta_sha is not None and meta_sha != authoritative["model_sha256"]:
        return None
    return authoritative


def recorded_fork_prerequisites(parent_run: Mapping) -> dict:
    """The purely RECORDED-RUN facts execution-fork planning needs, decoupled from any live worker/
    selected-runtime/checkpoint state. `plan_execution_fork` below reuses this verbatim for its own
    response-token-boundary gate (`missing_response_token_boundary` / `position_out_of_range`) and for
    the reconstruction path's `reconstruction_prompt_unavailable` gate, so there is exactly one
    implementation of "does this run have complete, aligned recorded evidence" -- never a second,
    subtly different one. `clozn.replay.rewind_fidelity.build_rewind_fidelity` (the read-only Rewind
    Fidelity projection) is the other caller: tests in both `tests/test_execution_fork_planner.py` and
    `tests/test_rewind_fidelity.py` cross-check against this same function so the live planner and the
    offline projection can never drift apart.

    Deliberately does NOT decide runtime-identity MATCH (there is no "selected" runtime without a live
    caller to select one) -- only whether the PARENT's own recorded runtime identity resolves at all
    (`parent_runtime_identity_available`, via the already-pure `parent_runtime_projection` above).

    Never raises; a malformed or absent run degrades every field to its unavailable state.
    """
    parent_run = parent_run if isinstance(parent_run, Mapping) else {}
    trace = parent_run.get("trace")
    pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
    token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None

    pieces_available = (
        isinstance(pieces, list) and bool(pieces) and all(isinstance(piece, str) for piece in pieces)
    )
    ids_available = (
        isinstance(token_ids, list) and bool(token_ids) and all(_is_int(tid) for tid in token_ids)
    )
    token_alignment_available = (
        pieces_available and ids_available and len(pieces) == len(token_ids)
    )

    final_prompt = parent_run.get("final_prompt")
    final_prompt_available = isinstance(final_prompt, str) and bool(final_prompt)

    return {
        "token_pieces_available": pieces_available,
        "token_ids_available": ids_available,
        "token_alignment_available": token_alignment_available,
        "recorded_token_count": len(pieces) if token_alignment_available else None,
        "final_prompt_available": final_prompt_available,
        "parent_runtime_identity_available": parent_runtime_projection(parent_run) is not None,
    }


def _worker_projection(value: Any) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    worker_id = value.get("worker_id")
    generation = value.get("worker_generation_id")
    protocol = value.get("protocol_version")
    if not (
        isinstance(worker_id, str) and worker_id
        and isinstance(generation, str) and generation
        and isinstance(protocol, str) and protocol
    ):
        return None
    return {
        "worker_id": worker_id,
        "worker_generation_id": generation,
        "protocol_version": protocol,
    }


def _normalize_change(change: Mapping) -> tuple[dict | None, dict | None]:
    """Return ``(exact_execution_shape, invalid_reason)`` for a known intervention."""
    raw = dict(change)
    kind = raw.get("type")
    if kind not in KNOWN_CHANGES:
        return None, _reason(
            "unsupported_intervention",
            f"intervention type {kind!r} is not supported by execution-fork v1",
        )

    if kind == "none":
        if set(raw) != {"type"}:
            return None, _reason("invalid_intervention", "none accepts no fields besides type")
        return {"type": "none"}, None

    if kind == "force_token":
        allowed = {"type", "token_id", "token_piece"}
        if set(raw) - allowed:
            return None, _reason("invalid_intervention", "force_token has unknown fields")
        token_id = raw.get("token_id")
        piece = raw.get("token_piece")
        if token_id is not None and not _is_int(token_id):
            return None, _reason(
                "invalid_intervention", "force_token token_id must be a non-negative integer")
        if piece is not None and (not isinstance(piece, str) or not piece):
            return None, _reason(
                "invalid_intervention", "force_token token_piece must be a non-empty string")
        if token_id is None and piece is None:
            return None, _reason(
                "invalid_intervention", "force_token needs token_id or token_piece")
        out = {"type": "force_token"}
        if token_id is not None:
            out["token_id"] = token_id
        if piece is not None:
            out["token_piece"] = piece
        return out, None

    if kind == "sampling":
        allowed = {"type", "temperature", "top_k", "top_p", "seed", "rep_penalty"}
        if set(raw) - allowed or len(raw) == 1:
            return None, _reason(
                "invalid_intervention", "sampling needs at least one supported sampler override")
        checks = {
            "temperature": lambda v: _finite_number(v, minimum=0),
            "top_k": lambda v: _is_int(v),
            "top_p": lambda v: _finite_number(v, minimum=0, maximum=1),
            "seed": lambda v: _is_int(v),
            "rep_penalty": lambda v: _finite_number(v) and v > 0,
        }
        if any(name in raw and not check(raw[name]) for name, check in checks.items()):
            return None, _reason("invalid_intervention", "sampling override values are out of range")
        return raw, None

    if kind == "steer":
        if raw.get("clear") is True:
            if set(raw) != {"type", "clear"}:
                return None, _reason(
                    "invalid_intervention", "steer clear cannot be combined with a vector")
            return {"type": "steer", "clear": True}, None
        allowed = {"type", "steer_vec", "steer_layer", "steer_coef"}
        vector = raw.get("steer_vec")
        if (
            set(raw) - allowed
            or not isinstance(vector, list)
            or not vector
            or any(not _finite_number(v) for v in vector)
            or ("steer_layer" in raw and not _is_int(raw["steer_layer"]))
            or ("steer_coef" in raw and not _finite_number(raw["steer_coef"]))
        ):
            return None, _reason(
                "invalid_intervention",
                "steer needs a finite non-empty steer_vec and valid optional layer/coefficient",
            )
        return raw, None

    allowed = {"type", "layer", "position", "values"}
    values = raw.get("values")
    if (
        set(raw) != allowed
        or not _is_int(raw.get("layer"), minimum=1)
        or not _is_int(raw.get("position"))
        or not isinstance(values, list)
        or not values
        or any(not _finite_number(v) for v in values)
    ):
        return None, _reason(
            "invalid_intervention",
            "residual_write needs layer>=1, position>=0, and finite non-empty values",
        )
    return raw, None


def _checkpoint_projection(value: Any) -> dict | None:
    if not isinstance(value, Mapping):
        return None
    raw = _json_copy(dict(value))
    required = ("checkpoint_id", "worker_generation_id", "state", "parent_run_id")
    if any(not isinstance(raw.get(name), str) or not raw[name] for name in required):
        return None
    if raw["state"] not in {"available", "missing", "expired"}:
        return None
    allowed = {
        "checkpoint_id", "worker_generation_id", "state", "parent_run_id",
        "prompt_tokens", "n_past", "size_bytes",
    }
    out = {name: raw[name] for name in allowed if name in raw}
    for name in ("prompt_tokens", "n_past", "size_bytes"):
        if name in out and not _is_int(out[name]):
            return None
    return out


def _artifact_base(parent_run_id: str, parent_fingerprint: str,
                   position: int, requested_change: dict,
                   checkpoint: dict | None, identity: dict) -> dict:
    change_sha = _sha256({"position": position, "change": requested_change})
    plan_subject = {
        "parent_run_id": parent_run_id,
        "parent_fingerprint_sha256": parent_fingerprint,
        "position": position,
        "change_sha256": change_sha,
        "checkpoint": (
            {
                "checkpoint_id": checkpoint["checkpoint_id"],
                "worker_generation_id": checkpoint["worker_generation_id"],
            }
            if checkpoint is not None
            else None
        ),
    }
    out = {
        "schema_version": SCHEMA_VERSION,
        "plan_id": f"fork_plan_{_sha256(plan_subject)[:20]}",
        "phase": "planned",
        "classification": "unavailable",
        "parent_run_id": parent_run_id,
        "parent_fingerprint_sha256": parent_fingerprint,
        "request": {
            "position": position,
            "change": requested_change,
            "change_sha256": change_sha,
        },
        "identity": identity,
        "exactness": {
            "regime": "unavailable",
            "source": "unavailable",
            "proof_status": "not_applicable",
        },
        "unavoidable_differences": [],
        "unchanged_control": {"required": True, "status": "unavailable"},
        "child_lineage": {
            "parent_run_id": parent_run_id,
            "source": "fork",
            "change_sha256": change_sha,
            "receipt_status": "not_created",
        },
        "reasons": [],
    }
    if checkpoint is not None:
        out["checkpoint_reference"] = checkpoint
    return out


def _finish(plan: dict, reason: dict) -> dict:
    plan["reasons"] = [reason]
    schemas.validate(plan, SCHEMA_VERSION)
    return plan


def plan_execution_fork(
    parent_run: Mapping,
    request: Mapping,
    *,
    checkpoint: Mapping | None = None,
    worker_identity: Mapping | None = None,
    runtime_identity: Mapping | None = None,
) -> dict:
    """Return one validated ``clozn.execution-fork.v1`` eligibility artifact.

    ``request`` is ``{"position": <response token index>, "change": {...}}``.  ``checkpoint`` is
    absent only when the caller is intentionally asking the classifier about legacy reconstruction;
    an explicit reference carries its availability state and is fail-closed.  ``worker_identity`` is
    the selected live worker, while ``runtime_identity`` is its exact runtime identity.  The parent
    runtime is derived from ``parent_run.identity`` plus ``parent_run.meta.n_ctx/device``.

    Malformed non-JSON inputs raise ``ValueError``.  Valid requests whose execution prerequisites are
    not met return ``classification == "unavailable"`` with one stable reason code.
    """
    if not isinstance(parent_run, Mapping):
        raise ValueError("parent_run must be an object")
    parent_id = parent_run.get("id")
    if not isinstance(parent_id, str) or not parent_id:
        raise ValueError("parent_run.id must be a non-empty string")
    if not isinstance(request, Mapping):
        raise ValueError("request must be an object")
    position = request.get("position")
    if not _is_int(position):
        raise ValueError("request.position must be a non-negative integer")
    requested_change = request.get("change")
    if not isinstance(requested_change, Mapping):
        raise ValueError("request.change must be an object")
    requested_change = _json_copy(dict(requested_change))
    if not isinstance(requested_change.get("type"), str) or not requested_change["type"]:
        raise ValueError("request.change.type must be a non-empty string")

    supplied_checkpoint = checkpoint is not None
    checkpoint_ref = _checkpoint_projection(checkpoint) if supplied_checkpoint else None
    parent_runtime = parent_runtime_projection(parent_run)
    selected_runtime = _runtime_projection(runtime_identity)
    selected_worker = _worker_projection(worker_identity)
    identity = {}
    if parent_runtime is not None:
        identity["parent_runtime"] = parent_runtime
    if selected_runtime is not None:
        identity["selected_runtime"] = selected_runtime
    if selected_worker is not None:
        identity["selected_worker"] = selected_worker

    plan = _artifact_base(
        parent_id, parent_execution_fingerprint(parent_run),
        position, requested_change, checkpoint_ref, identity)
    execution_change, change_error = _normalize_change(requested_change)
    if change_error is not None:
        return _finish(plan, change_error)

    prerequisites = recorded_fork_prerequisites(parent_run)
    if not prerequisites["token_alignment_available"]:
        return _finish(plan, _reason(
            "missing_response_token_boundary",
            "the parent needs complete, aligned response token pieces and token ids",
        ))
    token_count = prerequisites["recorded_token_count"]
    if position >= token_count:
        return _finish(plan, _reason(
            "position_out_of_range",
            f"position {position} is outside the parent response's {token_count} token boundaries",
        ))

    if parent_runtime is None or selected_runtime is None:
        return _finish(plan, _reason(
            "runtime_identity_unavailable",
            "parent and selected runtime identity must include model, template, engine, context, "
            "backend, and adapter state",
        ))
    if parent_runtime != selected_runtime:
        return _finish(plan, _reason(
            "runtime_identity_mismatch",
            "the selected runtime identity does not exactly match the parent run",
        ))
    if selected_worker is None:
        return _finish(plan, _reason(
            "worker_identity_unavailable",
            "the selected worker needs id, process-generation id, and protocol version",
        ))

    if not supplied_checkpoint:
        if execution_change["type"] not in RECONSTRUCTED_CHANGES:
            return _finish(plan, _reason(
                "reconstruction_unsupported_intervention",
                f"{execution_change['type']} requires an exact execution checkpoint",
            ))
        if not prerequisites["final_prompt_available"]:
            return _finish(plan, _reason(
                "reconstruction_prompt_unavailable",
                "reconstructed replay needs the parent's exact rendered final_prompt",
            ))
        if execution_change["type"] == "force_token" and not execution_change.get("token_piece"):
            return _finish(plan, _reason(
                "reconstruction_token_piece_unavailable",
                "reconstructed force_token needs token_piece because text is re-tokenized",
            ))
        plan["classification"] = "reconstructed_replay"
        plan["request"]["execution_change"] = execution_change
        plan["exactness"] = {
            "regime": "reconstructed_text",
            "source": "text_retokenization",
            "proof_status": "not_applicable",
        }
        plan["unavoidable_differences"] = list(RECONSTRUCTION_DIFFERENCES)
        plan["unchanged_control"] = {"required": True, "status": "required_not_run"}
        return _finish(plan, _reason(
            "checkpoint_not_supplied",
            "no exact checkpoint was supplied; the eligible path explicitly reconstructs text",
        ))

    if checkpoint_ref is None:
        return _finish(plan, _reason(
            "checkpoint_missing",
            "the supplied checkpoint reference is malformed or incomplete",
        ))
    if checkpoint_ref["state"] == "missing":
        return _finish(plan, _reason(
            "checkpoint_missing",
            "the referenced checkpoint is not present in the selected worker",
        ))
    if checkpoint_ref["state"] == "expired":
        return _finish(plan, _reason(
            "checkpoint_expired",
            "the referenced checkpoint has expired or been evicted",
        ))
    if checkpoint_ref["parent_run_id"] != parent_id:
        return _finish(plan, _reason(
            "checkpoint_parent_mismatch",
            "the checkpoint was not captured for this parent run",
        ))
    if checkpoint_ref["worker_generation_id"] != selected_worker["worker_generation_id"]:
        return _finish(plan, _reason(
            "stale_worker_generation",
            "the checkpoint belongs to a different worker process generation",
        ))
    prompt_tokens = checkpoint_ref.get("prompt_tokens")
    if not _is_int(prompt_tokens, minimum=1):
        return _finish(plan, _reason(
            "missing_prompt_boundary",
            "the checkpoint has no positive recorded prompt-token boundary",
        ))
    n_past = checkpoint_ref.get("n_past")
    truncate_to = prompt_tokens + position
    if not _is_int(n_past, minimum=1) or truncate_to > n_past:
        return _finish(plan, _reason(
            "checkpoint_range_mismatch",
            "the requested response boundary is outside the checkpoint token history",
        ))
    # The worker's exact wire path requires a numeric token id.  A text piece is valid only on the
    # explicitly reconstructed path; do not turn it into that path when an exact reference exists.
    if execution_change["type"] == "force_token" and "token_id" not in execution_change:
        return _finish(plan, _reason(
            "invalid_intervention",
            "exact force_token needs token_id; token_piece alone is reconstruction-only",
        ))

    plan["classification"] = "exact_execution_fork"
    plan["request"]["execution_change"] = execution_change
    if truncate_to > prompt_tokens:
        plan["exactness"] = {
            "regime": "generated_token_live_kv",
            "source": "live_kv",
            "proof_status": "planned",
            "truncate_to": truncate_to,
            "boundary_shape_true": True,
        }
    else:
        plan["exactness"] = {
            "regime": "prompt_boundary_reprefill",
            "source": "reprefill",
            "proof_status": "planned",
            "truncate_to": truncate_to,
            "boundary_shape_true": True,
        }
    plan["unchanged_control"] = {"required": True, "status": "required_not_run"}
    return _finish(plan, _reason(
        "exact_preconditions_met",
        "the checkpoint, identities, boundaries, and intervention are eligible for exact execution",
    ))
