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


def normalize_intervention(change: Mapping) -> tuple[dict | None, dict | None]:
    """Validate one execution-fork intervention using the canonical fork rules.

    Test This is a dispatcher, not a second intervention validator.  Keep this small public seam
    beside the existing private implementation so callers reuse the exact same allowed fields and
    numeric ranges as ``plan_execution_fork``.
    """
    if not isinstance(change, Mapping):
        return None, _reason("invalid_intervention", "intervention must be an object")
    return _normalize_change(change)


def sampling_intervention_contract() -> dict:
    """Describe the sampler fields accepted by ``normalize_intervention``.

    This is metadata for read-side affordances, not a second validator.  The executor/planner still
    calls :func:`normalize_intervention`; keeping this description beside the canonical checks lets
    Inspector clients render the same contract without maintaining a parallel range table elsewhere.
    """
    return {
        "type": "object",
        "properties": {
            "temperature": {"type": "number", "minimum": 0},
            "top_k": {"type": "integer", "minimum": 0},
            "top_p": {"type": "number", "minimum": 0, "maximum": 1},
            "seed": {"type": "integer", "minimum": 0},
            "rep_penalty": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": [],
        "min_fields": 1,
    }


def recorded_sampling_state(parent_run: Mapping) -> dict | None:
    """Deprecated alias for the neutral recorded-sampler projection.

    The immutable fact has one owner in the experimental kernel; this name stays only for the
    existing planning callers that still import it from here.
    """
    from clozn.experiments.execution_facts import recorded_sampler_state

    return recorded_sampler_state(parent_run)


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
    # A reconstructed text replay only needs the recorded response pieces.  Exact execution still
    # requires aligned numeric token ids, so the checkpoint path remains fail-closed below.
    if not prerequisites["token_pieces_available"]:
        return _finish(plan, _reason(
            "missing_response_token_boundary",
            "the parent needs complete recorded response token pieces",
        ))
    if supplied_checkpoint and not prerequisites["token_alignment_available"]:
        return _finish(plan, _reason(
            "missing_response_token_boundary",
            "exact execution needs complete, aligned response token pieces and token ids",
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


# Legacy planner compatibility: the old public planner remains available for its existing route
# callers, but the immutable execution facts it consumes have one neutral kernel owner.  These
# assignments deliberately preserve the old import surface without making the experimental kernel
# import this module.
from clozn.experiments.execution_facts import (  # noqa: E402  (after legacy planner definitions)
    KNOWN_CHANGES as _NEUTRAL_KNOWN_CHANGES,
    RECONSTRUCTED_CHANGES as _NEUTRAL_RECONSTRUCTED_CHANGES,
    RECONSTRUCTION_DIFFERENCES as _NEUTRAL_RECONSTRUCTION_DIFFERENCES,
    parent_execution_fingerprint as _neutral_parent_execution_fingerprint,
    parent_runtime_projection as _neutral_parent_runtime_projection,
    recorded_execution_prerequisites as _neutral_recorded_execution_prerequisites,
    normalize_checkpoint_reference as _neutral_normalize_checkpoint_reference,
    runtime_projection as _neutral_runtime_projection,
    worker_identity_projection as _neutral_worker_identity_projection,
)

KNOWN_CHANGES = _NEUTRAL_KNOWN_CHANGES
RECONSTRUCTED_CHANGES = _NEUTRAL_RECONSTRUCTED_CHANGES
RECONSTRUCTION_DIFFERENCES = _NEUTRAL_RECONSTRUCTION_DIFFERENCES
parent_execution_fingerprint = _neutral_parent_execution_fingerprint
parent_runtime_projection = _neutral_parent_runtime_projection
_runtime_projection = _neutral_runtime_projection
recorded_fork_prerequisites = _neutral_recorded_execution_prerequisites
_checkpoint_projection = _neutral_normalize_checkpoint_reference
_worker_projection = _neutral_worker_identity_projection
