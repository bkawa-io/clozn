"""Fail-closed contract helpers for exact appended-turn continuation.

This module is deliberately model-free.  It owns the gateway-side evidence that can be checked
without loading a model: the closed public request, the token-prefix proof, the closed receipt, and
the private worker wire document.  In particular, the full rendered conversation is evidence only;
only ``append_token_ids`` cross the worker boundary.

``orchestrate_continuation`` is an injected seam rather than an HTTP client.  Callers supply a
``worker_post(endpoint, body)`` function and an atomic ``persist_child(payload)`` function.  This
keeps the contract unit-testable and prevents this module from acquiring a networking dependency.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import time
import uuid
from typing import Any

from clozn import schemas


SCHEMA_VERSION = "clozn.time-machine-continuation.v1"
WORKER_CONTINUE_ENDPOINT = "/v1/time-machine/continue"

_HEX = frozenset("0123456789abcdef")
_TERMINAL_STATUSES = frozenset({"unavailable", "failed", "cancelled"})
_WORKER_CODES = frozenset({
    "checkpoint_unavailable", "checkpoint_expired", "checkpoint_corrupt",
    "checkpoint_identity_mismatch", "worker_generation_stale", "worker_capability_missing",
    "append_tokens_invalid", "batch_shape_unsupported", "checkpoint_import_failed",
    "worker_restore_failed", "worker_append_failed", "generation_failed", "request_cancelled",
})
_WORKER_STAGE = {
    "checkpoint_unavailable": "checkpoint",
    "checkpoint_expired": "checkpoint",
    "checkpoint_corrupt": "checkpoint",
    "checkpoint_identity_mismatch": "identity",
    "worker_generation_stale": "worker_restore",
    "worker_capability_missing": "worker_restore",
    "append_tokens_invalid": "worker_append",
    "batch_shape_unsupported": "worker_append",
    "checkpoint_import_failed": "worker_restore",
    "worker_restore_failed": "worker_restore",
    "worker_append_failed": "worker_append",
    "generation_failed": "generation",
    "request_cancelled": "request",
}


class ContinuationContractError(ValueError):
    """A caller attempted to construct a continuation outside this v1 contract."""


class ClosedRequestError(ContinuationContractError):
    """The public continuation body has an unknown, missing, or malformed field."""

    code = "request_invalid"


class AppendDerivationError(ContinuationContractError):
    """The validation render cannot prove a non-empty append-only token suffix."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


class WorkerProtocolError(ContinuationContractError):
    """The private worker reply did not prove the requested exact operation."""


@dataclass(frozen=True)
class ContinuationRequest:
    """The three and only three fields allowed in the public v1 request body."""

    turn: int
    user_content: str
    max_tokens: int

    def receipt(self, *, request_id: str, generation_config_sha256: str) -> dict:
        """Return the privacy-preserving request facet stored in the terminal receipt."""
        _require_nonempty_string(request_id, "request_id")
        _require_sha256(generation_config_sha256, "generation_config_sha256")
        encoded = self.user_content.encode("utf-8")
        return {
            "request_id": request_id,
            "turn": self.turn,
            "append_kind": "new_user_turn",
            "user_content_sha256": _sha_bytes(encoded),
            "user_content_bytes": len(encoded),
            "max_tokens": self.max_tokens,
            "generation_config_sha256": generation_config_sha256,
        }


@dataclass(frozen=True)
class AppendDerivation:
    """A full-render validation result whose suffix is safe to give to the worker."""

    historical_token_ids: tuple[int, ...]
    full_render_token_ids: tuple[int, ...]
    append_token_ids: tuple[int, ...]
    rendered_append: str
    template_fingerprint: str
    tokenizer_sha256: str
    generation_prefix_token_count: int

    def receipt(self) -> dict:
        return {
            "status": "validated",
            "derivation": "validated_full_render_suffix",
            "decode_regime": "sequential_single_token",
            "historical_token_count": len(self.historical_token_ids),
            "historical_token_ids_sha256": token_ids_sha256(self.historical_token_ids),
            "full_render_token_count": len(self.full_render_token_ids),
            "full_render_token_ids_sha256": token_ids_sha256(self.full_render_token_ids),
            "prefix_match": True,
            "append_token_ids": list(self.append_token_ids),
            "append_token_count": len(self.append_token_ids),
            "append_token_ids_sha256": token_ids_sha256(self.append_token_ids),
            "rendered_append_sha256": _sha_bytes(self.rendered_append.encode("utf-8")),
            "generation_prefix_token_count": self.generation_prefix_token_count,
            "template_fingerprint": self.template_fingerprint,
            "tokenizer_sha256": self.tokenizer_sha256,
            "special_tokens_preserved": True,
        }


def _json_copy(value: Any) -> Any:
    """Return a detached JSON wire value and reject non-finite/non-JSON facts eagerly."""
    try:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return json.loads(raw)
    except (TypeError, ValueError) as exc:
        raise ContinuationContractError(
            f"continuation evidence must be JSON-serializable: {exc}") from None


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_json(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return _sha_bytes(raw)


def token_ids_sha256(token_ids: Sequence[int]) -> str:
    """Digest token IDs using the exact private-worker v1 binary domain.

    This is intentionally *not* a JSON hash.  The NUL-terminated domain marker prevents accidental
    collision with another receipt's JSON digest, and fixed-width little-endian values make the
    gateway and C++ worker agree independent of JSON formatting.
    """
    values = _checked_token_ids(token_ids, "token_ids")
    wire = bytearray(b"clozn.time-machine.token-ids.v1\0")
    wire.extend(struct.pack("<I", len(values)))
    for token_id in values:
        wire.extend(struct.pack("<I", token_id))
    return _sha_bytes(bytes(wire))


def sampler_config_sha256(
    *,
    has_sampler: bool,
    temperature: float = 0.0,
    repeat_penalty: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
) -> str:
    """Return the worker-v1 checkpoint sampler configuration digest.

    The C++ protocol hashes the NUL-terminated domain, a one-byte presence marker, then the
    checkpoint's fixed-width fields.  Keeping the canonical form here lets the gateway verify the
    worker's sampler receipt without trusting worker-returned hashes on their own.
    """
    if not isinstance(has_sampler, bool):
        raise ContinuationContractError("has_sampler must be boolean")
    if not (
        isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
        and math.isfinite(float(temperature))
        and isinstance(repeat_penalty, (int, float)) and not isinstance(repeat_penalty, bool)
        and math.isfinite(float(repeat_penalty))
        and _is_int(top_k)
        and isinstance(top_p, (int, float)) and not isinstance(top_p, bool)
        and math.isfinite(float(top_p))
    ):
        raise ContinuationContractError("sampler configuration contains an invalid value")
    wire = bytearray(b"clozn.time-machine.sampler-config.v1\0")
    wire.extend(b"\x01" if has_sampler else b"\x00")
    wire.extend(struct.pack("<d", float(temperature)))
    wire.extend(struct.pack("<d", float(repeat_penalty)))
    wire.extend(struct.pack("<I", top_k))
    wire.extend(struct.pack("<d", float(top_p)))
    return _sha_bytes(bytes(wire))


def sampler_state_sha256(*, seed: int, rng_draws: int) -> str:
    """Return the worker-v1 checkpoint sampler RNG-state digest."""
    if not _is_int(seed) or seed > 0xFFFFFFFFFFFFFFFF:
        raise ContinuationContractError("sampler seed must be a uint64")
    if not _is_int(rng_draws) or rng_draws > 0xFFFFFFFFFFFFFFFF:
        raise ContinuationContractError("sampler rng_draws must be a uint64")
    wire = bytearray(b"clozn.time-machine.sampler-state.v1\0")
    wire.extend(struct.pack("<Q", seed))
    wire.extend(struct.pack("<Q", rng_draws))
    return _sha_bytes(bytes(wire))


def _is_int(value: Any, *, minimum: int = 0) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _checked_token_ids(value: Any, name: str, *, nonempty: bool = False) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AppendDerivationError("append_tokens_invalid", f"{name} must be an array of non-negative integers")
    values = list(value)
    if nonempty and not values:
        raise AppendDerivationError("append_empty", f"{name} must not be empty")
    if any(not _is_int(token_id) or token_id > 0xFFFFFFFF for token_id in values):
        raise AppendDerivationError("append_tokens_invalid", f"{name} contains an invalid token ID")
    return values


def _require_nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContinuationContractError(f"{name} must be a non-empty string")
    return value


def _require_sha256(value: Any, name: str) -> str:
    if not (
        isinstance(value, str)
        and len(value) == 64
        and all(char in _HEX for char in value)
    ):
        raise ContinuationContractError(f"{name} must be a lower-case SHA-256 digest")
    return value


def _require_timestamp(value: Any, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or value < 0
    ):
        raise ContinuationContractError(f"{name} must be a finite non-negative timestamp")
    return float(value)


def parse_continuation_request(payload: Mapping[str, Any]) -> ContinuationRequest:
    """Parse the closed public ``{turn, user.content, max_tokens}`` v1 body.

    A later product feature that needs a sampling/template/model override must use a new receipt
    regime.  Accepting it here would make an inherited-runtime claim untrue.
    """
    if not isinstance(payload, Mapping):
        raise ClosedRequestError("continuation request must be a JSON object")
    if set(payload) != {"turn", "user", "max_tokens"}:
        raise ClosedRequestError("v1 continuation request permits only turn, user, and max_tokens")
    turn = payload.get("turn")
    user = payload.get("user")
    max_tokens = payload.get("max_tokens")
    if not _is_int(turn):
        raise ClosedRequestError("turn must be a non-negative integer")
    if not isinstance(user, Mapping) or set(user) != {"content"}:
        raise ClosedRequestError("user must be an object containing only content")
    content = user.get("content")
    if not isinstance(content, str) or not content:
        raise ClosedRequestError("user.content must be a non-empty string")
    if not _is_int(max_tokens, minimum=1):
        raise ClosedRequestError("max_tokens must be a positive integer")
    return ContinuationRequest(turn=turn, user_content=content, max_tokens=max_tokens)


def derive_append_tokens(
    historical_token_ids: Sequence[int],
    full_render_token_ids: Sequence[int],
    *,
    rendered_append: str,
    template_fingerprint: str,
    tokenizer_sha256: str,
    generation_prefix_token_count: int = 0,
) -> AppendDerivation:
    """Prove a full validation render extends, rather than replaces, the saved token history.

    The returned full render must never be submitted to the worker.  This function returns only the
    newly appended suffix and all hashes/counts needed to audit that boundary.
    """
    historical = _checked_token_ids(historical_token_ids, "historical_token_ids", nonempty=True)
    full = _checked_token_ids(full_render_token_ids, "full_render_token_ids", nonempty=True)
    if not isinstance(rendered_append, str):
        raise AppendDerivationError("append_render_failed", "rendered_append must be a string")
    if not _is_int(generation_prefix_token_count):
        raise AppendDerivationError(
            "append_render_failed", "generation_prefix_token_count must be a non-negative integer")
    try:
        _require_sha256(tokenizer_sha256, "tokenizer_sha256")
    except ContinuationContractError as exc:
        raise AppendDerivationError("append_render_failed", str(exc)) from None
    if not (
        isinstance(template_fingerprint, str)
        and 16 <= len(template_fingerprint) <= 64
        and all(char in _HEX for char in template_fingerprint)
    ):
        raise AppendDerivationError(
            "append_render_failed", "template_fingerprint must be a lower-case hexadecimal fingerprint")
    if len(full) < len(historical) or full[:len(historical)] != historical:
        raise AppendDerivationError(
            "append_prefix_mismatch",
            "full validation render does not retain the checkpoint token history as an exact prefix",
        )
    suffix = full[len(historical):]
    if not suffix:
        raise AppendDerivationError("append_empty", "full validation render contains no appended tokens")
    return AppendDerivation(
        historical_token_ids=tuple(historical),
        full_render_token_ids=tuple(full),
        append_token_ids=tuple(suffix),
        rendered_append=rendered_append,
        template_fingerprint=template_fingerprint,
        tokenizer_sha256=tokenizer_sha256,
        generation_prefix_token_count=generation_prefix_token_count,
    )


def new_receipt_base(
    request: ContinuationRequest,
    *,
    requested_run_id: str,
    request_id: str,
    generation_config_sha256: str,
    continuation_id: str | None = None,
    created_ts: float | None = None,
) -> dict:
    """Create the common, privacy-safe envelope used by every terminal receipt builder."""
    _require_nonempty_string(requested_run_id, "requested_run_id")
    if continuation_id is None:
        continuation_id = f"tmc_{uuid.uuid4().hex[:20]}"
    if not (
        isinstance(continuation_id, str)
        and len(continuation_id) == 24
        and continuation_id.startswith("tmc_")
        and all(char in _HEX for char in continuation_id[4:])
    ):
        raise ContinuationContractError("continuation_id must have the tmc_<20 lowercase hex> form")
    stamp = time.time() if created_ts is None else created_ts
    stamp = _require_timestamp(stamp, "created_ts")
    return {
        "schema_version": SCHEMA_VERSION,
        "continuation_id": continuation_id,
        "created_ts": stamp,
        "finished_ts": stamp,
        "requested_run_id": requested_run_id,
        "source_turn": request.turn,
        "status": "failed",
        "request": request.receipt(
            request_id=request_id, generation_config_sha256=generation_config_sha256),
        "source": _unresolved_source("historical_source_unavailable", "source has not been resolved"),
        "source_checkpoint": _unavailable_checkpoint(
            "checkpoint_unavailable", "source checkpoint has not been selected"),
        "identity": _unmatched_identity("checkpoint_identity_mismatch", "runtime identity has not been verified"),
        "append": _unavailable_append("append_render_failed", "append boundary has not been derived"),
        "sampler": _unavailable_sampler("checkpoint_identity_mismatch", "sampler state has not been verified"),
        "exactness": _unconfirmed_exactness("checkpoint_identity_mismatch", "exactness is not confirmed"),
        "worker": _incomplete_worker("not_run", "checkpoint_unavailable", "worker was not invoked"),
        "child_lineage": _uncreated_child(
            requested_run_id, "not_created", "checkpoint_unavailable", "child was not created"),
        "unavoidable_differences": [],
        "reasons": [_reason("checkpoint_identity_mismatch", "continuation has not completed")],
        "failure": {
            "stage": "checkpoint",
            "code": "checkpoint_identity_mismatch",
            "message": "continuation has not completed",
            "retryable": False,
        },
    }


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _unresolved_source(code: str, message: str, *, status: str = "unavailable") -> dict:
    return {"status": status, "reasons": [_reason(code, message)]}


def _unavailable_checkpoint(code: str, message: str, *, status: str = "unavailable") -> dict:
    return {"status": status, "reasons": [_reason(code, message)]}


def _unmatched_identity(code: str, message: str, *, status: str = "unavailable") -> dict:
    return {"status": status, "reasons": [_reason(code, message)]}


def _unavailable_append(code: str, message: str, *, status: str = "failed") -> dict:
    return {"status": status, "reasons": [_reason(code, message)]}


def _unavailable_sampler(code: str, message: str, *, status: str = "unavailable") -> dict:
    return {"status": status, "reasons": [_reason(code, message)]}


def _unconfirmed_exactness(code: str, message: str, *, status: str = "not_confirmed") -> dict:
    return {
        "status": status,
        "claim": "unavailable",
        "structural_fallback_used": False,
        "reasons": [_reason(code, message)],
    }


def _incomplete_worker(status: str, code: str, message: str, *, request_id: str | None = None,
                       generation_id: str | None = None, cancelled: bool = False) -> dict:
    out = {"status": status, "cancelled": cancelled, "reasons": [_reason(code, message)]}
    if request_id:
        out["request_id"] = request_id
    if generation_id:
        out["worker_generation_id"] = generation_id
    return out


def _uncreated_child(requested_run_id: str, status: str, code: str, message: str,
                     *, source_checkpoint_run_id: str | None = None) -> dict:
    out = {
        "status": status,
        "requested_parent_run_id": requested_run_id,
        "relation": "exact_continuation",
        "parent_immutable": True,
        "reasons": [_reason(code, message)],
    }
    if source_checkpoint_run_id:
        out["source_checkpoint_run_id"] = source_checkpoint_run_id
    return out


def _base_with_evidence(base: Mapping, evidence: Mapping[str, Any] | None) -> dict:
    out = _json_copy(dict(base))
    if evidence:
        for name in ("source", "source_checkpoint", "identity", "append", "sampler"):
            if name in evidence and evidence[name] is not None:
                out[name] = _json_copy(evidence[name])
    return out


def _finish_terminal(
    base: Mapping,
    *,
    status: str,
    stage: str,
    code: str,
    message: str,
    retryable: bool,
    evidence: Mapping[str, Any] | None = None,
    finished_ts: float | None = None,
) -> dict:
    if status not in _TERMINAL_STATUSES:
        raise ContinuationContractError(f"terminal status must be one of {_TERMINAL_STATUSES}")
    out = _base_with_evidence(base, evidence)
    stamp = _require_timestamp(time.time() if finished_ts is None else finished_ts, "finished_ts")
    out.update({
        "finished_ts": stamp,
        "status": status,
        "reasons": [_reason(code, message)],
        "failure": {"stage": stage, "code": code, "message": message, "retryable": bool(retryable)},
        "exactness": _unconfirmed_exactness(code, message),
    })
    source = out["source"]
    source_run_id = source.get("source_run_id") if isinstance(source, Mapping) else None
    checkpoint = out["source_checkpoint"]
    generation = checkpoint.get("executing_worker_generation_id") if isinstance(checkpoint, Mapping) else None
    request_id = out["request"]["request_id"]
    if status == "cancelled":
        out["worker"] = _incomplete_worker(
            "cancelled", code, message, request_id=request_id, generation_id=generation, cancelled=True)
        out["child_lineage"] = _uncreated_child(
            out["requested_run_id"], "cancelled", code, message,
            source_checkpoint_run_id=source_run_id)
    else:
        worker_status = "failed" if stage in {"worker_restore", "worker_append", "generation"} else "not_run"
        out["worker"] = _incomplete_worker(
            worker_status, code, message, request_id=request_id if worker_status == "failed" else None,
            generation_id=generation if worker_status == "failed" else None)
        out["child_lineage"] = _uncreated_child(
            out["requested_run_id"], "failed" if status == "failed" else "not_created", code, message,
            source_checkpoint_run_id=source_run_id)
    validate_receipt(out)
    return out


def build_unavailable_receipt(base: Mapping, *, stage: str, code: str, message: str,
                              retryable: bool = True, evidence: Mapping[str, Any] | None = None,
                              finished_ts: float | None = None) -> dict:
    """Build and schema-validate an unavailable terminal receipt."""
    return _finish_terminal(
        base, status="unavailable", stage=stage, code=code, message=message, retryable=retryable,
        evidence=evidence, finished_ts=finished_ts)


def build_failed_receipt(base: Mapping, *, stage: str, code: str, message: str,
                         retryable: bool = False, evidence: Mapping[str, Any] | None = None,
                         finished_ts: float | None = None) -> dict:
    """Build and schema-validate a failed terminal receipt, without fabricating a child."""
    return _finish_terminal(
        base, status="failed", stage=stage, code=code, message=message, retryable=retryable,
        evidence=evidence, finished_ts=finished_ts)


def build_cancelled_receipt(base: Mapping, *, message: str = "continuation request was cancelled",
                            evidence: Mapping[str, Any] | None = None,
                            finished_ts: float | None = None) -> dict:
    """Build and schema-validate the terminal receipt for a cancellation before persistence."""
    return _finish_terminal(
        base, status="cancelled", stage="request", code="request_cancelled", message=message,
        retryable=True, evidence=evidence, finished_ts=finished_ts)


def build_completed_receipt(
    base: Mapping,
    *,
    source: Mapping,
    source_checkpoint: Mapping,
    identity: Mapping,
    append: AppendDerivation | Mapping,
    sampler: Mapping,
    worker: Mapping,
    child_run_id: str,
    worker_reply: Mapping,
    finished_ts: float | None = None,
) -> dict:
    """Build and schema-validate a completed exact-continuation receipt.

    The caller must have already validated the private worker reply with
    :func:`completed_worker_receipt`; arbitrary worker JSON cannot be promoted to a completion.
    """
    _require_nonempty_string(child_run_id, "child_run_id")
    append_receipt = append.receipt() if isinstance(append, AppendDerivation) else _json_copy(append)
    source_copy = _json_copy(source)
    checkpoint_copy = _json_copy(source_checkpoint)
    worker_copy = _json_copy(worker)
    if source_copy.get("status") != "resolved":
        raise ContinuationContractError("a completed continuation requires a resolved source")
    if checkpoint_copy.get("status") != "available":
        raise ContinuationContractError("a completed continuation requires an available checkpoint")
    if checkpoint_copy.get("capture_regime") not in {
        "organic_live_kv", "verified_prompt_boundary_reprefill",
    }:
        raise ContinuationContractError("checkpoint capture regime is not an approved continuation source")
    if worker_copy.get("status") != "completed":
        raise ContinuationContractError("a completed continuation requires a completed worker receipt")
    provenance = checkpoint_copy.get("provenance")
    if provenance == "live_worker_checkpoint":
        state_source, restore_mode = "live_kv", "live_checkpoint"
    elif provenance == "durable_pin_import":
        state_source, restore_mode = "durable_import", "durable_import"
    else:
        raise ContinuationContractError("checkpoint provenance is not an exact continuation source")
    if worker_copy.get("restore_mode") != restore_mode:
        raise ContinuationContractError("worker restore mode does not match the selected checkpoint")
    if worker_copy.get("worker_generation_id") != checkpoint_copy.get("executing_worker_generation_id"):
        raise ContinuationContractError("worker receipt generation does not match selected execution generation")
    if worker_copy.get("checkpoint_id") != checkpoint_copy.get("checkpoint_id"):
        raise ContinuationContractError("worker receipt checkpoint does not match selected checkpoint")
    stamp = time.time() if finished_ts is None else finished_ts
    stamp = _require_timestamp(stamp, "finished_ts")
    differences = ["new_append_tokens", "new_generated_suffix"]
    if sampler.get("mode") == "sample":
        differences.append("sampler_rng_advanced_for_new_generation")
    if provenance == "durable_pin_import" and (
        checkpoint_copy.get("source_worker_generation_id")
        != checkpoint_copy.get("executing_worker_generation_id")
    ):
        differences.append("worker_process_generation_changed_after_durable_import")
    out = _json_copy(dict(base))
    out.update({
        "finished_ts": stamp,
        "status": "completed",
        "source": source_copy,
        "source_checkpoint": checkpoint_copy,
        "identity": _json_copy(identity),
        "append": append_receipt,
        "sampler": _json_copy(sampler),
        "exactness": {
            "status": "confirmed",
            "claim": "exact_historical_state_append",
            "historical_state_source": state_source,
            "source_capture_regime": checkpoint_copy["capture_regime"],
            "historical_prefix_recomputed": False,
            "historical_prefix_retokenized_for_execution": False,
            "append_only_execution": True,
            "append_decode_regime": "sequential_single_token",
            "historical_boundary_preserved": True,
            "source_checkpoint_identity_verified": True,
            "worker_generation_verified": True,
            "structural_fallback_used": False,
            "fresh_full_prompt_equivalence_claimed": False,
            "worker_receipt_sha256": _sha_json(_json_copy(worker_reply)),
        },
        "worker": worker_copy,
        "child_lineage": {
            "status": "created",
            "requested_parent_run_id": out["requested_run_id"],
            "source_checkpoint_run_id": checkpoint_copy["source_run_id"],
            "child_run_id": child_run_id,
            "relation": "exact_continuation",
            "parent_immutable": True,
            "source_immutable": True,
            "receipt_persisted": True,
        },
        "unavoidable_differences": differences,
        "reasons": [_reason("continuation_completed", "exact appended-turn continuation completed")],
        "failure": None,
    })
    validate_receipt(out)
    return out


def validate_receipt(receipt: Mapping) -> None:
    """Validate a terminal receipt against the closed v1 artifact schema."""
    schemas.validate(_json_copy(dict(receipt)), SCHEMA_VERSION)


def build_worker_request(
    *,
    request: Mapping,
    source_checkpoint: Mapping,
    append: AppendDerivation | Mapping,
    sampler: Mapping,
    checkpoint_on_finish: bool | None = None,
) -> dict:
    """Create the closed gateway-to-worker append-and-generate request.

    This intentionally contains no model/template/LoRA identity or adapter hash.  Those are gateway
    responsibilities; the worker verifies its restored checkpoint payload, token history/position,
    and checkpoint-bound sampler/steering provenance.  Cancellation is keyed by ``request_id`` in
    the worker's existing registry; it is not a second request field.
    """
    checkpoint = _json_copy(source_checkpoint)
    append_receipt = append.receipt() if isinstance(append, AppendDerivation) else _json_copy(append)
    if checkpoint.get("status") != "available":
        raise ContinuationContractError("worker request requires an available checkpoint")
    if append_receipt.get("status") != "validated":
        raise ContinuationContractError("worker request requires a validated append suffix")
    for field in (
        "checkpoint_id", "executing_worker_generation_id", "payload_sha256", "token_history_sha256",
    ):
        _require_nonempty_string(checkpoint.get(field), f"source_checkpoint.{field}")
    _require_sha256(checkpoint["payload_sha256"], "source_checkpoint.payload_sha256")
    _require_sha256(checkpoint["token_history_sha256"], "source_checkpoint.token_history_sha256")
    if not _is_int(checkpoint.get("n_past"), minimum=1):
        raise ContinuationContractError("source_checkpoint.n_past must be a positive integer")
    if not _is_int(request.get("max_tokens"), minimum=1):
        raise ContinuationContractError("request.max_tokens must be a positive integer")
    _require_nonempty_string(request.get("request_id"), "request.request_id")
    token_ids = _checked_token_ids(append_receipt.get("append_token_ids"), "append_token_ids", nonempty=True)
    if append_receipt.get("append_token_count") != len(token_ids):
        raise ContinuationContractError("append token count does not match append token IDs")
    if append_receipt.get("append_token_ids_sha256") != token_ids_sha256(token_ids):
        raise ContinuationContractError("append token hash does not match append token IDs")
    # The gateway keeps expected sampler hashes as receipt evidence.  The worker restores their
    # checkpoint-bound provenance and reports its computed hashes in the reply; the closed v1 wire
    # deliberately does not duplicate those expectations.
    _require_sha256(sampler.get("config_sha256"), "sampler.config_sha256")
    sampler_state = sampler.get("state_sha256")
    if sampler_state is not None:
        _require_sha256(sampler_state, "sampler.state_sha256")
    out = {
        "checkpoint_id": checkpoint["checkpoint_id"],
        "worker_generation_id": checkpoint["executing_worker_generation_id"],
        "expected_checkpoint_payload_sha256": checkpoint["payload_sha256"],
        "expected_n_past": checkpoint["n_past"],
        "expected_token_history_sha256": checkpoint["token_history_sha256"],
        "append_token_ids": token_ids,
        "append_token_ids_sha256": append_receipt["append_token_ids_sha256"],
        "max_tokens": request["max_tokens"],
        "request_id": request["request_id"],
    }
    if checkpoint_on_finish is not None:
        if not isinstance(checkpoint_on_finish, bool):
            raise ContinuationContractError("checkpoint_on_finish must be boolean when supplied")
        out["checkpoint_on_finish"] = checkpoint_on_finish
    return out


def invoke_worker_continue(worker_post: Callable[[str, Mapping], Mapping], worker_request: Mapping) -> Mapping:
    """Call the one private worker endpoint through an injected transport function."""
    if not callable(worker_post):
        raise ContinuationContractError("worker_post must be callable")
    try:
        reply = worker_post(WORKER_CONTINUE_ENDPOINT, _json_copy(dict(worker_request)))
    except Exception as exc:
        raise WorkerProtocolError("worker invocation failed") from exc
    if not isinstance(reply, Mapping):
        raise WorkerProtocolError("worker returned a non-object continuation response")
    return _json_copy(dict(reply))


def _worker_failure(reply: Mapping) -> tuple[str, str, str, bool, str]:
    """Translate only stable worker failures; unrecognized details never enter an artifact."""
    nested = reply.get("error")
    nested = nested if isinstance(nested, Mapping) else {}
    raw_code = reply.get("code", nested.get("code"))
    if raw_code not in _WORKER_CODES:
        raw_code = "worker_protocol_error"
    if raw_code == "request_cancelled":
        return "cancelled", "request", raw_code, True, "worker cancelled the continuation request"
    stage = _WORKER_STAGE.get(raw_code, "generation")
    status = "unavailable" if raw_code in {
        "checkpoint_unavailable", "checkpoint_expired", "checkpoint_corrupt", "worker_generation_stale",
    } else "failed"
    retryable = raw_code not in {
        "checkpoint_corrupt", "checkpoint_identity_mismatch", "append_tokens_invalid", "batch_shape_unsupported",
    }
    return status, stage, raw_code, retryable, "worker could not complete the exact continuation"


def completed_worker_receipt(
    reply: Mapping,
    *,
    source_checkpoint: Mapping,
    append: AppendDerivation | Mapping,
    sampler: Mapping,
    request_id: str,
) -> dict:
    """Validate a completed private worker reply and project its closed receipt facet.

    Worker-computed sampler hashes must equal the checkpoint provenance supplied by the gateway.  The
    private worker's direct exactness fields are transformed into the nested public exactness receipt
    by :func:`build_completed_receipt`.  The worker reports an unmutated adapter state but receives no
    adapter material over this protocol; the v1 public schema's historic
    ``adapter_state_preserved`` field records that confirmation.
    """
    if not isinstance(reply, Mapping) or reply.get("status") != "completed" or reply.get("cancelled") is True:
        raise WorkerProtocolError("worker reply is not a completed continuation")
    checkpoint = _json_copy(source_checkpoint)
    append_receipt = append.receipt() if isinstance(append, AppendDerivation) else _json_copy(append)
    public_restore_mode = (
        "live_checkpoint" if checkpoint.get("provenance") == "live_worker_checkpoint" else "durable_import"
    )
    expected_append_ids = _checked_token_ids(
        append_receipt.get("append_token_ids"), "append_token_ids", nonempty=True)
    generated = _checked_token_ids(reply.get("tokens"), "tokens")
    token_pieces = reply.get("token_pieces")
    if (
        not isinstance(token_pieces, list)
        or len(token_pieces) != len(generated)
        or any(not isinstance(piece, str) for piece in token_pieces)
    ):
        raise WorkerProtocolError("worker omitted aligned generated token pieces")
    text = reply.get("text")
    if not isinstance(text, str):
        raise WorkerProtocolError("worker omitted generated text")
    observed_sampler = reply.get("sampler")
    if not isinstance(observed_sampler, Mapping):
        raise WorkerProtocolError("worker omitted sampler provenance")
    required = (
        ("request_id", request_id),
        ("worker_generation_id", checkpoint.get("executing_worker_generation_id")),
        ("checkpoint_id", checkpoint.get("checkpoint_id")),
        ("checkpoint_payload_sha256", checkpoint.get("payload_sha256")),
        # A durable pin is imported into a live worker checkpoint before this endpoint runs.  The
        # worker therefore always sees (and returns) a live checkpoint, while the public receipt
        # keeps the durable-import provenance visible.
        ("restore_mode", "live_checkpoint"),
        ("n_past_restored", checkpoint.get("n_past")),
        ("n_past_after_append", checkpoint.get("n_past", 0) + len(expected_append_ids)),
        ("append_token_count", len(expected_append_ids)),
        ("append_token_ids_sha256", token_ids_sha256(expected_append_ids)),
    )
    for field, expected in required:
        if reply.get(field) != expected:
            raise WorkerProtocolError(f"worker reply disagreed about {field}")
    sampler_required = (
        ("source", "checkpoint"),
        ("mode", sampler.get("mode")),
        ("config_sha256", sampler.get("config_sha256")),
        ("state_sha256", sampler.get("state_sha256")),
        ("rng_draws_before_append", sampler.get("rng_draws_before_append")),
    )
    for field, expected in sampler_required:
        if observed_sampler.get(field) != expected:
            raise WorkerProtocolError(f"worker sampler reply disagreed about {field}")
    if reply.get("sampler_state_preserved") is not True:
        raise WorkerProtocolError("worker did not preserve checkpoint sampler state")
    if reply.get("steering_state_preserved") is not True:
        raise WorkerProtocolError("worker did not preserve checkpoint steering state")
    exact_expectations = {
        "historical_prefix_recomputed": False,
        "historical_prefix_retokenized_for_execution": False,
        "append_only_execution": True,
        "append_decode_regime": "sequential_single_token",
        "native_grammar_constraints_applied": False,
        "additional_stop_constraints_applied": False,
        "adapter_state_mutated": False,
    }
    for field, expected in exact_expectations.items():
        if reply.get(field) != expected:
            raise WorkerProtocolError(f"worker did not confirm exactness field {field}")
    finish_reason = reply.get("finish_reason")
    if finish_reason not in {"eos", "length", "stop"}:
        raise WorkerProtocolError("worker returned an invalid finish reason")
    return {
        "status": "completed",
        "request_id": request_id,
        "worker_generation_id": checkpoint["executing_worker_generation_id"],
        "checkpoint_id": checkpoint["checkpoint_id"],
        "restore_mode": public_restore_mode,
        "n_past_restored": checkpoint["n_past"],
        "n_past_after_append": checkpoint["n_past"] + len(expected_append_ids),
        "append_token_count": len(expected_append_ids),
        "generated_token_count": len(generated),
        "generated_token_ids_sha256": token_ids_sha256(generated),
        "text_sha256": _sha_bytes(text.encode("utf-8")),
        "finish_reason": finish_reason,
        "cancelled": False,
        "sampler_state_preserved": True,
        "adapter_state_preserved": True,
        **({"final_checkpoint_id": reply["final_checkpoint_id"]}
           if isinstance(reply.get("final_checkpoint_id"), str) and reply["final_checkpoint_id"] else {}),
    }


def _cancelled(cancel_check: Callable[[], bool] | None) -> bool:
    try:
        return bool(cancel_check and cancel_check())
    except Exception:
        # A failed cancellation probe must not falsely claim a cancellation.  The worker still owns
        # cooperative cancellation through the request/cancellation ID carried on its private wire.
        return False


def _evidence(source: Mapping | None, checkpoint: Mapping | None, identity: Mapping | None,
              append: AppendDerivation | Mapping | None, sampler: Mapping | None) -> dict:
    out: dict[str, Any] = {}
    if source is not None:
        out["source"] = source
    if checkpoint is not None:
        out["source_checkpoint"] = checkpoint
    if identity is not None:
        out["identity"] = identity
    if append is not None:
        out["append"] = append.receipt() if isinstance(append, AppendDerivation) else append
    if sampler is not None:
        out["sampler"] = sampler
    return out


def _requires_unsupported_constraints(source_generation_settings: Mapping | None) -> bool:
    """Whether inherited source settings demand a constraint the append worker cannot apply.

    The caller may pass its exact stored settings object; these explicit booleans are intentionally
    named after the private worker's capability receipt.  Unknown settings do not imply a constraint,
    but a known ``true`` cannot be silently dropped or reinterpreted as a plain continuation.
    """
    if not isinstance(source_generation_settings, Mapping):
        return False
    return any(
        source_generation_settings.get(name) is True
        for name in (
            "native_grammar_constraints_required",
            "native_grammar_constraints_applied",
            "additional_stop_constraints_required",
            "additional_stop_constraints_applied",
        )
    )


def _rendered_append_evidence(value: Any) -> dict:
    if not isinstance(value, Mapping):
        raise AppendDerivationError("append_render_failed", "render_append must return an object")
    return dict(value)


def _new_child_id() -> str:
    # The run store remains the authority for its own identifier namespace.  This ID is only a stable
    # transactional reservation for the injected persister; production can supply its own factory.
    return f"run_tmc_{uuid.uuid4().hex[:20]}"


def orchestrate_continuation(
    request_payload: Mapping[str, Any],
    *,
    requested_run_id: str,
    request_id: str,
    generation_config_sha256: str,
    source: Mapping | None,
    source_checkpoint: Mapping | None,
    identity: Mapping | None,
    sampler: Mapping | None,
    historical_token_ids: Sequence[int],
    render_append: Callable[[ContinuationRequest], Mapping],
    worker_post: Callable[[str, Mapping], Mapping],
    persist_child: Callable[[Mapping], Any],
    source_generation_settings: Mapping | None = None,
    cancel_check: Callable[[], bool] | None = None,
    clock: Callable[[], float] = time.time,
    continuation_id: str | None = None,
    child_run_id_factory: Callable[[], str] | None = None,
    checkpoint_on_finish: bool | None = None,
) -> dict:
    """Execute the pure gateway continuation seam and return one schema-valid terminal receipt.

    ``render_append`` receives the parsed public request and returns
    ``full_render_token_ids``, ``rendered_append``, ``template_fingerprint``, ``tokenizer_sha256``,
    and optionally ``generation_prefix_token_count``.  It is the only callback allowed to render or
    tokenize.  ``persist_child`` receives a fully closed completed receipt plus the raw child-only
    content and must atomically persist both; a false/``None`` return is treated as failure.
    """
    request = parse_continuation_request(request_payload)
    created = _require_timestamp(clock(), "clock result")
    base = new_receipt_base(
        request,
        requested_run_id=requested_run_id,
        request_id=request_id,
        generation_config_sha256=generation_config_sha256,
        continuation_id=continuation_id,
        created_ts=created,
    )
    initial_evidence = _evidence(source, source_checkpoint, identity, None, sampler)
    if not isinstance(source, Mapping) or source.get("status") != "resolved":
        return build_unavailable_receipt(
            base, stage="source_resolution", code="historical_source_unavailable",
            message="no exact immutable source run is available for the requested turn", evidence=initial_evidence,
            finished_ts=clock())
    if not isinstance(source_checkpoint, Mapping) or source_checkpoint.get("status") != "available":
        return build_unavailable_receipt(
            base, stage="checkpoint", code="checkpoint_unavailable",
            message="no exact checkpoint is available for the resolved source run", evidence=initial_evidence,
            finished_ts=clock())
    if not isinstance(identity, Mapping) or identity.get("status") != "matched":
        return build_unavailable_receipt(
            base, stage="identity", code="checkpoint_identity_mismatch",
            message="source runtime identity is not exactly matched by the selected worker", evidence=initial_evidence,
            finished_ts=clock())
    if not isinstance(sampler, Mapping) or sampler.get("status") != "preserved":
        return build_unavailable_receipt(
            base, stage="identity", code="checkpoint_identity_mismatch",
            message="checkpoint sampler provenance is not available", evidence=initial_evidence,
            finished_ts=clock())
    if source.get("source_turn") != request.turn:
        return build_failed_receipt(
            base, stage="source_resolution", code="historical_source_unavailable",
            message="resolved source turn does not match the requested turn", evidence=initial_evidence,
            finished_ts=clock())
    if source_checkpoint.get("source_run_id") != source.get("source_run_id"):
        return build_failed_receipt(
            base, stage="checkpoint", code="checkpoint_identity_mismatch",
            message="selected checkpoint does not belong to the resolved source run", evidence=initial_evidence,
            finished_ts=clock())
    if source_checkpoint.get("capture_regime") not in {
        "organic_live_kv", "verified_prompt_boundary_reprefill",
    }:
        return build_unavailable_receipt(
            base, stage="checkpoint", code="checkpoint_unavailable",
            message="checkpoint capture regime is not eligible for exact continuation", evidence=initial_evidence,
            retryable=False, finished_ts=clock())
    if _cancelled(cancel_check):
        return build_cancelled_receipt(base, evidence=initial_evidence, finished_ts=clock())
    try:
        rendered = _rendered_append_evidence(render_append(request))
        append = derive_append_tokens(
            historical_token_ids,
            rendered["full_render_token_ids"],
            rendered_append=rendered["rendered_append"],
            template_fingerprint=rendered["template_fingerprint"],
            tokenizer_sha256=rendered["tokenizer_sha256"],
            generation_prefix_token_count=rendered.get("generation_prefix_token_count", 0),
        )
    except (AppendDerivationError, KeyError) as exc:
        code = exc.code if isinstance(exc, AppendDerivationError) else "append_render_failed"
        return build_failed_receipt(
            base, stage="append_derivation", code=code,
            message="could not derive a validated append-only token suffix", evidence=initial_evidence,
            finished_ts=clock())
    evidence = _evidence(source, source_checkpoint, identity, append, sampler)
    if source_checkpoint.get("token_history_sha256") != token_ids_sha256(append.historical_token_ids):
        return build_failed_receipt(
            base, stage="checkpoint", code="checkpoint_identity_mismatch",
            message="checkpoint token-history digest does not match the saved historical token IDs",
            evidence=evidence, finished_ts=clock())
    if append.template_fingerprint != identity.get("template_fingerprint"):
        return build_unavailable_receipt(
            base, stage="identity", code="template_identity_mismatch",
            message="validation render did not use the source chat template identity", evidence=evidence,
            finished_ts=clock())
    if append.tokenizer_sha256 != identity.get("tokenizer_sha256"):
        return build_unavailable_receipt(
            base, stage="identity", code="tokenizer_identity_mismatch",
            message="validation render did not use the source tokenizer identity", evidence=evidence,
            finished_ts=clock())
    if _requires_unsupported_constraints(source_generation_settings):
        return build_unavailable_receipt(
            base, stage="identity", code="worker_capability_missing",
            message=("source generation settings require grammar or additional stop constraints that "
                     "the exact append worker cannot apply"),
            evidence=evidence, retryable=False, finished_ts=clock())
    if _cancelled(cancel_check):
        return build_cancelled_receipt(base, evidence=evidence, finished_ts=clock())
    try:
        worker_request = build_worker_request(
            request=base["request"], source_checkpoint=source_checkpoint, append=append, sampler=sampler,
            checkpoint_on_finish=checkpoint_on_finish)
        reply = invoke_worker_continue(worker_post, worker_request)
    except (ContinuationContractError, WorkerProtocolError):
        return build_failed_receipt(
            base, stage="worker_restore", code="worker_protocol_error",
            message="worker protocol could not establish an exact continuation", evidence=evidence,
            retryable=True, finished_ts=clock())
    if reply.get("status") != "completed" or reply.get("cancelled") is True:
        status, stage, code, retryable, message = _worker_failure(reply)
        if status == "cancelled":
            return build_cancelled_receipt(base, message=message, evidence=evidence, finished_ts=clock())
        builder = build_unavailable_receipt if status == "unavailable" else build_failed_receipt
        return builder(
            base, stage=stage, code=code, message=message, retryable=retryable,
            evidence=evidence, finished_ts=clock())
    try:
        worker = completed_worker_receipt(
            reply, source_checkpoint=source_checkpoint, append=append, sampler=sampler,
            request_id=base["request"]["request_id"])
    except (ContinuationContractError, WorkerProtocolError):
        return build_failed_receipt(
            base, stage="worker_append", code="worker_protocol_error",
            message="worker completion receipt did not prove the requested append-only operation",
            evidence=evidence, retryable=False, finished_ts=clock())
    if _cancelled(cancel_check):
        return build_cancelled_receipt(base, evidence=evidence, finished_ts=clock())
    factory = child_run_id_factory or _new_child_id
    try:
        child_run_id = factory()
        _require_nonempty_string(child_run_id, "child_run_id")
        completed = build_completed_receipt(
            base, source=source, source_checkpoint=source_checkpoint, identity=identity,
            append=append, sampler=sampler, worker=worker, child_run_id=child_run_id,
            worker_reply=reply, finished_ts=clock())
        persistence_payload = {
            "child_run_id": child_run_id,
            "requested_parent_run_id": requested_run_id,
            "source_checkpoint_run_id": source_checkpoint["source_run_id"],
            "source_turn": request.turn,
            "user": {"content": request.user_content},
            "max_tokens": request.max_tokens,
            "worker_result": _json_copy(reply),
            "receipt": completed,
        }
        persisted = persist_child(_json_copy(persistence_payload))
        if persisted is None or persisted is False:
            raise ContinuationContractError("persistence returned no success result")
        if isinstance(persisted, Mapping) and persisted.get("child_run_id", child_run_id) != child_run_id:
            raise ContinuationContractError("persistence changed the reserved child run ID")
    except Exception:
        return build_failed_receipt(
            base, stage="persistence", code="child_persistence_failed",
            message="could not durably create the continuation child and terminal receipt",
            evidence=evidence, retryable=True, finished_ts=clock())
    return completed
