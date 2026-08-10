"""Deterministic, run-scoped ``sel1`` references for recorded selections.

The checksum catches damaged or non-canonical URL data.  It is deliberately not authentication, a
signature, or an authorization grant.  Semantic validity is established only when the decoded
selection is re-normalized and rebound to the supplied immutable run.
"""
from __future__ import annotations

from collections.abc import Mapping
import base64
import binascii
import hashlib
import json
import re
from typing import Any

from clozn.runs.selection_contract import (
    SelectionContractError,
    build_selection_binding,
    normalize_selection,
    public_selection,
    run_id,
)

SCHEMA_VERSION = "clozn.selection-reference.v1"
PREFIX = "sel1"
MAX_REFERENCE_CHARS = 2048
MAX_PAYLOAD_BYTES = 1024
CHECKSUM_HEX_CHARS = 24
_PAYLOAD_RE = re.compile(r"^[A-Za-z0-9_-]+$")
_CHECKSUM_RE = re.compile(r"^[0-9a-f]{24}$")


class SelectionReferenceInputError(ValueError):
    """Malformed reference or malformed raw selection input."""

    __test__ = False

    def __init__(self, code: str, message: str, *, state: str = "invalid"):
        super().__init__(message)
        self.code = code
        self.state = state


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SelectionReferenceInputError("invalid_reference_json", "reference payload is not JSON-safe") from exc


def _checksum(payload_bytes: bytes) -> str:
    return hashlib.sha256(payload_bytes).hexdigest()[:CHECKSUM_HEX_CHARS]


def _encode_payload(payload: dict) -> tuple[str, str, bytes]:
    payload_bytes = _canonical_json_bytes(payload)
    if len(payload_bytes) > MAX_PAYLOAD_BYTES:
        raise SelectionReferenceInputError("reference_payload_too_large", "selection reference payload is too large")
    payload_segment = base64.urlsafe_b64encode(payload_bytes).decode("ascii").rstrip("=")
    token = f"{PREFIX}.{payload_segment}.{_checksum(payload_bytes)}"
    if len(token) > MAX_REFERENCE_CHARS:
        raise SelectionReferenceInputError("reference_too_large", "selection reference is too large")
    return token, payload_segment, payload_bytes


def _public_artifact(*, state: str, reference: str | None, run_id_value: str,
                     selection: dict | None, binding: dict | None,
                     checksum: str | None, reason: dict | None = None) -> dict:
    out = {
        "schema_version": SCHEMA_VERSION,
        "state": state,
        "reference": reference,
        "run_id": run_id_value,
        "selection": selection,
        "binding": binding,
        "integrity": {
            "algorithm": "sha256",
            "scope": "encoded_payload",
            "checksum_hex_chars": CHECKSUM_HEX_CHARS,
            "checksum": checksum,
            "status": "matched" if checksum is not None else "unavailable",
        },
        "api_href": f"/runs/{run_id_value}/selection/inspect?ref={reference}" if reference else None,
        "deep_link": {
            "run_id": run_id_value,
            "selection_ref": reference,
        },
    }
    if reason is not None:
        out["reason"] = reason
    return out


def _contract_error(exc: SelectionContractError) -> SelectionReferenceInputError:
    unavailable_codes = {
        "selection_unbindable", "source_span_unavailable", "source_span_drifted",
        "source_span_not_found_or_drifted", "source_span_redacted",
        "influence_artifact_unavailable", "relationship_missing",
    }
    state = "unavailable" if exc.code in unavailable_codes else "invalid"
    return SelectionReferenceInputError(exc.code, str(exc), state=state)


def encode_selection_reference(run: dict, selection: dict) -> dict:
    """Normalize and encode one selection, without persistence or live work."""
    try:
        normalized = normalize_selection(run, selection)
    except SelectionContractError as exc:
        raise _contract_error(exc) from None

    canonical_selection = public_selection(normalized)
    run_id_value = run_id(run)
    try:
        binding = build_selection_binding(run, normalized)
    except SelectionContractError as exc:
        # A valid selection can be inspectable while not having enough current metadata to create a
        # portable reference.  Callers expose this as HTTP 422; Selection Inspection can still show
        # the typed unavailable reference state without fabricating an address.
        if exc.code in {
            "source_span_unavailable", "source_span_drifted", "source_span_redacted",
            "source_span_not_found_or_drifted", "influence_artifact_unavailable",
            "relationship_missing", "selection_unbindable",
        }:
            code = "source_span_unavailable" if exc.code == "source_span_not_found_or_drifted" else exc.code
            return _public_artifact(
                state="unavailable", reference=None, run_id_value=run_id_value,
                selection=canonical_selection, binding=None, checksum=None,
                reason={"code": code, "message": str(exc)},
            )
        raise _contract_error(exc) from None

    payload = {"v": 1, "run_id": run_id_value, "selection": canonical_selection, "binding": binding}
    reference, _payload_segment, payload_bytes = _encode_payload(payload)
    artifact = _public_artifact(
        state="resolved", reference=reference, run_id_value=run_id_value,
        selection=canonical_selection, binding=binding, checksum=_checksum(payload_bytes),
    )
    # The selection-reference schema is the contract for this derived artifact.  Import lazily so
    # the pure codec remains usable during schema-loader startup.
    from clozn import schemas
    schemas.validate(artifact, SCHEMA_VERSION)
    return artifact


def _invalid(code: str, message: str):
    raise SelectionReferenceInputError(code, message)


def _decode_base64_payload(segment: str) -> bytes:
    if not segment or "=" in segment or not _PAYLOAD_RE.fullmatch(segment):
        _invalid("invalid_reference_encoding", "selection reference payload encoding is invalid")
    padding = "=" * ((4 - len(segment) % 4) % 4)
    try:
        decoded = base64.urlsafe_b64decode(segment + padding)
    except (ValueError, binascii.Error):
        _invalid("invalid_reference_encoding", "selection reference payload encoding is invalid")
    if len(decoded) > MAX_PAYLOAD_BYTES:
        _invalid("reference_payload_too_large", "selection reference payload is too large")
    if base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != segment:
        _invalid("noncanonical_reference", "selection reference payload is not canonical base64url")
    return decoded


def _validate_claimed_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        _invalid("invalid_reference_json", "selection reference payload must be an object")
    if set(payload) != {"v", "run_id", "selection", "binding"}:
        _invalid("invalid_reference_json", "selection reference payload has unexpected fields")
    if payload.get("v") != 1:
        _invalid("unsupported_reference_version", "selection reference version is not supported")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"]:
        _invalid("invalid_reference_json", "selection reference run id is invalid")
    selection = payload.get("selection")
    if not isinstance(selection, dict):
        _invalid("invalid_reference_json", "selection reference selection is invalid")
    kind = selection.get("kind")
    allowed = {
        "response_token": {"kind", "position"},
        "sampling": {"kind", "position"},
        "answer_span": {"kind", "start", "end"},
        "context_span": {"kind", "source_span_id", "answer_span_id"},
    }.get(kind)
    required = {
        "response_token": {"kind", "position"},
        "sampling": {"kind", "position"},
        "answer_span": {"kind", "start", "end"},
        "context_span": {"kind", "source_span_id"},
    }.get(kind, set())
    if allowed is None or set(selection) - allowed or not required.issubset(selection):
        _invalid("invalid_reference_json", "selection reference selection is invalid")
    binding = payload.get("binding")
    if not isinstance(binding, dict):
        _invalid("invalid_reference_json", "selection reference binding is invalid")
    binding_kind = binding.get("kind")
    binding_fields = {
        "parent_execution": {"kind", "sha256"},
        "recorded_answer": {"kind", "sha256"},
        "text_span_address": {"kind", "source_span_id", "source_address_sha256"},
        "measured_relationship": {
            "kind", "source_span_id", "answer_span_id", "influence_artifact_sha256", "relationship_sha256",
        },
    }.get(binding_kind)
    binding_required = {
        "parent_execution": {"kind", "sha256"},
        "recorded_answer": {"kind", "sha256"},
        "text_span_address": {"kind", "source_span_id", "source_address_sha256"},
        "measured_relationship": {
            "kind", "source_span_id", "answer_span_id", "influence_artifact_sha256", "relationship_sha256",
        },
    }.get(binding_kind, set())
    if binding_fields is None or set(binding) - binding_fields or not binding_required.issubset(binding):
        _invalid("invalid_reference_json", "selection reference binding is invalid")
    return payload


def decode_selection_reference(reference: str) -> dict:
    """Decode only the untrusted wire envelope; no run is consulted."""
    if not isinstance(reference, str) or not reference:
        _invalid("invalid_reference_encoding", "selection reference must be a non-empty string")
    if len(reference) > MAX_REFERENCE_CHARS:
        _invalid("reference_too_large", "selection reference is too large")
    parts = reference.split(".")
    if len(parts) != 3 or parts[0] != PREFIX:
        _invalid("invalid_reference_prefix", "selection reference prefix is invalid")
    payload_segment, checksum = parts[1], parts[2]
    if not _CHECKSUM_RE.fullmatch(checksum):
        _invalid("invalid_reference_encoding", "selection reference checksum encoding is invalid")
    payload_bytes = _decode_base64_payload(payload_segment)
    if _checksum(payload_bytes) != checksum:
        _invalid("reference_checksum_mismatch", "selection reference checksum did not match")
    try:
        payload_text = payload_bytes.decode("utf-8")
        payload = json.loads(payload_text, parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        _invalid("invalid_reference_json", "selection reference payload is not valid UTF-8 JSON")
    payload = _validate_claimed_payload(payload)
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "decoded",
        "reference": reference,
        "payload": payload,
        "payload_segment": payload_segment,
        "integrity": {
            "algorithm": "sha256",
            "scope": "encoded_payload",
            "checksum_hex_chars": CHECKSUM_HEX_CHARS,
            "checksum": checksum,
            "status": "matched",
        },
    }


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _resolution_artifact(run: Mapping, decoded: dict, *, state: str, reason: dict | None,
                         normalized: dict | None = None, binding: dict | None = None) -> dict:
    payload = decoded["payload"]
    reference = decoded["reference"]
    run_id_value = str(run.get("id") or payload.get("run_id") or "")
    selection = dict(payload.get("selection") or {})
    expected = normalized if normalized is not None else None
    artifact = _public_artifact(
        state=state, reference=reference, run_id_value=run_id_value,
        selection=selection, binding=binding or payload.get("binding"),
        checksum=decoded["integrity"]["checksum"], reason=reason,
    )
    if expected is not None:
        artifact["resolved_selection"] = expected
    return artifact


def resolve_selection_reference(run: dict, reference: str | dict) -> dict:
    """Revalidate a decoded reference against the supplied immutable run."""
    decoded = reference if isinstance(reference, dict) and "payload" in reference else decode_selection_reference(reference)
    payload = decoded["payload"]
    supplied_run_id = run.get("id") if isinstance(run, Mapping) else None
    if payload.get("run_id") != supplied_run_id:
        return _resolution_artifact(
            run, decoded, state="stale",
            reason=_reason("reference_run_mismatch", "reference does not belong to this run"),
        )

    try:
        normalized = normalize_selection(run, payload["selection"])
    except SelectionContractError as exc:
        state = "unavailable" if exc.code in {
            "invalid_selection_basis", "invalid_token_selection", "source_span_unavailable",
        } else "stale"
        code = exc.code if state == "unavailable" else "selection_no_longer_valid"
        return _resolution_artifact(run, decoded, state=state, reason=_reason(code, str(exc)))
    if public_selection(normalized) != payload["selection"]:
        return _resolution_artifact(
            run, decoded, state="stale",
            reason=_reason("selection_no_longer_valid", "reference selection is not canonical"),
            normalized=normalized,
        )

    try:
        expected_binding = build_selection_binding(run, normalized)
    except SelectionContractError as exc:
        if exc.code in {"source_span_drifted", "source_span_not_found_or_drifted", "relationship_missing"}:
            state = "stale"
        else:
            state = "unavailable"
        code = "source_span_drifted" if exc.code == "source_span_not_found_or_drifted" else exc.code
        return _resolution_artifact(
            run, decoded, state=state, reason=_reason(code, str(exc)),
            normalized=normalized,
        )

    if expected_binding != payload.get("binding"):
        kind = normalized.get("kind")
        if kind in {"response_token", "sampling"}:
            code = "parent_execution_changed"
        elif kind == "answer_span":
            code = "recorded_answer_changed"
        elif payload.get("binding", {}).get("kind") == "measured_relationship":
            code = "influence_artifact_changed"
        else:
            code = "source_span_drifted"
        return _resolution_artifact(
            run, decoded, state="stale", reason=_reason(code, "immutable selection evidence changed"),
            normalized=normalized, binding=expected_binding,
        )

    canonical_payload = {
        "v": 1, "run_id": supplied_run_id,
        "selection": public_selection(normalized), "binding": expected_binding,
    }
    canonical_reference, _segment, payload_bytes = _encode_payload(canonical_payload)
    artifact = _resolution_artifact(
        run, decoded, state="resolved", reason=None,
        normalized=normalized, binding=expected_binding,
    )
    artifact["reference"] = canonical_reference
    artifact["integrity"]["checksum"] = _checksum(payload_bytes)
    artifact["api_href"] = f"/runs/{supplied_run_id}/selection/inspect?ref={canonical_reference}"
    artifact["deep_link"] = {"run_id": supplied_run_id, "selection_ref": canonical_reference}
    from clozn import schemas
    schemas.validate(artifact, SCHEMA_VERSION)
    return artifact


__all__ = [
    "CHECKSUM_HEX_CHARS",
    "MAX_PAYLOAD_BYTES",
    "MAX_REFERENCE_CHARS",
    "PREFIX",
    "SCHEMA_VERSION",
    "SelectionReferenceInputError",
    "decode_selection_reference",
    "encode_selection_reference",
    "resolve_selection_reference",
]
