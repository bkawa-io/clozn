"""The shared canonical contract for selections inside an immutable run.

Selection Inspection and Selection References intentionally use the same pure normalizer.  This
module owns only the selection's shape and the immutable evidence binding; it does not compose
inspection evidence or perform any live work.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from typing import Any

from clozn.replay.execution_fork import parent_execution_fingerprint
from clozn.runs import influence_geometry as geometry


class SelectionContractError(ValueError):
    """A malformed or currently unbindable canonical selection."""

    __test__ = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def is_span_id(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == len("span_") + 24
        and value.startswith("span_")
        and all(char in "0123456789abcdef" for char in value[len("span_"):])
    )


def sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_id(run: Mapping) -> str:
    value = run.get("id")
    if not isinstance(value, str) or not value:
        raise SelectionContractError("invalid_run", "recorded run id is unavailable")
    return value


def _trace(run: Mapping) -> dict:
    value = run.get("trace")
    return dict(value) if isinstance(value, Mapping) else {}


def _answer(run: Mapping) -> tuple[str | None, str]:
    return geometry.resolve_answer_text(dict(run))


def _token_count(run: Mapping) -> int:
    tokens = _trace(run).get("tokens")
    if not isinstance(tokens, list) or not tokens or not all(isinstance(item, str) for item in tokens):
        raise SelectionContractError(
            "invalid_token_selection", "recorded response token trace is unavailable")
    return len(tokens)


def _selection_id(run_id_value: str, subject: dict) -> str:
    return "selection_" + sha256_json({"run_id": run_id_value, **subject})[:24]


def normalize_selection(run: Mapping, selection: Mapping) -> dict:
    """Return one canonical, metadata-only selection bound to ``run``.

    The returned ``selection_id`` and ``basis`` are inspection metadata.  The portable reference
    codec deliberately strips those derived fields and stores the raw canonical coordinates plus its
    separately recomputed binding.
    """
    if not isinstance(selection, Mapping):
        raise SelectionContractError("invalid_selection", "selection must be an object")
    kind = selection.get("kind")
    if kind not in {"response_token", "answer_span", "context_span", "sampling"}:
        raise SelectionContractError("invalid_selection_kind", "selection kind is not supported")
    current_run_id = run_id(run)

    if kind in {"response_token", "sampling"}:
        if set(selection) - {"kind", "position"}:
            raise SelectionContractError("invalid_selection", "selection has unknown fields")
        position = selection.get("position", 0 if kind == "sampling" else None)
        if not is_int(position) or position < 0 or position >= _token_count(run):
            raise SelectionContractError(
                "invalid_position", "position is outside the recorded response token range")
        fingerprint = parent_execution_fingerprint(run)
        return {
            "selection_id": _selection_id(current_run_id, {
                "kind": kind, "position": position,
                "parent_execution_fingerprint": fingerprint,
            }),
            "kind": kind,
            "position": position,
            "basis": {
                "type": "recorded_execution",
                "parent_execution_fingerprint": fingerprint,
            },
        }

    if kind == "answer_span":
        if set(selection) - {"kind", "start", "end"}:
            raise SelectionContractError("invalid_selection", "selection has unknown fields")
        start, end = selection.get("start"), selection.get("end")
        response, reason = _answer(run)
        if response is None:
            raise SelectionContractError("invalid_selection_basis", reason)
        if not is_int(start) or not is_int(end) or start < 0 or end <= start or end > len(response):
            raise SelectionContractError(
                "invalid_answer_span", "answer span must be a non-empty recorded-answer range")
        response_sha256 = geometry.text_sha256(response)
        return {
            "selection_id": _selection_id(current_run_id, {
                "kind": kind, "start": start, "end": end,
                "response_sha256": response_sha256,
            }),
            "kind": kind,
            "start": start,
            "end": end,
            "basis": {
                "type": "recorded_response",
                "sha256": response_sha256,
                "unit": "unicode_code_points",
                "interval": "half_open",
            },
        }

    if set(selection) - {"kind", "source_span_id", "answer_span_id"}:
        raise SelectionContractError("invalid_selection", "selection has unknown fields")
    source_id = selection.get("source_span_id")
    answer_id = selection.get("answer_span_id")
    if not is_span_id(source_id):
        raise SelectionContractError(
            "invalid_source_span_id", "source_span_id must be a stable span id")
    if answer_id is not None and not is_span_id(answer_id):
        raise SelectionContractError(
            "invalid_answer_span_id", "answer_span_id must be a stable span id")
    subject = {"kind": kind, "source_span_id": source_id}
    if answer_id is not None:
        subject["answer_span_id"] = answer_id
    return {
        "selection_id": _selection_id(current_run_id, subject),
        "kind": kind,
        "source_span_id": source_id,
        **({"answer_span_id": answer_id} if answer_id is not None else {}),
        "basis": {"type": "stable_text_span_address"},
    }


def public_selection(normalized: Mapping) -> dict:
    """Remove inspection-only derived fields from a normalized selection."""
    return {
        key: value for key, value in normalized.items()
        if key not in {"selection_id", "basis"}
    }


def _valid_hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        char in "0123456789abcdef" for char in value)


def _address_for(run: Mapping, source_id: str) -> tuple[dict | None, str | None]:
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses

    redaction = run.get("redaction") if isinstance(run, Mapping) else None
    if ((isinstance(redaction, Mapping) and redaction.get("status") in {"redacted", "literal_redacted"})
            or "redacted" in (run.get("flags") or [])):
        return None, "source_span_redacted"

    try:
        document = build_persisted_text_span_addresses(dict(run), privacy="metadata_only")
    except Exception:
        return None, "source_span_unavailable"
    for address in document.get("addresses", []):
        if isinstance(address, Mapping) and address.get("address_id") == source_id:
            return dict(address), None
    # The current address projection intentionally changes the public address id when its canonical
    # basis drifts, so an old id may no longer be present.  Let the authoritative bridge distinguish
    # that closed failure from a generic unavailable projection without trusting stale offsets.
    try:
        from clozn.replay.span_bridge import resolve_span_address
        bridge = resolve_span_address(dict(run), source_id)
        reason = bridge.get("reason") if isinstance(bridge, Mapping) else None
        code = reason.get("code") if isinstance(reason, Mapping) else None
        if code == "span_address_not_found_or_drifted":
            return None, "source_span_not_found_or_drifted"
    except Exception:
        pass
    return None, "source_span_unavailable"


def _source_binding(run: Mapping, source_id: str) -> dict:
    address, reason = _address_for(run, source_id)
    if address is None:
        raise SelectionContractError(reason or "source_span_unavailable", "source span cannot be bound")
    resolution = address.get("resolution")
    resolution = resolution if isinstance(resolution, Mapping) else {}
    state = resolution.get("state")
    if state in {"drifted"}:
        raise SelectionContractError("source_span_drifted", "source span no longer matches recorded evidence")
    if state in {"redacted"}:
        raise SelectionContractError("source_span_redacted", "source span content is redacted")
    if state not in {"exact", "metadata_only"}:
        raise SelectionContractError("source_span_unavailable", "source span cannot be resolved")
    # Hash only the metadata-only address object.  The address itself is already the canonical span
    # identity; this digest is a supplementary drift detector and never contains source text.
    safe_address = dict(address)
    safe_resolution = dict(resolution)
    safe_resolution.pop("canonical", None)
    safe_address["resolution"] = safe_resolution
    return {
        "kind": "text_span_address",
        "source_span_id": source_id,
        "source_address_sha256": sha256_json(safe_address),
    }


def _measurement_binding(run: Mapping, source_id: str, answer_id: str) -> dict:
    from clozn.runs.influence_counterfactual import _measurement

    measurement = _measurement(run, source_id, answer_id)
    state = measurement.get("measurement_state")
    reason = measurement.get("measurement_reason")
    if state != "available":
        if state == "available" and reason in {"influence_link_not_found", "answer_span_not_found"}:
            raise SelectionContractError("relationship_missing", "measured influence relationship is missing")
        code = "influence_artifact_unavailable"
        if state in {"not_measured", "unavailable", "error"}:
            code = "influence_artifact_unavailable"
        raise SelectionContractError(code, "measured influence evidence is unavailable")
    # The helper uses measurement_state=available for a missing pair, so handle that separately.
    if reason in {"influence_link_not_found", "answer_span_not_found"}:
        raise SelectionContractError("relationship_missing", "measured influence relationship is missing")

    influence_map = run.get("influence_map")
    artifact_sha = influence_map.get("artifact_sha256") if isinstance(influence_map, Mapping) else None
    if not _valid_hash(artifact_sha):
        # Older native artifacts did not always persist an artifact digest.  Bind a compact,
        # text-free projection of the exact native measurement instead of inventing a scientific
        # score or changing any measured values.
        artifact_sha = sha256_json({
            "schema": influence_map.get("schema_version", influence_map.get("schema"))
            if isinstance(influence_map, Mapping) else None,
            "status": influence_map.get("status") if isinstance(influence_map, Mapping) else None,
            "method": influence_map.get("method") if isinstance(influence_map, Mapping) else None,
            "thresholds": influence_map.get("thresholds") if isinstance(influence_map, Mapping) else None,
            "measurement": {
                key: measurement.get(key)
                for key in ("source_span_id", "answer_span_id", "effect", "evidence_state",
                            "clears_floor", "delta_nats", "abs_delta_nats")
            },
        })
    relationship_sha = sha256_json({
        key: measurement.get(key)
        for key in ("source_span_id", "answer_span_id", "effect", "evidence_state",
                    "clears_floor", "delta_nats", "abs_delta_nats")
    })
    return {
        "kind": "measured_relationship",
        "source_span_id": source_id,
        "answer_span_id": answer_id,
        "influence_artifact_sha256": artifact_sha,
        "relationship_sha256": relationship_sha,
    }


def build_selection_binding(run: Mapping, normalized: Mapping) -> dict:
    """Derive the authoritative immutable binding for a normalized selection."""
    kind = normalized.get("kind")
    if kind in {"response_token", "sampling"}:
        fingerprint = parent_execution_fingerprint(run)
        if not _valid_hash(fingerprint):
            raise SelectionContractError(
                "selection_unbindable", "parent execution fingerprint is unavailable")
        return {"kind": "parent_execution", "sha256": fingerprint}
    if kind == "answer_span":
        response_sha = normalized.get("basis", {}).get("sha256")
        if not _valid_hash(response_sha):
            raise SelectionContractError("selection_unbindable", "recorded answer identity is unavailable")
        return {"kind": "recorded_answer", "sha256": response_sha}
    if kind == "context_span":
        source_id = normalized.get("source_span_id")
        source_binding = _source_binding(run, source_id)
        answer_id = normalized.get("answer_span_id")
        if answer_id is None:
            return source_binding
        return _measurement_binding(run, source_id, answer_id)
    raise SelectionContractError("invalid_selection_kind", "selection kind is not supported")


__all__ = [
    "SelectionContractError",
    "build_selection_binding",
    "is_int",
    "is_span_id",
    "normalize_selection",
    "public_selection",
    "run_id",
    "sha256_json",
]
