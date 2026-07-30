"""Stable, privacy-safe text-span addresses derived from existing run evidence.

``clozn.text-span-addresses.v1`` is a projection, never a run migration.  It
gives context receipts and context-answer influence maps one address shape
without changing either artifact's native identifiers, status, method, or
measurement semantics.

Offsets are always half-open Unicode code-point offsets into one exact string.
The basis and selected span each carry an exact UTF-8 SHA-256.  A byte count or
the context receipt's historical 16-hex content hash is useful integrity
evidence, but is not enough to invent code-point offsets or a full digest after
text has been removed.  Such references remain explicitly redacted or
unavailable.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import re
from typing import Any

SCHEMA_VERSION = "clozn.text-span-addresses.v1"
CONTEXT_SCHEMA = "clozn.context-receipt.v1"
LEGACY_CONTEXT_SCHEMA = "clozn.context_receipt.v1"
INFLUENCE_SCHEMA = "clozn.context_answer_influence.v1"
INFLUENCE_EXPORT_SCHEMA = "clozn.context-answer-influence-export.v1"

OFFSET_CONTRACT = {
    "unit": "unicode_code_points",
    "interval": "half_open",
    "hash_algorithm": "sha256",
    "canonicalization": "exact_string_utf8_v1",
}

KINDS = frozenset({
    "delivered_message",
    "rendered_prompt_segment",
    "attached_source_span",
    "answer_span",
    "claim",
})
BASES = frozenset({
    "delivered_message",
    "rendered_prompt",
    "attached_source",
    "scored_answer",
    "recorded_answer",
})
_FULL_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SHORT_SHA256 = re.compile(r"^[0-9a-f]{16}$")


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode("utf-8")


def _object_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _text_details(text: str) -> dict[str, Any]:
    encoded = text.encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "code_points": len(text),
        "utf8_bytes": len(encoded),
    }


def _valid_full_hash(value: Any) -> str | None:
    return value if isinstance(value, str) and _FULL_SHA256.fullmatch(value) else None


def _valid_short_hash(value: Any) -> str | None:
    return value if isinstance(value, str) and _SHORT_SHA256.fullmatch(value) else None


def _logical_relation_key(collection: str, anchor: Any) -> str:
    """Hash a text-free native anchor so it remains usable after redaction."""
    return "rel_" + _object_digest({"collection": collection, "anchor": anchor})[:24]


def _public_native_ref(native_ref: dict) -> dict:
    allowed = {
        "artifact_schema", "collection", "id", "parent_id", "segment_id",
        "client_source_id", "source_label", "selected", "recorded_hash",
    }
    return {key: deepcopy(value) for key, value in native_ref.items() if key in allowed}


def _address_id(run_id: str, kind: str, relation_key: str, native_ref: dict,
                resolution: dict) -> str:
    canonical = resolution.get("canonical")
    identity = {
        "run_id": run_id,
        "kind": kind,
        "relation_key": relation_key,
        "native": {
            key: native_ref.get(key)
            for key in ("artifact_schema", "collection", "id", "parent_id")
            if native_ref.get(key) is not None
        },
    }
    if isinstance(canonical, dict):
        identity["canonical"] = {
            key: canonical.get(key)
            for key in ("basis", "start", "end", "basis_sha256", "span_sha256")
        }
    else:
        identity["unresolved_reason"] = resolution.get("reason")
    return "span_" + _object_digest(identity)[:24]


def _unresolved_address(*, run_id: str, kind: str, native_ref: dict,
                        relation_anchor: Any, reason: str, redacted: bool) -> dict:
    relation_key = _logical_relation_key(native_ref["collection"], relation_anchor)
    resolution = {
        "state": "redacted" if redacted else "unavailable",
        "reason": reason,
    }
    public_ref = _public_native_ref(native_ref)
    return {
        "address_id": _address_id(run_id, kind, relation_key, public_ref, resolution),
        "run_id": run_id,
        "kind": kind,
        "relation_key": relation_key,
        "native_ref": public_ref,
        "resolution": resolution,
    }


def make_text_span_address(
    *,
    run_id: str,
    kind: str,
    native_ref: dict,
    relation_anchor: Any,
    basis: str,
    start: int | None,
    end: int | None,
    privacy: str = "metadata_only",
    basis_text: str | None = None,
    basis_sha256: str | None = None,
    span_text: str | None = None,
    span_sha256: str | None = None,
    recorded_hash: tuple[str, str, str] | None = None,
    recorded_hash_mismatch_reason: str = "basis_hash_mismatch",
    redacted: bool = False,
) -> dict:
    """Build one deterministic address without extracting or interpreting text.

    ``relation_anchor`` identifies the same logical native slot across a
    parent/child pair and deliberately excludes text.  ``recorded_hash`` is
    optional source-artifact integrity evidence as ``(algorithm, value, scope)``.
    It is retained on ``native_ref`` and checked when exact text is available.
    """
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("run_id must be a non-empty string")
    if kind not in KINDS:
        raise ValueError(f"unknown text-span kind {kind!r}")
    if basis not in BASES:
        raise ValueError(f"unknown canonical basis {basis!r}")
    if privacy not in {"full", "metadata_only"}:
        raise ValueError("privacy must be full or metadata_only")
    if not isinstance(native_ref, dict):
        raise ValueError("native_ref must be an object")
    native_ref = deepcopy(native_ref)
    for required in ("artifact_schema", "collection", "id"):
        if not isinstance(native_ref.get(required), str) or not native_ref[required]:
            raise ValueError(f"native_ref.{required} must be a non-empty string")

    if recorded_hash is not None:
        algorithm, value, scope = recorded_hash
        valid_recorded = (
            (algorithm == "sha256" and _valid_full_hash(value) is not None)
            or (algorithm == "sha256_truncated_64bit" and _valid_short_hash(value) is not None)
        )
        if not valid_recorded or scope not in {"canonical_basis", "span"}:
            raise ValueError("recorded_hash must be a valid sha256 or truncated-64bit tuple")
        native_ref["recorded_hash"] = {
            "algorithm": algorithm,
            "value": value,
            "scope": scope,
        }

    if (not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool)
            or start < 0 or end < start):
        reason = (
            "source_text_redacted"
            if redacted and (start is None or end is None)
            else "invalid_offsets"
        )
        return _unresolved_address(
            run_id=run_id, kind=kind, native_ref=native_ref,
            relation_anchor=relation_anchor, reason=reason, redacted=redacted,
        )

    expected_basis = _valid_full_hash(basis_sha256)
    expected_span = _valid_full_hash(span_sha256)
    drift_reason = None
    actual_span_text = None
    basis_details = None
    span_details = None

    if isinstance(basis_text, str):
        try:
            basis_details = _text_details(basis_text)
        except UnicodeEncodeError:
            return _unresolved_address(
                run_id=run_id, kind=kind, native_ref=native_ref,
                relation_anchor=relation_anchor, reason="invalid_text_encoding", redacted=redacted,
            )
        if end > len(basis_text):
            return _unresolved_address(
                run_id=run_id, kind=kind, native_ref=native_ref,
                relation_anchor=relation_anchor, reason="offset_out_of_bounds", redacted=redacted,
            )
        actual_span_text = basis_text[start:end]
        span_details = _text_details(actual_span_text)

        if expected_basis is not None and expected_basis != basis_details["sha256"]:
            drift_reason = "basis_hash_mismatch"
        if expected_span is not None and expected_span != span_details["sha256"]:
            drift_reason = drift_reason or "span_text_hash_mismatch"
        if isinstance(span_text, str) and span_text != actual_span_text:
            drift_reason = drift_reason or "span_text_hash_mismatch"

        if recorded_hash is not None:
            algorithm, value, _scope = recorded_hash
            compared = basis_details["sha256"]
            if algorithm == "sha256_truncated_64bit":
                compared = compared[:16]
            if value != compared:
                # A native receipt mismatch is the most specific diagnosis,
                # even when the same recorded digest was also supplied as the
                # portable basis hash.
                drift_reason = recorded_hash_mismatch_reason
    else:
        # A portable metadata-only influence export already has exact offsets
        # and full basis/span digests.  That is sufficient to address the span,
        # but never sufficient to expose its text.
        if expected_basis is None or expected_span is None:
            reason = "source_text_redacted" if redacted else "canonical_basis_unavailable"
            return _unresolved_address(
                run_id=run_id, kind=kind, native_ref=native_ref,
                relation_anchor=relation_anchor, reason=reason, redacted=redacted,
            )
        if isinstance(span_text, str):
            try:
                supplied = _text_details(span_text)
            except UnicodeEncodeError:
                return _unresolved_address(
                    run_id=run_id, kind=kind, native_ref=native_ref,
                    relation_anchor=relation_anchor, reason="invalid_text_encoding", redacted=redacted,
                )
            if supplied["sha256"] != expected_span:
                drift_reason = "span_text_hash_mismatch"

    canonical = {
        "basis": basis,
        "unit": "unicode_code_points",
        "interval": "half_open",
        "start": start,
        "end": end,
        "basis_sha256": basis_details["sha256"] if basis_details else expected_basis,
        "span_sha256": span_details["sha256"] if span_details else expected_span,
        "span_code_points": end - start,
    }
    if basis_details:
        canonical["basis_code_points"] = basis_details["code_points"]
        canonical["basis_utf8_bytes"] = basis_details["utf8_bytes"]
    if span_details:
        canonical["span_utf8_bytes"] = span_details["utf8_bytes"]
    # Drifted evidence reports the observed hashes and offsets, but never the
    # disputed literal.  This also keeps every drifted resolution compatible
    # with one closed, text-free schema shape regardless of top-level privacy.
    if privacy == "full" and actual_span_text is not None and drift_reason is None:
        canonical["text"] = actual_span_text

    state = "drifted" if drift_reason else (
        "exact" if privacy == "full" and actual_span_text is not None else "metadata_only"
    )
    resolution = {"state": state, "canonical": canonical}
    if drift_reason:
        resolution["reason"] = drift_reason
    relation_key = _logical_relation_key(native_ref["collection"], relation_anchor)
    public_ref = _public_native_ref(native_ref)
    return {
        "address_id": _address_id(run_id, kind, relation_key, public_ref, resolution),
        "run_id": run_id,
        "kind": kind,
        "relation_key": relation_key,
        "native_ref": public_ref,
        "resolution": resolution,
    }


def _run_is_redacted(run: dict) -> bool:
    redaction = run.get("redaction")
    return (
        isinstance(redaction, dict)
        and redaction.get("status") in {"redacted", "literal_redacted"}
    ) or "redacted" in (run.get("flags") or [])


def _run_is_fully_redacted(run: dict) -> bool:
    redaction = run.get("redaction")
    return (
        isinstance(redaction, dict) and redaction.get("status") == "redacted"
    ) or "redacted" in (run.get("flags") or [])


def _context_source_record(receipt: dict, schema: str, run: dict) -> dict:
    source = {"schema": schema}
    privacy = receipt.get("privacy")
    if isinstance(privacy, str) and privacy:
        source["privacy"] = privacy
    redaction = run.get("redaction")
    redaction_status = (
        redaction.get("status") if isinstance(redaction, dict) else None
    )
    if redaction_status == "redacted" or "redacted" in (run.get("flags") or []):
        source["native_status"] = "redacted"
        source["privacy"] = "redacted"
        source["reason"] = "run text was removed by the persisted redaction lifecycle"
    elif redaction_status == "literal_redacted":
        source["native_status"] = "literal_redacted"
        source["reason"] = (
            "one or more stored text literals were replaced; receipt hashes may report drift"
        )
    return source


def _context_projection(run: dict, *, privacy: str) -> tuple[list[dict], dict]:
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")
    receipt = run.get("context_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    schema = (
        receipt.get("schema_version")
        if isinstance(receipt.get("schema_version"), str) else
        receipt.get("schema")
        if isinstance(receipt.get("schema"), str) else
        "clozn.run-record.legacy"
    )
    messages = run.get("messages")
    messages = messages if isinstance(messages, list) else []
    redacted = _run_is_redacted(run)
    fully_redacted = _run_is_fully_redacted(run)
    if fully_redacted:
        # A tombstone's content lifecycle is authoritative even if a malformed
        # imported record still happens to carry stale duplicated text.
        messages = []
    addresses: list[dict] = []

    delivered = receipt.get("delivered")
    delivered = delivered if isinstance(delivered, list) else None
    if delivered is not None:
        for fallback_index, segment in enumerate(delivered):
            if not isinstance(segment, dict):
                continue
            index = segment.get("original_order")
            if not isinstance(index, int) or isinstance(index, bool) or index < 0:
                index = fallback_index
            message = messages[index] if index < len(messages) and isinstance(messages[index], dict) else {}
            content = message.get("content")
            content = content if isinstance(content, str) else None
            client_source_id = segment.get("client_source_id")
            if not isinstance(client_source_id, str) or not client_source_id:
                supplied = message.get("source_id")
                client_source_id = supplied if isinstance(supplied, str) and supplied else None
            segment_id = segment.get("segment_id")
            native_id = segment_id if isinstance(segment_id, str) and segment_id else f"message-{index}"
            native_ref = {
                "artifact_schema": schema,
                "collection": "context_receipt.delivered",
                "id": native_id,
            }
            if isinstance(segment_id, str) and segment_id:
                native_ref["segment_id"] = segment_id
            if client_source_id:
                native_ref["client_source_id"] = client_source_id
            if isinstance(segment.get("source_label"), str):
                native_ref["source_label"] = segment["source_label"]
            legacy_hash = _valid_short_hash(segment.get("content_hash"))
            recorded = (
                ("sha256_truncated_64bit", legacy_hash, "canonical_basis")
                if legacy_hash else None
            )
            relation_anchor = (
                {"client_source_id": client_source_id}
                if client_source_id else {"original_order": index}
            )
            addresses.append(make_text_span_address(
                run_id=run_id,
                kind="attached_source_span" if client_source_id else "delivered_message",
                native_ref=native_ref,
                relation_anchor=relation_anchor,
                basis="attached_source" if client_source_id else "delivered_message",
                start=0 if content is not None else None,
                end=len(content) if content is not None else None,
                privacy=privacy,
                basis_text=content,
                recorded_hash=recorded,
                recorded_hash_mismatch_reason="native_content_hash_mismatch",
                redacted=redacted or content is None,
            ))
    else:
        # Pre-schema runs are projected from their immutable run messages.  No
        # context receipt is fabricated or written back.
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if not isinstance(content, str):
                continue
            client_source_id = message.get("source_id")
            client_source_id = (
                client_source_id if isinstance(client_source_id, str) and client_source_id else None
            )
            native_ref = {
                "artifact_schema": schema,
                "collection": "run.messages",
                "id": f"message-{index}",
            }
            if client_source_id:
                native_ref["client_source_id"] = client_source_id
            relation_anchor = (
                {"client_source_id": client_source_id}
                if client_source_id else {"original_order": index}
            )
            addresses.append(make_text_span_address(
                run_id=run_id,
                kind="attached_source_span" if client_source_id else "delivered_message",
                native_ref=native_ref,
                relation_anchor=relation_anchor,
                basis="attached_source" if client_source_id else "delivered_message",
                start=0,
                end=len(content),
                privacy=privacy,
                basis_text=content,
                redacted=redacted,
            ))

    rendered = receipt.get("rendered")
    rendered = rendered if isinstance(rendered, dict) else {}
    final_prompt = run.get("final_prompt")
    if fully_redacted:
        final_prompt = None
    elif not isinstance(final_prompt, str):
        survived = receipt.get("survived")
        survived = survived if isinstance(survived, dict) else {}
        final_prompt = survived.get("final_prompt")
    final_prompt = final_prompt if isinstance(final_prompt, str) else None
    rendered_hash = _valid_full_hash(rendered.get("sha256"))
    if final_prompt is not None or rendered:
        native_ref = {
            "artifact_schema": schema,
            "collection": "context_receipt.rendered",
            "id": "rendered-prompt",
        }
        recorded = ("sha256", rendered_hash, "canonical_basis") if rendered_hash else None
        addresses.append(make_text_span_address(
            run_id=run_id,
            kind="rendered_prompt_segment",
            native_ref=native_ref,
            relation_anchor={"rendered_prompt": True},
            basis="rendered_prompt",
            start=0 if final_prompt is not None else None,
            end=len(final_prompt) if final_prompt is not None else None,
            privacy=privacy,
            basis_text=final_prompt,
            basis_sha256=rendered_hash,
            span_sha256=rendered_hash,
            recorded_hash=recorded,
            recorded_hash_mismatch_reason="native_rendered_hash_mismatch",
            redacted=redacted or final_prompt is None,
        ))

    return addresses, _context_source_record(receipt, schema, run)


def project_context_addresses(run: dict, *, privacy: str = "metadata_only") -> list[dict]:
    """Project current or legacy context evidence without changing the run."""
    return _context_projection(run, privacy=privacy)[0]


def _text_hash_from(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    return _valid_full_hash(value.get("text_sha256"))


def _source_kind(item: dict) -> tuple[str, str]:
    client_source = item.get("client_source_id") or item.get("external_source_id")
    if isinstance(client_source, (str, int)) and str(client_source):
        return "attached_source_span", "attached_source"
    if item.get("source_kind") == "prompt_block":
        return "rendered_prompt_segment", "rendered_prompt"
    return "delivered_message", "delivered_message"


def _native_influence_ref(schema: str, collection: str, item: dict, fallback_id: str) -> dict:
    native_id = item.get("id")
    native_ref = {
        "artifact_schema": schema,
        "collection": collection,
        "id": str(native_id) if isinstance(native_id, (str, int)) and str(native_id) else fallback_id,
    }
    for key in ("parent_id", "segment_id", "client_source_id", "source_label"):
        value = item.get(key)
        if isinstance(value, str) and value:
            native_ref[key] = value
    if isinstance(item.get("selected"), bool):
        native_ref["selected"] = item["selected"]
    return native_ref


def _reconstruct_answer_basis(spans: list[dict]) -> str | None:
    ordered = sorted(
        (item for item in spans if isinstance(item, dict)),
        key=lambda item: item.get("start", -1),
    )
    cursor = 0
    parts = []
    for item in ordered:
        start, end, text = item.get("start"), item.get("end"), item.get("text")
        if (not isinstance(start, int) or isinstance(start, bool)
                or not isinstance(end, int) or isinstance(end, bool)
                or not isinstance(text, str)
                or start != cursor or end < start or len(text) != end - start):
            return None
        parts.append(text)
        cursor = end
    return "".join(parts) if parts else None


def _influence_projection(run_id: str, artifact: dict, *,
                          privacy: str) -> tuple[list[dict], dict]:
    if not isinstance(artifact, dict):
        raise ValueError("influence artifact must be an object")
    schema = artifact.get("schema_version") or artifact.get("schema")
    if schema not in {INFLUENCE_SCHEMA, INFLUENCE_EXPORT_SCHEMA}:
        raise ValueError(
            f"influence artifact must be {INFLUENCE_SCHEMA} or {INFLUENCE_EXPORT_SCHEMA}"
        )
    source_record = {"schema": schema}
    status = artifact.get("status")
    if isinstance(status, str) and status:
        source_record["native_status"] = status
    if isinstance(artifact.get("available"), bool):
        source_record["available"] = artifact["available"]
    method = artifact.get("method")
    if isinstance(method, dict):
        source_record["method"] = deepcopy(method)
    artifact_privacy = artifact.get("privacy")
    if isinstance(artifact_privacy, str) and artifact_privacy:
        source_record["privacy"] = artifact_privacy

    prompt_sources = [
        item for item in (artifact.get("prompt_sources") or []) if isinstance(item, dict)
    ]
    source_by_id = {
        str(item["id"]): item
        for item in prompt_sources
        if isinstance(item.get("id"), (str, int)) and str(item["id"])
    }
    addresses: list[dict] = []

    for index, source in enumerate(prompt_sources):
        native_ref = _native_influence_ref(
            schema, "influence.prompt_sources", source, f"prompt-source-{index}",
        )
        kind, basis = _source_kind(source)
        source_text = source.get("text")
        source_text = source_text if isinstance(source_text, str) else None
        start = source.get("start", 0)
        end = source.get("end")
        if not isinstance(end, int) and source_text is not None:
            end = len(source_text)
        source_hash = _valid_full_hash(source.get("text_sha256"))
        client_anchor = source.get("client_source_id") or source.get("external_source_id")
        relation_anchor = (
            {"client_source_id": str(client_anchor), "source": native_ref["id"]}
            if isinstance(client_anchor, (str, int)) and str(client_anchor)
            else {"source": native_ref["id"]}
        )
        addresses.append(make_text_span_address(
            run_id=run_id,
            kind=kind,
            native_ref=native_ref,
            relation_anchor=relation_anchor,
            basis=basis,
            start=start,
            end=end,
            privacy=privacy,
            basis_text=source_text,
            basis_sha256=source_hash,
            span_sha256=source_hash,
            span_text=source_text,
            redacted=source_text is None and artifact_privacy != "full",
        ))

    for index, span in enumerate(artifact.get("prompt_spans") or []):
        if not isinstance(span, dict):
            continue
        native_ref = _native_influence_ref(
            schema, "influence.prompt_spans", span, f"prompt-span-{index}",
        )
        parent = source_by_id.get(str(span.get("parent_id") or ""))
        parent = parent if isinstance(parent, dict) else {}
        combined = {**parent, **span}
        kind, basis = _source_kind(combined)
        basis_text = parent.get("text")
        basis_text = basis_text if isinstance(basis_text, str) else None
        basis_hash = _valid_full_hash(parent.get("text_sha256"))
        span_hash = _valid_full_hash(span.get("text_sha256"))
        span_text = span.get("text")
        span_text = span_text if isinstance(span_text, str) else None
        native_id = native_ref["id"]
        client_anchor = span.get("client_source_id") or parent.get("client_source_id")
        relation_anchor = {
            "span": native_id,
            **(
                {"client_source_id": str(client_anchor)}
                if isinstance(client_anchor, (str, int)) and str(client_anchor) else {}
            ),
        }
        addresses.append(make_text_span_address(
            run_id=run_id,
            kind=kind,
            native_ref=native_ref,
            relation_anchor=relation_anchor,
            basis=basis,
            start=span.get("start"),
            end=span.get("end"),
            privacy=privacy,
            basis_text=basis_text,
            basis_sha256=basis_hash,
            span_text=span_text,
            span_sha256=span_hash,
            redacted=basis_text is None and artifact_privacy != "full",
        ))

    answer = artifact.get("answer")
    answer = answer if isinstance(answer, dict) else {}
    answer_spans = [
        item for item in (artifact.get("answer_spans") or []) if isinstance(item, dict)
    ]
    answer_basis = answer.get("scored_text")
    answer_basis = answer_basis if isinstance(answer_basis, str) else _reconstruct_answer_basis(answer_spans)
    answer_hashes = artifact.get("answer_hashes")
    answer_hashes = answer_hashes if isinstance(answer_hashes, dict) else {}
    answer_basis_hash = _text_hash_from(answer_hashes.get("scored_text"))
    if answer_basis_hash is None and answer_basis is not None:
        answer_basis_hash = _text_details(answer_basis)["sha256"]

    for index, span in enumerate(answer_spans):
        native_ref = _native_influence_ref(
            schema, "influence.answer_spans", span, f"answer-span-{index}",
        )
        relation_anchor = {
            "token_index": span.get("token_index")
            if isinstance(span.get("token_index"), int) else native_ref["id"],
        }
        span_text = span.get("text")
        span_text = span_text if isinstance(span_text, str) else None
        addresses.append(make_text_span_address(
            run_id=run_id,
            kind="answer_span",
            native_ref=native_ref,
            relation_anchor=relation_anchor,
            basis="scored_answer",
            start=span.get("start"),
            end=span.get("end"),
            privacy=privacy,
            basis_text=answer_basis,
            basis_sha256=answer_basis_hash,
            span_text=span_text,
            span_sha256=_valid_full_hash(span.get("text_sha256")),
            redacted=answer_basis is None and artifact_privacy != "full",
        ))

    return addresses, source_record


def project_influence_addresses(run_id: str, artifact: dict, *,
                                privacy: str = "metadata_only") -> list[dict]:
    """Project native influence source/span IDs without copying its measurements."""
    return _influence_projection(run_id, artifact, privacy=privacy)[0]


def map_inherited_addresses(parent_document: dict, child_document: dict) -> list[dict]:
    """Map parent addresses to a child using text-free native relation anchors.

    An address is inherited only when kind, offsets, canonical basis hash, and
    selected-span hash are all unchanged.  Structural differences never become
    a claim about why either answer changed.
    """
    if parent_document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"parent_document must be {SCHEMA_VERSION}")
    if child_document.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"child_document must be {SCHEMA_VERSION}")
    child_by_relation: dict[str, list[dict]] = {}
    for address in child_document.get("addresses") or []:
        if isinstance(address, dict) and isinstance(address.get("relation_key"), str):
            child_by_relation.setdefault(address["relation_key"], []).append(address)

    mappings = []
    for parent in parent_document.get("addresses") or []:
        if not isinstance(parent, dict) or not isinstance(parent.get("relation_key"), str):
            continue
        relation_key = parent["relation_key"]
        base = {
            "relation_key": relation_key,
            "parent_address_id": parent["address_id"],
        }
        children = child_by_relation.get(relation_key, [])
        if not children:
            mappings.append({**base, "state": "unavailable", "reason": "missing_in_child"})
            continue
        if len(children) != 1:
            mappings.append({**base, "state": "unavailable", "reason": "ambiguous_relation"})
            continue
        child = children[0]
        base["child_address_id"] = child["address_id"]
        parent_resolution = parent.get("resolution") or {}
        child_resolution = child.get("resolution") or {}
        parent_canonical = parent_resolution.get("canonical")
        child_canonical = child_resolution.get("canonical")
        if not isinstance(parent_canonical, dict):
            mappings.append({**base, "state": "unavailable", "reason": "parent_unresolved"})
            continue
        if not isinstance(child_canonical, dict):
            mappings.append({**base, "state": "unavailable", "reason": "child_unresolved"})
            continue
        if parent.get("kind") != child.get("kind"):
            mappings.append({**base, "state": "drifted", "reason": "kind_changed"})
            continue
        if (parent_canonical.get("start"), parent_canonical.get("end")) != (
            child_canonical.get("start"), child_canonical.get("end"),
        ):
            mappings.append({**base, "state": "drifted", "reason": "offset_changed"})
            continue
        if parent_canonical.get("basis_sha256") != child_canonical.get("basis_sha256"):
            mappings.append({
                **base, "state": "drifted", "reason": "canonical_basis_hash_changed",
            })
            continue
        if parent_canonical.get("span_sha256") != child_canonical.get("span_sha256"):
            mappings.append({**base, "state": "drifted", "reason": "span_text_hash_changed"})
            continue
        mappings.append({
            **base, "state": "inherited", "reason": "exact_text_and_hashes_unchanged",
        })
    return mappings


def build_text_span_addresses(
    run: dict,
    *,
    influence: dict | None = None,
    privacy: str = "metadata_only",
    parent_document: dict | None = None,
) -> dict:
    """Build and validate one derived address document.

    Passing ``parent_document`` adds explicit lineage mappings.  The caller
    supplies a loaded influence artifact when it lives in the blob store; this
    helper performs no I/O and never triggers a measurement.
    """
    if privacy not in {"full", "metadata_only"}:
        raise ValueError("privacy must be full or metadata_only")
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")
    context_addresses, context_source = _context_projection(run, privacy=privacy)
    addresses = list(context_addresses)
    source_artifacts = [context_source]

    if influence is None and isinstance(run.get("influence_map"), dict):
        influence = run["influence_map"]
    if influence is not None:
        influence_addresses, influence_source = _influence_projection(
            run_id, influence, privacy=privacy,
        )
        addresses.extend(influence_addresses)
        source_artifacts.append(influence_source)

    parent_run_id = run.get("parent_run_id")
    parent_run_id = parent_run_id if isinstance(parent_run_id, str) and parent_run_id else None
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": privacy,
        "offset_contract": dict(OFFSET_CONTRACT),
        "source_artifacts": source_artifacts,
        "addresses": addresses,
        "lineage": {
            "parent_run_id": parent_run_id,
            "mappings": [],
        },
    }
    if parent_document is not None:
        if parent_run_id is not None and parent_document.get("run_id") != parent_run_id:
            raise ValueError("parent_document.run_id does not match run.parent_run_id")
        document["lineage"]["parent_run_id"] = parent_document.get("run_id")
        document["lineage"]["mappings"] = map_inherited_addresses(parent_document, document)

    # Keep schema drift a construction-time error.  No run is mutated if this
    # derived projection is invalid.
    from clozn import schemas
    schemas.validate(document)
    return document


def build_persisted_text_span_addresses(
    run: dict,
    *,
    privacy: str = "metadata_only",
    parent_document: dict | None = None,
) -> dict:
    """Project only evidence already attached to a stored run.

    ``clozn.runs.store.get_run`` resolves a blob-backed ``influence_map_ref``
    before returning the run.  A missing/corrupt/unreadable blob is represented
    there as ``{"unavailable": ..., "sha256": ...}``; this helper preserves
    that distinction from a run which never recorded influence.  It performs
    no blob I/O itself and never invokes an evidence producer.
    """
    from clozn import schemas

    influence_present = "influence_map" in run
    influence = run.get("influence_map")
    source_status = None

    if not influence_present:
        source_status = {
            "schema": INFLUENCE_SCHEMA,
            "native_status": "not_recorded",
            "available": False,
            "reason": "no persisted influence artifact is attached to this run",
        }
        influence = None
    elif isinstance(influence, dict) and isinstance(influence.get("unavailable"), str):
        source_status = {
            "schema": INFLUENCE_SCHEMA,
            "native_status": "unavailable",
            "available": False,
            "reason": influence["unavailable"],
        }
        digest = _valid_full_hash(influence.get("sha256"))
        if digest:
            source_status["artifact_sha256"] = digest
        influence = None
    elif not isinstance(influence, dict) or not influence:
        source_status = {
            "schema": INFLUENCE_SCHEMA,
            "native_status": "failed",
            "available": False,
            "reason": "persisted influence evidence is not a non-empty object",
        }
        influence = None
    else:
        source_schema = influence.get("schema_version") or influence.get("schema")
        if source_schema not in {INFLUENCE_SCHEMA, INFLUENCE_EXPORT_SCHEMA}:
            source_status = {
                "schema": INFLUENCE_SCHEMA,
                "native_status": "failed",
                "available": False,
                "reason": "persisted influence evidence has an unsupported or missing schema",
            }
            influence = None
        else:
            try:
                schemas.validate(influence, source_schema)
            except schemas.ValidationError:
                source_status = {
                    "schema": str(source_schema),
                    "native_status": "failed",
                    "available": False,
                    "reason": "persisted influence evidence does not satisfy its native schema",
                }
                influence = None

    base_run = {key: value for key, value in run.items() if key != "influence_map"}
    try:
        document = build_text_span_addresses(
            base_run,
            influence=influence,
            privacy=privacy,
            parent_document=parent_document,
        )
    except (TypeError, ValueError, UnicodeError, schemas.ValidationError):
        # A malformed source must not hide valid context addresses.  Do not put
        # exception values in the reason: old artifacts may contain private
        # strings in unexpected fields.
        document = build_text_span_addresses(
            base_run,
            privacy=privacy,
            parent_document=parent_document,
        )
        source_status = {
            "schema": INFLUENCE_SCHEMA,
            "native_status": "failed",
            "available": False,
            "reason": "persisted influence evidence could not be projected safely",
        }

    if source_status is not None:
        document["source_artifacts"].append(source_status)
        schemas.validate(document)
    return document


__all__ = [
    "BASES",
    "KINDS",
    "OFFSET_CONTRACT",
    "SCHEMA_VERSION",
    "build_persisted_text_span_addresses",
    "build_text_span_addresses",
    "make_text_span_address",
    "map_inherited_addresses",
    "project_context_addresses",
    "project_influence_addresses",
]
