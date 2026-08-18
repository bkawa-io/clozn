"""span_bridge.py -- the arbitrary-span bridge (C3 core slice, item 3).

B3 span addressing (`clozn.runs.text_span_addresses`) already gives every passage in a run a stable,
content-addressed identity: an `address_id`, a `(message_index-shaped) native_ref`, and a
`resolution.canonical` block carrying Unicode-code-point offsets plus SHA-256 hashes of both the
canonical basis text and the selected span. What it does NOT do is turn that address back into
something `clozn.replay.replay.replay()` can execute -- `replay()`'s only content-level ablation lever is
`changes["exclude_sections"]`, a list of NAMED sections against the run's persisted `sections` manifest.
An arbitrary address is neither named nor guaranteed to align with a section boundary.

This module is the bridge. `resolve_span_address()` re-derives a FRESH `clozn.text-span-addresses.v1`
document from the run's CURRENT stored content (never a cached/stale one) and looks the requested
`address_id` up in it. Because `address_id` is itself a hash over `{relation_key, native_ref,
canonical.basis_sha256, canonical.span_sha256, canonical.start, canonical.end}` (see
`text_span_addresses._address_id`), a content change since the id was minted changes the id too -- so a
stale id simply is not found in the fresh rebuild, and this function refuses
(`span_address_not_found_or_drifted`) rather than resolving to the WRONG text at the same coordinates.
This is deliberately one merged reason, not two: telling "this id never existed" apart from "the content
at this same logical spot moved" would need the relation_key reversed, which is a hash and not
reversible from the id alone. Both cases are equally disqualifying, so a caller a re-run
`clozn text-span-addresses RUN_ID` (or the investigation surface) gets a fresh id either way.

Spans anchored to an ordinary chat message (`native_ref.collection` in `{"run.messages",
"context_receipt.delivered"}`) resolve to a `message_index` this module can hand to `excise_spans()`,
via either of the manifest's two native-id shapes: the positional `f"message-{index}"` fallback (parsed
directly out of the id, no lookup needed -- pre-schema/legacy runs, which never stamp a `segment_id`), or
an opaque `segment_id` -- looked up against a `segment_id -> (message_index, content_hash)` index this
module builds by READING the run's OWN context receipt (`_segment_id_index()`, via
`clozn.runs.context_receipt.read_receipt`; the index reuses the receipt's own `original_order` field,
which is a message-list position recorded once at delivery time -- see that module's "segment identity"
section). A `segment_id` lookup is a lookup, not a proof: before trusting it, `_resolved_span_from_address()`
re-hashes the message CURRENTLY sitting at the resolved index (`clozn.runs.context_receipt._content_hash`,
reused rather than reimplemented -- two content-hash implementations that could drift apart would defeat
the purpose) and compares it against that segment's own recorded `content_hash`, refusing rather than
silently mis-mapping when the two disagree, or when the receipt records no usable hash to check against in
the first place. This makes the segment_id path self-verifying: a stale or corrupted `original_order`
almost always points at content with a different hash, and gets caught here rather than producing a
confident, wrong ablation. Two kinds are refused by construction, honestly, never guessed, regardless of
native-id shape:
  * `rendered_prompt_segment` (basis `rendered_prompt`, i.e. `final_prompt`) -- `replay()`'s only surface
    is a MESSAGE list (`chat(messages, ...)`), never a raw-prompt override; raw-prompt reconstruction
    belongs to the Branch Fan execution seam. Splicing here would require inventing a capability replay does
    not have.
  * `answer_span` (basis `scored_answer`) -- a piece of the model's OWN reply, not prompt content. There
    is nothing to ablate out of a PROMPT here; that is a different kind of question this slice does not
    answer.

`excise_spans()` is the OTHER half: given one or more resolved spans, build a NEW messages list with each
span's [start, end) removed or replaced, right-to-left within a message so an earlier span's offsets in
the SAME message are never invalidated by a later span's removal -- exactly `replay.py`'s own
`_strip_message_parts` algorithm, generalized from a NAMED section's `parts` to an arbitrary offset list
(this module does not import that private helper; the two are independent, intentionally small,
implementations of the identical splice rule, kept apart because one is keyed by section name against a
persisted manifest and the other by raw offsets against a resolved address -- forcing them through one
shared function would need a manifest-shaped wrapper for every arbitrary-span caller for no shared
benefit). The result is an ordinary message list `clozn.replay.replay.replay()` already knows how to
consume via its `messages_override` parameter -- no change to replay.py itself was needed.

`pick_random_control_span()` builds the ONE piece of new machinery C3's five-arm-style discipline needs
that `clozn.receipts.forced`'s existing `_matched_length_filler` does not cover: a same-LENGTH,
NON-OVERLAPPING window elsewhere in the SAME basis text, chosen deterministically (SHA-256 over canonical
JSON, never Python's process-randomized `hash()` -- mirrors `clozn.analysis.causal_bisect.
_derive_single_site_seed`'s own reasoning) so the "did ANY similarly-sized change do this" control is
reproducible from the run id and the target span alone. `None` when the basis text is too short to fit a
disjoint window of the same length -- never a guessed overlapping or truncated one.

Stdlib-only, model-free, fully unit-testable against fixture run dicts -- no substrate, no generation.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any, Iterable

from clozn.receipts.forced import (
    MATCHED_LENGTH_NEUTRAL_FILLER_RECIPE,
    matched_length_neutral_filler,
)


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


_ID_MESSAGE_RE_PREFIX = "message-"


class ContextReceiptSourceResolutionError(ValueError):
    """A canonical Context Receipt source set cannot be deleted faithfully.

    This is intentionally separate from the older span/source bridge below.
    ``resolve_source_spans`` is a best-effort bridge for legacy client source
    IDs and may return the subset of occurrences it can splice.  Context
    Dependence source deletion has the opposite contract: it either proves the
    complete canonical receipt-to-message correspondence and deletes whole
    messages, or it refuses before handing anything to replay.
    """


def _canonical_json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()


def _strict_message_list(value: Any, *, name: str) -> list[dict]:
    """A message-list basis we can prove against a Context Receipt.

    A partial message list is not a harmless degradation for a whole-message
    source deletion: it would make the source set look complete while deleting
    it from a different prompt.  Require every entry and both template-visible
    fields to be explicit.
    """
    if not isinstance(value, list) or not value:
        raise ContextReceiptSourceResolutionError(f"{name} must be a non-empty message list")
    copied: list[dict] = []
    for index, message in enumerate(value):
        if not isinstance(message, dict):
            raise ContextReceiptSourceResolutionError(f"{name}[{index}] is not a message object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not role:
            raise ContextReceiptSourceResolutionError(f"{name}[{index}].role is not a non-empty string")
        if not isinstance(content, str):
            raise ContextReceiptSourceResolutionError(f"{name}[{index}].content is not a string")
        copied.append(deepcopy(message))
    return copied


def _validated_receipt_segments(*, segments: Any, messages: list[dict], label: str) -> list[dict]:
    """Validate one receipt segment list against its exact message-list basis.

    ``original_order`` is never a hint here.  It must be the complete identity
    mapping ``0..N-1`` and every position must agree on both the receipt's
    stable segment ID and content hash.  This catches stale order metadata,
    duplicate IDs, malformed receipt rows, and prompt drift before deletion.
    """
    from clozn.runs.context_receipt import _content_hash, segment_id

    if not isinstance(segments, list) or len(segments) != len(messages):
        raise ContextReceiptSourceResolutionError(
            f"Context Receipt {label} segments do not completely match the message basis")
    seen_ids: set[str] = set()
    seen_occurrences: dict[tuple[str, str], int] = {}
    out: list[dict] = []
    for expected_index, (segment, message) in enumerate(zip(segments, messages)):
        if not isinstance(segment, dict):
            raise ContextReceiptSourceResolutionError(
                f"Context Receipt {label}[{expected_index}] is malformed")
        if segment.get("source_type") != "message":
            raise ContextReceiptSourceResolutionError(
                f"Context Receipt {label}[{expected_index}] is not a message segment")
        if segment.get("original_order") != expected_index:
            raise ContextReceiptSourceResolutionError(
                f"Context Receipt {label} original_order is incomplete, stale, or reordered")
        role, content = message["role"], message["content"]
        key = (role, content)
        occurrence = seen_occurrences.get(key, 0)
        seen_occurrences[key] = occurrence + 1
        expected_id = segment_id(role, content, occurrence=occurrence)
        source_id = segment.get("segment_id")
        if not isinstance(source_id, str) or not source_id or source_id != expected_id:
            raise ContextReceiptSourceResolutionError(
                f"Context Receipt {label}[{expected_index}] segment_id does not match its message")
        if source_id in seen_ids:
            raise ContextReceiptSourceResolutionError(
                f"Context Receipt {label} has duplicate segment_id {source_id!r}")
        seen_ids.add(source_id)
        if segment.get("content_hash") != _content_hash(content):
            raise ContextReceiptSourceResolutionError(
                f"Context Receipt {label}[{expected_index}] content_hash does not match its message")
        out.append(deepcopy(segment))
    return out


def _canonical_removed_ids(source_ids: Iterable[str]) -> list[str]:
    if isinstance(source_ids, (str, bytes)):
        raise ContextReceiptSourceResolutionError(
            "removed_source_ids must be an iterable of canonical Context Receipt segment IDs")
    try:
        result = list(source_ids)
    except TypeError as exc:
        raise ContextReceiptSourceResolutionError(
            "removed_source_ids must be an iterable of canonical Context Receipt segment IDs") from exc
    if not result or any(not isinstance(value, str) or not value for value in result):
        raise ContextReceiptSourceResolutionError(
            "removed_source_ids must contain at least one non-empty canonical segment ID")
    if len(set(result)) != len(result):
        raise ContextReceiptSourceResolutionError("removed_source_ids must not contain duplicate segment IDs")
    return sorted(result)


def _strict_range(value: Any, *, upper: int, name: str) -> list[int]:
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(item, int) and not isinstance(item, bool) for item in value)):
        raise ContextReceiptSourceResolutionError(f"{name} must be a two-integer range")
    start, end = int(value[0]), int(value[1])
    if start < 0 or end <= start or end > upper:
        raise ContextReceiptSourceResolutionError(f"{name} is empty or outside the current message")
    return [start, end]


def _receipt_source_catalog(segments: list[dict], messages: list[dict], *, label: str) -> list[dict]:
    """Strictly prove every canonical root/span source against current bytes."""
    from clozn.runs.context_receipt import _canonical_source_id, _content_sha256

    catalog: list[dict] = []
    for message_index, (segment, message) in enumerate(zip(segments, messages)):
        content = message["content"]
        segment_id = str(segment["segment_id"])
        root = {
            "source_id": segment_id, "segment_id": segment_id, "message_index": message_index,
            "unicode_range": [0, len(content)], "byte_range": [0, len(content.encode("utf-8"))],
            "content_sha256": _content_sha256(content), "role": message["role"],
            "source_kind": "whole_message", "provenance_kind": "message",
        }
        for key in ("client_source_id", "source_label"):
            if isinstance(segment.get(key), str) and segment[key]:
                root[key] = segment[key]
        catalog.append(root)
        raw_spans = segment.get("sources")
        if raw_spans is None:
            continue
        if not isinstance(raw_spans, list):
            raise ContextReceiptSourceResolutionError(f"Context Receipt {label}.sources is malformed")
        spans: dict[str, dict] = {}
        for raw in raw_spans:
            if not isinstance(raw, dict) or not isinstance(raw.get("source_id"), str):
                raise ContextReceiptSourceResolutionError("Context Receipt span source is malformed")
            source_id = raw["source_id"]
            if not source_id.startswith("src_") or source_id in spans:
                raise ContextReceiptSourceResolutionError("Context Receipt span source IDs are malformed or duplicate")
            if raw.get("segment_id") != segment_id or raw.get("message_index") != message_index:
                raise ContextReceiptSourceResolutionError("Context Receipt span source is partially mapped")
            unicode_range = _strict_range(raw.get("unicode_range"), upper=len(content), name="span unicode_range")
            byte_range = [len(content[:unicode_range[0]].encode("utf-8")),
                          len(content[:unicode_range[1]].encode("utf-8"))]
            if _strict_range(raw.get("byte_range"), upper=len(content.encode("utf-8")), name="span byte_range") != byte_range:
                raise ContextReceiptSourceResolutionError("Context Receipt span byte range does not match UTF-8")
            selected = content[unicode_range[0]:unicode_range[1]]
            content_sha256 = _content_sha256(selected)
            if raw.get("content_sha256") != content_sha256:
                raise ContextReceiptSourceResolutionError("Context Receipt span does not match current message bytes")
            provenance_kind = raw.get("provenance_kind")
            if not isinstance(provenance_kind, str) or not provenance_kind:
                raise ContextReceiptSourceResolutionError("Context Receipt span provenance_kind is malformed")
            item = {
                "source_id": source_id, "segment_id": segment_id, "message_index": message_index,
                "unicode_range": unicode_range, "byte_range": byte_range,
                "content_sha256": content_sha256, "role": message["role"],
                "source_kind": "source_span", "provenance_kind": provenance_kind,
            }
            for key in ("client_source_id", "source_label", "parent_source_id"):
                if raw.get(key) is not None:
                    if not isinstance(raw[key], str) or not raw[key]:
                        raise ContextReceiptSourceResolutionError(f"Context Receipt span {key} is malformed")
                    item[key] = raw[key]
            spans[source_id] = item

        def verify(item: dict, active: set[str]) -> None:
            if item.get("_verified"):
                return
            source_id = item["source_id"]
            if source_id in active:
                raise ContextReceiptSourceResolutionError("Context Receipt span hierarchy contains a cycle")
            active.add(source_id)
            parent_id = item.get("parent_source_id")
            if parent_id is not None:
                parent = spans.get(parent_id)
                if parent is None:
                    raise ContextReceiptSourceResolutionError("Context Receipt structural parent is unavailable")
                verify(parent, active)
                if not (parent["unicode_range"][0] <= item["unicode_range"][0]
                        and item["unicode_range"][1] <= parent["unicode_range"][1]):
                    raise ContextReceiptSourceResolutionError("Context Receipt structural child is outside parent")
            expected_id = _canonical_source_id(
                segment=segment_id, unicode_range=item["unicode_range"], byte_range=item["byte_range"],
                content_sha256=item["content_sha256"], provenance_kind=item["provenance_kind"],
                client_source_id=item.get("client_source_id"), parent_source_id=parent_id,
            )
            if source_id != expected_id:
                raise ContextReceiptSourceResolutionError("Context Receipt source ID does not bind current bytes")
            item["_verified"] = True
            active.remove(source_id)

        for item in spans.values():
            verify(item, set())
        source_values = list(spans.values())
        for item in source_values:
            item.pop("_verified", None)
        def ancestor(ancestor_id: str, child: dict) -> bool:
            parent_id = child.get("parent_source_id")
            while isinstance(parent_id, str):
                if parent_id == ancestor_id:
                    return True
                parent = spans.get(parent_id)
                parent_id = parent.get("parent_source_id") if parent else None
            return False
        for left_index, left in enumerate(source_values):
            for right in source_values[left_index + 1:]:
                ls, le = left["unicode_range"]; rs, re = right["unicode_range"]
                if max(ls, rs) < min(le, re):
                    if not ((ls <= rs and re <= le and ancestor(left["source_id"], right))
                            or (rs <= ls and le <= re and ancestor(right["source_id"], left))):
                        raise ContextReceiptSourceResolutionError(
                            "Context Receipt spans overlap without structural nesting")
        catalog.extend(sorted(source_values, key=lambda x: (x["unicode_range"], x["source_id"])))
    return catalog


def _merged_ranges(ranges: list[list[int]]) -> list[list[int]]:
    out: list[list[int]] = []
    for start, end in sorted(ranges):
        if out and start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return out


def resolve_context_receipt_source_set(run: dict, removed_source_ids: Iterable[str]) -> dict:
    """Strictly resolve canonical Context Receipt roots and exact span sources.

    This is the strict source-removal/regeneration seam.  It only accepts a
    current, new-shape receipt whose delivered list completely verifies the raw
    messages.  When a run records an assembled message basis (the prompt that
    generation actually received), that list and its receipt projection must
    be a trivial, one-to-one, same-order correspondence with delivery.  A
    transformed/inserted/omitted assembly is deliberately refused rather than
    translating IDs through text similarity or stale positions.

    ``seg_`` roots retain their historical whole-message deletion semantics;
    ``src_`` descriptors delete only their proved exact range, right-to-left.
    The return includes a detached render-safe ``messages`` override, exact
    ranges, source catalog, and deterministic digest evidence.  No caller may
    receive a partially mapped deletion.
    """
    if not isinstance(run, dict) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ContextReceiptSourceResolutionError("a stored run with a non-empty id is required")
    from clozn.runs.context_receipt import read_receipt

    receipt_view = read_receipt(run)
    if receipt_view.get("shape") != "new":
        raise ContextReceiptSourceResolutionError("a current schema-backed Context Receipt is required")
    receipt = receipt_view.get("receipt")
    if not isinstance(receipt, dict) or receipt.get("schema_validation_error"):
        raise ContextReceiptSourceResolutionError("the Context Receipt is malformed or failed validation")
    if receipt.get("run_id") != run["id"]:
        raise ContextReceiptSourceResolutionError("the Context Receipt does not belong to this run")

    raw_messages = _strict_message_list(run.get("messages"), name="run.messages")
    delivered = _validated_receipt_segments(
        segments=receipt.get("delivered"), messages=raw_messages, label="delivered")

    assembled_value = run.get("assembled_messages")
    if assembled_value is None:
        if "assembled" in receipt:
            raise ContextReceiptSourceResolutionError(
                "the receipt has an assembled projection but the run has no assembled message basis")
        basis_name, basis_messages, basis_segments = "messages", raw_messages, delivered
    else:
        assembled_messages = _strict_message_list(assembled_value, name="run.assembled_messages")
        assembled = _validated_receipt_segments(
            segments=receipt.get("assembled"), messages=assembled_messages, label="assembled")
        if len(assembled) != len(delivered):  # redundant, but protects a future relaxation above
            raise ContextReceiptSourceResolutionError("assembled Context Receipt mapping is nontrivial")
        for index, (raw, assembled_segment, delivered_segment) in enumerate(
            zip(raw_messages, assembled, delivered)
        ):
            assembled_message = assembled_messages[index]
            if (
                assembled_segment.get("segment_id") != delivered_segment.get("segment_id")
                or assembled_segment.get("content_hash") != delivered_segment.get("content_hash")
                or raw["role"] != assembled_message["role"]
                or raw["content"] != assembled_message["content"]
                or delivered_segment.get("included") is not True
            ):
                raise ContextReceiptSourceResolutionError(
                    "the assembled prompt is not a trivial verified delivery correspondence")
        basis_name, basis_messages, basis_segments = "assembled_messages", assembled_messages, assembled

    catalog = _receipt_source_catalog(
        basis_segments, basis_messages,
        label=("assembled" if basis_name == "assembled_messages" else "delivered"),
    )
    requested = _canonical_removed_ids(removed_source_ids)
    by_id = {item["source_id"]: item for item in catalog}
    unknown = [source_id for source_id in requested if source_id not in by_id]
    if unknown:
        raise ContextReceiptSourceResolutionError(
            "unknown canonical Context Receipt source ID(s): " + ", ".join(unknown))

    exact_removed_ranges = []
    # Source-set identity/provenance is lexically canonical; ranges retain
    # their prompt-basis order so a reviewer can inspect the actual deletion
    # surgery without mentally re-sorting opaque IDs.  This also matches the
    # direct teacher-forced source-removal experiment record.
    for source_id in sorted(requested, key=lambda value: (
        by_id[value]["message_index"], by_id[value]["unicode_range"], value
    )):
        source = by_id[source_id]
        exact_removed_ranges.append({
            "source_id": source_id,
            "message_index": source["message_index"],
            "unicode_range": list(source["unicode_range"]),
            "byte_range": list(source["byte_range"]),
            "content_sha256": source["content_sha256"],
            "source_kind": source["source_kind"],
        })
    selected = [by_id[source_id] for source_id in requested]
    whole_removed = {
        item["message_index"] for item in selected if item["source_kind"] == "whole_message"
    }
    # Receipt/journal metadata must never ride into either the baseline score
    # condition or an intervened replay.  Return the same clean basis callers
    # need for their baseline, so the sole score mutation is exact source text.
    clean_basis_messages = [deepcopy(message) for message in basis_messages]
    for message in clean_basis_messages:
        for key in ("_clozn_sources", "source_id", "source_label"):
            message.pop(key, None)
    remaining = [deepcopy(message) for message in clean_basis_messages]
    ranges_by_message: dict[int, list[list[int]]] = {}
    for source in selected:
        if source["source_kind"] == "whole_message" or source["message_index"] in whole_removed:
            continue
        ranges_by_message.setdefault(source["message_index"], []).append(list(source["unicode_range"]))
    # Delete a merged structural source union right-to-left.  Merging makes a
    # selected parent+child one exact deletion, rather than applying stale
    # child coordinates after the parent splice.
    for message_index, ranges in ranges_by_message.items():
        content = remaining[message_index]["content"]
        for start, end in reversed(_merged_ranges(ranges)):
            content = content[:start] + content[end:]
        remaining[message_index]["content"] = content
    remaining = [message for index, message in enumerate(remaining) if index not in whole_removed]
    # These keys are durable journal/receipt evidence, never chat-template
    # input.  A replay can therefore use the same exact mutation result as a
    # score arm without accidentally rendering its source annotations.
    for message in remaining:
        for key in ("_clozn_sources", "source_id", "source_label"):
            message.pop(key, None)
    basis_digest = _canonical_json_digest({"basis": basis_name, "messages": clean_basis_messages})
    intervened_digest = _canonical_json_digest({"basis": basis_name, "messages": remaining})
    return {
        "basis": basis_name,
        "available_source_ids": [item["source_id"] for item in catalog],
        "sources": deepcopy(catalog),
        "canonical_source_ids": requested,
        "basis_messages": clean_basis_messages,
        "messages": remaining,
        "exact_removed_ranges": exact_removed_ranges,
        "basis_digest": basis_digest,
        "intervened_context_digest": intervened_digest,
    }


def delete_context_receipt_sources(run: dict, removed_source_ids: Iterable[str]) -> dict:
    """Alias naming the operation performed by :func:`resolve_context_receipt_source_set`.

    Keeping this public verb makes callers choose the strict whole-message
    path instead of mistaking the legacy span bridge for canonical source-set
    deletion.
    """
    return resolve_context_receipt_source_set(run, removed_source_ids)


NEUTRALIZATION_OPERATOR = "neutralize_source"
NEUTRALIZATION_STRATEGY = "matched_length_neutral_filler"
NEUTRALIZATION_RECIPE = MATCHED_LENGTH_NEUTRAL_FILLER_RECIPE
NEUTRALIZATION_LENGTH_CONTRACT = "unicode_code_points_exact"
NEUTRALIZATION_UTF8_BYTE_LENGTH_CONTRACT = "not_guaranteed"


def _neutral_fill(unicode_length: int) -> str:
    """The canonical neutral payload for one exact Unicode source range.

    The control preserves every message object, role, message-list position,
    and Unicode-code-point offset by using the same public matched-length
    recipe as the existing Influence controls.  It intentionally does *not*
    claim to preserve UTF-8 byte length: the recipe's ASCII payload can change
    byte count when the source range contains non-ASCII code points.  The
    before/after byte counts are persisted for every exact effective range.
    """
    replacement = matched_length_neutral_filler(unicode_length)
    if len(replacement) != unicode_length:
        raise ContextReceiptSourceResolutionError(
            "matched-length neutral filler did not preserve Unicode code-point length")
    return replacement


def neutralize_context_receipt_sources(run: dict, source_ids: Iterable[str]) -> dict:
    """Strict matched-Unicode-length neutralization control for source spans.

    This is deliberately a sibling of ``delete_context_receipt_sources``:
    deletion remains the canonical source-removal intervention, while this
    function records a separately named robustness control.  It reuses the
    deletion resolver only for the source catalog/current-byte proof and never
    uses its deleted messages.  Thus a parent/child source selection is merged
    into one exact replacement union, while its complete requested-source
    evidence remains available for audit.
    """
    resolved = resolve_context_receipt_source_set(run, source_ids)
    basis_messages = resolved.get("basis_messages")
    catalog = resolved.get("sources")
    requested = resolved.get("canonical_source_ids")
    if not (isinstance(basis_messages, list) and isinstance(catalog, list)
            and isinstance(requested, list)):
        raise ContextReceiptSourceResolutionError(
            "canonical source resolver returned an incomplete neutralization basis")
    by_id = {
        item.get("source_id"): item for item in catalog
        if isinstance(item, dict) and isinstance(item.get("source_id"), str)
    }
    selected: list[dict] = []
    for source_id in requested:
        item = by_id.get(source_id)
        if item is None:
            raise ContextReceiptSourceResolutionError(
                "canonical source resolver omitted a requested neutralization source")
        offsets = item.get("unicode_range")
        if not isinstance(offsets, list) or len(offsets) != 2 or offsets[0] == offsets[1]:
            # An empty whole-message root is meaningful for delete (role and
            # template structure still change), but it has no source text for
            # a neutralization control.  Refuse, never label a no-op as one.
            raise ContextReceiptSourceResolutionError(
                "matched-length neutralization requires non-empty source content")
        selected.append(item)

    messages = deepcopy(basis_messages)
    requested_ranges: list[dict] = []
    ranges_by_message: dict[int, list[list[int]]] = {}
    for source in sorted(selected, key=lambda item: (
        item["message_index"], item["unicode_range"], item["source_id"]
    )):
        start, end = source["unicode_range"]
        replacement = _neutral_fill(end - start)
        requested_ranges.append({
            "source_id": source["source_id"],
            "message_index": source["message_index"],
            "unicode_range": [start, end],
            "byte_range": list(source["byte_range"]),
            "content_sha256": source["content_sha256"],
            "source_kind": source["source_kind"],
            "replacement_content_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
            "original_unicode_code_points": end - start,
            "replacement_unicode_code_points": len(replacement),
            "original_utf8_bytes": source["byte_range"][1] - source["byte_range"][0],
            "replacement_utf8_bytes": len(replacement.encode("utf-8")),
        })
        ranges_by_message.setdefault(source["message_index"], []).append([start, end])

    effective_ranges: list[dict] = []
    for message_index, ranges in ranges_by_message.items():
        original = messages[message_index]["content"]
        for start, end in reversed(_merged_ranges(ranges)):
            selected_text = original[start:end]
            replacement = _neutral_fill(end - start)
            messages[message_index]["content"] = (
                messages[message_index]["content"][:start]
                + replacement
                + messages[message_index]["content"][end:]
            )
            effective_ranges.append({
                "message_index": message_index,
                "unicode_range": [start, end],
                "original_content_sha256": hashlib.sha256(selected_text.encode("utf-8")).hexdigest(),
                "replacement_content_sha256": hashlib.sha256(replacement.encode("utf-8")).hexdigest(),
                "original_unicode_code_points": end - start,
                "replacement_unicode_code_points": len(replacement),
                "original_utf8_bytes": len(selected_text.encode("utf-8")),
                "replacement_utf8_bytes": len(replacement.encode("utf-8")),
            })
        if len(messages[message_index]["content"]) != len(original):
            raise ContextReceiptSourceResolutionError(
                "neutralization failed to preserve its Unicode length contract")

    effective_ranges.sort(key=lambda item: (item["message_index"], item["unicode_range"]))
    intervened_digest = _canonical_json_digest({"basis": resolved["basis"], "messages": messages})
    return {
        "basis": resolved["basis"],
        "available_source_ids": deepcopy(resolved["available_source_ids"]),
        "sources": deepcopy(catalog),
        "canonical_source_ids": list(requested),
        "basis_messages": deepcopy(basis_messages),
        "messages": messages,
        "exact_neutralized_ranges": requested_ranges,
        "effective_neutralized_ranges": effective_ranges,
        "neutralization": {
            "operator": NEUTRALIZATION_OPERATOR,
            "strategy": NEUTRALIZATION_STRATEGY,
            "recipe": NEUTRALIZATION_RECIPE,
            "length_contract": NEUTRALIZATION_LENGTH_CONTRACT,
            "utf8_byte_length_contract": NEUTRALIZATION_UTF8_BYTE_LENGTH_CONTRACT,
            "message_structure": "preserved",
        },
        "basis_digest": resolved["basis_digest"],
        "intervened_context_digest": intervened_digest,
    }


def _segment_id_index(run: dict) -> dict:
    """`segment_id -> (message_index, content_hash)` for every delivered segment on a NEW-shape context
    receipt (`clozn.runs.context_receipt.read_receipt` shape == "new"; legacy/absent/unrecognized shapes
    return `{}` -- those never stamp a `segment_id` in the first place, see
    `clozn.runs.text_span_addresses._context_projection`).

    `message_index` reuses the receipt's own `original_order` (falling back to the segment's position
    within `delivered` when `original_order` is missing/invalid), MIRRORING
    `clozn.runs.text_span_addresses._context_projection`'s own identical fallback -- so this index agrees
    with the message_index already baked into each address's canonical offsets, rather than computing a
    second, independently-derived mapping that could silently drift from it.

    `content_hash` is the segment's own recorded hash (`clozn.runs.context_receipt._content_hash`) -- kept
    alongside so `_resolved_span_from_address` can verify the message actually sitting at that index still
    matches before trusting this lookup. A lookup alone is not a proof (see this module's docstring)."""
    from clozn.runs.context_receipt import read_receipt
    view = read_receipt(run)
    if view["shape"] != "new":
        return {}
    delivered = view["receipt"].get("delivered")
    if not isinstance(delivered, list):
        return {}
    index: dict = {}
    for fallback_index, segment in enumerate(delivered):
        if not isinstance(segment, dict):
            continue
        segment_id = segment.get("segment_id")
        if not isinstance(segment_id, str) or not segment_id:
            continue
        message_index = segment.get("original_order")
        if not isinstance(message_index, int) or isinstance(message_index, bool) or message_index < 0:
            message_index = fallback_index
        content_hash = segment.get("content_hash")
        index[segment_id] = (
            message_index,
            content_hash if isinstance(content_hash, str) and content_hash else None,
        )
    return index


def _parse_message_index(native_ref: dict, segment_index: dict) -> "int | None":
    """`native_ref["id"]` -> a message-list index. Two native-id shapes: the manifest's own positional
    fallback (`f"message-{index}"`, parsed directly, never present alongside a `segment_id`), or an opaque
    `segment_id`, looked up in `segment_index` (built by `_segment_id_index()` from the run's OWN context
    receipt -- see that function's and this module's docstrings). None when neither resolves, including a
    malformed "message-" id or a `segment_id` this run's receipt does not (or no longer) recognize --
    never guessed. Note this only resolves the INDEX; `_resolved_span_from_address` still verifies the
    segment_id path's content_hash before trusting it -- this function alone is not the safety property."""
    segment_id = native_ref.get("segment_id")
    if isinstance(segment_id, str) and segment_id:
        entry = segment_index.get(segment_id)
        return entry[0] if entry is not None else None
    raw_id = native_ref.get("id")
    if not isinstance(raw_id, str) or not raw_id.startswith(_ID_MESSAGE_RE_PREFIX):
        return None
    tail = raw_id[len(_ID_MESSAGE_RE_PREFIX):]
    if not tail.isdigit():
        return None
    return int(tail)


def _resolved_span_from_address(run: dict, address: dict, segment_index: dict) -> dict:
    """One already-matched, already-content-fresh address -> `{"ok": True, "span": ResolvedSpan}` or
    `{"ok": False, "reason": Reason}`. Never raises; every failure mode is a typed reason. `segment_index`
    is `_segment_id_index(run)`, built once by the caller and threaded through rather than rebuilt per
    address."""
    kind = address.get("kind")
    if kind not in ("delivered_message", "attached_source_span"):
        return {"ok": False, "reason": _reason(
            "span_basis_unsupported",
            f"span kind {kind!r} has no prompt-message basis this bridge can splice through "
            "replay()'s message-list surface (rendered_prompt_segment/answer_span/claim are out of "
            "scope for a prompt-span ablation in this slice)")}

    native_ref = address.get("native_ref") if isinstance(address.get("native_ref"), dict) else {}
    collection = native_ref.get("collection")
    message_index = None
    if collection in ("influence.prompt_spans", "influence.prompt_sources"):
        # Influence Query exposes projected IDs for measured fine/coarse source spans.  Those
        # addresses are authored against the source text, while replay needs the corresponding
        # message-list slot.  Resolve that slot only when the explicit source identity maps to one
        # message whose current content hash is the same canonical basis hash; never guess by index.
        source_id = native_ref.get("client_source_id")
        if not isinstance(source_id, str) or not source_id:
            influence = run.get("influence_map") if isinstance(run.get("influence_map"), dict) else {}
            source_records = influence.get("prompt_sources") if isinstance(influence, dict) else []
            source_keys = {native_ref.get("id"), native_ref.get("parent_id")}
            for source in source_records if isinstance(source_records, list) else []:
                if not isinstance(source, dict) or source.get("id") not in source_keys:
                    continue
                candidate = source.get("client_source_id")
                if isinstance(candidate, str) and candidate:
                    source_id = candidate
                    break
        messages = run.get("messages") if isinstance(run.get("messages"), list) else []
        matches = [
            index for index, message in enumerate(messages)
            if isinstance(message, dict) and isinstance(source_id, str) and source_id
            and message.get("source_id") == source_id
        ]
        if not matches:
            resolution = address.get("resolution") if isinstance(address.get("resolution"), dict) else {}
            canonical = resolution.get("canonical") if isinstance(resolution.get("canonical"), dict) else {}
            basis_sha256 = canonical.get("basis_sha256")
            if isinstance(basis_sha256, str) and basis_sha256:
                matches = [
                    index for index, message in enumerate(messages)
                    if isinstance(message, dict)
                    and isinstance(message.get("content"), str)
                    and hashlib.sha256(message["content"].encode("utf-8")).hexdigest() == basis_sha256
                ]
        if len(matches) != 1:
            return {"ok": False, "reason": _reason(
                "span_message_index_unresolvable",
                "the measured source span does not map to exactly one recorded message")}
        message_index = matches[0]
    elif collection not in ("run.messages", "context_receipt.delivered"):
        return {"ok": False, "reason": _reason(
            "span_basis_unsupported",
            f"native_ref.collection {collection!r} is not an ordinary chat message")}
    else:
        message_index = _parse_message_index(native_ref, segment_index)
    if message_index is None:
        return {"ok": False, "reason": _reason(
            "span_message_index_unresolvable",
            "this span's native reference uses an opaque segment id (or an unrecognized id shape) that "
            "this slice's bridge cannot map back to a message-list index")}

    resolution = address.get("resolution") if isinstance(address.get("resolution"), dict) else {}
    state = resolution.get("state")
    if state not in ("exact", "metadata_only"):
        return {"ok": False, "reason": _reason(
            "span_unavailable",
            f"span resolution state is {state!r} ({resolution.get('reason') or 'no further reason recorded'})")}

    canonical = resolution.get("canonical") if isinstance(resolution.get("canonical"), dict) else {}
    start, end = canonical.get("start"), canonical.get("end")
    basis_sha256 = canonical.get("basis_sha256")
    if (not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool) or end < start):
        return {"ok": False, "reason": _reason(
            "span_unavailable", "the resolved span carries no usable integer offsets")}

    messages = run.get("messages") if isinstance(run.get("messages"), list) else []
    if not (0 <= message_index < len(messages)) or not isinstance(messages[message_index], dict):
        return {"ok": False, "reason": _reason(
            "span_unavailable", f"message index {message_index} no longer exists on this run")}
    content = messages[message_index].get("content")
    content = content if isinstance(content, str) else ""
    if end > len(content):
        return {"ok": False, "reason": _reason(
            "span_unavailable", "the resolved span's offsets fall outside the current message content")}

    # segment_id self-verification: the segment_id -> message_index mapping above is a LOOKUP against the
    # run's own receipt, not a proof -- see _segment_id_index's docstring. Before trusting it, re-hash the
    # message CURRENTLY at message_index (clozn.runs.context_receipt._content_hash, reused rather than
    # reimplemented) and require it to match the segment's own recorded content_hash. A missing hash to
    # check against is treated the same as "unresolvable" (never silently trusted); an observed mismatch
    # is treated as drift, same vocabulary the full-hash check just below uses for the same failure mode.
    # This is what makes a stale/corrupted original_order fail loud instead of splicing the wrong message.
    segment_id = native_ref.get("segment_id")
    if isinstance(segment_id, str) and segment_id:
        from clozn.runs.context_receipt import _content_hash
        entry = segment_index.get(segment_id)
        recorded_content_hash = entry[1] if entry is not None else None
        if not recorded_content_hash:
            return {"ok": False, "reason": _reason(
                "span_message_index_unresolvable",
                "this segment's receipt entry carries no recorded content_hash this bridge can verify "
                "the segment_id -> message_index lookup against, so it cannot be trusted")}
        if _content_hash(content) != recorded_content_hash:
            return {"ok": False, "reason": _reason(
                "span_address_not_found_or_drifted",
                "the message content now sitting at this segment's recorded message-list position no "
                "longer matches the segment_id's recorded content_hash -- refusing rather than risk "
                "ablating the wrong message")}

    # Belt-and-suspenders re-verification: the freshly-rebuilt document already recomputed this hash from
    # CURRENT content (see module docstring), so this should always agree -- but this module never trusts
    # an upstream computation silently. A mismatch here is a bug elsewhere, and MUST refuse, not proceed.
    actual_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
    if isinstance(basis_sha256, str) and basis_sha256 and basis_sha256 != actual_hash:
        return {"ok": False, "reason": _reason(
            "span_address_not_found_or_drifted",
            "the message content's hash no longer matches this span's recorded canonical basis")}

    span = {
        "message_index": message_index,
        "start": start,
        "end": end,
        "basis_sha256_verified": True,
        "span_address_id": address.get("address_id"),
    }
    return {"ok": True, "span": span}


def _fresh_addresses(run: dict) -> list:
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    document = build_persisted_text_span_addresses(run, privacy="metadata_only")
    addresses = document.get("addresses")
    return addresses if isinstance(addresses, list) else []


def resolve_span_address(run: dict, address_id: str) -> dict:
    """`{"ok": True, "span": ResolvedSpan}` or `{"ok": False, "reason": Reason}` for one `address_id`
    against `run`'s CURRENT content. Never raises: a malformed `run` degrades to a `span_unavailable`
    refusal rather than an exception, mirroring every sibling read-only synthesis module in this
    codebase (`clozn.runs.investigation`, `clozn.receipts.explain`)."""
    try:
        addresses = _fresh_addresses(run)
    except Exception as exc:  # noqa: BLE001 -- a malformed run must refuse, never crash the caller
        return {"ok": False, "reason": _reason(
            "span_unavailable", f"could not rebuild this run's span addresses: {type(exc).__name__}: {exc}")}
    segment_index = _segment_id_index(run)
    for address in addresses:
        if isinstance(address, dict) and address.get("address_id") == address_id:
            return _resolved_span_from_address(run, address, segment_index)
    return {"ok": False, "reason": _reason(
        "span_address_not_found_or_drifted",
        f"no address {address_id!r} exists in this run's current span-address projection -- either it "
        "never existed, or the content it pointed at has changed since the id was minted")}


def resolve_source_spans(run: dict, source_id: str) -> dict:
    """Every resolvable span attached to `source_id` (a `client_source_id`) -> `{"ok": True, "spans":
    [ResolvedSpan, ...]}`, or `{"ok": False, "reason": Reason}` when the source has no addresses at all
    (`source_not_found`) or none of its addresses could be resolved to a message-anchored span
    (`source_has_no_resolvable_spans`, e.g. every occurrence is rendered_prompt/answer_span-anchored, or a
    segment_id occurrence whose content_hash no longer matches -- see `_resolved_span_from_address`). A
    source with SOME resolvable and some unresolvable occurrences still succeeds with the resolvable
    subset -- "omit this source" ablates what this bridge CAN reach, never silently claims to have removed
    what it could not."""
    try:
        addresses = _fresh_addresses(run)
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "reason": _reason(
            "span_unavailable", f"could not rebuild this run's span addresses: {type(exc).__name__}: {exc}")}
    matching = [
        a for a in addresses
        if isinstance(a, dict) and isinstance(a.get("native_ref"), dict)
        and a["native_ref"].get("client_source_id") == source_id
    ]
    if not matching:
        return {"ok": False, "reason": _reason(
            "source_not_found", f"no attached source with client_source_id {source_id!r} was found on this run")}
    segment_index = _segment_id_index(run)
    spans = []
    for address in matching:
        resolved = _resolved_span_from_address(run, address, segment_index)
        if resolved.get("ok"):
            spans.append(resolved["span"])
    if not spans:
        return {"ok": False, "reason": _reason(
            "source_has_no_resolvable_spans",
            f"source {source_id!r} was found but none of its occurrences resolve to a message-anchored "
            "span this bridge can splice (see span_basis_unsupported/span_message_index_unresolvable)")}
    return {"ok": True, "spans": spans}


def excise_spans(messages: list, spans: list, *, replacement=None) -> list:
    """`messages` with every `ResolvedSpan`'s `[start, end)` removed (`replacement=None`) or replaced
    (`replacement(n_chars) -> str`, e.g. `clozn.receipts.forced._matched_length_filler`). Multiple spans in
    the SAME message are spliced right-to-left (highest start first) so an earlier span's offsets stay
    valid; spans in different messages are independent. Never raises: an out-of-range message index, or a
    non-dict message, is left untouched -- same "never take the rest of the list down with one bad entry"
    discipline `replay.py`'s own `_strip_message_parts` uses. `messages` and its dicts are never mutated
    in place; only the touched messages are shallow-copied."""
    by_message: dict = {}
    for span in spans or []:
        if not isinstance(span, dict):
            continue
        idx = span.get("message_index")
        if not isinstance(idx, int) or isinstance(idx, bool):
            continue
        by_message.setdefault(idx, []).append((span.get("start"), span.get("end")))

    out = [dict(m) if isinstance(m, dict) else m for m in (messages or [])]
    for idx, ranges in by_message.items():
        if not (0 <= idx < len(out)) or not isinstance(out[idx], dict):
            continue
        content = out[idx].get("content")
        content = content if isinstance(content, str) else ""
        for start, end in sorted(ranges, key=lambda pair: (pair[0] if isinstance(pair[0], int) else -1),
                                 reverse=True):
            if not isinstance(start, int) or isinstance(start, bool):
                continue
            if not isinstance(end, int) or isinstance(end, bool):
                continue
            start = max(0, min(start, len(content)))
            end = max(0, min(end, len(content)))
            if end < start:
                start, end = end, start
            fill = "" if replacement is None else replacement(end - start)
            content = content[:start] + fill + content[end:]
        out[idx]["content"] = content
    return out


def pick_random_control_span(run: dict, span: dict, *, extra: str = "") -> "dict | None":
    """A same-length window elsewhere in `span["message_index"]`'s current content, disjoint from `span`
    itself, chosen deterministically from a SHA-256 digest over `{run_id, message_index, start, end,
    extra}` (never Python's process-randomized `hash()` -- see module docstring). `None` when the message
    is too short to fit a disjoint same-length window -- never a guessed overlapping or shorter one."""
    idx = span.get("message_index")
    start, end = span.get("start"), span.get("end")
    if not isinstance(idx, int) or isinstance(idx, bool):
        return None
    if (not isinstance(start, int) or isinstance(start, bool)
            or not isinstance(end, int) or isinstance(end, bool) or end < start):
        return None
    messages = run.get("messages") if isinstance(run.get("messages"), list) else []
    if not (0 <= idx < len(messages)) or not isinstance(messages[idx], dict):
        return None
    content = messages[idx].get("content")
    content = content if isinstance(content, str) else ""
    length = end - start
    if length <= 0 or len(content) < length:
        return None

    candidates = [
        i for i in range(0, len(content) - length + 1)
        if i + length <= start or i >= end
    ]
    if not candidates:
        return None

    key = {"run_id": run.get("id"), "message_index": idx, "start": start, "end": end, "extra": extra}
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    pick = int.from_bytes(digest[:4], byteorder="big") % len(candidates)
    chosen_start = candidates[pick]
    return {
        "message_index": idx,
        "start": chosen_start,
        "end": chosen_start + length,
        "basis_sha256_verified": True,
    }


def derive_seed(run: dict, *, purpose: str) -> int:
    """A deterministic uint32 seed from `{run_id, purpose}` -- SHA-256, never `hash()` (module docstring).
    Used to pin `sampler_change` arms to the SAME seed across baseline/no-op/treatment so an observed
    difference is attributable to the requested sampler override, not to seed variance."""
    key = {"run_id": run.get("id"), "purpose": purpose}
    canonical = json.dumps(key, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    digest = hashlib.sha256(canonical).digest()
    return int.from_bytes(digest[:4], byteorder="big")


__all__ = [
    "ContextReceiptSourceResolutionError",
    "delete_context_receipt_sources",
    "derive_seed",
    "excise_spans",
    "NEUTRALIZATION_LENGTH_CONTRACT",
    "NEUTRALIZATION_OPERATOR",
    "NEUTRALIZATION_RECIPE",
    "NEUTRALIZATION_STRATEGY",
    "NEUTRALIZATION_UTF8_BYTE_LENGTH_CONTRACT",
    "neutralize_context_receipt_sources",
    "pick_random_control_span",
    "resolve_context_receipt_source_set",
    "resolve_source_spans",
    "resolve_span_address",
]
