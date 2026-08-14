"""Zero-config canonical context units for recorded chat runs.

Context units are a journal/evidence layer.  They are derived after a request
has crossed the model boundary and their private source annotations are added
only to the copy used to build the Context Receipt.  The messages sent to a
chat template are never modified here.

The unit identity is always the canonical Context Receipt identity: a
``seg_`` whole-message root or a ``src_`` exact span.  This module deliberately
does not mint IDs.  The receipt builder remains the one source-canonicalization
authority, and the strict span bridge remains the one source-to-message
resolver.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping

from . import sections


SCHEMA = "clozn.context-units.v1"


def _text_of(message: Any) -> str:
    return message.get("content") if isinstance(message, dict) and isinstance(message.get("content"), str) else ""


def protected_message_indices(messages: list[dict]) -> set[int]:
    """Return the conservatively protected messages in a chat-shaped list.

    The final user message is the current request.  It and everything after it
    are protected, including a trailing assistant prefill.  When no user
    message exists, only the final message is protected.
    """
    if not isinstance(messages, list) or not messages:
        return set()
    last_user = next(
        (
            index
            for index in range(len(messages) - 1, -1, -1)
            if isinstance(messages[index], dict) and messages[index].get("role") == "user"
        ),
        None,
    )
    if last_user is None:
        return {len(messages) - 1}
    return set(range(last_user, len(messages)))


def _has_explicit_sources(message: Any) -> bool:
    """Whether a message carries caller source metadata in the journal shape."""
    raw = message.get("_clozn_sources") if isinstance(message, dict) else None
    if isinstance(raw, list) and bool(raw):
        return True
    # Direct receipt callers from the pre-exact-source API may still carry a
    # message-level source identity without the normalized private list.  Keep
    # that caller-defined root intact instead of adding auto children beside
    # it.
    return isinstance(message, dict) and any(
        isinstance(message.get(key), str) and bool(message.get(key))
        for key in ("source_id", "source_label")
    )


def _partition(text: str, semantic_spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Turn trimmed semantic spans into a contiguous partition of ``text``.

    The structural chunker intentionally trims separator whitespace.  Context
    units cannot do that because deletion must account for every code point.
    Leading whitespace is assigned to the first unit, inter-span separators to
    the following unit, and trailing whitespace to the final unit.  The rule is
    deterministic and yields no gaps or overlaps.
    """
    n = len(text)
    clean: list[tuple[int, int]] = []
    for value in semantic_spans:
        if not (isinstance(value, (list, tuple)) and len(value) == 2):
            continue
        start, end = value
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        start = max(0, min(start, n))
        end = max(0, min(end, n))
        if end > start:
            clean.append((start, end))
    clean.sort()
    if not clean:
        return [(0, n)] if n else []

    # structural_spans() is ordered and non-overlapping today, but keeping
    # this check local makes the partition contract explicit if that helper is
    # ever extended.
    normalized: list[tuple[int, int]] = []
    for start, end in clean:
        if normalized and start < normalized[-1][1]:
            if end <= normalized[-1][1]:
                continue
            start = normalized[-1][1]
        if end > start:
            normalized.append((start, end))
    if len(normalized) <= 1:
        return [(0, n)] if n else []

    out: list[tuple[int, int]] = []
    for index, (_semantic_start, semantic_end) in enumerate(normalized):
        start = 0 if index == 0 else normalized[index - 1][1]
        end = n if index == len(normalized) - 1 else semantic_end
        if end <= start:
            return [(0, n)] if n else []
        out.append((start, end))
    return out


def _message_auto_spans(message_index: int, message: dict) -> list[dict]:
    text = _text_of(message)
    if not text:
        return []
    semantic = sections.structural_spans(text)
    partition = _partition(text, semantic)
    # A one-span result is deliberately retained as a useful diagnostic result
    # but is not registered as src_: the receipt's existing seg_ root is the
    # canonical whole-message variable in that case.
    return [
        {
            "message_index": message_index,
            "unicode_range": [start, end],
            "source_kind": "source_span",
            "derivation": "auto_structural",
        }
        for start, end in partition
        if end > start
    ]


def _message_auto_source_descriptors(message_index: int, message: dict) -> list[dict]:
    """Build a deterministic binary hierarchy over one auto leaf partition.

    The leaf ranges remain the default Context Units.  The additional parent
    ranges are journal-only canonical identities that a bounded search
    planner may select later.  Every parent is made from adjacent children in
    document order; no semantic boundary or message boundary is crossed.
    """
    leaves = _message_auto_spans(message_index, message)
    if len(leaves) <= 1:
        return []

    nodes: list[dict] = []
    current: list[dict] = []
    for index, leaf in enumerate(leaves):
        node = {
            "message_index": message_index,
            "unicode_range": list(leaf["unicode_range"]),
            "source_kind": "source_span",
            "provenance_kind": "message",
            "source_id": f"__clozn_auto_{message_index}_leaf_{index}",
        }
        current.append(node)
        nodes.append(node)

    level = 0
    while len(current) > 1:
        next_level: list[dict] = []
        for pair_index in range(0, len(current), 2):
            left = current[pair_index]
            if pair_index + 1 >= len(current):
                next_level.append(left)
                continue
            right = current[pair_index + 1]
            parent = {
                "message_index": message_index,
                "unicode_range": [left["unicode_range"][0], right["unicode_range"][1]],
                "source_kind": "source_span",
                "provenance_kind": "message",
                "source_id": f"__clozn_auto_{message_index}_node_{level}_{pair_index // 2}",
                "children": [left, right],
            }
            left["parent_source_id"] = parent["source_id"]
            right["parent_source_id"] = parent["source_id"]
            next_level.append(parent)
            nodes.append(parent)
        current = next_level
        level += 1

    # The receipt canonicalizer consumes a flat list and resolves parent
    # client IDs after all descriptors exist.  Parents first makes the stored
    # catalog pleasant to inspect while the identity itself is order-free.
    return sorted(nodes, key=lambda item: (
        -(item["unicode_range"][1] - item["unicode_range"][0]),
        item["unicode_range"][0],
        item["source_id"],
    ))


def derive_auto_source_spans(messages: list[dict]) -> list[dict]:
    """Derive uncapped structural spans for eligible chat messages.

    Unlike :func:`clozn.runs.sections.auto_chunk_messages`, this includes
    previous user messages, has no global section cap, and returns no section
    IDs.  Returned ranges are already contiguous partitions for messages that
    have more than one effective structural span; a one-span message is
    returned as its whole-message span so callers can intentionally keep the
    existing ``seg_`` root instead of minting a duplicate ``src_``.
    """
    if not isinstance(messages, list) or not messages:
        return []
    protected = protected_message_indices(messages)
    out: list[dict] = []
    for index, message in enumerate(messages):
        if index in protected or not isinstance(message, dict) or _has_explicit_sources(message):
            continue
        out.extend(_message_auto_spans(index, message))
    return out


def annotate_auto_sources(messages: list[dict]) -> list[dict]:
    """Return a journal-only copy carrying auto source descriptors.

    Only genuinely subdivided eligible messages receive metadata.  Existing
    caller metadata is copied byte-for-byte and never merged or relabeled.
    The descriptors contain ranges but no source IDs; Context Receipt capture
    canonicalizes them through its existing source machinery.
    """
    if not isinstance(messages, list):
        return []
    out = deepcopy(messages)
    spans = derive_auto_source_spans(out)
    by_message: dict[int, list[dict]] = {}
    for span in spans:
        index = span["message_index"]
        by_message.setdefault(index, []).append(span)
    for index, message_spans in by_message.items():
        if len(message_spans) <= 1:
            continue
        if not (0 <= index < len(out)) or not isinstance(out[index], dict):
            continue
        # The input has already been checked by derive_auto_source_spans(),
        # but check again at the mutation boundary so this remains safe for a
        # future caller that supplies a custom list-like object.
        if _has_explicit_sources(out[index]):
            continue
        out[index]["_clozn_sources"] = _message_auto_source_descriptors(index, out[index])
    return out


def _source_rows_by_message(receipt: Mapping[str, Any]) -> dict[int, dict[str, list[dict]]]:
    rows_by_message: dict[int, dict[str, list[dict]]] = {}
    delivered = receipt.get("delivered")
    if not isinstance(delivered, list):
        return rows_by_message
    for index, row in enumerate(delivered):
        if not isinstance(row, dict):
            continue
        root_id = row.get("segment_id")
        roots: list[dict] = []
        if isinstance(root_id, str) and root_id:
            roots.append({
                "source_id": root_id,
                "segment_id": root_id,
                "message_index": index,
                "unicode_range": None,
                "byte_range": None,
                "source_kind": "whole_message",
                "provenance_kind": "message",
            })
        for source in row.get("sources") if isinstance(row.get("sources"), list) else []:
            if isinstance(source, dict) and isinstance(source.get("source_id"), str):
                roots.append({**source, "source_kind": "source_span"})
        rows_by_message[index] = {
            "root": roots[:1],
            "spans": roots[1:],
        }
    return rows_by_message


def _copy_source(source: Mapping[str, Any], *, message_index: int, role: str,
                 derivation: str) -> dict:
    item = {
        "source_id": source["source_id"],
        "message_index": message_index,
        "role": role,
        "unicode_range": list(source["unicode_range"]),
        "byte_range": list(source["byte_range"]),
        "source_kind": source["source_kind"],
        "derivation": derivation,
    }
    for key in ("segment_id", "content_sha256", "provenance_kind", "client_source_id",
                "source_label", "parent_source_id"):
        if source.get(key) is not None:
            item[key] = source[key]
    return item


def _range(source: Mapping[str, Any]) -> tuple[int, int] | None:
    value = source.get("unicode_range")
    if not (isinstance(value, (list, tuple)) and len(value) == 2
            and all(isinstance(part, int) and not isinstance(part, bool) for part in value)):
        return None
    return int(value[0]), int(value[1])


def _covers_partition(sources: list[Mapping[str, Any]], length: int, *, start: int = 0) -> bool:
    ordered = sorted(sources, key=lambda source: (_range(source) or (length + 1, length + 1),
                                                  str(source.get("source_id"))))
    cursor = start
    for source in ordered:
        span = _range(source)
        if span is None or span[0] != cursor or span[1] <= span[0] or span[1] > length:
            return False
        cursor = span[1]
    return cursor == length


def _explicit_partition(message: dict, spans: list[dict], length: int) -> list[dict] | None:
    """Choose an explicitly represented complete partition, if one is provable."""
    raw = message.get("_clozn_sources")
    if not isinstance(raw, list) or not raw:
        return None
    exact_raw = [item for item in raw if isinstance(item, dict) and "unicode_range" in item]
    if not exact_raw:
        # Legacy clozn_sources names the existing whole-message root.  It has
        # no exact src_ identity to preserve and is safely represented by seg_.
        return []

    matched: list[dict] = []
    for raw_item in exact_raw:
        raw_range = raw_item.get("unicode_range")
        client_id = raw_item.get("source_id", raw_item.get("client_source_id"))
        candidates = [
            source for source in spans
            if source.get("unicode_range") == raw_range
            and (client_id is None or source.get("client_source_id") == client_id)
        ]
        if len(candidates) != 1:
            return None
        matched.append(candidates[0])
    by_id = {source.get("source_id"): source for source in matched}
    if len(by_id) != len(matched):
        return None

    top = [
        source for source in matched
        if source.get("parent_source_id") not in by_id
    ]
    if not _covers_partition(top, length):
        return None

    def refine(source: dict) -> list[dict]:
        children = [
            child for child in matched
            if child.get("parent_source_id") == source.get("source_id")
        ]
        source_range = _range(source)
        if source_range is None:
            return [source]
        parent_start, parent_end = source_range
        if not children or not _covers_partition(children, parent_end, start=parent_start):
            return [source]
        return [item for child in sorted(children, key=lambda item: _range(item) or (0, 0))
                for item in refine(child)]

    refined = [item for source in sorted(top, key=lambda item: _range(item) or (0, 0))
               for item in refine(source)]
    return refined if _covers_partition(refined, length) else None


def _failed_manifest(run_id: str, protected: list[int], message: str) -> dict:
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "basis": "messages",
        "protected_message_indices": protected,
        "units": [],
        "default_source_ids": [],
        "error": message,
    }


def build_context_unit_manifest(run: dict) -> dict:
    """Build a finalized canonical context-unit manifest from a recorded run.

    The receipt is authoritative.  The builder first asks the strict Context
    Receipt resolver to prove the catalog against the run's current message
    bytes, then chooses only IDs returned by that catalog.  Any reconciliation
    failure returns an empty default universe with an explanatory error rather
    than guessing coordinates or inventing source IDs.
    """
    run_id = run.get("id") if isinstance(run, dict) and isinstance(run.get("id"), str) else ""
    messages = run.get("messages") if isinstance(run, dict) else None
    if not isinstance(messages, list):
        return _failed_manifest(run_id, [], "run.messages is not a message list")
    protected = sorted(protected_message_indices(messages))
    base = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "basis": "messages",
        "protected_message_indices": protected,
        "units": [],
        "default_source_ids": [],
    }
    receipt = run.get("context_receipt") if isinstance(run, dict) else None
    if not isinstance(receipt, dict):
        return _failed_manifest(run_id, protected, "run has no finalized Context Receipt")

    rows_by_message = _source_rows_by_message(receipt)
    candidate_ids = [
        source["source_id"]
        for rows in rows_by_message.values()
        for source in rows.get("root", []) + rows.get("spans", [])
        if isinstance(source.get("source_id"), str)
    ]
    candidate_ids = list(dict.fromkeys(candidate_ids))
    if not candidate_ids:
        return _failed_manifest(run_id, protected, "finalized Context Receipt has no source catalog")

    try:
        from clozn.replay.span_bridge import resolve_context_receipt_source_set

        resolved = resolve_context_receipt_source_set(run, candidate_ids)
        catalog = resolved.get("sources")
        if not isinstance(catalog, list):
            raise ValueError("strict resolver returned no source catalog")
        by_message: dict[int, list[dict]] = {}
        for source in catalog:
            if isinstance(source, dict) and isinstance(source.get("message_index"), int):
                by_message.setdefault(source["message_index"], []).append(source)
    except Exception as exc:  # noqa: BLE001 - journal derivation fails closed
        return _failed_manifest(run_id, protected, f"strict receipt reconciliation failed: {exc}")

    auto_spans = derive_auto_source_spans(messages)
    auto_by_message: dict[int, list[dict]] = {}
    for span in auto_spans:
        auto_by_message.setdefault(span["message_index"], []).append(span)

    selected: list[dict] = []
    for index, message in enumerate(messages):
        if index in protected or not isinstance(message, dict):
            continue
        content = _text_of(message)
        sources = by_message.get(index, [])
        root = next((source for source in sources if source.get("source_kind") == "whole_message"), None)
        spans = [source for source in sources if source.get("source_kind") == "source_span"]
        if root is None:
            return _failed_manifest(run_id, protected, f"message {index} has no canonical whole-message root")

        if _has_explicit_sources(message):
            explicit = _explicit_partition(message, spans, len(content))
            chosen = explicit if explicit else [root]
            derivation = "caller_explicit" if explicit else "caller_fallback_root"
        else:
            expected = auto_by_message.get(index, [])
            if len(expected) <= 1:
                chosen = [root]
                derivation = "message_root"
            else:
                expected_ranges = [tuple(item["unicode_range"]) for item in expected]
                matches = [
                    source for expected_range in expected_ranges
                    for source in spans
                    if _range(source) == expected_range
                ]
                if (len(matches) != len(expected_ranges)
                        or len({source.get("source_id") for source in matches}) != len(matches)
                        or not _covers_partition(matches, len(content))):
                    return _failed_manifest(
                        run_id, protected,
                        f"message {index} auto partition does not reconcile with the Context Receipt",
                    )
                chosen = sorted(matches, key=lambda source: _range(source) or (0, 0))
                derivation = "auto_structural"

        for source in chosen:
            selected.append(_copy_source(
                source, message_index=index, role=str(message.get("role") or ""),
                derivation=derivation,
            ))

    default_ids = [source["source_id"] for source in selected]
    if len(default_ids) != len(set(default_ids)):
        return _failed_manifest(run_id, protected, "default source universe contains duplicate canonical IDs")
    base["units"] = selected
    base["default_source_ids"] = default_ids
    return base


__all__ = [
    "SCHEMA",
    "annotate_auto_sources",
    "build_context_unit_manifest",
    "derive_auto_source_spans",
    "protected_message_indices",
]
