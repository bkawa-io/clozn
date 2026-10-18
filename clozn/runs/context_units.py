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

import re
from copy import deepcopy
from typing import Any, Mapping

SCHEMA = "clozn.context-units.v1"

_LONG_MESSAGE_CHARS = 600
_MIN_SPAN_CHARS = 200
_HEADER_RE = re.compile(r"^#{1,6} ", re.MULTILINE)
_HR_RE = re.compile(r"^(?:---|\*\*\*)[ \t]*$", re.MULTILINE)
_DOC_MARKER_RE = re.compile(
    r"^[ \t]*(?:Document[ \t]+\d+\b|\[\d+\])",
    re.MULTILINE | re.IGNORECASE,
)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_XML_WRAP_RE = re.compile(r"<([a-zA-Z_][\w:-]*)>.*?</\1>", re.DOTALL)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")
_WS = " \t\r\n"


def _text_of(message: Any) -> str:
    return message.get("content") if isinstance(message, dict) and isinstance(message.get("content"), str) else ""


def protected_message_indices(messages: list[dict]) -> set[int]:
    """Return the conservatively protected messages in a chat-shaped list.

    System/developer instructions are always protected. The final user message
    is the current question; it and everything after it are protected,
    including a trailing assistant prefill. When no user message exists, the
    final message is protected as well.
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
    instructions = {
        index for index, message in enumerate(messages)
        if isinstance(message, dict) and message.get("role") in {"system", "developer"}
    }
    if last_user is None:
        return instructions | {len(messages) - 1}
    return instructions | set(range(last_user, len(messages)))


def _has_explicit_sources(message: Any) -> bool:
    """Whether a message carries caller source metadata in the journal shape."""
    raw = message.get("_clozn_sources") if isinstance(message, dict) else None
    return isinstance(raw, list) and bool(raw)


def _protected_spans(text: str) -> list[tuple[int, int]]:
    spans = sorted(
        [(match.start(), match.end()) for match in _FENCE_RE.finditer(text)]
        + [(match.start(), match.end()) for match in _XML_WRAP_RE.finditer(text)]
    )
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _inside(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start] in _WS:
        start += 1
    while end > start and text[end - 1] in _WS:
        end -= 1
    return start, end


def _paragraph_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    position = start
    for match in _BLANK_LINE_RE.finditer(text, start, end):
        if match.start() > position:
            spans.append((position, match.start()))
        position = match.end()
    if position < end:
        spans.append((position, end))
    return spans or [(start, end)]


def _merge_small_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not spans:
        return []
    merged = [spans[0]]
    for start, end in spans[1:]:
        if end - start < _MIN_SPAN_CHARS:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    while len(merged) > 1 and merged[0][1] - merged[0][0] < _MIN_SPAN_CHARS:
        head = merged.pop(0)
        merged[0] = (head[0], merged[0][1])
    return merged


def _structural_spans(text: str) -> list[tuple[int, int]]:
    """Return deterministic semantic spans without minting another identity namespace."""
    if not text:
        return [(0, 0)]
    protected = _protected_spans(text)
    points = {
        match.start()
        for pattern in (_HEADER_RE, _HR_RE, _DOC_MARKER_RE)
        for match in pattern.finditer(text)
        if not _inside(match.start(), protected)
    }
    for start, end in protected:
        points.update((start, end))
    points.update((0, len(text)))

    spans: list[tuple[int, int]] = []
    for start, end in zip(sorted(points), sorted(points)[1:]):
        if end <= start:
            continue
        if (start, end) in protected:
            spans.append((start, end))
            continue
        trimmed_start, trimmed_end = _trim(text, start, end)
        if trimmed_end <= trimmed_start:
            continue
        if trimmed_end - trimmed_start > _LONG_MESSAGE_CHARS:
            fragments = [
                _trim(text, fragment_start, fragment_end)
                for fragment_start, fragment_end in _paragraph_spans(text, trimmed_start, trimmed_end)
            ]
            spans.extend(_merge_small_spans([
                span for span in fragments if span[1] > span[0]
            ]))
        else:
            spans.append((trimmed_start, trimmed_end))
    return spans or [(0, len(text))]


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
    semantic = _structural_spans(text)
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

    This includes previous user messages, has no global span cap, and returns
    canonical source descriptors rather than a second section namespace.
    Returned ranges are already contiguous partitions for messages that
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


def _covers_explicit_content(
    sources: list[Mapping[str, Any]], text: str, *, start: int = 0, end: int | None = None,
) -> bool:
    """Whether exact caller spans cover all non-whitespace content without overlap.

    Caller source annotations commonly omit separators between text blocks
    (for example the newline between two retrieved paragraphs).  Those bytes
    are fixed formatting, not a separately removable source.  Requiring the
    canonical source ranges to absorb them would force Minimal Context to
    replace the caller's identities with a whole-message root.
    """
    limit = len(text) if end is None else end
    if not (0 <= start <= limit <= len(text)):
        return False
    ordered = sorted(sources, key=lambda source: (_range(source) or (limit + 1, limit + 1),
                                                  str(source.get("source_id"))))
    cursor = start
    for source in ordered:
        span = _range(source)
        if span is None:
            return False
        source_start, source_end = span
        if source_start < cursor or source_end <= source_start or source_end > limit:
            return False
        if text[cursor:source_start].strip():
            return False
        cursor = source_end
    return not text[cursor:limit].strip()


def _explicit_partition(message: dict, spans: list[dict], text: str) -> list[dict] | None:
    """Choose an explicitly represented semantic partition, if one is provable."""
    raw = message.get("_clozn_sources")
    if not isinstance(raw, list) or not raw:
        return None
    exact_raw = [item for item in raw if isinstance(item, dict) and "unicode_range" in item]
    if len(exact_raw) != len(raw):
        return None

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
    if not _covers_explicit_content(top, text):
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
        if not children or not _covers_explicit_content(
            children, text, start=parent_start, end=parent_end,
        ):
            return [source]
        return [item for child in sorted(children, key=lambda item: _range(item) or (0, 0))
                for item in refine(child)]

    refined = [item for source in sorted(top, key=lambda item: _range(item) or (0, 0))
               for item in refine(source)]
    return refined if _covers_explicit_content(refined, text) else None


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
            explicit = _explicit_partition(message, spans, content)
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
