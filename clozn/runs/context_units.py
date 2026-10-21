"""Deterministic Markdown-aware context units for recorded chat runs.

Context units are derived from the exact message text after the request has
crossed the model boundary. They never alter the prompt. A single flat,
bounded partition is shared by Context Effects and Minimal Context; callers do
not provide source ranges, labels, or hierarchy.
"""
from __future__ import annotations

import math
import os
import re
from copy import deepcopy
from typing import Any, Mapping


SCHEMA = "clozn.context-units.v1"
MAX_CONTEXT_ARMS_ENV = "CLOZN_MAX_CONTEXT_ARMS"
DEFAULT_MAX_CONTEXT_ARMS = 16
MAX_CONTEXT_ARMS_CEILING = 64

_HEADER_RE = re.compile(r"^#{1,6}[ \t]+", re.MULTILINE)
_HR_RE = re.compile(r"^(?:---|\*\*\*|___)[ \t]*$", re.MULTILINE)
_DOC_MARKER_RE = re.compile(
    r"^[ \t]*(?:Document[ \t]+\d+\b|\[\d+\])",
    re.MULTILINE | re.IGNORECASE,
)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_XML_WRAP_RE = re.compile(r"<([a-zA-Z_][\w:-]*)\b[^>]*>.*?</\1>", re.DOTALL)
_BLANK_LINE_RE = re.compile(r"\n[ \t]*\n+")


def _text_of(message: Any) -> str:
    if isinstance(message, dict) and isinstance(message.get("content"), str):
        return message["content"]
    return ""


def max_context_arms(environ: Mapping[str, str] | None = None) -> int:
    """Return the configured analysis-arm cap, falling back safely on errors."""
    values = os.environ if environ is None else environ
    raw = values.get(MAX_CONTEXT_ARMS_ENV)
    if raw is None:
        return DEFAULT_MAX_CONTEXT_ARMS
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONTEXT_ARMS
    if not 1 <= value <= MAX_CONTEXT_ARMS_CEILING:
        return DEFAULT_MAX_CONTEXT_ARMS
    return value


def _indivisible_spans(text: str) -> list[tuple[int, int]]:
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
    return any(start < position < end for start, end in spans)


def _split_oversized(
    text: str, start: int, end: int, *, maximum: int,
) -> list[tuple[int, int]]:
    """Split a large atom at a line edge, then at an exact character bound."""
    out: list[tuple[int, int]] = []
    cursor = start
    while end - cursor > maximum:
        limit = cursor + maximum
        cut = text.rfind("\n", cursor + max(1, maximum // 2), limit + 1)
        if cut <= cursor:
            cut = limit
        else:
            cut += 1
        out.append((cursor, cut))
        cursor = cut
    if cursor < end:
        out.append((cursor, end))
    return out


def _structural_atoms(text: str, *, maximum: int) -> list[tuple[int, int]]:
    """Return a contiguous partition at basic Markdown/document boundaries."""
    if not text:
        return []
    indivisible = _indivisible_spans(text)
    points = {0, len(text)}
    for pattern in (_HEADER_RE, _HR_RE, _DOC_MARKER_RE):
        for match in pattern.finditer(text):
            if not _inside(match.start(), indivisible):
                points.add(match.start())
    for match in _BLANK_LINE_RE.finditer(text):
        if not _inside(match.end(), indivisible):
            points.add(match.end())
    for start, end in indivisible:
        points.update((start, end))

    ordered = sorted(points)
    atoms: list[tuple[int, int]] = []
    for start, end in zip(ordered, ordered[1:]):
        if end > start:
            atoms.extend(_split_oversized(text, start, end, maximum=maximum))
    return atoms or [(0, len(text))]


def _pack_atoms(
    atoms: list[tuple[int, int]], *, target: int, minimum: int, maximum: int,
) -> list[tuple[int, int]]:
    """Pack adjacent atoms toward the adaptive target without creating gaps."""
    if not atoms:
        return []
    chunks: list[tuple[int, int]] = []
    current_start, current_end = atoms[0]
    for start, end in atoms[1:]:
        current_length = current_end - current_start
        combined_length = end - current_start
        should_close = (
            current_length >= minimum
            and (current_length >= target or combined_length > maximum)
        )
        if should_close:
            chunks.append((current_start, current_end))
            current_start, current_end = start, end
        else:
            current_end = end
    chunks.append((current_start, current_end))

    # The minimum is a packing preference, not permission to erase all useful
    # structure. Keep a final small section when merging it would collapse the
    # entire message to one arm.
    if len(chunks) > 2 and chunks[-1][1] - chunks[-1][0] < minimum:
        tail = chunks.pop()
        previous = chunks.pop()
        chunks.append((previous[0], tail[1]))
    return chunks


def _adaptive_sizes(total_characters: int, arm_limit: int) -> tuple[int, int, int]:
    target = max(1, math.ceil(total_characters / arm_limit))
    minimum = max(40, math.ceil(target / 2))
    maximum = max(240, target * 2, minimum * 2)
    return target, minimum, maximum


def _merge_to_cap(
    partitions: dict[int, list[tuple[int, int]]], arm_limit: int,
) -> None:
    """Merge the smallest adjacent pair until the global arm cap is met."""
    while sum(len(spans) for spans in partitions.values()) > arm_limit:
        candidates: list[tuple[int, int, int, int]] = []
        for message_index, spans in partitions.items():
            for position, (left, right) in enumerate(zip(spans, spans[1:])):
                candidates.append((right[1] - left[0], message_index, left[0], position))
        if not candidates:
            # Separate messages cannot be represented by one exact receipt
            # source. The search-universe planner reports this rare case as a
            # typed bound failure; Studio's single context message avoids it.
            return
        _length, message_index, _start, position = min(candidates)
        spans = partitions[message_index]
        merged = (spans[position][0], spans[position + 1][1])
        spans[position:position + 2] = [merged]


def derive_auto_source_spans(
    messages: list[dict], *, max_arms: int | None = None,
) -> list[dict]:
    """Derive the one flat, Markdown-aware source universe for a chat."""
    if not isinstance(messages, list) or not messages:
        return []
    arm_limit = max_context_arms() if max_arms is None else max_arms
    if isinstance(arm_limit, bool) or not isinstance(arm_limit, int) or arm_limit <= 0:
        raise ValueError("max_arms must be a positive integer")

    eligible = [
        (index, message, _text_of(message))
        for index, message in enumerate(messages)
        if isinstance(message, dict) and _text_of(message)
    ]
    total = sum(len(text) for _index, _message, text in eligible)
    if not total:
        return []
    target, minimum, maximum = _adaptive_sizes(total, arm_limit)
    partitions: dict[int, list[tuple[int, int]]] = {}
    for index, _message, text in eligible:
        atoms = _structural_atoms(text, maximum=maximum)
        partitions[index] = _pack_atoms(
            atoms, target=target, minimum=minimum, maximum=maximum,
        )
    _merge_to_cap(partitions, arm_limit)

    return [
        {
            "message_index": message_index,
            "unicode_range": [start, end],
            "source_kind": "source_span",
            "derivation": "adaptive_markdown",
        }
        for message_index in sorted(partitions)
        for start, end in partitions[message_index]
        if end > start
    ]


def annotate_auto_sources(messages: list[dict]) -> list[dict]:
    """Return a detached receipt copy with only server-derived source spans."""
    if not isinstance(messages, list):
        return []
    out = deepcopy(messages)
    for message in out:
        if isinstance(message, dict):
            message.pop("_clozn_sources", None)
    by_message: dict[int, list[dict]] = {}
    for span in derive_auto_source_spans(out):
        by_message.setdefault(span["message_index"], []).append(span)
    for index, spans in by_message.items():
        if len(spans) <= 1:
            continue
        out[index]["_clozn_sources"] = [
            {
                "unicode_range": list(span["unicode_range"]),
                "provenance_kind": "message",
            }
            for span in spans
        ]
    return out


def _source_rows_by_message(receipt: Mapping[str, Any]) -> dict[int, dict[str, list[dict]]]:
    rows_by_message: dict[int, dict[str, list[dict]]] = {}
    delivered = receipt.get("delivered")
    if not isinstance(delivered, list):
        return rows_by_message
    for index, row in enumerate(delivered):
        if not isinstance(row, dict):
            continue
        roots: list[dict] = []
        root_id = row.get("segment_id")
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
        spans = [
            {**source, "source_kind": "source_span"}
            for source in row.get("sources", [])
            if isinstance(source, dict) and isinstance(source.get("source_id"), str)
        ] if isinstance(row.get("sources"), list) else []
        rows_by_message[index] = {"root": roots, "spans": spans}
    return rows_by_message


def _copy_source(
    source: Mapping[str, Any], *, message_index: int, role: str, derivation: str,
) -> dict:
    item = {
        "source_id": source["source_id"],
        "message_index": message_index,
        "role": role,
        "unicode_range": list(source["unicode_range"]),
        "byte_range": list(source["byte_range"]),
        "source_kind": source["source_kind"],
        "derivation": derivation,
    }
    for key in ("segment_id", "content_sha256", "provenance_kind"):
        if source.get(key) is not None:
            item[key] = source[key]
    return item


def _range(source: Mapping[str, Any]) -> tuple[int, int] | None:
    value = source.get("unicode_range")
    if not (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(part, int) and not isinstance(part, bool) for part in value)
    ):
        return None
    return int(value[0]), int(value[1])


def _covers_partition(sources: list[Mapping[str, Any]], length: int) -> bool:
    ordered = sorted(sources, key=lambda source: (_range(source) or (length + 1, length + 1)))
    cursor = 0
    for source in ordered:
        span = _range(source)
        if span is None or span[0] != cursor or span[1] <= span[0] or span[1] > length:
            return False
        cursor = span[1]
    return cursor == length


def _failed_manifest(run_id: str, message: str) -> dict:
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "basis": "messages",
        "units": [],
        "default_source_ids": [],
        "error": message,
    }


def build_context_unit_manifest(run: dict) -> dict:
    """Bind the derived flat partition to canonical Context Receipt IDs."""
    run_id = run.get("id") if isinstance(run, dict) and isinstance(run.get("id"), str) else ""
    messages = run.get("messages") if isinstance(run, dict) else None
    if not isinstance(messages, list):
        return _failed_manifest(run_id, "run.messages is not a message list")
    base = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "basis": "messages",
        "units": [],
        "default_source_ids": [],
    }
    receipt = run.get("context_receipt") if isinstance(run, dict) else None
    if not isinstance(receipt, dict):
        return _failed_manifest(run_id, "run has no finalized Context Receipt")

    rows_by_message = _source_rows_by_message(receipt)
    candidate_ids = list(dict.fromkeys(
        source["source_id"]
        for rows in rows_by_message.values()
        for source in rows.get("root", []) + rows.get("spans", [])
        if isinstance(source.get("source_id"), str)
    ))
    if not candidate_ids:
        return _failed_manifest(run_id, "finalized Context Receipt has no source catalog")

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
        return _failed_manifest(run_id, f"strict receipt reconciliation failed: {exc}")

    expected_by_message: dict[int, list[dict]] = {}
    for span in derive_auto_source_spans(messages):
        expected_by_message.setdefault(span["message_index"], []).append(span)

    selected: list[dict] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict) or not _text_of(message):
            continue
        content = _text_of(message)
        sources = by_message.get(index, [])
        root = next((source for source in sources if source.get("source_kind") == "whole_message"), None)
        spans = [source for source in sources if source.get("source_kind") == "source_span"]
        if root is None:
            return _failed_manifest(run_id, f"message {index} has no canonical whole-message root")

        expected = expected_by_message.get(index, [])
        if len(expected) <= 1:
            chosen = [root]
            derivation = "message_root"
        else:
            expected_ranges = [tuple(item["unicode_range"]) for item in expected]
            chosen = []
            for expected_range in expected_ranges:
                matches = [source for source in spans if _range(source) == expected_range]
                if len(matches) != 1:
                    return _failed_manifest(
                        run_id,
                        f"message {index} derived partition does not reconcile with the Context Receipt",
                    )
                chosen.append(matches[0])
            if not _covers_partition(chosen, len(content)):
                return _failed_manifest(
                    run_id,
                    f"message {index} derived partition does not cover the recorded context",
                )
            derivation = "adaptive_markdown"

        for source in chosen:
            selected.append(_copy_source(
                source,
                message_index=index,
                role=str(message.get("role") or ""),
                derivation=derivation,
            ))

    default_ids = [source["source_id"] for source in selected]
    if len(default_ids) != len(set(default_ids)):
        return _failed_manifest(run_id, "default source universe contains duplicate canonical IDs")
    base["units"] = selected
    base["default_source_ids"] = default_ids
    return base


__all__ = [
    "DEFAULT_MAX_CONTEXT_ARMS",
    "MAX_CONTEXT_ARMS_ENV",
    "SCHEMA",
    "annotate_auto_sources",
    "build_context_unit_manifest",
    "derive_auto_source_spans",
    "max_context_arms",
]
