"""Validation and identity for the shared flat Context Unit universe."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from copy import deepcopy
from typing import Any

from clozn import schemas
from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    resolve_context_receipt_source_set,
)


SCHEMA = "clozn.context-search-universe.v1"
POLICY = "flat_markdown_partition.v1"
POLICY_KIND = "flat_markdown_partition"


class ContextSearchUniverseError(ValueError):
    """Raised when a run cannot produce a faithful search partition."""


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _range(source: Mapping[str, Any]) -> tuple[int, int]:
    value = source.get("unicode_range")
    if not (
        isinstance(value, (list, tuple)) and len(value) == 2
        and all(isinstance(part, int) and not isinstance(part, bool) for part in value)
    ):
        raise ContextSearchUniverseError(
            f"source {source.get('source_id')!r} has no exact Unicode range"
        )
    start, end = int(value[0]), int(value[1])
    if start < 0 or end < start:
        raise ContextSearchUniverseError(
            f"source {source.get('source_id')!r} has an invalid Unicode range"
        )
    return start, end


def _validate_partition(
    *,
    selected_ids: Iterable[str],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    basis_messages: list[dict],
) -> None:
    by_message: dict[int, list[Mapping[str, Any]]] = {}
    for source_id in selected_ids:
        source = catalog_by_id[source_id]
        index = source.get("message_index")
        if not isinstance(index, int):
            raise ContextSearchUniverseError("a selected source has no message index")
        unit = unit_by_id[source_id]
        if unit.get("message_index") != index or _range(unit) != _range(source):
            raise ContextSearchUniverseError("Context Unit and receipt source ranges disagree")
        by_message.setdefault(index, []).append(source)

    for index, sources in by_message.items():
        if not 0 <= index < len(basis_messages):
            raise ContextSearchUniverseError("a selected source points outside the message basis")
        content = basis_messages[index].get("content", "")
        length = len(content)
        cursor = 0
        for source in sorted(sources, key=lambda item: (_range(item), str(item["source_id"]))):
            start, end = _range(source)
            if start != cursor or end <= start or end > length:
                raise ContextSearchUniverseError(
                    f"selected sources do not exactly partition message {index}"
                )
            cursor = end
        if cursor != length:
            raise ContextSearchUniverseError(
                f"selected sources do not cover removable content in message {index}"
            )


def plan_context_search_universe(
    run: Mapping[str, Any],
    context_units: Mapping[str, Any],
    *,
    max_units: int = 16,
    policy: str = POLICY,
) -> dict:
    """Validate and expose the same bounded units produced during capture."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ContextSearchUniverseError("a recorded run with a non-empty id is required")
    if policy != POLICY:
        raise ContextSearchUniverseError(f"unsupported context search-universe policy: {policy!r}")
    if isinstance(max_units, bool) or not isinstance(max_units, int) or max_units <= 0:
        raise ContextSearchUniverseError("max_units must be a positive integer")
    if not isinstance(context_units, Mapping):
        raise ContextSearchUniverseError("context_units must be a manifest object")

    manifest = deepcopy(dict(context_units))
    try:
        schemas.validate(manifest)
    except Exception as exc:  # noqa: BLE001 - normalize failures at this seam
        raise ContextSearchUniverseError(f"invalid Context Units manifest: {exc}") from exc
    run_id = str(run["id"])
    if manifest.get("run_id") != run_id:
        raise ContextSearchUniverseError("Context Units manifest does not belong to this run")
    if manifest.get("error"):
        raise ContextSearchUniverseError(f"Context Units manifest is unavailable: {manifest['error']}")

    messages = run.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContextSearchUniverseError("run.messages must be a non-empty message list")
    units = manifest.get("units")
    source_ids = manifest.get("default_source_ids")
    if not isinstance(units, list) or not isinstance(source_ids, list) or not source_ids:
        raise ContextSearchUniverseError("Context Units manifest has no removable source universe")
    unit_by_id = {unit.get("source_id"): unit for unit in units if isinstance(unit, dict)}
    if len(unit_by_id) != len(units) or set(source_ids) != set(unit_by_id):
        raise ContextSearchUniverseError("Context Units units and default_source_ids disagree")

    try:
        resolved = resolve_context_receipt_source_set(run, source_ids)
    except (ContextReceiptSourceResolutionError, TypeError, ValueError) as exc:
        raise ContextSearchUniverseError(f"strict source resolution failed: {exc}") from exc
    catalog = resolved.get("sources")
    basis_messages = resolved.get("basis_messages")
    if not isinstance(catalog, list) or not isinstance(basis_messages, list):
        raise ContextSearchUniverseError("strict source resolver returned incomplete catalog evidence")
    catalog_by_id = {
        source.get("source_id"): source
        for source in catalog
        if isinstance(source, dict) and isinstance(source.get("source_id"), str)
    }
    if len(catalog_by_id) != len(catalog) or any(source_id not in catalog_by_id for source_id in source_ids):
        raise ContextSearchUniverseError("strict source catalog is incomplete or malformed")
    _validate_partition(
        selected_ids=source_ids,
        unit_by_id=unit_by_id,
        catalog_by_id=catalog_by_id,
        basis_messages=basis_messages,
    )

    policy_doc = {
        "kind": POLICY_KIND,
        "version": POLICY,
        "max_units": max_units,
    }
    basis_digest = _digest(manifest)
    coverage = {
        "removable_message_indices": sorted({catalog_by_id[sid]["message_index"] for sid in source_ids}),
        "removable_content_covered": True,
        "partition_verified": True,
        "complete_catalog_source_count": len(catalog),
        "selected_source_count": len(source_ids),
    }
    status = "planned" if len(source_ids) <= max_units else "bound_exceeded"
    condition = None if status == "planned" else {
        "code": "derived_units_exceed_max_units",
        "message": (
            f"{len(source_ids)} exact message-scoped units exceed max_units={max_units}; "
            "units from separate messages cannot be merged"
        ),
    }
    identity = {
        "run_id": run_id,
        "basis_context_units_digest": basis_digest,
        "policy": policy_doc,
        "source_ids": list(source_ids),
        "status": status,
        "condition": condition,
    }
    artifact = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "universe_id": "mcu_" + _digest(identity)[:24],
        "basis_context_units_digest": basis_digest,
        "policy": policy_doc,
        "source_ids": list(source_ids),
        "source_count": len(source_ids),
        "coverage": coverage,
        "status": status,
    }
    if condition is not None:
        artifact["condition"] = condition
    schemas.validate(artifact)
    return artifact


__all__ = [
    "ContextSearchUniverseError",
    "POLICY",
    "POLICY_KIND",
    "SCHEMA",
    "plan_context_search_universe",
]
