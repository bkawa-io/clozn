"""Deterministic bounded search-universe planning for recorded context.

This module chooses a finite, non-overlapping partition from the canonical
Context Receipt source catalog.  It does not score, call a model, or mutate a
run.  Automatic parent sources are selected only when all of their adjacent
children are currently on the partition frontier; explicit caller sources are
never merged.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from collections.abc import Iterable, Mapping
from typing import Any

from clozn import schemas
from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    resolve_context_receipt_source_set,
)
from clozn.runs.context_units import protected_message_indices


SCHEMA = "clozn.context-search-universe.v1"
POLICY = "bounded_structural_partition.v1"
POLICY_KIND = "bounded_structural_partition"
MERGE_POLICY = "smallest_adjacent_auto_siblings_leftmost.v1"


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
        raise ContextSearchUniverseError(f"source {source.get('source_id')!r} has an invalid Unicode range")
    return start, end


def _source_order(source: Mapping[str, Any]) -> tuple[int, tuple[int, int], str]:
    return (
        int(source.get("message_index", 2**31 - 1)),
        _range(source),
        str(source.get("source_id", "")),
    )


def _ancestor(source_id: str, child_id: str, by_id: Mapping[str, Mapping[str, Any]]) -> bool:
    parent = by_id.get(child_id, {}).get("parent_source_id")
    seen: set[str] = set()
    while isinstance(parent, str) and parent and parent not in seen:
        if parent == source_id:
            return True
        seen.add(parent)
        parent = by_id.get(parent, {}).get("parent_source_id")
    return False


def _validate_partition(
    *,
    selected_ids: Iterable[str],
    unit_by_id: Mapping[str, Mapping[str, Any]],
    catalog_by_id: Mapping[str, Mapping[str, Any]],
    basis_messages: list[dict],
    protected: set[int],
) -> None:
    selected = [catalog_by_id[source_id] for source_id in selected_ids]
    by_message: dict[int, list[Mapping[str, Any]]] = {}
    for source in selected:
        index = source.get("message_index")
        if not isinstance(index, int):
            raise ContextSearchUniverseError("a selected source has no message index")
        if index in protected:
            raise ContextSearchUniverseError("the search universe includes a protected message")
        by_message.setdefault(index, []).append(source)
        if source.get("source_id") in unit_by_id:
            unit = unit_by_id[source["source_id"]]
            if unit.get("message_index") != index or _range(unit) != _range(source):
                raise ContextSearchUniverseError("Context Unit and receipt source ranges disagree")

    for index, sources in by_message.items():
        if not (0 <= index < len(basis_messages)):
            raise ContextSearchUniverseError("a selected source points outside the message basis")
        length = len(basis_messages[index].get("content", ""))
        ordered = sorted(sources, key=lambda source: (_range(source), str(source["source_id"])))
        for left_index, left in enumerate(ordered):
            for right in ordered[left_index + 1:]:
                if _ancestor(str(left["source_id"]), str(right["source_id"]), catalog_by_id) or _ancestor(
                    str(right["source_id"]), str(left["source_id"]), catalog_by_id
                ):
                    raise ContextSearchUniverseError("a search universe contains both a parent and child")
        cursor = 0
        for source in ordered:
            start, end = _range(source)
            if start != cursor or end > length:
                raise ContextSearchUniverseError(
                    f"selected sources do not exactly partition removable message {index}"
                )
            cursor = end
        if cursor != length:
            raise ContextSearchUniverseError(
                f"selected sources do not cover removable message {index}"
            )


def _condition_artifact(
    *,
    run_id: str,
    basis_digest: str,
    policy: dict,
    source_ids: list[str],
    coverage: dict,
    code: str,
    message: str,
) -> dict:
    identity = {
        "run_id": run_id,
        "basis_context_units_digest": basis_digest,
        "policy": policy,
        "source_ids": source_ids,
        "status": "bound_exceeded",
        "condition": {"code": code, "message": message},
    }
    return {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "universe_id": "mcu_" + _digest(identity)[:24],
        "basis_context_units_digest": basis_digest,
        "policy": policy,
        "source_ids": source_ids,
        "source_count": len(source_ids),
        "coverage": coverage,
        "status": "bound_exceeded",
        "condition": {"code": code, "message": message},
    }


def plan_context_search_universe(
    run: Mapping[str, Any],
    context_units: Mapping[str, Any],
    *,
    max_units: int = 50,
    policy: str = POLICY,
) -> dict:
    """Plan a stable bounded partition over a recorded run's context units.

    The source IDs in the returned artifact are always accepted by the strict
    Context Receipt resolver.  A caller-defined structure that cannot fit is
    returned as a typed ``bound_exceeded`` artifact, never silently merged.
    Invalid or drifted receipt/unit evidence raises
    :class:`ContextSearchUniverseError` before an artifact is returned.
    """
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
    except Exception as exc:  # noqa: BLE001 - normalize schema failures at the public seam
        raise ContextSearchUniverseError(f"invalid Context Units manifest: {exc}") from exc
    run_id = str(run["id"])
    if manifest.get("run_id") != run_id:
        raise ContextSearchUniverseError("Context Units manifest does not belong to this run")
    if manifest.get("error"):
        raise ContextSearchUniverseError(f"Context Units manifest is unavailable: {manifest['error']}")

    messages = run.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ContextSearchUniverseError("run.messages must be a non-empty message list")
    expected_protected = sorted(protected_message_indices(messages))
    protected = manifest.get("protected_message_indices")
    if protected != expected_protected:
        raise ContextSearchUniverseError("Context Units protected-message partition is stale")

    units = manifest.get("units")
    default_ids = manifest.get("default_source_ids")
    if not isinstance(units, list) or not isinstance(default_ids, list) or not default_ids:
        raise ContextSearchUniverseError("Context Units manifest has no removable source universe")
    unit_by_id = {unit.get("source_id"): unit for unit in units if isinstance(unit, dict)}
    if len(unit_by_id) != len(units) or set(default_ids) != set(unit_by_id):
        raise ContextSearchUniverseError("Context Units units and default_source_ids disagree")

    try:
        resolved = resolve_context_receipt_source_set(run, default_ids)
    except (ContextReceiptSourceResolutionError, TypeError, ValueError) as exc:
        raise ContextSearchUniverseError(f"strict source resolution failed: {exc}") from exc
    catalog = resolved.get("sources")
    basis_messages = resolved.get("basis_messages")
    if not isinstance(catalog, list) or not isinstance(basis_messages, list):
        raise ContextSearchUniverseError("strict source resolver returned incomplete catalog evidence")
    catalog_by_id = {
        source.get("source_id"): source for source in catalog
        if isinstance(source, dict) and isinstance(source.get("source_id"), str)
    }
    if len(catalog_by_id) != len(catalog):
        raise ContextSearchUniverseError("strict source catalog contains duplicate or malformed IDs")
    missing = [source_id for source_id in default_ids if source_id not in catalog_by_id]
    if missing:
        raise ContextSearchUniverseError("Context Units contains IDs absent from the strict source catalog")
    _validate_partition(
        selected_ids=default_ids,
        unit_by_id=unit_by_id,
        catalog_by_id=catalog_by_id,
        basis_messages=basis_messages,
        protected=set(protected),
    )

    policy_doc = {
        "kind": POLICY_KIND,
        "version": POLICY,
        "max_units": max_units,
        "merge": MERGE_POLICY,
        "explicit_over_cap": "typed_bound_exceeded",
    }
    basis_digest = _digest(manifest)
    selected_ids = list(default_ids)
    explicit_ids = {
        source_id for source_id, unit in unit_by_id.items()
        if unit.get("derivation") in {"caller_explicit", "caller_fallback_root"}
    }
    coverage_base = {
        "protected_message_indices": list(protected),
        "removable_message_indices": sorted({catalog_by_id[sid]["message_index"] for sid in selected_ids}),
        "removable_content_covered": True,
        "partition_verified": True,
        "complete_catalog_source_count": len(catalog),
        "explicit_source_count": len(explicit_ids),
    }

    if len(explicit_ids) > max_units:
        artifact = _condition_artifact(
            run_id=run_id, basis_digest=basis_digest, policy=policy_doc,
            source_ids=selected_ids, coverage=coverage_base,
            code="explicit_units_exceed_max_units",
            message=(f"{len(explicit_ids)} caller-defined source units exceed max_units={max_units}; "
                     "explicit boundaries were preserved"),
        )
        schemas.validate(artifact)
        return artifact

    # A frontier contains exactly one selected source for every point in an
    # auto-derived message.  Parent candidates are therefore safe to select
    # only when all of their direct children are frontier members.
    frontier = set(selected_ids)
    auto_message_indices = {
        int(unit["message_index"])
        for unit in units
        if unit.get("derivation") == "auto_structural"
    }
    children_by_parent: dict[str, list[str]] = {}
    for source in catalog:
        parent = source.get("parent_source_id")
        if isinstance(parent, str):
            children_by_parent.setdefault(parent, []).append(source["source_id"])
    for children in children_by_parent.values():
        children.sort(key=lambda source_id: _source_order(catalog_by_id[source_id]))

    while len(frontier) > max_units:
        candidates: list[tuple[tuple[Any, ...], str, list[str]]] = []
        for parent_id, children in children_by_parent.items():
            parent = catalog_by_id.get(parent_id)
            if not parent or parent.get("message_index") not in auto_message_indices:
                continue
            if not all(child in frontier for child in children):
                continue
            if any(child in explicit_ids for child in children):
                continue
            start, end = _range(parent)
            candidates.append(((end - start, int(parent["message_index"]), start, parent_id), parent_id, children))
        if not candidates:
            # A valid legacy/hand-built catalog may expose leaves without
            # intermediate auto parents.  The existing whole-message root is
            # still a canonical, strict source and is a safe final fallback.
            fallback: list[tuple[tuple[Any, ...], str]] = []
            for index in sorted(auto_message_indices):
                members = [source_id for source_id in frontier
                           if catalog_by_id[source_id].get("message_index") == index]
                if len(members) <= 1 or any(source_id in explicit_ids for source_id in members):
                    continue
                roots = [source for source in catalog
                         if source.get("message_index") == index
                         and source.get("source_kind") == "whole_message"]
                if len(roots) == 1:
                    root = roots[0]
                    fallback.append(((len(basis_messages[index].get("content", "")), index), root["source_id"]))
            if fallback:
                _score, root_id = min(fallback)
                frontier.difference_update({source_id for source_id in frontier
                                            if catalog_by_id[source_id].get("message_index") ==
                                            catalog_by_id[root_id].get("message_index")})
                frontier.add(root_id)
                continue
            break
        _score, parent_id, children = min(candidates)
        frontier.difference_update(children)
        frontier.add(parent_id)

    selected_ids = [source_id for source_id in sorted(frontier, key=lambda sid: _source_order(catalog_by_id[sid]))]
    _validate_partition(
        selected_ids=selected_ids,
        unit_by_id=unit_by_id,
        catalog_by_id=catalog_by_id,
        basis_messages=basis_messages,
        protected=set(protected),
    )
    coverage = {**coverage_base, "selected_source_count": len(selected_ids)}
    status = "planned" if len(selected_ids) <= max_units else "bound_exceeded"
    condition = None if status == "planned" else {
        "code": "bounded_partition_unreachable",
        "message": f"the strict canonical catalog cannot reach max_units={max_units} without crossing a boundary",
    }
    identity = {
        "run_id": run_id,
        "basis_context_units_digest": basis_digest,
        "policy": policy_doc,
        "source_ids": selected_ids,
        "status": status,
        "condition": condition,
    }
    artifact = {
        "schema_version": SCHEMA,
        "run_id": run_id,
        "universe_id": "mcu_" + _digest(identity)[:24],
        "basis_context_units_digest": basis_digest,
        "policy": policy_doc,
        "source_ids": selected_ids,
        "source_count": len(selected_ids),
        "coverage": coverage,
        "status": status,
    }
    if condition is not None:
        artifact["condition"] = condition
    schemas.validate(artifact)
    # This second resolver call is intentional: it proves the actual bounded
    # output, including selected canonical parents, rather than only proving
    # the original Context Unit leaves used to construct it.
    try:
        resolve_context_receipt_source_set(run, selected_ids)
    except (ContextReceiptSourceResolutionError, TypeError, ValueError) as exc:
        raise ContextSearchUniverseError(f"planned source IDs failed strict resolution: {exc}") from exc
    return artifact


__all__ = [
    "ContextSearchUniverseError",
    "MERGE_POLICY",
    "POLICY",
    "SCHEMA",
    "plan_context_search_universe",
]
