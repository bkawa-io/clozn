"""Budgeted, direct Context Dependence hierarchy studies.

This module is deliberately a small orchestration layer over
``clozn.receipts.context_dependence.ContextDependenceStudy``.  It does not
score tokens itself, fit an estimator, enumerate pairs, or infer whether a
source was used.  Its hierarchy is only a deterministic record of *which
source sets were actually submitted to the direct deletion experiment*.

The root set is always scheduled first.  Consequently a large direct effect
for a parent remains visible even when all directly measured children are
small: arithmetic between those measurements is retained only as explicitly
derived search metadata, never as a new measurement.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from typing import Any

from clozn import schemas
from clozn.receipts.context_dependence import ContextDependenceStudy as _MeasurementStudy


DEFAULT_PASSES_REQUESTED = 8
"""Conservative default: one baseline, root deletion, and up to six arms."""


class ContextDependenceStudyError(ValueError):
    """A requested direct hierarchy cannot be represented faithfully."""


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str,
    ).encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _as_source_ids(value: Iterable[str], *, name: str, source_order: tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ContextDependenceStudyError(f"{name} must be an iterable of source ID strings")
    try:
        supplied = tuple(value)
    except TypeError as exc:
        raise ContextDependenceStudyError(f"{name} must be an iterable of source ID strings") from exc
    if not supplied or any(not isinstance(source_id, str) or not source_id for source_id in supplied):
        raise ContextDependenceStudyError(f"{name} must contain at least one non-empty source ID")
    if len(set(supplied)) != len(supplied):
        raise ContextDependenceStudyError(f"{name} must not contain duplicate source IDs")
    unknown = set(supplied).difference(source_order)
    if unknown:
        raise ContextDependenceStudyError(
            f"{name} contains source IDs absent from the Context Receipt: {', '.join(sorted(unknown))}")
    # Prompt/receipt order, rather than caller container order, is the stable
    # display and traversal order.  Task 1 itself separately canonicalizes the
    # source set for its content-addressed experiment identity.
    return tuple(source_id for source_id in source_order if source_id in supplied)


@dataclass(frozen=True)
class _Group:
    group_id: str
    source_ids: tuple[str, ...]
    structure_origin: str
    children: tuple["_Group", ...]


def _group_items(value: Any, *, name: str) -> list[dict]:
    """Normalize a compact mapping or an explicit sequence of group objects."""
    if value is None:
        return []
    if isinstance(value, Mapping):
        # One explicit group object is convenient for the single-group case.
        if "group_id" in value or "source_ids" in value or "children" in value:
            return [dict(value)]
        result = []
        for group_id, group_value in value.items():
            if isinstance(group_value, Mapping):
                item = dict(group_value)
                item.setdefault("group_id", group_id)
            else:
                item = {"group_id": group_id, "source_ids": group_value}
            result.append(item)
        return result
    if isinstance(value, (str, bytes)):
        raise ContextDependenceStudyError(f"{name} must be a group mapping or iterable of group objects")
    try:
        items = list(value)
    except TypeError as exc:
        raise ContextDependenceStudyError(f"{name} must be a group mapping or iterable of group objects") from exc
    if any(not isinstance(item, Mapping) for item in items):
        raise ContextDependenceStudyError(f"{name} entries must be objects")
    return [dict(item) for item in items]


def _normalise_groups(
    value: Any,
    *,
    name: str,
    source_order: tuple[str, ...],
    allowed_source_ids: tuple[str, ...],
    seen_group_ids: set[str],
) -> tuple[_Group, ...]:
    items = _group_items(value, name=name)
    groups: list[_Group] = []
    sibling_sources: set[str] = set()
    allowed = set(allowed_source_ids)
    for index, item in enumerate(items):
        unsupported = set(item).difference({"group_id", "source_ids", "children", "structure_origin"})
        if unsupported:
            raise ContextDependenceStudyError(
                f"{name}[{index}] has unsupported fields: {', '.join(sorted(unsupported))}")
        group_id = item.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            raise ContextDependenceStudyError(f"{name}[{index}].group_id must be a non-empty string")
        if group_id in seen_group_ids:
            raise ContextDependenceStudyError(f"group_id must be unique: {group_id}")
        seen_group_ids.add(group_id)
        source_ids = _as_source_ids(
            item.get("source_ids"), name=f"{name}[{index}].source_ids", source_order=source_order,
        )
        outside_parent = set(source_ids).difference(allowed)
        if outside_parent:
            raise ContextDependenceStudyError(
                f"{name}[{index}] contains sources outside its parent set: {', '.join(sorted(outside_parent))}")
        overlap = sibling_sources.intersection(source_ids)
        if overlap:
            raise ContextDependenceStudyError(
                f"{name} sibling groups overlap: {', '.join(sorted(overlap))}")
        sibling_sources.update(source_ids)
        origin = item.get("structure_origin", "caller_supplied")
        if origin not in {"context_receipt", "caller_supplied", "search_generated"}:
            raise ContextDependenceStudyError(
                f"{name}[{index}].structure_origin must be context_receipt, caller_supplied, or search_generated")
        children = _normalise_groups(
            item.get("children"), name=f"{name}[{index}].children", source_order=source_order,
            allowed_source_ids=source_ids, seen_group_ids=seen_group_ids,
        )
        groups.append(_Group(group_id, source_ids, origin, children))
    # Stable ordering makes two equivalent mapping inputs render identically.
    return tuple(sorted(groups, key=lambda group: group.group_id))


def _node_id(*, kind: str, source_ids: tuple[str, ...], path: tuple[str, ...]) -> str:
    return "cdh_" + _canonical_digest({
        "kind": kind, "source_ids": source_ids, "path": path,
    })[:24]


def _node(
    *,
    kind: str,
    source_ids: tuple[str, ...],
    parent_node_id: str | None,
    path: tuple[str, ...],
    structure_origin: str,
    group_id: str | None = None,
) -> dict:
    result = {
        "node_id": _node_id(kind=kind, source_ids=source_ids, path=path),
        "parent_node_id": parent_node_id,
        "node_kind": kind,
        "source_ids": list(source_ids),
        "structure_origin": structure_origin,
        "measurement_state": "pending",
    }
    if group_id is not None:
        result["group_id"] = group_id
    return result


def _hierarchy_nodes(
    *, root_source_ids: tuple[str, ...], groups: tuple[_Group, ...], quick: bool,
) -> list[dict]:
    """Plan nodes without inspecting effects or running a scorer.

    ``quick`` means root plus only the explicitly supplied top-level structure.
    When no such structure exists, it is intentionally root-only; source units
    remain available to the ordinary bounded run without pretending they form a
    further natural grouping.
    """
    root = _node(
        kind="requested_root_set", source_ids=root_source_ids, parent_node_id=None,
        path=("root",), structure_origin="requested_source_set",
    )
    result = [root]

    def append_group(group: _Group, parent: dict, path: tuple[str, ...], *, include_descendants: bool) -> None:
        group_node = _node(
            kind="structural_source_group", source_ids=group.source_ids,
            parent_node_id=parent["node_id"], path=path + (group.group_id,),
            structure_origin=group.structure_origin, group_id=group.group_id,
        )
        result.append(group_node)
        if not include_descendants:
            return
        if group.children:
            for child in group.children:
                append_group(child, group_node, path + (group.group_id,), include_descendants=True)
            # A group may honestly contain sources that its natural children do
            # not further classify.  Those sources remain the receipt-native
            # leaf units, not invented paragraph subdivisions.
            covered = {source_id for child in group.children for source_id in child.source_ids}
            for source_id in group.source_ids:
                if source_id not in covered:
                    result.append(_node(
                        kind="source_unit", source_ids=(source_id,), parent_node_id=group_node["node_id"],
                        path=path + (group.group_id, source_id), structure_origin="context_receipt",
                    ))
        else:
            for source_id in group.source_ids:
                result.append(_node(
                    kind="source_unit", source_ids=(source_id,), parent_node_id=group_node["node_id"],
                    path=path + (group.group_id, source_id), structure_origin="context_receipt",
                ))

    if groups:
        for group in groups:
            append_group(group, root, ("root",), include_descendants=not quick)
        if not quick:
            covered = {source_id for group in groups for source_id in group.source_ids}
            for source_id in root_source_ids:
                if source_id not in covered:
                    result.append(_node(
                        kind="source_unit", source_ids=(source_id,), parent_node_id=root["node_id"],
                        path=("root", source_id), structure_origin="context_receipt",
                    ))
    elif not quick:
        for source_id in root_source_ids:
            result.append(_node(
                kind="source_unit", source_ids=(source_id,), parent_node_id=root["node_id"],
                path=("root", source_id), structure_origin="context_receipt",
            ))
    return result


class DirectContextDependenceStudy:
    """A deterministic, pass-budgeted hierarchy of Task 1 measurements.

    Construct with a recorded run and its scoring substrate, then call
    :meth:`run`.  Each node's ``direct_effect`` is copied from one Task 1
    experiment and carries its ``experiment_id``.  The class intentionally
    has no effect floor or source-level semantic conclusion API.
    """

    def __init__(
        self,
        run: dict,
        sub: Any,
        *,
        target: dict | None = None,
        root_source_ids: Iterable[str] | None = None,
        source_groups: Any = None,
        structural_groups: Any = None,
        passes_requested: int = DEFAULT_PASSES_REQUESTED,
        quick: bool = False,
        clock: Any = None,
    ):
        if not _is_int(passes_requested) or passes_requested < 1:
            raise ContextDependenceStudyError("passes_requested must be a positive integer")
        if target is not None:
            raise ContextDependenceStudyError(
                "target is not accepted by clozn.context-dependence-study.v2; "
                "run-level studies score the whole continuation and selections are projections"
            )
        if not isinstance(quick, bool):
            raise ContextDependenceStudyError("quick must be a boolean")
        if source_groups is not None and structural_groups is not None:
            raise ContextDependenceStudyError("supply source_groups or structural_groups, not both")
        kwargs = {}
        if clock is not None:
            kwargs["clock"] = clock
        self._measurement = _MeasurementStudy(run, sub, **kwargs)
        source_order = self._measurement.source_ids
        if not source_order:
            raise ContextDependenceStudyError("the Context Receipt has no canonical deletable sources")
        self._source_order = tuple(source_order)
        self._root_source_ids = (
            self._source_order if root_source_ids is None
            else _as_source_ids(root_source_ids, name="root_source_ids", source_order=self._source_order)
        )
        group_input = source_groups if source_groups is not None else structural_groups
        self._groups = _normalise_groups(
            group_input, name="source_groups", source_order=self._source_order,
            allowed_source_ids=self._root_source_ids, seen_group_ids=set(),
        )
        self._passes_requested = passes_requested
        self._quick = quick
        self._completed_document: dict | None = None

    @property
    def source_ids(self) -> tuple[str, ...]:
        """Canonical receipt source IDs in recorded prompt order."""
        return self._source_order

    def run(self) -> dict:
        """Run the direct hierarchy once and return a schema-valid study artifact.

        Baseline teacher-forcing costs one pass.  The root therefore needs two
        available passes on a new study (baseline plus deletion arm).  If the
        budget contains only the baseline, the returned hierarchy makes the
        unmeasured root explicit rather than borrowing a number from another
        set or exceeding the budget.
        """
        if self._completed_document is not None:
            return deepcopy(self._completed_document)

        nodes = _hierarchy_nodes(
            root_source_ids=self._root_source_ids, groups=self._groups, quick=self._quick,
        )
        experiment_by_set: dict[frozenset[str], dict] = {}
        passes_consumed = 0

        for node in nodes:
            source_set = frozenset(node["source_ids"])
            cached = experiment_by_set.get(source_set)
            if cached is not None:
                node["measurement_state"] = "measured_reused"
                node["direct_effect"] = _effect_reference(cached)
                continue
            # Until an arm has been measured, Task 1 must score the baseline as
            # well.  Every subsequent unique deletion adds exactly one arm.
            needed = 2 if not experiment_by_set else 1
            if passes_consumed + needed > self._passes_requested:
                node["measurement_state"] = "unmeasured_budget_exhausted"
                continue
            experiment = self._measurement.measure_removal_effect(node["source_ids"])
            experiment_by_set[source_set] = experiment
            passes_consumed += needed
            node["measurement_state"] = "measured"
            node["direct_effect"] = _effect_reference(experiment)

        # ``document`` scores the baseline for a 1-pass study which could not
        # afford a root arm.  It is otherwise a no-op with respect to scoring.
        direct_document = self._measurement.document()
        actual_passes = direct_document["budget"]["passes_consumed"]
        # The accounting invariant should hold even if the direct primitive is
        # changed in the future; never report an optimistic locally predicted
        # number.  A changed primitive that unexpectedly exceeds a requested
        # budget fails closed instead of being silently normalized away.
        if actual_passes > self._passes_requested:
            raise ContextDependenceStudyError(
                "direct measurement consumed more score passes than the requested budget")
        if actual_passes != passes_consumed and not experiment_by_set:
            passes_consumed = actual_passes
        elif actual_passes != passes_consumed:
            raise ContextDependenceStudyError("direct measurement pass accounting disagreed with the study")

        hierarchy = _build_hierarchy(nodes)
        document = deepcopy(direct_document)
        document["hierarchy"] = hierarchy
        document["budget"] = {
            "passes_requested": self._passes_requested,
            "passes_consumed": actual_passes,
            "passes_remaining": self._passes_requested - actual_passes,
            "state": "exhausted" if actual_passes == self._passes_requested else "complete",
        }
        # The Task 1 schema deliberately permits additive direct-study fields.
        # Validate after attaching hierarchy metadata rather than trusting that
        # a refactor accidentally leaves IDs or direct records detached.
        schemas.validate(document)
        self._completed_document = deepcopy(document)
        return document


def _effect_reference(experiment: Mapping[str, Any]) -> dict:
    """The only hierarchy effect projection; it is always experiment-backed."""
    return {
        "experiment_id": experiment["experiment_id"],
        "delta_nats": experiment["delta_nats"],
        "provenance": "measured",
    }


def _build_hierarchy(nodes: list[dict]) -> dict:
    by_parent: dict[str, list[dict]] = {}
    for node in nodes:
        parent_id = node.get("parent_node_id")
        if isinstance(parent_id, str):
            by_parent.setdefault(parent_id, []).append(node)
    nonadditivity: list[dict] = []
    for parent in nodes:
        children = by_parent.get(parent["node_id"], [])
        parent_effect = parent.get("direct_effect")
        if not children or not isinstance(parent_effect, Mapping):
            continue
        child_effects = [child.get("direct_effect") for child in children]
        if not all(isinstance(effect, Mapping) for effect in child_effects):
            continue
        # This quantity is deliberately not named an effect and does not get
        # a synthetic experiment ID: it is search metadata derived from the
        # referenced real experiments, not another intervention.
        nonadditivity.append({
            "parent_node_id": parent["node_id"],
            "parent_experiment_id": parent_effect["experiment_id"],
            "child_experiment_ids": [effect["experiment_id"] for effect in child_effects],
            "derived_value_nats": (
                parent_effect["delta_nats"]
                - sum(effect["delta_nats"] for effect in child_effects)
            ),
            "provenance": "derived_search_metadata",
            "not_a_measured_effect": True,
        })
    unmeasured = [node["node_id"] for node in nodes
                  if node["measurement_state"] == "unmeasured_budget_exhausted"]
    root = nodes[0]
    return {
        "provenance": "measured_direct_experiments",
        "root_node_id": root["node_id"],
        "nodes": deepcopy(nodes),
        "nonadditivity": nonadditivity,
        "unmeasured_node_ids": unmeasured,
    }


def run_context_dependence_study(
    run: dict,
    sub: Any,
    *,
    target: dict | None = None,
    root_source_ids: Iterable[str] | None = None,
    source_groups: Any = None,
    structural_groups: Any = None,
    passes_requested: int = DEFAULT_PASSES_REQUESTED,
    quick: bool = False,
    clock: Any = None,
) -> dict:
    """Build and execute a bounded Layers 1/2 direct study in one call."""
    return DirectContextDependenceStudy(
        run, sub, target=target, root_source_ids=root_source_ids,
        source_groups=source_groups, structural_groups=structural_groups,
        passes_requested=passes_requested, quick=quick, clock=clock,
    ).run()


# A descriptive alias for callers that prefer the noun phrasing used in the
# artifact title.  Unlike Task 1's historical alias, this always means the
# bounded hierarchy, not a one-arm measurement convenience wrapper.
build_context_dependence_study = run_context_dependence_study


__all__ = [
    "DEFAULT_PASSES_REQUESTED",
    "ContextDependenceStudyError",
    "DirectContextDependenceStudy",
    "build_context_dependence_study",
    "run_context_dependence_study",
]
