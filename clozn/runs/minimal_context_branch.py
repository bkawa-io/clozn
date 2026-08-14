"""Explicit user branches motivated by a Minimal Context result.

This module is intentionally separate from the bounded proof runner.  Proof
arms never call it and never become Runs.  Only an explicit action supplies a
validated result ID and source selection, after which the normal replay seam
records one auditable child and the canonical run diff is returned.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from clozn.analysis.model_diff import diff_runs
from clozn.replay.replay import replay as replay_run
from clozn.replay.span_bridge import ContextReceiptSourceResolutionError, resolve_context_receipt_source_set
from clozn.runs.minimal_context import _manifest_digest, validate_minimal_context_result


SCHEMA = "clozn.minimal-context-branch.v1"
_ACTION_ALIASES = {
    "remove": "remove_and_branch",
    "remove_and_branch": "remove_and_branch",
    "add_back": "add_back_and_branch",
    "add_back_and_branch": "add_back_and_branch",
    "only": "branch_with_only",
    "branch_with_only": "branch_with_only",
}


class MinimalContextBranchError(ValueError):
    """A malformed, stale, or unavailable explicit branch request."""

    def __init__(self, message: str, *, code: str = "minimal_context_branch_invalid", status: int = 409):
        super().__init__(message)
        self.code = code
        self.status = status


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False, default=str,
    ).encode("utf-8")).hexdigest()


def _parent_binding(parent: Mapping) -> str:
    return _digest({
        "id": parent.get("id"),
        "messages": parent.get("messages"),
        "assembled_messages": parent.get("assembled_messages"),
        "context_receipt": parent.get("context_receipt"),
        "context_units": parent.get("context_units"),
        "identity": parent.get("identity"),
        "behavior": parent.get("behavior"),
        "meta": parent.get("meta"),
    })


def _ordered_unique(values: Any, field: str) -> list[str]:
    if isinstance(values, str) or not isinstance(values, Iterable):
        raise MinimalContextBranchError(f"{field} must be a list of source IDs", code="invalid_source_selection", status=400)
    values = list(values)
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise MinimalContextBranchError(f"{field} must contain non-empty source IDs", code="invalid_source_selection", status=400)
    if len(set(values)) != len(values):
        raise MinimalContextBranchError(f"{field} must not contain duplicate source IDs", code="invalid_source_selection", status=400)
    return values


def _result_from_parent(parent: Mapping, result_id: str) -> dict:
    results = parent.get("minimal_context_results")
    result = results.get(result_id) if isinstance(results, Mapping) else None
    if not isinstance(result, Mapping):
        raise MinimalContextBranchError("the Minimal Context result was not found on this run",
                                         code="minimal_context_result_not_found", status=404)
    result = deepcopy(dict(result))
    try:
        validate_minimal_context_result(result)
    except Exception as exc:
        raise MinimalContextBranchError(
            f"the Minimal Context result is not a valid proof artifact: {exc}",
            code="minimal_context_result_invalid", status=409,
        ) from exc
    if result.get("run_id") != parent.get("id") or result.get("result_id") != result_id:
        raise MinimalContextBranchError("the Minimal Context result is bound to another run",
                                         code="minimal_context_result_stale", status=409)
    if result.get("status") != "found" or not isinstance(result.get("candidate"), Mapping):
        raise MinimalContextBranchError("only a found Minimal Context result can motivate a branch",
                                         code="minimal_context_result_unavailable", status=409)
    return result


def _target_set(result: Mapping, action: str, selected: list[str]) -> tuple[list[str], list[str], list[str]]:
    universe = result["source_universe"]["source_ids"]
    candidate = result["candidate"]["retained_source_ids"]
    universe_set = set(universe)
    selected_set = set(selected)
    unknown = selected_set - universe_set
    if unknown:
        raise MinimalContextBranchError(
            "selected source IDs are outside the Minimal Context result universe: " + ", ".join(sorted(unknown)),
            code="source_selection_outside_universe", status=400,
        )
    candidate_set = set(candidate)
    if action == "remove_and_branch":
        if not selected_set.issubset(candidate_set):
            raise MinimalContextBranchError(
                "REMOVE + BRANCH requires selected sources retained by the motivating candidate",
                code="source_not_retained", status=400,
            )
        target = candidate_set - selected_set
    elif action == "add_back_and_branch":
        if not selected_set.isdisjoint(candidate_set):
            raise MinimalContextBranchError(
                "ADD BACK + BRANCH requires selected sources omitted by the motivating candidate",
                code="source_already_retained", status=400,
            )
        target = candidate_set | selected_set
    elif action == "branch_with_only":
        target = selected_set
    else:  # pragma: no cover - aliases are normalized before this helper
        raise MinimalContextBranchError("unsupported Minimal Context branch action", code="invalid_action", status=400)
    ordered_target = [source_id for source_id in universe if source_id in target]
    removed = [source_id for source_id in universe if source_id not in target]
    return ordered_target, removed, list(candidate)


def _resolve_target(parent: Mapping, universe: list[str], removed: list[str]) -> dict:
    if not universe:
        raise MinimalContextBranchError("the Minimal Context result has no source universe",
                                         code="minimal_context_universe_empty", status=409)
    try:
        # The strict resolver requires a non-empty deletion.  For an unchanged
        # target, resolve one source only to obtain and validate the clean
        # ``basis_messages`` and complete current catalog; the branch uses the
        # basis, not the one-source deletion.
        resolved = resolve_context_receipt_source_set(parent, removed or [universe[0]])
    except ContextReceiptSourceResolutionError as exc:
        raise MinimalContextBranchError(
            f"the current Context Receipt no longer resolves the selected sources: {exc}",
            code="source_drift", status=409,
        ) from exc
    available = resolved.get("available_source_ids")
    if not isinstance(available, list) or any(source_id not in available for source_id in universe):
        raise MinimalContextBranchError(
            "the current Context Receipt no longer contains the complete result universe",
            code="source_drift", status=409,
        )
    if not isinstance(resolved.get("basis_messages"), list) or not isinstance(resolved.get("messages"), list):
        raise MinimalContextBranchError("the strict source resolver returned no message override",
                                         code="source_drift", status=409)
    return resolved


def plan_minimal_context_branch(parent: Mapping, result: Mapping, *, action: str,
                                source_ids: Iterable[str]) -> dict:
    """Build a model-free, source-bound user branch plan.

    The result and source resolver are both rechecked here.  The returned
    message override is detached and is consumed only by the normal replay
    executor; no model call or Run mutation happens during planning.
    """
    if not isinstance(parent, Mapping) or not isinstance(parent.get("id"), str) or not parent["id"]:
        raise MinimalContextBranchError("parent run must carry a non-empty id", code="invalid_parent", status=400)
    if not isinstance(result, Mapping):
        raise MinimalContextBranchError("result must be a Minimal Context object", code="invalid_result", status=400)
    result = deepcopy(dict(result))
    if result.get("run_id") != parent["id"]:
        raise MinimalContextBranchError("result does not belong to the requested parent",
                                         code="minimal_context_result_stale", status=409)
    try:
        validate_minimal_context_result(result)
    except Exception as exc:
        raise MinimalContextBranchError(f"result is not a valid Minimal Context artifact: {exc}",
                                         code="minimal_context_result_invalid", status=409) from exc
    normalized_action = _ACTION_ALIASES.get(action)
    if normalized_action is None:
        raise MinimalContextBranchError(
            "action must be remove_and_branch, add_back_and_branch, or branch_with_only",
            code="invalid_action", status=400,
        )
    selected = _ordered_unique(source_ids, "source_ids")
    universe = list(result["source_universe"]["source_ids"])
    target, removed, candidate = _target_set(result, normalized_action, selected)

    manifest = parent.get("context_units")
    expected_manifest = result["source_universe"].get("context_units_manifest_sha256")
    if not isinstance(manifest, Mapping) or not isinstance(expected_manifest, str):
        raise MinimalContextBranchError("the Minimal Context result has no current manifest binding",
                                         code="source_drift", status=409)
    current_manifest = _manifest_digest(manifest, tuple(universe))
    if current_manifest != expected_manifest:
        raise MinimalContextBranchError(
            "the Context Units manifest changed after the Minimal Context result was recorded",
            code="source_drift", status=409,
        )

    resolved = _resolve_target(parent, universe, removed)
    ranges = []
    by_id = {item.get("source_id"): item for item in resolved.get("sources", []) if isinstance(item, Mapping)}
    for source_id in removed:
        source = by_id.get(source_id)
        if source is None:
            raise MinimalContextBranchError("the strict resolver omitted a selected source", code="source_drift", status=409)
    if removed:
        ranges = deepcopy(resolved.get("exact_removed_ranges") or [])
        messages_override = deepcopy(resolved["messages"])
    else:
        messages_override = deepcopy(resolved["basis_messages"])
    intervention = {
        "operator": "delete_source",
        "action": normalized_action,
        "selected_source_ids": selected,
        "candidate_retained_source_ids": candidate,
        "target_retained_source_ids": target,
        "removed_source_ids": removed,
        "exact_removed_ranges": ranges,
        "source_basis": resolved.get("basis"),
        "basis_digest": resolved.get("basis_digest"),
        "intervened_context_digest": resolved.get("intervened_context_digest") if removed else resolved.get("basis_digest"),
    }
    return {
        "schema_version": SCHEMA,
        "state": "ready",
        "parent_run_id": parent["id"],
        "result_id": result["result_id"],
        "result_preservation": result["preservation"]["kind"],
        "result_certificate_kind": (result.get("certificate") or {}).get("kind"),
        "parent_binding_sha256": _parent_binding(parent),
        "context_units_manifest_sha256": expected_manifest,
        "search_universe_id": result["source_universe"].get("search_universe_id"),
        "intervention": intervention,
        "execution": {"requires_generation": True, "generation_calls": 1, "messages_override": messages_override},
    }


def execute_minimal_context_branch(parent: Mapping, result: Mapping, sub, *, action: str,
                                   source_ids: Iterable[str], plan: Mapping | None = None,
                                   reload_parent=None, max_new: int | None = None) -> dict:
    """Revalidate and generate one explicit child; failed generation is non-mutating."""
    current_parent = reload_parent(parent.get("id")) if callable(reload_parent) else parent
    if not isinstance(current_parent, Mapping):
        raise MinimalContextBranchError("the parent could not be reloaded", code="parent_unavailable", status=409)
    current = plan_minimal_context_branch(current_parent, result, action=action, source_ids=source_ids)
    if plan is not None:
        if plan.get("parent_binding_sha256") != current["parent_binding_sha256"]:
            raise MinimalContextBranchError("the parent changed after branch planning", code="source_drift", status=409)
        if plan.get("intervention") != current["intervention"]:
            raise MinimalContextBranchError("the selected source intervention changed after planning",
                                             code="source_drift", status=409)
    changes = {
        "minimal_context_branch": {
            "result_id": current["result_id"],
            "result_preservation": current["result_preservation"],
            "result_certificate_kind": current.get("result_certificate_kind"),
            **deepcopy(current["intervention"]),
        },
    }
    behavior = current_parent.get("behavior")
    active_dials = behavior.get("active_dials") if isinstance(behavior, Mapping) else None
    if isinstance(active_dials, Mapping) and active_dials:
        changes["behavior_overrides"] = deepcopy(dict(active_dials))
    kwargs: dict[str, Any] = {"messages_override": deepcopy(current["execution"]["messages_override"])}
    if isinstance(max_new, int) and not isinstance(max_new, bool) and max_new > 0:
        kwargs["max_new"] = max_new
    child = replay_run(dict(current_parent), changes, sub, **kwargs)
    if not isinstance(child, Mapping) or not child.get("id"):
        return {
            "schema_version": SCHEMA,
            "state": "failed",
            "parent_run_id": current["parent_run_id"],
            "result_id": current["result_id"],
            "intervention": deepcopy(current["intervention"]),
            "reason": "branch_generation_failed",
        }
    comparison = diff_runs(dict(current_parent), dict(child))
    return {
        "schema_version": SCHEMA,
        "state": "completed",
        "parent_run_id": current["parent_run_id"],
        "child_run_id": child["id"],
        "result_id": current["result_id"],
        "result_preservation": current["result_preservation"],
        "result_certificate_kind": current.get("result_certificate_kind"),
        "intervention": deepcopy(current["intervention"]),
        "comparison": comparison,
        "compare_path": f"#/compare/{current['parent_run_id']}/{child['id']}",
    }


__all__ = [
    "MinimalContextBranchError",
    "execute_minimal_context_branch",
    "plan_minimal_context_branch",
]
