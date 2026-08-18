"""Execution for bounded message-context bisect.

This module is orchestration only.  Influence geometry, span resolution and neutral filler remain
authoritative in their existing modules; replay persists ordinary direct child runs; the existing
model diff and Influence Counterfactual target relationship decide the observed criterion.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy

from clozn import schemas
from clozn.receipts.forced import matched_length_neutral_filler
from clozn.replay import influence_counterfactual as counterfactual
from clozn.replay import span_bridge
from clozn.replay.controlled import ExecutionBudget
from clozn.replay.execution_fork import parent_execution_fingerprint
from clozn.replay.replay import replay as replay_run
from clozn.runs.context_bisect import (
    FILLER_RECIPE,
    RESULT_SCHEMA_VERSION,
    ContextBisectInputError,
    _region_id,
    plan_context_bisect,
    split_region,
)


_SAFE_CODES = {
    "stale_parent", "runtime_identity_unavailable", "runtime_identity_mismatch",
    "worker_identity_unavailable", "source_span_unavailable", "span_address_not_found_or_drifted",
    "span_basis_unsupported", "message_basis_unavailable", "recorded_steering_malformed",
    "sampler_provenance_unavailable", "control_generation_failed", "treatment_generation_failed",
    "execution_cancelled", "comparison_unavailable", "inconclusive_children",
    "parent_runtime_identity_unavailable",
}


def _reason(code: str, message: str) -> dict:
    code = code if code in _SAFE_CODES else "context_bisect_unavailable"
    messages = {
        "stale_parent": "the immutable parent changed after planning",
        "runtime_identity_unavailable": "the parent or selected runtime identity is unavailable",
        "runtime_identity_mismatch": "the selected runtime does not match the recorded parent runtime",
        "worker_identity_unavailable": "the selected worker identity is unavailable",
        "source_span_unavailable": "the selected context span could not be resolved",
        "span_address_not_found_or_drifted": "the selected context span is stale or drifted",
        "span_basis_unsupported": "the selected context span has no supported message-list basis",
        "message_basis_unavailable": "the selected context span has no message-list basis",
        "recorded_steering_malformed": "the recorded steering state is malformed",
        "sampler_provenance_unavailable": "the recorded decode regime cannot be reproduced",
        "control_generation_failed": "the unchanged control replay did not produce a child",
        "treatment_generation_failed": "the context treatment replay did not produce a child",
        "execution_cancelled": "context bisect execution was cancelled",
        "comparison_unavailable": "the existing run comparison could not classify the trajectory",
        "inconclusive_children": "both partition children were not available for classification",
        "parent_runtime_identity_unavailable": "the recorded parent runtime identity is unavailable",
    }
    return {"code": code, "message": messages.get(code, message or "context bisect was unavailable")}


def _cancelled(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _max_new(parent: Mapping) -> int:
    return counterfactual._budget(parent)


def _decode(plan: Mapping):
    regime = plan.get("execution", {}).get("decode_regime")
    if not isinstance(regime, Mapping):
        return None
    source = regime.get("source")
    if source == "recorded_greedy":
        return False
    if source == "recorded_fixed_sampling" and isinstance(regime.get("sampling"), Mapping):
        return deepcopy(dict(regime["sampling"]))
    return None


def _changes(plan: Mapping, arm: str, region: Mapping | None = None) -> dict:
    changes = {
        "context_bisect": {
            "search_id": plan["search_id"],
            "arm": arm,
            "root_source_span_id": plan["influence"]["source_span_id"],
            "answer_span_id": plan["influence"]["answer_span_id"],
            "intervention_recipe": FILLER_RECIPE,
        },
    }
    if isinstance(region, Mapping):
        changes["context_bisect"].update({
            "region_id": region["region_id"],
            "root_relative_start": region["root_interval"]["start"],
            "root_relative_end": region["root_interval"]["end"],
        })
    return changes


def _run_arm(parent: Mapping, sub, budget: ExecutionBudget, changes: Mapping, *, messages: list,
             sampling) -> Mapping | None:
    try:
        timeout = budget.start_run()
        # replay's max_new is an output-token budget.  The controlled context arms all use the same
        # parent-derived horizon; the wall-clock timeout is enforced by the shared budget between arms.
        del timeout
        return replay_run(
            dict(parent), dict(changes), sub,
            max_new=_max_new(parent),
            messages_override=deepcopy(messages),
            sampling_override=deepcopy(sampling) if isinstance(sampling, Mapping) else sampling,
        )
    except Exception:
        return None


def _region_key(region: Mapping) -> tuple[int, int, str]:
    interval = region.get("root_interval") or {}
    return (int(region.get("depth", 0)), int(interval.get("start", 0)), str(region.get("region_id", "")))


def _criterion(comparison: Mapping) -> bool:
    return comparison.get("target_relationship") in {"before_target", "within_target"}


def _region_with_state(region: Mapping, state: str, child: Mapping | None, comparison: Mapping | None,
                       reason: Mapping | None = None) -> dict:
    out = deepcopy(dict(region))
    out["state"] = state
    if isinstance(child, Mapping) and child.get("id"):
        out["child_run_id"] = child["id"]
    if isinstance(comparison, Mapping):
        out["criterion"] = {
            "state": "matched" if _criterion(comparison) else "not_matched",
            "target_relationship": comparison.get("target_relationship", "unavailable"),
        }
        out["comparison"] = deepcopy(dict(comparison))
    if isinstance(reason, Mapping):
        out["reason"] = deepcopy(dict(reason))
    return out


def _comparison(baseline: Mapping, child: Mapping | None, target: Mapping) -> dict:
    try:
        projection, _diff = counterfactual._comparison(baseline, child, target)
        return projection
    except Exception:
        return {"state": "unavailable", "target_relationship": "unavailable"}


def _control_state(reproduction: Mapping, comparison: Mapping) -> str:
    relationship = comparison.get("target_relationship")
    if relationship in {"after_target", "no_divergence"}:
        if reproduction.get("state") == "exact_token_and_text":
            return "exact_full_reproduction"
        return "reproduced_through_target"
    if relationship in {"before_target", "within_target"}:
        return "not_reproduced_through_target"
    return "comparison_unavailable"


def _root_region(plan: Mapping) -> dict:
    return deepcopy(dict(plan["root_region"]))


def _treatment_messages(messages: list, source_span: Mapping, region: Mapping) -> list:
    root = region["root_interval"]
    start = int(source_span["start"]) + int(root["start"])
    end = int(source_span["start"]) + int(root["end"])
    selected = dict(source_span)
    selected["start"], selected["end"] = start, end
    return span_bridge.excise_spans(
        deepcopy(messages), [selected], replacement=matched_length_neutral_filler)


def _base_result(parent: Mapping, plan: Mapping, *, status: str = "unavailable") -> dict:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "search_id": plan["search_id"],
        "parent_run_id": parent["id"],
        "parent_execution_fingerprint": plan["parent_execution_fingerprint"],
        "influence": deepcopy(plan.get("influence", {})),
        "intervention": deepcopy(plan["intervention"]),
        "target": deepcopy(plan.get("target", {})),
        "span_resolution": deepcopy(plan.get("span_resolution", {})),
        "root_basis_sha256": (plan.get("span_resolution") or {}).get("basis_sha256"),
        "root_span_sha256": (plan.get("span_resolution") or {}).get("span_sha256"),
        "control": {"state": "not_attempted"},
        "execution": {
            "status": status,
            "children_created": 0,
            "fidelity": "controlled_regeneration",
            "requires_full_reprefill": True,
            "decode_regime": deepcopy((plan.get("execution") or {}).get("decode_regime", {})),
            "order": "breadth_first",
        },
        "search": deepcopy(plan["search"]),
        "regions": [],
        "findings": [],
        "coverage": {"state": "inconclusive"},
        "summary": {
            "state": "inconclusive",
            "generated_runs": 0,
            "regions_tested": 0,
            "retained_regions": 0,
            "not_retained_regions": 0,
            "deepest_tested_depth": 0,
            "minimality_scope": "tested_partition_only",
        },
    }


def _finalize(result: dict) -> dict:
    schemas.validate(result, RESULT_SCHEMA_VERSION)
    return result


def _mark_terminal(result: dict, region: dict, reason: str) -> None:
    region["terminal_reason"] = reason
    if region.get("state") == "retained":
        result["findings"].append({
            "type": "smallest_tested_retained_region",
            "region_id": region["region_id"],
            "depth": region["depth"],
            "code_points": region["code_points"],
            "minimality_scope": "tested_partition_only",
            "terminal_reason": reason,
            **({"child_run_id": region["child_run_id"]} if region.get("child_run_id") else {}),
        })


def _summary(result: dict) -> None:
    regions = result["regions"]
    tested = [r for r in regions if r.get("state") != "not_tested"]
    retained = [r for r in regions if r.get("state") == "retained"]
    not_retained = [r for r in regions if r.get("state") == "not_retained"]
    depths = [r.get("depth", 0) for r in tested]
    terminal = [r for r in retained if r.get("terminal_reason")]
    result["summary"].update({
        "state": result.get("coverage", {}).get("state", "inconclusive"),
        "generated_runs": _created_count(result),
        "regions_tested": len(tested),
        "retained_regions": len(retained),
        "not_retained_regions": len(not_retained),
        "deepest_tested_depth": max(depths, default=0),
        "terminal_retained_regions": len(terminal),
        "distributed_regions": sum(1 for r in regions if r.get("split", {}).get("classification") == "distributed_within_region"),
        "budget_limited_regions": sum(1 for r in regions if r.get("terminal_reason") == "budget_limit"),
        "inconclusive_regions": sum(1 for r in regions if r.get("state") == "unavailable"),
    })
    if terminal:
        result["summary"]["smallest_tested_retained_region_chars"] = min(r["code_points"] for r in terminal)


def _created_count(result: Mapping) -> int:
    ids = set()
    control = result.get("control") if isinstance(result, Mapping) else None
    if isinstance(control, Mapping) and isinstance(control.get("run_id"), str):
        ids.add(control["run_id"])
    for region in result.get("regions", []) if isinstance(result, Mapping) else []:
        if isinstance(region, Mapping) and isinstance(region.get("child_run_id"), str):
            ids.add(region["child_run_id"])
    return len(ids)


def _budget_state(budget: ExecutionBudget, runs_needed: int = 1) -> str:
    if budget.runs_used + runs_needed > budget.max_runs:
        return "budget_limited"
    return "time_limited"


def execute_context_bisect(
    parent_run: Mapping,
    sub,
    request=None,
    *,
    runtime_identity=None,
    worker_identity=None,
    reload_parent=None,
    cancel_check=None,
    plan=None,
) -> dict:
    """Execute one bounded breadth-first search over sibling controlled replays."""
    if plan is None and isinstance(request, Mapping) and request.get("schema_version") == "clozn.context-bisect-plan.v1":
        # Accepting a preview as the positional third argument is convenient for direct callers, but
        # still rebuild the logical request below so the preview is never execution authority.
        plan = request
        plan_search = request.get("search") if isinstance(request.get("search"), Mapping) else {}
        request = {
            "influence": {
                "source_span_id": request.get("influence", {}).get("source_span_id"),
                "answer_span_id": request.get("influence", {}).get("answer_span_id"),
            },
            "recipe": plan_search.get("recipe", "neutralize_bisect_v1"),
            "max_depth": plan_search.get("max_depth", 3),
            "max_runs": plan_search.get("max_runs", 8),
            "max_seconds": plan_search.get("max_seconds", 120),
            "min_region_chars": plan_search.get("min_region_chars", 32),
        }
    current_plan = plan_context_bisect(parent_run, request=request) if request is not None else plan_context_bisect(parent_run)
    result = _base_result(parent_run, current_plan)
    if current_plan["execution"]["state"] != "ready":
        result["coverage"] = {"state": "inconclusive", "reason": current_plan["execution"].get("reason", "plan_unavailable")}
        result["reasons"] = [_reason(current_plan["execution"].get("reason", "context_bisect_unavailable"), "")]
        _summary(result)
        return _finalize(result)

    runtime_ok, runtime_reason = counterfactual._runtime_match(parent_run, runtime_identity, worker_identity)
    result["execution"]["runtime_identity_match"] = bool(runtime_ok)
    if not runtime_ok:
        result["reasons"] = [runtime_reason or _reason("runtime_identity_unavailable", "")]
        _summary(result)
        return _finalize(result)

    current_parent = reload_parent(parent_run["id"]) if callable(reload_parent) else parent_run
    if not isinstance(current_parent, Mapping) or parent_execution_fingerprint(current_parent) != current_plan["parent_execution_fingerprint"]:
        result["execution"]["status"] = "stale_parent"
        result["coverage"] = {"state": "stale_parent"}
        result["reasons"] = [_reason("stale_parent", "")]
        return _finalize(result)
    parent_run = current_parent

    resolved = span_bridge.resolve_span_address(dict(parent_run), current_plan["influence"]["source_span_id"])
    source_span = resolved.get("span") if isinstance(resolved, Mapping) and resolved.get("ok") else None
    messages = parent_run.get("messages")
    if not isinstance(source_span, Mapping) or not isinstance(messages, list):
        result["reasons"] = [_reason("source_span_unavailable", "")]
        _summary(result)
        return _finalize(result)
    sampling = _decode(current_plan)
    if sampling is None:
        result["reasons"] = [_reason("sampler_provenance_unavailable", "")]
        _summary(result)
        return _finalize(result)

    budget = ExecutionBudget(
        current_plan["search"]["max_runs"], current_plan["search"]["max_seconds"])
    result["execution"]["budget"] = budget.snapshot()
    result["execution"]["status"] = "running"

    def stop_result(state: str, code: str | None = None):
        if state == "cancelled":
            result["execution"]["status"] = "partial_cancelled" if _created_count(result) else "cancelled"
        elif state == "stale_parent":
            result["execution"]["status"] = "stale_parent"
        elif result["execution"].get("status") == "running":
            result["execution"]["status"] = "completed"
        result["coverage"] = {"state": state}
        if code:
            result.setdefault("reasons", []).append(_reason(code, ""))
        _summary(result)
        result["execution"]["children_created"] = _created_count(result)
        result["execution"]["budget"] = budget.snapshot()
        return _finalize(result)

    if _cancelled(cancel_check):
        return stop_result("cancelled", "execution_cancelled")
    if not budget.can_start():
        return stop_result(_budget_state(budget))

    control_changes = {
        "context_bisect": {
            "search_id": current_plan["search_id"],
            "arm": "control",
            "root_source_span_id": current_plan["influence"]["source_span_id"],
            "answer_span_id": current_plan["influence"]["answer_span_id"],
        },
    }
    control = _run_arm(parent_run, sub, budget, control_changes, messages=messages, sampling=sampling)
    if not isinstance(control, Mapping) or not control.get("id"):
        result["execution"]["status"] = "failed"
        return stop_result("inconclusive", "control_generation_failed")
    result["control"] = {"state": "completed", "run_id": control["id"]}
    result["execution"]["children_created"] = _created_count(result)
    control_comparison = _comparison(parent_run, control, current_plan["target"])
    reproduction = counterfactual._reproduction(parent_run, control, control_comparison)
    result["control"]["reproduction"] = {
        "state": _control_state(reproduction, control_comparison),
        "parent_vs_control": deepcopy(control_comparison),
    }
    if result["control"]["reproduction"]["state"] not in {"exact_full_reproduction", "reproduced_through_target"}:
        result["execution"]["status"] = "completed"
        return stop_result("control_not_reproduced")

    root = _root_region(current_plan)
    root_child = None
    if _cancelled(cancel_check):
        return stop_result("cancelled", "execution_cancelled")
    if not budget.can_start():
        return stop_result("budget_limited")
    root_changes = _changes({**current_plan, "_parent": parent_run}, "treatment", root)
    root_messages = _treatment_messages(messages, source_span, root)
    root_child = _run_arm(parent_run, sub, budget, root_changes, messages=root_messages, sampling=sampling)
    if not isinstance(root_child, Mapping) or not root_child.get("id"):
        result["regions"].append(_region_with_state(root, "unavailable", None, None, _reason("treatment_generation_failed", "")))
        result["execution"]["status"] = "failed"
        return stop_result("inconclusive", "treatment_generation_failed")
    root_comparison = _comparison(control, root_child, current_plan["target"])
    root_state = "retained" if _criterion(root_comparison) else "not_retained"
    root_result = _region_with_state(root, root_state, root_child, root_comparison)
    result["regions"].append(root_result)
    if root_state != "retained":
        result["execution"]["status"] = "completed"
        result["coverage"] = {"state": "root_not_reproduced"}
        _summary(result)
        result["execution"]["children_created"] = _created_count(result)
        result["execution"]["budget"] = budget.snapshot()
        return _finalize(result)

    frontier = deque([(root_result, root)])
    coverage_state = "complete_within_limits"
    stop_all = False
    message_text = messages[source_span["message_index"]]["content"]
    while frontier and not stop_all:
        frontier = deque(sorted(frontier, key=lambda item: _region_key(item[0])))
        region_result, region_internal = frontier.popleft()
        if callable(reload_parent):
            latest_parent = reload_parent(parent_run["id"])
            if (not isinstance(latest_parent, Mapping)
                    or parent_execution_fingerprint(latest_parent) != current_plan["parent_execution_fingerprint"]):
                coverage_state = "stale_parent"
                result["reasons"] = [_reason("stale_parent", "")]
                stop_all = True
                break
        depth = int(region_internal["depth"])
        if depth >= int(current_plan["search"]["max_depth"]):
            _mark_terminal(result, region_result, "depth_limit")
            continue
        interval = region_internal["root_interval"]
        split = split_region(message_text[source_span["start"]:source_span["end"]], interval["start"], interval["end"], current_plan["search"]["min_region_chars"])
        if split is None:
            _mark_terminal(result, region_result, "unsplittable")
            continue
        if not budget.can_start(2):
            terminal_budget_state = _budget_state(budget, 2)
            _mark_terminal(result, region_result, "budget_limit" if terminal_budget_state == "budget_limited" else "time_limit")
            coverage_state = terminal_budget_state
            continue
        if _cancelled(cancel_check):
            coverage_state = "cancelled"
            stop_all = True
            _mark_terminal(result, region_result, "cancelled")
            break

        left_internal = {"start": split[0]["start"], "end": split[0]["end"], "depth": depth + 1}
        right_internal = {"start": split[1]["start"], "end": split[1]["end"], "depth": depth + 1}
        parent_region_id = region_result["region_id"]
        message_start = source_span["start"]
        left = _region_projection_for_result(
            current_plan["search_id"], left_internal, message_start,
            source_span["message_index"], message_text, parent_region_id,
        )
        right = _region_projection_for_result(current_plan["search_id"], right_internal, message_start,
                                              source_span["message_index"], message_text, parent_region_id)
        left_child = _run_arm(
            parent_run, sub, budget,
            _changes({**current_plan, "_parent": parent_run}, "treatment", left),
            messages=_treatment_messages(messages, source_span, left), sampling=sampling)
        left_result = _region_with_state(
            left,
            "unavailable",
            left_child,
            None,
            None if isinstance(left_child, Mapping) else _reason("treatment_generation_failed", ""),
        )
        result["regions"].append(left_result)
        if _cancelled(cancel_check):
            coverage_state = "cancelled"
            right_result = _region_with_state(right, "not_tested", None, None, _reason("execution_cancelled", ""))
            result["regions"].append(right_result)
            region_result["split"] = {"state": "incomplete", "classification": "incomplete_pair_cancelled", "left_region_id": left["region_id"], "right_region_id": right["region_id"]}
            stop_all = True
            break
        if _cancelled(cancel_check):
            coverage_state = "cancelled"
            stop_all = True
            break
        right_child = _run_arm(
            parent_run, sub, budget,
            _changes({**current_plan, "_parent": parent_run}, "treatment", right),
            messages=_treatment_messages(messages, source_span, right), sampling=sampling)
        if isinstance(left_child, Mapping) and left_child.get("id"):
            left_comparison = _comparison(control, left_child, current_plan["target"])
            left_result.update(_region_with_state(left, "retained" if _criterion(left_comparison) else "not_retained", left_child, left_comparison))
        right_result = _region_with_state(
            right,
            "unavailable",
            right_child,
            None,
            None if isinstance(right_child, Mapping) else _reason("treatment_generation_failed", ""),
        )
        result["regions"].append(right_result)
        if isinstance(right_child, Mapping) and right_child.get("id"):
            right_comparison = _comparison(control, right_child, current_plan["target"])
            right_result.update(_region_with_state(right, "retained" if _criterion(right_comparison) else "not_retained", right_child, right_comparison))
        left_ok, right_ok = left_result.get("state") == "retained", right_result.get("state") == "retained"
        if not left_result.get("child_run_id") or not right_result.get("child_run_id"):
            classification = "inconclusive"
            coverage_state = "inconclusive"
            result["findings"].append({
                "type": "inconclusive_region",
                "region_id": region_result["region_id"],
                "minimality_scope": "tested_partition_only",
            })
        elif left_ok and right_ok:
            classification = "multiple_retained_regions"
        elif left_ok:
            classification = "localized_left"
        elif right_ok:
            classification = "localized_right"
        else:
            classification = "distributed_within_region"
        region_result["split"] = {
            "state": "tested" if classification != "inconclusive" else "inconclusive",
            "classification": classification,
            "left_region_id": left["region_id"],
            "right_region_id": right["region_id"],
        }
        if classification == "distributed_within_region":
            result["findings"].append({"type": classification, "region_id": region_result["region_id"], "minimality_scope": "tested_partition_only"})
        elif classification == "multiple_retained_regions":
            result["findings"].append({"type": classification, "region_id": region_result["region_id"], "minimality_scope": "tested_partition_only"})
            frontier.extend([(left_result, left), (right_result, right)])
        elif classification == "localized_left":
            frontier.append((left_result, left))
        elif classification == "localized_right":
            frontier.append((right_result, right))

    if coverage_state == "complete_within_limits" and budget.stop_reason == "budget_exhausted":
        coverage_state = "budget_limited"
    if coverage_state == "cancelled":
        result["execution"]["status"] = "partial_cancelled" if _created_count(result) else "cancelled"
    elif coverage_state == "stale_parent":
        result["execution"]["status"] = "stale_parent"
    else:
        result["execution"]["status"] = "completed"
    result["coverage"] = {"state": coverage_state}
    _summary(result)
    result["execution"]["children_created"] = _created_count(result)
    result["execution"]["budget"] = budget.snapshot()
    return _finalize(result)


def _region_projection_for_result(search_id: str, region: Mapping, message_start: int,
                                  message_index: int, message_text: str, parent_region_id: str) -> dict:
    start, end = int(region["start"]), int(region["end"])
    return {
        "region_id": _region_id(search_id, start, end),
        "parent_region_id": parent_region_id,
        "depth": int(region["depth"]),
        "root_interval": {"start": start, "end": end, "unit": "unicode_code_points", "interval": "half_open"},
        "message_interval": {"start": message_start + start, "end": message_start + end, "unit": "unicode_code_points", "interval": "half_open"},
        "code_points": end - start,
        "sha256": __import__("clozn.runs.influence_geometry", fromlist=["text_sha256"]).text_sha256(
            message_text[message_start + start:message_start + end]),
    }


__all__ = ["execute_context_bisect"]
