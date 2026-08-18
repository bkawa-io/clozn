"""Execution dispatcher for the explicit Test This action.

There is intentionally very little execution logic here.  The module resolves the pure plan again,
then hands the request to the canonical Time Travel recipe, Branch Fan orchestrator, or exact
Execution Fork executor.  Force-token generation is evidence-first; child materialization is a
separate explicit operation. Child persistence, exactness proof, checkpoint handling, and diff
calculation remain owned by those lower-level primitives.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from clozn import schemas
from clozn.runs.test_this import (
    RESULT_SCHEMA_VERSION,
    TestThisInputError,
    build_test_this_plan,
    resolve_recorded_alternative,
)


def _reason(code: str, message: str) -> dict:
    return {"code": str(code or "test_unavailable"), "message": str(message or "test unavailable")}


def _public_reasons(raw) -> list[dict]:
    out = []
    for item in raw or []:
        if not isinstance(item, Mapping):
            continue
        code = item.get("code")
        if isinstance(code, str) and code:
            out.append(_reason(code, item.get("message") or "the requested test was unavailable"))
    return out


def _single_artifact(child: Mapping | None, *, outcome: str, execution_id: str | None = None) -> dict:
    receipt = child.get("execution_fork") if isinstance(child, Mapping) else None
    artifact = {
        "schema": "clozn.execution-fork.v1",
        "classification": outcome,
    }
    if execution_id:
        artifact["execution_id"] = execution_id
    if isinstance(receipt, Mapping):
        artifact["receipt"] = deepcopy(dict(receipt))
    return artifact


def _single_result(parent: Mapping, plan: Mapping, child_result: Mapping | None) -> dict:
    """Project one existing fork result without embedding its full child run."""
    from clozn.replay.branch_fan import comparison_projection

    operation = plan["resolved_test"]["operation"]
    if child_result is None:
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent["id"],
            "selection": deepcopy(dict(plan["selection"])),
            "test": deepcopy(dict(plan["test"])),
            "operation": operation,
            "outcome": "failed",
            "result": {"reasons": [_reason("child_result_unavailable", "the fork returned no child result")]},
            "artifact": None,
            "comparison": None,
        }
        schemas.validate(document, RESULT_SCHEMA_VERSION)
        return document

    outcome = child_result.get("outcome")
    if outcome in {"exact_execution_fork", "reconstructed_replay"} and child_result.get("id"):
        comparison = comparison_projection(parent, child_result)
        result = {
            "child_run_id": child_result["id"],
            "backend_outcome": outcome,
            "reasons": _public_reasons(child_result.get("reasons")),
        }
        for key in ("exactness", "unchanged_control", "unavoidable_differences", "execution_fork_execution_id"):
            if key in child_result:
                result[key] = deepcopy(child_result[key])
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent["id"],
            "selection": deepcopy(dict(plan["selection"])),
            "test": deepcopy(dict(plan["test"])),
            "operation": operation,
            "outcome": "completed",
            "result": result,
            "artifact": _single_artifact(
                child_result,
                outcome=outcome,
                execution_id=child_result.get("execution_fork_execution_id"),
            ),
            "comparison": comparison,
            "child_run_id": child_result["id"],
        }
    else:
        reasons = _public_reasons(child_result.get("reasons")) or [
            _reason("test_unavailable", "the selected fork could not be produced")
        ]
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent["id"],
            "selection": deepcopy(dict(plan["selection"])),
            "test": deepcopy(dict(plan["test"])),
            "operation": operation,
            "outcome": "unavailable",
            "result": {
                "reasons": reasons,
                **({"exactness": deepcopy(child_result["exactness"])}
                   if isinstance(child_result.get("exactness"), Mapping) else {}),
                **({"unchanged_control": deepcopy(child_result["unchanged_control"])}
                   if isinstance(child_result.get("unchanged_control"), Mapping) else {}),
            },
            "artifact": _single_artifact(
                child_result,
                outcome=outcome or "unavailable",
                execution_id=child_result.get("execution_fork_execution_id"),
            ),
            "comparison": None,
        }
    schemas.validate(document, RESULT_SCHEMA_VERSION)
    return document


def _execute_force_token(parent: Mapping, sub, plan: Mapping, *, runtime_identity,
                         worker_identity, reload_parent=None, cancel_check=None) -> dict:
    from clozn.experiments.persistence import ObservationStore
    from clozn.recipes.time_travel import run_time_travel

    selection = plan["selection"]
    candidate = resolve_recorded_alternative(
        parent,
        selection["position"],
        alternative_rank=plan["test"].get("alternative_rank"),
        token_id=plan["test"].get("token_id"),
    )
    # The resolved piece is read from the immutable parent's recorded alternatives. The canonical
    # recipe owns StateRef/Generate resolution and returns a durable GeneratedObservation reference;
    # this dispatcher never creates a child Run.
    try:
        travel = run_time_travel(
            parent,
            position=selection["position"],
            token_id=candidate.get("token_id"),
            token_piece=None if candidate.get("token_id") is not None else candidate.get("piece"),
            max_new=32,
            policy="exact_preferred",
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            substrate=sub,
            run_loader=reload_parent,
            observation_store=ObservationStore(),
            cancel=cancel_check,
        )
    except Exception as exc:
        return _time_travel_result(parent, plan, None, reason=_reason(
            "time_travel_execution_failed", str(exc)))
    return _time_travel_result(parent, plan, travel)


def _time_travel_result(parent: Mapping, plan: Mapping, travel, *, reason: Mapping | None = None) -> dict:
    """Wrap the canonical TimeTravelResult in the Test This envelope without child persistence."""
    operation = plan["resolved_test"]["operation"]
    projection = travel.to_dict() if travel is not None else None
    status = getattr(travel, "status", "failed") if travel is not None else "failed"
    outcome = "completed" if status == "completed" else ("failed" if status == "failed" else "unavailable")
    reasons = [_reason(reason.get("code"), reason.get("message"))] if isinstance(reason, Mapping) else []
    if isinstance(projection, Mapping):
        code = projection.get("reason_code") or projection.get("diagnostics", {}).get("reason_code")
        message = projection.get("reason") or projection.get("diagnostics", {}).get("message")
        if outcome != "completed" and isinstance(code, str) and code:
            reasons = [_reason(code, message or "the time-travel operation was unavailable")]
    document = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "run_id": parent["id"],
        "selection": deepcopy(dict(plan["selection"])),
        "test": deepcopy(dict(plan["test"])),
        "operation": operation,
        "outcome": outcome,
        "result": {
            "time_travel": deepcopy(projection) if isinstance(projection, Mapping) else {},
            "experiment_id": getattr(travel, "experiment_id", None),
            "arm_id": getattr(travel, "arm_id", None),
            "observation_id": getattr(travel, "observation_id", None),
            "reasons": reasons,
        },
        "artifact": {
            "schema": "clozn.time-travel-result.v1",
            "result": deepcopy(projection) if isinstance(projection, Mapping) else None,
        },
        "comparison": None,
    }
    schemas.validate(document, RESULT_SCHEMA_VERSION)
    return document


def _execute_sampling(parent: Mapping, sub, plan: Mapping, *, runtime_identity,
                      worker_identity, reload_parent=None, cancel_check=None) -> dict:
    from clozn.replay.execution_fork import (
        capture_exact_force_token_context,
        execute_exact_force_token,
        plan_exact_force_token,
    )

    engine = getattr(sub, "engine", None) if sub is not None else None
    if engine is None or not isinstance(runtime_identity, Mapping) or not isinstance(worker_identity, Mapping):
        return _single_result(parent, plan, {
            "outcome": "unavailable",
            "reasons": [_reason("exact_execution_unavailable", "exact sampler execution is unavailable")],
        })
    if callable(cancel_check) and cancel_check():
        return _single_result(parent, plan, {
            "outcome": "unavailable",
            "reasons": [_reason("execution_cancelled", "the sampler test was cancelled before execution")],
        })

    try:
        capture = capture_exact_force_token_context(
            parent, engine, runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
    except Exception:
        capture = {"status": "ineligible", "reason": _reason(
            "checkpoint_capture_unavailable", "an exact checkpoint could not be captured")}
    if capture.get("status") != "available" or not isinstance(capture.get("checkpoint_reference"), Mapping):
        return _single_result(parent, plan, {
            "outcome": "unavailable",
            "reasons": _public_reasons([capture.get("reason")]) or [
                _reason("checkpoint_capture_unavailable", "an exact sampler checkpoint is unavailable")
            ],
        })

    request = {
        "position": plan["selection"]["position"],
        "change": deepcopy(plan["resolved_test"]["change"]),
    }
    exact_plan = plan_exact_force_token(
        parent,
        request,
        checkpoint_reference=dict(capture["checkpoint_reference"]),
        runtime_identity=dict(runtime_identity),
        worker_identity=dict(worker_identity),
    )
    if exact_plan.get("classification") != "exact_execution_fork":
        return _single_result(parent, plan, {
            "outcome": "unavailable",
            "reasons": exact_plan.get("reasons") or [
                _reason("exact_execution_unavailable", "the sampler change is not exact-executable")
            ],
            "exactness": exact_plan.get("exactness"),
        })
    execution = execute_exact_force_token(
        parent,
        exact_plan,
        engine,
        runtime_identity=dict(runtime_identity),
        worker_identity=dict(worker_identity),
        reload_parent=reload_parent,
        cancel_check=cancel_check,
    )
    receipt = execution.get("receipt") or {}
    child = execution.get("child")
    if receipt.get("phase") == "completed" and isinstance(child, Mapping) and child.get("id"):
        child = dict(child)
        child["outcome"] = "exact_execution_fork"
        child["execution_fork_execution_id"] = receipt.get("execution_id")
        child["exactness"] = deepcopy(receipt.get("exactness") or {})
        child["unchanged_control"] = deepcopy(receipt.get("unchanged_control") or {})
        return _single_result(parent, plan, child)
    failure = {
        "outcome": "unavailable",
        "reasons": receipt.get("reasons") or [_reason(
            "exact_execution_failed", "the exact sampler execution did not complete")],
        "execution_fork_execution_id": receipt.get("execution_id"),
        "exactness": receipt.get("exactness"),
        "unchanged_control": receipt.get("unchanged_control"),
    }
    return _single_result(parent, plan, failure)


def _fan_outcome(fan_result: Mapping) -> str:
    status = (fan_result.get("summary") or {}).get("status")
    if status == "completed":
        return "completed"
    if status in {"partial", "partial_cancelled"}:
        return "partial"
    if status == "cancelled":
        return "cancelled"
    return "unavailable"


def _counterfactual_outcome(result: Mapping) -> str:
    status = (result.get("execution") or {}).get("status") if isinstance(result, Mapping) else None
    if status == "completed":
        return "completed"
    if status in {"partial", "partial_cancelled"}:
        return "partial"
    if status == "cancelled":
        return "cancelled"
    if status in {"unavailable", "stale_parent"}:
        return "unavailable"
    if status == "failed":
        return "failed"
    return "unavailable"


def _context_bisect_outcome(result: Mapping) -> str:
    status = (result.get("execution") or {}).get("status") if isinstance(result, Mapping) else None
    if status == "completed":
        return "completed"
    if status in {"partial_cancelled", "partial"}:
        return "partial"
    if status == "cancelled":
        return "cancelled"
    if status in {"unavailable", "stale_parent", "running"}:
        return "unavailable"
    if status == "failed":
        return "failed"
    return "unavailable"


def _sampler_sensitivity_outcome(result: Mapping) -> str:
    status = (result.get("summary") or {}).get("status") if isinstance(result, Mapping) else None
    if status == "completed":
        return "completed"
    if status in {"partial", "partial_cancelled"}:
        return "partial"
    if status in {"cancelled"}:
        return "cancelled"
    if status in {"unavailable", "stale_parent"}:
        return "unavailable"
    return "failed"


def execute_test_this(
    parent_run: Mapping,
    sub,
    request,
    *,
    runtime_identity=None,
    worker_identity=None,
    reload_parent=None,
    cancel_check=None,
    plan=None,
) -> dict:
    """Rebuild the current plan and dispatch one explicit Test This request."""
    current_plan = build_test_this_plan(parent_run, request)
    if plan is not None and (
        not isinstance(plan, Mapping)
        or plan.get("run_id") != current_plan.get("run_id")
        or plan.get("parent_fingerprint_sha256") != current_plan.get("parent_fingerprint_sha256")
    ):
        # A preview is explanatory only.  Execution authority is always the rebuilt plan; the
        # mismatch is intentionally not allowed to smuggle a stale selection into a live fork.
        current_plan = build_test_this_plan(parent_run, request)
    if current_plan["resolution"]["state"] != "ready":
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent_run["id"],
            "selection": deepcopy(current_plan["selection"]),
            "test": deepcopy(current_plan["test"]),
            "operation": current_plan["resolution"]["operation"],
            "outcome": "unavailable",
            "result": {"reasons": [current_plan["resolution"].get("reason", _reason(
                "test_unavailable", "the requested test is unavailable"))]},
            "artifact": None,
            "comparison": None,
        }
        schemas.validate(document, RESULT_SCHEMA_VERSION)
        return document

    operation = current_plan["resolved_test"]["operation"]
    if operation == "force_token":
        return _execute_force_token(
            parent_run, sub, current_plan, runtime_identity=runtime_identity,
            worker_identity=worker_identity, reload_parent=reload_parent, cancel_check=cancel_check)
    if operation == "branch_fan":
        from clozn.replay import branch_fan as branch_fan_module

        fan_result = branch_fan_module.branch_fan(
            parent_run,
            sub,
            current_plan["selection"]["position"],
            limit=current_plan["test"]["limit"],
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=reload_parent,
            cancel_check=cancel_check,
        )
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent_run["id"],
            "selection": deepcopy(current_plan["selection"]),
            "test": deepcopy(current_plan["test"]),
            "operation": operation,
            "outcome": _fan_outcome(fan_result),
            "result": {"branch_fan": deepcopy(fan_result)},
            "artifact": {"schema": "clozn.branch-fan.v1", "result": deepcopy(fan_result)},
            "comparison": None,
        }
        schemas.validate(document, RESULT_SCHEMA_VERSION)
        return document
    if operation == "sampler_sensitivity":
        from clozn.replay.sampler_sensitivity import execute_sampler_sensitivity

        sensitivity = execute_sampler_sensitivity(
            parent_run,
            sub,
            current_plan["resolved_test"]["sampler_sensitivity_plan"],
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=reload_parent,
            cancel_check=cancel_check,
        )
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent_run["id"],
            "selection": deepcopy(current_plan["selection"]),
            "test": deepcopy(current_plan["test"]),
            "operation": operation,
            "outcome": _sampler_sensitivity_outcome(sensitivity),
            "result": {"sampler_sensitivity": deepcopy(sensitivity)},
            "artifact": {
                "schema": "clozn.sampler-sensitivity.v1",
                "result": deepcopy(sensitivity),
            },
            "comparison": None,
        }
        schemas.validate(document, RESULT_SCHEMA_VERSION)
        return document
    if operation == "influence_counterfactual":
        from clozn.replay.influence_counterfactual import execute_influence_counterfactual

        result = execute_influence_counterfactual(
            parent_run,
            sub,
            {
                "influence": {
                    "source_span_id": current_plan["selection"]["source_span_id"],
                    "answer_span_id": current_plan["selection"]["answer_span_id"],
                },
                "intervention": {"kind": current_plan["test"]["kind"]},
                "specificity_control": current_plan["resolved_test"][
                    "counterfactual_plan"]["specificity_control"]["requested"],
            },
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=reload_parent,
            cancel_check=cancel_check,
            plan=current_plan["resolved_test"]["counterfactual_plan"],
        )
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent_run["id"],
            "selection": deepcopy(current_plan["selection"]),
            "test": deepcopy(current_plan["test"]),
            "operation": operation,
            "outcome": _counterfactual_outcome(result),
            "result": {"influence_counterfactual": deepcopy(result)},
            "artifact": {
                "schema": "clozn.influence-counterfactual.v1",
                "result": deepcopy(result),
            },
            "comparison": deepcopy(result.get("comparison")),
        }
        schemas.validate(document, RESULT_SCHEMA_VERSION)
        return document
    if operation == "context_bisect":
        from clozn.replay.context_bisect import execute_context_bisect

        bisect = execute_context_bisect(
            parent_run,
            sub,
            {
                "influence": {
                    "source_span_id": current_plan["selection"]["source_span_id"],
                    "answer_span_id": current_plan["selection"]["answer_span_id"],
                },
                **{key: value for key, value in current_plan["test"].items() if key != "kind"},
            },
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=reload_parent,
            cancel_check=cancel_check,
            plan=current_plan["resolved_test"]["context_bisect_plan"],
        )
        document = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "run_id": parent_run["id"],
            "selection": deepcopy(current_plan["selection"]),
            "test": deepcopy(current_plan["test"]),
            "operation": operation,
            "outcome": _context_bisect_outcome(bisect),
            "result": {"context_bisect": deepcopy(bisect)},
            "artifact": {"schema": "clozn.context-bisect.v1", "result": deepcopy(bisect)},
            "comparison": None,
        }
        schemas.validate(document, RESULT_SCHEMA_VERSION)
        return document
    return _execute_sampling(
        parent_run, sub, current_plan, runtime_identity=runtime_identity,
        worker_identity=worker_identity, reload_parent=reload_parent, cancel_check=cancel_check)


__all__ = ["execute_test_this"]
