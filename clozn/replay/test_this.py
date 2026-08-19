"""Execution dispatcher for the explicit Test This action.

There is intentionally very little execution logic here.  The module resolves the pure plan again,
then hands the request to the canonical Time Travel recipe or the Branch Fan orchestrator.  Every
operation is evidence-first: generation stops at a GeneratedObservation, and child materialization is
a separate explicit operation.  Exactness proof, checkpoint handling, and comparison remain owned by
those lower-level primitives.
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
            "time_travel_execution_failed", str(exc)), outcome="failed")
    return _time_travel_result(parent, plan, travel)


def _time_travel_result(parent: Mapping, plan: Mapping, travel, *, reason: Mapping | None = None,
                        outcome: str = "unavailable") -> dict:
    """Wrap the canonical TimeTravelResult in the Test This envelope without child persistence.

    ``outcome`` applies only when there is no result to project: a precondition that was never met
    is unavailable, while an execution that raised is a failure.  The two are not the same claim.
    """
    operation = plan["resolved_test"]["operation"]
    projection = travel.to_dict() if travel is not None else None
    if travel is not None:
        status = getattr(travel, "status", "failed")
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
    """Run one explicit sampler test as a canonical sampler probe.

    Like the force-token path, this stops at a GeneratedObservation: it creates no child Run and
    writes no legacy execution-fork receipt.  Keeping a result is a separate explicit
    materialization of the observation this returns.
    """
    from clozn.experiments.persistence import ObservationStore
    from clozn.recipes.time_travel import run_sampler_probe

    engine = getattr(sub, "engine", None) if sub is not None else None
    if engine is None or not isinstance(runtime_identity, Mapping) or not isinstance(worker_identity, Mapping):
        return _time_travel_result(parent, plan, None, reason=_reason(
            "exact_execution_unavailable", "exact sampler execution is unavailable"))
    if callable(cancel_check) and cancel_check():
        return _time_travel_result(parent, plan, None, reason=_reason(
            "execution_cancelled", "the sampler test was cancelled before execution"))

    override = {name: value for name, value in plan["resolved_test"]["change"].items() if name != "type"}
    baseline = _recorded_sampler(parent)
    if baseline is None:
        return _time_travel_result(parent, plan, None, reason=_reason(
            "recorded_sampler_unavailable",
            "the recorded run has no complete sampler state to probe against"))
    checkpoint, capture_reason = _capture_checkpoint(
        parent, engine, runtime_identity=runtime_identity, worker_identity=worker_identity)
    if checkpoint is None:
        return _time_travel_result(parent, plan, None, reason=capture_reason)
    try:
        travel = run_sampler_probe(
            parent,
            position=plan["selection"]["position"],
            sampler=override,
            sampling={**baseline, **override},
            max_new=_remaining_horizon(parent, plan["selection"]["position"]),
            checkpoint=checkpoint,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            substrate=sub,
            run_loader=reload_parent,
            observation_store=ObservationStore(),
            cancel=cancel_check,
        )
    except Exception as exc:
        return _time_travel_result(parent, plan, None, reason=_reason(
            "sampler_probe_execution_failed", str(exc)), outcome="failed")
    return _time_travel_result(parent, plan, travel)


def _recorded_sampler(parent: Mapping) -> dict | None:
    """The parent's recorded sampler in the exact-resume field names, or None."""
    from clozn.experiments.execution_facts import recorded_sampler_state

    return recorded_sampler_state(parent)


def _remaining_horizon(parent: Mapping, position: int) -> int:
    trace = parent.get("trace") if isinstance(parent.get("trace"), Mapping) else {}
    tokens = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
    return max(1, len(tokens) - int(position))


def _capture_checkpoint(parent: Mapping, engine, *, runtime_identity, worker_identity):
    """Capture the exact context this test resumes from, through the canonical capture seam."""
    from clozn.replay.checkpoint_capture import capture_parent_checkpoint

    try:
        capture = capture_parent_checkpoint(
            parent, engine,
            runtime_identity=dict(runtime_identity), worker_identity=dict(worker_identity))
    except Exception:
        return None, _reason("checkpoint_capture_unavailable", "an exact checkpoint could not be captured")
    if not isinstance(capture, Mapping) or capture.get("status") != "available":
        reasons = capture.get("reasons") if isinstance(capture, Mapping) else None
        reason = reasons[0] if isinstance(reasons, list) and reasons and isinstance(reasons[0], Mapping) else {}
        return None, _reason(
            str(reason.get("code") or "checkpoint_capture_unavailable"),
            str(reason.get("message") or "an exact sampler checkpoint is unavailable"))
    reference = capture.get("checkpoint_reference")
    if not isinstance(reference, Mapping):
        return None, _reason("checkpoint_capture_unavailable", "an exact checkpoint reference was unavailable")
    return deepcopy(dict(reference)), None


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
            "artifact": {"schema": "clozn.branch-fan.v2", "result": deepcopy(fan_result)},
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
