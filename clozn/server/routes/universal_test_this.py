"""HTTP routes for the explicit, backend-only Test This dispatcher."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_PLAN_SUFFIX = "/test-this/plan"
_EXECUTE_SUFFIX = "/test-this"

_CONTRACT_ERROR = {
    "error": "test-this result could not be composed",
    "code": "test_this_contract_invalid",
}


def _error(h, status: int, code: str, message: str):
    h._json(status, {"error": message, "code": code})
    return True


def _request_run_id(path: str, suffix: str) -> str:
    return path[len("/runs/"):-len(suffix)]


def try_post(h, p, body):
    is_plan = p.startswith("/runs/") and p.endswith(_PLAN_SUFFIX)
    is_execute = (
        p.startswith("/runs/") and p.endswith(_EXECUTE_SUFFIX)
        and not p.endswith(_PLAN_SUFFIX)
    )
    if not (is_plan or is_execute):
        return False

    import clozn.runs.store as runlog

    suffix = _PLAN_SUFFIX if is_plan else _EXECUTE_SUFFIX
    run_id = _request_run_id(p, suffix)
    parent = runlog.get_run(run_id)
    if parent is None:
        return _error(h, 404, "run_not_found", "run not found")

    from clozn.runs.test_this import TestThisInputError, build_test_this_plan

    try:
        plan = build_test_this_plan(parent, body)
    except TestThisInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, "test_this_plan_contract_invalid", _CONTRACT_ERROR["error"])

    if is_plan:
        # This branch intentionally stops before model routing, engine health, checkpoint capture,
        # or any other live operation.  ``live_state=not_checked`` is part of the plan contract.
        h._json(200, plan)
        return True

    if plan["resolution"]["state"] != "ready":
        from clozn.replay.test_this import execute_test_this
        try:
            result = execute_test_this(parent, None, body, plan=plan)
        except Exception:
            result = {
                "schema_version": "clozn.test-this-result.v1",
                "run_id": parent["id"],
                "selection": plan["selection"],
                "test": plan["test"],
                "operation": plan["resolution"]["operation"],
                "outcome": "unavailable",
                "result": {"reasons": [plan["resolution"].get("reason", {
                    "code": "test_unavailable", "message": "the requested test is unavailable"
                })]},
                "artifact": None,
                "comparison": None,
            }
        h._json(422, result)
        return True

    # Execution routing is always parent-scoped.  No request field is accepted as a model, worker,
    # or runtime override; the shared resolver handles managed multi-model and legacy deployments.
    from clozn.server.model_routing import select_run_model_facts

    facts = select_run_model_facts(h, parent, route="/runs/<id>/test-this")
    if facts is None:
        return True
    runtime_identity, worker_identity, engine, sub = facts
    if engine is None:
        return _error(
            h, 503, "test_this_worker_unavailable",
            "Test This requires a ready product model worker",
        )
    if plan["execution"]["fidelity_policy"] in {"exact_required", "controlled_regeneration"} and (
        runtime_identity is None or worker_identity is None
    ):
        return _error(
            h, 503, "test_this_worker_unavailable",
            "the exact sampler test requires identity-qualified worker facts",
        )

    from clozn.replay.test_this import execute_test_this

    try:
        result = execute_test_this(
            parent,
            sub,
            body,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=runlog.get_run,
            cancel_check=getattr(h, "_execution_fork_cancelled", None),
            plan=plan,
        )
    except TestThisInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])

    outcome = result.get("outcome")
    if outcome in {"completed", "partial"}:
        status = 201
    elif outcome == "cancelled":
        status = 409
    elif outcome == "unavailable":
        status = 422
    else:
        status = 500
    h._json(status, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
