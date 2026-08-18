"""HTTP routes for Influence -> Counterfactual Confirmation."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_PLAN_SUFFIX = "/influence-counterfactual/plan"
_EXECUTE_SUFFIX = "/influence-counterfactual"

_CONTRACT_ERROR = {
    "error": "influence counterfactual result could not be composed",
    "code": "influence_counterfactual_contract_invalid",
}


def _error(h, status: int, code: str, message: str):
    h._json(status, {"error": message, "code": code})
    return True


def _run_id(path: str, suffix: str) -> str:
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
    from clozn.runs.influence_counterfactual import (
        InfluenceCounterfactualInputError,
        build_influence_counterfactual_plan,
    )

    suffix = _PLAN_SUFFIX if is_plan else _EXECUTE_SUFFIX
    parent = runlog.get_run(_run_id(p, suffix))
    if parent is None:
        return _error(h, 404, "run_not_found", "run not found")
    if not isinstance(body, Mapping):
        return _error(h, 400, "invalid_body", "body must be an object")
    try:
        plan = build_influence_counterfactual_plan(parent, body)
    except InfluenceCounterfactualInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])

    if is_plan:
        # Planning deliberately stops before model routing, worker health, generation, or writes.
        h._json(200, plan)
        return True

    from clozn.replay.influence_counterfactual import execute_influence_counterfactual

    if plan["execution"]["state"] != "ready":
        try:
            result = execute_influence_counterfactual(parent, None, body, plan=plan)
        except Exception:
            return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])
        h._json(422, result)
        return True

    # Resolve only the immutable parent's recorded model.  The request has no model, worker, or
    # runtime override and this route never falls back to the gateway default.
    from clozn.server.model_routing import select_run_model_facts

    facts = select_run_model_facts(
        h, parent, route="/runs/<id>/influence-counterfactual")
    if facts is None:
        return True
    runtime_identity, worker_identity, engine, sub = facts
    if engine is None or runtime_identity is None or worker_identity is None:
        return _error(
            h, 503, "influence_counterfactual_worker_unavailable",
            "Influence counterfactual requires a ready identity-qualified product worker",
        )

    try:
        result = execute_influence_counterfactual(
            parent,
            sub,
            body,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=runlog.get_run,
            cancel_check=getattr(h, "_execution_fork_cancelled", None),
            plan=plan,
        )
    except InfluenceCounterfactualInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])

    execution = result.get("execution") if isinstance(result, Mapping) else {}
    status = execution.get("status") if isinstance(execution, Mapping) else None
    children = execution.get("children_created", 0) if isinstance(execution, Mapping) else 0
    if children:
        http_status = 201
    elif status in {"cancelled", "stale_parent"}:
        http_status = 409
    elif status in {"unavailable", "failed"}:
        http_status = 422 if status == "unavailable" else 500
    else:
        http_status = 500
    h._json(http_status, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
