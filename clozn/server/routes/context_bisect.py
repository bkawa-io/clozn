"""HTTP routes for the explicit bounded Context Bisect search."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_PLAN_SUFFIX = "/context-bisect/plan"
_EXECUTE_SUFFIX = "/context-bisect"


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
    from clozn.runs.context_bisect import ContextBisectInputError, plan_context_bisect

    suffix = _PLAN_SUFFIX if is_plan else _EXECUTE_SUFFIX
    parent = runlog.get_run(_run_id(p, suffix))
    if parent is None:
        return _error(h, 404, "run_not_found", "run not found")
    if not isinstance(body, Mapping):
        return _error(h, 400, "invalid_body", "body must be an object")
    try:
        plan = plan_context_bisect(parent, request=body)
    except ContextBisectInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, "context_bisect_contract_invalid", "context bisect plan could not be composed")

    if is_plan:
        # Planning is intentionally before model routing, worker health, replay, or persistence.
        h._json(200, plan)
        return True

    from clozn.replay.context_bisect import execute_context_bisect

    if plan["execution"]["state"] != "ready":
        try:
            result = execute_context_bisect(parent, None, body, plan=plan)
        except Exception:
            return _error(h, 500, "context_bisect_contract_invalid", "context bisect result could not be composed")
        h._json(422, result)
        return True

    # Parent-scoped routing only.  The request carries no model, worker, runtime, or adapter override.
    from clozn.server.model_routing import select_run_model_facts

    facts = select_run_model_facts(
        h, parent, route="/runs/<id>/context-bisect")
    if facts is None:
        return True
    runtime_identity, worker_identity, engine, sub = facts
    if engine is None or runtime_identity is None or worker_identity is None:
        return _error(
            h, 503, "context_bisect_worker_unavailable",
            "Context Bisect requires an identity-qualified parent model worker",
        )

    try:
        result = execute_context_bisect(
            parent,
            sub,
            body,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=runlog.get_run,
            cancel_check=getattr(h, "_execution_fork_cancelled", None),
            plan=plan,
        )
    except ContextBisectInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, "context_bisect_contract_invalid", "context bisect result could not be composed")

    execution = result.get("execution") if isinstance(result, Mapping) else {}
    status = execution.get("status") if isinstance(execution, Mapping) else None
    children = execution.get("children_created", 0) if isinstance(execution, Mapping) else 0
    if children:
        http_status = 201
    elif status in {"cancelled", "partial_cancelled", "stale_parent"}:
        http_status = 409
    elif status in {"unavailable", "failed"}:
        http_status = 422 if status == "unavailable" else 500
    else:
        http_status = 422
    h._json(http_status, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
