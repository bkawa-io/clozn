"""POST /runs/<id>/branch-fan -- bounded execution over recorded alternatives."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/branch-fan"

_CONTRACT_ERROR = {
    "error": "branch fan response could not be composed",
    "code": "branch_fan_contract_invalid",
}


def _bad(h, code: str, message: str):
    h._json(400, {"error": message, "code": code})
    return True


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    parent = runlog.get_run(run_id)
    if parent is None:
        h._json(404, {"error": "run not found"})
        return True
    if not isinstance(body, Mapping):
        return _bad(h, "invalid_body", "body must be an object")
    unknown = set(body) - {"position", "limit"}
    if unknown:
        return _bad(h, "invalid_body", "body may contain only position and limit")

    position = body.get("position")
    if not isinstance(position, int) or isinstance(position, bool) or position < 0:
        return _bad(h, "invalid_position", "position must be a non-negative integer")
    limit = body.get("limit", 3)
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 4:
        return _bad(h, "invalid_limit", "limit must be an integer from 1 to 4")

    # Candidate discovery is a pure read of the parent trace.  Do it before routing so an ordinary
    # token with no recorded alternatives returns its typed 422 without waking/cold-loading a worker.
    from clozn.replay.branch_fan import BranchFanInputError, _recorded_candidates, branch_fan
    try:
        candidates, _recorded_count = _recorded_candidates(parent, position, limit)
    except BranchFanInputError as exc:
        return _bad(h, exc.code, str(exc))
    if not candidates:
        try:
            result = branch_fan(parent, None, position, limit=limit)
        except Exception:
            h._json(500, _CONTRACT_ERROR)
            return True
        h._json(422, result)
        return True

    # The resolver uses parent.model and the same managed/legacy routing path as execution-fork.
    # It never accepts a model supplied in this body.  Keep the selected substrate itself so
    # reconstructed replay can reapply the parent's recorded dials/template seam.
    from clozn.server.model_routing import select_control_model_for_run
    from clozn.server.routes.execution_fork import _identity_facts
    selection = select_control_model_for_run(h, parent.get("model"), route="/runs/<id>/branch-fan")
    if selection is None:
        return True
    runtime_identity, worker_identity, engine = _identity_facts(selection)
    sub = selection.sub
    if engine is None or runtime_identity is None or worker_identity is None:
        h._json(503, {
            "error": "branch fan requires a ready identity-qualified product worker",
            "code": "branch_fan_worker_unavailable",
        })
        return True

    try:
        result = branch_fan(
            parent,
            sub,
            position,
            limit=limit,
            runtime_identity=runtime_identity,
            worker_identity=worker_identity,
            reload_parent=runlog.get_run,
            cancel_check=getattr(h, "_execution_fork_cancelled", None),
        )
    except BranchFanInputError as exc:
        h._json(400, {"error": str(exc), "code": exc.code})
        return True
    except Exception:
        # Do not expose worker/parser exception text from a contract failure.  Branch-level execution
        # failures are already represented inside branches[] by the domain orchestrator.
        h._json(500, _CONTRACT_ERROR)
        return True

    summary = result.get("summary") or {}
    children = summary.get("children_created", 0)
    status = (
        201 if children else
        409 if summary.get("status") == "cancelled" else
        422
    )
    h._json(status, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
