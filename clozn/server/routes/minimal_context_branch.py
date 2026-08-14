"""Explicit source-bound branch actions from a persisted Minimal Context result."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/minimal-context/branch"


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False
    import clozn.runs.store as runlog
    from clozn.runs.minimal_context_branch import (
        MinimalContextBranchError,
        execute_minimal_context_branch,
        plan_minimal_context_branch,
    )

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    parent = runlog.get_run(run_id)
    if parent is None:
        h._json(404, {"error": "run not found", "code": "run_not_found"})
        return True
    if not isinstance(body, Mapping):
        h._json(400, {"error": "body must be an object", "code": "invalid_body"})
        return True
    action = body.get("action")
    result_id = body.get("result_id")
    source_ids = body.get("source_ids")
    if not isinstance(action, str) or not action:
        h._json(400, {"error": "action is required", "code": "invalid_action"})
        return True
    if not isinstance(result_id, str) or not result_id:
        h._json(400, {"error": "result_id is required", "code": "invalid_result_id"})
        return True
    if not isinstance(source_ids, list):
        h._json(400, {"error": "source_ids must be a list", "code": "invalid_source_selection"})
        return True
    stored = parent.get("minimal_context_results")
    result = stored.get(result_id) if isinstance(stored, Mapping) else None
    if not isinstance(result, Mapping):
        h._json(404, {"error": "Minimal Context result not found", "code": "minimal_context_result_not_found"})
        return True
    try:
        plan = plan_minimal_context_branch(parent, result, action=action, source_ids=source_ids)
    except MinimalContextBranchError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True

    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, parent.get("model"), route="/runs/<id>/minimal-context/branch")
    if selection is None:
        return True
    sub = selection.sub
    if sub is None or not callable(getattr(sub, "chat", None)):
        h._json(503, {
            "error": "Minimal Context branch requires a ready product model worker",
            "code": "minimal_context_branch_worker_unavailable",
        })
        return True
    try:
        result = execute_minimal_context_branch(
            parent, result, sub, action=action, source_ids=source_ids,
            plan=plan, reload_parent=runlog.get_run,
            max_new=body.get("max_new"),
        )
    except MinimalContextBranchError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    status = 201 if result.get("state") == "completed" else 409
    h._json(status, result)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
