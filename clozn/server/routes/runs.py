"""The narrow v1 run collection: list, read one, and delete one."""
from __future__ import annotations


def _run_id(path: str) -> str | None:
    if not path.startswith("/runs/"):
        return None
    run_id = path[len("/runs/"):]
    return run_id if run_id and "/" not in run_id else None


def try_get(h, path: str) -> bool:
    if path != "/runs":
        return False
    import clozn.runs.store as runlog
    h._json(200, {"runs": runlog.list_runs(80)})
    return True


def try_get_fallback(h, path: str) -> bool:
    """Read exactly ``/runs/<id>`` and reject suffixes instead of treating them as IDs."""
    if not path.startswith("/runs/"):
        return False
    run_id = _run_id(path)
    if run_id is None:
        h._json(404, {
            "error": "unknown run subroute",
            "code": "unknown_run_subroute",
        })
        return True
    import clozn.runs.store as runlog
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found", "code": "run_not_found"})
    else:
        h._json(200, run)
    return True


def try_delete(h, path: str) -> bool:
    run_id = _run_id(path)
    if run_id is None:
        return False
    from clozn.runs.mutations import MutationError, RunHasChildrenError, delete_run

    try:
        result = delete_run(run_id)
    except RunHasChildrenError as exc:
        h._json(409, {
            "error": str(exc),
            "code": "run_has_children",
            "children": list(exc.children),
        })
    except MutationError as exc:
        status = 404 if "was not found" in str(exc) else 400
        h._json(status, {
            "error": str(exc),
            "code": "run_not_found" if status == 404 else "invalid_run_id",
        })
    else:
        h._json(200, result)
    return True


__all__ = ["try_delete", "try_get", "try_get_fallback"]
