"""GET /runs/<id>/suggested-breakpoints -- read-only breakpoint suggestions.

The route composes only evidence already persisted with the run.  It never starts influence
measurement, scores tokens, selects a worker, calls a model, creates a checkpoint, executes a fork,
or asks a live rewind planner for advice.  A breakpoint is a suggested test location, not a diagnosis.

Registered through the route autoloader so this backend-only endpoint does not require a shared app.py
route-list edit.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/suggested-breakpoints"

_CONTRACT_ERROR = {
    "error": "run suggested breakpoints could not be composed",
    "code": "suggested_breakpoints_contract_invalid",
}


def _parse_limit(raw: str | None) -> int | None:
    """Parse one strict base-10 query value without accepting floats or blank values."""
    if raw is None or raw == "":
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def _invalid_limit(h) -> bool:
    h._json(400, {
        "error": "limit must be an integer from 1 to 50",
        "code": "invalid_limit",
    })
    return True


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    query = parse_qs(urlparse(h.path).query, keep_blank_values=True)
    limit_values = query.get("limit")
    if limit_values is None:
        limit = 12
    elif len(limit_values) != 1:
        return _invalid_limit(h)
    else:
        limit = _parse_limit(limit_values[0])
        if limit is None or not (1 <= limit <= 50):
            return _invalid_limit(h)

    from clozn.runs.suggested_breakpoints import build_suggested_breakpoints

    try:
        document = build_suggested_breakpoints(run, limit=limit)
    except Exception:
        # A malformed legacy run/evidence artifact must not turn private parser details into a public
        # response.  The domain builder is still responsible for its normal explicit missing-evidence
        # states; this branch is reserved for an internal contract/schema failure.
        h._json(500, _CONTRACT_ERROR)
        return True

    h._json(200, document)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get"]
