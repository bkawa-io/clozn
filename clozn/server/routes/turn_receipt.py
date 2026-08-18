"""GET /runs/<id>/turn-receipt -- the everyday, read-side Turn Receipt v1.

This route only loads the recorded run and already-persisted optional evidence.  It never invokes an
influence scorer, model, worker, checkpoint, rewind, or live exact-resume planner.  Missing evidence is
represented by the projection's explicit ``not_measured``/unavailable states.
"""
from __future__ import annotations

from urllib.parse import parse_qs, urlparse

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/turn-receipt"

_CONTRACT_ERROR = {
    "error": "turn receipt could not be composed",
    "code": "turn_receipt_contract_invalid",
}


def _format(h) -> str:
    try:
        raw = (parse_qs(urlparse(h.path).query).get("format") or ["json"])[0]
    except Exception:
        raw = "json"
    return str(raw).strip().lower()


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    parent = None
    parent_id = run.get("parent_run_id") if isinstance(run, dict) else None
    if isinstance(parent_id, str) and parent_id:
        # Parent loading is a read-only identity lookup.  The builder still refuses to invent a diff:
        # only a persisted first-divergence view is eligible for comparison.
        parent = runlog.get_run(parent_id)

    historical_receipts = ()
    try:
        # The results store performs a read-only query and creates no checkpoint/worker on a miss.  A
        # receipt must remain useful even when that optional history store is unavailable.
        from clozn.replay import execution_fork_results
        historical_receipts = execution_fork_results.list_for_parent(run_id)
    except Exception:
        historical_receipts = ()

    from clozn.runs.turn_receipt import build_turn_receipt, to_markdown
    try:
        receipt = build_turn_receipt(
            run,
            parent_run=parent,
            rewind_history=historical_receipts,
        )
    except Exception:
        # Recorded evidence can be from an older or partially migrated run.  Do not expose source text or
        # internal parser details in a contract-failure response.
        h._json(500, _CONTRACT_ERROR)
        return True

    if _format(h) == "md":
        h._send(200, to_markdown(receipt), "text/markdown; charset=utf-8")
    else:
        h._json(200, receipt)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get"]
