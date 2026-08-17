"""GET /runs/<id>/diagnostics -- the canonical read-only Run diagnostics projection.

The route only loads the recorded Run and already-durable read-side evidence. It never selects a
worker, touches a substrate, executes a planner-backed experiment, or persists a derived result.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/diagnostics"


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog
    from clozn.runs.run_diagnostics import build_run_diagnostics

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    from clozn.replay import execution_fork_results
    try:
        historical_receipts = execution_fork_results.list_for_parent(run_id)
    except Exception:
        historical_receipts = []
    try:
        from clozn.replay.checkpoint_pin_store import resolve_pin
        checkpoint_pin = resolve_pin(run_id)
    except Exception as exc:
        checkpoint_pin = {"unavailable": f"checkpoint pin lookup failed: {type(exc).__name__}"}

    document = build_run_diagnostics(
        run,
        related_runs=runlog.iter_runs(limit=200),
        historical_receipts=historical_receipts,
        checkpoint_pin=checkpoint_pin,
    )
    h._json(200, document)
    return True
