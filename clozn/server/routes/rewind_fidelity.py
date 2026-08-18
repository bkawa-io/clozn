"""GET /runs/<id>/rewind-fidelity -- if I rewind this recorded run, what fidelity can Clozn truthfully
promise? (E10, `clozn.replay.rewind_fidelity.build_rewind_fidelity`.)

READ-ONLY AND OFFLINE-SAFE, ABSOLUTELY
-------------------------------------------
This route must never start a worker, cold-load a model, select/load a control model, capture or
hydrate a checkpoint, call `execution_fork(...)`/`execution_fork_checkpoint(...)`, run an unchanged
control, regenerate text, or score tokens. It imports only `clozn.runs.store` (to load the immutable
parent run), `clozn.replay.execution_fork_results` (a pure SQLite READ that creates nothing on a miss --
see that module's own `list_for_parent` docstring), and `clozn.replay.rewind_fidelity` (pure planning-
adjacent logic, no engine/worker import anywhere in its own dependency chain). It does NOT import
`clozn.server.app` (no `SUB`/`ENGINE` access) -- a fidelity indicator must be drawable while the runtime
is completely offline.

NOT A LIVE EXECUTION PLANNER
-------------------------------------------
The canonical Time Travel resolver is the authority for whether an exact rewind can be realized now;
it depends on live worker/checkpoint/runtime state this route deliberately never inspects. This route
answers a narrower, cheaper question from recorded evidence and prior proof alone; see
`clozn.replay.rewind_fidelity`'s own module docstring for the full three-concept boundary
(reconstructed / requires_live_plan / historically_verified_exact) this route's response encodes.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4), spliced in before the generic GET /runs/<id>
fallback -- see clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not
cosmetic, for every /runs/<id>/<suffix> family including this one.

No query parameters -- the response already carries the run's full valid coordinate range plus every
historically verified boundary; there is nothing to page or select for V1.

Wire shape:
  GET /runs/<id>/rewind-fidelity
      -> 200 <clozn.rewind-fidelity.v1>
      -> 404 the run was not found
      -> 500 {"error": ..., "code": "rewind_fidelity_contract_invalid"} -- a malformed run (no id) or an
         internal composition failure. Text-free: this endpoint has no per-request input that could
         legitimately warrant a 400, and recorded evidence may itself carry private literals.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/rewind-fidelity"

_CONTRACT_ERROR = {
    "error": "run rewind fidelity could not be composed",
    "code": "rewind_fidelity_contract_invalid",
}


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    from clozn import schemas
    from clozn.replay import execution_fork_results
    from clozn.replay.rewind_fidelity import build_rewind_fidelity

    try:
        historical_receipts = execution_fork_results.list_for_parent(run_id)
        document = build_rewind_fidelity(run, historical_receipts=historical_receipts)
    except (ValueError, TypeError, UnicodeError, schemas.ValidationError):
        # Metadata-only route with no per-request input, so there is no legitimate 400 case here. An
        # exception raised while reading malformed legacy run or receipt data may itself contain
        # private evidence, so the public failure stays generic and text-free.
        h._json(500, _CONTRACT_ERROR)
        return True

    h._json(200, document)
    return True
