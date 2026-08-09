"""GET /runs/<id>/context-utilization -- which prompt sources showed a clear measured effect on this
answer, which were measured but stayed below the measurement floor, and which were never measured at all
(E9, built on the same shared `clozn.runs.influence_geometry` primitives as `clozn.runs.influence_query`
"Why this?", E7, and `clozn.runs.context_tension`, E8).

NAMED "CONTEXT UTILIZATION", NOT "DEAD CONTEXT"
----------------------------------------------------
The persisted influence measurement is BOUNDED -- it scores at most `selection.max_context_spans` prompt
sources, chosen by a deterministic "earliest policy/system source, then most recent context" strategy.
A source this route reports as `not_measured` may have been the single most important piece of context
in the prompt; Clozn simply never scored it, and this route never pretends otherwise. See
`clozn.runs.context_utilization`'s own module docstring for the full three-state vocabulary
(`clear_measured_effect` / `below_measured_floor` / `not_measured`) and why `below_measured_floor` is
never a claim of irrelevance.

A PURE PROJECTION, NEVER A NEW MEASUREMENT
--------------------------------------------
`clozn.runs.context_utilization.build_context_utilization` reads only the run already on disk and its
already-persisted `run["influence_map"]` (`clozn.context_answer_influence.v1`). It never calls
`context_answer_influence(...)`, never starts an influence-map job, never touches `select_control_model_
for_run`/`score_tokens`/an engine client/a worker, and never mutates the run. When no measurement is
stored, this route says so (`measurement.state == "not_measured"`) rather than computing one.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4), spliced in before the generic GET /runs/<id>
fallback -- see clozn/server/routes/_autoload.py's own docstring for why that ordering is semantic, not
cosmetic, for every /runs/<id>/<suffix> family including this one.

Fixed at `privacy="metadata_only"`, matching `influence_query.py`, `context_tension.py`,
`claim_support.py`, and `span_addresses.py`: the response never embeds prompt or answer text, only real
`clozn.text-span-addresses.v1` `span_...` address IDs a caller already has a route to resolve.

No query parameters -- this is a coverage artifact over every prompt source the persisted measurement
knows about; silently returning only some of them would undermine the whole point of the view.

Wire shape:
  GET /runs/<id>/context-utilization
      -> 200 <clozn.context-utilization.v1>
      -> 404 the run was not found
      -> 500 {"error": ..., "code": "context_utilization_contract_invalid"} -- a malformed or internally
         inconsistent stored influence artifact (a selection/`selected` disagreement, a selected source
         missing its expected coarse spans or links, ...) could not be composed into a valid response.
         Text-free (mirrors influence_query.py's, context_tension.py's, claim_support.py's, and
         span_addresses.py's own contract-failure handlers): persisted evidence may itself carry private
         source literals, and there is no per-request input here that could legitimately warrant a 400.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/context-utilization"

_CONTRACT_ERROR = {
    "error": "run context utilization could not be composed",
    "code": "context_utilization_contract_invalid",
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
    from clozn.runs.context_utilization import build_context_utilization

    try:
        document = build_context_utilization(run)
    except (ValueError, TypeError, UnicodeError, schemas.ValidationError):
        # Metadata-only route with no per-request input, so there is no legitimate 400 case here -- a
        # ValueError from build_context_utilization always means the persisted evidence itself is
        # malformed or internally inconsistent (bad run id, bad selection bookkeeping, a selected source
        # missing coarse spans/links, ...), never a caller mistake. An exception raised while reading a
        # malformed legacy run or influence artifact may itself contain private source literals, so the
        # public failure stays generic and text-free.
        h._json(500, _CONTRACT_ERROR)
        return True

    h._json(200, document)
    return True
