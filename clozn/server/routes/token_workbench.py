"""GET /runs/<id>/tokens/<index>/workbench[?reference_run_id=<id>] -- compose existing evidence about
one token of one recorded run, without starting a measurement. -> clozn.runs.token_workbench.build.

Registered via CLOZN_ROUTE_AUTOLOAD (docs/SEAMS.md Seam 4) -- no edit to clozn/server/app.py. Mirrors
clozn/server/routes/investigation.py's composition pattern: every source here is a journal read or a
deterministic projection; checking whether the active substrate exposes score_tokens or an engine does
not call either, it only informs the `capabilities` block of what a SEPARATE POST action could do.

Status codes: 404 (no such run), 400 (index is not an integer, or the run has no trace / the index is
out of range for it), 500 (the composed document failed its own schema contract -- a bug here, not a
caller error), 200 otherwise. An unresolvable `?reference_run_id=` degrades the comparison and
mechanistic_diff sections to a labeled `unavailable`/`available: false` -- it never 404s the whole
request, since the primary run is still valid.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/workbench"
_MARKER = "/tokens/"


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False
    middle = p[len("/runs/"):-len(_SUFFIX)]
    if _MARKER not in middle:
        return False
    run_id, _, index_part = middle.partition(_MARKER)
    if not run_id or not index_part:
        return False
    try:
        index = int(index_part)
    except ValueError:
        h._json(400, {"error": "token index must be an integer"})
        return True

    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    from urllib.parse import parse_qs, urlparse

    query = parse_qs(urlparse(h.path).query)
    reference_run_id = ((query.get("reference_run_id") or [""])[0] or "").strip() or None
    reference_run = runlog.get_run(reference_run_id) if reference_run_id else None

    # Every source below is a journal read or deterministic projection. In particular, checking whether
    # the worker exposes score_tokens/an engine does not call either -- expensive measurements remain
    # capability descriptors pointing at their own separate POST action. The run's OWN model resolves
    # whose capabilities are being reported (see investigation.py's identical use of
    # peek_control_model_for_run for why this must not fail closed under a managed gateway, nor turn
    # "worker unavailable" into a hard refusal on a route that starts no measurement either way).
    related = list(runlog.iter_runs(limit=200))
    from clozn.server.model_routing import peek_control_model_for_run
    sub = peek_control_model_for_run(h, run.get("model"), route="/runs/<id>/tokens/<index>/workbench")
    scoring_available = bool(sub and callable(getattr(sub, "score_tokens", None)))
    worker_ready = bool(sub and getattr(sub, "engine", None))

    from clozn.behavior import corrective_flow
    from clozn.runs import investigation
    from clozn.runs import token_workbench
    from clozn import schemas

    try:
        registry = corrective_flow.registry_for_run(run, steer=getattr(sub, "steer", None))
        investigation_doc = investigation.build(
            run, related_runs=related, corrective_registry=registry, scoring_available=scoring_available)
        schemas.validate(investigation_doc, "clozn.run-investigation.v1")
    except (ValueError, schemas.ValidationError) as exc:
        h._json(500, {
            "error": "the composed investigation this workbench builds on is invalid",
            "code": "investigation_contract_invalid",
            "detail": str(exc),
        })
        return True

    try:
        document = token_workbench.build(
            run, index,
            investigation_doc=investigation_doc,
            related_runs=related,
            reference_run_id=reference_run_id,
            reference_run=reference_run,
            worker_ready=worker_ready,
        )
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    try:
        schemas.validate(document, "clozn.token-workbench.v1")
    except schemas.ValidationError as exc:
        h._json(500, {
            "error": "token workbench could not be composed",
            "code": "token_workbench_contract_invalid",
            "detail": str(exc),
        })
        return True
    h._json(200, document)
    return True
