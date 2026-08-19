"""GET /runs/<id>/investigation -- compose existing evidence without starting measurements."""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/investigation"


def try_get(h, p):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog

    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    # Every source is a journal read or deterministic projection.  In particular, checking whether the
    # worker exposes score_tokens does not call it; expensive measurements remain action descriptors.
    related = list(runlog.iter_runs(limit=200))
    # The run's OWN model resolves whose capabilities are being reported -- a bare ctx.active_sub(h)
    # fails closed under a managed gateway (see clozn.server.app.active_sub's docstring) and would
    # silently under-report scoring_available as False for every run, not just ones whose worker is
    # genuinely unavailable. peek_control_model_for_run never turns that unavailability into a hard
    # refusal here -- this route composes existing evidence and starts no measurement either way, so
    # an unresolvable worker degrades to scoring_available:false (still 200), exactly like legacy mode
    # already does when the one engine is down.
    from clozn.server.model_routing import peek_control_model_for_run
    sub = peek_control_model_for_run(h, run.get("model"), route="/runs/<id>/investigation")
    scoring_available = bool(sub and callable(getattr(sub, "score_tokens", None)))

    from clozn.behavior import corrective_flow
    from clozn.runs.investigation import build
    from clozn import schemas

    registry = corrective_flow.registry_for_run(run)
    try:
        document = build(
            run,
            related_runs=related,
            corrective_registry=registry,
            scoring_available=scoring_available,
        )
        schemas.validate(document, "clozn.run-investigation.v1")
    except (ValueError, schemas.ValidationError) as exc:
        h._json(500, {
            "error": "run investigation could not be composed",
            "code": "investigation_contract_invalid",
            "detail": str(exc),
        })
        return True
    h._json(200, document)
    return True
