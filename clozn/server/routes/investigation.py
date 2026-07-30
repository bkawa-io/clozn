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
    from clozn.server import app as ctx
    sub = ctx.active_sub(h)
    scoring_available = bool(sub and callable(getattr(sub, "score_tokens", None)))

    from clozn.behavior import corrective_flow
    from clozn.runs.investigation import build
    from clozn import schemas

    registry = corrective_flow.registry_for_run(
        run,
        steer=getattr(sub, "steer", None),
        active_profile=ctx._active_profile_name(),
    )
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
