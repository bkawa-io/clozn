"""GET /runs/<id>/span-addresses -- metadata-only stable text-span projection.

The run store has already resolved any content-addressed influence blob before
this route sees the record.  Composition is pure: it neither starts an
influence job nor calls the active worker.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/span-addresses"


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
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses

    try:
        document = build_persisted_text_span_addresses(run, privacy="metadata_only")
        schemas.validate(document, "clozn.text-span-addresses.v1")
    except (TypeError, ValueError, UnicodeError, schemas.ValidationError):
        # This is a metadata-only route. Validation exceptions raised while
        # reading malformed legacy artifacts may contain source literals, so
        # keep the public failure stable and text-free.
        h._json(500, {
            "error": "run span addresses could not be composed",
            "code": "span_address_contract_invalid",
        })
        return True
    h._json(200, document)
    return True
