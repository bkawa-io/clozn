"""HTTP read/manage surface for durable checkpoint pins.

Pin creation remains run-scoped (``POST /runs/<id>/snapshot/pin``) because it must
materialize a checkpoint on the worker that served that run.  This module exposes the
local pin ledger to Studio and keeps unpinning behind the same typed store errors as the
CLI.  No checkpoint bytes are returned by the list route.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

SCHEMA_VERSION = "clozn.pinned-checkpoint-list.v1"
_PREFIX = "/snapshots"


def _document() -> dict:
    from clozn.replay.checkpoint_pin_store import list_pins

    document = {"schema_version": SCHEMA_VERSION, "snapshots": list_pins()}
    from clozn import schemas
    schemas.validate(document, SCHEMA_VERSION)
    return document


def try_get(h, p):
    if p != _PREFIX:
        return False
    h._json(200, _document())
    return True


def try_post(h, p, body):
    if not p.startswith(_PREFIX + "/") or not p.endswith("/unpin"):
        return False
    run_id = p[len(_PREFIX) + 1:-len("/unpin")]
    if not run_id or "/" in run_id:
        return False
    body = body if isinstance(body, dict) else {}
    cascade = body.get("cascade", False)
    if not isinstance(cascade, bool):
        h._json(400, {"error": "cascade must be a boolean", "code": "snapshot_unpin_invalid_cascade"})
        return True

    from clozn.replay import checkpoint_pin_store as pins
    try:
        receipt = pins.unpin_checkpoint(run_id, cascade=cascade)
    except pins.PinHasDependentsError as exc:
        h._json(409, {
            "error": str(exc),
            "code": "snapshot_unpin_has_dependents",
            "run_id": run_id,
            "children": list(exc.children),
        })
        return True
    except pins.PinStoreError as exc:
        h._json(404, {"error": str(exc), "code": "snapshot_unpin_not_found", "run_id": run_id})
        return True

    h._json(200, receipt)
    return True


__all__ = ["try_get", "try_post"]
