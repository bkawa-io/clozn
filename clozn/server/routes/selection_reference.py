"""POST /runs/<id>/selection/reference -- create a deterministic read-only ``sel1`` reference."""
from __future__ import annotations

from collections.abc import Mapping

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/selection/reference"
_CONTRACT_ERROR = {
    "error": "selection reference could not be composed",
    "code": "selection_reference_contract_invalid",
}


def _error(h, status: int, code: str, message: str):
    h._json(status, {"error": message, "code": code})
    return True


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog
    run_id = p[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        return _error(h, 404, "run_not_found", "run not found")
    if not isinstance(body, Mapping):
        return _error(h, 400, "invalid_body", "body must be an object")
    if set(body) != {"selection"} or not isinstance(body.get("selection"), Mapping):
        return _error(h, 400, "invalid_selection", "body must contain one selection object")

    from clozn import schemas
    from clozn.runs.selection_reference import (
        SelectionReferenceInputError,
        encode_selection_reference,
    )

    try:
        document = encode_selection_reference(run, body["selection"])
        if document.get("state") == "unavailable":
            return _error(h, 422, (document.get("reason") or {}).get("code", "selection_unavailable"),
                          "selection cannot be bound to immutable recorded evidence")
        schemas.validate(document, "clozn.selection-reference.v1")
    except SelectionReferenceInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except (TypeError, ValueError, UnicodeError, schemas.ValidationError):
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])
    except Exception:
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])

    h._json(200, document)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_post"]
