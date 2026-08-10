"""Read-only Select -> Inspect composition for raw selections and ``sel1`` references."""
from __future__ import annotations

from collections.abc import Mapping
from urllib.parse import parse_qs, urlparse

CLOZN_ROUTE_AUTOLOAD = True
_SUFFIX = "/selection/inspect"
_CONTRACT_ERROR = {
    "error": "selection inspection could not be composed",
    "code": "selection_inspection_contract_invalid",
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
    from clozn.runs.selection_inspection import (
        SelectionInspectionInputError,
        build_selection_inspection,
    )

    try:
        document = build_selection_inspection(run, selection=body["selection"])
        schemas.validate(document, "clozn.selection-inspection.v1")
    except SelectionInspectionInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except (TypeError, ValueError, UnicodeError, schemas.ValidationError):
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])
    except Exception:
        # Do not expose arbitrary legacy-artifact exception text; source and prompt data may be present
        # in malformed records even though the normal document is metadata-only.
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])

    h._json(200, document)
    return True


def try_get(h, p):
    path = p.split("?", 1)[0]
    if not (path.startswith("/runs/") and path.endswith(_SUFFIX)):
        return False

    import clozn.runs.store as runlog
    run_id = path[len("/runs/"):-len(_SUFFIX)]
    run = runlog.get_run(run_id)
    if run is None:
        return _error(h, 404, "run_not_found", "run not found")

    query = parse_qs(urlparse(getattr(h, "path", p)).query, keep_blank_values=True)
    values = query.get("ref") or []
    if len(values) != 1 or not values[0]:
        return _error(h, 400, "invalid_reference_encoding", "selection reference is required")

    from clozn import schemas
    from clozn.runs.selection_contract import public_selection
    from clozn.runs.selection_inspection import build_selection_inspection
    from clozn.runs.selection_reference import (
        SelectionReferenceInputError,
        resolve_selection_reference,
    )

    try:
        resolution = resolve_selection_reference(run, values[0])
    except SelectionReferenceInputError as exc:
        return _error(h, 400, exc.code, str(exc))
    except Exception:
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])

    if resolution.get("state") == "stale":
        return _error(h, 409, (resolution.get("reason") or {}).get("code", "selection_reference_stale"),
                      "selection reference is stale")
    if resolution.get("state") == "unavailable":
        return _error(h, 422, (resolution.get("reason") or {}).get("code", "selection_reference_unavailable"),
                      "selection reference cannot currently be resolved")
    selection = resolution.get("resolved_selection")
    if not isinstance(selection, Mapping):
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])
    try:
        document = build_selection_inspection(run, selection=public_selection(selection))
        schemas.validate(document, "clozn.selection-inspection.v1")
    except Exception:
        return _error(h, 500, _CONTRACT_ERROR["code"], _CONTRACT_ERROR["error"])
    h._json(200, document)
    return True


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get", "try_post"]
