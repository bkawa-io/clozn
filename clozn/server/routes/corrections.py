"""HTTP surface for the durable, explicit F5/F6 correction store.

The correction implementation is deliberately model-free and already owns all validation and
transactions.  This route is only an adapter: it never infers scope, reads prompt text, or silently
confirms a draft.  F6 verification accepts two immutable run ids; it does not generate a retry.
"""
from __future__ import annotations

from clozn import schemas


CLOZN_ROUTE_AUTOLOAD = True


def _store():
    from clozn.runs import corrections
    return corrections


def _error(h, exc, *, status: int = 400):
    h._json(status, {"error": str(exc), "type": type(exc).__name__})
    return True


def _state_status(exc) -> int:
    from clozn.runs.corrections import CorrectionStateError, CorrectionNotFoundError
    if isinstance(exc, (CorrectionStateError, CorrectionNotFoundError)):
        return 409 if isinstance(exc, CorrectionStateError) else 404
    return 400


def try_get(h, p):
    corrections = _store()
    if p == "/corrections":
        h._json(200, {
            "schema_version": corrections.SCHEMA_NAME,
            "corrections": corrections.list_corrections(),
        })
        return True
    if not p.startswith("/corrections/"):
        return False
    rest = p[len("/corrections/"):]
    if not rest or "/" in rest and not rest.endswith("/export"):
        return False
    if rest.endswith("/export"):
        correction_id = rest[:-len("/export")]
        exported = corrections.export_correction(correction_id)
        if exported is None:
            h._json(404, {"error": "correction not found"})
            return True
        h._json(200, exported)
        return True
    correction = corrections.get_correction(rest)
    if correction is None:
        h._json(404, {"error": "correction not found"})
        return True
    h._json(200, correction)
    return True


def try_post(h, p, body):
    corrections = _store()
    body = body if isinstance(body, dict) else {}

    if p == "/corrections":
        try:
            result = corrections.draft_correction(
                scope_kind=body.get("scope_kind"),
                scope_value=body.get("scope_value"),
                correction_type=body.get("type"),
                content=body.get("content"),
            )
        except corrections.CorrectionError as exc:
            return _error(h, exc)
        h._json(201, result)
        return True

    if p == "/corrections/resolve":
        try:
            result = corrections.resolve_corrections(
                session_id=body.get("session_id"),
                client_id=body.get("client_id"),
                project_id=body.get("project_id"),
                model_sha256=body.get("model_sha256"),
                include_global_local=bool(body.get("include_global_local", True)),
            )
        except corrections.CorrectionError as exc:
            return _error(h, exc)
        h._json(200, result)
        return True

    if not p.startswith("/corrections/"):
        return False
    rest = p[len("/corrections/"):]
    if "/" not in rest:
        return False
    correction_id, action = rest.split("/", 1)
    try:
        if action == "confirm":
            result = corrections.confirm_correction(correction_id)
        elif action == "disable":
            result = corrections.disable_correction(correction_id)
        elif action == "enable":
            result = corrections.enable_correction(correction_id)
        elif action == "delete":
            result = corrections.delete_correction(correction_id, reason=body.get("reason"))
        elif action == "undo":
            result = corrections.undo_last_change(correction_id)
        elif action == "verify":
            from clozn.runs import teaching_loop
            result = teaching_loop.verify_and_promote(
                correction_id,
                target_run_id=body.get("target_run_id"),
                child_run_id=body.get("child_run_id"),
                match_criterion=body.get("match_criterion", "exact_output"),
            )
            schemas.validate(result, teaching_loop.SCHEMA_NAME)
        else:
            return False
    except corrections.CorrectionError as exc:
        return _error(h, exc, status=_state_status(exc))
    except Exception as exc:
        from clozn.runs.teaching_loop import TeachingLoopError
        if isinstance(exc, TeachingLoopError):
            return _error(h, exc, status=400)
        raise
    h._json(200, result)
    return True
