"""Registry-driven preview -> confirm -> keep routes for reversible answer fixes."""
from __future__ import annotations

from clozn.server import app as ctx


CLOZN_ROUTE_AUTOLOAD = True


def _run(run_id: str):
    import clozn.runs.store as runlog
    return runlog.get_run(run_id)


def _error(h, exc, *, status: int = 409):
    h._json(status, {"error": str(exc)})
    return True


def try_get(h, p):
    from clozn.behavior import corrective_flow, registry

    if p == "/actions/registry":
        sub = ctx.active_sub(h)
        h._json(200, registry.build_registry(steer=getattr(sub, "steer", None)))
        return True

    if p.startswith("/runs/") and p.endswith("/corrective-actions"):
        run_id = p[len("/runs/"):-len("/corrective-actions")]
        run = _run(run_id)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        # Describes what's available for this run; it never executes anything, so an unresolvable
        # worker degrades to steer=None (fewer eligible actions) rather than a hard refusal -- the
        # same "compose, don't block" contract investigation.py uses for the identical reason.
        from clozn.server.model_routing import peek_control_model_for_run
        sub = peek_control_model_for_run(h, run.get("model"), route="/runs/<id>/corrective-actions")
        h._json(200, corrective_flow.registry_for_run(
            run,
            steer=getattr(sub, "steer", None),
            active_profile=ctx._active_profile_name(),
        ))
        return True

    if p.startswith("/corrective-results/"):
        result_id = p[len("/corrective-results/"):]
        if "/" in result_id:
            return False
        result = corrective_flow.get_result(result_id)
        if result is None:
            h._json(404, {"error": "corrective action result not found"})
            return True
        h._json(200, result)
        return True
    return False


def try_post(h, p, body):
    from clozn.behavior import corrective_flow
    import clozn.runs.store as runlog

    body = body if isinstance(body, dict) else {}

    if p.startswith("/runs/") and p.endswith("/corrective-actions/preview"):
        run_id = p[len("/runs/"):-len("/corrective-actions/preview")]
        run = runlog.get_run(run_id)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        # Same "compose, don't block" reasoning as GET .../corrective-actions above: a preview
        # never executes anything, so an unresolvable worker degrades to steer=None instead of
        # refusing.
        from clozn.server.model_routing import peek_control_model_for_run
        sub = peek_control_model_for_run(
            h, run.get("model"), route="/runs/<id>/corrective-actions/preview")
        try:
            preview = corrective_flow.create_preview(
                run,
                str(body.get("action_id") or ""),
                str(body.get("requested_backend") or "prompt_policy"),
                steer=getattr(sub, "steer", None),
                active_profile=ctx._active_profile_name(),
            )
        except corrective_flow.CorrectiveFlowError as exc:
            return _error(h, exc, status=400)
        h._json(201, preview)
        return True

    if p.startswith("/corrective-previews/") and p.endswith("/confirm"):
        preview_id = p[len("/corrective-previews/"):-len("/confirm")]
        # Read the preview from the flow receipt so the route never trusts a caller-supplied run id.
        preview = corrective_flow.get_preview(preview_id)
        if not isinstance(preview, dict):
            h._json(404, {"error": "corrective action preview not found"})
            return True
        run = runlog.get_run(str(preview.get("parent_run_id") or ""))
        if run is None:
            h._json(409, {"error": "parent run is missing; refusing confirmation"})
            return True
        # Confirmation actually regenerates through the product model -- unlike the preview above,
        # this must fail closed (typed refusal, never SUB) when the run's own model has no ready
        # worker under a managed gateway.
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(
            h, run.get("model"), route="/corrective-previews/<id>/confirm")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and callable(getattr(sub, "chat", None))):
            h._json(503, {"error": "corrective action requires a ready product model worker"})
            return True
        from clozn.server.routes.corrective_retries import _identity_conflict
        mismatch = _identity_conflict(run, sub)
        if mismatch:
            h._json(409, {
                "error": f"active worker {mismatch} does not match the target run"
            })
            return True
        from clozn.replay.corrective import retry_compare

        def execute(saved_preview):
            return retry_compare(
                run,
                saved_preview["action"]["id"],
                sub,
                backend=saved_preview["execution"]["requested_backend"],
                structured=True,
            )

        try:
            result = corrective_flow.confirm_preview(
                preview_id,
                str(body.get("idempotency_key") or ""),
                run,
                execute,
            )
        except corrective_flow.CorrectiveFlowError as exc:
            return _error(h, exc)
        h._json(200, result)
        return True

    if p.startswith("/corrective-previews/") and p.endswith("/cancel"):
        preview_id = p[len("/corrective-previews/"):-len("/cancel")]
        try:
            preview = corrective_flow.cancel_preview(preview_id)
        except corrective_flow.CorrectiveFlowError as exc:
            return _error(h, exc)
        h._json(200, preview)
        return True

    if p.startswith("/corrective-results/") and p.endswith("/keep"):
        result_id = p[len("/corrective-results/"):-len("/keep")]
        try:
            result = corrective_flow.keep_result(
                result_id,
                str(body.get("scope") or ""),
                str(body.get("expected_prior_hash") or ""),
                str(body.get("idempotency_key") or ""),
                get_run=runlog.get_run,
                replace_run=runlog.replace_run,
            )
        except corrective_flow.CorrectiveFlowError as exc:
            return _error(h, exc)
        h._json(200, result)
        return True

    if p.startswith("/corrective-results/") and p.endswith("/source-use"):
        result_id = p[len("/corrective-results/"):-len("/source-use")]
        try:
            comparison = corrective_flow.compare_source_use(
                result_id, get_run=runlog.get_run
            )
        except corrective_flow.CorrectiveFlowError as exc:
            return _error(h, exc)
        h._json(200, comparison)
        return True

    if p.startswith("/corrective-actions/") and p.endswith("/undo"):
        transaction_id = p[len("/corrective-actions/"):-len("/undo")]
        try:
            result = corrective_flow.undo_keep(
                transaction_id,
                get_run=runlog.get_run,
                replace_run=runlog.replace_run,
            )
        except corrective_flow.CorrectiveFlowError as exc:
            return _error(h, exc)
        h._json(200, result)
        return True
    return False
