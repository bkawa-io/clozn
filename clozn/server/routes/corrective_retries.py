"""Prompt-first corrective retry, persistent scope, and conflict-safe undo routes."""
from __future__ import annotations

from clozn.server import app as ctx


def _identity_conflict(run: dict, sub) -> str | None:
    recorded = run.get("identity") or {}
    if not recorded or not hasattr(sub, "identity_meta"):
        return None
    try:
        active = sub.identity_meta() or {}
    except Exception:
        return None
    for field in ("model_sha256", "template_fingerprint"):
        if recorded.get(field) and active.get(field) and recorded[field] != active[field]:
            return field
    return None


def try_post(h, p, body):
    if p.startswith("/runs/") and p.endswith("/retry"):
        rid = p[len("/runs/"):-len("/retry")]
        import clozn.runs.store as runlog
        run = runlog.get_run(rid)
        if run is None:
            h._json(404, {"error": "run not found"})
            return True
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/retry")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and callable(getattr(sub, "chat", None))):
            h._json(503, {"error": "corrective retry requires a ready product model worker"})
            return True
        preset = str(body.get("preset") or "")
        scope = str(body.get("scope") or "once")
        # Optional -- absent/omitted keeps the default prompt_policy behavior byte-for-byte (spec:
        # "must not expose raw scientific dials as the default interaction"). "control_vector" is an
        # explicit opt-in that may still fall back to prompt_policy (comparison["backend_fallback"])
        # when this exact model has no calibrated dial for `preset` -- see retry_compare's docstring.
        backend = body.get("backend")
        from clozn.replay.corrective import CORRECTION_PRESETS, retry_compare
        if preset not in CORRECTION_PRESETS:
            h._json(400, {"error": "preset must be one of: " + ", ".join(CORRECTION_PRESETS)})
            return True
        if scope not in {"once", "session", "profile"}:
            h._json(400, {"error": "scope must be once, session, or profile"})
            return True
        if backend is not None and backend not in ("prompt_policy", "control_vector"):
            h._json(400, {"error": "backend must be omitted, 'prompt_policy', or 'control_vector'"})
            return True
        mismatch = _identity_conflict(run, sub)
        if mismatch:
            h._json(409, {"error": f"active worker {mismatch} does not match the target run"})
            return True

        target = None
        if scope == "session":
            target = run.get("session_key")
            if not target:
                h._json(409, {"error": "that run has no exact session association; use --scope once"})
                return True
        elif scope == "profile":
            target = (run.get("meta") or {}).get("active_profile")
            if not target:
                h._json(409, {"error": "that run has no captured active profile; use --scope once"})
                return True
            if target != ctx._active_profile_name():
                h._json(409, {"error": "the profile that shaped that run is not currently active"})
                return True

        try:
            from clozn.behavior import corrective_retries
            active_presets = corrective_retries.effective_presets(
                session_key=run.get("session_key"),
                profile_name=ctx._active_profile_name(),
            )
            comparison = retry_compare(
                run, preset, sub, scope=scope, active_presets=active_presets, backend=backend,
            )
        except ValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        if comparison is None:
            h._json(500, {"error": "corrective retry comparison failed"})
            return True

        if scope == "once":
            policy = {"status": "request_local", "scope": "once", "target": None,
                      "presets": [preset], "undo_id": None}
            undo = {"status": "automatic_restored", "available": False,
                    "note": "the intervention applied only to the candidate replay"}
        elif comparison.get("coherence", {}).get("degenerate"):
            policy = {"status": "not_activated", "scope": scope, "target": target,
                      "reason": "candidate output was degenerate"}
            undo = {"status": "not_needed", "available": False}
        elif not comparison.get("intervention_observed"):
            policy = {"status": "not_activated", "scope": scope, "target": target,
                      "reason": "the correction was not present in survived prompt evidence"}
            undo = {"status": "not_needed", "available": False}
        else:
            try:
                policy = corrective_retries.activate(scope, target, preset)
            except corrective_retries.CorrectivePolicyError as exc:
                h._json(409, {"error": str(exc), "comparison": comparison})
                return True
            undo_id = policy.get("undo_id")
            undo = {"status": "available" if undo_id else "not_needed",
                    "available": bool(undo_id), "id": undo_id}
        h._json(200, {**comparison, "policy": policy, "undo": undo})
        return True

    if p.startswith("/corrective-retries/") and p.endswith("/undo"):
        transaction_id = p[len("/corrective-retries/"):-len("/undo")]
        from clozn.behavior import corrective_retries
        try:
            result = corrective_retries.undo(transaction_id)
        except corrective_retries.CorrectivePolicyError as exc:
            h._json(409, {"error": str(exc)})
            return True
        h._json(200, result)
        return True
    return False
