"""Prompt-first corrective retry: a request-local counterfactual debugging route.

``POST /runs/<id>/retry`` regenerates a matched greedy baseline and a corrected candidate (baseline
plus one bounded system instruction) and returns the comparison. Nothing here persists a policy for a
later, unrelated request to pick up -- durable, auto-applied corrections (session/profile scope,
persistent activation, explicit undo) were retired; see docs/CAPABILITIES.md. ``policy.status`` is
always ``"request_local"``, kept as a small compatibility field for existing consumers.
"""
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
        # Optional -- absent/omitted keeps the default prompt_policy behavior byte-for-byte (spec:
        # "must not expose raw scientific dials as the default interaction"). A calibrated named-dial
        # backend ("control_vector") existed here before named-dial personalization was retired; the
        # corrected arm is prompt_policy-only now -- see retry_compare's docstring.
        backend = body.get("backend")
        from clozn.replay.corrective import CORRECTION_PRESETS, retry_compare
        if preset not in CORRECTION_PRESETS:
            h._json(400, {"error": "preset must be one of: " + ", ".join(CORRECTION_PRESETS)})
            return True
        if backend is not None and backend != "prompt_policy":
            h._json(400, {"error": "backend must be omitted or 'prompt_policy'"})
            return True
        mismatch = _identity_conflict(run, sub)
        if mismatch:
            h._json(409, {"error": f"active worker {mismatch} does not match the target run"})
            return True

        try:
            comparison = retry_compare(run, preset, sub, backend=backend)
        except ValueError as exc:
            h._json(400, {"error": str(exc)})
            return True
        if comparison is None:
            h._json(500, {"error": "corrective retry comparison failed"})
            return True

        policy = {"status": "request_local", "scope": "once", "target": None, "presets": [preset]}
        undo = {"status": "automatic_restored", "available": False,
                "note": "the intervention applied only to the candidate replay"}
        h._json(200, {**comparison, "policy": policy, "undo": undo})
        return True
    return False
