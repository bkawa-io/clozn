"""Compute/attach (POST) and export (GET) the fast context<->answer influence map for a recorded run."""
from clozn.server import app as ctx


_MAX_CONTEXT_SPANS = 8
_SCHEMA = "clozn.context_answer_influence.v1"


def try_get(h, p):
    """GET /runs/<id>/influence-map -- the persistence/export path (Phase 3.7): return the already-
    computed, versioned evidence object exactly as stored, never triggering a new scoring job. A pure
    journal read, so it works even with no worker attached -- the counterpart to POST, which computes."""
    if not (p.startswith("/runs/") and p.endswith("/influence-map")):
        return False

    rid = p[len("/runs/"):-len("/influence-map")]
    import clozn.runs.store as runlog

    run = runlog.get_run(rid)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    stored = run.get("influence_map")
    if not (isinstance(stored, dict) and stored.get("schema") == _SCHEMA):
        h._json(404, {
            "error": "no context-answer influence map has been computed for this run yet",
            "schema": _SCHEMA,
            "available": False,
        })
        return True
    h._json(200, stored)
    return True


def _max_spans(body: dict) -> int:
    raw = body.get("max_context_spans", _MAX_CONTEXT_SPANS)
    if isinstance(raw, bool):
        raise ValueError("max_context_spans must be an integer from 1 to 8")
    try:
        value = int(raw)
    except (TypeError, ValueError):
        raise ValueError("max_context_spans must be an integer from 1 to 8") from None
    if value < 1 or value > _MAX_CONTEXT_SPANS:
        raise ValueError("max_context_spans must be an integer from 1 to 8")
    return value


def try_post(h, p, body):
    if not (p.startswith("/runs/") and p.endswith("/influence-map")):
        return False

    rid = p[len("/runs/"):-len("/influence-map")]
    import clozn.runs.store as runlog

    run = runlog.get_run(rid)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True

    body = body if isinstance(body, dict) else {}
    cached = run.get("influence_map")
    if (isinstance(cached, dict)
            and cached.get("schema") == _SCHEMA
            and cached.get("available") is True and not body.get("refresh")):
        h._json(200, cached)
        return True

    sub = ctx.active_sub(h)
    if not (sub and callable(getattr(sub, "score_tokens", None))):
        h._json(503, {"error": "influence-map requires worker token scoring"})
        return True

    try:
        max_spans = _max_spans(body)
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    from clozn.receipts.context_answer_influence import SCHEMA, context_answer_influence

    result = context_answer_influence(run, sub, max_context_spans=max_spans)
    if not isinstance(result, dict):
        h._json(500, {"error": "influence-map failed without an evidence object"})
        return True

    # Seam 2: validate every freshly-computed shape (ok, unavailable, or error) against its registered
    # schema before it is persisted or returned -- a malformed artifact must fail loudly here, not sail
    # through and mislead a reader later (roadmap rule 3: no silent fallback). Cached reads (the branch
    # above) and the pure GET export path (try_get) are intentionally NOT re-validated on every read --
    # this is a write-boundary check, not a read tax on already-stored data.
    from clozn import schemas

    try:
        schemas.validate(result, SCHEMA)
    except schemas.ValidationError as exc:
        h._json(500, {"error": f"influence-map produced an artifact that failed its own schema: {exc}"})
        return True

    if result.get("available") is not True:
        status = 500 if result.get("status") == "error" else 422
        h._json(status, result)
        return True

    # The run record is immutable evidence plus explicit derived attachments.  Persisting the map makes
    # the Studio view, JSON export, and offline HTML card all render the exact same scored artifact.
    updated = dict(run)
    updated["influence_map"] = result
    if not runlog.replace_run(updated):
        h._json(500, {"error": "influence-map was computed but could not be attached to the run"})
        return True
    h._json(200, result)
    return True
