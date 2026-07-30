"""Compute/attach (POST) and export (GET) the fast context<->answer influence map for a recorded run."""


_MAX_CONTEXT_SPANS = 8
_SCHEMA = "clozn.context_answer_influence.v1"
_EXPORT_SCHEMA = "clozn.context-answer-influence-export.v1"


def _job_path(p: str):
    """Return (run_id, job_id|None, cancel) for exact async job routes."""
    parts = p.strip("/").split("/")
    if len(parts) == 4 and parts[0] == "runs" and parts[2:] == ["influence-map", "jobs"]:
        return parts[1], None, False
    if (
        len(parts) == 5
        and parts[0] == "runs"
        and parts[2:4] == ["influence-map", "jobs"]
    ):
        return parts[1], parts[4], False
    if (
        len(parts) == 6
        and parts[0] == "runs"
        and parts[2:4] == ["influence-map", "jobs"]
        and parts[5] == "cancel"
    ):
        return parts[1], parts[4], True
    return None


def _export(h, rid: str, privacy: str) -> bool:
    import clozn.runs.store as runlog
    from clozn import schemas
    from clozn.receipts.context_answer_influence import portable_export

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
    try:
        exported = portable_export(stored, privacy=privacy)
        schemas.validate(exported, _EXPORT_SCHEMA)
    except (ValueError, schemas.ValidationError) as exc:
        h._json(400, {"error": str(exc)})
        return True
    h._json(200, exported)
    return True


def try_get(h, p):
    """GET /runs/<id>/influence-map -- the persistence/export path (Phase 3.7): return the already-
    computed, versioned evidence object exactly as stored, never triggering a new scoring job. A pure
    journal read, so it works even with no worker attached -- the counterpart to POST, which computes."""
    job_route = _job_path(p)
    if job_route is not None:
        rid, job_id, cancel = job_route
        if job_id is None or cancel:
            return False
        from clozn.server.influence_jobs import JOBS

        job = JOBS.get(rid, job_id)
        if job is None:
            h._json(404, {"error": "influence-map job not found"})
        else:
            h._json(200, job)
        return True

    if p.startswith("/runs/") and p.endswith("/influence-map/export"):
        rid = p[len("/runs/"):-len("/influence-map/export")]
        return _export(h, rid, "metadata_only")
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


def _job_worker(run: dict, sub, max_spans: int):
    """Build the closure run by the bounded local job executor."""
    def worker(control):
        import clozn.runs.store as runlog
        from clozn import schemas
        from clozn.receipts.context_answer_influence import (
            InfluenceComputationCancelled,
            SCHEMA,
            context_answer_influence,
        )
        from clozn.server.influence_jobs import JobCancelled

        try:
            result = context_answer_influence(
                run,
                sub,
                max_context_spans=max_spans,
                progress=control.checkpoint,
                cancel_requested=control.cancel_requested,
            )
        except InfluenceComputationCancelled as exc:
            raise JobCancelled(str(exc)) from None
        control.checkpoint(phase="validating", completed=0, total=1)
        if not isinstance(result, dict):
            return {
                "state": "failed",
                "error": {
                    "code": "influence_map_job_failed",
                    "message": "influence-map failed without an evidence object",
                },
            }
        try:
            schemas.validate(result, SCHEMA)
        except schemas.ValidationError as exc:
            return {
                "state": "failed",
                "error": {
                    "code": "influence_map_schema_invalid",
                    "message": f"influence-map artifact failed its schema: {exc}",
                },
            }
        control.checkpoint(phase="validating", completed=1, total=1)
        if result.get("available") is not True:
            error = result.get("error") if isinstance(result.get("error"), dict) else {}
            return {
                "state": "failed",
                "error": {
                    "code": error.get("code") or "influence_map_unavailable",
                    "message": error.get("message") or "influence-map evidence is unavailable",
                    "artifact_status": result.get("status"),
                },
            }

        def persist():
            latest = runlog.get_run(run.get("id"))
            if latest is None:
                return False
            updated = dict(latest)
            updated["influence_map"] = result
            return runlog.replace_run(updated)

        control.commit(persist)
        return {"state": "completed"}

    return worker


def _start_job(h, rid: str, body: dict) -> bool:
    import clozn.runs.store as runlog
    from clozn.receipts.context_answer_influence import cache_matches
    from clozn.server.influence_jobs import JOBS, JobCapacityError

    run = runlog.get_run(rid)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True
    try:
        max_spans = _max_spans(body)
    except ValueError as exc:
        h._json(400, {"error": str(exc)})
        return True

    cached = run.get("influence_map")
    try:
        if cache_matches(run, cached) and not body.get("refresh"):
            job = JOBS.start(rid, cached=True)
            h._json(202, job, extra_headers={"X-Clozn-Influence-Cache": "hit"})
            return True
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(
            h, run.get("model"), route="/runs/<id>/influence-map/jobs")
        if selection is None:
            return True   # typed clozn.model-routing.v1 refusal already written
        sub = selection.sub
        if not (sub and callable(getattr(sub, "score_tokens", None))):
            h._json(503, {"error": "influence-map requires worker token scoring"})
            return True
        job = JOBS.start(rid, _job_worker(run, sub, max_spans))
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, job, extra_headers={"X-Clozn-Influence-Cache": "miss"})
    return True


def try_post(h, p, body):
    job_route = _job_path(p)
    if job_route is not None:
        rid, job_id, cancel = job_route
        body = body if isinstance(body, dict) else {}
        if job_id is None and not cancel:
            return _start_job(h, rid, body)
        if job_id is not None and cancel:
            from clozn.server.influence_jobs import JOBS

            job = JOBS.cancel(rid, job_id)
            if job is None:
                h._json(404, {"error": "influence-map job not found"})
            else:
                h._json(200, job)
            return True
        return False

    if p.startswith("/runs/") and p.endswith("/influence-map/export"):
        rid = p[len("/runs/"):-len("/influence-map/export")]
        body = body if isinstance(body, dict) else {}
        return _export(h, rid, body.get("privacy", "metadata_only"))
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
    from clozn.receipts.context_answer_influence import cache_matches
    if cache_matches(run, cached) and not body.get("refresh"):
        h._json(200, cached, extra_headers={"X-Clozn-Influence-Cache": "hit"})
        return True

    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/influence-map")
    if selection is None:
        return True   # typed clozn.model-routing.v1 refusal already written
    sub = selection.sub
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
    h._json(200, result, extra_headers={"X-Clozn-Influence-Cache": "miss"})
    return True
