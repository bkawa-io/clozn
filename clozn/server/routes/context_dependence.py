"""Async, persisted Context Dependence studies for recorded runs.

This is intentionally a sibling of, never a replacement for,
``/runs/<id>/influence-map``.  GETs only return a completed separately-stored
artifact; POST starts one bounded shared-registry job selected against the
recorded run's model.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_SCHEMA = "clozn.context-dependence-study.v1"
_MARKER = "/context-dependence"
_ROUTE = "/runs/<id>/context-dependence/jobs"


def _split(path: str):
    """Return ``(run_id, tail)`` only for this exact route family."""
    if not path.startswith("/runs/"):
        return None
    rest = path[len("/runs/"):]
    run_id, marker, tail = rest.partition(_MARKER)
    if marker != _MARKER or not run_id:
        return None
    return run_id, tail


def _job_worker(run: dict, sub, request: dict):
    def worker(control):
        from clozn import schemas
        import clozn.runs.store as runlog
        from clozn.runs.context_dependence_execution import (
            ContextDependenceExecutionError,
            SCHEMA,
            run_context_dependence_execution,
        )
        from clozn.server.influence_jobs import JobCancelled

        try:
            artifact = run_context_dependence_execution(
                run, sub, request, checkpoint=control.checkpoint,
            )
        except JobCancelled:
            raise
        except ContextDependenceExecutionError as exc:
            return {"state": "failed", "error": {
                "code": "context_dependence_execution_invalid",
                "message": str(exc),
            }}
        except Exception as exc:  # the job registry turns unexpected failures into terminal evidence
            return {"state": "failed", "error": {
                "code": "context_dependence_execution_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }}

        # A producer's own pre-persistence schema check is intentionally
        # repeated at this persistence boundary.  No malformed result reaches
        # a run attachment even if a test/subclass bypassed producer validation.
        control.checkpoint(phase="validating", completed=0, total=1)
        try:
            schemas.validate(artifact, SCHEMA)
        except schemas.ValidationError as exc:
            return {"state": "failed", "error": {
                "code": "context_dependence_schema_invalid",
                "message": f"Context Dependence artifact failed its schema: {exc}",
            }}
        control.checkpoint(phase="validating", completed=1, total=1)

        def persist():
            latest = runlog.get_run(run.get("id"))
            if latest is None:
                return False
            updated = dict(latest)
            updated["context_dependence_study"] = artifact
            return runlog.replace_run(updated)

        control.commit(persist)
        control.attach_result(artifact)
        return {"state": "completed"}

    return worker


def _get_artifact(h, run_id: str) -> bool:
    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True
    artifact = run.get("context_dependence_study")
    if not (isinstance(artifact, dict) and artifact.get("schema_version") == _SCHEMA):
        h._json(404, {
            "error": "no Context Dependence study has been computed for this run yet",
            "schema": _SCHEMA,
            "available": False,
        })
        return True
    h._json(200, artifact)
    return True


def _start_job(h, run_id: str, body: dict) -> bool:
    import clozn.runs.store as runlog
    from clozn.runs.context_dependence_execution import (
        ContextDependenceExecutionError,
        cache_matches,
        normalize_request,
    )
    from clozn.server.influence_jobs import JOBS, JobCapacityError

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True
    try:
        request = normalize_request(body)
    except ContextDependenceExecutionError as exc:
        h._json(400, {"error": str(exc)})
        return True

    cached = run.get("context_dependence_study")
    try:
        if cache_matches(run, cached, request) and not request["refresh"]:
            job = JOBS.start(run_id, cached=True, kind="context_dependence")
            h._json(202, job, extra_headers={"X-Clozn-Context-Dependence-Cache": "hit"})
            return True
        # Never select an ambient worker.  This resolves the immutable recorded
        # run model and produces the usual typed managed-routing refusal.
        from clozn.server.model_routing import select_control_model_for_run
        selection = select_control_model_for_run(h, run.get("model"), route=_ROUTE)
        if selection is None:
            return True
        sub = selection.sub
        if not (sub and callable(getattr(sub, "score_tokens", None))):
            h._json(503, {
                "error": "Context Dependence requires worker token scoring",
                "code": "context_dependence_worker_unavailable",
            })
            return True
        job = JOBS.start(run_id, _job_worker(run, sub, request), kind="context_dependence")
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc)})
        return True
    h._json(202, job, extra_headers={"X-Clozn-Context-Dependence-Cache": "miss"})
    return True


def try_get(h, path):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "":
        return _get_artifact(h, run_id)
    if not tail.startswith("/jobs/") or tail.endswith("/cancel"):
        return False
    job_id = tail[len("/jobs/"):]
    if not job_id or "/" in job_id:
        return False
    from clozn.server.influence_jobs import JOBS
    job = JOBS.get(run_id, job_id)
    if job is None:
        h._json(404, {"error": "Context Dependence job not found"})
    else:
        h._json(200, job)
    return True


def try_post(h, path, body):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    body = body if isinstance(body, dict) else {}
    if tail == "/jobs":
        return _start_job(h, run_id, body)
    if tail.startswith("/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.cancel(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Context Dependence job not found"})
        else:
            h._json(200, job)
        return True
    return False


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get", "try_post"]
