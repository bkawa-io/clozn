"""Async, persisted Context Dependence studies for recorded runs.

This is intentionally a sibling of, never a replacement for,
``/runs/<id>/influence-map``.  GETs only return a completed separately-stored
artifact; POST starts one bounded shared-registry job selected against the
recorded run's model.
"""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_SCHEMA_V1 = "clozn.context-dependence-study.v1"
_SCHEMA_V2 = "clozn.context-dependence-study.v2"
# New computations persist the run-level v2 evidence.  Keep the name private
# and singular because the worker imports its authoritative schema from the
# execution module; this only controls route availability messaging.
_SCHEMA = _SCHEMA_V2
_READABLE_SCHEMAS = {_SCHEMA_V1, _SCHEMA_V2}
_MARKER = "/context-dependence"
_ROUTE = "/runs/<id>/context-dependence/jobs"
_REGENERATION_ROUTE = "/runs/<id>/context-dependence/experiments/<experiment_id>/regenerate"


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
    if not (isinstance(artifact, dict) and artifact.get("schema_version") in _READABLE_SCHEMAS):
        h._json(404, {
            "error": "no Context Dependence study has been computed for this run yet",
            "schema": _SCHEMA,
            "available": False,
        })
        return True
    h._json(200, artifact)
    return True


def _parse_query_range(h, path: str) -> tuple[int | None, int | None, str | None]:
    """Read exactly one strict pair of Unicode output coordinates.

    The server dispatches ``path`` without its query component but direct route
    tests and the HTTP handler retain it on ``h.path``.  Do not infer a value
    from a repeated parameter: callers must name one unambiguous selection.
    """
    from urllib.parse import parse_qs, urlsplit

    raw_path = getattr(h, "path", path)
    query = parse_qs(urlsplit(raw_path).query, keep_blank_values=True)
    starts, ends = query.get("output_start"), query.get("output_end")
    if not (isinstance(starts, list) and len(starts) == 1 and isinstance(ends, list) and len(ends) == 1):
        return None, None, "output_start and output_end are each required exactly once"

    def parse(value: str) -> int | None:
        if not value:
            return None
        try:
            # int accepts leading +/- signs but never float-looking values;
            # the projection independently rejects negative coordinates.
            return int(value)
        except (TypeError, ValueError):
            return None

    start, end = parse(starts[0]), parse(ends[0])
    if start is None or end is None:
        return None, None, "output_start and output_end must be base-10 integers"
    return start, end, None


def _query_artifact(h, run_id: str, path: str) -> bool:
    import clozn.runs.store as runlog
    from clozn.runs.context_dependence_projection import (
        ContextDependenceProjectionError,
        build_context_dependence_query,
    )

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found"})
        return True
    output_start, output_end, error = _parse_query_range(h, path)
    if error is not None:
        h._json(400, {"error": error, "code": "invalid_output_range"})
        return True
    try:
        # Pure projection: the module only checks the stored study and sums
        # its persisted full vectors.  In particular it selects no model and
        # starts no Context Dependence job.
        document = build_context_dependence_query(
            run, output_start=output_start, output_end=output_end,
        )
    except ContextDependenceProjectionError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    h._json(200, document)
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


def _regenerate_experiment(h, run_id: str, experiment_id: str) -> bool:
    """Generate exactly one child from an already measured deletion arm.

    Planning happens before worker selection.  Thus a missing/stale/tampered
    study has no route to a model call, and the executor repeats the plan
    against the latest stored parent immediately before its single replay.
    """
    import clozn.runs.store as runlog
    from clozn.runs.context_dependence_regeneration import (
        ContextDependenceRegenerationError,
        ContextDependenceRegenerationExperimentNotFoundError,
        ContextDependenceRegenerationStaleError,
        execute_context_dependence_regeneration,
        plan_context_dependence_regeneration,
    )

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {
            "error": "run not found",
            "code": "context_dependence_regeneration_run_not_found",
        })
        return True
    study = run.get("context_dependence_study")
    if not isinstance(study, dict):
        h._json(404, {
            "error": "no Context Dependence study has been computed for this run yet",
            "code": "context_dependence_regeneration_study_unavailable",
        })
        return True
    try:
        # Model-free strict re-resolution, including source bytes/ranges and
        # the measured score-context hash, before reaching model routing.
        plan = plan_context_dependence_regeneration(run, study, experiment_id)
    except ContextDependenceRegenerationExperimentNotFoundError as exc:
        h._json(404, {"error": str(exc), "code": "context_dependence_regeneration_experiment_not_found"})
        return True
    except ContextDependenceRegenerationStaleError as exc:
        h._json(409, {"error": str(exc), "code": "context_dependence_regeneration_stale"})
        return True
    except ContextDependenceRegenerationError as exc:
        h._json(409, {"error": str(exc), "code": "context_dependence_regeneration_invalid"})
        return True

    # The run record, not request data, determines the immutable worker
    # target.  The shared resolver writes its own typed model-routing refusal.
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, run.get("model"), route=_REGENERATION_ROUTE)
    if selection is None:
        return True
    sub = selection.sub
    if not (sub and callable(getattr(sub, "chat", None))):
        h._json(503, {
            "error": "Context Dependence regeneration requires a ready product model worker",
            "code": "context_dependence_regeneration_worker_unavailable",
        })
        return True
    try:
        result = execute_context_dependence_regeneration(
            run, study, experiment_id, sub,
            plan=plan,
            reload_parent=runlog.get_run,
        )
    except ContextDependenceRegenerationStaleError as exc:
        # The run could change between the pre-routing plan and execution;
        # execute() re-plans against the latest copy and refuses before chat.
        h._json(409, {"error": str(exc), "code": "context_dependence_regeneration_stale"})
        return True
    except ContextDependenceRegenerationError as exc:
        h._json(409, {"error": str(exc), "code": "context_dependence_regeneration_invalid"})
        return True
    if result.get("regeneration", {}).get("state") != "completed":
        h._json(502, {
            "error": "Context Dependence regeneration did not produce a child run",
            "code": "context_dependence_regeneration_failed",
            "result": result,
        })
        return True
    h._json(200, result)
    return True


def try_get(h, path):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "":
        return _get_artifact(h, run_id)
    if tail == "/query":
        return _query_artifact(h, run_id, path)
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
    experiment_prefix = "/experiments/"
    experiment_suffix = "/regenerate"
    if tail.startswith(experiment_prefix) and tail.endswith(experiment_suffix):
        experiment_id = tail[len(experiment_prefix):-len(experiment_suffix)]
        if not experiment_id or "/" in experiment_id:
            return False
        return _regenerate_experiment(h, run_id, experiment_id)
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
