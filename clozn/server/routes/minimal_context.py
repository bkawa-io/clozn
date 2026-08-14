"""Cancellable, cache-bound Minimal Context jobs and immutable result lookup."""
from __future__ import annotations

CLOZN_ROUTE_AUTOLOAD = True

_MARKER = "/minimal-context"
_KIND = "minimal_context"


def _split(path: str):
    if not path.startswith("/runs/"):
        return None
    rest = path[len("/runs/"):]
    run_id, marker, tail = rest.partition(_MARKER)
    if marker != _MARKER or not run_id:
        return None
    return run_id, tail


def _job_worker(run: dict, sub, request: dict, universe: dict, binding: dict):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn import schemas
        from clozn.runs.minimal_context_execution import (
            MinimalContextExecutionError,
            cache_binding,
            execute_minimal_context,
            planned_universe,
        )
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {
                "code": "minimal_context_run_deleted",
                "message": "recorded run no longer exists",
            }}
        try:
            control.checkpoint(phase="planning_context", completed=0, total=1)
            current_universe = planned_universe(latest, request)
            if current_universe.get("universe_id") != universe.get("universe_id"):
                return {"state": "failed", "error": {
                    "code": "minimal_context_run_changed",
                    "message": "recorded run or Context Search Universe changed while the job was queued",
                }}
            if cache_binding(latest, request, current_universe, sub) != binding:
                return {"state": "failed", "error": {
                    "code": "minimal_context_run_changed",
                    "message": "recorded run or runtime identity changed while the job was queued",
                }}
            control.checkpoint(phase="planning_context", completed=1, total=1)
            if request["preservation"]["kind"] == "exact_recorded_output":
                control.checkpoint(phase="checking_exact_eligibility", completed=0, total=1)
                control.checkpoint(phase="checking_exact_eligibility", completed=1, total=1)
            result, support = execute_minimal_context(
                latest, sub, request, universe, binding, checkpoint=control.checkpoint,
                cancel=control.cancel_requested,
            )
            candidate = result.get("candidate") if isinstance(result.get("candidate"), dict) else {}
            certificate = result.get("certificate") if isinstance(result.get("certificate"), dict) else {}
            control.checkpoint(
                phase="validating", completed=0, total=1,
                best_retained_source_count=candidate.get("retained_source_count")
                if isinstance(candidate.get("retained_source_count"), int) else None,
                certificate_candidate_kind=certificate.get("kind")
                if isinstance(certificate.get("kind"), str) else None,
            )
            schemas.validate(result, "clozn.minimal-context-result.v1")
        except JobCancelled:
            raise
        except MinimalContextExecutionError as exc:
            return {"state": "failed", "error": {"code": exc.code, "message": str(exc)}}
        except Exception as exc:
            return {"state": "failed", "error": {
                "code": "minimal_context_job_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }}

        def persist():
            current = runlog.get_run(latest.get("id"))
            if current is None:
                return False
            results = current.get("minimal_context_results")
            results = dict(results) if isinstance(results, dict) else {}
            results[result["result_id"]] = result
            current["minimal_context_results"] = results
            studies = current.get("minimal_context_support")
            studies = dict(studies) if isinstance(studies, dict) else {}
            if "unavailable" in studies:
                studies = {}
            support_id = support.get("study_id") if isinstance(support, dict) else None
            if isinstance(support_id, str):
                studies[support_id] = support
            current["minimal_context_support"] = studies
            universes = current.get("minimal_context_universes")
            universes = dict(universes) if isinstance(universes, dict) else {}
            universes[universe["universe_id"]] = universe
            current["minimal_context_universes"] = universes
            return runlog.replace_run(current)

        control.commit(persist)
        control.attach_result(result)
        return {"state": "completed"}

    return worker


def _get_run_or_404(h, run_id: str):
    import clozn.runs.store as runlog
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found", "code": "minimal_context_run_not_found"})
        return None
    return run


def _start_job(h, run_id: str, body: dict) -> bool:
    import clozn.runs.store as runlog
    from clozn.runs.minimal_context_execution import (
        MinimalContextExecutionError,
        cache_binding,
        cache_matches,
        normalize_request,
        planned_universe,
    )
    from clozn.server.influence_jobs import JOBS, JobCapacityError

    run = _get_run_or_404(h, run_id)
    if run is None:
        return True
    try:
        request = normalize_request(body)
        universe = planned_universe(run, request)
    except MinimalContextExecutionError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    if universe.get("status") != "planned":
        h._json(409, {
            "error": universe.get("condition", {}).get("message", "search universe exceeds its bound"),
            "code": universe.get("condition", {}).get("code", "minimal_context_universe_bound_exceeded"),
            "universe": universe,
        })
        return True

    # Request validation and pure planning happen before this worker selection.
    # The target remains the recorded run's model identity, never ambient state.
    from clozn.server.model_routing import select_control_model_for_run
    selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/minimal-context/jobs")
    if selection is None:
        return True
    sub = selection.sub
    kind = request["preservation"]["kind"]
    if kind == "exact_recorded_output":
        if not callable(getattr(sub, "probe_reference_match", None)):
            h._json(503, {
                "error": "exact_recorded_output requires direct generation probes",
                "code": "minimal_context_exact_capability_unavailable",
            })
            return True
        from clozn.runs.answer_preservation import assess_exact_eligibility
        eligibility = assess_exact_eligibility(run, sub)
        if not eligibility.get("eligible"):
            h._json(409, {
                "error": eligibility.get("reason", "exact_recorded_output_ineligible"),
                "code": "minimal_context_exact_unavailable",
                "eligibility": eligibility,
            })
            return True
    elif not callable(getattr(sub, "score_tokens", None)):
        h._json(503, {
            "error": "teacher_forced_likelihood requires token scoring",
            "code": "minimal_context_likelihood_capability_unavailable",
        })
        return True

    binding = cache_binding(run, request, universe, sub)
    stored = run.get("minimal_context_results")
    candidates = stored.values() if isinstance(stored, dict) else ()
    cached_result = next((result for result in candidates if cache_matches(result, binding)), None)
    try:
        if cached_result is not None and not request["refresh"]:
            job = JOBS.start(
                run_id, cached=True, kind=_KIND, cached_result=cached_result,
            )
            h._json(202, job, extra_headers={"X-Clozn-Minimal-Context-Cache": "hit"})
            return True
        job = JOBS.start(
            run_id, _job_worker(run, sub, request, universe, binding), kind=_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "minimal_context_job_capacity"})
        return True
    h._json(202, job, extra_headers={"X-Clozn-Minimal-Context-Cache": "miss"})
    return True


def _get_result(h, run_id: str, result_id: str) -> bool:
    run = _get_run_or_404(h, run_id)
    if run is None:
        return True
    results = run.get("minimal_context_results")
    result = results.get(result_id) if isinstance(results, dict) else None
    if not isinstance(result, dict):
        h._json(404, {"error": "Minimal Context result not found", "code": "minimal_context_result_not_found"})
        return True
    from clozn import schemas
    try:
        schemas.validate(result, "clozn.minimal-context-result.v1")
    except Exception as exc:
        h._json(500, {"error": f"stored Minimal Context result is invalid: {exc}",
                      "code": "minimal_context_result_invalid"})
        return True
    h._json(200, result)
    return True


def _list_results(h, run_id: str) -> bool:
    run = _get_run_or_404(h, run_id)
    if run is None:
        return True
    from clozn.runs.minimal_context_execution import result_summary
    results = run.get("minimal_context_results")
    rows = [result_summary(result) for result in (results.values() if isinstance(results, dict) else ())
            if isinstance(result, dict)]
    rows.sort(key=lambda row: str(row.get("result_id") or ""))
    h._json(200, {"run_id": run_id, "results": rows})
    return True


def try_get(h, path):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "":
        return _list_results(h, run_id)
    if tail.startswith("/jobs/"):
        job_id = tail[len("/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Minimal Context job not found", "code": "minimal_context_job_not_found"})
        else:
            h._json(200, job)
        return True
    result_id = tail[1:] if tail.startswith("/") else ""
    if result_id and "/" not in result_id:
        return _get_result(h, run_id, result_id)
    return False


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
            h._json(404, {"error": "Minimal Context job not found", "code": "minimal_context_job_not_found"})
        else:
            h._json(200, job)
        return True
    return False


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get", "try_post"]
