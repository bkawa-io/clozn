"""Product Minimal Context route over the generic experiment kernel."""
from __future__ import annotations

from collections.abc import Mapping

from clozn.recipes.minimal_context import MinimalContextUnavailable, run_minimal_context

CLOZN_ROUTE_AUTOLOAD = True

_MARKER = "/minimal-context"
_KIND = "minimal_context"


class MinimalContextRouteError(ValueError):
    def __init__(self, message: str, *, code: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _split(path: str):
    if not path.startswith("/runs/"):
        return None
    rest = path[len("/runs/"): ]
    run_id, marker, tail = rest.partition(_MARKER)
    if marker != _MARKER or not run_id:
        return None
    return run_id, tail


def normalize_request(body: object) -> dict:
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise MinimalContextRouteError("request body must be an object", code="invalid_body")
    preservation = body.get("preservation")
    if preservation is not None:
        if not isinstance(preservation, Mapping):
            raise MinimalContextRouteError("preservation must be an object", code="invalid_preservation")
        kind = preservation.get("kind")
        if kind == "teacher_forced_likelihood":
            raise MinimalContextRouteError(
                "teacher_forced_likelihood is no longer a Minimal Context product mode; use Context Effects",
                code="minimal_context_mode_removed",
            )
        if kind not in {None, "exact_recorded_output"}:
            raise MinimalContextRouteError(
                "Minimal Context supports exact_recorded_output only",
                code="unsupported_preservation",
            )
    universe = body.get("universe") or {}
    if not isinstance(universe, Mapping):
        raise MinimalContextRouteError("universe must be an object", code="invalid_universe")
    max_units = universe.get("max_units", 50)
    max_new = body.get("max_new_counterfactual_observations", 32)
    for name, value in (("universe.max_units", max_units), ("max_new_counterfactual_observations", max_new)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MinimalContextRouteError(f"{name} must be a non-negative integer", code="invalid_budget")
    if max_units == 0:
        raise MinimalContextRouteError("universe.max_units must be positive", code="invalid_universe")
    attempt = body.get("attempt_inclusion_check", True)
    if not isinstance(attempt, bool):
        raise MinimalContextRouteError("attempt_inclusion_check must be a boolean", code="invalid_search_policy")
    return {
        "max_units": max_units,
        "max_new_counterfactual_observations": max_new,
        "attempt_inclusion_check": attempt,
    }


def planned_universe(run: Mapping[str, object], request: Mapping[str, object]) -> dict:
    from clozn.runs.context_search_universe import plan_context_search_universe
    return plan_context_search_universe(
        run, run.get("context_units"), max_units=int(request["max_units"]),
    )


def _get_run_or_404(h, run_id: str):
    import clozn.runs.store as runlog
    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found", "code": "minimal_context_run_not_found"})
        return None
    return run


def _job_worker(run: dict, sub, engine, request: dict, universe: dict):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.experiments.persistence import ObservationStore
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
            control.checkpoint(phase="planning_context", completed=1, total=1)
            result = run_minimal_context(
                latest, max_new_counterfactual_observations=request["max_new_counterfactual_observations"],
                max_units=request["max_units"], attempt_inclusion_check=request["attempt_inclusion_check"],
                substrate=sub, engine=engine, observation_store=ObservationStore(),
                cancel=control.cancel_requested,
            )
        except JobCancelled:
            raise
        except MinimalContextUnavailable as exc:
            return {"state": "failed", "error": {"code": exc.reason, "message": str(exc)}}
        except Exception as exc:
            return {"state": "failed", "error": {
                "code": "minimal_context_job_failed",
                "message": f"{type(exc).__name__}: {exc}",
            }}
        control.checkpoint(
            phase="validating", completed=1, total=1,
            best_retained_source_count=len(result.best.retained_source_ids) if result.best else None,
            certificate_candidate_kind=result.certificate,
        )
        control.attach_result(result.to_dict())
        return {"state": "completed"}

    return worker


def _start_job(h, run_id: str, body: dict) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run_or_404(h, run_id)
    if run is None:
        return True
    try:
        request = normalize_request(body)
        universe = planned_universe(run, request)
    except MinimalContextRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    except Exception as exc:
        h._json(409, {"error": str(exc), "code": "minimal_context_universe_unavailable"})
        return True
    if universe.get("status") != "planned":
        condition = universe.get("condition") if isinstance(universe.get("condition"), Mapping) else {}
        h._json(409, {
            "error": condition.get("message", "search universe is unavailable"),
            "code": condition.get("code", "minimal_context_universe_unavailable"),
            "universe": universe,
        })
        return True

    selection = select_control_model_for_run(h, run.get("model"), route="/runs/<id>/minimal-context/jobs")
    if selection is None:
        return True
    if not callable(getattr(selection.sub, "probe_reference_match", None)):
        h._json(503, {
            "error": "exact recorded-answer probes are unavailable",
            "code": "minimal_context_exact_capability_unavailable",
        })
        return True
    try:
        job = JOBS.start(
            run_id, _job_worker(run, selection.sub, selection.engine, request, universe), kind=_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "minimal_context_job_capacity"})
        return True
    h._json(202, job)
    return True


def _get_run_or_jobs(h, run_id: str) -> bool:
    run = _get_run_or_404(h, run_id)
    if run is None:
        return True
    h._json(200, {"run_id": run_id, "results": []})
    return True


def try_get(h, path):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "":
        return _get_run_or_jobs(h, run_id)
    if tail.startswith("/jobs/"):
        job_id = tail[len("/jobs/"): ]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Minimal Context job not found", "code": "minimal_context_job_not_found"})
        else:
            h._json(200, job)
        return True
    return False


def try_post(h, path, body):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "/jobs":
        return _start_job(h, run_id, body if isinstance(body, dict) else {})
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


__all__ = [
    "CLOZN_ROUTE_AUTOLOAD", "MinimalContextRouteError", "normalize_request",
    "planned_universe", "try_get", "try_post",
]
