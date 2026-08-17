"""Product Minimal Context route over the generic experiment kernel."""
from __future__ import annotations

from collections.abc import Mapping

from clozn.recipes.minimal_context import MinimalContextUnavailable, run_minimal_context
from clozn.recipes.minimal_context_result_store import (
    MinimalContextResultStore, MinimalContextResultStoreError, current_binding,
)

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
        payload = result.to_dict()
        # The commit boundary is deliberately after search and validation but
        # before the job exposes a terminal result.  A cancellation racing
        # this boundary therefore cannot leave a durable result that the job
        # reports as cancelled.
        if callable(getattr(control, "commit", None)) and payload.get("schema_version") == "clozn.minimal-context-search-result.v2":
            result_store = MinimalContextResultStore()
            control.commit(lambda: (result_store.put(result) is not None))
        control.attach_result(payload)
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


def try_get(h, path):
    parsed = _split(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "/results":
        run = _get_run_or_404(h, run_id)
        if run is None:
            return True
        try:
            store = MinimalContextResultStore()
            summaries = store.list_for_run(run_id, limit=50)
            for summary in summaries:
                result = store.get(summary["result_id"])
                summary["current_binding"] = current_binding(result, run) if result is not None else {
                    "status": "run_unavailable", "reason": "result could not be loaded",
                }
        except MinimalContextResultStoreError as exc:
            h._json(500, {"error": str(exc), "code": "minimal_context_result_unavailable"})
            return True
        h._json(200, {"results": summaries})
        return True
    if tail.startswith("/results/"):
        result_id = tail[len("/results/"):]
        if not result_id or "/" in result_id:
            return False
        run = _get_run_or_404(h, run_id)
        if run is None:
            return True
        try:
            result = MinimalContextResultStore().get(result_id)
        except MinimalContextResultStoreError as exc:
            h._json(500, {"error": str(exc), "code": "minimal_context_result_unavailable"})
            return True
        if result is None or result.run_id != run_id:
            h._json(404, {"error": "Minimal Context result not found", "code": "minimal_context_result_not_found"})
        else:
            payload = result.to_dict()
            payload["current_binding"] = current_binding(result, run)
            h._json(200, payload)
        return True
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
    if tail.startswith("/results/") and tail.endswith("/materialize"):
        return _materialize_result(h, run_id, tail[len("/results/"):-len("/materialize")], body)
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


def _materialize_result(h, run_id: str, result_id: str, body: object) -> bool:
    """Materialize only the winner bound by a persisted result document."""
    import clozn.runs.store as runlog
    from clozn.experiments.interventions import DeleteSource
    from clozn.experiments.materialize import MaterializationError, materialize_arm
    from clozn.experiments.persistence import ObservationStore
    from clozn.server.model_routing import select_control_model_for_run

    if not result_id or "/" in result_id:
        return False
    parent = runlog.get_run(run_id)
    if parent is None:
        h._json(404, {"error": "run not found", "code": "run_not_found"})
        return True
    if body not in (None, {}) and not (isinstance(body, Mapping) and not body):
        h._json(400, {
            "error": "winner references are bound to the persisted result; request body must be empty",
            "code": "winner_reference_override_forbidden",
        })
        return True
    try:
        result_store = MinimalContextResultStore()
        result = result_store.get(result_id)
    except MinimalContextResultStoreError as exc:
        h._json(500, {"error": str(exc), "code": "minimal_context_result_unavailable"})
        return True
    if result is None or result.run_id != run_id:
        h._json(404, {"error": "Minimal Context result not found", "code": "minimal_context_result_not_found"})
        return True
    binding = current_binding(result, parent)
    if binding["status"] != "current":
        h._json(409, {"error": binding["reason"], "code": "minimal_context_result_stale",
                      "current_binding": binding})
        return True
    if result.status != "completed":
        h._json(409, {"error": "only a completed Minimal Context result can be materialized",
                      "code": "minimal_context_result_not_completed"})
        return True
    if result.certificate not in {"BEST_VERIFIED", "INCLUSION_MINIMUM", "EXACT_MINIMUM"} or result.best is None:
        h._json(409, {"error": "Minimal Context result has no recognized materializable winner",
                      "code": "minimal_context_winner_unavailable"})
        return True
    winner = result.best
    if not winner.removed_source_ids:
        h._json(409, {"error": "the winning context makes no reduction",
                      "code": "no_reduction_to_materialize"})
        return True
    universe_ids = tuple(result.universe.get("source_ids") or [])
    if (
        len(set(winner.retained_source_ids)) != len(winner.retained_source_ids)
        or len(set(winner.removed_source_ids)) != len(winner.removed_source_ids)
        or set(winner.retained_source_ids) | set(winner.removed_source_ids) != set(universe_ids)
        or set(winner.retained_source_ids) & set(winner.removed_source_ids)
    ):
        h._json(409, {"error": "winner source sets do not match the recorded universe",
                      "code": "minimal_context_winner_source_binding_mismatch"})
        return True
    if not all(isinstance(value, str) and value for value in (
        winner.experiment_id, winner.arm_id, winner.observation_id,
    )):
        h._json(409, {"error": "winner is missing durable experiment references",
                      "code": "minimal_context_winner_evidence_missing"})
        return True

    observations = ObservationStore()
    try:
        experiment = observations.get_experiment(winner.experiment_id)
        arm = experiment.arm_for(winner.arm_id)
        observation = observations.get_observation(winner.observation_id)
    except Exception as exc:
        h._json(409, {"error": str(exc), "code": "minimal_context_winner_evidence_missing"})
        return True
    if arm.observation_id != winner.observation_id or arm.observation is None or observation.status != "exact_preserved":
        h._json(409, {"error": "winner observation binding is not exact_preserved",
                      "code": "minimal_context_winner_evidence_mismatch"})
        return True
    if not isinstance(arm.intervention, DeleteSource):
        h._json(409, {"error": "winner arm is not a DeleteSource condition",
                      "code": "minimal_context_winner_intervention_mismatch"})
        return True
    if tuple(arm.intervention.source_ids) != tuple(winner.removed_source_ids):
        h._json(409, {"error": "winner arm deletion does not match the persisted result",
                      "code": "minimal_context_winner_intervention_mismatch"})
        return True

    selection = select_control_model_for_run(h, parent.get("model"), route="/runs/<id>/minimal-context/results/materialize")
    if selection is None:
        return True
    try:
        outcome = materialize_arm(
            parent, winner.experiment_id, winner.arm_id, substrate=selection.sub,
            reload_parent=runlog.get_run, observation_id=winner.observation_id,
            require_preserved=True, observation_store=observations,
            materialization_context={
                "minimal_context": {
                    "recipe": "minimal_context",
                    "search_id": result.search_id,
                    "result_id": result.result_id,
                    "certificate": result.certificate,
                    "objective": dict(result.objective),
                    "retained_source_ids": list(winner.retained_source_ids),
                    "removed_source_ids": list(winner.removed_source_ids),
                },
            },
        )
    except MaterializationError as exc:
        h._json(409, {"error": str(exc), "code": "minimal_context_materialization_rejected"})
        return True
    h._json(201 if outcome.get("state") == "completed" else 409, outcome)
    return True


__all__ = [
    "CLOZN_ROUTE_AUTOLOAD", "MinimalContextRouteError", "normalize_request",
    "planned_universe", "try_get", "try_post",
]
