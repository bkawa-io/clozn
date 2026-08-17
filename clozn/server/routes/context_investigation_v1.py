"""Backend Context Investigation v1 routes over the generic experiment kernel.

The reader and answer query are pure reads. Explicit Q1/Q2/Q3 jobs are the
only paths that select a substrate and execute generic Experiments; Q3 ends in
durable GeneratedObservation evidence and only its explicit materialization
route can create a child Run.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
from urllib.parse import parse_qs, urlsplit

from clozn.experiments.context_investigation import (
    AnswerSelectionProjectionUnavailable,
    DEFAULT_MEASUREMENT_FLOOR_NATS,
    build_context_investigation_reader,
    query_answer_effects,
)
from clozn.experiments.persistence import (
    ExperimentPersistenceError,
    ObservationNotFound,
    ObservationStore,
)
from clozn.experiments.materialize import (
    MaterializationError, MaterializationStaleError,
    materialize_generated_observation,
)
from clozn.experiments.selections import AnswerSelection
from clozn.recipes.context_effects import measure_context_effects, plan_context_effects
from clozn.recipes.context_counterfactual import (
    ContextCounterfactualUnavailable, generate_without_source,
    plan_context_counterfactual,
)
from clozn.recipes.removability import can_remove, plan_removability


CLOZN_ROUTE_AUTOLOAD = True

_PREFIX = "/runs/"
_SUFFIX = "/context-investigation"
_KIND = "context_investigation_effects"
_REMOVE_KIND = "context_investigation_remove_test"
_COUNTERFACTUAL_KIND = "context_investigation_counterfactual"
_READER_SCHEMA = "clozn.context-investigation-reader.v1"


class ContextInvestigationRouteError(ValueError):
    def __init__(self, message: str, *, code: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _run_id(path: str) -> tuple[str, str] | None:
    clean = path.split("?", 1)[0]
    if not clean.startswith(_PREFIX):
        return None
    rest = clean[len(_PREFIX):]
    run_id, marker, tail = rest.partition(_SUFFIX)
    if marker != _SUFFIX or not run_id:
        return None
    return run_id, tail


def _get_run(h, run_id: str):
    import clozn.runs.store as runlog

    run = runlog.get_run(run_id)
    if run is None:
        h._json(404, {"error": "run not found", "code": "context_investigation_run_not_found"})
        return None
    return run


def _query_floor(h) -> float | None:
    query = parse_qs(urlsplit(getattr(h, "path", "")).query, keep_blank_values=True)
    raw = (query.get("floor_nats") or [None])[-1]
    if raw in {None, ""}:
        return DEFAULT_MEASUREMENT_FLOOR_NATS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ContextInvestigationRouteError(
            "floor_nats must be a finite non-negative number", code="invalid_floor"
        ) from None
    if not math.isfinite(value) or value < 0:
        raise ContextInvestigationRouteError(
            "floor_nats must be a finite non-negative number", code="invalid_floor"
        )
    return value


def _plan_or_error(h, run: Mapping[str, object], source_ids=None):
    try:
        return plan_context_effects(run, source_ids=source_ids)
    except Exception as exc:
        h._json(409, {
            "error": str(exc),
            "code": "context_investigation_plan_unavailable",
        })
        return None


def _reader(h, run_id: str) -> bool:
    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        floor = _query_floor(h)
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    plan = _plan_or_error(h, run)
    if plan is None:
        return True
    document = build_context_investigation_reader(
        run, plan, observation_store=ObservationStore(), floor_nats=floor,
    )
    h._json(200, document)
    return True


def _compact_job_result(result) -> dict:
    completed = unavailable = failed = not_measured = 0
    for row in result.arms:
        if row.observation is not None and row.observation.completed:
            completed += 1
            continue
        status = row.diagnostics.get("observation_status")
        if status == "unavailable":
            unavailable += 1
        elif status == "failed" or row.state == "failed":
            failed += 1
        else:
            not_measured += 1
    return {
        "schema_version": "clozn.context-investigation-effects-job-result.v1",
        "experiment_id": result.experiment_id,
        "source_count": len(result.arms),
        "completed_source_count": completed,
        "unavailable_source_count": unavailable,
        "failed_source_count": failed,
        "not_measured_source_count": not_measured,
        "state": result.state,
    }


def _job_worker(run: Mapping[str, object], sub, engine, source_ids, plan_id: str):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {
                "code": "context_investigation_run_deleted",
                "message": "recorded run no longer exists",
            }}
        current_plan = plan_context_effects(latest, source_ids=source_ids)
        if current_plan.experiment_id != plan_id:
            return {"state": "failed", "error": {
                "code": "context_investigation_run_changed",
                "message": "recorded Run or Context Receipt changed while the job was queued",
            }}
        control.checkpoint(
            phase="scoring_recorded_continuation",
            completed=0,
            total=len(current_plan.measurement_source_ids),
        )
        result = measure_context_effects(
            latest,
            source_ids=source_ids,
            substrate=sub,
            execution_adapter=None,
            observation_store=ObservationStore(),
            cancel=control.cancel_requested,
        )
        if control.cancel_requested():
            raise JobCancelled("Context Investigation effects job cancelled")
        control.checkpoint(
            phase="persisting_observations",
            completed=len(current_plan.measurement_source_ids),
            total=len(current_plan.measurement_source_ids),
        )
        control.attach_result(_compact_job_result(result))
        return {"state": "completed"}

    return worker


def _normalize_single_source_request(body: object, *, label: str) -> str:
    if not isinstance(body, Mapping):
        raise ContextInvestigationRouteError("request body must be an object", code="invalid_body")
    if set(body) != {"source_id"}:
        raise ContextInvestigationRouteError(
            f"only source_id is accepted for {label}", code=f"unsupported_{label}_request",
        )
    source_id = body.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise ContextInvestigationRouteError(
            "source_id must be one canonical Context Receipt ID", code="invalid_source_id",
        )
    return source_id


def _compact_remove_result(result, plan) -> dict:
    row = result.arm_for(plan.arm_id)
    observation = row.observation
    status = observation.status if observation is not None else row.status
    return {
        "schema_version": "clozn.context-investigation-remove-test-result.v1",
        "run_id": plan.run_id,
        "source_id": plan.source_ids[0],
        "experiment_id": result.experiment_id,
        "arm_id": plan.arm_id,
        "observation_id": row.observation_id,
        "status": status,
        "matched_token_count": getattr(observation, "matched_token_count", None),
        "first_divergence_index": getattr(observation, "first_divergence_index", None),
        "divergence_kind": getattr(observation, "divergence_kind", None),
        "reason": (observation.diagnostics.get("message") if observation is not None else row.error.get("error")),
        "reason_code": (observation.diagnostics.get("reason_code") if observation is not None
                        else row.diagnostics.get("reason")),
        "experiment_state": result.state,
    }


def _compact_counterfactual_result(result, plan) -> dict:
    row = result.arm_for(plan.arm_id)
    observation = row.observation
    status = observation.status if observation is not None else row.status
    return {
        "schema_version": "clozn.context-investigation-counterfactual-result.v1",
        "run_id": plan.run_id,
        "source_id": plan.source_id,
        "experiment_id": result.experiment_id,
        "arm_id": plan.arm_id,
        "observation_id": row.observation_id,
        "status": status,
        "generated_text": getattr(observation, "generated_suffix_text", "") if observation is not None else "",
        "generated_token_count": len(getattr(observation, "generated_token_ids", ()) or ()) if observation is not None else 0,
        "finish_reason": getattr(observation, "finish_reason", None),
        "reason": (observation.diagnostics.get("message") if observation is not None else row.error.get("error")),
        "reason_code": (observation.diagnostics.get("reason_code") if observation is not None
                        else row.diagnostics.get("reason")),
        "experiment_state": result.state,
    }


def _remove_test_worker(run: Mapping[str, object], sub, source_id: str, plan_id: str):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {"code": "context_investigation_run_deleted",
                                                   "message": "recorded run no longer exists"}}
        current_plan = plan_removability(latest, [source_id])
        if current_plan.experiment_id != plan_id:
            return {"state": "failed", "error": {"code": "context_investigation_run_changed",
                                                   "message": "recorded Run or Context Receipt changed while queued"}}
        control.checkpoint(phase="exact_reference_match", completed=0, total=1)
        result = can_remove(
            latest, [source_id], substrate=sub, observation_store=ObservationStore(),
            cancel=control.cancel_requested,
        )
        if control.cancel_requested():
            raise JobCancelled("Context Investigation remove-test job cancelled")
        control.attach_result(_compact_remove_result(result, current_plan))
        control.checkpoint(phase="persisting_observation", completed=1, total=1)
        return {"state": "completed"}
    return worker


def _counterfactual_worker(run: Mapping[str, object], sub, source_id: str, plan_id: str):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {"code": "context_investigation_run_deleted",
                                                   "message": "recorded run no longer exists"}}
        current_plan = plan_context_counterfactual(latest, source_id)
        if current_plan.experiment_id != plan_id:
            return {"state": "failed", "error": {"code": "context_investigation_run_changed",
                                                   "message": "recorded Run or Context Receipt changed while queued"}}
        control.checkpoint(phase="counterfactual_generation", completed=0, total=1)
        result = generate_without_source(
            latest, source_id, substrate=sub, observation_store=ObservationStore(),
            cancel=control.cancel_requested, run_loader=runlog.get_run,
        )
        if control.cancel_requested():
            raise JobCancelled("Context Investigation counterfactual job cancelled")
        control.attach_result(_compact_counterfactual_result(result, current_plan))
        control.checkpoint(phase="persisting_observation", completed=1, total=1)
        return {"state": "completed"}
    return worker


def _normalize_effect_request(body: object) -> dict:
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ContextInvestigationRouteError("request body must be an object", code="invalid_body")
    unknown = set(body).difference({"source_ids"})
    if unknown:
        raise ContextInvestigationRouteError(
            "only source_ids is accepted for the recorded continuation measurement",
            code="unsupported_effect_request",
        )
    source_ids = body.get("source_ids")
    if source_ids is not None:
        if isinstance(source_ids, (str, bytes)) or not isinstance(source_ids, (list, tuple)):
            raise ContextInvestigationRouteError(
                "source_ids must be an array of canonical Context Receipt IDs",
                code="invalid_source_ids",
            )
        if not source_ids or len(source_ids) != len(set(source_ids)):
            raise ContextInvestigationRouteError(
                "source_ids must be a non-empty duplicate-free array",
                code="invalid_source_ids",
            )
    return {"source_ids": list(source_ids) if source_ids is not None else None}


def _start_effects_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        request = _normalize_effect_request(body)
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    plan = _plan_or_error(h, run, source_ids=request["source_ids"])
    if plan is None:
        return True
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/context-investigation/effects/jobs",
    )
    if selection is None:
        return True
    if not callable(getattr(selection.sub, "score_tokens", None)):
        h._json(503, {
            "error": "recorded continuation scoring is unavailable",
            "code": "context_investigation_score_capability_unavailable",
        })
        return True
    try:
        job = JOBS.start(
            run_id,
            _job_worker(run, selection.sub, selection.engine, request["source_ids"], plan.experiment_id),
            kind=_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "context_investigation_job_capacity"})
        return True
    h._json(202, job)
    return True


def _start_remove_test_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        source_id = _normalize_single_source_request(body, label="remove-test")
        plan = plan_removability(run, [source_id])
    except Exception as exc:
        if isinstance(exc, ContextInvestigationRouteError):
            h._json(exc.status, {"error": str(exc), "code": exc.code})
        else:
            h._json(409, {"error": str(exc), "code": "context_investigation_remove_test_unavailable"})
        return True
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/context-investigation/remove-test/jobs",
    )
    if selection is None:
        return True
    if not (callable(getattr(selection.sub, "probe_reference_match", None))
            or callable(getattr(selection.sub, "probe_reference_match_many", None))):
        h._json(503, {"error": "exact reference matching is unavailable",
                       "code": "context_investigation_exact_match_capability_unavailable"})
        return True
    try:
        job = JOBS.start(
            run_id, _remove_test_worker(run, selection.sub, source_id, plan.experiment_id),
            kind=_REMOVE_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "context_investigation_job_capacity"})
        return True
    h._json(202, job)
    return True


def _start_counterfactual_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        source_id = _normalize_single_source_request(body, label="counterfactual")
        plan = plan_context_counterfactual(run, source_id)
    except Exception as exc:
        if isinstance(exc, ContextInvestigationRouteError):
            h._json(exc.status, {"error": str(exc), "code": exc.code})
        elif isinstance(exc, ContextCounterfactualUnavailable):
            h._json(409, {"error": str(exc), "code": f"context_investigation_{exc.reason}"})
        else:
            h._json(409, {"error": str(exc), "code": "context_investigation_counterfactual_unavailable"})
        return True
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/context-investigation/counterfactuals/jobs",
    )
    if selection is None:
        return True
    if not callable(getattr(selection.sub, "chat", None)):
        h._json(503, {"error": "counterfactual generation is unavailable",
                       "code": "context_investigation_generation_capability_unavailable"})
        return True
    try:
        job = JOBS.start(
            run_id, _counterfactual_worker(run, selection.sub, source_id, plan.experiment_id),
            kind=_COUNTERFACTUAL_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "context_investigation_job_capacity"})
        return True
    h._json(202, job)
    return True


def _materialize_counterfactual(h, run_id: str, body) -> bool:
    import clozn.runs.store as runlog

    run = _get_run(h, run_id)
    if run is None:
        return True
    if not isinstance(body, Mapping) or set(body) != {"experiment_id", "arm_id", "observation_id"}:
        h._json(400, {"error": "experiment_id, arm_id, and observation_id are required",
                       "code": "invalid_counterfactual_materialization"})
        return True
    if any(not isinstance(body.get(key), str) or not body[key]
           for key in ("experiment_id", "arm_id", "observation_id")):
        h._json(400, {"error": "experiment_id, arm_id, and observation_id must be non-empty strings",
                       "code": "invalid_counterfactual_materialization"})
        return True
    try:
        result = materialize_generated_observation(
            run, None, body["arm_id"], experiment_id=body["experiment_id"],
            observation_id=body["observation_id"], observation_store=ObservationStore(),
            reload_parent=runlog.get_run,
        )
    except (MaterializationStaleError, MaterializationError, ExperimentPersistenceError) as exc:
        h._json(409, {"error": str(exc), "code": "context_investigation_materialization_unavailable"})
        return True
    h._json(201, result)
    return True


def _query(h, run_id: str, body) -> bool:
    run = _get_run(h, run_id)
    if run is None:
        return True
    if not isinstance(body, Mapping):
        h._json(400, {"error": "body must be an object", "code": "invalid_body"})
        return True
    if set(body).difference({"answer_start", "answer_end", "floor_nats"}):
        h._json(400, {
            "error": "answer_start, answer_end, and optional floor_nats are accepted",
            "code": "unsupported_query",
        })
        return True
    start, end = body.get("answer_start"), body.get("answer_end")
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or start < 0 or end <= start
        or not isinstance(run.get("response"), str) or end > len(run["response"])
    ):
        h._json(400, {
            "error": "answer_start and answer_end must be non-negative half-open Unicode offsets",
            "code": "invalid_answer_range",
        })
        return True
    try:
        floor = body.get("floor_nats", DEFAULT_MEASUREMENT_FLOOR_NATS)
        if isinstance(floor, bool) or not isinstance(floor, (int, float)) or not math.isfinite(float(floor)) or floor < 0:
            raise ValueError
        selection = AnswerSelection.from_range(start, end, selected_text=run.get("response", "")[start:end])
    except (ValueError, TypeError, IndexError):
        h._json(400, {"error": "floor_nats must be finite and answer range must be valid", "code": "invalid_query"})
        return True
    plan = _plan_or_error(h, run)
    if plan is None:
        return True
    store = ObservationStore()
    reader = build_context_investigation_reader(run, plan, observation_store=store, floor_nats=float(floor))
    if reader.get("status") == "not_measured":
        h._json(200, {
            "schema_version": "clozn.context-investigation-query.v1",
            "status": "not_measured",
            "selection": None,
            "effects": [],
            "measurement_floor_nats": float(floor),
        })
        return True
    if reader.get("status") in {"stale", "unavailable"}:
        h._json(200, {
            "schema_version": "clozn.context-investigation-query.v1",
            "status": reader.get("status"),
            "reason": reader.get("reason"),
            "reason_code": reader.get("reason_code"),
            "selection": None,
            "effects": [],
            "measurement_floor_nats": float(floor),
        })
        return True
    try:
        view = store.get_experiment(plan.experiment_id)
        result = query_answer_effects(view, selection, floor_nats=float(floor))
    except AnswerSelectionProjectionUnavailable as exc:
        h._json(422, {
            "error": str(exc),
            "code": "answer_selection_unavailable",
        })
        return True
    except (ObservationNotFound, ExperimentPersistenceError) as exc:
        h._json(409, {"error": str(exc), "code": "context_investigation_evidence_unavailable"})
        return True
    h._json(200, result)
    return True


def try_get(h, path):
    parsed = _run_id(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "/reader":
        return _reader(h, run_id)
    if tail.startswith("/effects/jobs/"):
        job_id = tail[len("/effects/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Context Investigation effects job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    for marker, label in (
        ("/remove-test/jobs/", "Context Investigation remove-test"),
        ("/counterfactuals/jobs/", "Context Investigation counterfactual"),
    ):
        if tail.startswith(marker):
            job_id = tail[len(marker):]
            if not job_id or "/" in job_id:
                return False
            from clozn.server.influence_jobs import JOBS
            job = JOBS.get(run_id, job_id)
            if job is None:
                h._json(404, {"error": f"{label} job not found", "code": "job_not_found"})
            else:
                h._json(200, job)
            return True
    return False


def try_post(h, path, body):
    parsed = _run_id(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "/effects/jobs":
        return _start_effects_job(h, run_id, body)
    if tail.startswith("/effects/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/effects/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.cancel(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Context Investigation effects job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail == "/remove-test/jobs":
        return _start_remove_test_job(h, run_id, body)
    if tail.startswith("/remove-test/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/remove-test/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.cancel(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Context Investigation remove-test job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail == "/counterfactuals/jobs":
        return _start_counterfactual_job(h, run_id, body)
    if tail.startswith("/counterfactuals/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/counterfactuals/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.cancel(run_id, job_id)
        if job is None:
            h._json(404, {"error": "Context Investigation counterfactual job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail == "/counterfactuals/materialize":
        return _materialize_counterfactual(h, run_id, body)
    if tail == "/query":
        return _query(h, run_id, body)
    return False


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get", "try_post"]
