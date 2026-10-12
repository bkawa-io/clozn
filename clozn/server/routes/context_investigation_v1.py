"""Backend Context Investigation v1 routes over the generic experiment kernel.

The reader and answer query are pure reads. Explicit Q1/Q2/Q3 jobs are the
only paths that select a substrate and execute generic Experiments; Q3 ends in
durable GeneratedObservation evidence and only its explicit materialization
route can create a child Run.
"""
from __future__ import annotations

from collections.abc import Mapping
import math
import threading
from urllib.parse import parse_qs, urlsplit

from clozn import schemas
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
from clozn.experiments.evaluators import ScoreRecordedContinuation
from clozn.experiments.observations import (
    TokenScoreObservation,
    execution_observation_identity,
)
from clozn.experiments.materialize import (
    MaterializationError, MaterializationStaleError,
    materialize_generated_observation,
)
from clozn.experiments.selections import AnswerSelection
from clozn.recipes.context_effects import (
    measure_context_effects,
    plan_context_effects,
    plan_context_effects_reader,
)
from clozn.recipes.context_effects_search import (
    ContextEffectsSearchBudget,
    measure_context_effects_coarse_to_fine,
)
from clozn.recipes.context_counterfactual import (
    ContextCounterfactualUnavailable, generate_without_source,
    plan_context_counterfactual,
)
from clozn.recipes.removability import can_remove, plan_removability


CLOZN_ROUTE_AUTOLOAD = True

_PREFIX = "/runs/"
_SUFFIX = "/context-investigation"
_KIND = "context_investigation_effects"
_SEARCH_KIND = "context_investigation_effects_search"
_REMOVE_KIND = "context_investigation_remove_test"
_COUNTERFACTUAL_KIND = "context_investigation_counterfactual"
_READER_SCHEMA = "clozn.context-investigation-reader.v1"
_QUERY_SCHEMA = "clozn.context-investigation-query.v1"
_EFFECTS_RESULT_SCHEMA = "clozn.context-investigation-effects-job-result.v1"
_SEARCH_REQUEST_SCHEMA = "clozn.context-investigation-effects-search-job-request.v1"
_SEARCH_RESULT_SCHEMA = "clozn.context-investigation-effects-search-job-result.v1"
_REMOVE_RESULT_SCHEMA = "clozn.context-investigation-remove-test-result.v1"
_COUNTERFACTUAL_RESULT_SCHEMA = "clozn.context-investigation-counterfactual-result.v1"
_MATERIALIZATION_SCHEMA = "clozn.experiment-materialization.v1"
_MATERIALIZATION_LOCK = threading.Lock()


class ContextInvestigationRouteError(ValueError):
    def __init__(self, message: str, *, code: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _validated(document: dict, schema: str) -> dict:
    """Fail closed before a Context Investigation document crosses the route boundary."""
    schemas.validate(document, schema)
    return document


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
    try:
        plan = plan_context_effects_reader(run)
    except Exception as exc:
        h._json(409, {
            "error": str(exc),
            "code": "context_investigation_plan_unavailable",
        })
        return True
    document = build_context_investigation_reader(
        run, plan, observation_store=ObservationStore(), floor_nats=floor,
    )
    h._json(200, _validated(document, _READER_SCHEMA))
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
        control.attach_result(_validated(_compact_job_result(result), _EFFECTS_RESULT_SCHEMA))
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
        control.attach_result(_validated(
            _compact_remove_result(result, current_plan), _REMOVE_RESULT_SCHEMA,
        ))
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
        control.attach_result(_validated(
            _compact_counterfactual_result(result, current_plan), _COUNTERFACTUAL_RESULT_SCHEMA,
        ))
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


def _normalize_effects_search_request(body: object) -> dict:
    """Validate the additive coarse-to-fine request without widening Q2."""
    if not isinstance(body, Mapping):
        raise ContextInvestigationRouteError(
            "request body must be an object", code="invalid_body",
        )
    request = dict(body)
    try:
        schemas.validate(request, _SEARCH_REQUEST_SCHEMA)
    except schemas.ValidationError as exc:
        raise ContextInvestigationRouteError(
            str(exc), code="invalid_effects_search_request",
        ) from None

    if "coarse_groups" in request and "coarse_group_size" in request:
        raise ContextInvestigationRouteError(
            "coarse_groups and coarse_group_size are mutually exclusive",
            code="invalid_effects_search_request",
        )

    def unique(values, *, name):
        if len(values) != len(set(values)):
            raise ContextInvestigationRouteError(
                f"{name} must not contain duplicates",
                code="invalid_effects_search_request",
            )
        return list(values)

    source_ids = request.get("source_ids")
    if source_ids is not None:
        request["source_ids"] = unique(source_ids, name="source_ids")
    request["drilldown_source_ids"] = unique(
        request.get("drilldown_source_ids", []), name="drilldown_source_ids",
    )

    groups = request.get("coarse_groups")
    if groups is not None:
        normalized_groups = []
        seen: set[str] = set()
        for index, group in enumerate(groups):
            item = unique(group, name=f"coarse_groups[{index}]")
            overlap = sorted(seen.intersection(item))
            if overlap:
                raise ContextInvestigationRouteError(
                    "coarse_groups must not overlap",
                    code="invalid_effects_search_request",
                )
            seen.update(item)
            normalized_groups.append(item)
        request["coarse_groups"] = normalized_groups

    joints = []
    joint_keys: set[tuple[str, ...]] = set()
    for index, group in enumerate(request.get("joint_source_sets", [])):
        item = unique(group, name=f"joint_source_sets[{index}]")
        key = tuple(sorted(item))
        if key in joint_keys:
            raise ContextInvestigationRouteError(
                "joint_source_sets must not contain duplicate conditions",
                code="invalid_effects_search_request",
            )
        joint_keys.add(key)
        joints.append(item)
    request["joint_source_sets"] = joints

    threshold = float(request.get("refine_threshold_nats", 0.25))
    if not math.isfinite(threshold):
        raise ContextInvestigationRouteError(
            "refine_threshold_nats must be finite",
            code="invalid_effects_search_request",
        )
    request["refine_threshold_nats"] = threshold
    request["coarse_group_size"] = int(request.get("coarse_group_size", 4))
    request["budget"] = dict(request["budget"])
    return request


def _resolve_effects_search_spec(request: Mapping[str, object], canonical_plan) -> dict:
    canonical_ids = tuple(canonical_plan.measurement_source_ids)
    canonical_set = set(canonical_ids)
    source_ids = tuple(request.get("source_ids") or canonical_ids)
    unknown = sorted(set(source_ids) - canonical_set)
    if unknown:
        raise ContextInvestigationRouteError(
            "source_ids contains sources outside the canonical measurement universe: "
            + ", ".join(unknown),
            code="invalid_effects_search_request",
        )
    universe = set(source_ids)

    def canonical_set_list(values, *, name, minimum=1):
        unknown_items = sorted(set(values) - universe)
        if unknown_items:
            raise ContextInvestigationRouteError(
                f"{name} contains sources outside the selected measurement universe: "
                + ", ".join(unknown_items),
                code="invalid_effects_search_request",
            )
        if len(values) < minimum:
            raise ContextInvestigationRouteError(
                f"{name} must contain at least {minimum} source IDs",
                code="invalid_effects_search_request",
            )
        return sorted(values)

    raw_groups = request.get("coarse_groups")
    if raw_groups is None:
        size = int(request["coarse_group_size"])
        groups = [
            sorted(source_ids[index:index + size])
            for index in range(0, len(source_ids), size)
        ]
    else:
        groups = [
            canonical_set_list(group, name=f"coarse_groups[{index}]")
            for index, group in enumerate(raw_groups)
        ]
    drilldowns = canonical_set_list(
        request.get("drilldown_source_ids", []),
        name="drilldown_source_ids", minimum=0,
    )
    joints = [
        canonical_set_list(group, name=f"joint_source_sets[{index}]", minimum=2)
        for index, group in enumerate(request.get("joint_source_sets", []))
    ]
    budget = dict(request["budget"])
    if request["execution_policy"] == "native_screen_then_scalar_refine":
        if budget["max_coarse_arms"] > 64 or (
            budget["max_fine_arms"] + budget["max_joint_arms"] > 64
        ):
            raise ContextInvestigationRouteError(
                "native screening supports at most 64 coarse conditions and 64 combined fine/joint conditions",
                code="invalid_effects_search_request",
            )
    return {
        "source_ids": list(source_ids),
        "coarse_groups": groups,
        "coarse_group_size": (
            int(request["coarse_group_size"]) if raw_groups is None else None
        ),
        "refine_threshold_nats": float(request["refine_threshold_nats"]),
        "drilldown_source_ids": drilldowns,
        "joint_source_sets": joints,
        "budget": budget,
        "execution_policy": request["execution_policy"],
    }


def _publish_search_reader_projection(
    store: ObservationStore, run: Mapping[str, object], canonical_plan,
) -> dict:
    """Associate only trusted direct observations with the canonical Q2 plan."""
    experiment = canonical_plan.experiment
    store.create_experiment(
        experiment,
        requested_by={"recipe": "context_effects_search_reader_projection"},
    )
    evaluator = ScoreRecordedContinuation()
    control_identity = execution_observation_identity(
        canonical_plan.execution_state, evaluator, None,
    )
    control = store.find_observation(control_identity["observation_key_sha256"])
    control_id = None
    if (
        isinstance(control, TokenScoreObservation)
        and control.completed and control.trusted
    ):
        control_id = store.associate_observation(
            experiment.experiment_id, "control", control,
        )

    trusted_source_ids: list[str] = []
    for arm in experiment.arms:
        identity = execution_observation_identity(
            canonical_plan.execution_state, evaluator, arm.intervention,
        )
        observation = store.find_observation(identity["observation_key_sha256"])
        if not (
            isinstance(observation, TokenScoreObservation)
            and observation.completed and observation.trusted
        ):
            continue
        store.associate_observation(experiment.experiment_id, arm.arm_id, observation)
        trusted_source_ids.append(arm.intervention.source_ids[0])
    store.set_experiment_state(
        experiment.experiment_id,
        "completed",
        diagnostics={
            "recipe": "context_effects_search_reader_projection",
            "trusted_singleton_source_ids": trusted_source_ids,
        },
    )
    reader = build_context_investigation_reader(
        run, canonical_plan, observation_store=store,
    )
    return {
        "experiment_id": experiment.experiment_id,
        "status": reader.get("status", "not_measured"),
        "control_observation_id": control_id,
        "trusted_singleton_source_ids": trusted_source_ids,
    }


def _effects_search_worker(
    run: Mapping[str, object], sub, request: Mapping[str, object],
    search_spec: Mapping[str, object], canonical_plan_id: str,
):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {
                "code": "context_investigation_effects_search_run_deleted",
                "message": "recorded run no longer exists",
            }}
        try:
            canonical_plan = plan_context_effects(latest)
        except Exception as exc:
            return {"state": "failed", "error": {
                "code": "context_investigation_effects_search_run_changed",
                "message": str(exc),
            }}
        if canonical_plan.experiment_id != canonical_plan_id:
            return {"state": "failed", "error": {
                "code": "context_investigation_effects_search_run_changed",
                "message": "recorded Run or Context Receipt changed while the job was queued",
            }}

        control.checkpoint(phase="planning", completed=0, total=1)
        budget = ContextEffectsSearchBudget(**dict(search_spec["budget"]))
        control.checkpoint(phase="planning", completed=1, total=1)
        use_native = (
            search_spec["execution_policy"] == "native_screen_then_scalar_refine"
        )
        store = ObservationStore()

        def scheduler_progress(**event):
            # Once cancellation is acknowledged, retain the last truthful
            # boundary and let the scheduler assemble actual-work accounting.
            if not control.cancel_requested():
                control.checkpoint(**event)

        result = measure_context_effects_coarse_to_fine(
            latest,
            budget=budget,
            source_ids=search_spec["source_ids"],
            coarse_groups=search_spec["coarse_groups"],
            coarse_group_size=int(request["coarse_group_size"]),
            refine_threshold_nats=float(search_spec["refine_threshold_nats"]),
            drilldown_source_ids=search_spec["drilldown_source_ids"],
            joint_source_sets=search_spec["joint_source_sets"],
            substrate=sub,
            run_loader=runlog.get_run,
            use_native_screening=use_native,
            observation_store=store,
            cancel=control.cancel_requested,
            progress=scheduler_progress,
        )
        if control.cancel_requested():
            cancelled_payload = result.to_dict()
            cancelled_payload.update({
                "schema_version": _SEARCH_RESULT_SCHEMA,
                "search_spec": dict(search_spec),
                "reader_projection": {
                    "experiment_id": canonical_plan.experiment_id,
                    "status": "not_published_cancelled",
                    "control_observation_id": None,
                    "trusted_singleton_source_ids": [],
                },
            })
            control.attach_cancellation_result(_validated(
                cancelled_payload, _SEARCH_RESULT_SCHEMA,
            ))
            raise JobCancelled("Context Investigation effects search cancelled")
        if use_native and any(
            arm.phase in {"fine", "joint"}
            and arm.evidence_status == "screening_only"
            for arm in result.arms
        ):
            return {"state": "failed", "error": {
                "code": "context_investigation_effects_search_scalar_refinement_required",
                "message": "selected fine or joint evidence was not scalar-refined",
            }}

        control.checkpoint(
            phase="publishing_reader_projection", completed=0, total=1,
        )
        projection: dict = {}

        def publish() -> bool:
            projection.update(_publish_search_reader_projection(
                store, latest, canonical_plan,
            ))
            return True

        control.commit(publish)
        payload = result.to_dict()
        payload.update({
            "schema_version": _SEARCH_RESULT_SCHEMA,
            "search_spec": dict(search_spec),
            "reader_projection": projection,
        })
        control.attach_result(_validated(payload, _SEARCH_RESULT_SCHEMA))
        return {"state": "completed"}

    return worker


def _start_effects_search_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        request = _normalize_effects_search_request(body)
        canonical_plan = plan_context_effects(run)
        search_spec = _resolve_effects_search_spec(request, canonical_plan)
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    except Exception as exc:
        h._json(409, {
            "error": str(exc),
            "code": "context_investigation_plan_unavailable",
        })
        return True

    route_name = "/runs/<id>/context-investigation/effects-search/jobs"
    selection = select_control_model_for_run(
        h, run.get("model"), route=route_name,
    )
    if selection is None:
        return True
    if not callable(getattr(selection.sub, "score_tokens", None)):
        h._json(503, {
            "error": "recorded continuation scoring is unavailable",
            "code": "context_investigation_score_capability_unavailable",
        })
        return True
    if (
        search_spec["execution_policy"] == "native_screen_then_scalar_refine"
        and not (
            getattr(selection.sub, "native_score_variants_available", False) is True
            or getattr(selection.sub, "score_tokens_many_execution_strategy", None)
            == "native_prefix_reuse_step_batch"
        )
    ):
        h._json(503, {
            "error": "native related-prompt screening is unavailable on the selected worker",
            "code": "context_investigation_native_screening_unavailable",
        })
        return True
    try:
        job = JOBS.start(
            run_id,
            _effects_search_worker(
                run, selection.sub, request, search_spec,
                canonical_plan.experiment_id,
            ),
            kind=_SEARCH_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {
            "error": str(exc), "code": "context_investigation_job_capacity",
        })
        return True
    h._json(202, job)
    return True


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


def _existing_counterfactual_materialization(
    parent: Mapping[str, object], request: Mapping[str, object],
) -> dict | None:
    """Return the child already created from this exact persisted observation, if any."""
    import clozn.runs.store as runlog
    from clozn.analysis.model_diff import diff_runs

    parent_id = parent.get("id")
    for child in runlog.iter_runs():
        if child.get("parent_run_id") != parent_id:
            continue
        changes = child.get("changes_applied")
        experiment = changes.get("experiment") if isinstance(changes, Mapping) else None
        if not isinstance(experiment, Mapping) or any(
            experiment.get(key) != request.get(key)
            for key in ("experiment_id", "arm_id", "observation_id")
        ):
            continue
        if experiment.get("operation") != "delete_source_generate":
            continue
        base_state = experiment.get("base_state")
        fidelity = (
            base_state.get("realized_fidelity")
            if isinstance(base_state, Mapping) else None
        )
        trace = child.get("trace")
        return {
            "schema_version": _MATERIALIZATION_SCHEMA,
            "state": "completed",
            "parent_run_id": parent_id,
            "child_run_id": child["id"],
            "experiment_id": request["experiment_id"],
            "arm_id": request["arm_id"],
            "observation_id": request["observation_id"],
            "realized_fidelity": fidelity or "direct_context_generation",
            "trace_state": (
                "available" if isinstance(trace, Mapping) and trace else "unavailable"
            ),
            "comparison": diff_runs(parent, child),
            "compare_path": f"#/compare/{parent_id}/{child['id']}",
        }
    return None


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
        # This product action is idempotent even though the generic materializer intentionally
        # remains a branch primitive for Time Travel and other callers.  Serialize the lookup and
        # create inside this process so two repeated UI requests cannot create sibling duplicates.
        with _MATERIALIZATION_LOCK:
            result = _existing_counterfactual_materialization(run, body)
            if result is None:
                result = materialize_generated_observation(
                    run, None, body["arm_id"], experiment_id=body["experiment_id"],
                    observation_id=body["observation_id"], observation_store=ObservationStore(),
                    reload_parent=runlog.get_run,
                )
    except (MaterializationStaleError, MaterializationError, ExperimentPersistenceError) as exc:
        h._json(409, {"error": str(exc), "code": "context_investigation_materialization_unavailable"})
        return True
    h._json(201, _validated(result, _MATERIALIZATION_SCHEMA))
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
        h._json(200, _validated({
            "schema_version": "clozn.context-investigation-query.v1",
            "status": "not_measured",
            "selection": None,
            "effects": [],
            "measurement_floor_nats": float(floor),
        }, _QUERY_SCHEMA))
        return True
    if reader.get("status") in {"stale", "unavailable"}:
        h._json(200, _validated({
            "schema_version": "clozn.context-investigation-query.v1",
            "status": reader.get("status"),
            "reason": reader.get("reason"),
            "reason_code": reader.get("reason_code"),
            "selection": None,
            "effects": [],
            "measurement_floor_nats": float(floor),
        }, _QUERY_SCHEMA))
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
    h._json(200, _validated(result, _QUERY_SCHEMA))
    return True


def try_get(h, path):
    parsed = _run_id(path)
    if parsed is None:
        return False
    run_id, tail = parsed
    if tail == "/reader":
        return _reader(h, run_id)
    if tail.startswith("/effects-search/jobs/"):
        job_id = tail[len("/effects-search/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None or job.get("kind") != _SEARCH_KIND:
            h._json(404, {
                "error": "Context Investigation effects-search job not found",
                "code": "job_not_found",
            })
        else:
            h._json(200, job)
        return True
    if tail.startswith("/effects/jobs/"):
        job_id = tail[len("/effects/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None or job.get("kind") != _KIND:
            h._json(404, {"error": "Context Investigation effects job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    for marker, label, expected_kind in (
        ("/remove-test/jobs/", "Context Investigation remove-test", _REMOVE_KIND),
        ("/counterfactuals/jobs/", "Context Investigation counterfactual", _COUNTERFACTUAL_KIND),
    ):
        if tail.startswith(marker):
            job_id = tail[len(marker):]
            if not job_id or "/" in job_id:
                return False
            from clozn.server.influence_jobs import JOBS
            job = JOBS.get(run_id, job_id)
            if job is None or job.get("kind") != expected_kind:
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
    if tail == "/effects-search/jobs":
        return _start_effects_search_job(h, run_id, body)
    if tail.startswith("/effects-search/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/effects-search/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        current = JOBS.get(run_id, job_id)
        job = (
            JOBS.cancel(run_id, job_id)
            if current is not None and current.get("kind") == _SEARCH_KIND else None
        )
        if job is None:
            h._json(404, {
                "error": "Context Investigation effects-search job not found",
                "code": "job_not_found",
            })
        else:
            h._json(200, job)
        return True
    if tail == "/effects/jobs":
        return _start_effects_job(h, run_id, body)
    if tail.startswith("/effects/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/effects/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        current = JOBS.get(run_id, job_id)
        job = (
            JOBS.cancel(run_id, job_id)
            if current is not None and current.get("kind") == _KIND else None
        )
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
        current = JOBS.get(run_id, job_id)
        job = (
            JOBS.cancel(run_id, job_id)
            if current is not None and current.get("kind") == _REMOVE_KIND else None
        )
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
        current = JOBS.get(run_id, job_id)
        job = (
            JOBS.cancel(run_id, job_id)
            if current is not None and current.get("kind") == _COUNTERFACTUAL_KIND else None
        )
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
