"""Canonical Context Effects reader, query, and direct-measurement routes."""
from __future__ import annotations

from collections.abc import Mapping
import math
import statistics
import time
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
from clozn.experiments.selections import AnswerSelection
from clozn.recipes.context_effects import (
    configure_context_effect_sources,
    measure_context_effects,
    plan_context_effects,
    plan_context_effects_reader,
)
from clozn.recipes.lean_context import run_lean_context
from clozn.recipes.random_coalitions import run_random_coalitions
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.context_units import max_context_arms, max_context_units


CLOZN_ROUTE_AUTOLOAD = True

_PREFIX = "/runs/"
_SUFFIX = "/context-investigation"
_KIND = "context_investigation_effects"
_LEAN_KIND = "context_investigation_lean_context"
_INTERACTIONS_KIND = "context_investigation_interactions"
_READER_SCHEMA = "clozn.context-investigation-reader.v1"
_QUERY_SCHEMA = "clozn.context-investigation-query.v1"
_EFFECTS_RESULT_SCHEMA = "clozn.context-investigation-effects-job-result.v1"
_LEAN_RESULT_SCHEMA = "clozn.lean-context-result.v1"
_INTERACTIONS_RESULT_SCHEMA = "clozn.context-interactions-result.v1"
_FAST_MAX_SEQUENCE_BATCH = 16
_FAST_FALLBACK_SEQUENCE_BATCH = 4
_FAST_LOGICAL_ROWS_PER_BATCH = 4096


class ContextInvestigationRouteError(ValueError):
    def __init__(self, message: str, *, code: str, status: int = 400):
        super().__init__(message)
        self.code = code
        self.status = status


def _validated(document: dict, schema: str) -> dict:
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
        h._json(404, {
            "error": "run not found",
            "code": "context_investigation_run_not_found",
        })
        return None
    return run


def _query_floor(h) -> float:
    query = parse_qs(urlsplit(getattr(h, "path", "")).query, keep_blank_values=True)
    raw = (query.get("floor_nats") or [None])[-1]
    if raw in {None, ""}:
        return DEFAULT_MEASUREMENT_FLOOR_NATS
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ContextInvestigationRouteError(
            "floor_nats must be a finite non-negative number", code="invalid_floor",
        ) from None
    if not math.isfinite(value) or value < 0:
        raise ContextInvestigationRouteError(
            "floor_nats must be a finite non-negative number", code="invalid_floor",
        )
    return value


def _normalize_effect_configuration(value: Mapping[str, object]) -> dict:
    granularity = value.get("granularity")
    max_arms = value.get("max_arms")
    if granularity is None and max_arms is None:
        return {"granularity": None, "max_arms": None}
    if granularity is None:
        granularity = "adaptive"
    if max_arms is None:
        max_arms = max_context_arms()
    if granularity not in {"message", "adaptive"}:
        raise ContextInvestigationRouteError(
            "granularity must be 'message' or 'adaptive'",
            code="invalid_granularity",
        )
    if isinstance(max_arms, bool) or not isinstance(max_arms, int):
        raise ContextInvestigationRouteError(
            "max_arms must be a positive integer",
            code="invalid_max_arms",
        )
    return {"granularity": granularity, "max_arms": max_arms}


def _configured_source_ids(run: Mapping[str, object], configuration: Mapping[str, object]):
    granularity = configuration.get("granularity")
    max_arms = configuration.get("max_arms")
    if granularity is None and max_arms is None:
        return None
    try:
        return list(configure_context_effect_sources(
            run, granularity=str(granularity), max_arms=int(max_arms),
        ))
    except (TypeError, ValueError) as exc:
        raise ContextInvestigationRouteError(
            str(exc), code="invalid_effect_configuration",
        ) from exc


def _query_effect_request(h) -> dict:
    query = parse_qs(urlsplit(getattr(h, "path", "")).query, keep_blank_values=True)
    raw_granularity = (query.get("granularity") or [None])[-1]
    raw_max_arms = (query.get("max_arms") or [None])[-1]
    scoring_mode = (query.get("scoring_mode") or [None])[-1]
    source_ids = query.get("source_id")
    try:
        max_arms = int(raw_max_arms) if raw_max_arms is not None else None
    except (TypeError, ValueError):
        max_arms = raw_max_arms
    return _normalize_effect_request({
        **({"source_ids": source_ids} if source_ids is not None else {}),
        "granularity": raw_granularity,
        "max_arms": max_arms,
        **({"scoring_mode": scoring_mode} if scoring_mode is not None else {}),
    })


def _plan_or_error(h, run: Mapping[str, object], source_ids=None, *, scoring_mode="exact"):
    try:
        return plan_context_effects(
            run, source_ids=source_ids, scoring_mode=scoring_mode,
        )
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
        request = _query_effect_request(h)
        source_ids = request["source_ids"] or _configured_source_ids(run, request)
        plan = plan_context_effects_reader(
            run, source_ids=source_ids, scoring_mode=request["scoring_mode"],
        )
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
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


def _fast_sequence_batch_size(run: Mapping[str, object]) -> int:
    """Conservatively pack native sequences from recorded worker token facts.

    The native worker performs better accounting with its exact prompt-token
    trie. This gateway estimate deliberately assumes no sharing, so every batch
    accepted here also fits when prefixes happen not to overlap at all. The
    logical-row ceiling bounds one uninterrupted background worker turn, which
    lets foreground chat acquire the model between large experiment batches.
    Eight prompt rows cover small BPE-boundary changes caused by deleting source
    text.
    """
    meta = run.get("meta")
    if not isinstance(meta, Mapping):
        return _FAST_FALLBACK_SEQUENCE_BATCH
    n_ctx = meta.get("n_ctx")
    prompt_tokens = meta.get("prompt_tokens")
    if (
        isinstance(n_ctx, bool) or not isinstance(n_ctx, int) or n_ctx <= 0
        or isinstance(prompt_tokens, bool) or not isinstance(prompt_tokens, int)
        or prompt_tokens <= 0
    ):
        return _FAST_FALLBACK_SEQUENCE_BATCH
    trace = run.get("trace")
    token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    continuation_tokens = len(token_ids) if isinstance(token_ids, list) else run.get("token_count")
    if (
        isinstance(continuation_tokens, bool)
        or not isinstance(continuation_tokens, int)
        or continuation_tokens <= 0
    ):
        return _FAST_FALLBACK_SEQUENCE_BATCH
    conservative_rows_per_sequence = prompt_tokens + 8 + max(0, continuation_tokens - 1)
    available_rows = min(n_ctx, _FAST_LOGICAL_ROWS_PER_BATCH)
    return max(
        1,
        min(_FAST_MAX_SEQUENCE_BATCH, available_rows // conservative_rows_per_sequence),
    )


def _fast_execution_plan(run: Mapping[str, object], *, sequence_count: int) -> dict:
    """Choose the measured-fast scoring path from immutable run facts.

    Native trie scoring wins when prompt prefill dominates and the answer is
    short. Scalar prefix-cached scoring wins when its single large continuation
    pass avoids many native one-token decode calls. The conservative v1 rule is
    intentionally explicit and reported with every job rather than pretending
    one backend is universally faster.
    """
    meta = run.get("meta")
    prompt_tokens = meta.get("prompt_tokens") if isinstance(meta, Mapping) else None
    trace = run.get("trace")
    token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    continuation_tokens = len(token_ids) if isinstance(token_ids, list) else run.get("token_count")
    facts_available = bool(
        isinstance(prompt_tokens, int) and not isinstance(prompt_tokens, bool)
        and prompt_tokens > 0
        and isinstance(continuation_tokens, int)
        and not isinstance(continuation_tokens, bool)
        and continuation_tokens > 0
    )
    native_selected = bool(
        sequence_count >= 2
        and (
            not facts_available
            or (
                continuation_tokens <= 32
                and prompt_tokens >= 4 * continuation_tokens
            )
        )
    )
    return {
        "policy": "clozn.fast-path-auto.v1",
        "selected_strategy": (
            "native_prefix_trie" if native_selected else "scalar_prefix_cache"
        ),
        "reason": (
            "prompt_dominant_short_continuation"
            if native_selected and facts_available
            else "recorded_token_facts_unavailable_native_fallback"
            if native_selected
            else "continuation_decode_dominates"
        ),
        "prompt_tokens": prompt_tokens if facts_available else None,
        "continuation_tokens": continuation_tokens if facts_available else None,
        "sequence_count": sequence_count,
        "native_sequence_batch_size": _fast_sequence_batch_size(run),
    }


def _compact_job_result(result, *, elapsed_seconds: float, scoring_mode: str) -> dict:
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
    dispositions = (result.control_disposition, *result.arm_dispositions)
    result_diagnostics = getattr(result, "diagnostics", {})
    if not isinstance(result_diagnostics, Mapping):
        result_diagnostics = {}
    arm_diagnostics = [row.diagnostics for row in result.arms if isinstance(row.diagnostics, Mapping)]
    strategies = {
        str(row["execution_strategy"])
        for row in arm_diagnostics if row.get("execution_strategy")
    }
    regimes = {
        str(row["numerical_regime"])
        for row in arm_diagnostics if row.get("numerical_regime")
    }
    execution_strategy = (
        next(iter(strategies)) if len(strategies) == 1
        else "mixed" if strategies
        else str(result_diagnostics.get("execution_strategy") or "scalar_serial")
    )
    numerical_regime = (
        next(iter(regimes)) if len(regimes) == 1
        else "mixed" if regimes
        else str(
            result_diagnostics.get("numerical_regime")
            or ("native_prefix_trie_target_logprob_v1" if scoring_mode == "fast"
                else "scalar_reference")
        )
    )
    observations = [
        observation for observation in (
            getattr(result, "control", None),
            *(row.observation for row in result.arms),
        )
        if observation is not None and getattr(observation, "completed", False)
    ]
    evidence_quality = (
        "approximate"
        if scoring_mode == "fast" and (
            not observations
            or any(not getattr(observation, "trusted", False) for observation in observations)
        )
        else "exact"
    )
    payload = {
        "schema_version": _EFFECTS_RESULT_SCHEMA,
        "experiment_id": result.experiment_id,
        "source_count": len(result.arms),
        "completed_source_count": completed,
        "unavailable_source_count": unavailable,
        "failed_source_count": failed,
        "not_measured_source_count": not_measured,
        "executed_score_count": sum(value == "executed" for value in dispositions),
        "reused_score_count": sum(value == "reused" for value in dispositions),
        "elapsed_seconds": max(0.0, float(elapsed_seconds)),
        "scoring_mode": scoring_mode,
        "evidence_quality": evidence_quality,
        "execution_strategy": execution_strategy,
        "numerical_regime": numerical_regime,
        "state": result.state,
    }
    timing = result_diagnostics.get("performance_timing")
    if isinstance(timing, Mapping):
        payload["timing_detail"] = dict(timing)
    if any(key in result_diagnostics for key in (
        "requested_sequence_batch_size", "batch_count", "sequence_batch_sizes",
        "control_batched",
    )):
        payload["batching"] = {
            "requested_sequence_batch_size": result_diagnostics.get(
                "requested_sequence_batch_size"
            ),
            "batch_count": result_diagnostics.get("batch_count", 0),
            "arm_batch_sizes": list(result_diagnostics.get("batch_sizes") or []),
            "sequence_batch_sizes": list(
                result_diagnostics.get("sequence_batch_sizes") or []
            ),
            "control_batched": result_diagnostics.get("control_batched") is True,
        }
    execution_plan = result_diagnostics.get("fast_execution_plan")
    if isinstance(execution_plan, Mapping):
        payload["execution_plan"] = dict(execution_plan)
    return payload


def _job_worker(run: Mapping[str, object], sub, source_ids, plan_id: str,
                scoring_mode: str = "exact"):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {
                "code": "context_investigation_run_deleted",
                "message": "recorded run no longer exists",
            }}
        current_plan = plan_context_effects(
            latest, source_ids=source_ids, scoring_mode=scoring_mode,
        )
        if current_plan.experiment_id != plan_id:
            return {"state": "failed", "error": {
                "code": "context_investigation_run_changed",
                "message": "recorded Run or Context Receipt changed while the job was queued",
            }}
        source_count = len(current_plan.measurement_source_ids)
        control.checkpoint(
            phase="scoring_recorded_continuation", completed=0, total=source_count,
        )

        def publish_progress(event):
            if control.cancel_requested():
                return
            try:
                control.checkpoint(
                    phase=str(event.get("phase", "scoring_recorded_continuation")),
                    completed=int(event.get("completed_units", 0)),
                    total=int(event.get("total_units", source_count)),
                )
            except JobCancelled:
                # The runner observes the same cancellation flag before it
                # dispatches another arm. Completed observations are already
                # durable at this point and remain available to the reader.
                return

        started = time.perf_counter()
        fast_plan = _fast_execution_plan(
            latest, sequence_count=source_count + 1,
        ) if scoring_mode == "fast" else None
        use_native = bool(
            fast_plan is not None
            and fast_plan["selected_strategy"] == "native_prefix_trie"
        )
        execution_sub = sub
        if scoring_mode == "fast" and not use_native:
            configure_mode = getattr(sub, "with_scoring_mode", None)
            if callable(configure_mode):
                execution_sub = configure_mode("exact")
        sequence_batch_size = (
            int(fast_plan["native_sequence_batch_size"]) if use_native else 1
        )
        result = measure_context_effects(
            latest,
            source_ids=source_ids,
            substrate=execution_sub,
            observation_store=ObservationStore(),
            cancel=control.cancel_requested,
            arm_batch_size=sequence_batch_size,
            # Fast-auto always sends the control through the same detailed
            # score-many contract as its arms. In the scalar selection this is
            # a one-sequence batch, preserving exact execution provenance.
            batch_control=scoring_mode == "fast",
            scoring_mode=scoring_mode,
            on_progress=publish_progress,
            execution_diagnostics=(
                {"fast_execution_plan": fast_plan} if fast_plan is not None else None
            ),
        )
        if control.cancel_requested():
            raise JobCancelled("Context Effects measurement cancelled")
        control.checkpoint(
            phase="persisting_observations", completed=source_count, total=source_count,
        )
        control.attach_result(_validated(
            _compact_job_result(
                result,
                elapsed_seconds=time.perf_counter() - started,
                scoring_mode=scoring_mode,
            ),
            _EFFECTS_RESULT_SCHEMA,
        ))
        return {"state": "completed"}

    return worker


def _normalize_effect_request(body: object) -> dict:
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ContextInvestigationRouteError("request body must be an object", code="invalid_body")
    unknown = set(body).difference({"source_ids", "granularity", "max_arms", "scoring_mode"})
    if unknown:
        raise ContextInvestigationRouteError(
            "only source_ids, granularity, max_arms, and scoring_mode are accepted for the recorded continuation measurement",
            code="unsupported_effect_request",
        )
    source_ids = body.get("source_ids")
    if source_ids is not None:
        if isinstance(source_ids, (str, bytes)) or not isinstance(source_ids, (list, tuple)):
            raise ContextInvestigationRouteError(
                "source_ids must be an array of canonical Context Receipt IDs",
                code="invalid_source_ids",
            )
        if (
            not source_ids
            or any(not isinstance(source_id, str) or not source_id for source_id in source_ids)
            or len(source_ids) != len(set(source_ids))
        ):
            raise ContextInvestigationRouteError(
                "source_ids must be a non-empty duplicate-free array",
                code="invalid_source_ids",
            )
    configuration = _normalize_effect_configuration(body)
    scoring_mode = body.get("scoring_mode", "exact")
    if scoring_mode not in {"fast", "exact"}:
        raise ContextInvestigationRouteError(
            "scoring_mode must be 'fast' or 'exact'",
            code="invalid_scoring_mode",
        )
    if source_ids is not None and configuration["granularity"] is not None:
        raise ContextInvestigationRouteError(
            "source_ids cannot be combined with granularity or max_arms",
            code="ambiguous_effect_request",
        )
    return {
        "source_ids": list(source_ids) if source_ids is not None else None,
        "scoring_mode": scoring_mode,
        **configuration,
    }


def _start_effects_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        request = _normalize_effect_request(body)
        source_ids = request["source_ids"] or _configured_source_ids(run, request)
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    plan = _plan_or_error(
        h, run, source_ids=source_ids, scoring_mode=request["scoring_mode"],
    )
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
    sub = selection.sub
    configure_mode = getattr(sub, "with_scoring_mode", None)
    if callable(configure_mode):
        sub = configure_mode(request["scoring_mode"])
    try:
        job = JOBS.start(
            run_id,
            _job_worker(
                run, sub, source_ids, plan.experiment_id,
                scoring_mode=request["scoring_mode"],
            ),
            kind=_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "context_investigation_job_capacity"})
        return True
    h._json(202, job)
    return True


def _normalize_lean_request(body: object) -> dict:
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ContextInvestigationRouteError("request body must be an object", code="invalid_body")
    unsupported = set(body) - {"max_new_executions"}
    if unsupported:
        raise ContextInvestigationRouteError(
            "Lean Context accepts only max_new_executions; the answer-change tolerance remains fixed",
            code="unsupported_lean_context_request",
        )
    max_new_executions = body.get("max_new_executions", 16)
    if (
        isinstance(max_new_executions, bool)
        or not isinstance(max_new_executions, int)
        or not 1 <= max_new_executions <= 64
    ):
        raise ContextInvestigationRouteError(
            "max_new_executions must be an integer from 1 to 64",
            code="invalid_lean_context_budget",
        )
    return {"max_new_executions": max_new_executions}


def _lean_job_worker(
    run: Mapping[str, object], sub, engine, universe_id: str,
    request: Mapping[str, object] | None = None,
):
    max_new_executions = int((request or {}).get("max_new_executions", 16))

    def worker(control):
        import clozn.runs.store as runlog
        from clozn.runs.think_tags import prompt_opens_think, sanitize_reply
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {
                "code": "lean_context_run_deleted",
                "message": "recorded run no longer exists",
            }}
        current_universe = plan_context_search_universe(
            latest, latest.get("context_units"), max_units=max_context_units(),
        )
        if current_universe.get("universe_id") != universe_id:
            return {"state": "failed", "error": {
                "code": "lean_context_run_changed",
                "message": "recorded Run or context-unit universe changed while the job was queued",
            }}

        def regenerate(messages, max_new, stop):
            manifest = {}
            raw = sub.chat(
                [dict(message) for message in messages],
                int(max_new),
                False,
                mem_out=manifest,
                stop=list(stop),
            )
            return sanitize_reply(
                raw,
                implicit_open=prompt_opens_think(manifest.get("final_prompt")),
            ).public_text

        def generation_progress(label, completed, total):
            control.checkpoint(
                phase=f"regenerating_{label}_answer",
                completed=completed,
                total=total,
            )

        control.checkpoint(
            phase="searching_lean_context", completed=0, total=max_new_executions,
        )
        result = run_lean_context(
            latest,
            max_new_executions=max_new_executions,
            substrate=sub,
            engine=engine,
            observation_store=ObservationStore(),
            regenerate=regenerate,
            generation_progress=generation_progress,
            cancel=control.cancel_requested,
        )
        if control.cancel_requested():
            raise JobCancelled("Lean Context search cancelled")
        control.checkpoint(
            phase="validating_lean_context",
            completed=result.get("budget", {}).get("used_new_executions", 0),
            total=result.get("budget", {}).get("max_new_executions", 16),
        )
        control.attach_result(_validated(result, _LEAN_RESULT_SCHEMA))
        return {"state": "completed"}

    return worker


def _start_lean_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        request = _normalize_lean_request(body)
        universe = plan_context_search_universe(
            run, run.get("context_units"), max_units=max_context_units(),
        )
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    except Exception as exc:
        h._json(409, {"error": str(exc), "code": "lean_context_universe_unavailable"})
        return True
    if universe.get("status") != "planned":
        condition = universe.get("condition") if isinstance(universe.get("condition"), Mapping) else {}
        h._json(409, {
            "error": condition.get("message", "Lean Context universe is unavailable"),
            "code": condition.get("code", "lean_context_universe_unavailable"),
            "universe": universe,
        })
        return True
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/context-investigation/lean-context/jobs",
    )
    if selection is None:
        return True
    if not callable(getattr(selection.sub, "score_tokens", None)):
        h._json(503, {
            "error": "recorded continuation scoring is unavailable",
            "code": "lean_context_score_capability_unavailable",
        })
        return True
    if not callable(getattr(selection.sub, "chat", None)):
        h._json(503, {
            "error": "fresh answer generation is unavailable",
            "code": "lean_context_generation_capability_unavailable",
        })
        return True
    try:
        job = JOBS.start(
            run_id,
            _lean_job_worker(
                run, selection.sub, selection.engine, str(universe["universe_id"]), request,
            ),
            kind=_LEAN_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "context_investigation_job_capacity"})
        return True
    h._json(202, job)
    return True


def _normalize_interactions_request(body: object) -> dict:
    if body is None:
        body = {}
    if not isinstance(body, Mapping):
        raise ContextInvestigationRouteError("request body must be an object", code="invalid_body")
    unknown = set(body).difference({"max_new_executions", "max_pairs", "seed"})
    if unknown:
        raise ContextInvestigationRouteError(
            "only max_new_executions, max_pairs, and seed are accepted",
            code="unsupported_interactions_request",
        )
    values = {
        "max_new_executions": body.get("max_new_executions", 16),
        "max_pairs": body.get("max_pairs", 32),
        "seed": body.get("seed", 0),
    }
    for key, minimum, maximum in (
        ("max_new_executions", 0, 64),
        ("max_pairs", 1, 256),
        ("seed", 0, 2**31 - 1),
    ):
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
            raise ContextInvestigationRouteError(
                f"{key} must be an integer between {minimum} and {maximum}",
                code="invalid_interactions_request",
            )
    return values


def _sampled_estimates(source_ids, samples):
    estimates = []
    for source_id in source_ids:
        draws = [sample for sample in samples if sample.get("source_id") == source_id]
        values = [float(sample["delta_nats"]) for sample in draws]
        estimates.append({
            "source_id": source_id,
            "sample_count": len(draws),
            "unique_pair_count": len({
                tuple(sample.get("without_source_ids", [])) for sample in draws
            }),
            "mean_marginal_nats": statistics.fmean(values) if values else None,
            "sample_stddev_nats": statistics.stdev(values) if len(values) > 1 else None,
            "leave_one_out_nats": None,
            "mean_minus_leave_one_out_nats": None,
        })
    return estimates


def _compact_interactions_result(result: Mapping[str, object]) -> dict:
    universe = result.get("universe") if isinstance(result.get("universe"), Mapping) else {}
    budget = result.get("budget") if isinstance(result.get("budget"), Mapping) else {}
    estimates = result.get("source_estimates")
    estimates = estimates if isinstance(estimates, list) else []
    best = result.get("best") if isinstance(result.get("best"), Mapping) else None
    return {
        "schema_version": _INTERACTIONS_RESULT_SCHEMA,
        "run_id": result.get("run_id"),
        "status": result.get("status", "partial"),
        "stop_reason": result.get("stop_reason"),
        "seed": result.get("seed", 0),
        "claim": {
            "kind": "exploratory_sampled_marginals",
            "redundancy_proven": False,
        },
        "source_ids": list(universe.get("source_ids", [])),
        "budget": {
            "max_new_executions": budget.get("max_new_executions", 0),
            "used_new_executions": budget.get("used_new_executions", 0),
            "reused_observation_count": budget.get("reused_observation_count", 0),
            "completed_pairs": budget.get("completed_pairs", 0),
        },
        "source_estimates": [{
            key: row.get(key)
            for key in (
                "source_id", "sample_count", "unique_pair_count", "mean_marginal_nats",
                "sample_stddev_nats", "leave_one_out_nats", "mean_minus_leave_one_out_nats",
            )
        } for row in estimates if isinstance(row, Mapping)],
        "elapsed_seconds": result.get("elapsed_seconds"),
        "best": ({
            "retained_source_ids": list(best.get("retained_source_ids", [])),
            "rendered_prompt_tokens": best.get("rendered_prompt_tokens"),
            "loss": ({
                key: best["loss"].get(key)
                for key in (
                    "answer_token_count", "positive_loss_token_count",
                    "mean_positive_support_loss_nats_per_token",
                    "max_token_support_loss_nats",
                )
            } if isinstance(best.get("loss"), Mapping) else None),
        } if best is not None else None),
    }


def _interactions_job_worker(
    run: Mapping[str, object], sub, engine, universe: Mapping[str, object], request: Mapping[str, int],
):
    def worker(control):
        import clozn.runs.store as runlog
        from clozn.server.influence_jobs import JobCancelled

        latest = runlog.get_run(run.get("id"))
        if latest is None:
            return {"state": "failed", "error": {
                "code": "context_interactions_run_deleted",
                "message": "recorded run no longer exists",
            }}
        current_universe = plan_context_search_universe(
            latest, latest.get("context_units"), max_units=max_context_units(),
        )
        if current_universe.get("universe_id") != universe.get("universe_id"):
            return {"state": "failed", "error": {
                "code": "context_interactions_run_changed",
                "message": "recorded Run or context-unit universe changed while the job was queued",
            }}
        source_ids = list(current_universe.get("source_ids", []))
        samples = []
        max_new = int(request["max_new_executions"])
        control.checkpoint(phase="sampling_context_interactions", completed=0, total=max_new)

        def publish_score(event):
            if control.cancel_requested():
                return
            try:
                control.checkpoint(
                    phase="sampling_context_interactions",
                    completed=int(event.get("used_new_executions", 0)),
                    total=max_new,
                )
            except JobCancelled:
                return

        def publish_sample(event):
            sample = event.get("sample")
            if isinstance(sample, Mapping):
                samples.append(dict(sample))
            preview = {
                "schema_version": _INTERACTIONS_RESULT_SCHEMA,
                "run_id": latest["id"],
                "status": "running",
                "stop_reason": None,
                "seed": request["seed"],
                "claim": {"kind": "exploratory_sampled_marginals", "redundancy_proven": False},
                "source_ids": source_ids,
                "budget": {
                    "max_new_executions": max_new,
                    "used_new_executions": int(event.get("used_new_executions", 0)),
                    "reused_observation_count": int(event.get("reused_observation_count", 0)),
                    "completed_pairs": int(event.get("completed_pairs", len(samples))),
                },
                "source_estimates": _sampled_estimates(source_ids, samples),
                "elapsed_seconds": None,
                "best": None,
            }
            if control.cancel_requested():
                return
            try:
                control.attach_result(_validated(preview, _INTERACTIONS_RESULT_SCHEMA))
                control.checkpoint(
                    phase="sampling_context_interactions",
                    completed=int(event.get("used_new_executions", 0)),
                    total=max_new,
                )
            except JobCancelled:
                return

        result = run_random_coalitions(
            latest,
            substrate=sub,
            engine=engine,
            observation_store=ObservationStore(),
            max_new_executions=max_new,
            max_pairs=int(request["max_pairs"]),
            seed=int(request["seed"]),
            max_units=max_context_units(),
            cancel=control.cancel_requested,
            progress=publish_score,
            sample_progress=publish_sample,
        )
        compact = _validated(
            _compact_interactions_result(result), _INTERACTIONS_RESULT_SCHEMA,
        )
        if control.cancel_requested():
            control.attach_cancellation_result(compact)
            raise JobCancelled("Context interaction sampling cancelled")
        control.attach_result(compact)
        return {"state": "completed"}

    return worker


def _start_interactions_job(h, run_id: str, body) -> bool:
    from clozn.server.influence_jobs import JOBS, JobCapacityError
    from clozn.server.model_routing import select_control_model_for_run

    run = _get_run(h, run_id)
    if run is None:
        return True
    try:
        request = _normalize_interactions_request(body)
        universe = plan_context_search_universe(
            run, run.get("context_units"), max_units=max_context_units(),
        )
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    except Exception as exc:
        h._json(409, {"error": str(exc), "code": "context_interactions_universe_unavailable"})
        return True
    if universe.get("status") != "planned":
        condition = universe.get("condition") if isinstance(universe.get("condition"), Mapping) else {}
        h._json(409, {
            "error": condition.get("message", "Context interaction universe is unavailable"),
            "code": condition.get("code", "context_interactions_universe_unavailable"),
        })
        return True
    selection = select_control_model_for_run(
        h, run.get("model"), route="/runs/<id>/context-investigation/interactions/jobs",
    )
    if selection is None:
        return True
    if not callable(getattr(selection.sub, "score_tokens", None)):
        h._json(503, {
            "error": "recorded continuation scoring is unavailable",
            "code": "context_interactions_score_capability_unavailable",
        })
        return True
    try:
        job = JOBS.start(
            run_id,
            _interactions_job_worker(run, selection.sub, selection.engine, universe, request),
            kind=_INTERACTIONS_KIND,
        )
    except JobCapacityError as exc:
        h._json(429, {"error": str(exc), "code": "context_investigation_job_capacity"})
        return True
    h._json(202, job)
    return True


def _query(h, run_id: str, body) -> bool:
    run = _get_run(h, run_id)
    if run is None:
        return True
    if not isinstance(body, Mapping):
        h._json(400, {"error": "body must be an object", "code": "invalid_body"})
        return True
    if set(body).difference({
        "answer_start", "answer_end", "floor_nats", "source_ids", "granularity", "max_arms",
        "scoring_mode",
    }):
        h._json(400, {
            "error": "answer_start, answer_end, optional floor_nats, source_ids, granularity, max_arms, and scoring_mode are accepted",
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
        if (
            isinstance(floor, bool) or not isinstance(floor, (int, float))
            or not math.isfinite(float(floor)) or floor < 0
        ):
            raise ValueError
        selection = AnswerSelection.from_range(
            start, end, selected_text=run.get("response", "")[start:end],
        )
    except (ValueError, TypeError, IndexError):
        h._json(400, {
            "error": "floor_nats must be finite and answer range must be valid",
            "code": "invalid_query",
        })
        return True
    try:
        request = _normalize_effect_request({
            key: body[key]
            for key in ("source_ids", "granularity", "max_arms", "scoring_mode")
            if key in body
        })
        source_ids = request["source_ids"] or _configured_source_ids(run, request)
    except ContextInvestigationRouteError as exc:
        h._json(exc.status, {"error": str(exc), "code": exc.code})
        return True
    plan = _plan_or_error(
        h, run, source_ids=source_ids, scoring_mode=request["scoring_mode"],
    )
    if plan is None:
        return True
    store = ObservationStore()
    reader = build_context_investigation_reader(
        run, plan, observation_store=store, floor_nats=float(floor),
    )
    if reader.get("status") == "not_measured":
        h._json(200, _validated({
            "schema_version": _QUERY_SCHEMA,
            "status": "not_measured",
            "selection": None,
            "effects": [],
            "measurement_floor_nats": float(floor),
        }, _QUERY_SCHEMA))
        return True
    if reader.get("status") in {"stale", "unavailable"}:
        h._json(200, _validated({
            "schema_version": _QUERY_SCHEMA,
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
        h._json(422, {"error": str(exc), "code": "answer_selection_unavailable"})
        return True
    except (ObservationNotFound, ExperimentPersistenceError) as exc:
        h._json(409, {
            "error": str(exc),
            "code": "context_investigation_evidence_unavailable",
        })
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
    if tail.startswith("/effects/jobs/"):
        job_id = tail[len("/effects/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None or job.get("kind") != _KIND:
            h._json(404, {"error": "Context Effects job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail.startswith("/lean-context/jobs/"):
        job_id = tail[len("/lean-context/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None or job.get("kind") != _LEAN_KIND:
            h._json(404, {"error": "Lean Context job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail.startswith("/interactions/jobs/"):
        job_id = tail[len("/interactions/jobs/"):]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        job = JOBS.get(run_id, job_id)
        if job is None or job.get("kind") != _INTERACTIONS_KIND:
            h._json(404, {"error": "Interactions job not found", "code": "job_not_found"})
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
    if tail == "/lean-context/jobs":
        return _start_lean_job(h, run_id, body)
    if tail == "/interactions/jobs":
        return _start_interactions_job(h, run_id, body)
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
            h._json(404, {"error": "Context Effects job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail.startswith("/lean-context/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/lean-context/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        current = JOBS.get(run_id, job_id)
        job = (
            JOBS.cancel(run_id, job_id)
            if current is not None and current.get("kind") == _LEAN_KIND else None
        )
        if job is None:
            h._json(404, {"error": "Lean Context job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail.startswith("/interactions/jobs/") and tail.endswith("/cancel"):
        job_id = tail[len("/interactions/jobs/"):-len("/cancel")]
        if not job_id or "/" in job_id:
            return False
        from clozn.server.influence_jobs import JOBS
        current = JOBS.get(run_id, job_id)
        job = (
            JOBS.cancel(run_id, job_id)
            if current is not None and current.get("kind") == _INTERACTIONS_KIND else None
        )
        if job is None:
            h._json(404, {"error": "Interactions job not found", "code": "job_not_found"})
        else:
            h._json(200, job)
        return True
    if tail == "/query":
        return _query(h, run_id, body)
    return False


__all__ = ["CLOZN_ROUTE_AUTOLOAD", "try_get", "try_post"]
