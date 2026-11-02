"""Budgeted tolerant context reduction over recorded-answer token scores."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import time
from typing import Any

from clozn.experiments.context_search import ContextSearchUnavailable
from clozn.experiments.effective_prompt import render_effective_prompt_for_retained
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.score_context_search import ScoreContextSearchDispatcher
from clozn.experiments.search import canonical_search_policy, run_adaptive_search
from clozn.experiments.state import ExecutionState
from clozn.runs.answer_preservation import generation_contract_from_run
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.context_units import max_context_units


SCHEMA_VERSION = "clozn.lean-context-result.v1"
POLICY_VERSION = "balanced_recorded_continuation_loss.v1"
DEFAULT_MEAN_LOSS_TOLERANCE_NATS = 0.05
DEFAULT_MAX_TOKEN_LOSS_TOLERANCE_NATS = 0.70
DEFAULT_MAX_NEW_EXECUTIONS = 16


def _with_elapsed(result: dict[str, Any], started: float) -> dict[str, Any]:
    result["elapsed_seconds"] = max(0.0, time.perf_counter() - started)
    return result


def _digest(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _classification(value: Any) -> str:
    if isinstance(value, Mapping) and value.get("classification") in {
        "preserves", "diverged", "unknown",
    }:
        return str(value["classification"])
    return "unknown"


def _tolerance(mean_loss: float, max_token_loss: float) -> dict[str, Any]:
    return {
        "profile": "balanced.v1",
        "mean_positive_support_loss_nats_per_token": mean_loss,
        "max_token_support_loss_nats": max_token_loss,
    }


def _policy(max_new_executions: int) -> dict[str, Any]:
    return {
        "kind": "tolerant_recorded_continuation_deletion",
        "version": POLICY_VERSION,
        "claim": "best_found_within_budget",
        "max_new_executions": max_new_executions,
        "one_source_reduction_sweep": True,
    }


def _generation_unavailable(reason: str, reason_code: str) -> dict[str, Any]:
    return {
        "status": "unavailable",
        "decode_mode": "greedy",
        "max_new_tokens": None,
        "full_context": None,
        "lean_context": None,
        "exact_match": None,
        "normalized_match": None,
        "reason": reason,
        "reason_code": reason_code,
    }


def _generation_comparison(
    run: Mapping[str, Any],
    *,
    source_ids: Sequence[str],
    retained_source_ids: Sequence[str],
    regenerate: Callable[[Sequence[Mapping[str, Any]], int, Sequence[str]], str] | None,
    generation_progress: Callable[[str, int, int], None] | None,
    cancel: Callable[[], bool] | None,
) -> dict[str, Any]:
    """Run a deterministic full-vs-lean generation pair.

    The historical response is deliberately not used as the full-context arm:
    both arms are freshly decoded greedily with the same prompt renderer and
    token budget, so any text difference is attributable to context rather than
    sampling or a stale worker process.
    """
    if regenerate is None:
        return _generation_unavailable(
            "fresh answer generation was not requested",
            "generation_runner_unavailable",
        )
    contract, contract_reason = generation_contract_from_run(run)
    if contract is None:
        return _generation_unavailable(
            "the recorded generation token budget could not be reconstructed",
            contract_reason or "generation_contract_unavailable",
        )
    max_new = int(contract["max_new"])
    stop = tuple(str(value) for value in contract["stop"])
    try:
        full_messages = render_effective_prompt_for_retained(run, source_ids, source_ids)
        lean_messages = render_effective_prompt_for_retained(
            run, source_ids, retained_source_ids,
        )
        if cancel is not None and cancel():
            return _generation_unavailable(
                "fresh answer generation was cancelled",
                "generation_cancelled",
            )
        if generation_progress is not None:
            generation_progress("full_context", 0, 2)
        full_text = str(regenerate(full_messages, max_new, stop))
        if cancel is not None and cancel():
            return _generation_unavailable(
                "fresh answer generation was cancelled",
                "generation_cancelled",
            )
        if generation_progress is not None:
            generation_progress("lean_context", 1, 2)
        lean_text = str(regenerate(lean_messages, max_new, stop))
    except Exception as exc:
        return _generation_unavailable(
            f"fresh answer generation failed: {exc}",
            "generation_failed",
        )

    recorded = run.get("response") if isinstance(run.get("response"), str) else None
    return {
        "status": "completed",
        "decode_mode": "greedy",
        "max_new_tokens": max_new,
        "full_context": {
            "text": full_text,
            "matches_recorded": full_text == recorded if recorded is not None else None,
        },
        "lean_context": {
            "text": lean_text,
            "matches_recorded": lean_text == recorded if recorded is not None else None,
        },
        "exact_match": full_text == lean_text,
        "normalized_match": full_text.strip() == lean_text.strip(),
    }


def _unavailable(
    *,
    run_id: str,
    state: ExecutionState,
    universe: Mapping[str, Any],
    search_id: str,
    tolerance: Mapping[str, Any],
    policy: Mapping[str, Any],
    max_new_executions: int,
    reason: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "search_id": search_id,
        "status": "unavailable",
        "reason": reason,
        "reason_code": reason_code,
        "base_execution_fingerprint": state.execution_fingerprint,
        "claim": {
            "kind": "best_found_within_budget",
            "minimum_proven": False,
        },
        "tolerance": deepcopy(dict(tolerance)),
        "policy": deepcopy(dict(policy)),
        "universe": deepcopy(dict(universe)),
        "original": None,
        "best": None,
        "generation_comparison": _generation_unavailable(
            "Lean Context search did not produce a candidate",
            "search_unavailable",
        ),
        "reduction": None,
        "trials": [],
        "budget": {
            "max_new_executions": max_new_executions,
            "used_new_executions": 0,
            "reused_observation_count": 0,
            "exhausted": max_new_executions == 0,
            "blocked_by_budget": False,
        },
        "stopping_reason": "search_unavailable",
    }


def run_lean_context(
    run: Mapping[str, Any],
    *,
    max_new_executions: int = DEFAULT_MAX_NEW_EXECUTIONS,
    mean_loss_tolerance_nats: float = DEFAULT_MEAN_LOSS_TOLERANCE_NATS,
    max_token_loss_tolerance_nats: float = DEFAULT_MAX_TOKEN_LOSS_TOLERANCE_NATS,
    max_units: int | None = None,
    observation_store: ObservationStore | None = None,
    substrate: Any = None,
    engine: Any = None,
    execution_adapter: Any = None,
    prompt_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
    render_messages: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]] | None = None,
    regenerate: Callable[[Sequence[Mapping[str, Any]], int, Sequence[str]], str] | None = None,
    generation_progress: Callable[[str, int, int], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Find the cheapest directly tested context within the fixed score tolerance.

    This is intentionally not a minimum-context proof. It reuses compatible
    Context Effects observations, spends at most ``max_new_executions`` on new
    combined-deletion arms, and reports the best candidate encountered by the
    shared bounded reducer.
    """
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ValueError("run_lean_context requires a recorded run")
    if isinstance(max_new_executions, bool) or not isinstance(max_new_executions, int) or max_new_executions < 0:
        raise ValueError("max_new_executions must be a non-negative integer")
    if observation_store is not None and not isinstance(observation_store, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if regenerate is not None and not callable(regenerate):
        raise TypeError("regenerate must be callable when supplied")
    if generation_progress is not None and not callable(generation_progress):
        raise TypeError("generation_progress must be callable when supplied")
    started = time.perf_counter()
    resolved_max_units = max_context_units() if max_units is None else max_units
    state = ExecutionState.from_run(run)
    universe = plan_context_search_universe(
        run, run.get("context_units"), max_units=resolved_max_units,
    )
    tolerance = _tolerance(
        float(mean_loss_tolerance_nats), float(max_token_loss_tolerance_nats),
    )
    policy = _policy(max_new_executions)
    search_id = "lean_" + _digest({
        "base_execution_fingerprint": state.execution_fingerprint,
        "universe_id": universe.get("universe_id"),
        "tolerance": tolerance,
        "policy": policy,
    })[:24]
    if universe.get("status") != "planned":
        condition = universe.get("condition") if isinstance(universe.get("condition"), Mapping) else {}
        return _with_elapsed(_unavailable(
            run_id=run["id"], state=state, universe=universe, search_id=search_id,
            tolerance=tolerance, policy=policy, max_new_executions=max_new_executions,
            reason=str(condition.get("message") or "context search universe is unavailable"),
            reason_code=str(condition.get("code") or "universe_unavailable"),
        ), started)

    dispatcher: ScoreContextSearchDispatcher | None = None
    try:
        dispatcher = ScoreContextSearchDispatcher(
            run,
            universe,
            mean_loss_tolerance_nats=float(mean_loss_tolerance_nats),
            max_token_loss_tolerance_nats=float(max_token_loss_tolerance_nats),
            substrate=substrate,
            engine=engine,
            observation_store=observation_store,
            execution_adapter=execution_adapter,
            prompt_token_counter=prompt_token_counter,
            render_messages=render_messages,
            cancel=cancel,
        )
        searched = run_adaptive_search(
            tuple(universe["source_ids"]),
            max_new_executions,
            dispatcher.prepare_candidate,
            dispatcher.probe_many,
            classify_evidence=_classification,
            candidate_is_reusable=dispatcher.candidate_is_reusable,
            # The shared reducer's bounded child sweep is useful here as a
            # reduction step, especially because its one-source deletions are
            # normally already present from Context Effects. The outer result
            # intentionally does not promote its certificate: Lean Context's
            # public claim remains only "best found within budget".
            attempt_inclusion_check=True,
            search_id=search_id,
            base_execution_fingerprint=state.execution_fingerprint,
            universe_id=universe.get("universe_id"),
            policy=canonical_search_policy(attempt_inclusion_check=True),
            objective={
                "kind": "rendered_prompt_tokens_with_recorded_continuation_loss_tolerance",
                "version": POLICY_VERSION,
                "tolerance": tolerance,
            },
        )
    except (ContextSearchUnavailable, ValueError, TypeError) as exc:
        return _with_elapsed(_unavailable(
            run_id=run["id"], state=state, universe=universe, search_id=search_id,
            tolerance=tolerance, policy=policy, max_new_executions=max_new_executions,
            reason=str(exc), reason_code=getattr(exc, "reason", "search_execution_unavailable"),
        ), started)
    finally:
        if dispatcher is not None:
            try:
                dispatcher.close()
            except Exception:
                pass

    control = searched.control_evidence
    if searched.status == "control_failed" or control is None or not control.observation_id:
        return _with_elapsed(_unavailable(
            run_id=run["id"], state=state, universe=universe, search_id=search_id,
            tolerance=tolerance, policy=policy, max_new_executions=max_new_executions,
            reason=control.observation_status if control is not None else "control unavailable",
            reason_code="score_control_unavailable",
        ), started)

    source_ids = tuple(str(value) for value in universe["source_ids"])

    def evidence_for(observation_id: str | None) -> dict[str, Any] | None:
        return dispatcher.evidence_for_observation(observation_id) if dispatcher is not None else None

    def candidate(retained_ids, cost: int, evidence: Mapping[str, Any] | None) -> dict[str, Any]:
        retained = tuple(str(value) for value in retained_ids)
        kept = set(retained)
        return {
            "retained_source_ids": list(retained),
            "removed_source_ids": [source_id for source_id in source_ids if source_id not in kept],
            "rendered_prompt_token_cost": cost,
            "tolerance_status": evidence.get("tolerance_status") if evidence else None,
            "loss": deepcopy(evidence.get("loss")) if evidence and isinstance(evidence.get("loss"), Mapping) else None,
            "evidence": {
                key: evidence.get(key)
                for key in ("experiment_id", "arm_id", "observation_id", "observation_status", "disposition")
                if evidence.get(key) is not None
            } if evidence else None,
        }

    control_evidence = evidence_for(control.observation_id)
    original = candidate(source_ids, searched.original_candidate.cost, control_evidence)
    best_ref = control if searched.best_candidate.retained_ids == source_ids else None
    if best_ref is None:
        for trial in searched.trials:
            if trial.preserves and trial.retained_ids == searched.best_candidate.retained_ids:
                best_ref = trial.evidence_ref
                break
    best_evidence = evidence_for(best_ref.observation_id if best_ref else None)
    best = candidate(searched.best_candidate.retained_ids, searched.best_candidate.cost, best_evidence)

    trials: list[dict[str, Any]] = []
    for trial in searched.trials:
        direct = evidence_for(trial.observation_id)
        retained = tuple(str(value) for value in trial.retained_ids)
        retained_set = set(retained)
        trials.append({
            "ordinal": trial.ordinal,
            "stage": trial.stage,
            "retained_source_ids": list(retained),
            "removed_source_ids": [source_id for source_id in source_ids if source_id not in retained_set],
            "rendered_prompt_token_cost": trial.cost,
            "classification": trial.classification,
            "disposition": trial.disposition,
            "tolerance_status": direct.get("tolerance_status") if direct else None,
            "loss": deepcopy(direct.get("loss")) if direct and isinstance(direct.get("loss"), Mapping) else None,
            "evidence": trial.evidence_ref.to_dict() if trial.evidence_ref else None,
        })

    original_cost = original["rendered_prompt_token_cost"]
    best_cost = best["rendered_prompt_token_cost"]
    removed_tokens = max(0, original_cost - best_cost)
    generation_comparison = _generation_comparison(
        run,
        source_ids=source_ids,
        retained_source_ids=best["retained_source_ids"],
        regenerate=regenerate,
        generation_progress=generation_progress,
        cancel=cancel,
    )
    return _with_elapsed({
        "schema_version": SCHEMA_VERSION,
        "run_id": run["id"],
        "search_id": search_id,
        "status": "completed",
        "base_execution_fingerprint": state.execution_fingerprint,
        "claim": {
            "kind": "best_found_within_budget",
            "minimum_proven": False,
        },
        "tolerance": tolerance,
        "policy": policy,
        "universe": deepcopy(dict(universe)),
        "original": original,
        "best": best,
        "generation_comparison": generation_comparison,
        "reduction": {
            "removed_prompt_tokens": removed_tokens,
            "retained_prompt_token_fraction": (best_cost / original_cost) if original_cost else 1.0,
            "removed_source_count": len(best["removed_source_ids"]),
            "retained_source_count": len(best["retained_source_ids"]),
        },
        "trials": trials,
        "budget": searched.budget.to_dict(),
        "stopping_reason": (
            "budget_exhausted" if searched.budget.blocked_by_budget else "search_policy_complete"
        ),
    }, started)


__all__ = [
    "DEFAULT_MAX_NEW_EXECUTIONS",
    "DEFAULT_MAX_TOKEN_LOSS_TOLERANCE_NATS",
    "DEFAULT_MEAN_LOSS_TOLERANCE_NATS",
    "POLICY_VERSION",
    "SCHEMA_VERSION",
    "run_lean_context",
]
