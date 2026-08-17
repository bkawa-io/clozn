"""Adaptive exact Minimal Context recipe over the generic experiment kernel."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from typing import Any

from clozn.experiments.context_search import ContextSearchDispatcher, ContextSearchUnavailable
from clozn.experiments.evaluators import ExactReferenceMatch
from clozn.experiments.search import (
    BEST_VERIFIED,
    INCLUSION_MINIMUM,
    canonical_search_policy,
    SearchBudget,
    SearchTrial,
    SearchTrajectoryEntry,
    run_adaptive_search,
)
from clozn.experiments.state import ExecutionState
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.experiments.persistence import ObservationStore


SCHEMA_VERSION = "clozn.minimal-context-search-result.v1"
OBJECTIVE_VERSION = "rendered_prompt_tokens.v1"


class MinimalContextUnavailable(ValueError):
    """The exact direct-observation search cannot be performed faithfully."""

    def __init__(self, message: str, *, reason: str = "minimal_context_unavailable"):
        super().__init__(message)
        self.reason = reason


def _digest(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WinningCandidate:
    retained_source_ids: tuple[str, ...]
    removed_source_ids: tuple[str, ...]
    rendered_prompt_token_cost: int
    experiment_id: str | None
    arm_id: str | None
    observation_id: str | None
    observation_status: str | None = None

    @property
    def cost(self) -> int:
        return self.rendered_prompt_token_cost

    @property
    def retained_ids(self) -> tuple[str, ...]:
        return self.retained_source_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "retained_source_ids": list(self.retained_source_ids),
            "removed_source_ids": list(self.removed_source_ids),
            "rendered_prompt_token_cost": self.rendered_prompt_token_cost,
            "experiment_id": self.experiment_id,
            "arm_id": self.arm_id,
            "observation_id": self.observation_id,
            "observation_status": self.observation_status,
        }


@dataclass(frozen=True)
class MinimalContextResult:
    """Reference-only result; ObservationStore remains evidence authority."""

    search_id: str
    status: str
    base_execution_fingerprint: str
    universe: Mapping[str, Any]
    objective: Mapping[str, Any]
    control_observation_id: str | None
    trials: tuple[SearchTrial, ...]
    trajectory: tuple[SearchTrajectoryEntry, ...]
    best: WinningCandidate | None
    certificate: str | None
    policy: Mapping[str, Any]
    budget: SearchBudget
    inclusion_check: Mapping[str, Any]
    reason: str | None = None
    reason_code: str | None = None
    search_status: str | None = None

    @property
    def best_candidate(self) -> WinningCandidate | None:
        return self.best

    @property
    def certificate_level(self) -> str | None:
        return self.certificate

    @property
    def winning_experiment_id(self) -> str | None:
        return self.best.experiment_id if self.best else None

    @property
    def winning_arm_id(self) -> str | None:
        return self.best.arm_id if self.best else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "search_id": self.search_id,
            "status": self.status,
            "search_status": self.search_status,
            "reason": self.reason,
            "reason_code": self.reason_code,
            "base_execution_fingerprint": self.base_execution_fingerprint,
            "universe": deepcopy(dict(self.universe)),
            "objective": deepcopy(dict(self.objective)),
            "control_observation_id": self.control_observation_id,
            "trials": [trial.to_dict() for trial in self.trials],
            "trajectory": [item.to_dict() for item in self.trajectory],
            "best": self.best.to_dict() if self.best else None,
            "certificate": self.certificate,
            "policy": deepcopy(dict(self.policy)),
            "budget": self.budget.to_dict(),
            "inclusion_check": dict(self.inclusion_check),
        }

    def to_json(self) -> str:
        from clozn.experiments.state import canonical_json
        return canonical_json(self.to_dict())


def _empty_budget(max_new: int) -> SearchBudget:
    return SearchBudget(max_new_executions=max_new, used_new_executions=0, exhausted=max_new == 0)


def _identity(*, state: ExecutionState, universe: Mapping[str, Any], max_new: int,
              policy: Mapping[str, Any]) -> str:
    value = {
        "base_execution_fingerprint": state.execution_fingerprint,
        "universe_id": universe.get("universe_id"),
        "ordered_source_ids": list(universe.get("source_ids") or []),
        "policy": deepcopy(dict(policy)),
        "objective_version": OBJECTIVE_VERSION,
        "max_new_counterfactual_observations": max_new,
    }
    return "search_" + _digest(value)[:24]


def _unavailable(*, state: ExecutionState, universe: Mapping[str, Any], max_new: int,
                 search_id: str, reason: str, policy: Mapping[str, Any],
                 reason_code: str = "minimal_context_unavailable",
                 status: str = "unavailable") -> MinimalContextResult:
    objective = {"kind": "rendered_prompt_tokens", "version": OBJECTIVE_VERSION}
    return MinimalContextResult(
        search_id=search_id,
        status=status,
        search_status="unavailable",
        base_execution_fingerprint=state.execution_fingerprint,
        universe=universe,
        objective=objective,
        control_observation_id=None,
        trials=(),
        trajectory=(),
        best=None,
        certificate=None,
        policy=policy,
        budget=_empty_budget(max_new),
        inclusion_check={"attempted": False, "complete": False, "tested_child_count": 0,
                         "total_child_count": 0, "all_children_failed": False},
        reason=reason,
        reason_code=reason_code,
    )


def run_minimal_context(
    run: Mapping[str, Any], *,
    evaluator: ExactReferenceMatch | None = None,
    max_new_counterfactual_observations: int = 32,
    observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None,
    substrate: Any = None,
    engine: Any = None,
    execution_adapter: Any = None,
    prompt_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
    render_messages: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]] | None = None,
    max_units: int = 50,
    attempt_inclusion_check: bool = True,
    cancel: Callable[[], bool] | None = None,
    execution_strategy: str = "auto",
) -> MinimalContextResult:
    """Search directly observed exact preservation over a canonical universe."""
    if observation_store is not None and store is not None and observation_store is not store:
        raise ValueError("pass only one observation store")
    durable = observation_store or store
    if durable is not None and not isinstance(durable, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise MinimalContextUnavailable("a recorded run with a non-empty id is required", reason="run_unavailable")
    if evaluator is not None and not isinstance(evaluator, ExactReferenceMatch):
        raise MinimalContextUnavailable(
            "Batch 5 Minimal Context supports exact recorded-answer preservation only",
            reason="unsupported_evaluator",
        )
    if (isinstance(max_new_counterfactual_observations, bool)
            or not isinstance(max_new_counterfactual_observations, int)
            or max_new_counterfactual_observations < 0):
        raise ValueError("max_new_counterfactual_observations must be a non-negative integer")
    try:
        policy = canonical_search_policy(attempt_inclusion_check=attempt_inclusion_check)
    except ValueError as exc:
        raise MinimalContextUnavailable(str(exc), reason="invalid_search_policy") from exc
    try:
        state = ExecutionState.from_run(run)
    except Exception as exc:
        raise MinimalContextUnavailable(
            f"recorded execution state unavailable: {exc}",
            reason="execution_state_unavailable",
        ) from exc
    manifest = run.get("context_units")
    try:
        universe = plan_context_search_universe(run, manifest, max_units=max_units)
    except Exception as exc:  # normalize planner failures at this recipe boundary
        raise MinimalContextUnavailable(f"context search universe unavailable: {exc}", reason="universe_unavailable") from exc
    search_id = _identity(
        state=state, universe=universe, max_new=max_new_counterfactual_observations,
        policy=policy,
    )
    if universe.get("status") != "planned":
        return _unavailable(
            state=state, universe=universe, max_new=max_new_counterfactual_observations,
            search_id=search_id, policy=policy,
            reason=(universe.get("condition") or {}).get("message", "context search universe unavailable"),
            reason_code=(universe.get("condition") or {}).get("code", "universe_unavailable"),
        )
    objective = {"kind": "rendered_prompt_tokens", "version": OBJECTIVE_VERSION}
    dispatcher: ContextSearchDispatcher | None = None
    try:
        dispatcher = ContextSearchDispatcher(
            run, universe, substrate=substrate, engine=engine,
            observation_store=durable, execution_adapter=execution_adapter,
            evaluator=ExactReferenceMatch(), prompt_token_counter=prompt_token_counter,
            render_messages=render_messages, cancel=cancel, execution_strategy=execution_strategy,
        )
        searched = run_adaptive_search(
            tuple(universe["source_ids"]),
            max_new_counterfactual_observations,
            dispatcher.prepare_candidate,
            dispatcher.probe_many,
            candidate_is_reusable=dispatcher.candidate_is_reusable,
            attempt_inclusion_check=attempt_inclusion_check,
            search_id=search_id,
            base_execution_fingerprint=state.execution_fingerprint,
            universe_id=universe.get("universe_id"),
            policy=policy,
            objective=objective,
        )
    except (ContextSearchUnavailable, ValueError, TypeError) as exc:
        return _unavailable(
            state=state, universe=universe, max_new=max_new_counterfactual_observations,
            search_id=search_id, policy=policy, reason=str(exc),
            reason_code=getattr(exc, "reason", "search_execution_unavailable"),
        )
    finally:
        if dispatcher is not None:
            try:
                dispatcher.close()
            except Exception:
                # Evidence already established remains valid; lifecycle cleanup
                # is operational metadata and cannot rewrite the result.
                pass

    control_ref = searched.control_evidence
    control_id = control_ref.observation_id if control_ref else None
    if searched.status == "control_failed":
        return MinimalContextResult(
            search_id=search_id, status="unavailable", search_status=searched.status,
            base_execution_fingerprint=state.execution_fingerprint, universe=universe,
            objective=objective, control_observation_id=control_id,
            trials=searched.trials, trajectory=searched.trajectory, best=None,
            certificate=None, policy=policy, budget=searched.budget,
            inclusion_check=searched.inclusion_check.to_dict(),
            reason=(control_ref.observation_status if control_ref else "control_unavailable"),
            reason_code="exact_control_unavailable",
        )

    best_trial = None
    for trial in searched.trials:
        if trial.retained_ids == searched.best_candidate.retained_ids and trial.preserves:
            best_trial = trial
            break
    if best_trial is None:
        best_ref = control_ref if searched.best_candidate.retained_ids == tuple(universe["source_ids"]) else None
    else:
        best_ref = best_trial.evidence_ref
    retained = tuple(str(item) for item in searched.best_candidate.retained_ids)
    removed = tuple(source_id for source_id in universe["source_ids"] if source_id not in set(retained))
    best = WinningCandidate(
        retained_source_ids=retained,
        removed_source_ids=removed,
        rendered_prompt_token_cost=searched.best_candidate.cost,
        experiment_id=best_ref.experiment_id if best_ref else None,
        arm_id=best_ref.arm_id if best_ref else None,
        observation_id=best_ref.observation_id if best_ref else None,
        observation_status=best_ref.observation_status if best_ref else None,
    )
    return MinimalContextResult(
        search_id=search_id, status="completed", search_status=searched.status,
        base_execution_fingerprint=state.execution_fingerprint, universe=universe,
        objective=objective, control_observation_id=control_id,
        trials=searched.trials, trajectory=searched.trajectory, best=best,
        certificate=searched.certificate, policy=policy, budget=searched.budget,
        inclusion_check=searched.inclusion_check.to_dict(),
    )


__all__ = [
    "BEST_VERIFIED", "INCLUSION_MINIMUM", "MinimalContextError", "MinimalContextUnavailable",
    "MinimalContextResult", "WinningCandidate", "run_minimal_context",
]


MinimalContextError = MinimalContextUnavailable
