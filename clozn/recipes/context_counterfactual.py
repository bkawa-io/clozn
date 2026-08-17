"""Context-boundary counterfactual generation recipe.

This recipe is deliberately a small planner/orchestrator.  Generation ends in
the experimental kernel's GeneratedObservation; only the explicit generic
materialization API may create a child Run.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from clozn.experiments.evaluators import Generate
from clozn.experiments.execution import resolve_delete_source
from clozn.experiments.generation import DeleteSourceGenerateAdapter
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState


class ContextCounterfactualError(ValueError):
    """The recorded generation contract cannot support this recipe."""


class ContextCounterfactualUnavailable(ContextCounterfactualError):
    """A typed, model-free refusal before worker selection or dispatch."""

    def __init__(self, message: str, *, reason: str = "context_counterfactual_unavailable"):
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class ContextCounterfactualPlan:
    """Immutable Q3 plan for deleting one canonical source and generating."""

    run_id: str
    execution_state: ExecutionState
    source_id: str
    experiment: Experiment

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    @property
    def arm_id(self) -> str:
        return self.experiment.arms[0].arm_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_state": self.execution_state.to_dict(),
            "source_id": self.source_id,
            "experiment": self.experiment.to_dict(),
        }


def _generation_contract(state: ExecutionState) -> Mapping[str, Any]:
    contract = state.generation_contract
    if not isinstance(contract, Mapping) or state.generation_contract_reason:
        raise ContextCounterfactualUnavailable(
            state.generation_contract_reason
            or "the recorded Run does not contain a complete generation contract",
            reason="generation_contract_unavailable",
        )
    required = {"max_new", "decode_mode", "sampling", "stop"}
    if not required.issubset(contract):
        raise ContextCounterfactualUnavailable(
            "the recorded Run generation contract is incomplete",
            reason="generation_contract_unavailable",
        )
    return contract


def plan_context_counterfactual(
    run: Mapping[str, Any], source_id: str,
) -> ContextCounterfactualPlan:
    """Build Q3's typed plan without selecting a worker or calling a model."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ContextCounterfactualError(
            "plan_context_counterfactual requires a recorded Run with a non-empty id"
        )
    if not isinstance(source_id, str) or not source_id:
        raise ContextCounterfactualError("source_id must be a canonical Context Receipt ID")
    state = ExecutionState.from_run(run)
    contract = _generation_contract(state)
    try:
        selection = ContextSelection([source_id])
        intervention = DeleteSource(selection)
        # Source existence, protected-message rules, and receipt binding are
        # all model-free.  Refuse them before a route selects or touches a
        # worker; the adapter repeats the check after its stale reload.
        resolve_delete_source(run, intervention)
    except Exception as exc:
        raise ContextCounterfactualUnavailable(
            f"the requested source cannot be used for counterfactual generation: {exc}",
            reason=getattr(exc, "reason", None) or "source_unavailable",
        ) from exc
    try:
        evaluator = Generate(
            max_new=contract["max_new"],
            decode_mode=contract["decode_mode"],
            sampling=contract["sampling"],
            stop=contract["stop"],
        )
        experiment = Experiment(
            base=state,
            evaluator=evaluator,
            arms=[intervention],
        )
    except Exception as exc:
        raise ContextCounterfactualUnavailable(
            f"the recorded generation contract cannot be used: {exc}",
            reason="generation_contract_unavailable",
        ) from exc
    return ContextCounterfactualPlan(
        run_id=run["id"], execution_state=state,
        source_id=source_id, experiment=experiment,
    )


def generate_without_source(
    run: Mapping[str, Any], source_id: str, *, substrate: Any,
    observation_store: ObservationStore | None = None,
    cancel=None, run_loader=None,
) -> ExperimentResult:
    """Execute the planned arm into GeneratedObservation evidence only."""
    plan = plan_context_counterfactual(run, source_id)
    adapter = DeleteSourceGenerateAdapter(
        substrate, run=run, run_loader=run_loader,
    )
    return run_experiment(
        plan.experiment, adapter, include_control=False, cancel=cancel,
        observation_store=observation_store,
        requested_by={"recipe": "context_counterfactual"},
    )


__all__ = [
    "ContextCounterfactualError", "ContextCounterfactualPlan",
    "ContextCounterfactualUnavailable", "generate_without_source",
    "plan_context_counterfactual",
]
