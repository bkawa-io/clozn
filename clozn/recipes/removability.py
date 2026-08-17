"""The first thin recipe built on the experimental kernel."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from clozn.experiments.evaluators import ExactReferenceMatch
from clozn.experiments.execution import DeleteSourceExactReferenceAdapter, resolve_delete_source
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState


@dataclass(frozen=True)
class RemovabilityPlan:
    """Model-free Q1 plan for one typed DeleteSource condition."""

    run_id: str
    execution_state: ExecutionState
    source_ids: tuple[str, ...]
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
            "source_ids": list(self.source_ids),
            "experiment": self.experiment.to_dict(),
        }


def plan_removability(run: Mapping[str, Any], source_ids) -> RemovabilityPlan:
    """Build the canonical Q1 Experiment without selecting a worker or executing."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run["id"]:
        raise ValueError("plan_removability requires a recorded run with a non-empty id")
    state = ExecutionState.from_run(run)
    selection = ContextSelection(source_ids)
    # Validate the canonical receipt binding before any product route selects
    # a worker.  The execution adapter repeats this against its fresh reload.
    resolve_delete_source(run, DeleteSource(selection))
    experiment = Experiment(
        base=state,
        evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(selection)],
    )
    return RemovabilityPlan(
        run_id=run["id"],
        execution_state=state,
        source_ids=tuple(selection.source_ids),
        experiment=experiment,
    )


def can_remove(
    run: Mapping[str, Any],
    source_ids,
    execution_adapter: Any | None = None,
    *,
    substrate: Any | None = None,
    run_loader=None,
    include_control: bool = True,
    observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None,
    cancel=None,
) -> ExperimentResult:
    """Measure whether deleting the requested canonical sources preserves the answer."""
    plan = plan_removability(run, source_ids)
    if execution_adapter is None:
        if substrate is None:
            raise ValueError("can_remove requires execution_adapter or substrate")
        execution_adapter = DeleteSourceExactReferenceAdapter(
            substrate, run=run, run_loader=run_loader,
        )
    return run_experiment(
        plan.experiment, execution_adapter, include_control=include_control,
        observation_store=observation_store, store=store,
        cancel=cancel,
        requested_by={"recipe": "removability"},
    )


def removability_message(result: ExperimentResult, arm_id: str | None = None) -> str:
    """Render direct preservation evidence without semantic overclaiming."""
    observation = result.control if arm_id is None else result.observation_for(arm_id)
    if observation is None:
        return "Exact execution unavailable."
    if observation.status == "exact_preserved":
        return "This source was deleted and the recorded answer was exactly preserved."
    if observation.status == "diverged":
        index = observation.first_divergence_index
        return f"Deleting this source caused divergence at recorded answer token {index}."
    if observation.status == "unavailable":
        return "Exact execution unavailable; the deletion was not established faithfully."
    return "Exact preservation experiment failed."


__all__ = ["RemovabilityPlan", "can_remove", "plan_removability", "removability_message"]
