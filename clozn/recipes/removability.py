"""The first thin recipe built on the experimental kernel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clozn.experiments.evaluators import ExactReferenceMatch
from clozn.experiments.execution import DeleteSourceExactReferenceAdapter
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState


def can_remove(
    run: Mapping[str, Any],
    source_ids,
    execution_adapter: Any | None = None,
    *,
    substrate: Any | None = None,
    run_loader=None,
    include_control: bool = True,
) -> ExperimentResult:
    """Measure whether deleting the requested canonical sources preserves the answer."""
    state = ExecutionState.from_run(run)
    selection = ContextSelection(source_ids)
    experiment = Experiment(
        base=state,
        evaluator=ExactReferenceMatch(),
        arms=[DeleteSource(selection)],
    )
    if execution_adapter is None:
        if substrate is None:
            raise ValueError("can_remove requires execution_adapter or substrate")
        execution_adapter = DeleteSourceExactReferenceAdapter(
            substrate, run=run, run_loader=run_loader,
        )
    return run_experiment(experiment, execution_adapter, include_control=include_control)


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


__all__ = ["can_remove", "removability_message"]
