"""Thin StateRef + ForceToken + Generate recipe."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from clozn.experiments.evaluators import Generate
from clozn.experiments.generation import GenerateExecutionAdapter
from clozn.experiments.interventions import ForceToken
from clozn.experiments.kernel import Experiment
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.state_ref import StateRef, StateRefError, resolve_state


def run_time_travel(
    run: Mapping[str, Any], *, position: int, token_id: int | None = None,
    token_piece: str | None = None, max_new: int, policy: str = "exact_preferred",
    checkpoint: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
    worker_identity: Mapping[str, Any] | None = None,
    execution_adapter: Any | None = None, substrate: Any | None = None,
    run_loader=None, observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None, include_control: bool = True,
) -> ExperimentResult:
    """Run one durable time-travel Generate experiment.

    Resolution and execution remain delegated to the kernel; this recipe only
    wires the user-facing recorded boundary and forced token together.
    """
    if not isinstance(run, Mapping):
        raise TypeError("run_time_travel requires a recorded run mapping")
    state_ref = StateRef.before_answer_token(run, position)
    resolved = resolve_state(
        state_ref, run=run, policy=policy, checkpoint=checkpoint,
        runtime_identity=runtime_identity, worker_identity=worker_identity,
    )
    if not resolved.available:
        reason = resolved.diagnostics.get("reason_code", "state_unavailable")
        raise StateRefError(f"time-travel state is unavailable: {reason}")
    experiment = Experiment(
        base=resolved, evaluator=Generate(max_new=max_new),
        arms=[ForceToken(token_id=token_id, token_piece=token_piece)],
    )
    adapter = execution_adapter
    if adapter is None:
        if substrate is None:
            raise ValueError("run_time_travel requires execution_adapter or substrate")
        adapter = GenerateExecutionAdapter(
            substrate, run=run, run_loader=run_loader,
            runtime_identity=runtime_identity, worker_identity=worker_identity,
        )
    return run_experiment(
        experiment, adapter, include_control=include_control,
        observation_store=observation_store, store=store,
        requested_by={"recipe": "time_travel"},
    )


time_travel = run_time_travel


__all__ = ["run_time_travel", "time_travel"]
