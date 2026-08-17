"""Durable orchestration for the feature-independent experimental kernel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

from .evaluators import Evaluator, ExactReferenceMatch, ScoreRecordedContinuation, Generate, evaluator_from_dict
from .interventions import DeleteSource, ForceToken, Intervention, intervention_from_dict
from .kernel import Experiment
from .observations import (
    Observation, ObservationError, ObservationIntegrityError, TokenScoreObservation,
    GeneratedObservation, TokenScoreDelta, execution_observation_identity, observation_from_dict,
)
from .persistence import ExperimentArmView, ExperimentView, ObservationStore
from .state import ExecutionState, canonical_json
from .state_ref import ResolvedState


SCHEMA_VERSION = "clozn.experiment-view.v2"
ObservationLike = Union[Observation, TokenScoreObservation, GeneratedObservation]


def _identity_for(state: ExecutionState | ResolvedState, evaluator: Evaluator,
                 intervention: Intervention | None) -> dict[str, Any]:
    return execution_observation_identity(state, evaluator, intervention)


def _expected_type(evaluator: Evaluator):
    if isinstance(evaluator, ScoreRecordedContinuation):
        return TokenScoreObservation
    if isinstance(evaluator, Generate):
        return GeneratedObservation
    return Observation


def _allows_arms(evaluator: Evaluator, control: ObservationLike | None) -> bool:
    if control is None:
        return False
    if isinstance(evaluator, ScoreRecordedContinuation):
        return isinstance(control, TokenScoreObservation) and control.status == "completed"
    if isinstance(evaluator, Generate):
        return isinstance(control, GeneratedObservation) and control.status == "completed"
    return type(control) is Observation and control.status == "exact_preserved"


def _reusable(observation: ObservationLike | None) -> bool:
    """Only completed direct measurements may occupy the canonical store key."""
    return observation is not None and observation.completed


class ExperimentResult:
    """Convenient read model; durable stores remain the authority when present."""

    __slots__ = (
        "experiment_id", "base", "evaluator", "control", "arm_rows", "arm_observations",
        "arm_interventions", "state", "diagnostics", "timing", "execution_provenance",
        "requested_by", "persisted", "observation_store",
    )

    def __init__(self, *, experiment_id: str, base: ExecutionState | ResolvedState, evaluator: Evaluator,
                 control: ObservationLike | None, arm_observations: list[ObservationLike | None] | tuple[ObservationLike | None, ...],
                 arm_interventions: list[Intervention] | tuple[Intervention, ...],
                 arm_ids: list[str] | tuple[str, ...] | None = None,
                 arm_states: list[str] | tuple[str, ...] | None = None,
                 observation_ids: list[str | None] | tuple[str | None, ...] | None = None,
                 arm_errors: list[Mapping[str, Any] | None] | tuple[Mapping[str, Any] | None, ...] | None = None,
                 arm_diagnostics: list[Mapping[str, Any] | None] | tuple[Mapping[str, Any] | None, ...] | None = None,
                 state: str, diagnostics: Mapping[str, Any] | None = None,
                 timing: Mapping[str, Any] | None = None,
                 execution_provenance: Mapping[str, Any] | None = None,
                 arm_rows: tuple[ExperimentArmView, ...] | None = None,
                 requested_by: Mapping[str, Any] | None = None,
                 persisted: bool = False, observation_store: ObservationStore | None = None):
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("ExperimentResult.experiment_id must be non-empty")
        if not isinstance(base, (ExecutionState, ResolvedState)):
            raise TypeError("ExperimentResult.base must be an ExecutionState or ResolvedState")
        if not isinstance(evaluator, (ExactReferenceMatch, ScoreRecordedContinuation, Generate)):
            raise TypeError("ExperimentResult.evaluator must be a supported evaluator")
        if state not in {"pending", "running", "completed", "cancelled", "failed", "blocked"}:
            raise ValueError("unsupported ExperimentResult state")
        interventions = tuple(arm_interventions)
        observations = tuple(arm_observations)
        if len(interventions) != len(observations):
            raise ValueError("ExperimentResult arm interventions must align with observations")
        if any(not isinstance(item, (DeleteSource, ForceToken)) for item in interventions):
            raise TypeError("ExperimentResult arm interventions must be typed interventions")
        ids = tuple(arm_ids or [
            "arm_" + str(index) for index in range(len(interventions))
        ])
        if len(ids) != len(interventions):
            raise ValueError("ExperimentResult arm IDs must align with interventions")
        states = tuple(arm_states or ["completed" if item is not None else "pending" for item in observations])
        obs_ids = tuple(observation_ids or [item.observation_id if item is not None else None for item in observations])
        errors = tuple(arm_errors or [{} for _ in interventions])
        row_diagnostics = tuple(arm_diagnostics or [{} for _ in interventions])
        if not (len(states) == len(obs_ids) == len(errors) == len(row_diagnostics) == len(interventions)):
            raise ValueError("ExperimentResult arm metadata must align with interventions")
        if arm_rows is None:
            rows = tuple(
                ExperimentArmView(
                    experiment_id=experiment_id, arm_id=ids[index], ordinal=index,
                    intervention=interventions[index], condition=observations[index].condition if observations[index] else
                    _identity_for(base, evaluator, interventions[index])["observation_key"]["condition"],
                    state=states[index], observation_id=obs_ids[index], observation=observations[index],
                    error=errors[index], diagnostics=row_diagnostics[index],
                )
                for index in range(len(interventions))
            )
        else:
            rows = tuple(arm_rows)
        self.experiment_id = experiment_id
        self.base = base
        self.evaluator = evaluator
        self.control = control
        self.arm_rows = rows
        self.arm_observations = observations
        self.arm_interventions = interventions
        self.state = state
        self.diagnostics = dict(diagnostics or {})
        self.timing = dict(timing or {})
        self.execution_provenance = dict(execution_provenance or {})
        self.requested_by = dict(requested_by or {})
        self.persisted = bool(persisted)
        self.observation_store = observation_store

    @classmethod
    def from_view(cls, view: ExperimentView, *, observation_store: ObservationStore | None = None) -> "ExperimentResult":
        return cls(
            experiment_id=view.experiment_id, base=view.base, evaluator=view.evaluator, control=view.control,
            arm_observations=tuple(row.observation for row in view.arms),
            arm_interventions=tuple(row.intervention for row in view.arms),
            arm_ids=tuple(row.arm_id for row in view.arms),
            arm_states=tuple(row.state for row in view.arms),
            observation_ids=tuple(row.observation_id for row in view.arms),
            arm_errors=tuple(row.error for row in view.arms), state=view.state,
            diagnostics=view.diagnostics, timing=view.timing,
            execution_provenance=view.execution_provenance, arm_rows=view.arm_rows,
            requested_by=view.requested_by,
            persisted=True, observation_store=observation_store,
        )

    @property
    def arms(self) -> tuple[ExperimentArmView, ...]:
        return self.arm_rows

    def arm_for(self, arm_id: str) -> ExperimentArmView:
        for row in self.arm_rows:
            if row.arm_id == arm_id:
                return row
        raise KeyError(arm_id)

    def observation_for(self, arm_id: str) -> ObservationLike:
        row = self.arm_for(arm_id)
        if row.observation is None:
            raise KeyError(arm_id)
        return row.observation

    def score_delta_for(self, arm_id: str) -> TokenScoreDelta:
        row = self.arm_for(arm_id)
        if not isinstance(self.control, TokenScoreObservation) or not isinstance(row.observation, TokenScoreObservation):
            return TokenScoreDelta(
                observation_id=row.observation_id or "obs_unavailable", status="unavailable",
                diagnostics={"reason": "baseline_score_unavailable"},
            )
        return TokenScoreDelta.from_observations(self.control, row.observation)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id, "base": self.base.to_dict(),
            "evaluator": self.evaluator.to_dict(), "state": self.state,
            "control": self.control.to_dict() if self.control else None,
            "arms": [row.to_dict() for row in self.arm_rows],
            "diagnostics": dict(self.diagnostics), "timing": dict(self.timing),
            "execution_provenance": dict(self.execution_provenance),
            "requested_by": dict(self.requested_by), "persisted": self.persisted,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentResult":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"ExperimentResult must declare {SCHEMA_VERSION}")
        evaluator = evaluator_from_dict(value.get("evaluator"))
        control_value = value.get("control")
        control = _observation_from_dict(control_value) if isinstance(control_value, Mapping) else None
        rows: list[ExperimentArmView] = []
        interventions: list[Intervention] = []
        observations: list[ObservationLike | None] = []
        states: list[str] = []
        ids: list[str] = []
        obs_ids: list[str | None] = []
        for raw in value.get("arms") or []:
            intervention = intervention_from_dict(raw.get("intervention"))
            observation_value = raw.get("observation")
            observation = _observation_from_dict(observation_value) if isinstance(observation_value, Mapping) else None
            row = ExperimentArmView(
                experiment_id=value.get("experiment_id"), arm_id=raw.get("arm_id"), ordinal=int(raw.get("ordinal", len(rows))),
                intervention=intervention, condition=raw.get("condition") or {}, state=raw.get("state"),
                observation_id=raw.get("observation_id"), observation=observation,
                error=raw.get("error"), diagnostics=raw.get("diagnostics"),
            )
            rows.append(row); interventions.append(intervention); observations.append(observation)
            states.append(row.state); ids.append(row.arm_id); obs_ids.append(row.observation_id)
        base_value = value.get("base")
        base = (ResolvedState.from_dict(base_value)
                if isinstance(base_value, Mapping)
                and base_value.get("schema_version") == "clozn.experiment-resolved-state.v1"
                else ExecutionState.from_dict(base_value))
        return cls(
            experiment_id=value.get("experiment_id"), base=base,
            evaluator=evaluator, control=control, arm_observations=observations,
            arm_interventions=interventions, arm_ids=ids, arm_states=states, observation_ids=obs_ids,
            state=value.get("state"), diagnostics=value.get("diagnostics"), timing=value.get("timing"),
            execution_provenance=value.get("execution_provenance"), arm_rows=tuple(rows),
            requested_by=value.get("requested_by"),
            persisted=bool(value.get("persisted")),
        )


def _observation_from_dict(value: Mapping[str, Any]) -> ObservationLike:
    return observation_from_dict(value)


def _validate_returned_observation(observation: Any, expected_type: type,
                                   state: ExecutionState | ResolvedState, evaluator: Evaluator,
                                   intervention: Intervention | None) -> ObservationLike:
    compatible = (
        isinstance(observation, TokenScoreObservation)
        if expected_type is TokenScoreObservation
        else type(observation) is expected_type
    )
    if not compatible:
        raise ObservationError("execution adapter returned an observation incompatible with evaluator")
    expected = _identity_for(state, evaluator, intervention)
    if observation.observation_id != expected["observation_id"] or observation.observation_key_sha256 != expected["observation_key_sha256"]:
        raise ObservationIntegrityError("execution adapter returned evidence for a different condition")
    return observation


def _execute(adapter: Any, state: ExecutionState | ResolvedState, evaluator: Evaluator,
             intervention: Intervention | None, *, arm_id: str | None) -> ObservationLike:
    execute = getattr(adapter, "execute", None)
    if not callable(execute):
        raise TypeError("execution adapter exposes no execute method")
    return execute(state, intervention, evaluator=evaluator, arm_id=arm_id)


def run_experiment(experiment: Experiment, execution_adapter: Any, *, include_control: bool = True,
                   cancel: Any = None, observation_store: ObservationStore | None = None,
                   store: ObservationStore | None = None,
                   requested_by: Mapping[str, Any] | None = None,
                   diagnostics: Mapping[str, Any] | None = None) -> ExperimentResult:
    """Run a plan, reusing durable direct observations when a store is supplied."""
    if not isinstance(experiment, Experiment):
        raise TypeError("run_experiment requires a new-kernel Experiment")
    if execution_adapter is None:
        raise TypeError("run_experiment requires an execution adapter")
    if observation_store is not None and store is not None and observation_store is not store:
        raise ValueError("pass only one observation store")
    durable = observation_store or store
    if durable is not None and not isinstance(durable, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if durable is not None:
        durable.create_experiment(experiment, requested_by=requested_by)
        prior_view = durable.get_experiment(experiment.experiment_id)
        prior_arms = {row.arm_id: row for row in prior_view.arms}
        durable.set_experiment_state(experiment.experiment_id, "running")
    else:
        prior_view = None
        prior_arms = {}

    control: ObservationLike | None = None
    diagnostics = dict(diagnostics or {})
    if requested_by is not None:
        diagnostics["requested_by"] = dict(requested_by)
    observation_cache: dict[str, ObservationLike] = {}
    cancelled = bool(cancel()) if callable(cancel) else False
    if cancelled:
        diagnostics["cancelled"] = True
    control_identity = _identity_for(experiment.base, experiment.evaluator, None)
    if include_control and not cancelled:
        try:
            if durable is not None:
                control = durable.find_observation(control_identity["observation_key_sha256"])
                if control is not None:
                    diagnostics["control_reused"] = True
            if control is None:
                execute_control = getattr(execution_adapter, "execute_control", None)
                if callable(execute_control):
                    control = execute_control(experiment.base, evaluator=experiment.evaluator)
                else:
                    control = _execute(execution_adapter, experiment.base, experiment.evaluator, None, arm_id=None)
                control = _validate_returned_observation(
                    control, _expected_type(experiment.evaluator), experiment.base, experiment.evaluator, None,
                )
                if durable is not None and _reusable(control):
                    durable.associate_observation(experiment.experiment_id, "control", control)
                elif durable is not None:
                    durable.update_arm(
                        experiment.experiment_id, "control", state="failed",
                        diagnostics={"observation_status": control.status, "diagnostics": control.diagnostics},
                    )
            elif durable is not None:
                durable.update_arm(experiment.experiment_id, "control", state="completed", observation_id=control.observation_id)
        except Exception as exc:
            diagnostics["control_error"] = str(exc)
            if durable is not None:
                durable.update_arm(experiment.experiment_id, "control", state="failed", error={"error": str(exc)})
    elif include_control and cancelled and prior_view is not None:
        control = prior_view.control
        if durable is not None and control is None:
            durable.update_arm(experiment.experiment_id, "control", state="cancelled", diagnostics={"reason": "cancelled"})
    elif durable is not None:
        durable.update_arm(experiment.experiment_id, "control", state="cancelled", diagnostics={"reason": "cancelled"})

    blocked = include_control and not cancelled and not _allows_arms(experiment.evaluator, control)
    arm_observations: list[ObservationLike | None] = []
    arm_states: list[str] = []
    arm_observation_ids: list[str | None] = []
    arm_errors: list[Mapping[str, Any] | None] = []
    arm_diagnostics: list[Mapping[str, Any] | None] = []
    expected_type = _expected_type(experiment.evaluator)
    for arm in experiment.arms:
        if blocked:
            state, observation, observation_id = "blocked", None, None
            if durable is not None:
                durable.update_arm(experiment.experiment_id, arm.arm_id, state="blocked",
                                   diagnostics={
                                       "reason": "control_observation_not_available",
                                       "observation_status": control.status if control is not None else "unavailable",
                                   })
            arm_observations.append(observation); arm_states.append(state); arm_observation_ids.append(observation_id); arm_errors.append({})
            arm_diagnostics.append({
                "reason": "control_observation_not_available",
                "observation_status": control.status if control is not None else "unavailable",
            })
            continue
        if cancelled or (callable(cancel) and cancel()):
            cancelled = True
            prior_arm = prior_arms.get(arm.arm_id)
            if prior_arm is not None and prior_arm.observation is not None:
                observation = prior_arm.observation
                state, observation_id = prior_arm.state, prior_arm.observation_id
                arm_observations.append(observation); arm_states.append(state); arm_observation_ids.append(observation_id); arm_errors.append(prior_arm.error)
                arm_diagnostics.append(prior_arm.diagnostics)
                continue
            if durable is not None:
                durable.update_arm(experiment.experiment_id, arm.arm_id, state="cancelled", diagnostics={"reason": "cancelled"})
            arm_observations.append(None); arm_states.append("cancelled"); arm_observation_ids.append(None); arm_errors.append({})
            arm_diagnostics.append({"reason": "cancelled"})
            continue
        identity = _identity_for(experiment.base, experiment.evaluator, arm.intervention)
        observation = (
            durable.find_observation(identity["observation_key_sha256"])
            if durable is not None else observation_cache.get(identity["observation_key_sha256"])
        )
        error: dict[str, Any] = {}
        if observation is not None:
            diagnostics["reused_observations"] = diagnostics.get("reused_observations", 0) + 1
            state, observation_id = "completed", observation.observation_id
            if durable is not None:
                durable.update_arm(experiment.experiment_id, arm.arm_id, state=state, observation_id=observation_id)
        else:
            if durable is not None:
                durable.update_arm(experiment.experiment_id, arm.arm_id, state="running")
            try:
                observation = _execute(execution_adapter, experiment.base, experiment.evaluator, arm.intervention,
                                       arm_id=arm.arm_id)
                observation = _validate_returned_observation(
                    observation, expected_type, experiment.base, experiment.evaluator, arm.intervention,
                )
                if durable is not None and _reusable(observation):
                    observation_id = durable.associate_observation(experiment.experiment_id, arm.arm_id, observation)
                elif durable is not None:
                    observation_id = None
                    durable.update_arm(
                        experiment.experiment_id, arm.arm_id, state="failed",
                        diagnostics={"observation_status": observation.status,
                                     "diagnostics": observation.diagnostics},
                    )
                else:
                    observation_id = observation.observation_id if _reusable(observation) else None
                    if _reusable(observation):
                        observation_cache[identity["observation_key_sha256"]] = observation
                state = "completed" if _reusable(observation) else "failed"
            except Exception as exc:
                error = {"error": str(exc)}
                state, observation_id = "failed", None
                if durable is not None:
                    durable.update_arm(experiment.experiment_id, arm.arm_id, state="failed", error=error)
                observation = None
        arm_observations.append(observation); arm_states.append(state); arm_observation_ids.append(observation_id); arm_errors.append(error)
        arm_diagnostics.append(dict(getattr(observation, "diagnostics", {}) or {}))

    if cancelled:
        final_state = "cancelled"
    elif blocked:
        final_state = "blocked"
    elif any(state == "failed" for state in arm_states) or (include_control and control is None):
        final_state = "failed"
    else:
        final_state = "completed"
    if durable is not None:
        durable.set_experiment_state(experiment.experiment_id, final_state, diagnostics=diagnostics)
        return ExperimentResult.from_view(
            durable.get_experiment(experiment.experiment_id), observation_store=durable,
        )
    result = ExperimentResult(
        experiment_id=experiment.experiment_id, base=experiment.base, evaluator=experiment.evaluator,
        control=control, arm_observations=tuple(arm_observations), arm_interventions=experiment.arms and tuple(arm.intervention for arm in experiment.arms) or (),
        arm_ids=tuple(arm.arm_id for arm in experiment.arms), arm_states=tuple(arm_states),
        observation_ids=tuple(arm_observation_ids), arm_errors=tuple(arm_errors), state=final_state,
        arm_diagnostics=tuple(arm_diagnostics),
        diagnostics=diagnostics, timing={"control_executed": bool(include_control), "arms_executed": sum(item is not None for item in arm_observations)},
        execution_provenance={"runner": "experimental_kernel", "arms_ephemeral": True},
        requested_by=requested_by,
    )
    return result


__all__ = ["ExperimentResult", "SCHEMA_VERSION", "run_experiment"]
