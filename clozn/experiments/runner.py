"""Deterministic orchestration for the experimental kernel."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

from .evaluators import Evaluator, ExactReferenceMatch, ScoreRecordedContinuation, evaluator_from_dict
from .interventions import DeleteSource
from .kernel import Experiment
from .observations import Observation, TokenScoreObservation
from .state import ExecutionState, canonical_json


SCHEMA_VERSION = "clozn.experiment-result.v1"


ObservationLike = Union[Observation, TokenScoreObservation]


def _failed(arm_id: str, evaluator: Evaluator, reason: str, error: str | None = None) -> ObservationLike:
    diagnostics = {"reason": reason}
    if error:
        diagnostics["error"] = error
    if isinstance(evaluator, ScoreRecordedContinuation):
        return TokenScoreObservation(
            arm_id=arm_id, status="failed",
            evaluator_provenance={"evaluator": "score_recorded_continuation"},
            execution_provenance={"runner": "experimental_kernel"}, diagnostics=diagnostics,
        )
    return Observation(arm_id=arm_id, status="failed",
                       execution_provenance={"runner": "experimental_kernel"},
                       diagnostics=diagnostics)


class ExperimentResult:
    """One result document; its arms remain evidence, never persisted Runs."""

    __slots__ = (
        "experiment_id", "base", "evaluator", "control", "arm_observations",
        "arm_interventions",
        "state", "diagnostics", "timing", "execution_provenance",
    )

    def __init__(self, *, experiment_id: str, base: ExecutionState,
                 evaluator: Evaluator, control: ObservationLike | None,
                 arm_observations: list[ObservationLike] | tuple[ObservationLike, ...],
                 arm_interventions: list[DeleteSource] | tuple[DeleteSource, ...] = (),
                 state: str, diagnostics: Mapping[str, Any] | None = None,
                 timing: Mapping[str, Any] | None = None,
                 execution_provenance: Mapping[str, Any] | None = None):
        if not isinstance(experiment_id, str) or not experiment_id:
            raise ValueError("ExperimentResult.experiment_id must be non-empty")
        if not isinstance(base, ExecutionState):
            raise TypeError("ExperimentResult.base must be an ExecutionState")
        if not isinstance(evaluator, (ExactReferenceMatch, ScoreRecordedContinuation)):
            raise TypeError("ExperimentResult.evaluator must be a supported evaluator")
        if state not in {"completed", "blocked", "failed"}:
            raise ValueError("ExperimentResult.state must be completed, blocked, or failed")
        observations = tuple(arm_observations)
        expected_type = TokenScoreObservation if isinstance(evaluator, ScoreRecordedContinuation) else Observation
        if any(not isinstance(item, expected_type) for item in observations):
            raise TypeError("ExperimentResult arm observations do not match its evaluator")
        if control is not None and not isinstance(control, expected_type):
            raise TypeError("ExperimentResult control does not match its evaluator")
        interventions = tuple(arm_interventions)
        if len(interventions) != len(observations):
            raise ValueError("ExperimentResult arm interventions must align with arm observations")
        if any(not isinstance(item, DeleteSource) for item in interventions):
            raise TypeError("ExperimentResult arm interventions must be DeleteSource objects")
        self.experiment_id = experiment_id
        self.base = base
        self.evaluator = evaluator
        self.control = control
        self.arm_observations = observations
        self.arm_interventions = interventions
        self.state = state
        self.diagnostics = dict(diagnostics or {})
        self.timing = dict(timing or {})
        self.execution_provenance = dict(execution_provenance or {})

    @property
    def arms(self) -> tuple[ObservationLike, ...]:
        return self.arm_observations

    def observation_for(self, arm_id: str) -> ObservationLike:
        for observation in self.arm_observations:
            if observation.arm_id == arm_id:
                return observation
        raise KeyError(arm_id)

    def score_delta_for(self, arm_id: str):
        """Derive one full signed score vector without executing anything."""
        from .observations import TokenScoreDelta
        if not isinstance(self.evaluator, ScoreRecordedContinuation):
            raise TypeError("score_delta_for requires ScoreRecordedContinuation")
        if not isinstance(self.control, TokenScoreObservation):
            return TokenScoreDelta(arm_id=arm_id, status="unavailable",
                                   diagnostics={"reason": "baseline_score_unavailable"})
        return TokenScoreDelta.from_observations(self.control, self.observation_for(arm_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "base": self.base.to_dict(),
            "evaluator": self.evaluator.to_dict(),
            "state": self.state,
            "control": self.control.to_dict() if self.control is not None else None,
            "arms": [observation.to_dict() for observation in self.arm_observations],
            "arm_interventions": [item.to_dict() for item in self.arm_interventions],
            "diagnostics": dict(self.diagnostics),
            "timing": dict(self.timing),
            "execution_provenance": dict(self.execution_provenance),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExperimentResult":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"ExperimentResult must declare {SCHEMA_VERSION}")
        from .observations import Observation, TokenScoreObservation
        arms = value.get("arms")
        if not isinstance(arms, list):
            raise ValueError("ExperimentResult.arms must be a list")
        raw_interventions = value.get("arm_interventions")
        if not isinstance(raw_interventions, list) or len(raw_interventions) != len(arms):
            raise ValueError("ExperimentResult.arm_interventions must align with arms")
        control = value.get("control")
        return cls(
            experiment_id=value.get("experiment_id"),
            base=ExecutionState.from_dict(value.get("base")),
            evaluator=evaluator_from_dict(value.get("evaluator")),
            control=_observation_from_dict(control) if isinstance(control, Mapping) else None,
            arm_observations=[_observation_from_dict(item) for item in arms],
            arm_interventions=[DeleteSource.from_dict(item) for item in raw_interventions],
            state=value.get("state"),
            diagnostics=value.get("diagnostics"),
            timing=value.get("timing"),
            execution_provenance=value.get("execution_provenance"),
        )


def _observation_from_dict(value: Mapping[str, Any]) -> ObservationLike:
    if value.get("schema_version") == "clozn.experiment-observation.v1":
        return Observation.from_dict(value)
    if value.get("schema_version") == "clozn.experiment-token-score-observation.v1":
        return TokenScoreObservation.from_dict(value)
    raise ValueError("unsupported observation schema")


def _blocked_observation(arm_id: str, control: ObservationLike, evaluator: Evaluator) -> ObservationLike:
    if isinstance(evaluator, ScoreRecordedContinuation):
        return TokenScoreObservation(
            arm_id=arm_id, status="unavailable",
            evaluator_provenance={"evaluator": "score_recorded_continuation"},
            execution_provenance={"runner": "experimental_kernel"},
            diagnostics={"reason": "unchanged_control_failed", "control_status": control.status},
        )
    return Observation(
        arm_id=arm_id,
        status="unavailable",
        execution_provenance={"runner": "experimental_kernel"},
        diagnostics={
            "reason": "unchanged_control_failed",
            "control_status": control.status,
        },
    )


def _control_allows_arms(evaluator: Evaluator, control: ObservationLike | None) -> bool:
    if control is None:
        return False
    if isinstance(evaluator, ScoreRecordedContinuation):
        return isinstance(control, TokenScoreObservation) and control.status == "completed"
    return isinstance(control, Observation) and control.status == "exact_preserved"


def run_experiment(experiment: Experiment, execution_adapter: Any, *,
                   include_control: bool = True, cancel: Any = None) -> ExperimentResult:
    """Execute an ordered plan while keeping arms ephemeral.

    The adapter is the only execution dependency.  This function never imports
    the legacy experiment dispatcher and never calls the run store.
    """
    if not isinstance(experiment, Experiment):
        raise TypeError("run_experiment requires a new-kernel Experiment")
    if execution_adapter is None:
        raise TypeError("run_experiment requires an execution adapter")

    control: ObservationLike | None = None
    diagnostics: dict[str, Any] = {}
    if include_control:
        try:
            execute_control = getattr(execution_adapter, "execute_control", None)
            if callable(execute_control):
                control = execute_control(experiment.base, evaluator=experiment.evaluator)
            else:
                execute = getattr(execution_adapter, "execute", None)
                if not callable(execute):
                    raise TypeError("execution adapter exposes neither execute_control nor execute")
                control = execute(experiment.base, None, evaluator=experiment.evaluator, arm_id="control")
            if isinstance(experiment.evaluator, ScoreRecordedContinuation):
                if not isinstance(control, TokenScoreObservation):
                    raise TypeError("execution adapter returned a non-token-score control")
            elif not isinstance(control, Observation):
                raise TypeError("execution adapter returned a non-Observation control")
        except Exception as exc:
            control = _failed("control", experiment.evaluator, "control_execution_failed", str(exc))
            diagnostics["control_error"] = str(exc)

    observations: list[ObservationLike] = []
    blocked = include_control and not _control_allows_arms(experiment.evaluator, control)
    execution_cache: dict[str, ObservationLike] = {}
    cancelled = False
    for arm in experiment.arms:
        if blocked:
            observations.append(_blocked_observation(arm.arm_id, control, experiment.evaluator))
            continue
        if cancelled or (callable(cancel) and cancel()):
            cancelled = True
            if isinstance(experiment.evaluator, ScoreRecordedContinuation):
                observations.append(TokenScoreObservation(
                    arm_id=arm.arm_id, status="unavailable",
                    evaluator_provenance={"evaluator": "score_recorded_continuation"},
                    execution_provenance={"runner": "experimental_kernel"},
                    diagnostics={"reason": "cancelled"},
                ))
            else:
                observations.append(Observation(
                    arm_id=arm.arm_id, status="unavailable",
                    execution_provenance={"runner": "experimental_kernel"},
                    diagnostics={"reason": "cancelled"},
                ))
            continue
        cache_key = canonical_json(arm.intervention.to_dict())
        if cache_key in execution_cache:
            observations.append(execution_cache[cache_key].rebind_arm_id(arm.arm_id))
            diagnostics["duplicate_arms_reused"] = diagnostics.get("duplicate_arms_reused", 0) + 1
            continue
        try:
            execute = getattr(execution_adapter, "execute", None)
            if not callable(execute):
                raise TypeError("execution adapter exposes no execute method")
            observation = execute(
                experiment.base,
                arm.intervention,
                evaluator=experiment.evaluator,
                arm_id=arm.arm_id,
            )
            expected_type = TokenScoreObservation if isinstance(experiment.evaluator, ScoreRecordedContinuation) else Observation
            if not isinstance(observation, expected_type):
                raise TypeError("execution adapter returned an observation incompatible with evaluator")
        except Exception as exc:
            observation = _failed(arm.arm_id, experiment.evaluator, "arm_execution_failed", str(exc))
        execution_cache[cache_key] = observation
        observations.append(observation)

    state = "blocked" if blocked else "completed"
    if control is not None and control.status == "failed" and not observations:
        state = "failed"
    diagnostics.setdefault("arms_executed", 0 if blocked else len(observations))
    if blocked:
        diagnostics.setdefault("blocked_by_control", True)
    return ExperimentResult(
        experiment_id=experiment.experiment_id,
        base=experiment.base,
        evaluator=experiment.evaluator,
        control=control,
        arm_observations=observations,
        arm_interventions=[arm.intervention for arm in experiment.arms],
        state=state,
        diagnostics=diagnostics,
        timing={"control_executed": bool(include_control), "arms_executed": 0 if blocked else len(observations)},
        execution_provenance={"runner": "experimental_kernel", "arms_ephemeral": True},
    )


__all__ = ["ExperimentResult", "SCHEMA_VERSION", "run_experiment"]
