"""The feature-independent, model-free experiment plan."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from typing import Any

from .evaluators import Evaluator, ExactReferenceMatch, ScoreRecordedContinuation, Generate, evaluator_from_dict
from .interventions import DeleteSource, ForceToken, Intervention, SampleWith, intervention_from_dict
from .observations import Observation
from .selections import ContextSelection
from .state import ExecutionState, canonical_json
from .state_ref import ResolvedState


SCHEMA_VERSION = "clozn.experiment.v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ExperimentArm:
    """An ephemeral, ordered intervention with a deterministic ID."""

    __slots__ = ("arm_id", "intervention", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ExperimentArm is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, arm_id: str, intervention: Intervention | None):
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("ExperimentArm.arm_id must be a non-empty string")
        if intervention is not None and not isinstance(intervention, (DeleteSource, ForceToken, SampleWith)):
            raise TypeError("Experiment arms must be supported typed interventions or None")
        self.arm_id = arm_id
        self.intervention = intervention
        self._sealed = True

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id,
                "intervention": self.intervention.to_dict() if self.intervention is not None else None}


class Experiment:
    """An immutable base binding, evaluator, and ordered ephemeral arms."""

    __slots__ = ("base", "evaluator", "arms", "experiment_id", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Experiment is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, base: ExecutionState | ResolvedState, evaluator: Evaluator,
                 arms: Iterable[Intervention | None]):
        if not isinstance(base, (ExecutionState, ResolvedState)):
            raise TypeError("Experiment.base must be an ExecutionState or ResolvedState")
        if not isinstance(evaluator, (ExactReferenceMatch, ScoreRecordedContinuation, Generate)):
            raise TypeError("Experiment.evaluator must be a supported new-kernel evaluator")
        if isinstance(arms, (str, bytes)):
            raise TypeError("Experiment.arms must be an iterable of interventions")
        raw_arms = list(arms)
        expected_intervention = (ForceToken, DeleteSource, SampleWith) if isinstance(evaluator, Generate) else DeleteSource
        if not all(
            (arm is None and isinstance(evaluator, Generate))
            or isinstance(arm, expected_intervention)
            for arm in raw_arms
        ):
            suffix = " or an unchanged condition" if isinstance(evaluator, Generate) else ""
            name = "ForceToken, SampleWith, or DeleteSource" if isinstance(evaluator, Generate) else "DeleteSource"
            raise TypeError(f"{type(evaluator).__name__} experiments require {name} arms{suffix}")
        if isinstance(evaluator, Generate) and isinstance(base, ResolvedState) and base.classification == "unavailable":
            raise ValueError("Generate experiments cannot be created from an unavailable ResolvedState")
        if isinstance(evaluator, Generate) and isinstance(base, ExecutionState):
            if any(arm is None or not isinstance(arm, DeleteSource) for arm in raw_arms):
                raise TypeError("ExecutionState Generate experiments require DeleteSource arms")
        if isinstance(evaluator, Generate) and isinstance(base, ResolvedState):
            if any(arm is not None and not isinstance(arm, (ForceToken, SampleWith)) for arm in raw_arms):
                raise TypeError("ResolvedState Generate experiments require ForceToken, SampleWith, or unchanged arms")
        arm_payload = [arm.to_dict() if arm is not None else None for arm in raw_arms]
        base_identity = _base_identity(base)
        binding = {
            "base": base_identity,
            "evaluator": evaluator.to_dict(),
            "arms": arm_payload,
        }
        experiment_id = "exp_" + _digest(binding)[:24]
        assigned = []
        for index, intervention in enumerate(raw_arms):
            arm_id = "arm_" + _digest({
                "base": base_identity,
                "index": index,
                "intervention": intervention.to_dict() if intervention is not None else None,
            })[:24]
            assigned.append(ExperimentArm(arm_id, intervention))
        self.base = base
        self.evaluator = evaluator
        self.arms = tuple(assigned)
        self.experiment_id = experiment_id
        self._sealed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "experiment_id": self.experiment_id,
            "base": self.base.to_dict(),
            "evaluator": self.evaluator.to_dict(),
            "arms": [arm.to_dict() for arm in self.arms],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Experiment":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"Experiment must declare {SCHEMA_VERSION}")
        raw_arms = value.get("arms")
        if not isinstance(raw_arms, list):
            raise ValueError("Experiment.arms must be a list")
        base_value = value.get("base")
        if isinstance(base_value, Mapping) and base_value.get("schema_version") == "clozn.experiment-resolved-state.v1":
            base = ResolvedState.from_dict(base_value)
        else:
            base = ExecutionState.from_dict(base_value)
        experiment = cls(
            base=base,
            evaluator=evaluator_from_dict(value.get("evaluator")),
            arms=[
                intervention_from_dict(item.get("intervention"))
                if item.get("intervention") is not None else None
                for item in raw_arms
            ],
        )
        if value.get("experiment_id") != experiment.experiment_id:
            raise ValueError("Experiment ID does not match its deterministic plan")
        supplied_ids = [item.get("arm_id") for item in raw_arms]
        if supplied_ids != [arm.arm_id for arm in experiment.arms]:
            raise ValueError("Experiment arm IDs do not match their deterministic plan")
        return experiment


def _base_identity(base: ExecutionState | ResolvedState) -> dict[str, Any]:
    if isinstance(base, ResolvedState):
        return {
            "kind": "resolved_state",
            "state_ref": base.state_ref.identity_payload(),
            "classification": base.classification,
            "realization_fingerprint": base.realization_fingerprint,
        }
    return {"kind": "execution_state", "execution_fingerprint": base.execution_fingerprint}


__all__ = ["Experiment", "ExperimentArm", "Observation", "SCHEMA_VERSION"]
