"""The model-free experiment plan shared by Context Effects and Minimal Context."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
from typing import Any

from .evaluators import Evaluator, ExactReferenceMatch, ScoreRecordedContinuation, evaluator_from_dict
from .interventions import DeleteSource, Intervention, intervention_from_dict
from .observations import Observation
from .state import ExecutionState, canonical_json


SCHEMA_VERSION = "clozn.experiment.v1"


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


class ExperimentArm:
    """An ephemeral, ordered source-deletion intervention with a deterministic ID."""

    __slots__ = ("arm_id", "intervention", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ExperimentArm is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, arm_id: str, intervention: Intervention):
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("ExperimentArm.arm_id must be a non-empty string")
        if not isinstance(intervention, DeleteSource):
            raise TypeError("Experiment arms require DeleteSource interventions")
        self.arm_id = arm_id
        self.intervention = intervention
        self._sealed = True

    def to_dict(self) -> dict[str, Any]:
        return {"arm_id": self.arm_id, "intervention": self.intervention.to_dict()}


class Experiment:
    """An immutable execution binding, retained evaluator, and ordered deletion arms."""

    __slots__ = ("base", "evaluator", "arms", "experiment_id", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Experiment is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, base: ExecutionState, evaluator: Evaluator,
                 arms: Iterable[Intervention]):
        if not isinstance(base, ExecutionState):
            raise TypeError("Experiment.base must be an ExecutionState")
        if not isinstance(evaluator, (ExactReferenceMatch, ScoreRecordedContinuation)):
            raise TypeError("Experiment.evaluator must be a retained v1 evaluator")
        if isinstance(arms, (str, bytes)):
            raise TypeError("Experiment.arms must be an iterable of interventions")
        raw_arms = list(arms)
        if not all(isinstance(arm, DeleteSource) for arm in raw_arms):
            raise TypeError(f"{type(evaluator).__name__} experiments require DeleteSource arms")
        arm_payload = [arm.to_dict() for arm in raw_arms]
        base_identity = {"kind": "execution_state", "execution_fingerprint": base.execution_fingerprint}
        experiment_id = "exp_" + _digest({
            "base": base_identity,
            "evaluator": evaluator.to_dict(),
            "arms": arm_payload,
        })[:24]
        assigned = []
        for index, intervention in enumerate(raw_arms):
            arm_id = "arm_" + _digest({
                "base": base_identity,
                "index": index,
                "intervention": intervention.to_dict(),
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
        experiment = cls(
            base=ExecutionState.from_dict(value.get("base")),
            evaluator=evaluator_from_dict(value.get("evaluator")),
            arms=[intervention_from_dict(item.get("intervention")) for item in raw_arms],
        )
        if value.get("experiment_id") != experiment.experiment_id:
            raise ValueError("Experiment ID does not match its deterministic plan")
        if [item.get("arm_id") for item in raw_arms] != [arm.arm_id for arm in experiment.arms]:
            raise ValueError("Experiment arm IDs do not match their deterministic plan")
        return experiment


__all__ = ["Experiment", "ExperimentArm", "Observation", "SCHEMA_VERSION"]
