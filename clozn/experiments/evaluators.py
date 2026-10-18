"""Model-free evaluator descriptors."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union


class EvaluatorError(ValueError):
    """A malformed evaluator descriptor."""


class ExactReferenceMatch:
    """Compare an intervention execution to the exact recorded continuation."""

    __slots__ = ("_sealed",)

    def __init__(self):
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        raise AttributeError("ExactReferenceMatch is immutable")

    def __repr__(self) -> str:
        return "ExactReferenceMatch()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ExactReferenceMatch)

    def __hash__(self) -> int:
        return hash(type(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "exact_reference_match",
            "reference": "recorded_output",
        }

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExactReferenceMatch":
        if not isinstance(value, Mapping) or value.get("kind") != "exact_reference_match":
            raise EvaluatorError("expected an exact_reference_match object")
        if value.get("reference", "recorded_output") != "recorded_output":
            raise EvaluatorError("ExactReferenceMatch reference must be recorded_output")
        return cls()


class ScoreRecordedContinuation:
    """Score the complete recorded continuation under a counterfactual prompt."""

    __slots__ = ("_sealed",)

    def __init__(self):
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name, value):
        raise AttributeError("ScoreRecordedContinuation is immutable")

    def __repr__(self) -> str:
        return "ScoreRecordedContinuation()"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ScoreRecordedContinuation)

    def __hash__(self) -> int:
        return hash(type(self))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "score_recorded_continuation",
            "target": "full_recorded_continuation",
        }

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ScoreRecordedContinuation":
        if not isinstance(value, Mapping) or value.get("kind") != "score_recorded_continuation":
            raise EvaluatorError("expected a score_recorded_continuation object")
        if value.get("target", "full_recorded_continuation") != "full_recorded_continuation":
            raise EvaluatorError("ScoreRecordedContinuation target must be full_recorded_continuation")
        return cls()


Evaluator = Union[ExactReferenceMatch, ScoreRecordedContinuation]


def evaluator_from_dict(value: Mapping[str, Any]) -> Evaluator:
    if not isinstance(value, Mapping):
        raise EvaluatorError("evaluator must be an object")
    kind = value.get("kind")
    if kind == "exact_reference_match":
        return ExactReferenceMatch.from_dict(value)
    if kind == "score_recorded_continuation":
        return ScoreRecordedContinuation.from_dict(value)
    raise EvaluatorError(f"unsupported evaluator kind: {kind!r}")


__all__ = [
    "Evaluator", "EvaluatorError", "ExactReferenceMatch", "ScoreRecordedContinuation",
    "evaluator_from_dict",
]
