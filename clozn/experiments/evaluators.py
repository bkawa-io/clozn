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


class Generate:
    """Continue generation from a resolved state under a deterministic contract.

    ``max_new`` is the total number of generated tokens in the observed suffix;
    a forced token, when present, consumes the first slot.
    """

    __slots__ = ("max_new", "decode_mode", "sampling", "stop", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Generate is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, max_new: int, decode_mode: str = "greedy",
                 sampling: Mapping[str, Any] | None = None,
                 stop: list[str] | tuple[str, ...] = ()):
        if isinstance(max_new, bool) or not isinstance(max_new, int) or max_new <= 0:
            raise EvaluatorError("Generate.max_new must be a positive integer")
        if decode_mode not in {"greedy", "sample"}:
            raise EvaluatorError("Generate.decode_mode must be greedy or sample")
        if decode_mode == "sample":
            if not isinstance(sampling, Mapping):
                raise EvaluatorError("sample Generate requires explicit sampling parameters")
            required = {"temperature", "top_p", "top_k", "repeat_penalty", "seed"}
            if not required.issubset(sampling):
                raise EvaluatorError("sample Generate requires a complete sampler contract")
        elif sampling is not None:
            raise EvaluatorError("greedy Generate cannot carry sampling parameters")
        if not isinstance(stop, (list, tuple)) or any(not isinstance(item, str) for item in stop):
            raise EvaluatorError("Generate.stop must contain strings")
        self.max_new = max_new
        self.decode_mode = decode_mode
        self.sampling = dict(sampling) if sampling is not None else None
        self.stop = tuple(stop)
        self._sealed = True

    def __repr__(self) -> str:
        return f"Generate(max_new={self.max_new!r}, decode_mode={self.decode_mode!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Generate) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash((type(self), self.max_new, self.decode_mode,
                     tuple(sorted((self.sampling or {}).items())), self.stop))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "generate", "max_new": self.max_new,
            "decode_mode": self.decode_mode,
            "sampling": dict(self.sampling) if self.sampling is not None else None,
            "stop": list(self.stop),
        }

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Generate":
        if not isinstance(value, Mapping) or value.get("kind") != "generate":
            raise EvaluatorError("expected a generate object")
        return cls(max_new=value.get("max_new"), decode_mode=value.get("decode_mode", "greedy"),
                   sampling=value.get("sampling"), stop=value.get("stop") or ())


Evaluator = Union[ExactReferenceMatch, ScoreRecordedContinuation, Generate]


def evaluator_from_dict(value: Mapping[str, Any]) -> Evaluator:
    if not isinstance(value, Mapping):
        raise EvaluatorError("evaluator must be an object")
    kind = value.get("kind")
    if kind == "exact_reference_match":
        return ExactReferenceMatch.from_dict(value)
    if kind == "score_recorded_continuation":
        return ScoreRecordedContinuation.from_dict(value)
    if kind == "generate":
        return Generate.from_dict(value)
    raise EvaluatorError(f"unsupported evaluator kind: {kind!r}")


__all__ = [
    "Evaluator", "EvaluatorError", "ExactReferenceMatch", "ScoreRecordedContinuation", "Generate",
    "evaluator_from_dict",
]
