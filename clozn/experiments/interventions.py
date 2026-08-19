"""Typed intervention declarations.  Execution adapters own their effects."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any, Union

from .selections import ContextSelection


class InterventionError(ValueError):
    """A malformed typed intervention."""


def _non_negative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _finite(value: Any, *, minimum: float | None = None, maximum: float | None = None) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return False
    if minimum is not None and value < minimum:
        return False
    return not (maximum is not None and value > maximum)


class DeleteSource:
    """Declare deletion of canonical Context Receipt sources.

    This object does not resolve a receipt or modify prompt messages.
    """

    __slots__ = ("target", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("DeleteSource is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, target: ContextSelection):
        if not isinstance(target, ContextSelection):
            raise InterventionError("DeleteSource.target must be a ContextSelection")
        self.target = target
        self._sealed = True

    @property
    def source_ids(self) -> tuple[str, ...]:
        return self.target.source_ids

    def __repr__(self) -> str:
        return f"DeleteSource(target={self.target!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DeleteSource) and self.target == other.target

    def __hash__(self) -> int:
        return hash((type(self), self.target))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "delete_source", "target": self.target.to_dict()}

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DeleteSource":
        if not isinstance(value, Mapping) or value.get("kind") != "delete_source":
            raise InterventionError("expected a delete_source object")
        return cls(ContextSelection.from_dict(value.get("target")))


class ForceToken:
    """Force one token at the boundary addressed by ``StateRef``.

    The location is deliberately absent here: a StateRef owns the logical
    position, while this declaration owns only the replacement token.
    """

    __slots__ = ("token_id", "token_piece", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ForceToken is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, token_id: int | None = None, token_piece: str | None = None):
        if token_id is None and token_piece is None:
            raise InterventionError("ForceToken requires token_id or token_piece")
        if token_id is not None and (
            isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        ):
            raise InterventionError("ForceToken.token_id must be a non-negative integer")
        if token_piece is not None and (not isinstance(token_piece, str) or not token_piece):
            raise InterventionError("ForceToken.token_piece must be a non-empty string when supplied")
        self.token_id = token_id
        self.token_piece = token_piece
        self._sealed = True

    def __repr__(self) -> str:
        return f"ForceToken(token_id={self.token_id!r}, token_piece={self.token_piece!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ForceToken) and (
            self.token_id, self.token_piece
        ) == (other.token_id, other.token_piece)

    def __hash__(self) -> int:
        return hash((type(self), self.token_id, self.token_piece))

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"kind": "force_token"}
        if self.token_id is not None:
            value["token_id"] = self.token_id
        if self.token_piece is not None:
            value["token_piece"] = self.token_piece
        return value

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ForceToken":
        if not isinstance(value, Mapping) or value.get("kind") != "force_token":
            raise InterventionError("expected a force_token object")
        return cls(token_id=value.get("token_id"), token_piece=value.get("token_piece"))


class SampleWith:
    """Declare a sampler override for the continuation from a resolved state.

    Like :class:`ForceToken`, this owns only the change, never the location.  It names the sampler
    fields the caller wants changed relative to the recorded regime; the fully resolved five-field
    sampler that actually ran is worker evidence, reported on the resulting observation rather than
    assumed here.  Field names and ranges are the canonical exact-resume sampler contract.
    """

    __slots__ = ("temperature", "top_k", "top_p", "seed", "rep_penalty", "_sealed")

    FIELDS = ("temperature", "top_k", "top_p", "seed", "rep_penalty")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("SampleWith is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, temperature=None, top_k=None, top_p=None, seed=None, rep_penalty=None):
        supplied = {
            "temperature": temperature, "top_k": top_k, "top_p": top_p,
            "seed": seed, "rep_penalty": rep_penalty,
        }
        if all(value is None for value in supplied.values()):
            raise InterventionError("SampleWith needs at least one sampler override")
        if temperature is not None and not _finite(temperature, minimum=0):
            raise InterventionError("SampleWith.temperature must be a finite number >= 0")
        if top_p is not None and not _finite(top_p, minimum=0, maximum=1):
            raise InterventionError("SampleWith.top_p must be a finite number in [0, 1]")
        if top_k is not None and not _non_negative_int(top_k):
            raise InterventionError("SampleWith.top_k must be a non-negative integer")
        if seed is not None and not _non_negative_int(seed):
            raise InterventionError("SampleWith.seed must be a non-negative integer")
        if rep_penalty is not None and not (_finite(rep_penalty) and rep_penalty > 0):
            raise InterventionError("SampleWith.rep_penalty must be a finite number > 0")
        for name, value in supplied.items():
            object.__setattr__(self, name, value)
        self._sealed = True

    @property
    def overrides(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.FIELDS if getattr(self, name) is not None}

    def __repr__(self) -> str:
        return f"SampleWith({self.overrides!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SampleWith) and self.overrides == other.overrides

    def __hash__(self) -> int:
        return hash((type(self), tuple(sorted(self.overrides.items()))))

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "sample_with", **self.overrides}

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    def wire_change(self) -> dict[str, Any]:
        """The override in the worker's own exact-resume intervention vocabulary."""
        return {"type": "sampling", **self.overrides}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SampleWith":
        if not isinstance(value, Mapping) or value.get("kind") != "sample_with":
            raise InterventionError("expected a sample_with object")
        unknown = set(value) - {"kind", *cls.FIELDS}
        if unknown:
            raise InterventionError("sample_with has unknown fields")
        return cls(**{name: value.get(name) for name in cls.FIELDS})


def sampler_override_contract() -> dict[str, Any]:
    """Describe the sampler fields :class:`SampleWith` accepts, for read-side affordances.

    Metadata, not a second validator: the ranges below are the ones ``SampleWith.__init__`` actually
    enforces, so an Inspector client can render the contract without maintaining a parallel table
    that could drift from the rule it describes.
    """
    return {
        "type": "object",
        "properties": {
            "temperature": {"type": "number", "minimum": 0},
            "top_k": {"type": "integer", "minimum": 0},
            "top_p": {"type": "number", "minimum": 0, "maximum": 1},
            "seed": {"type": "integer", "minimum": 0},
            "rep_penalty": {"type": "number", "exclusiveMinimum": 0},
        },
        "required": [],
        "min_fields": 1,
    }


Intervention = Union[DeleteSource, ForceToken, SampleWith]


def intervention_from_dict(value: Mapping[str, Any]) -> Intervention:
    if not isinstance(value, Mapping):
        raise InterventionError("intervention must be an object")
    kind = value.get("kind")
    if kind == "delete_source":
        return DeleteSource.from_dict(value)
    if kind == "force_token":
        return ForceToken.from_dict(value)
    if kind == "sample_with":
        return SampleWith.from_dict(value)
    raise InterventionError(f"unsupported intervention kind: {kind!r}")


__all__ = ["DeleteSource", "ForceToken", "Intervention", "InterventionError", "SampleWith",
           "intervention_from_dict", "sampler_override_contract"]
