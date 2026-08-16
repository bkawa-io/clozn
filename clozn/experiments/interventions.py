"""Typed intervention declarations.  Execution adapters own their effects."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .selections import ContextSelection


class InterventionError(ValueError):
    """A malformed typed intervention."""


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


__all__ = ["DeleteSource", "InterventionError"]
