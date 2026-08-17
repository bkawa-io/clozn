"""Typed intervention declarations.  Execution adapters own their effects."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Union

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


Intervention = Union[DeleteSource, ForceToken]


def intervention_from_dict(value: Mapping[str, Any]) -> Intervention:
    if not isinstance(value, Mapping):
        raise InterventionError("intervention must be an object")
    kind = value.get("kind")
    if kind == "delete_source":
        return DeleteSource.from_dict(value)
    if kind == "force_token":
        return ForceToken.from_dict(value)
    raise InterventionError(f"unsupported intervention kind: {kind!r}")


__all__ = ["DeleteSource", "ForceToken", "Intervention", "InterventionError", "intervention_from_dict"]
