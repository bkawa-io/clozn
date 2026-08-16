"""Explicit immutable selections used by the experimental kernel."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
import hashlib
import re
from typing import Any


_CANONICAL_SOURCE_ID = re.compile(r"^(?:seg|src)_[A-Za-z0-9_-]+$")


class SelectionError(ValueError):
    """A selection is not a valid canonical Context Receipt selection."""


class AnswerSelectionUnavailable(SelectionError):
    """A recorded-answer selection cannot be resolved without approximation."""


class ContextSelection:
    """A sorted, duplicate-free set of canonical ``seg_``/``src_`` IDs.

    The selection is only a declaration.  It never reads the run, resolves a
    receipt, or mutates messages.
    """

    __slots__ = ("source_ids", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ContextSelection is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, source_ids: Iterable[str]):
        if isinstance(source_ids, (str, bytes)):
            raise SelectionError("ContextSelection.source_ids must be an iterable of source IDs")
        try:
            values = list(source_ids)
        except TypeError as exc:
            raise SelectionError("ContextSelection.source_ids must be an iterable of source IDs") from exc
        if not values:
            raise SelectionError("ContextSelection cannot be empty")
        if any(not isinstance(value, str) or not value for value in values):
            raise SelectionError("ContextSelection source IDs must be non-empty strings")
        if len(values) != len(set(values)):
            raise SelectionError("ContextSelection source IDs must not contain duplicates")
        invalid = [value for value in values if not _CANONICAL_SOURCE_ID.fullmatch(value)]
        if invalid:
            raise SelectionError(
                "ContextSelection source IDs must be canonical Context Receipt IDs (seg_ or src_): "
                + ", ".join(invalid)
            )
        self.source_ids = tuple(sorted(values))
        self._sealed = True

    def __iter__(self):
        return iter(self.source_ids)

    def __len__(self):
        return len(self.source_ids)

    def __repr__(self) -> str:
        return f"ContextSelection(source_ids={self.source_ids!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ContextSelection) and self.source_ids == other.source_ids

    def __hash__(self) -> int:
        return hash(self.source_ids)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "context_selection", "source_ids": list(self.source_ids)}

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ContextSelection":
        if not isinstance(value, Mapping) or value.get("kind") != "context_selection":
            raise SelectionError("expected a context_selection object")
        return cls(value.get("source_ids"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _recorded_answer_tokens(run: Mapping[str, Any]) -> tuple[list[int], list[str], str]:
    """Return the exact recorded continuation IDs/pieces, or fail closed.

    This deliberately does not use the text fallback in ``with_arm_conditions``:
    answer addressing is only meaningful when the stored trace itself rebuilds the
    recorded answer byte-for-byte in Python's Unicode code-point space.
    """
    if not isinstance(run, Mapping):
        raise AnswerSelectionUnavailable("recorded answer is unavailable")
    response = run.get("response")
    if not isinstance(response, str):
        raise AnswerSelectionUnavailable("recorded answer text is unavailable")
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    raw_ids = trace.get("token_ids")
    raw_steps = trace.get("steps")
    pieces: list[Any]
    ids: list[Any]
    if isinstance(raw_steps, list) and raw_steps:
        pieces = [item.get("piece") for item in raw_steps if isinstance(item, Mapping)]
        ids = [item.get("token_id") for item in raw_steps if isinstance(item, Mapping)]
    elif isinstance(trace.get("tokens"), list):
        pieces = list(trace["tokens"])
        ids = list(raw_ids) if isinstance(raw_ids, list) else []
    else:
        pieces, ids = [], []
    if (
        not pieces
        or len(pieces) != len(ids)
        or any(not isinstance(piece, str) for piece in pieces)
        or any(not _is_int(token_id) for token_id in ids)
        or "".join(pieces) != response
    ):
        raise AnswerSelectionUnavailable("recorded answer token trace cannot be established exactly")
    return [int(token_id) for token_id in ids], list(pieces), response


class AnswerSelection:
    """A read-side request for a visible span of the recorded answer.

    The preferred form is ``AnswerSelection(start, end)`` or
    ``AnswerSelection(character_range=(start, end), selected_text=...)`` with
    Unicode code-point, half-open offsets.  ``AnswerSelection("some text")`` is
    also supported for callers that have text but not offsets; resolution accepts
    it only when the text occurs exactly once in the recorded answer.

    This declaration is intentionally not part of :class:`Experiment`; it is a
    projection input, never expensive measurement identity.
    """

    __slots__ = ("character_range", "text", "selected_text", "answer_sha256", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("AnswerSelection is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, value: Any = None, end: int | None = None, *,
                 start: int | None = None, text: str | None = None,
                 character_range: Iterable[int] | None = None,
                 selected_text: str | None = None,
                 answer_sha256: str | None = None):
        if start is not None:
            if value is not None or character_range is not None:
                raise SelectionError("AnswerSelection accepts one character range")
            value = start
        if text is not None:
            if value is not None or end is not None or character_range is not None:
                raise SelectionError("AnswerSelection accepts one answer target")
            value = text
        if character_range is not None:
            if value is not None or end is not None:
                raise SelectionError("AnswerSelection accepts one character range")
            value = character_range
        if end is not None:
            if not _is_int(value) or not _is_int(end):
                raise SelectionError("AnswerSelection character range must contain integers")
            value = (value, end)
        range_value: tuple[int, int] | None = None
        text_value: str | None = None
        if isinstance(value, str):
            if not value:
                raise SelectionError("AnswerSelection text cannot be empty")
            text_value = value
            if selected_text is not None and selected_text != value:
                raise SelectionError("selected_text must match the text selection")
        else:
            if value is None:
                raise SelectionError("AnswerSelection requires text or a character range")
            try:
                pair = tuple(value)
            except TypeError as exc:
                raise SelectionError("AnswerSelection requires text or a character range") from exc
            if len(pair) != 2 or not all(_is_int(item) for item in pair):
                raise SelectionError("AnswerSelection character range must be a pair of integers")
            start, finish = pair
            if start < 0 or finish <= start:
                raise SelectionError("AnswerSelection range must be non-empty and half-open")
            range_value = (start, finish)
            if selected_text is not None and not isinstance(selected_text, str):
                raise SelectionError("AnswerSelection.selected_text must be a string")
        if answer_sha256 is not None and (
            not isinstance(answer_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", answer_sha256)
        ):
            raise SelectionError("AnswerSelection.answer_sha256 must be a lowercase SHA-256 digest")
        self.character_range = range_value
        self.text = text_value
        self.selected_text = text_value if text_value is not None else selected_text
        self.answer_sha256 = answer_sha256
        self._sealed = True

    @classmethod
    def from_range(cls, start: int, end: int, *, selected_text: str | None = None,
                   answer_sha256: str | None = None) -> "AnswerSelection":
        return cls(start, end, selected_text=selected_text, answer_sha256=answer_sha256)

    @property
    def range(self) -> tuple[int, int] | None:
        """Alias for callers that use the shorter range terminology."""
        return self.character_range

    def __repr__(self) -> str:
        if self.text is not None:
            return f"AnswerSelection({self.text!r})"
        return f"AnswerSelection({self.character_range!r}, selected_text={self.selected_text!r})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, AnswerSelection)
            and self.character_range == other.character_range
            and self.text == other.text
            and self.selected_text == other.selected_text
            and self.answer_sha256 == other.answer_sha256
        )

    def __hash__(self) -> int:
        return hash((self.character_range, self.text, self.selected_text, self.answer_sha256))

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": "answer_selection",
            "character_range": list(self.character_range) if self.character_range is not None else None,
            "text": self.selected_text,
        }
        if self.answer_sha256 is not None:
            result["answer_sha256"] = self.answer_sha256
        return result

    def to_json(self) -> str:
        from .state import canonical_json
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnswerSelection":
        if not isinstance(value, Mapping) or value.get("kind") != "answer_selection":
            raise SelectionError("expected an answer_selection object")
        raw_range = value.get("character_range")
        if raw_range is not None:
            return cls(raw_range, selected_text=value.get("text"), answer_sha256=value.get("answer_sha256"))
        return cls(value.get("text"), answer_sha256=value.get("answer_sha256"))


class ResolvedAnswerSelection:
    """Exact recorded-token evidence for one :class:`AnswerSelection`."""

    __slots__ = (
        "run_id", "answer_sha256", "character_range", "selected_text",
        "token_indices", "token_ids", "token_pieces", "token_spans",
        "boundary", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ResolvedAnswerSelection is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, run_id: str, answer_sha256: str, character_range: tuple[int, int],
                 selected_text: str, token_indices: Iterable[int], token_ids: Iterable[int],
                 token_pieces: Iterable[str], token_spans: Iterable[tuple[int, int]],
                 boundary: str = "intersecting_recorded_tokens"):
        self.run_id = run_id
        self.answer_sha256 = answer_sha256
        self.character_range = tuple(character_range)
        self.selected_text = selected_text
        self.token_indices = tuple(token_indices)
        self.token_ids = tuple(token_ids)
        self.token_pieces = tuple(token_pieces)
        self.token_spans = tuple(tuple(span) for span in token_spans)
        self.boundary = boundary
        self._sealed = True

    @property
    def token_range(self) -> tuple[int, int]:
        return (self.token_indices[0], self.token_indices[-1] + 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "resolved_answer_selection",
            "run_id": self.run_id,
            "answer_sha256": self.answer_sha256,
            "character_range": list(self.character_range),
            "selected_text": self.selected_text,
            "token_indices": list(self.token_indices),
            "token_ids": list(self.token_ids),
            "token_pieces": list(self.token_pieces),
            "token_spans": [{"start": start, "end": end} for start, end in self.token_spans],
            "token_range": list(self.token_range),
            "boundary": self.boundary,
        }


def _resolve_against_text(response: str, selection: AnswerSelection, *, run_id: str | None,
                          answer_sha256: str | None, token_ids: list[int], token_pieces: list[str]) -> ResolvedAnswerSelection:
    actual_answer_sha = _sha256_text(response)
    if selection.answer_sha256 is not None and selection.answer_sha256 != actual_answer_sha:
        raise AnswerSelectionUnavailable("selection is stale for the recorded answer")
    if answer_sha256 is not None and answer_sha256 != actual_answer_sha:
        raise AnswerSelectionUnavailable("recorded answer identity does not match its text")
    if selection.character_range is not None:
        start, finish = selection.character_range
    else:
        assert selection.text is not None
        occurrences: list[int] = []
        cursor = 0
        while True:
            found = response.find(selection.text, cursor)
            if found < 0:
                break
            occurrences.append(found)
            cursor = found + 1
        if len(occurrences) != 1:
            reason = "selected answer text was not found" if not occurrences else "selected answer text is ambiguous"
            raise AnswerSelectionUnavailable(reason)
        start = occurrences[0]
        finish = start + len(selection.text)
    if start < 0 or finish <= start or finish > len(response):
        raise AnswerSelectionUnavailable("answer selection is outside the recorded answer")
    selected_text = response[start:finish]
    if selection.character_range is not None and selection.selected_text is not None and selected_text != selection.selected_text:
        raise AnswerSelectionUnavailable("selected answer text does not match the recorded answer")
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in token_pieces:
        next_cursor = cursor + len(piece)
        spans.append((cursor, next_cursor))
        cursor = next_cursor
    indices = [index for index, (token_start, token_end) in enumerate(spans)
               if token_start < finish and token_end > start]
    if not indices:
        raise AnswerSelectionUnavailable("answer selection does not intersect a recorded token")
    return ResolvedAnswerSelection(
        run_id=run_id or "recorded-answer",
        answer_sha256=actual_answer_sha,
        character_range=(start, finish),
        selected_text=selected_text,
        token_indices=indices,
        token_ids=[token_ids[index] for index in indices],
        token_pieces=[token_pieces[index] for index in indices],
        token_spans=[spans[index] for index in indices],
    )


def resolve_answer_selection(run: Mapping[str, Any], selection: AnswerSelection) -> ResolvedAnswerSelection:
    """Resolve a selection against one stored run without model or text approximation."""
    if not isinstance(selection, AnswerSelection):
        raise TypeError("resolve_answer_selection requires an AnswerSelection")
    token_ids, token_pieces, response = _recorded_answer_tokens(run)
    return _resolve_against_text(
        response, selection, run_id=run.get("id") if isinstance(run, Mapping) else None,
        answer_sha256=None, token_ids=token_ids, token_pieces=token_pieces,
    )


def resolve_answer_selection_from_observation(
    observation: Any, selection: AnswerSelection, *, run_id: str | None = None,
    answer_sha256: str | None = None,
) -> ResolvedAnswerSelection:
    """Resolve against persisted score-observation token pieces for read projection."""
    token_ids = list(getattr(observation, "recorded_token_ids", ()) or ())
    token_pieces = list(getattr(observation, "token_pieces", ()) or ())
    if not token_ids or len(token_ids) != len(token_pieces):
        raise AnswerSelectionUnavailable("score observation has no exact recorded token trace")
    response = "".join(token_pieces)
    return _resolve_against_text(
        response, selection, run_id=run_id, answer_sha256=answer_sha256,
        token_ids=token_ids, token_pieces=token_pieces,
    )


__all__ = [
    "AnswerSelection", "AnswerSelectionUnavailable", "ContextSelection",
    "ResolvedAnswerSelection", "SelectionError", "resolve_answer_selection",
    "resolve_answer_selection_from_observation",
]
