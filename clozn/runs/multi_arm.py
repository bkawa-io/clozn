"""Deterministic multi-arm adapters for direct Minimal Context measurements.

The public helpers in this module are deliberately small.  They provide one
batch seam for the two independent evidence protocols used by Minimal
Context, while keeping the scalar substrate methods as the semantic source of
truth.  A substrate may implement a native ``*_many`` method; otherwise the
helpers dispatch compatible arms serially and preserve input ordering.

An arm is a mapping containing the keyword arguments for its scalar method.
``block`` is the only mutable context field excluded from compatibility
grouping.  Everything else is an immutable condition and is therefore never
combined with a different model, continuation, contract, or steering setup.
"""
from __future__ import annotations

from copy import deepcopy
import inspect
import json
from collections.abc import Callable, Iterable, Mapping
from typing import Any


class MultiArmError(ValueError):
    """A malformed arm or a failed batch dispatch.

    ``completed`` contains detached results from groups completed before the
    failure, in original input order.  Callers may safely retain those direct
    observations without treating the overall operation as complete.
    """

    def __init__(self, message: str, *, arm_index: int | None = None,
                 completed: Iterable[Any] = ()) -> None:
        super().__init__(message)
        self.arm_index = arm_index
        self.completed = list(completed)


class BatchCancelled(RuntimeError):
    """Cancellation stopped dispatch of queued arms or batches."""

    def __init__(self, message: str = "multi-arm dispatch cancelled", *,
                 completed: Iterable[Any] = (), next_index: int | None = None) -> None:
        super().__init__(message)
        self.completed = list(completed)
        self.next_index = next_index


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    method = getattr(cancel, "is_set", None)
    if callable(method):
        return bool(method())
    return bool(cancel)


def _jsonable(value: Any) -> Any:
    """Make an immutable compatibility key without exposing live mutable state."""
    try:
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _freeze(value: Any) -> str:
    return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), default=str)


def _validate_arm(kind: str, index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MultiArmError(f"{kind} arm {index} must be a mapping", arm_index=index)
    arm = deepcopy(dict(raw))
    if "messages" not in arm or not isinstance(arm["messages"], (list, tuple)):
        raise MultiArmError(f"{kind} arm {index} must carry a messages list", arm_index=index)
    if kind == "score_tokens":
        if "continuation_ids" in arm:
            ids = arm["continuation_ids"]
            if ids is not None and (not isinstance(ids, (list, tuple)) or any(
                    isinstance(item, bool) or not isinstance(item, int) for item in ids)):
                raise MultiArmError(f"score_tokens arm {index} has invalid continuation_ids", arm_index=index)
        if "continuation" in arm and arm["continuation"] is not None and not isinstance(arm["continuation"], str):
            raise MultiArmError(f"score_tokens arm {index} has invalid continuation", arm_index=index)
    else:
        ids = arm.get("reference_token_ids")
        if not isinstance(ids, (list, tuple)) or not ids or any(
                isinstance(item, bool) or not isinstance(item, int) for item in ids):
            raise MultiArmError(f"probe_reference_match arm {index} has invalid reference_token_ids", arm_index=index)
        if not isinstance(arm.get("generation_contract"), Mapping):
            raise MultiArmError(f"probe_reference_match arm {index} has invalid generation_contract", arm_index=index)
        if "explicit_conditions" in arm and arm["explicit_conditions"] is not None \
                and not isinstance(arm["explicit_conditions"], Mapping):
            raise MultiArmError(f"probe_reference_match arm {index} has invalid explicit_conditions", arm_index=index)
    return arm


def _compatibility_key(kind: str, arm: Mapping[str, Any]) -> str:
    immutable = dict(arm)
    immutable.pop("block", None)
    if kind == "probe_reference_match":
        conditions = dict(immutable.get("explicit_conditions") or {})
        conditions.pop("block", None)
        if "explicit_conditions" in immutable:
            immutable["explicit_conditions"] = conditions
    return _freeze(immutable)


def _groups(kind: str, arms: list[dict[str, Any]]) -> list[tuple[str, list[tuple[int, dict[str, Any]]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    order: list[str] = []
    for index, arm in enumerate(arms):
        key = _compatibility_key(kind, arm)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, arm))
    return [(key, groups[key]) for key in order]


def _invoke_batch(method: Callable[..., Any], arms: list[dict[str, Any]], cancel: Any) -> list[Any]:
    try:
        signature = inspect.signature(method)
        accepts_cancel = "cancel" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        )
    except (TypeError, ValueError):
        accepts_cancel = True
    raw = method(arms, cancel=cancel) if accepts_cancel else method(arms)
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), list):
        raw = raw["results"]
    if not isinstance(raw, (list, tuple)) or len(raw) != len(arms):
        raise ValueError("a multi-arm method must return one result per input arm")
    # A native scheduler may complete arms out of order.  It can make that
    # explicit with ``{"arm_index": i, "result": value}`` rows; restore the
    # caller's order before any evidence is exposed.
    if raw and all(isinstance(item, Mapping) and isinstance(item.get("arm_index"), int)
                   and "result" in item for item in raw):
        restored: list[Any] = [None] * len(raw)
        seen: set[int] = set()
        for item in raw:
            index = int(item["arm_index"])
            if index < 0 or index >= len(raw) or index in seen:
                raise ValueError("native multi-arm results contain invalid arm indexes")
            seen.add(index)
            restored[index] = item["result"]
        if len(seen) != len(raw):
            raise ValueError("native multi-arm results do not cover every arm")
        raw = restored
    return [deepcopy(item) for item in raw]


def _serial_results(method: Callable[..., Any], arms: list[dict[str, Any]], cancel: Any,
                    completed: list[Any], base_index: int, indexes: list[int] | None = None) -> list[Any]:
    results: list[Any] = []
    for offset, arm in enumerate(arms):
        index = indexes[offset] if indexes is not None else base_index + offset
        if _cancelled(cancel):
            raise BatchCancelled(completed=completed + results, next_index=index)
        try:
            result = method(**arm)
        except BatchCancelled:
            raise
        except Exception as exc:
            raise MultiArmError(f"multi-arm scalar dispatch failed for arm {index}: {exc}",
                                arm_index=index, completed=completed + results) from exc
        results.append(deepcopy(result))
    return results


def _many(kind: str, substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None) -> list[Any]:
    try:
        raw_arms = list(arms)
    except TypeError as exc:
        raise MultiArmError(f"{kind} arms must be iterable") from exc
    normalised: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_arms):
        normalised.append(_validate_arm(kind, index, raw))
    if _cancelled(cancel):
        raise BatchCancelled(completed=[], next_index=0)

    scalar = getattr(substrate, kind, None)
    if not callable(scalar):
        raise MultiArmError(f"substrate has no scalar {kind} method")
    native = getattr(substrate, f"{kind}_many", None)
    # The EngineSubstrate implementation is a serial fallback but is still a
    # valid batch adapter.  Native adapters can replace it without changing
    # the scheduling or proof layer.
    if not callable(native):
        native = None

    ordered: list[Any] = [None] * len(normalised)
    completed: list[Any] = []
    for _key, group in _groups(kind, normalised):
        indices = [index for index, _arm in group]
        group_arms = [arm for _index, arm in group]
        if _cancelled(cancel):
            raise BatchCancelled(completed=completed, next_index=indices[0])
        try:
            if native is not None:
                group_results = _invoke_batch(native, group_arms, cancel)
            else:
                group_results = _serial_results(scalar, group_arms, cancel, completed, indices[0], indices)
        except BatchCancelled as exc:
            partial = list(completed)
            # Native adapters may report detached completed rows; scalar
            # fallback already includes them in the exception.
            partial.extend(exc.completed)
            raise BatchCancelled(completed=partial, next_index=exc.next_index or indices[0]) from exc
        except MultiArmError as exc:
            raise MultiArmError(str(exc), arm_index=exc.arm_index, completed=completed + exc.completed) from exc
        except Exception as exc:
            raise MultiArmError(f"multi-arm batch dispatch failed for arm {indices[0]}: {exc}",
                                arm_index=indices[0], completed=completed) from exc
        for index, result in zip(indices, group_results):
            ordered[index] = result
        completed.extend(group_results)
    return ordered


def score_tokens_many(substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None) -> list[Any]:
    """Score compatible teacher-forced arms, preserving input order."""
    return _many("score_tokens", substrate, arms, cancel=cancel)


def probe_reference_match_many(substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None) -> list[Any]:
    """Probe compatible exact-output arms, preserving input order."""
    return _many("probe_reference_match", substrate, arms, cancel=cancel)


def serial_many(method: Callable[..., Any], arms: Iterable[Mapping[str, Any]], *, cancel: Any = None) -> list[Any]:
    """Small helper for a substrate's explicit serial fallback method."""
    normalised = [deepcopy(dict(arm)) for arm in arms]
    return _serial_results(method, normalised, cancel, [], 0)


__all__ = [
    "BatchCancelled",
    "MultiArmError",
    "probe_reference_match_many",
    "score_tokens_many",
    "serial_many",
]
