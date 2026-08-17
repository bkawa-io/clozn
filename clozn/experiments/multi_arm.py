"""Generic substrate multi-arm dispatch.

This layer only schedules already-prepared semantic requests.  It does not
know recipes, source IDs, or how evidence is classified.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from copy import deepcopy
import inspect
import json
from typing import Any


class MultiArmError(RuntimeError):
    def __init__(self, message: str, *, arm_index: int | None = None,
                 completed: Iterable[Any] = ()):
        super().__init__(message)
        self.arm_index = arm_index
        self.completed = list(completed)


class BatchCancelled(RuntimeError):
    def __init__(self, message: str = "multi-arm dispatch cancelled", *,
                 completed: Iterable[Any] = (), next_index: int | None = None):
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


def _freeze(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


_PER_ARM_FIELDS = frozenset({
    "messages", "prompt", "removed_source_ids", "retained_source_ids",
    "context_digest", "intervened_context_digest",
})


def _validate(kind: str, index: int, raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise MultiArmError(f"{kind} arm {index} must be a mapping", arm_index=index)
    arm = deepcopy(dict(raw))
    if not isinstance(arm.get("messages"), (list, tuple)):
        raise MultiArmError(f"{kind} arm {index} must carry a messages list", arm_index=index)
    if kind == "probe_reference_match":
        ids = arm.get("reference_token_ids")
        if not isinstance(ids, (list, tuple)) or not ids or any(
                isinstance(item, bool) or not isinstance(item, int) for item in ids):
            raise MultiArmError(f"{kind} arm {index} has invalid reference_token_ids", arm_index=index)
        if not isinstance(arm.get("generation_contract"), Mapping):
            raise MultiArmError(f"{kind} arm {index} has invalid generation_contract", arm_index=index)
    else:
        ids = arm.get("continuation_ids")
        if ids is not None and (not isinstance(ids, (list, tuple)) or any(
                isinstance(item, bool) or not isinstance(item, int) for item in ids)):
            raise MultiArmError(f"{kind} arm {index} has invalid continuation_ids", arm_index=index)
    return arm


def _groups(kind: str, arms: list[dict[str, Any]]) -> list[list[tuple[int, dict[str, Any]]]]:
    groups: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    order: list[str] = []
    for index, arm in enumerate(arms):
        immutable = {key: value for key, value in arm.items() if key not in _PER_ARM_FIELDS}
        key = _freeze(immutable)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((index, arm))
    return [groups[key] for key in order]


def _invoke_native(method: Callable[..., Any], arms: list[dict[str, Any]], *, cancel: Any) -> list[Any]:
    try:
        signature = inspect.signature(method)
        accepts_cancel = "cancel" in signature.parameters or any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values())
    except (TypeError, ValueError):
        accepts_cancel = True
    raw = method(arms, **({"cancel": cancel} if accepts_cancel else {}))
    if isinstance(raw, Mapping) and isinstance(raw.get("results"), list):
        raw = raw["results"]
    if not isinstance(raw, (list, tuple)) or len(raw) != len(arms):
        raise MultiArmError("a multi-arm method must return one result per input arm")
    if raw and all(isinstance(row, Mapping) and isinstance(row.get("arm_index"), int)
                   and "result" in row for row in raw):
        restored = [None] * len(raw)
        seen: set[int] = set()
        for row in raw:
            index = row["arm_index"]
            if index < 0 or index >= len(raw) or index in seen:
                raise MultiArmError("native multi-arm results contain invalid arm indexes")
            seen.add(index)
            restored[index] = row["result"]
        raw = restored
    return [deepcopy(row) for row in raw]


def _scalar(method: Callable[..., Any], arms: list[dict[str, Any]], *, cancel: Any,
            base_index: int = 0) -> list[Any]:
    result: list[Any] = []
    for offset, arm in enumerate(arms):
        if _cancelled(cancel):
            raise BatchCancelled(completed=result, next_index=base_index + offset)
        try:
            result.append(deepcopy(method(**arm)))
        except Exception as exc:
            raise MultiArmError(str(exc), arm_index=base_index + offset, completed=result) from exc
    return result


def _many(kind: str, substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
          proof_grade: bool = True) -> list[Any]:
    normalised = [_validate(kind, index, raw) for index, raw in enumerate(arms)]
    if _cancelled(cancel):
        raise BatchCancelled(completed=[], next_index=0)
    scalar = getattr(substrate, kind, None)
    if not callable(scalar):
        raise MultiArmError(f"substrate has no scalar {kind} method")
    native = getattr(substrate, f"{kind}_many", None)
    native_ok = bool(proof_grade and getattr(substrate, f"{kind}_many_proof_grade", False))
    if not native_ok:
        return _scalar(scalar, normalised, cancel=cancel)
    ordered: list[Any] = [None] * len(normalised)
    completed: list[Any] = []
    for group in _groups(kind, normalised):
        indexes = [index for index, _arm in group]
        group_arms = [arm for _index, arm in group]
        if _cancelled(cancel):
            raise BatchCancelled(completed=completed, next_index=indexes[0])
        try:
            rows = _invoke_native(native, group_arms, cancel=cancel)
        except BatchCancelled as exc:
            raise BatchCancelled(completed=completed + exc.completed, next_index=exc.next_index) from exc
        except MultiArmError:
            raise
        except Exception as exc:
            raise MultiArmError(str(exc), arm_index=indexes[0], completed=completed) from exc
        for index, row in zip(indexes, rows):
            ordered[index] = row
        completed.extend(rows)
    return ordered


def score_tokens_many(substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
                      proof_grade: bool = True) -> list[Any]:
    return _many("score_tokens", substrate, arms, cancel=cancel, proof_grade=proof_grade)


def probe_reference_match_many(substrate: Any, arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
                               proof_grade: bool = True) -> list[Any]:
    return _many("probe_reference_match", substrate, arms, cancel=cancel, proof_grade=proof_grade)


def serial_many(method: Callable[..., Any], arms: Iterable[Mapping[str, Any]], *, cancel: Any = None) -> list[Any]:
    return _scalar(method, [deepcopy(dict(arm)) for arm in arms], cancel=cancel)


def concurrent_many(method: Callable[..., Any], arms: Iterable[Mapping[str, Any]], *, cancel: Any = None,
                    max_workers: int = 2) -> list[Any]:
    normalised = [deepcopy(dict(arm)) for arm in arms]
    if max_workers <= 1:
        return _scalar(method, normalised, cancel=cancel)
    if _cancelled(cancel):
        raise BatchCancelled(completed=[], next_index=0)
    executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)), thread_name_prefix="clozn-batch")
    futures = {executor.submit(method, **arm): index for index, arm in enumerate(normalised)}
    results: list[Any] = [None] * len(normalised)
    try:
        while futures:
            done, _pending = wait(tuple(futures), return_when=FIRST_COMPLETED)
            for future in done:
                index = futures.pop(future)
                try:
                    results[index] = deepcopy(future.result())
                except Exception as exc:
                    completed = [row for row in results if row is not None]
                    raise MultiArmError(str(exc), arm_index=index, completed=completed) from exc
            if _cancelled(cancel):
                for future in futures:
                    future.cancel()
                raise BatchCancelled(completed=[row for row in results if row is not None], next_index=min(futures.values(), default=None))
        return results
    finally:
        executor.shutdown(wait=not _cancelled(cancel), cancel_futures=True)


__all__ = [
    "BatchCancelled", "MultiArmError", "concurrent_many", "probe_reference_match_many",
    "score_tokens_many", "serial_many",
]
