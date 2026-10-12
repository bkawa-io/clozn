"""Persist outcome-grounded calibration profiles in one authoritative registry.

The store is dependency-light and imports no server code. Every reader selects a
profile from the same model/task-indexed document; there is no separate active
report or fallback representation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
import os
import time
import unicodedata

_PATH = os.path.join(os.path.expanduser("~/.clozn"), "calibration_profiles.json")
_PROFILE_SCHEMA = "clozn.calibration_profile.v1"
_REGISTRY_SCHEMA = "clozn.calibration_profile_registry.v1"


def _registry_path(path: str | None = None) -> str:
    return os.fspath(path or _PATH)


def _normal_task(task) -> str:
    if isinstance(task, bool) or not isinstance(task, str):
        raise ValueError("calibration profile task must be a non-empty string")
    if any(unicodedata.category(char) == "Cc" for char in task):
        raise ValueError("calibration profile task must not contain control characters")
    normalized = " ".join(task.strip().lower().split())
    if not normalized:
        raise ValueError("calibration profile task must be a non-empty string")
    if len(normalized) > 80:
        raise ValueError("calibration profile task must be at most 80 characters")
    return normalized


def _model(payload: dict) -> str:
    model = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model, bool) or not isinstance(model, str) or not model.strip():
        raise ValueError("calibration profile payload must carry a non-empty model")
    return model.strip()


def _saved_ts(payload: dict) -> float:
    stamp = payload.get("saved_ts")
    if stamp is None:
        return time.time()
    if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(float(stamp)):
        raise ValueError("calibration profile saved_ts must be a finite number when provided")
    return float(stamp)


def _profile(payload: dict, task: str) -> dict:
    if not isinstance(payload, dict):
        raise ValueError("calibration profile payload must be a dictionary")
    model = _model(payload)
    normalized_task = _normal_task(task)
    source_provenance = payload.get("provenance")
    source_schema = payload.get("schema")
    out = deepcopy(payload)
    out.update({
        "schema": _PROFILE_SCHEMA,
        "model": model,
        "task": normalized_task,
        "saved_ts": _saved_ts(payload),
        "provenance": {
            "kind": "outcome_grounded_calibration",
            "model_match": "exact",
            "task_match": "exact_normalized_task",
            "score_aggregate": payload.get("score"),
            "probe_set": payload.get("set"),
            "claim_limit": (
                "thresholds are fitted to labeled outcomes for this exact model and task; "
                "they are not a per-answer correctness probability or guarantee"
            ),
        },
    })
    if isinstance(source_provenance, dict) and source_provenance:
        out["provenance"]["source"] = deepcopy(source_provenance)
    if source_schema is not None and source_schema != _PROFILE_SCHEMA:
        out["provenance"]["source_schema"] = source_schema
    try:
        json.dumps(out)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"calibration profile must be JSON serializable: {exc}") from exc
    return out


def _valid_profile(value) -> bool:
    if not isinstance(value, dict) or value.get("schema") != _PROFILE_SCHEMA:
        return False
    try:
        model = _model(value)
        task = _normal_task(value.get("task"))
    except ValueError:
        return False
    stamp = value.get("saved_ts")
    if (isinstance(stamp, bool) or not isinstance(stamp, (int, float))
            or not math.isfinite(float(stamp))):
        return False
    return (value.get("model") == model and value.get("task") == task
            and isinstance(value.get("provenance"), dict))


def _read_registry(path: str | None = None) -> list[dict]:
    """Read only valid v1 entries. Missing/corrupt registries are an empty registry."""
    try:
        with open(_registry_path(path), encoding="utf-8") as handle:
            registry = json.load(handle)
        if not isinstance(registry, dict) or registry.get("schema") != _REGISTRY_SCHEMA:
            return []
        profiles = registry.get("profiles")
        if not isinstance(profiles, list):
            return []
        return [deepcopy(profile) for profile in profiles if _valid_profile(profile)]
    except (OSError, TypeError, ValueError):
        return []


def _write_registry(profiles: list[dict], path: str | None = None) -> str:
    registry_path = _registry_path(path)
    os.makedirs(os.path.dirname(registry_path) or ".", exist_ok=True)
    tmp = registry_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump({"schema": _REGISTRY_SCHEMA, "profiles": profiles}, handle)
    os.replace(tmp, registry_path)
    return registry_path


def save_profile(payload: dict, task: str, path: str | None = None) -> dict:
    """Validate and persist one exact model/normalized-task calibration profile.

    Only the prior entry for the same exact ``(model, task)`` pair is replaced.
    Returns the stored profile.
    """
    profile = _profile(payload, task)
    profiles = _read_registry(path)
    profiles = [
        existing for existing in profiles
        if not (existing["model"] == profile["model"] and existing["task"] == profile["task"])
    ]
    profiles.append(profile)
    # Stable registry order makes diffs and recovery inspection predictable.
    profiles.sort(key=lambda item: (item["model"], item["task"], item["saved_ts"]))
    _write_registry(profiles, path)
    return deepcopy(profile)


def list_profiles(path: str | None = None) -> list[dict]:
    """All valid profiles, newest first. Missing/corrupt registries return ``[]``."""
    profiles = _read_registry(path)
    profiles.sort(key=lambda item: (-float(item["saved_ts"]), item["model"], item["task"]))
    return profiles


def load_latest_profile(path: str | None = None) -> dict | None:
    """Newest valid profile across all models and tasks, or None."""
    profiles = list_profiles(path)
    return deepcopy(profiles[0]) if profiles else None


def load_profile(model: str, task: str | None = None, path: str | None = None) -> dict | None:
    """Load an exact model/task profile, or that model's newest profile when task is omitted.

    Explicit tasks are normalized by the same rule as ``save_profile`` and never
    fall back to another task. Invalid inputs and corrupt registries return None.
    """
    if isinstance(model, bool) or not isinstance(model, str) or not model.strip():
        return None
    exact_model = model.strip()
    try:
        exact_task = _normal_task(task) if task is not None else None
    except ValueError:
        return None
    matches = [profile for profile in _read_registry(path) if profile["model"] == exact_model]
    if exact_task is not None:
        matches = [profile for profile in matches if profile["task"] == exact_task]
    if not matches:
        return None
    newest = max(
        enumerate(matches),
        key=lambda indexed: (float(indexed[1]["saved_ts"]), indexed[0]),
    )[1]
    return deepcopy(newest)
