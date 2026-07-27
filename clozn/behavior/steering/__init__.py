"""Steering axes, catalogs, and substrate adapters."""

from . import axes as _axes

__all__ = [
    "AXES",
    "ConceptSteer",
    "DEV",
    "EngineSteer",
    "SEED_PROMPTS",
    "SteeringControl",
    "_DIAL_DEFAULT_MAG",
    "_DIAL_LEXICON",
    "_DIAL_REDUCERS",
    "suggest_dial_for_preference",
]

_AXIS_EXPORTS = {"AXES", "SEED_PROMPTS", "_DIAL_DEFAULT_MAG", "_DIAL_LEXICON", "_DIAL_REDUCERS"}


def __getattr__(name: str):
    if name in _AXIS_EXPORTS:
        return getattr(_axes, name)
    if name == "suggest_dial_for_preference":
        from .catalog import suggest_dial_for_preference
        return suggest_dial_for_preference
    if name in {"DEV", "SteeringControl"}:
        from .hf_adapter import DEV, SteeringControl
        return {"DEV": DEV, "SteeringControl": SteeringControl}[name]
    if name == "EngineSteer":
        from .engine_adapter import EngineSteer
        return EngineSteer
    if name == "ConceptSteer":
        from .concept_dir import ConceptSteer
        return ConceptSteer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
