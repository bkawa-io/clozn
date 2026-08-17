"""Thin user-facing compositions over the experimental kernel."""

from .removability import can_remove, removability_message
from .context_effects import context_effect_message, measure_context_effects, project_context_effects
from .minimal_context import MinimalContextResult, WinningCandidate, run_minimal_context

__all__ = [
    "can_remove", "context_effect_message", "measure_context_effects",
    "project_context_effects", "removability_message",
    "MinimalContextResult", "WinningCandidate", "run_minimal_context",
]
