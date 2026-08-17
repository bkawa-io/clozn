"""Thin user-facing compositions over the experimental kernel."""

from .removability import can_remove, removability_message
from .context_effects import context_effect_message, measure_context_effects, project_context_effects
from .minimal_context import MinimalContextResult, WinningCandidate, run_minimal_context
from .time_travel import (
    TIME_TRAVEL_RESULT_SCHEMA_VERSION, TimeTravelError, TimeTravelResult,
    continue_from_here, enumerate_answer_boundaries, force_token_and_continue, list_answer_token_boundaries,
    materialize_time_travel,
    resolve_time_travel, run_time_travel, time_travel, time_travel_capabilities,
)

__all__ = [
    "can_remove", "context_effect_message", "measure_context_effects",
    "project_context_effects", "removability_message",
    "MinimalContextResult", "WinningCandidate", "run_minimal_context",
    "TIME_TRAVEL_RESULT_SCHEMA_VERSION", "TimeTravelError", "TimeTravelResult",
    "continue_from_here", "force_token_and_continue", "materialize_time_travel",
    "resolve_time_travel", "run_time_travel", "time_travel_capabilities",
    "time_travel", "enumerate_answer_boundaries", "list_answer_token_boundaries",
]
