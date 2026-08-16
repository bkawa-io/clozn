"""Pure answer-span projections over persisted score observations."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .evaluators import ScoreRecordedContinuation
from .observations import TokenScoreDelta, TokenScoreObservation
from .persistence import ExperimentArmView, ExperimentView
from .runner import ExperimentResult
from .selections import (
    AnswerSelection,
    AnswerSelectionUnavailable,
    ResolvedAnswerSelection,
    resolve_answer_selection_from_observation,
)
from .state import canonical_json


class ProjectionError(ValueError):
    """A score result cannot be projected into an answer-span effect."""


class AnswerSpanEffect:
    """One signed, intervention-specific answer-span measurement."""

    __slots__ = (
        "experiment_id", "run_id", "arm_id", "intervention", "source_ids",
        "selection", "resolved_selection", "selected_token_range", "selected_token_indices", "selected_text",
        "status", "baseline_selected_logp", "intervened_selected_logp", "delta_nats",
        "provenance", "diagnostics", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("AnswerSpanEffect is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, experiment_id: str, run_id: str, arm_id: str,
                 intervention: Mapping[str, Any], source_ids: tuple[str, ...],
                 selection: AnswerSelection | ResolvedAnswerSelection | Mapping[str, Any],
                 resolved_selection: ResolvedAnswerSelection | None = None,
                 status: str, baseline_selected_logp: float | None,
                 intervened_selected_logp: float | None, delta_nats: float | None,
                 provenance: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None):
        if isinstance(selection, (AnswerSelection, ResolvedAnswerSelection)):
            original = selection
            resolved = resolved_selection if resolved_selection is not None else (
                selection if isinstance(selection, ResolvedAnswerSelection) else None
            )
        else:
            original = dict(selection)
            resolved = dict(selection)
        if resolved is None:
            raise ProjectionError("AnswerSpanEffect requires resolved answer selection evidence")
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.arm_id = arm_id
        self.intervention = dict(intervention)
        self.source_ids = tuple(source_ids)
        self.selection = original
        self.resolved_selection = resolved
        if isinstance(resolved, ResolvedAnswerSelection):
            self.selected_token_range = resolved.token_range
            self.selected_token_indices = resolved.token_indices
            self.selected_text = resolved.selected_text
        else:
            token_range = resolved.get("token_range") or resolved.get("selected_token_range") or []
            indices = resolved.get("token_indices") or resolved.get("selected_token_indices") or []
            self.selected_token_range = tuple(token_range)
            self.selected_token_indices = tuple(indices)
            self.selected_text = resolved.get("selected_text")
        if status not in {"completed", "unavailable", "failed"}:
            raise ProjectionError(f"unsupported answer-span effect status: {status!r}")
        self.status = status
        self.baseline_selected_logp = _number_or_none(baseline_selected_logp)
        self.intervened_selected_logp = _number_or_none(intervened_selected_logp)
        self.delta_nats = _number_or_none(delta_nats)
        self.provenance = dict(provenance)
        self.diagnostics = dict(diagnostics or {})
        self._sealed = True

    @property
    def signed_delta_nats(self) -> float | None:
        return self.delta_nats

    def to_dict(self) -> dict[str, Any]:
        selection = self.selection.to_dict() if isinstance(self.selection, (AnswerSelection, ResolvedAnswerSelection)) else dict(self.selection)
        resolved = self.resolved_selection.to_dict() if isinstance(self.resolved_selection, ResolvedAnswerSelection) else dict(self.resolved_selection)
        return {
            "kind": "answer_span_effect",
            "experiment_id": self.experiment_id,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "intervention": dict(self.intervention),
            "source_ids": list(self.source_ids),
            "selection": selection,
            "resolved_selection": resolved,
            "selected_token_range": list(self.selected_token_range),
            "selected_token_indices": list(self.selected_token_indices),
            "selected_text": self.selected_text,
            "status": self.status,
            "baseline_selected_logp": self.baseline_selected_logp,
            "intervened_selected_logp": self.intervened_selected_logp,
            "delta_nats": self.delta_nats,
            "provenance": dict(self.provenance),
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def __repr__(self) -> str:
        return f"AnswerSpanEffect(arm_id={self.arm_id!r}, delta_nats={self.delta_nats!r})"


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ProjectionError("effect values must be finite numbers or None")
    return float(value)


def _selected_sum(values: tuple[float, ...], indices: tuple[int, ...]) -> float | None:
    if not values or any(index < 0 or index >= len(values) for index in indices):
        return None
    return sum(values[index] for index in indices)


def _resolved_selection(result: ExperimentResult, selection: AnswerSelection | ResolvedAnswerSelection) -> ResolvedAnswerSelection:
    if isinstance(selection, ResolvedAnswerSelection):
        resolved = selection
    else:
        baseline = result.control
        if not isinstance(baseline, TokenScoreObservation) or not baseline.completed:
            raise AnswerSelectionUnavailable("baseline score observation is unavailable")
        expected_ids = result.base.recorded_answer_token_identity.get("token_ids_sha256")
        from .state import digest
        if expected_ids != digest(list(baseline.recorded_token_ids)):
            raise AnswerSelectionUnavailable("baseline score is not bound to the recorded answer")
        expected_response = result.base.recorded_answer_token_identity.get("response_sha256")
        if expected_response != digest("".join(baseline.token_pieces)):
            raise AnswerSelectionUnavailable("baseline score text is not bound to the recorded answer")
        resolved = resolve_answer_selection_from_observation(
            baseline, selection, run_id=result.base.run_id,
        )
    if resolved.run_id not in {result.base.run_id, "recorded-answer"}:
        raise AnswerSelectionUnavailable("selection belongs to a different run")
    return resolved


def _effect_for_arm(result: ExperimentResult | ExperimentView, selection: AnswerSelection | ResolvedAnswerSelection,
                    resolved: ResolvedAnswerSelection,
                    arm: ExperimentArmView) -> AnswerSpanEffect:
    baseline = result.control if isinstance(result.control, TokenScoreObservation) else None
    observation = arm.observation if isinstance(arm.observation, TokenScoreObservation) else None
    delta = TokenScoreDelta.from_observations(baseline, observation) if baseline is not None and observation is not None else None
    indices = resolved.token_indices
    baseline_sum = _selected_sum(baseline.token_logprobs, indices) if baseline and baseline.completed else None
    intervened_sum = _selected_sum(observation.token_logprobs, indices) if observation is not None and observation.completed else None
    selected_delta = _selected_sum(delta.deltas, indices) if delta is not None and delta.status == "completed" else None
    status = delta.status if delta is not None else "unavailable"
    diagnostics = {} if status == "completed" else dict(
        (delta.diagnostics if delta is not None else {}) or {}
    )
    if observation is None:
        diagnostics.setdefault("reason", "intervention_observation_unavailable")
    if baseline is None:
        diagnostics.setdefault("baseline_status", "unavailable")
    return AnswerSpanEffect(
        experiment_id=result.experiment_id, run_id=result.base.run_id, arm_id=arm.arm_id,
        intervention=arm.intervention.to_dict(), source_ids=tuple(arm.intervention.source_ids),
        selection=selection, resolved_selection=resolved, status=status, baseline_selected_logp=baseline_sum,
        intervened_selected_logp=intervened_sum, delta_nats=selected_delta,
        provenance={
            "run_id": result.base.run_id,
            "experiment_id": result.experiment_id,
            "arm_id": arm.arm_id,
            "source_ids": list(arm.intervention.source_ids),
            "intervention": arm.intervention.to_dict(),
            "evaluator": result.evaluator.to_dict(),
            "selected_answer": {
                **resolved.to_dict(),
                "selection": selection.to_dict() if isinstance(selection, (AnswerSelection, ResolvedAnswerSelection)) else dict(selection),
            },
            "measurement": {
                "basis": "persisted_full_continuation_token_vectors",
                "sign": "baseline_minus_intervention",
                "baseline_observation_id": baseline.observation_id if baseline else None,
                "intervention_observation_id": observation.observation_id if observation else None,
                "selected_baseline_logp": baseline_sum,
                "selected_intervened_logp": intervened_sum,
                "selected_delta_nats": selected_delta,
            },
        },
        diagnostics=diagnostics,
    )


def project_answer_effects(result: ExperimentResult | ExperimentView, selection: AnswerSelection | ResolvedAnswerSelection,
                           *, ordering: str = "absolute") -> list[AnswerSpanEffect]:
    """Project every direct deletion arm without making model or substrate calls."""
    if not isinstance(result, (ExperimentResult, ExperimentView)):
        raise TypeError("project_answer_effects requires an experiment read model")
    if not isinstance(result.evaluator, ScoreRecordedContinuation):
        raise ProjectionError("answer-span effects require ScoreRecordedContinuation evidence")
    if ordering not in {"absolute", "source"}:
        raise ProjectionError("ordering must be 'absolute' or 'source'")
    resolved = _resolved_selection(result, selection)
    effects: list[AnswerSpanEffect] = []
    for arm in result.arms:
        if not isinstance(arm, ExperimentArmView) or arm.intervention is None:
            raise ProjectionError("score result contains an invalid arm association")
        effects.append(_effect_for_arm(result, selection, resolved, arm))
    if ordering == "absolute":
        indexed = list(enumerate(effects))
        indexed.sort(key=lambda pair: (
            0 if pair[1].delta_nats is not None else 1,
            -abs(pair[1].delta_nats) if pair[1].delta_nats is not None else 0.0,
            pair[0],
        ))
        return [effect for _index, effect in indexed]
    return effects


def project_answer_selection(result: ExperimentResult | ExperimentView, selection: AnswerSelection | ResolvedAnswerSelection,
                             *, ordering: str = "absolute") -> list[AnswerSpanEffect]:
    return project_answer_effects(result, selection, ordering=ordering)


__all__ = [
    "AnswerSpanEffect", "ProjectionError", "project_answer_effects", "project_answer_selection",
]
