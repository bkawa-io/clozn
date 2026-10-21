"""Score-based context search dispatch over recorded-continuation evidence."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import math
from typing import Any

from .context_search import ContextSearchDispatcher
from .evaluators import ScoreRecordedContinuation
from .interventions import DeleteSource
from .observations import TokenScoreObservation
from .persistence import ObservationStore
from .runner import ExperimentResult
from .scoring import DeleteSourceRecordedContinuationScoreAdapter


def continuation_support_loss(
    control: TokenScoreObservation,
    candidate: TokenScoreObservation,
) -> dict[str, Any] | None:
    """Summarize only losses in support for the exact recorded answer tokens.

    Improvements are deliberately clamped to zero. Lean Context asks whether
    deleting context makes any recorded-answer token meaningfully less likely;
    an improvement on another token does not compensate for that local loss.
    """
    if not (control.completed and candidate.completed and control.trusted and candidate.trusted):
        return None
    if (
        not control.recorded_token_ids
        or control.recorded_token_ids != candidate.recorded_token_ids
        or len(control.token_logprobs) != len(candidate.token_logprobs)
    ):
        return None
    losses = tuple(
        max(0.0, float(baseline) - float(altered))
        for baseline, altered in zip(control.token_logprobs, candidate.token_logprobs)
    )
    if any(not math.isfinite(value) for value in losses):
        return None
    return {
        "answer_token_count": len(losses),
        "positive_loss_token_count": sum(value > 0 for value in losses),
        "mean_positive_support_loss_nats_per_token": sum(losses) / len(losses),
        "max_token_support_loss_nats": max(losses, default=0.0),
    }


class ScoreContextSearchDispatcher(ContextSearchDispatcher):
    """Translate retained-source candidates into teacher-forced score arms."""

    def __init__(
        self,
        run: Mapping[str, Any],
        universe: Mapping[str, Any],
        *,
        mean_loss_tolerance_nats: float,
        max_token_loss_tolerance_nats: float,
        substrate: Any = None,
        engine: Any = None,
        observation_store: ObservationStore | None = None,
        execution_adapter: Any = None,
        prompt_token_counter: Callable[[Sequence[Mapping[str, Any]]], int] | None = None,
        render_messages: Callable[[tuple[str, ...]], Sequence[Mapping[str, Any]]] | None = None,
        cancel: Callable[[], bool] | None = None,
    ):
        for name, value in (
            ("mean_loss_tolerance_nats", mean_loss_tolerance_nats),
            ("max_token_loss_tolerance_nats", max_token_loss_tolerance_nats),
        ):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative number")
        self.mean_loss_tolerance_nats = float(mean_loss_tolerance_nats)
        self.max_token_loss_tolerance_nats = float(max_token_loss_tolerance_nats)
        self._evidence_by_observation_id: dict[str, dict[str, Any]] = {}
        adapter = execution_adapter or DeleteSourceRecordedContinuationScoreAdapter(
            substrate, run=run,
        )
        super().__init__(
            run,
            universe,
            substrate=substrate,
            engine=engine,
            observation_store=observation_store,
            execution_adapter=adapter,
            evaluator=ScoreRecordedContinuation(),
            prompt_token_counter=prompt_token_counter,
            render_messages=render_messages,
            cancel=cancel,
            _allow_non_exact_evaluator=True,
        )

    @property
    def tolerance(self) -> dict[str, Any]:
        return {
            "profile": "balanced.v1",
            "mean_positive_support_loss_nats_per_token": self.mean_loss_tolerance_nats,
            "max_token_support_loss_nats": self.max_token_loss_tolerance_nats,
        }

    def evidence_for_observation(self, observation_id: str | None) -> dict[str, Any] | None:
        if not isinstance(observation_id, str):
            return None
        value = self._evidence_by_observation_id.get(observation_id)
        return deepcopy(value) if value is not None else None

    def _one_ref(
        self,
        result: ExperimentResult,
        arm_id: str,
        intervention: DeleteSource | None,
        *,
        reused: bool = False,
    ) -> dict[str, Any]:
        evidence = super()._one_ref(result, arm_id, intervention, reused=reused)
        observation = result.control if arm_id == "control" else result.arm_for(arm_id).observation
        if not isinstance(observation, TokenScoreObservation) or not observation.completed:
            evidence["classification"] = "unknown"
            return evidence

        if arm_id == "control":
            metrics = {
                "answer_token_count": len(observation.recorded_token_ids),
                "positive_loss_token_count": 0,
                "mean_positive_support_loss_nats_per_token": 0.0,
                "max_token_support_loss_nats": 0.0,
            }
        elif isinstance(result.control, TokenScoreObservation):
            metrics = continuation_support_loss(result.control, observation)
        else:
            metrics = None
        if metrics is None:
            evidence["classification"] = "unknown"
            return evidence

        within_tolerance = (
            metrics["mean_positive_support_loss_nats_per_token"]
            <= self.mean_loss_tolerance_nats
            and metrics["max_token_support_loss_nats"]
            <= self.max_token_loss_tolerance_nats
        )
        evidence.update({
            "classification": "preserves" if within_tolerance else "diverged",
            "tolerance_status": "within_tolerance" if within_tolerance else "outside_tolerance",
            "loss": metrics,
        })
        observation_id = evidence.get("observation_id")
        if isinstance(observation_id, str):
            self._evidence_by_observation_id[observation_id] = deepcopy(evidence)
        return evidence


__all__ = ["ScoreContextSearchDispatcher", "continuation_support_loss"]
