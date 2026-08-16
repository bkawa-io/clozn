"""Explicit observation vocabulary for direct preservation experiments."""
from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from .state import canonical_json


SCHEMA_VERSION = "clozn.experiment-observation.v1"
OBSERVATION_STATUSES = frozenset({"exact_preserved", "diverged", "unavailable", "failed"})


class ObservationError(ValueError):
    """A malformed experiment observation."""


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


class Observation:
    """One typed result for the control or one ephemeral experiment arm."""

    __slots__ = (
        "arm_id", "status", "matched_token_count", "first_divergence_index",
        "divergence_kind", "execution_provenance", "proof_grade", "trusted",
        "diagnostics", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("Observation is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, arm_id: str, status: str,
                 matched_token_count: int | None = None,
                 first_divergence_index: int | None = None,
                 divergence_kind: str | None = None,
                 execution_provenance: Mapping[str, Any] | None = None,
                 proof_grade: str = "unavailable",
                 trusted: bool = False,
                 diagnostics: Mapping[str, Any] | None = None):
        if not isinstance(arm_id, str) or not arm_id:
            raise ObservationError("Observation.arm_id must be a non-empty string")
        if status not in OBSERVATION_STATUSES:
            raise ObservationError(f"unsupported observation status: {status!r}")
        if matched_token_count is not None and (
            isinstance(matched_token_count, bool) or not isinstance(matched_token_count, int)
            or matched_token_count < 0
        ):
            raise ObservationError("matched_token_count must be a non-negative integer or None")
        if first_divergence_index is not None and (
            isinstance(first_divergence_index, bool) or not isinstance(first_divergence_index, int)
            or first_divergence_index < 0
        ):
            raise ObservationError("first_divergence_index must be a non-negative integer or None")
        if not isinstance(proof_grade, str) or not proof_grade:
            raise ObservationError("proof_grade must be a non-empty string")
        if not isinstance(trusted, bool):
            raise ObservationError("trusted must be a boolean")
        self.arm_id = arm_id
        self.status = status
        self.matched_token_count = matched_token_count
        self.first_divergence_index = first_divergence_index
        self.divergence_kind = divergence_kind
        self.execution_provenance = _copy_mapping(execution_provenance)
        self.proof_grade = proof_grade
        self.trusted = trusted
        self.diagnostics = _copy_mapping(diagnostics)
        self._sealed = True

    @property
    def preservation_status(self) -> str:
        """Readable alias; this is preservation evidence, not an influence score."""
        return self.status

    @property
    def completed(self) -> bool:
        return self.status in {"exact_preserved", "diverged"}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "status": self.status,
            "matched_token_count": self.matched_token_count,
            "first_divergence_index": self.first_divergence_index,
            "divergence_kind": self.divergence_kind,
            "execution_provenance": dict(self.execution_provenance),
            "proof_grade": self.proof_grade,
            "trusted": self.trusted,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def rebind_arm_id(self, arm_id: str) -> "Observation":
        """Copy evidence for a duplicate arm while preserving its measured values."""
        value = self.to_dict()
        value["arm_id"] = arm_id
        return type(self).from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Observation":
        if not isinstance(value, Mapping) or value.get("schema_version") != SCHEMA_VERSION:
            raise ObservationError(f"Observation must declare {SCHEMA_VERSION}")
        return cls(
            arm_id=value.get("arm_id"),
            status=value.get("status"),
            matched_token_count=value.get("matched_token_count"),
            first_divergence_index=value.get("first_divergence_index"),
            divergence_kind=value.get("divergence_kind"),
            execution_provenance=value.get("execution_provenance"),
            proof_grade=value.get("proof_grade", "unavailable"),
            trusted=value.get("trusted", False),
            diagnostics=value.get("diagnostics"),
        )


TOKEN_SCORE_SCHEMA_VERSION = "clozn.experiment-token-score-observation.v1"
TOKEN_SCORE_STATUSES = frozenset({"completed", "unavailable", "failed"})


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ObservationError(f"{name} must contain finite numbers")
    return float(value)


def _mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


class TokenScoreObservation:
    """One full teacher-forced score vector for the recorded continuation."""

    __slots__ = (
        "arm_id", "status", "recorded_token_ids", "token_pieces", "token_spans",
        "token_logprobs", "total_continuation_logprob", "evaluator_provenance",
        "score_basis", "execution_provenance", "proof_grade", "trusted", "diagnostics",
        "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("TokenScoreObservation is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, arm_id: str, status: str,
                 recorded_token_ids: list[int] | tuple[int, ...] = (),
                 token_pieces: list[str] | tuple[str, ...] = (),
                 token_spans: list[Any] | tuple[Any, ...] = (),
                 token_logprobs: list[float] | tuple[float, ...] = (),
                 total_continuation_logprob: float | None = None,
                 evaluator_provenance: Mapping[str, Any] | None = None,
                 score_basis: Mapping[str, Any] | None = None,
                 execution_provenance: Mapping[str, Any] | None = None,
                 proof_grade: str = "unavailable", trusted: bool = False,
                 diagnostics: Mapping[str, Any] | None = None):
        if not isinstance(arm_id, str) or not arm_id:
            raise ObservationError("TokenScoreObservation.arm_id must be a non-empty string")
        if status not in TOKEN_SCORE_STATUSES:
            raise ObservationError(f"unsupported token-score observation status: {status!r}")
        ids = tuple(recorded_token_ids)
        pieces = tuple(token_pieces)
        spans = tuple(_span_pair(item) for item in token_spans)
        logprobs = tuple(_finite_float(item, name="token_logprobs") for item in token_logprobs)
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in ids):
            raise ObservationError("recorded_token_ids must contain non-negative integers")
        if any(not isinstance(item, str) for item in pieces):
            raise ObservationError("token_pieces must contain strings")
        if not (len(ids) == len(pieces) == len(spans) == len(logprobs)):
            if status == "completed":
                raise ObservationError("completed token-score fields must have equal lengths")
            ids = pieces = spans = logprobs = ()
        if status == "completed" and total_continuation_logprob is None:
            raise ObservationError("completed token-score observations require a total logprob")
        total = None if total_continuation_logprob is None else _finite_float(
            total_continuation_logprob, name="total_continuation_logprob")
        if not isinstance(proof_grade, str) or not proof_grade:
            raise ObservationError("proof_grade must be a non-empty string")
        if not isinstance(trusted, bool):
            raise ObservationError("trusted must be a boolean")
        self.arm_id = arm_id
        self.status = status
        self.recorded_token_ids = ids
        self.token_pieces = pieces
        self.token_spans = spans
        self.token_logprobs = logprobs
        self.total_continuation_logprob = total
        self.evaluator_provenance = _mapping(evaluator_provenance)
        self.score_basis = _mapping(score_basis)
        self.execution_provenance = _mapping(execution_provenance)
        self.proof_grade = proof_grade
        self.trusted = trusted
        self.diagnostics = _mapping(diagnostics)
        self._sealed = True

    @property
    def completed(self) -> bool:
        return self.status == "completed"

    @property
    def runtime_binding(self) -> Any:
        return self.score_basis.get("runtime_binding")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TOKEN_SCORE_SCHEMA_VERSION,
            "arm_id": self.arm_id,
            "status": self.status,
            "recorded_token_ids": list(self.recorded_token_ids),
            "token_pieces": list(self.token_pieces),
            "token_spans": [{"start": start, "end": end} for start, end in self.token_spans],
            "token_logprobs": list(self.token_logprobs),
            "total_continuation_logprob": self.total_continuation_logprob,
            "evaluator_provenance": dict(self.evaluator_provenance),
            "score_basis": dict(self.score_basis),
            "execution_provenance": dict(self.execution_provenance),
            "proof_grade": self.proof_grade,
            "trusted": self.trusted,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def rebind_arm_id(self, arm_id: str) -> "TokenScoreObservation":
        value = self.to_dict()
        value["arm_id"] = arm_id
        return type(self).from_dict(value)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenScoreObservation":
        if not isinstance(value, Mapping) or value.get("schema_version") != TOKEN_SCORE_SCHEMA_VERSION:
            raise ObservationError(f"TokenScoreObservation must declare {TOKEN_SCORE_SCHEMA_VERSION}")
        return cls(
            arm_id=value.get("arm_id"), status=value.get("status"),
            recorded_token_ids=value.get("recorded_token_ids") or (),
            token_pieces=value.get("token_pieces") or (), token_spans=value.get("token_spans") or (),
            token_logprobs=value.get("token_logprobs") or (),
            total_continuation_logprob=value.get("total_continuation_logprob"),
            evaluator_provenance=value.get("evaluator_provenance"), score_basis=value.get("score_basis"),
            execution_provenance=value.get("execution_provenance"),
            proof_grade=value.get("proof_grade", "unavailable"), trusted=value.get("trusted", False),
            diagnostics=value.get("diagnostics"),
        )


def _span_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, Mapping):
        start, end = value.get("start"), value.get("end")
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        start, end = value
    else:
        raise ObservationError("token_spans must contain start/end pairs")
    if (
        isinstance(start, bool) or isinstance(end, bool)
        or not isinstance(start, int) or not isinstance(end, int)
        or start < 0 or end < start
    ):
        raise ObservationError("token_spans must contain non-negative half-open ranges")
    return int(start), int(end)


class TokenScoreDelta:
    """Pure signed baseline-minus-intervention token evidence."""

    __slots__ = (
        "arm_id", "status", "recorded_token_ids", "token_pieces", "token_spans",
        "baseline_logprobs", "intervened_logprobs", "deltas", "total_delta_nats",
        "baseline_total_logprob", "intervened_total_logprob", "provenance", "diagnostics",
        "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("TokenScoreDelta is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, arm_id: str, status: str, recorded_token_ids=(), token_pieces=(),
                 token_spans=(), baseline_logprobs=(), intervened_logprobs=(), deltas=(),
                 total_delta_nats=None, baseline_total_logprob=None, intervened_total_logprob=None,
                 provenance=None, diagnostics=None):
        if status not in TOKEN_SCORE_STATUSES:
            raise ObservationError(f"unsupported token-score delta status: {status!r}")
        self.arm_id = arm_id
        self.status = status
        self.recorded_token_ids = tuple(recorded_token_ids)
        self.token_pieces = tuple(token_pieces)
        self.token_spans = tuple(tuple(item) for item in token_spans)
        self.baseline_logprobs = tuple(baseline_logprobs)
        self.intervened_logprobs = tuple(intervened_logprobs)
        self.deltas = tuple(deltas)
        self.total_delta_nats = total_delta_nats
        self.baseline_total_logprob = baseline_total_logprob
        self.intervened_total_logprob = intervened_total_logprob
        self.provenance = _mapping(provenance)
        self.diagnostics = _mapping(diagnostics)
        self._sealed = True

    @classmethod
    def from_observations(cls, baseline: TokenScoreObservation,
                          intervention: TokenScoreObservation) -> "TokenScoreDelta":
        if not isinstance(baseline, TokenScoreObservation) or not isinstance(intervention, TokenScoreObservation):
            raise TypeError("TokenScoreDelta requires TokenScoreObservation objects")
        if not baseline.completed or not intervention.completed:
            status = "failed" if "failed" in {baseline.status, intervention.status} else "unavailable"
            return cls(
                arm_id=intervention.arm_id, status=status,
                provenance={"basis": "persisted_token_score_observations"},
                diagnostics={"baseline_status": baseline.status, "intervention_status": intervention.status},
            )
        fields = (
            baseline.recorded_token_ids == intervention.recorded_token_ids,
            baseline.token_pieces == intervention.token_pieces,
            baseline.token_spans == intervention.token_spans,
            len(baseline.token_logprobs) == len(intervention.token_logprobs),
        )
        if not all(fields):
            return cls(
                arm_id=intervention.arm_id, status="unavailable",
                provenance={"basis": "persisted_token_score_observations"},
                diagnostics={"reason": "token_alignment_mismatch"},
            )
        deltas = tuple(base - altered for base, altered in zip(
            baseline.token_logprobs, intervention.token_logprobs))
        return cls(
            arm_id=intervention.arm_id, status="completed",
            recorded_token_ids=baseline.recorded_token_ids,
            token_pieces=baseline.token_pieces, token_spans=baseline.token_spans,
            baseline_logprobs=baseline.token_logprobs,
            intervened_logprobs=intervention.token_logprobs,
            deltas=deltas,
            total_delta_nats=baseline.total_continuation_logprob - intervention.total_continuation_logprob,
            baseline_total_logprob=baseline.total_continuation_logprob,
            intervened_total_logprob=intervention.total_continuation_logprob,
            provenance={
                "basis": "persisted_token_score_observations",
                "sign": "baseline_minus_intervention",
                "baseline_arm_id": baseline.arm_id,
                "intervention_arm_id": intervention.arm_id,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "token_score_delta", "arm_id": self.arm_id, "status": self.status,
            "recorded_token_ids": list(self.recorded_token_ids), "token_pieces": list(self.token_pieces),
            "token_spans": [{"start": start, "end": end} for start, end in self.token_spans],
            "baseline_logprobs": list(self.baseline_logprobs),
            "intervened_logprobs": list(self.intervened_logprobs), "deltas": list(self.deltas),
            "total_delta_nats": self.total_delta_nats,
            "baseline_total_logprob": self.baseline_total_logprob,
            "intervened_total_logprob": self.intervened_total_logprob,
            "provenance": dict(self.provenance), "diagnostics": dict(self.diagnostics),
        }


__all__ = [
    "Observation", "ObservationError", "OBSERVATION_STATUSES", "SCHEMA_VERSION",
    "TokenScoreDelta", "TokenScoreObservation", "TOKEN_SCORE_SCHEMA_VERSION", "TOKEN_SCORE_STATUSES",
]
