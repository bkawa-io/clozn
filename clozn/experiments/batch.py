"""Generic per-arm execution contracts.

The experiment runner deals in semantic arm identities, while execution
adapters are free to choose scalar, native-many, or shared-state mechanics.
This module is deliberately model- and recipe-independent.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from copy import deepcopy
from typing import Any


EXECUTION_DISPOSITIONS = frozenset({"reused", "executed", "not_executed"})
EXECUTION_STATES = frozenset({"completed", "failed", "cancelled", "not_executed"})


@dataclass(frozen=True)
class ArmExecutionRequest:
    """One complete semantic condition awaiting execution."""

    arm_id: str
    state: Any
    intervention: Any
    evaluator: Any


@dataclass
class ArmExecutionOutcome:
    """Detached result for one requested arm.

    ``execution_disposition`` describes orchestration, not epistemic status:
    an executed arm may carry unavailable or failed evidence and therefore no
    reusable Observation.
    """

    arm_id: str
    observation: Any = None
    execution_disposition: str = "executed"
    state: str = "completed"
    error: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.arm_id, str) or not self.arm_id:
            raise ValueError("ArmExecutionOutcome.arm_id must be non-empty")
        if self.execution_disposition not in EXECUTION_DISPOSITIONS:
            raise ValueError("unsupported execution disposition")
        if self.state not in EXECUTION_STATES:
            raise ValueError("unsupported execution state")
        self.error = dict(self.error or {})
        self.diagnostics = dict(self.diagnostics or {})

    def detached(self) -> "ArmExecutionOutcome":
        return ArmExecutionOutcome(
            arm_id=self.arm_id, observation=self.observation,
            execution_disposition=self.execution_disposition, state=self.state,
            error=deepcopy(self.error), diagnostics=deepcopy(self.diagnostics),
        )


@dataclass
class BatchExecutionResult:
    """Ordered, arm-addressed results from one dispatch."""

    outcomes: tuple[ArmExecutionOutcome, ...] = ()
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        self.outcomes = tuple(self.outcomes)
        ids = [item.arm_id for item in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("batch outcomes must have unique arm IDs")
        self.diagnostics = dict(self.diagnostics or {})

    @property
    def by_arm_id(self) -> dict[str, ArmExecutionOutcome]:
        return {item.arm_id: item for item in self.outcomes}


class BatchExecutionError(RuntimeError):
    """Typed dispatch failure retaining all detached per-arm outcomes."""

    def __init__(self, message: str, *, outcomes: Iterable[ArmExecutionOutcome] = (),
                 diagnostics: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.outcomes = tuple(item.detached() for item in outcomes)
        self.diagnostics = dict(diagnostics or {})


def scalar_batch(adapter: Any, requests: Iterable[ArmExecutionRequest], *, cancel: Any = None) -> BatchExecutionResult:
    """Reference batch strategy: call the scalar adapter once per request."""
    outcomes: list[ArmExecutionOutcome] = []
    requests = tuple(requests)
    for index, request in enumerate(requests):
        if _cancelled(cancel):
            outcomes.extend(
                ArmExecutionOutcome(
                    arm_id=pending.arm_id, execution_disposition="not_executed",
                    state="cancelled", diagnostics={"reason": "cancelled_before_dispatch"},
                )
                for pending in requests[index:]
            )
            break
        try:
            observation = adapter.execute(
                request.state, request.intervention,
                evaluator=request.evaluator, arm_id=request.arm_id,
            )
            outcomes.append(ArmExecutionOutcome(
                arm_id=request.arm_id, observation=observation,
                execution_disposition="executed",
                state="completed" if getattr(observation, "completed", False) else "failed",
                diagnostics={"execution_strategy": "scalar"},
            ))
        except Exception as exc:
            # The scalar call was dispatched before the exception was raised.
            outcomes.append(ArmExecutionOutcome(
                arm_id=request.arm_id, execution_disposition="executed", state="failed",
                error={"error": str(exc)}, diagnostics={"execution_strategy": "scalar"},
            ))
    return BatchExecutionResult(tuple(outcomes), {"execution_strategy": "scalar", "batch_count": 1})


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    method = getattr(cancel, "is_set", None)
    if callable(method):
        return bool(method())
    return bool(cancel)


__all__ = [
    "ArmExecutionOutcome", "ArmExecutionRequest", "BatchExecutionError",
    "BatchExecutionResult", "EXECUTION_DISPOSITIONS", "EXECUTION_STATES", "scalar_batch",
]
