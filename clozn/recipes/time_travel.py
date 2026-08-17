"""Thin product recipe for recorded-execution time travel.

The recipe owns request/result shaping only. State resolution, token forcing,
generation, evidence persistence, and Run materialization remain kernel seams.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from clozn.experiments.evaluators import Generate
from clozn.experiments.generation import GenerateExecutionAdapter
from clozn.experiments.interventions import ForceToken, InterventionError
from clozn.experiments.kernel import Experiment
from clozn.experiments.materialize import materialize_generated_observation
from clozn.experiments.observations import GeneratedObservation
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.state_ref import (
    ResolvedState, StateRef, StateRefError, enumerate_answer_boundaries, list_answer_token_boundaries, resolve_state,
)

TIME_TRAVEL_RESULT_SCHEMA_VERSION = "clozn.time-travel-result.v1"
_FIDELITY = {"exact_execution_fork": "EXACT", "reconstructed_replay": "RECONSTRUCTED", "unavailable": "UNAVAILABLE"}


class TimeTravelError(ValueError):
    """A typed product-level time-travel request failure."""

    def __init__(self, message: str, *, code: str, status: str = "unavailable"):
        super().__init__(message)
        self.code = code
        self.status = status


@dataclass(frozen=True)
class TimeTravelResult:
    """Read-side projection pointing at a GeneratedObservation, not its store."""

    run_id: str
    state_ref: StateRef
    resolved_state: ResolvedState
    operation: Mapping[str, Any]
    experiment_id: str | None
    arm_id: str | None
    observation_id: str | None
    status: str
    continuation: Mapping[str, Any]
    branch_point: Mapping[str, Any]
    diagnostics: Mapping[str, Any]

    def __post_init__(self):
        if self.status not in {"ready", "completed", "unavailable", "failed"}:
            raise ValueError("unsupported TimeTravelResult status")
        if not isinstance(self.run_id, str) or not self.run_id:
            raise ValueError("TimeTravelResult.run_id is required")

    @property
    def fidelity(self) -> str:
        value = self.continuation.get("fidelity")
        classification = value.get("classification") if isinstance(value, Mapping) else None
        if classification in _FIDELITY:
            return _FIDELITY[classification]
        # Planning an exact checkpoint is not confirmation. An exact result
        # without a completed GeneratedObservation must therefore remain
        # explicitly unconfirmed.
        if self.status == "ready" and self.resolved_state.classification == "reconstructed_replay":
            return "RECONSTRUCTED"
        return "UNAVAILABLE"

    @property
    def resolution(self) -> str:
        return _FIDELITY.get(self.resolved_state.classification, "UNAVAILABLE")

    @property
    def operation_kind(self) -> str:
        return str(self.operation.get("kind") or "")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": TIME_TRAVEL_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id, "state_ref": self.state_ref.to_dict(),
            "resolved_state": self.resolved_state.to_dict(), "operation": deepcopy(dict(self.operation)),
            "experiment_id": self.experiment_id, "arm_id": self.arm_id, "observation_id": self.observation_id,
            "status": self.status, "resolution": self.resolution, "fidelity": self.fidelity,
            "operation_kind": self.operation_kind, "continuation": deepcopy(dict(self.continuation)),
            "available_operations": {
                "continue": self.resolved_state.available,
                "force_token": self.resolved_state.available,
            },
            "reason_code": self.diagnostics.get("reason_code"),
            "reason": self.diagnostics.get("message") or self.diagnostics.get("reason"),
            "branch_point": deepcopy(dict(self.branch_point)), "diagnostics": deepcopy(dict(self.diagnostics)),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimeTravelResult":
        if not isinstance(value, Mapping) or value.get("schema_version") != TIME_TRAVEL_RESULT_SCHEMA_VERSION:
            raise TimeTravelError(f"TimeTravelResult must declare {TIME_TRAVEL_RESULT_SCHEMA_VERSION}", code="malformed_time_travel_result")
        try:
            state_ref = StateRef.from_dict(value.get("state_ref"))
            resolved = ResolvedState.from_dict(value.get("resolved_state"))
        except Exception as exc:
            raise TimeTravelError("TimeTravelResult contains malformed state evidence", code="malformed_time_travel_result") from exc
        return cls(
            run_id=value.get("run_id"), state_ref=state_ref, resolved_state=resolved,
            operation=value.get("operation") or {}, experiment_id=value.get("experiment_id"), arm_id=value.get("arm_id"),
            observation_id=value.get("observation_id"), status=value.get("status"),
            continuation=value.get("continuation") or {}, branch_point=value.get("branch_point") or {},
            diagnostics=value.get("diagnostics") or {},
        )


def _run_tokens(run: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    trace = run.get("trace")
    pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
    ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    if not (isinstance(pieces, list) and isinstance(ids, list) and pieces and ids and len(pieces) == len(ids)
            and all(isinstance(piece, str) for piece in pieces)
            and all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0 for token_id in ids)):
        raise TimeTravelError("recorded answer token evidence is unavailable", code="token_trace_unavailable")
    return pieces, ids


def _branch_point(run: Mapping[str, Any], state_ref: StateRef) -> dict[str, Any]:
    pieces, _ids = _run_tokens(run)
    index = state_ref.position.index
    if index > len(pieces):
        raise TimeTravelError("the requested token boundary is outside the recorded answer", code="token_boundary_out_of_range")
    return {"state_ref": state_ref.to_dict(), "answer_token_index": index,
            "recorded_prefix_token_count": index, "recorded_prefix_text": "".join(pieces[:index])}


def _parent_generation_contract(run: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("generation_contract", "output_contract"):
        value = run.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    meta = run.get("meta")
    if isinstance(meta, Mapping) and isinstance(meta.get("generation_contract"), Mapping):
        return dict(meta["generation_contract"])
    return {}


def _generate(run: Mapping[str, Any], *, max_new: int, decode_mode: str | None,
              sampling: Mapping[str, Any] | None, stop: list[str] | tuple[str, ...] | None) -> Generate:
    contract = _parent_generation_contract(run)
    parent_decode = contract.get("decode_mode")
    if parent_decode is None and isinstance(contract.get("decode"), Mapping):
        parent_decode = contract["decode"].get("mode")
    mode = decode_mode or parent_decode or "greedy"
    parent_sampling = contract.get("sampling")
    resolved_sampling = dict(sampling) if sampling is not None else (dict(parent_sampling) if isinstance(parent_sampling, Mapping) else None)
    parent_stop = contract.get("stop")
    resolved_stop = stop if stop is not None else (parent_stop if isinstance(parent_stop, (list, tuple)) else ())
    try:
        return Generate(max_new=max_new, decode_mode=mode, sampling=resolved_sampling, stop=resolved_stop)
    except Exception as exc:
        raise TimeTravelError(f"generation contract is unavailable: {exc}", code="generation_unsupported") from exc


def _operation(token_id: int | None, token_piece: str | None) -> tuple[dict[str, Any], ForceToken | None]:
    if token_id is None and token_piece is None:
        return {"kind": "continue"}, None
    try:
        intervention = ForceToken(token_id=token_id, token_piece=token_piece)
    except InterventionError as exc:
        raise TimeTravelError(str(exc), code="force_token_unsupported") from exc
    return {"kind": "force_token", **intervention.to_dict()}, intervention


def _continuation(observation: GeneratedObservation | None, *, status: str = "unavailable",
                  diagnostics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not isinstance(observation, GeneratedObservation):
        return {"status": status, "diagnostics": deepcopy(dict(diagnostics or {}))}
    return {
        "status": observation.status, "observation_id": observation.observation_id,
        "fidelity": deepcopy(observation.fidelity), "generated_suffix_text": observation.generated_suffix_text,
        "generated_token_ids": list(observation.generated_token_ids), "generated_steps": deepcopy(observation.generated_steps),
        "finish_reason": observation.finish_reason, "generation_contract": deepcopy(observation.generation_contract),
        "runtime_provenance": deepcopy(observation.runtime_provenance), "exact_control_proof": deepcopy(observation.exact_control_proof),
        "diagnostics": deepcopy(observation.diagnostics),
    }


def resolve_time_travel(run: Mapping[str, Any], *, position: int, policy: str = "exact_preferred",
                        checkpoint: Mapping[str, Any] | None = None, runtime_identity: Mapping[str, Any] | None = None,
                        worker_identity: Mapping[str, Any] | None = None, token_id: int | None = None,
                        token_piece: str | None = None) -> TimeTravelResult:
    """Resolve one logical boundary without model or worker calls."""
    if not isinstance(run, Mapping):
        raise TimeTravelError("recorded run is required", code="run_not_found")
    try:
        state_ref = StateRef.before_answer_token(run, position)
    except StateRefError as exc:
        code = "token_trace_unavailable" if "history" in str(exc) or "malformed" in str(exc) else "token_boundary_out_of_range"
        raise TimeTravelError(str(exc), code=code) from exc
    operation, _intervention = _operation(token_id, token_piece)
    try:
        resolved = resolve_state(state_ref, run=run, policy=policy, checkpoint=checkpoint,
                                 runtime_identity=runtime_identity, worker_identity=worker_identity)
    except StateRefError as exc:
        code = "invalid_resolution_policy" if "unsupported state resolution" in str(exc) else "exact_restore_unavailable"
        raise TimeTravelError(str(exc), code=code) from exc
    status = "ready" if resolved.available else "unavailable"
    return TimeTravelResult(
        run_id=state_ref.run_id, state_ref=state_ref, resolved_state=resolved, operation=operation,
        experiment_id=None, arm_id=None, observation_id=None, status=status, continuation={"status": status},
        branch_point=_branch_point(run, state_ref), diagnostics=deepcopy(resolved.diagnostics),
    )


def run_time_travel(
    run: Mapping[str, Any], *, position: int, token_id: int | None = None, token_piece: str | None = None,
    max_new: int, policy: str = "exact_preferred", checkpoint: Mapping[str, Any] | None = None,
    runtime_identity: Mapping[str, Any] | None = None, worker_identity: Mapping[str, Any] | None = None,
    execution_adapter: Any | None = None, substrate: Any | None = None, run_loader=None,
    observation_store: ObservationStore | None = None, store: ObservationStore | None = None,
    include_control: bool = True, decode_mode: str | None = None, sampling: Mapping[str, Any] | None = None,
    stop: list[str] | tuple[str, ...] | None = None,
) -> TimeTravelResult:
    """Resolve, execute, and project one Continue or ForceToken experiment."""
    if not isinstance(run, Mapping):
        raise TimeTravelError("recorded run is required", code="run_not_found")
    try:
        state_ref = StateRef.before_answer_token(run, position)
    except StateRefError as exc:
        code = "token_trace_unavailable" if "history" in str(exc) or "malformed" in str(exc) else "token_boundary_out_of_range"
        raise TimeTravelError(str(exc), code=code) from exc
    operation, intervention = _operation(token_id, token_piece)
    try:
        resolved = resolve_state(state_ref, run=run, policy=policy, checkpoint=checkpoint,
                                 runtime_identity=runtime_identity, worker_identity=worker_identity)
    except StateRefError as exc:
        code = "invalid_resolution_policy" if "unsupported state resolution" in str(exc) else "exact_restore_unavailable"
        raise TimeTravelError(str(exc), code=code) from exc
    branch_point = _branch_point(run, state_ref)
    if not resolved.available:
        return TimeTravelResult(
            run_id=state_ref.run_id, state_ref=state_ref, resolved_state=resolved, operation=operation,
            experiment_id=None, arm_id=None, observation_id=None, status="unavailable",
            continuation={"status": "unavailable"}, branch_point=branch_point, diagnostics=deepcopy(resolved.diagnostics),
        )
    try:
        evaluator = _generate(run, max_new=max_new, decode_mode=decode_mode, sampling=sampling, stop=stop)
    except TimeTravelError as exc:
        return TimeTravelResult(
            run_id=state_ref.run_id, state_ref=state_ref, resolved_state=resolved, operation=operation,
            experiment_id=None, arm_id=None, observation_id=None, status="unavailable",
            continuation={"status": "unavailable", "diagnostics": {"reason_code": exc.code, "message": str(exc)}},
            branch_point=branch_point, diagnostics={"reason_code": exc.code, "message": str(exc)},
        )
    experiment = Experiment(base=resolved, evaluator=evaluator, arms=[intervention])
    adapter = execution_adapter
    if adapter is None:
        if substrate is None:
            raise TimeTravelError("execution adapter or substrate is required", code="generation_unsupported")
        adapter = GenerateExecutionAdapter(substrate, run=run, run_loader=run_loader,
                                           runtime_identity=runtime_identity, worker_identity=worker_identity)
    durable = observation_store or store or ObservationStore()
    result = run_experiment(experiment, adapter, include_control=include_control,
                            observation_store=durable, requested_by={"recipe": "time_travel"})
    arm = result.arms[0] if result.arms else None
    observation = arm.observation if arm is not None else None
    if isinstance(observation, GeneratedObservation) and observation.status == "completed":
        status = "completed"
    elif isinstance(observation, GeneratedObservation) and observation.status == "failed":
        status = "failed"
    elif arm is not None and (
        arm.state == "failed" or arm.diagnostics.get("observation_status") == "failed"
    ):
        status = "failed"
    elif result.control is not None and result.control.status == "failed":
        status = "failed"
    else:
        status = "unavailable"
    diagnostics = {**dict(result.diagnostics), **dict(arm.diagnostics if arm is not None else {})}
    return TimeTravelResult(
        run_id=state_ref.run_id, state_ref=state_ref, resolved_state=resolved, operation=operation,
        experiment_id=result.experiment_id, arm_id=arm.arm_id if arm is not None else None,
        observation_id=arm.observation_id if arm is not None else None, status=status,
        continuation=_continuation(observation, status=status, diagnostics=diagnostics),
        branch_point=branch_point, diagnostics=diagnostics,
    )


def time_travel_capabilities(run: Mapping[str, Any], *, checkpoint: Mapping[str, Any] | None = None,
                             runtime_identity: Mapping[str, Any] | None = None,
                             worker_identity: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Expose cheap Run-bound capabilities without trial execution."""
    try:
        boundaries = enumerate_answer_boundaries(run)
        exact = reconstructed = False
        if boundaries:
            ref = StateRef.before_answer_token(run, boundaries[0].index)
            reconstructed = resolve_state(ref, run=run, policy="reconstructed_only").available
            exact = checkpoint is not None and resolve_state(
                ref, run=run, policy="exact_required", checkpoint=checkpoint,
                runtime_identity=runtime_identity, worker_identity=worker_identity).available
        return {
            "answer_token_boundaries": {"available": bool(boundaries), "count": len(boundaries)},
            "exact_checkpoint_restore": {"available": exact}, "reconstructed_replay": {"available": reconstructed},
            "generate": {"available": exact or reconstructed}, "force_token": {"available": exact or reconstructed},
            "sampler_restore": {"available": exact},
        }
    except StateRefError as exc:
        return {
            "answer_token_boundaries": {"available": False, "reason_code": "token_trace_unavailable"},
            "exact_checkpoint_restore": {"available": False}, "reconstructed_replay": {"available": False},
            "generate": {"available": False}, "force_token": {"available": False},
            "sampler_restore": {"available": False}, "reason": str(exc),
        }


def materialize_time_travel(base_run: Mapping[str, Any], result: TimeTravelResult | str,
                            arm_id: str | None = None, *, observation_store: ObservationStore | None = None,
                            observation_id: str | None = None, reload_parent=None) -> dict[str, Any]:
    """Promote already-generated evidence through the generic materializer."""
    experiment_id = result.experiment_id if isinstance(result, TimeTravelResult) else result
    selected_arm = arm_id or (result.arm_id if isinstance(result, TimeTravelResult) else None)
    selected_observation = observation_id or (result.observation_id if isinstance(result, TimeTravelResult) else None)
    if not experiment_id or not selected_arm:
        raise TimeTravelError("a completed time-travel arm is required", code="materialization_failed")
    try:
        return materialize_generated_observation(
            base_run, experiment_id, selected_arm, observation_id=selected_observation,
            observation_store=observation_store or ObservationStore(), reload_parent=reload_parent,
        )
    except Exception as exc:
        from clozn.experiments.materialize import MaterializationStaleError
        code = "observation_stale" if isinstance(exc, MaterializationStaleError) else "materialization_failed"
        raise TimeTravelError(str(exc), code=code, status="failed") from exc


continue_from_here = run_time_travel
force_token_and_continue = run_time_travel
time_travel = run_time_travel

__all__ = [
    "TIME_TRAVEL_RESULT_SCHEMA_VERSION", "TimeTravelError", "TimeTravelResult", "continue_from_here",
    "enumerate_answer_boundaries", "list_answer_token_boundaries", "force_token_and_continue", "materialize_time_travel", "resolve_time_travel",
    "run_time_travel", "time_travel", "time_travel_capabilities",
]
