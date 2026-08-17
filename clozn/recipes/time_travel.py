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
    ResolvedState, StateRef, StateRefError, enumerate_answer_boundaries, list_answer_token_boundaries,
    operation_readiness, resolve_state,
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
        operation_readiness_value = self.diagnostics.get("operation_readiness")
        if not isinstance(operation_readiness_value, Mapping):
            operation_readiness_value = operation_readiness(
                self.resolved_state,
                operation=self.operation_kind or "continue",
                token_id=self.operation.get("token_id"),
                token_piece=self.operation.get("token_piece"),
            )
        continue_readiness = operation_readiness_value if self.operation_kind == "continue" else operation_readiness(
            self.resolved_state, operation="continue",
        )
        force_readiness = operation_readiness_value if self.operation_kind == "force_token" else operation_readiness(
            self.resolved_state, operation="force_token",
        )
        return {
            "schema_version": TIME_TRAVEL_RESULT_SCHEMA_VERSION,
            "run_id": self.run_id, "state_ref": self.state_ref.to_dict(),
            "resolved_state": self.resolved_state.to_dict(), "operation": deepcopy(dict(self.operation)),
            "experiment_id": self.experiment_id, "arm_id": self.arm_id, "observation_id": self.observation_id,
            "status": self.status, "resolution": self.resolution, "fidelity": self.fidelity,
            "operation_kind": self.operation_kind, "continuation": deepcopy(dict(self.continuation)),
            "available_operations": {
                "continue": deepcopy(dict(continue_readiness)),
                "force_token": deepcopy(dict(force_readiness)),
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
    # Reuse the canonical recorded-run contract reader used by ExecutionState. In particular, do
    # not let a sampled ``meta.decode`` be silently reinterpreted as greedy merely because the
    # recipe's historical projection did not look there.
    try:
        from clozn.runs.answer_preservation import _generation_contract_from_run
        canonical, _reason = _generation_contract_from_run(run)
        if isinstance(canonical, Mapping):
            return dict(canonical)
    except Exception:
        pass
    for key in ("generation_contract", "output_contract"):
        value = run.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    meta = run.get("meta")
    if isinstance(meta, Mapping) and isinstance(meta.get("generation_contract"), Mapping):
        return dict(meta["generation_contract"])
    return {}


def checkpoint_reference_from_pin(pin_result: Mapping[str, Any], *, run_id: str) -> dict[str, Any] | None:
    """Project a verified durable pin into the execution-fork planner's reference shape.

    The pin store deliberately returns the complete export envelope because checkpoint hydration
    needs its bytes.  State planning must not pass that envelope to ``plan_execution_fork`` as if it
    were a live worker reference; this projection keeps the two contracts explicit and carries only
    the immutable planner facts.  The execution seam remains responsible for importing/hydrating the
    envelope when the selected worker needs it.
    """
    if not isinstance(pin_result, Mapping) or pin_result.get("ok") is not True:
        return None
    direct = pin_result.get("checkpoint_reference")
    if isinstance(direct, Mapping):
        return dict(direct)
    manifest = pin_result.get("manifest")
    envelope = pin_result.get("envelope")
    if isinstance(envelope, Mapping) and all(name in envelope for name in
                                             ("checkpoint_id", "worker_generation_id", "state", "parent_run_id")):
        return dict(envelope)
    source = manifest.get("source") if isinstance(manifest, Mapping) else None
    state = envelope.get("state") if isinstance(envelope, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(state, Mapping):
        return None
    checkpoint_id = source.get("checkpoint_id")
    generation = source.get("worker_generation_id")
    if not all(isinstance(value, str) and value for value in (checkpoint_id, generation)):
        return None
    parent_id = manifest.get("run_id") if isinstance(manifest, Mapping) else None
    parent_id = parent_id if isinstance(parent_id, str) and parent_id else run_id
    reference = {
        "checkpoint_id": checkpoint_id,
        "worker_generation_id": generation,
        "state": "available",
        "parent_run_id": parent_id,
        "prompt_tokens": state.get("prompt_tokens"),
        "n_past": state.get("n_past"),
    }
    return reference if all(reference.get(name) is not None for name in
                            ("prompt_tokens", "n_past")) else None


def checkpoint_reference(value: Mapping[str, Any] | None, *, run_id: str) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    required = ("checkpoint_id", "worker_generation_id", "state", "parent_run_id")
    if all(name in value for name in required):
        return dict(value)
    return checkpoint_reference_from_pin(value, run_id=run_id)


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


def _readiness(resolved: ResolvedState, *, operation: Mapping[str, Any],
               decode_mode: str | None = None) -> dict[str, Any]:
    return operation_readiness(
        resolved,
        operation=str(operation.get("kind") or "continue"),
        token_id=operation.get("token_id"), token_piece=operation.get("token_piece"),
        decode_mode=decode_mode,
    )


def _confirmed_readiness(value: Mapping[str, Any], observation: GeneratedObservation) -> dict[str, Any]:
    """Promote only the operation that produced completed direct evidence."""
    result = deepcopy(dict(value))
    result.update({
        "available": True,
        "plannable": True,
        "state": "confirmed",
        "proof_status": "confirmed" if observation.fidelity_classification == "exact_execution_fork" else "not_applicable",
        "reason_code": "exact_confirmed" if observation.fidelity_classification == "exact_execution_fork" else "reconstructed_replay",
    })
    if isinstance(result.get("sampler"), Mapping):
        sampler = dict(result["sampler"])
        if observation.fidelity_classification == "exact_execution_fork":
            sampler["status"] = "confirmed" if observation.fidelity.get("sampler_state") == "confirmed" else "not_required"
        result["sampler"] = sampler
    return result


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
    readiness = _readiness(resolved, operation=operation)
    status = "ready" if resolved.available and readiness.get("plannable", False) else "unavailable"
    diagnostics = {**dict(resolved.diagnostics), "operation_readiness": readiness}
    return TimeTravelResult(
        run_id=state_ref.run_id, state_ref=state_ref, resolved_state=resolved, operation=operation,
        experiment_id=None, arm_id=None, observation_id=None, status=status, continuation={"status": status},
        branch_point=_branch_point(run, state_ref), diagnostics=deepcopy(diagnostics),
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
    readiness = _readiness(resolved, operation=operation, decode_mode=evaluator.decode_mode)
    if not readiness.get("plannable", False):
        diagnostics = {**dict(resolved.diagnostics), "operation_readiness": readiness}
        return TimeTravelResult(
            run_id=state_ref.run_id, state_ref=state_ref, resolved_state=resolved, operation=operation,
            experiment_id=None, arm_id=None, observation_id=None, status="unavailable",
            continuation={"status": "unavailable", "diagnostics": diagnostics},
            branch_point=branch_point, diagnostics=diagnostics,
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
        readiness = _confirmed_readiness(readiness, observation)
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
    diagnostics = {
        **dict(result.diagnostics), **dict(arm.diagnostics if arm is not None else {}),
        "operation_readiness": readiness,
    }
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
        pin_status: dict[str, Any] = {"state": "not_requested"}
        if checkpoint is None:
            try:
                from clozn.replay.checkpoint_pin_store import resolve_pin
                pinned = resolve_pin(run.get("id"))
            except Exception as exc:
                pinned = {"unavailable": f"checkpoint pin lookup failed: {type(exc).__name__}"}
            if isinstance(pinned, Mapping) and pinned.get("ok") is True and isinstance(pinned.get("envelope"), Mapping):
                checkpoint = checkpoint_reference_from_pin(pinned, run_id=str(run.get("id") or ""))
                if checkpoint is None:
                    pin_status = {
                        "state": "unavailable", "reason_code": "checkpoint_pin_unavailable",
                        "reason": "the durable checkpoint pin has no valid planner reference",
                    }
                    checkpoint = None
                else:
                    pin_status = {
                        "state": "stored", "pin_id": (pinned.get("manifest") or {}).get("pin_id"),
                    }
            else:
                pin_reason = str((pinned or {}).get("unavailable") or "no durable checkpoint is pinned for this run")
                pin_code = "checkpoint_missing" if "has no pinned checkpoint" in pin_reason else "checkpoint_pin_unavailable"
                pin_status = {
                    "state": "unavailable", "reason_code": pin_code, "reason": pin_reason,
                }
        else:
            checkpoint = checkpoint_reference(checkpoint, run_id=str(run.get("id") or ""))

        if not boundaries:
            unavailable = {
                "available": False, "state": "unavailable", "reason_code": "token_trace_unavailable",
                "reason": "the run has no aligned recorded answer-token history",
            }
            return {
                "answer_token_boundaries": {**unavailable, "count": 0},
                "exact_checkpoint_restore": unavailable, "reconstructed_replay": unavailable,
                "available_operations": {"continue": unavailable, "force_token": unavailable},
                "generate": unavailable, "force_token": unavailable,
                "sampler_restore": {**unavailable, "required": True}, "checkpoint": pin_status,
            }

        ref = StateRef.before_answer_token(run, boundaries[0].index)
        # A capability read has no live worker selection.  The parent runtime is a static identity
        # fact, and a durable pin supplies the worker generation needed to classify an exact plan.
        # The synthetic worker label is explicitly planning-only and is never used for execution.
        from clozn.replay.execution_fork import parent_runtime_projection
        static_runtime = runtime_identity or parent_runtime_projection(run)
        static_worker = worker_identity
        if static_worker is None and isinstance(checkpoint, Mapping):
            checkpoint_identity = checkpoint.get("identity")
            generation = checkpoint.get("worker_generation_id")
            if generation is None and isinstance(checkpoint_identity, Mapping):
                generation = checkpoint_identity.get("worker_generation_id")
            if isinstance(generation, str) and generation:
                static_worker = {
                    "worker_id": "planning-only:pinned-checkpoint",
                    "worker_generation_id": generation,
                    "protocol_version": "1.0",
                }
        reconstructed_state = resolve_state(
            ref, run=run, policy="reconstructed_only",
        )
        exact_state = resolve_state(
            ref, run=run, policy="exact_required", checkpoint=checkpoint,
            runtime_identity=static_runtime, worker_identity=static_worker,
        ) if checkpoint is not None else ResolvedState(
            state_ref=ref, classification="unavailable", proof_status="not_available",
            realization={"regime": "unavailable"},
            diagnostics={"reason_code": "checkpoint_missing", "message": "exact resolution requires a checkpoint reference"},
        )
        continue_state = exact_state if exact_state.available else reconstructed_state
        continue_ready = operation_readiness(continue_state, operation="continue")
        force_ready = operation_readiness(continue_state, operation="force_token")
        exact_projection = {
            "available": exact_state.available,
            "state": "planned" if exact_state.available else "unavailable",
            "proof_status": exact_state.proof_status,
            "reason_code": exact_state.diagnostics.get("reason_code"),
            "reason": exact_state.diagnostics.get("message"),
            "classification": exact_state.classification,
        }
        if pin_status.get("reason_code") == "checkpoint_pin_unavailable":
            exact_projection.update({
                "reason_code": pin_status["reason_code"], "reason": pin_status.get("reason"),
            })
        reconstructed_projection = {
            "available": reconstructed_state.available,
            "state": "available" if reconstructed_state.available else "unavailable",
            "proof_status": reconstructed_state.proof_status,
            "reason_code": reconstructed_state.diagnostics.get("reason_code"),
            "reason": reconstructed_state.diagnostics.get("message"),
            "classification": reconstructed_state.classification,
            "unavoidable_differences": reconstructed_state.realization.get("unavoidable_differences", []),
        }
        return {
            "answer_token_boundaries": {"available": True, "state": "available", "count": len(boundaries)},
            "exact_checkpoint_restore": exact_projection,
            "reconstructed_replay": reconstructed_projection,
            "available_operations": {"continue": continue_ready, "force_token": force_ready},
            # Keep these names as projections for existing clients, but make them operation-aware
            # instead of aliases of one global state boolean.
            "generate": {**continue_ready, "operation": "continue"},
            "force_token": {**force_ready, "operation": "force_token"},
            "sampler_restore": {
                # Planning an exact checkpoint is not sampler proof.  Only a completed execution
                # receipt may ever turn this into available=true.
                "available": continue_ready.get("sampler", {}).get("status") == "confirmed",
                "state": (
                    "unavailable" if continue_ready.get("sampler", {}).get("status") == "unbound"
                    else "not_required" if continue_ready.get("sampler", {}).get("status") == "not_required"
                    else "requires_verification" if exact_state.available
                    else "unavailable"
                ),
                "required": bool(continue_ready.get("sampler", {}).get("required")),
                "reason_code": (
                    "stochastic_execution_unbound" if continue_ready.get("sampler", {}).get("status") == "unbound" else
                    None if continue_ready.get("sampler", {}).get("status") == "not_required" else
                    "sampler_state_requires_execution_proof" if exact_state.available else
                    exact_state.diagnostics.get("reason_code")
                ),
            },
            "checkpoint": pin_status,
        }
    except StateRefError as exc:
        unavailable = {"available": False, "state": "unavailable", "reason_code": "token_trace_unavailable", "reason": str(exc)}
        return {
            "answer_token_boundaries": unavailable,
            "exact_checkpoint_restore": unavailable, "reconstructed_replay": unavailable,
            "available_operations": {"continue": unavailable, "force_token": unavailable},
            "generate": unavailable, "force_token": unavailable,
            "sampler_restore": {**unavailable, "required": True}, "reason": str(exc),
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
    "run_time_travel", "time_travel", "time_travel_capabilities", "checkpoint_reference", "checkpoint_reference_from_pin",
]
