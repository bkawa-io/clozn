"""Model-free addressing and realization contracts for recorded execution states."""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from .state import ExecutionState, canonical_json, digest


STATE_REF_SCHEMA_VERSION = "clozn.experiment-state-ref.v1"
RESOLVED_STATE_SCHEMA_VERSION = "clozn.experiment-resolved-state.v1"
STATE_CLASSIFICATIONS = frozenset({
    "exact_execution_fork", "reconstructed_replay", "unavailable",
})
RESOLUTION_POLICIES = frozenset({
    "exact_required", "exact_preferred", "reconstructed_only",
})
STOCHASTIC_EXECUTION_UNBOUND = "stochastic_execution_unbound"
STOCHASTIC_EXECUTION_UNBOUND_MESSAGE = (
    "the current replay protocol does not bind the sampler/RNG state for a reusable continuation"
)


class StateRefError(ValueError):
    """A logical execution address cannot be bound or resolved honestly."""


class AnswerTokenBoundary:
    """Canonical coordinate: after tokens ``0..index-1``, before token ``index``."""

    __slots__ = ("index", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("AnswerTokenBoundary is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, index: int):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise StateRefError("answer token boundary index must be a non-negative integer")
        self.index = index
        self._sealed = True

    def to_dict(self) -> dict[str, Any]:
        return {"kind": "answer_token_boundary", "index": self.index}

    def __repr__(self) -> str:
        return f"AnswerTokenBoundary(index={self.index})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AnswerTokenBoundary) and self.index == other.index

    def __hash__(self) -> int:
        return hash((type(self), self.index))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "AnswerTokenBoundary":
        if not isinstance(value, Mapping) or value.get("kind") != "answer_token_boundary":
            raise StateRefError("expected an answer_token_boundary position")
        return cls(value.get("index"))


class RecordedAnswerBoundary:
    """Read-only projection of one addressable recorded answer-token boundary.

    ``recorded_token_id`` and ``recorded_token_piece`` describe the token whose
    decision is next at this boundary.  They are evidence projections, not a
    second coordinate system; the canonical address remains ``index``.
    """

    __slots__ = (
        "index", "recorded_token_id", "recorded_token_piece", "response_offset",
        "state_fingerprint", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("RecordedAnswerBoundary is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, index: int, recorded_token_id: int,
                 recorded_token_piece: str, response_offset: int,
                 state_fingerprint: str):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise StateRefError("boundary index must be a non-negative integer")
        if isinstance(recorded_token_id, bool) or not isinstance(recorded_token_id, int) or recorded_token_id < 0:
            raise StateRefError("recorded boundary token ID must be a non-negative integer")
        if not isinstance(recorded_token_piece, str):
            raise StateRefError("recorded boundary token piece must be a string")
        if isinstance(response_offset, bool) or not isinstance(response_offset, int) or response_offset < 0:
            raise StateRefError("recorded boundary response offset must be non-negative")
        if not isinstance(state_fingerprint, str) or not state_fingerprint:
            raise StateRefError("recorded boundary state fingerprint is required")
        self.index = index
        self.recorded_token_id = recorded_token_id
        self.recorded_token_piece = recorded_token_piece
        self.response_offset = response_offset
        self.state_fingerprint = state_fingerprint
        self._sealed = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "answer_token_boundary",
            "index": self.index,
            "recorded_token_id": self.recorded_token_id,
            "recorded_token_piece": self.recorded_token_piece,
            "token_id": self.recorded_token_id,
            "token_piece": self.recorded_token_piece,
            "response_offset": self.response_offset,
            "state_fingerprint": self.state_fingerprint,
        }

    @property
    def token_id(self) -> int:
        return self.recorded_token_id

    @property
    def token_piece(self) -> str:
        return self.recorded_token_piece


def _recorded_tokens(run: Mapping[str, Any]) -> tuple[list[str], list[int]]:
    if not isinstance(run, Mapping):
        raise StateRefError("recorded token history requires a run mapping")
    trace = run.get("trace")
    pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
    token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
    if not (
        isinstance(pieces, list) and pieces
        and all(isinstance(piece, str) for piece in pieces)
        and isinstance(token_ids, list) and token_ids
        and all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
                for token_id in token_ids)
        and len(pieces) == len(token_ids)
    ):
        raise StateRefError("recorded token history is malformed or unavailable")
    return pieces, token_ids


def enumerate_answer_boundaries(run: Mapping[str, Any]) -> tuple[RecordedAnswerBoundary, ...]:
    """Enumerate the canonical, model-free answer-token decision boundaries.

    The final recorded token is the last addressable decision.  A boundary
    does not imply that a checkpoint or a worker is available; that is the
    responsibility of :func:`resolve_state`.
    """
    pieces, token_ids = _recorded_tokens(run)
    result = []
    offset = 0
    for index, (piece, token_id) in enumerate(zip(pieces, token_ids)):
        ref = StateRef.before_answer_token(run, index)
        result.append(RecordedAnswerBoundary(
            index=index, recorded_token_id=token_id, recorded_token_piece=piece,
            response_offset=offset, state_fingerprint=ref.state_fingerprint,
        ))
        offset += len(piece)
    return tuple(result)


list_answer_token_boundaries = enumerate_answer_boundaries


class StateRef:
    """An immutable reference to one recorded answer-token boundary."""

    __slots__ = ("execution", "position", "state_fingerprint", "_sealed")

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("StateRef is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, execution: ExecutionState, position: AnswerTokenBoundary | int):
        if not isinstance(execution, ExecutionState):
            raise TypeError("StateRef.execution must be an ExecutionState")
        if isinstance(position, int) and not isinstance(position, bool):
            position = AnswerTokenBoundary(position)
        if not isinstance(position, AnswerTokenBoundary):
            raise TypeError("StateRef.position must be an AnswerTokenBoundary")
        token_count = execution.recorded_answer_token_identity.get("token_count")
        token_hash = execution.recorded_answer_token_identity.get("token_ids_sha256")
        if (isinstance(token_count, bool) or not isinstance(token_count, int) or token_count <= 0
                or not isinstance(token_hash, str) or not token_hash):
            raise StateRefError("recorded answer token history is unavailable")
        if position.index >= token_count:
            raise StateRefError(
                f"answer token boundary {position.index} is outside {token_count} recorded tokens"
            )
        self.execution = execution
        self.position = position
        self.state_fingerprint = digest({
            "run_id": execution.run_id,
            "execution_fingerprint": execution.execution_fingerprint,
            "position": position.to_dict(),
            "recorded_answer_token_identity": dict(execution.recorded_answer_token_identity),
        })
        self._sealed = True

    @classmethod
    def from_run(cls, run: Mapping[str, Any], index: int) -> "StateRef":
        if not isinstance(run, Mapping):
            raise StateRefError("StateRef requires a recorded run mapping")
        trace = run.get("trace")
        pieces = trace.get("tokens") if isinstance(trace, Mapping) else None
        token_ids = trace.get("token_ids") if isinstance(trace, Mapping) else None
        if not (
            isinstance(pieces, list) and pieces
            and all(isinstance(piece, str) for piece in pieces)
            and isinstance(token_ids, list) and token_ids
            and all(isinstance(token_id, int) and not isinstance(token_id, bool) and token_id >= 0
                    for token_id in token_ids)
            and len(pieces) == len(token_ids)
        ):
            raise StateRefError("recorded token history is malformed or unavailable")
        return cls(execution=ExecutionState.from_run(run), position=index)

    @classmethod
    def before_answer_token(cls, run_or_execution: Mapping[str, Any] | ExecutionState, index: int) -> "StateRef":
        if isinstance(run_or_execution, ExecutionState):
            return cls(execution=run_or_execution, position=index)
        return cls.from_run(run_or_execution, index)

    @classmethod
    def prompt_boundary(cls, run_or_execution: Mapping[str, Any] | ExecutionState) -> "StateRef":
        return cls.before_answer_token(run_or_execution, 0)

    @property
    def run_id(self) -> str:
        return self.execution.run_id

    @property
    def execution_fingerprint(self) -> str:
        return self.execution.execution_fingerprint

    @property
    def index(self) -> int:
        return self.position.index

    def identity_payload(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_fingerprint": self.execution_fingerprint,
            "position": self.position.to_dict(),
            "state_fingerprint": self.state_fingerprint,
        }

    def __repr__(self) -> str:
        return f"StateRef(run_id={self.run_id!r}, index={self.index})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, StateRef) and self.state_fingerprint == other.state_fingerprint

    def __hash__(self) -> int:
        return hash((type(self), self.state_fingerprint))

    def assert_current(self, run: Mapping[str, Any]) -> ExecutionState:
        try:
            current = ExecutionState.from_run(run)
        except Exception as exc:
            raise StateRefError(f"current parent execution state is unavailable: {exc}") from exc
        if current.run_id != self.run_id or current.execution_fingerprint != self.execution_fingerprint:
            raise StateRefError("StateRef is stale relative to the current parent execution")
        return current

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STATE_REF_SCHEMA_VERSION,
            "execution": self.execution.to_dict(),
            "position": self.position.to_dict(),
            "state_fingerprint": self.state_fingerprint,
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "StateRef":
        if not isinstance(value, Mapping) or value.get("schema_version") != STATE_REF_SCHEMA_VERSION:
            raise StateRefError(f"StateRef must declare {STATE_REF_SCHEMA_VERSION}")
        result = cls(
            execution=ExecutionState.from_dict(value.get("execution")),
            position=AnswerTokenBoundary.from_dict(value.get("position")),
        )
        if value.get("state_fingerprint") != result.state_fingerprint:
            raise StateRefError("StateRef fingerprint does not match its canonical position")
        return result


class ResolvedState:
    """A state address plus an explicit, non-proven realization regime."""

    __slots__ = (
        "state_ref", "classification", "realization", "realization_fingerprint",
        "proof_status", "diagnostics", "plan", "_sealed",
    )

    def __setattr__(self, name, value):
        if getattr(self, "_sealed", False):
            raise AttributeError("ResolvedState is immutable")
        object.__setattr__(self, name, value)

    def __init__(self, *, state_ref: StateRef, classification: str,
                 realization: Mapping[str, Any] | None = None,
                 realization_fingerprint: str | None = None,
                 proof_status: str = "planned",
                 diagnostics: Mapping[str, Any] | None = None,
                 plan: Mapping[str, Any] | None = None):
        if not isinstance(state_ref, StateRef):
            raise TypeError("ResolvedState.state_ref must be a StateRef")
        if classification not in STATE_CLASSIFICATIONS:
            raise StateRefError(f"unsupported resolved-state classification: {classification!r}")
        if not isinstance(proof_status, str) or not proof_status:
            raise StateRefError("ResolvedState.proof_status must be non-empty")
        realization_value = deepcopy(dict(realization or {}))
        calculated = digest({
            "state_ref": state_ref.identity_payload(),
            "classification": classification,
            "realization": realization_value,
        })
        if realization_fingerprint is not None and realization_fingerprint != calculated:
            raise StateRefError("resolved-state realization fingerprint does not match its contents")
        self.state_ref = state_ref
        self.classification = classification
        self.realization = realization_value
        self.realization_fingerprint = calculated
        self.proof_status = proof_status
        self.diagnostics = deepcopy(dict(diagnostics or {}))
        self.plan = deepcopy(dict(plan)) if isinstance(plan, Mapping) else None
        self._sealed = True

    @property
    def execution(self) -> ExecutionState:
        return self.state_ref.execution

    @property
    def run_id(self) -> str:
        return self.execution.run_id

    @property
    def execution_fingerprint(self) -> str:
        return self.execution.execution_fingerprint

    @property
    def position(self) -> AnswerTokenBoundary:
        return self.state_ref.position

    @property
    def state_fingerprint(self) -> str:
        return digest({
            "state_ref": self.state_ref.state_fingerprint,
            "realization_fingerprint": self.realization_fingerprint,
        })

    @property
    def available(self) -> bool:
        return self.classification != "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESOLVED_STATE_SCHEMA_VERSION,
            "state_ref": self.state_ref.to_dict(),
            "classification": self.classification,
            "realization": deepcopy(self.realization),
            "realization_fingerprint": self.realization_fingerprint,
            "proof_status": self.proof_status,
            "diagnostics": deepcopy(self.diagnostics),
            "plan": deepcopy(self.plan),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResolvedState":
        if not isinstance(value, Mapping) or value.get("schema_version") != RESOLVED_STATE_SCHEMA_VERSION:
            raise StateRefError(f"ResolvedState must declare {RESOLVED_STATE_SCHEMA_VERSION}")
        return cls(
            state_ref=StateRef.from_dict(value.get("state_ref")),
            classification=value.get("classification"), realization=value.get("realization"),
            realization_fingerprint=value.get("realization_fingerprint"),
            proof_status=value.get("proof_status"), diagnostics=value.get("diagnostics"),
            plan=value.get("plan"),
        )


def _unavailable(state_ref: StateRef, code: str, message: str, *, plan: Mapping[str, Any] | None = None) -> ResolvedState:
    return ResolvedState(
        state_ref=state_ref, classification="unavailable", proof_status="not_available",
        realization={"regime": "unavailable"}, diagnostics={"reason_code": code, "message": message},
        plan=plan,
    )


def _reconstructed(state_ref: StateRef, *, plan: Mapping[str, Any] | None = None,
                   reason_code: str = "reconstructed_only",
                   message: str = "the caller explicitly permitted reconstructed replay") -> ResolvedState:
    from clozn.replay.execution_fork import RECONSTRUCTION_DIFFERENCES
    exactness = plan.get("exactness") if isinstance(plan, Mapping) else {}
    realization = {
        "regime": exactness.get("regime", "reconstructed_text"),
        "source": exactness.get("source", "text_retokenization"),
        "unavoidable_differences": list(
            plan.get("unavoidable_differences") or RECONSTRUCTION_DIFFERENCES
        ) if isinstance(plan, Mapping) else list(RECONSTRUCTION_DIFFERENCES),
    }
    return ResolvedState(
        state_ref=state_ref, classification="reconstructed_replay", proof_status="planned",
        realization=realization,
        diagnostics={"reason_code": reason_code, "message": message}, plan=plan,
    )


def _model_free_reconstruction_plan(state_ref: StateRef, source: Mapping[str, Any]) -> dict[str, Any]:
    """Build the small reconstruction plan when no live identity is available.

    ``plan_execution_fork`` deliberately requires a selected worker because its normal caller is
    preparing a live fork.  A read-only capability request, and a reconstructed experiment, do not
    need a worker identity: the replay path uses the recorded prompt and explicitly reports that KV,
    sampler state, and batch shape are not restored.  Keep the plan's semantic fields aligned with
    the execution-fork planner without manufacturing a live worker binding.
    """
    from clozn.replay.execution_fork import RECONSTRUCTION_DIFFERENCES

    return {
        "classification": "reconstructed_replay",
        "parent_run_id": source.get("id"),
        "request": {
            "position": state_ref.position.index,
            "change": {"type": "none"},
            "execution_change": {"type": "none"},
        },
        "exactness": {
            "regime": "reconstructed_text",
            "source": "text_retokenization",
            "proof_status": "not_applicable",
        },
        "unavoidable_differences": list(RECONSTRUCTION_DIFFERENCES),
        "unchanged_control": {"required": True, "status": "required_not_run"},
        "reasons": [{
            "code": "checkpoint_not_supplied",
            "message": "no exact checkpoint was supplied; the eligible path explicitly reconstructs text",
        }],
    }


def operation_readiness(
    resolved_state: "ResolvedState", *, operation: str,
    token_id: int | None = None, token_piece: str | None = None,
    decode_mode: str | None = None,
) -> dict[str, Any]:
    """Classify whether one Continue/ForceToken operation is requestable.

    This is a planning projection only.  ``available`` means that the operation can be sent to the
    execution seam without violating its input contract; it never means that an exact operation has
    already established fidelity.  Exact operations therefore remain ``requires_verification``
    until the unchanged control and worker receipt confirm them.
    """
    if not isinstance(resolved_state, ResolvedState):
        raise TypeError("operation_readiness requires a ResolvedState")
    if operation not in {"continue", "force_token"}:
        raise StateRefError(f"unsupported time-travel operation: {operation!r}")

    classification = resolved_state.classification
    base: dict[str, Any] = {
        "available": False,
        "plannable": classification != "unavailable",
        "state": "unavailable" if classification == "unavailable" else "requires_verification",
        "resolution": classification,
        "proof_status": resolved_state.proof_status,
        "sampler": {
            "required": classification == "exact_execution_fork",
            "status": "not_required" if classification == "reconstructed_replay" else "requires_control_proof",
        },
    }
    if classification == "unavailable":
        base["reason_code"] = resolved_state.diagnostics.get("reason_code", "state_unavailable")
        base["reason"] = resolved_state.diagnostics.get("message", "the resolved state is unavailable")
        return base

    contract = resolved_state.execution.generation_contract
    if not isinstance(contract, Mapping):
        # A recorded sampled run can have enough metadata to identify its regime while still
        # lacking the complete immutable sampler contract. Do not let the historical greedy
        # default turn that into an apparently deterministic operation. Runs with no decode
        # regime at all retain the legacy greedy planning default.
        if resolved_state.execution.generation_contract_reason == "sampled_replay_not_proven":
            base.update({
                "state": "unavailable", "plannable": False,
                "reason_code": STOCHASTIC_EXECUTION_UNBOUND,
                "reason": STOCHASTIC_EXECUTION_UNBOUND_MESSAGE,
                "sampler": {"required": True, "mode": "sample", "status": "unbound"},
            })
            return base
    recorded_mode = contract.get("decode_mode") if isinstance(contract, Mapping) else None
    mode = decode_mode or recorded_mode or "greedy"
    if mode not in {"greedy", "sample"}:
        base.update({
            "state": "unavailable", "plannable": False,
            "reason_code": "generation_contract_incomplete",
            "reason": "the recorded generation contract does not identify greedy or sampled decoding",
        })
        return base
    base["sampler"] = {
        "required": mode == "sample",
        "mode": mode,
        "status": (
            "unbound" if mode == "sample"
            else "not_required" if classification == "reconstructed_replay"
            else "not_required" if mode == "greedy"
            else "requires_control_proof"
        ),
    }
    if mode == "sample":
        base.update({
            "state": "unavailable", "plannable": False,
            "reason_code": STOCHASTIC_EXECUTION_UNBOUND,
            "reason": STOCHASTIC_EXECUTION_UNBOUND_MESSAGE,
        })
        return base

    if operation == "continue":
        if classification == "reconstructed_replay":
            base.update({"available": True, "state": "available", "proof_status": "not_applicable"})
        else:
            base.update({
                "reason_code": "exact_control_required",
                "reason": "exact Continue requires a matching unchanged control before fidelity is confirmed",
            })
        return base

    # ForceToken owns the replacement token, while StateRef owns its location.  The missing-input
    # branches are intentionally explicit so a capability client cannot mistake state availability
    # for intervention readiness.
    if classification == "exact_execution_fork" and token_id is None:
        base.update({
            "state": "requires_input", "plannable": False,
            "required_inputs": ["token_id"],
            "reason_code": "force_token_id_required",
            "reason": "exact ForceToken execution requires a numeric token_id",
        })
        return base
    if classification == "reconstructed_replay" and token_piece is None:
        base.update({
            "available": False, "state": "requires_input", "plannable": False,
            "required_inputs": ["token_piece"],
            "reason_code": "reconstruction_token_piece_unavailable",
            "reason": "reconstructed ForceToken execution requires token_piece",
        })
        return base
    if classification == "reconstructed_replay":
        base.update({"available": True, "state": "available", "proof_status": "not_applicable"})
    else:
        base.update({
            "reason_code": "exact_control_required",
            "reason": "exact ForceToken execution requires a matching unchanged control before fidelity is confirmed",
        })
    return base


def resolve_state(state_ref: StateRef, *, run: Mapping[str, Any] | None = None,
                  parent_run: Mapping[str, Any] | None = None,
                  policy: str = "exact_preferred", checkpoint: Mapping[str, Any] | None = None,
                  worker_identity: Mapping[str, Any] | None = None,
                  runtime_identity: Mapping[str, Any] | None = None) -> ResolvedState:
    """Resolve one logical state without contacting a model, worker, or checkpoint store."""
    if not isinstance(state_ref, StateRef):
        raise TypeError("resolve_state requires a StateRef")
    if policy not in RESOLUTION_POLICIES:
        raise StateRefError(f"unsupported state resolution policy: {policy!r}")
    source = run if run is not None else parent_run
    if not isinstance(source, Mapping):
        return _unavailable(state_ref, "parent_run_required", "state resolution requires the current parent run")
    try:
        state_ref.assert_current(source)
    except StateRefError as exc:
        return _unavailable(state_ref, "stale_parent_execution", str(exc))

    from clozn.replay.execution_fork import plan_execution_fork, recorded_fork_prerequisites
    prerequisites = recorded_fork_prerequisites(source)
    if not prerequisites["token_alignment_available"]:
        return _unavailable(state_ref, "recorded_token_history_unavailable",
                            "recorded token IDs and pieces are not aligned")
    if policy == "reconstructed_only" or (policy == "exact_preferred" and checkpoint is None):
        if not prerequisites["final_prompt_available"]:
            return _unavailable(state_ref, "reconstruction_prompt_unavailable",
                                "the parent has no exact rendered prompt for reconstruction")
        # Preserve exact parity with the trusted planner when live identities are supplied.  A
        # model-free caller such as GET /capabilities is allowed to plan reconstructed replay from
        # recorded evidence alone, so it uses the explicit model-free projection instead of
        # manufacturing a worker identity.
        if runtime_identity is not None or worker_identity is not None:
            plan = plan_execution_fork(
                source, {"position": state_ref.position.index, "change": {"type": "none"}},
                checkpoint=None, worker_identity=worker_identity, runtime_identity=runtime_identity,
            )
            if plan.get("classification") != "reconstructed_replay":
                reason = (plan.get("reasons") or [{}])[0]
                return _unavailable(
                    state_ref, str(reason.get("code") or "reconstruction_unavailable"),
                    str(reason.get("message") or "reconstructed replay is unavailable"), plan=plan,
                )
        else:
            plan = _model_free_reconstruction_plan(state_ref, source)
        return _reconstructed(state_ref, plan=plan, reason_code="checkpoint_not_supplied")
    if checkpoint is None:
        return _unavailable(state_ref, "checkpoint_missing", "exact resolution requires a checkpoint reference")

    plan = plan_execution_fork(
        source,
        {"position": state_ref.position.index, "change": {"type": "none"}},
        checkpoint=checkpoint, worker_identity=worker_identity, runtime_identity=runtime_identity,
    )
    if plan.get("classification") != "exact_execution_fork":
        reason = (plan.get("reasons") or [{}])[0]
        return _unavailable(
            state_ref, str(reason.get("code") or "exact_state_unavailable"),
            str(reason.get("message") or "exact execution state is unavailable"), plan=plan,
        )
    realization = {
        "regime": plan.get("exactness", {}).get("regime", "exact_execution_fork"),
        "source": plan.get("exactness", {}).get("source", "checkpoint"),
        "checkpoint_reference": deepcopy(plan.get("checkpoint_reference")),
        "runtime_identity": deepcopy(plan.get("identity", {}).get("selected_runtime")),
        "worker_identity": deepcopy(plan.get("identity", {}).get("selected_worker")),
    }
    return ResolvedState(
        state_ref=state_ref, classification="exact_execution_fork", proof_status="planned",
        realization=realization, diagnostics={
            "reason_code": "exact_preconditions_met",
            "message": "exact execution is planned; unchanged control must still confirm fidelity",
        }, plan=plan,
    )


__all__ = [
    "AnswerTokenBoundary", "RecordedAnswerBoundary", "enumerate_answer_boundaries",
    "list_answer_token_boundaries",
    "RESOLUTION_POLICIES", "RESOLVED_STATE_SCHEMA_VERSION",
    "ResolvedState", "STATE_CLASSIFICATIONS", "STATE_REF_SCHEMA_VERSION", "StateRef",
    "StateRefError", "STOCHASTIC_EXECUTION_UNBOUND", "operation_readiness", "resolve_state",
]
