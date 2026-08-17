"""Trusted teacher-forced scoring for the new experiment kernel.

This adapter is deliberately parallel to Batch 1's exact-reference adapter at
the evaluator boundary, but it reuses its ``resolve_delete_source`` helper for
the intervention itself.  The only model call is the existing
``receipts.rederive.score_arm`` seam, with recorded token IDs as the primary
continuation representation.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
import math
from typing import Any

from clozn.receipts.rederive import score_arm, with_arm_conditions
from clozn.replay.execution_fork import _runtime_projection, parent_runtime_projection

from .batch import ArmExecutionOutcome, ArmExecutionRequest, BatchExecutionResult
from .evaluators import ScoreRecordedContinuation
from .execution import ExecutionAdapterError, ExecutionStateStaleError, _identity_kwargs, resolve_delete_source
from .interventions import DeleteSource
from .multi_arm import score_tokens_many
from .observations import TokenScoreObservation
from .selections import _recorded_answer_tokens
from .state import ExecutionState, digest


class ScoreExecutionError(ExecutionAdapterError):
    """The score adapter cannot prove or execute a recorded-token score."""


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    method = getattr(cancel, "is_set", None)
    if callable(method):
        return bool(method())
    return bool(cancel)


def _runtime_binding(run: Mapping[str, Any], substrate: Any) -> tuple[dict[str, Any] | None, str | None]:
    try:
        recorded = parent_runtime_projection(run)
    except Exception:
        recorded = None
    identity_fn = getattr(substrate, "identity_meta", None)
    meta_fn = getattr(substrate, "run_meta", None)
    if not callable(identity_fn) or not callable(meta_fn):
        return None, "runtime_identity_unavailable"
    try:
        current_raw = {
            "id": run.get("id", "current"),
            "model": run.get("model"),
            "identity": dict(identity_fn() or {}),
            "meta": dict(meta_fn() or {}),
        }
        current = _runtime_projection(current_raw.get("identity"), run_meta=current_raw.get("meta"))
    except Exception:
        current = None
    if recorded is None or current is None:
        return None, "runtime_identity_unavailable"
    if current != recorded:
        reason = "template_mismatch" if current.get("template_fingerprint") != recorded.get("template_fingerprint") else "runtime_identity_mismatch"
        return None, reason
    return recorded, None


def _score_observation(*, arm_id: str | None, run: Mapping[str, Any], state: ExecutionState,
                       evaluator: ScoreRecordedContinuation, intervention: DeleteSource | None,
                       conditions: Mapping[str, Any], raw_tokens: Any,
                       provenance: Mapping[str, Any], diagnostics: Mapping[str, Any] | None = None) -> TokenScoreObservation:
    identity = _identity_kwargs(state, evaluator, intervention)
    try:
        expected_ids, expected_pieces, response = _recorded_answer_tokens(run)
    except Exception as exc:
        return TokenScoreObservation(
            **identity, status="unavailable", evaluator_provenance=provenance,
            score_basis={"runtime_binding": _thaw(state.model_runtime_identity)},
            execution_provenance=provenance,
            diagnostics={"reason": "recorded_continuation_unavailable", "error": str(exc)},
        )
    if not isinstance(raw_tokens, list) or len(raw_tokens) != len(expected_ids):
        return TokenScoreObservation(
            **identity, status="unavailable", evaluator_provenance=provenance,
            score_basis={"runtime_binding": _thaw(state.model_runtime_identity)},
            execution_provenance=provenance,
            diagnostics={"reason": "score_token_count_mismatch", "expected": len(expected_ids),
                         "actual": len(raw_tokens) if isinstance(raw_tokens, list) else None},
        )
    logprobs: list[float] = []
    for index, (expected_id, expected_piece, token) in enumerate(zip(expected_ids, expected_pieces, raw_tokens)):
        if not isinstance(token, Mapping):
            return TokenScoreObservation(
                **identity, status="unavailable", evaluator_provenance=provenance,
                score_basis={"runtime_binding": _thaw(state.model_runtime_identity)},
                execution_provenance=provenance,
                diagnostics={"reason": "malformed_score_token", "index": index},
            )
        actual_id = token.get("id", token.get("token_id"))
        actual_piece = token.get("piece")
        logprob = token.get("logprob")
        if actual_id != expected_id or actual_piece != expected_piece:
            return TokenScoreObservation(
                **identity, status="unavailable", evaluator_provenance=provenance,
                score_basis={"runtime_binding": _thaw(state.model_runtime_identity)},
                execution_provenance=provenance,
                diagnostics={"reason": "score_token_alignment_mismatch", "index": index,
                             "expected_token_id": expected_id, "actual_token_id": actual_id},
            )
        if isinstance(logprob, bool) or not isinstance(logprob, (int, float)) or not math.isfinite(float(logprob)):
            return TokenScoreObservation(
                **identity, status="unavailable", evaluator_provenance=provenance,
                score_basis={"runtime_binding": _thaw(state.model_runtime_identity)},
                execution_provenance=provenance,
                diagnostics={"reason": "score_logprob_unavailable", "index": index},
            )
        logprobs.append(float(logprob))
    spans: list[tuple[int, int]] = []
    cursor = 0
    for piece in expected_pieces:
        end = cursor + len(piece)
        spans.append((cursor, end))
        cursor = end
    basis = dict(provenance)
    basis.update({
        "runtime_binding": provenance.get("runtime_binding"),
        "recorded_answer_token_ids_sha256": digest(expected_ids),
        "recorded_answer_text_sha256": digest(response),
        "prompt_conditions_digest": digest({
            "messages": conditions.get("messages"), "block": conditions.get("block"),
            "steer_strengths": conditions.get("steer_strengths"),
        }),
        "continuation_basis": "recorded_token_ids",
    })
    return TokenScoreObservation(
        **identity, status="completed", recorded_token_ids=expected_ids,
        token_pieces=expected_pieces, token_spans=spans, token_logprobs=logprobs,
        total_continuation_logprob=sum(logprobs), evaluator_provenance=provenance,
        score_basis=basis, execution_provenance=provenance,
        proof_grade="trusted", trusted=True, diagnostics=diagnostics,
    )


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_thaw(item) for item in value]
    return value


class DeleteSourceRecordedContinuationScoreAdapter:
    """Execute control and direct ``DeleteSource`` score arms."""

    def __init__(self, substrate: Any, *, run: Mapping[str, Any] | None = None,
                 run_loader: Callable[[str], Mapping[str, Any] | None] | None = None):
        if substrate is None:
            raise ValueError("DeleteSourceRecordedContinuationScoreAdapter requires a substrate")
        if run is not None and not isinstance(run, Mapping):
            raise TypeError("run must be a run mapping when supplied")
        if run_loader is not None and not callable(run_loader):
            raise TypeError("run_loader must be callable")
        self.substrate = substrate
        self._run = deepcopy(dict(run)) if isinstance(run, Mapping) else None
        self._run_loader = run_loader

    def load_run(self, state: ExecutionState) -> Mapping[str, Any] | None:
        if self._run_loader is not None:
            return self._run_loader(state.run_id)
        if self._run is not None and self._run.get("id") == state.run_id:
            return deepcopy(self._run)
        return None

    def _validated_run(self, state: ExecutionState) -> dict[str, Any]:
        run = self.load_run(state)
        if not isinstance(run, Mapping):
            raise ScoreExecutionError("the base run could not be loaded")
        current = ExecutionState.from_run(run)
        if current.execution_fingerprint != state.execution_fingerprint:
            raise ExecutionStateStaleError("the base run no longer matches the experiment execution fingerprint")
        if current.context_receipt_identity.get("digest") != state.context_receipt_identity.get("digest"):
            raise ExecutionStateStaleError("the Context Receipt binding changed after experiment creation")
        return dict(run)

    def execute(self, state: ExecutionState, intervention: DeleteSource | None = None, *,
                evaluator: ScoreRecordedContinuation | None = None,
                arm_id: str | None = None) -> TokenScoreObservation:
        if not isinstance(state, ExecutionState):
            raise TypeError("execution state must be an ExecutionState")
        evaluator = evaluator or ScoreRecordedContinuation()
        if not isinstance(evaluator, ScoreRecordedContinuation):
            raise TypeError("score execution supports ScoreRecordedContinuation only")
        identity = _identity_kwargs(state, evaluator, intervention)
        try:
            run = self._validated_run(state)
        except ExecutionStateStaleError:
            raise
        except ExecutionAdapterError as exc:
            return TokenScoreObservation(
                **identity, status="unavailable",
                evaluator_provenance={"evaluator": "score_recorded_continuation"},
                execution_provenance={"adapter": "delete_source_recorded_continuation_score"},
                diagnostics={"reason": "base_run_unavailable", "error": str(exc)},
            )
        try:
            expected_ids, _pieces, _response = _recorded_answer_tokens(run)
        except Exception as exc:
            return TokenScoreObservation(
                **identity, status="unavailable",
                evaluator_provenance={"evaluator": "score_recorded_continuation"},
                execution_provenance={"adapter": "delete_source_recorded_continuation_score"},
                diagnostics={"reason": "recorded_continuation_unavailable", "error": str(exc)},
            )
        if not expected_ids or state.recorded_answer_token_identity.get("token_ids_sha256") != digest(expected_ids):
            return TokenScoreObservation(
                **identity, status="unavailable",
                evaluator_provenance={"evaluator": "score_recorded_continuation"},
                execution_provenance={"adapter": "delete_source_recorded_continuation_score"},
                diagnostics={"reason": "recorded_continuation_identity_mismatch"},
            )
        runtime, runtime_reason = _runtime_binding(run, self.substrate)
        base_provenance: dict[str, Any] = {
            "adapter": "delete_source_recorded_continuation_score",
            "resolver": "resolve_context_receipt_source_set",
            "scorer": "clozn.receipts.rederive.score_arm",
            "evaluator": "score_recorded_continuation",
            "method": "teacher_forced_score_tokens",
            "proof_grade": "trusted",
            "runtime_binding": runtime,
        }
        if runtime_reason:
            return TokenScoreObservation(
                **identity, status="unavailable", evaluator_provenance=base_provenance,
                score_basis={"runtime_binding": runtime}, execution_provenance=base_provenance,
                diagnostics={"reason": runtime_reason},
            )
        conditions = with_arm_conditions(dict(run))
        if conditions.get("continuation_ids") != expected_ids:
            return TokenScoreObservation(
                **identity, status="unavailable", evaluator_provenance=base_provenance,
                score_basis={"runtime_binding": runtime}, execution_provenance=base_provenance,
                diagnostics={"reason": "recorded_continuation_identity_mismatch"},
            )
        messages = list(conditions.get("messages") or [])
        block = conditions.get("block")
        provenance = dict(base_provenance)
        if intervention is not None:
            if not isinstance(intervention, DeleteSource):
                raise ScoreExecutionError("score execution supports DeleteSource only")
            try:
                resolved = resolve_delete_source(run, intervention)
            except Exception as exc:
                return TokenScoreObservation(
                    **identity, status="unavailable", evaluator_provenance=provenance,
                    score_basis={"runtime_binding": runtime}, execution_provenance=provenance,
                    diagnostics={"reason": "intervention_unavailable", "error": str(exc)},
                )
            messages = list(resolved.get("messages") or [])
            if resolved.get("basis") == "assembled_messages":
                block = None
            provenance.update({
                "source_basis": resolved.get("basis"),
                "basis_digest": resolved.get("basis_digest"),
                "intervened_context_digest": resolved.get("intervened_context_digest"),
                "canonical_source_ids": list(resolved.get("canonical_source_ids") or intervention.source_ids),
                "removed_ranges": deepcopy(resolved.get("exact_removed_ranges") or []),
                "removed_source_ids": list(intervention.source_ids),
            })
        score_conditions = dict(conditions)
        score_conditions["messages"] = messages
        score_conditions["block"] = block
        if not callable(getattr(self.substrate, "score_tokens", None)):
            return TokenScoreObservation(
                **identity, status="unavailable", evaluator_provenance=provenance,
                score_basis={"runtime_binding": runtime}, execution_provenance=provenance,
                diagnostics={"reason": "scoring_substrate_unavailable"},
            )
        tokens, ok = score_arm(
            self.substrate, score_conditions, messages=messages, block=block,
            steer_strengths=conditions.get("steer_strengths"), topk=0,
        )
        if not ok:
            return TokenScoreObservation(
                **identity, status="failed", evaluator_provenance=provenance,
                score_basis={"runtime_binding": runtime}, execution_provenance=provenance,
                diagnostics={"reason": "score_failed"},
            )
        return _score_observation(
            arm_id=arm_id, run=run, state=state, evaluator=evaluator, intervention=intervention,
            conditions=score_conditions,
            raw_tokens=tokens, provenance=provenance,
        )

    def execute_control(self, state: ExecutionState, *,
                        evaluator: ScoreRecordedContinuation | None = None) -> TokenScoreObservation:
        return self.execute(state, None, evaluator=evaluator, arm_id="control")

    def execute_many(self, requests: tuple[ArmExecutionRequest, ...] | list[ArmExecutionRequest], *,
                     cancel: Any = None) -> BatchExecutionResult:
        """Batch the substrate score calls while retaining one host validator."""
        requests = tuple(requests)
        if not requests:
            return BatchExecutionResult((), {"execution_strategy": "none", "batch_count": 0})
        first = requests[0]
        if not isinstance(first.state, ExecutionState) or not isinstance(first.evaluator, ScoreRecordedContinuation):
            raise TypeError("score batch requests require ExecutionState and ScoreRecordedContinuation")
        try:
            run = self._validated_run(first.state)
        except ExecutionAdapterError as exc:
            return BatchExecutionResult(tuple(
                ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="not_executed", state="failed",
                    diagnostics={"reason": "base_run_unavailable", "error": str(exc)},
                ) for request in requests
            ), {"execution_strategy": "preflight", "batch_count": 0})
        prepared: list[tuple[ArmExecutionRequest, dict[str, Any]]] = []
        outcomes: list[ArmExecutionOutcome] = []
        for request in requests:
            if request.state != first.state or not isinstance(request.evaluator, ScoreRecordedContinuation):
                raise ScoreExecutionError("score batch requests must share one execution state/evaluator")
            if _cancelled(cancel):
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="not_executed", state="cancelled",
                    diagnostics={"reason": "cancelled_before_dispatch"},
                ))
                continue
            try:
                expected_ids, _pieces, _response = _recorded_answer_tokens(run)
            except Exception as exc:
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="not_executed", state="failed",
                    diagnostics={"reason": "recorded_continuation_unavailable", "error": str(exc)},
                ))
                continue
            identity = _identity_kwargs(request.state, request.evaluator, request.intervention)
            runtime, runtime_reason = _runtime_binding(run, self.substrate)
            provenance: dict[str, Any] = {
                "adapter": "delete_source_recorded_continuation_score",
                "resolver": "resolve_context_receipt_source_set",
                "scorer": "clozn.receipts.rederive.score_arm",
                "evaluator": "score_recorded_continuation",
                "method": "teacher_forced_score_tokens",
                "proof_grade": "trusted",
                "runtime_binding": runtime,
            }
            if runtime_reason or not expected_ids or request.state.recorded_answer_token_identity.get("token_ids_sha256") != digest(expected_ids):
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id,
                    observation=TokenScoreObservation(
                        **identity, status="unavailable", evaluator_provenance=provenance,
                        score_basis={"runtime_binding": runtime}, execution_provenance=provenance,
                        diagnostics={"reason": runtime_reason or "recorded_continuation_identity_mismatch"},
                    ),
                    execution_disposition="not_executed", state="failed",
                ))
                continue
            conditions = with_arm_conditions(dict(run))
            if conditions.get("continuation_ids") != expected_ids:
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id,
                    observation=TokenScoreObservation(
                        **identity, status="unavailable", evaluator_provenance=provenance,
                        score_basis={"runtime_binding": runtime}, execution_provenance=provenance,
                        diagnostics={"reason": "recorded_continuation_identity_mismatch"},
                    ),
                    execution_disposition="not_executed", state="failed",
                ))
                continue
            messages = list(conditions.get("messages") or [])
            block = conditions.get("block")
            if request.intervention is not None:
                if not isinstance(request.intervention, DeleteSource):
                    raise ScoreExecutionError("score execution supports DeleteSource only")
                try:
                    resolved = resolve_delete_source(run, request.intervention)
                except Exception as exc:
                    outcomes.append(ArmExecutionOutcome(
                        arm_id=request.arm_id,
                        observation=TokenScoreObservation(
                            **identity, status="unavailable", evaluator_provenance=provenance,
                            score_basis={"runtime_binding": runtime}, execution_provenance=provenance,
                            diagnostics={"reason": "intervention_unavailable", "error": str(exc)},
                        ),
                        execution_disposition="not_executed", state="failed",
                    ))
                    continue
                messages = list(resolved.get("messages") or [])
                if resolved.get("basis") == "assembled_messages":
                    block = None
                provenance.update({
                    "source_basis": resolved.get("basis"),
                    "basis_digest": resolved.get("basis_digest"),
                    "intervened_context_digest": resolved.get("intervened_context_digest"),
                    "canonical_source_ids": list(resolved.get("canonical_source_ids") or request.intervention.source_ids),
                    "removed_ranges": deepcopy(resolved.get("exact_removed_ranges") or []),
                    "removed_source_ids": list(request.intervention.source_ids),
                })
            prepared.append((request, {
                "messages": messages,
                "continuation_ids": list(expected_ids),
                "block": block,
                "steer_strengths": deepcopy(conditions.get("steer_strengths") or {}),
                "provenance": provenance,
            }))
        if not prepared:
            return BatchExecutionResult(tuple(outcomes), {"execution_strategy": "preflight", "batch_count": 0})
        if not callable(getattr(self.substrate, "score_tokens", None)):
            raise ScoreExecutionError("scoring_substrate_unavailable")
        strategy = "native_many" if bool(getattr(self.substrate, "score_tokens_many_proof_grade", False)) else "scalar"
        arms = [{key: deepcopy(value) for key, value in payload.items() if key != "provenance"}
                for _request, payload in prepared]
        raw_rows = list(score_tokens_many(self.substrate, arms, cancel=cancel, proof_grade=True))
        if len(raw_rows) != len(prepared):
            raise ScoreExecutionError("score batch returned the wrong arm count")
        for (request, payload), raw_tokens in zip(prepared, raw_rows):
            observation = _score_observation(
                arm_id=request.arm_id, run=run, state=request.state, evaluator=request.evaluator,
                intervention=request.intervention, conditions=payload, raw_tokens=raw_tokens,
                provenance=payload["provenance"],
            )
            outcomes.append(ArmExecutionOutcome(
                arm_id=request.arm_id, observation=observation,
                execution_disposition="executed",
                state="completed" if observation.completed else "failed",
                diagnostics={"execution_strategy": strategy},
            ))
        return BatchExecutionResult(tuple(outcomes), {
            "execution_strategy": strategy, "batch_count": 1, "batch_size": len(prepared),
        })


ScoreRecordedContinuationAdapter = DeleteSourceRecordedContinuationScoreAdapter
DeleteSourceScoreAdapter = DeleteSourceRecordedContinuationScoreAdapter
RecordedContinuationScoreAdapter = DeleteSourceRecordedContinuationScoreAdapter


__all__ = [
    "DeleteSourceRecordedContinuationScoreAdapter", "DeleteSourceScoreAdapter",
    "RecordedContinuationScoreAdapter", "ScoreRecordedContinuationAdapter", "ScoreExecutionError",
]
