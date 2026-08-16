"""Execution seam for delete-source exact-reference observations.

Only this adapter knows how to turn a typed ``DeleteSource`` declaration into
messages.  It delegates canonical source proof to the existing strict receipt
resolver and delegates exact generation/classification to the existing scalar
``probe_reference_match`` substrate method.  No arm is persisted here.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

from clozn.receipts.rederive import with_arm_conditions
from clozn.replay.span_bridge import (
    ContextReceiptSourceResolutionError,
    resolve_context_receipt_source_set,
)
from clozn.runs.answer_preservation import assess_exact_eligibility
from clozn.runs.context_units import protected_message_indices

from .evaluators import ExactReferenceMatch
from .interventions import DeleteSource
from .kernel import Experiment
from .observations import Observation, execution_observation_identity
from .state import ExecutionState, digest


class ExecutionAdapterError(ValueError):
    """The adapter cannot faithfully bind or execute an experiment arm."""


class ExecutionStateStaleError(ExecutionAdapterError):
    """The reloaded run no longer matches the experiment's base state."""


def _diagnostics(raw: Mapping[str, Any]) -> dict[str, Any]:
    allowed = (
        "reason", "error", "termination_match", "finish_reason",
        "termination", "expected_token_id", "actual_token_id",
    )
    result = {key: deepcopy(raw[key]) for key in allowed if key in raw}
    generated = raw.get("generated_token_ids")
    if isinstance(generated, list) and all(isinstance(item, int) and not isinstance(item, bool)
                                           for item in generated):
        result["generated_token_count"] = len(generated)
        result["generated_token_ids_sha256"] = digest(generated)
    return result


def _identity_kwargs(state: ExecutionState, evaluator: Any, intervention: DeleteSource | None) -> dict[str, Any]:
    identity = execution_observation_identity(state, evaluator, intervention)
    key = identity["observation_key"]
    return {
        "observation_id": identity["observation_id"],
        "observation_key_sha256": identity["observation_key_sha256"],
        "observation_key": key,
        "run_id": state.run_id,
        "base_execution_fingerprint": state.execution_fingerprint,
        "evaluator": key["evaluator"],
        "condition": key["condition"],
        "contract": key["contract"],
    }


def _observation_from_probe(state: ExecutionState, evaluator: ExactReferenceMatch,
                            intervention: DeleteSource | None, raw: Mapping[str, Any], *,
                            provenance: Mapping[str, Any]) -> Observation:
    status = raw.get("status")
    if status == "matched":
        observation_status = "exact_preserved"
        proof_grade, trusted = "trusted", True
    elif status == "diverged":
        observation_status = "diverged"
        proof_grade, trusted = "trusted", True
    elif status == "unavailable":
        observation_status = "unavailable"
        proof_grade, trusted = "unavailable", False
    else:
        observation_status = "failed"
        proof_grade, trusted = "unavailable", False
    return Observation(
        **_identity_kwargs(state, evaluator, intervention),
        status=observation_status,
        matched_token_count=raw.get("matched_token_count"),
        first_divergence_index=raw.get("first_divergence_index"),
        divergence_kind=raw.get("divergence_kind"),
        execution_provenance=provenance,
        proof_grade=proof_grade,
        trusted=trusted,
        diagnostics=_diagnostics(raw),
    )


def resolve_delete_source(run: Mapping[str, Any], intervention: DeleteSource) -> dict[str, Any]:
    """Resolve one typed delete declaration through the canonical receipt bridge."""
    if not isinstance(intervention, DeleteSource):
        raise ExecutionAdapterError("Batch 1 execution supports DeleteSource only")
    try:
        resolved = resolve_context_receipt_source_set(dict(run), list(intervention.source_ids))
    except ContextReceiptSourceResolutionError:
        raise
    protected = protected_message_indices(run.get("messages"))
    selected_ranges = resolved.get("exact_removed_ranges") or []
    protected_ranges = [
        item for item in selected_ranges
        if isinstance(item, Mapping) and item.get("message_index") in protected
    ]
    if protected_ranges:
        raise ExecutionAdapterError(
            "the current request and following message suffix are protected from source deletion"
        )
    return resolved


class DeleteSourceExactReferenceAdapter:
    """Run control/delete arms without invoking the legacy experiment API."""

    def __init__(self, substrate: Any, *, run: Mapping[str, Any] | None = None,
                 run_loader: Callable[[str], Mapping[str, Any] | None] | None = None):
        if substrate is None:
            raise ValueError("DeleteSourceExactReferenceAdapter requires a substrate")
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
            raise ExecutionAdapterError("the base run could not be loaded")
        current = ExecutionState.from_run(run)
        if current.execution_fingerprint != state.execution_fingerprint:
            raise ExecutionStateStaleError("the base run no longer matches the experiment execution fingerprint")
        if current.context_receipt_identity.get("digest") != state.context_receipt_identity.get("digest"):
            raise ExecutionStateStaleError("the Context Receipt binding changed after experiment creation")
        return dict(run)

    def _eligibility(self, run: Mapping[str, Any], state: ExecutionState) -> dict[str, Any]:
        if state.generation_contract is None:
            return {"eligible": False, "reason": state.generation_contract_reason or "generation_contract_incomplete"}
        token_identity = state.recorded_answer_token_identity
        if not token_identity.get("token_ids_sha256"):
            return {"eligible": False, "reason": "missing_exact_recorded_token_ids"}
        # Runtime parity is checked by the trusted low-level eligibility oracle
        # when the substrate exposes its recorded/current runtime identity seam.
        if callable(getattr(self.substrate, "identity_meta", None)) and callable(
                getattr(self.substrate, "run_meta", None)):
            return assess_exact_eligibility(run, self.substrate)
        return {"eligible": False, "reason": "runtime_identity_unavailable"}

    def _run_probe(self, state: ExecutionState, run: Mapping[str, Any], *, arm_id: str | None,
                   intervention: DeleteSource | None, evaluator: ExactReferenceMatch) -> Observation:
        eligibility = self._eligibility(run, state)
        provenance: dict[str, Any] = {
            "adapter": "delete_source_exact_reference",
            "resolver": "resolve_context_receipt_source_set",
            "evaluator": "scalar_probe_reference_match",
            "proof_grade": "trusted",
        }
        if not eligibility.get("eligible"):
            return Observation(
                **_identity_kwargs(state, evaluator, intervention),
                status="unavailable",
                execution_provenance=provenance,
                diagnostics={"reason": eligibility.get("reason", "exact_execution_unavailable")},
            )

        conditions = with_arm_conditions(dict(run))
        messages = list(conditions.get("messages") or [])
        block = conditions.get("block")
        if intervention is not None:
            try:
                resolved = resolve_delete_source(run, intervention)
            except (ContextReceiptSourceResolutionError, ExecutionAdapterError) as exc:
                return Observation(
                    **_identity_kwargs(state, evaluator, intervention),
                    status="unavailable",
                    execution_provenance=provenance,
                    diagnostics={"reason": "intervention_unavailable", "error": str(exc)},
                )
            messages = list(resolved.get("messages") or [])
            if resolved.get("basis") == "assembled_messages":
                block = None
            provenance.update({
                "source_basis": resolved.get("basis"),
                "basis_digest": resolved.get("basis_digest"),
                "intervened_context_digest": resolved.get("intervened_context_digest"),
                "removed_source_ids": list(intervention.source_ids),
            })
        else:
            provenance["source_basis"] = conditions.get("block_source")

        probe = getattr(self.substrate, "probe_reference_match", None)
        if not callable(probe):
            return Observation(
                **_identity_kwargs(state, evaluator, intervention),
                status="unavailable",
                execution_provenance=provenance,
                diagnostics={"reason": "exact_probe_unsupported"},
            )
        try:
            state_contract = state.to_dict().get("generation_contract")
            raw = probe(
                messages,
                list(conditions.get("continuation_ids") or []),
                generation_contract=state_contract,
                explicit_conditions={
                    "block": block,
                    "steer_strengths": deepcopy(conditions.get("steer_strengths") or {}),
                },
            )
        except Exception as exc:
            return Observation(
                **_identity_kwargs(state, evaluator, intervention),
                status="failed",
                execution_provenance=provenance,
                proof_grade="unavailable",
                diagnostics={"reason": "probe_failed", "error": str(exc)},
            )
        if not isinstance(raw, Mapping):
            return Observation(
                **_identity_kwargs(state, evaluator, intervention),
                status="failed",
                execution_provenance=provenance,
                diagnostics={"reason": "probe_malformed"},
            )
        return _observation_from_probe(state, evaluator, intervention, raw, provenance=provenance)

    def execute(self, state: ExecutionState, intervention: DeleteSource | None = None, *,
                evaluator: ExactReferenceMatch | None = None, arm_id: str | None = None) -> Observation:
        if not isinstance(state, ExecutionState):
            raise TypeError("execution state must be an ExecutionState")
        evaluator = evaluator or ExactReferenceMatch()
        if not isinstance(evaluator, ExactReferenceMatch):
            raise TypeError("Batch 1 execution supports ExactReferenceMatch only")
        try:
            run = self._validated_run(state)
        except ExecutionStateStaleError:
            raise
        except ExecutionAdapterError as exc:
            return Observation(
                **_identity_kwargs(state, evaluator, intervention),
                status="unavailable",
                execution_provenance={"adapter": "delete_source_exact_reference"},
                diagnostics={"reason": "base_run_unavailable", "error": str(exc)},
            )
        return self._run_probe(state, run, arm_id=arm_id, intervention=intervention, evaluator=evaluator)

    def execute_control(self, state: ExecutionState, *, evaluator: ExactReferenceMatch | None = None) -> Observation:
        return self.execute(state, None, evaluator=evaluator, arm_id="control")


# Short explicit names for callers constructing the Batch 1 seam.
ExecutionAdapter = DeleteSourceExactReferenceAdapter
ExactReferenceMatchAdapter = DeleteSourceExactReferenceAdapter


__all__ = [
    "DeleteSourceExactReferenceAdapter", "ExecutionAdapter", "ExactReferenceMatchAdapter",
    "ExecutionAdapterError", "ExecutionStateStaleError", "resolve_delete_source",
]
