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
from .batch import ArmExecutionOutcome, ArmExecutionRequest, BatchExecutionResult
from .effective_prompt import EffectivePromptUnavailable, resolve_effective_prompt
from .interventions import DeleteSource
from .multi_arm import probe_reference_match_many
from .observations import Observation, execution_observation_identity
from .shared_parent import SharedParentSessionClient, SharedParentSessionError, condition_candidate_id
from .state import ExecutionState, digest


class ExecutionAdapterError(ValueError):
    """The adapter cannot faithfully bind or execute an experiment arm."""


class ExecutionStateStaleError(ExecutionAdapterError):
    """The reloaded run no longer matches the experiment's base state."""


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    method = getattr(cancel, "is_set", None)
    if callable(method):
        return bool(method())
    return bool(cancel)


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
                 run_loader: Callable[[str], Mapping[str, Any] | None] | None = None,
                 engine: Any = None, execution_strategy: str = "auto"):
        if substrate is None:
            raise ValueError("DeleteSourceExactReferenceAdapter requires a substrate")
        if run is not None and not isinstance(run, Mapping):
            raise TypeError("run must be a run mapping when supplied")
        if run_loader is not None and not callable(run_loader):
            raise TypeError("run_loader must be callable")
        self.substrate = substrate
        self._run = deepcopy(dict(run)) if isinstance(run, Mapping) else None
        self._run_loader = run_loader
        if execution_strategy not in {"auto", "scalar", "native_many", "shared_parent"}:
            raise ValueError("unsupported exact execution strategy")
        self.engine = engine or getattr(substrate, "engine", None)
        self.execution_strategy = execution_strategy
        self._shared_parent: SharedParentSessionClient | None = None
        self._shared_parent_disabled = False
        self._shared_parent_children: dict[str, str] = {}
        self._optimization_diagnostics: list[dict[str, Any]] = []

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

    def _prepare_probe(self, state: ExecutionState, run: Mapping[str, Any], *,
                       intervention: DeleteSource | None, evaluator: ExactReferenceMatch) -> tuple[dict[str, Any] | None, Observation | None]:
        """Prepare one exact probe without dispatching model work."""
        eligibility = self._eligibility(run, state)
        provenance: dict[str, Any] = {
            "adapter": "delete_source_exact_reference",
            "resolver": "resolve_context_receipt_source_set",
            "evaluator": "exact_reference_match",
            "method": "direct_generation",
            "proof_grade": "trusted",
        }
        if not eligibility.get("eligible"):
            return None, Observation(
                **_identity_kwargs(state, evaluator, intervention), status="unavailable",
                execution_provenance=provenance,
                diagnostics={"reason": eligibility.get("reason", "exact_execution_unavailable")},
            )
        try:
            prompt = resolve_effective_prompt(run, intervention)
        except EffectivePromptUnavailable as exc:
            return None, Observation(
                **_identity_kwargs(state, evaluator, intervention), status="unavailable",
                execution_provenance=provenance,
                diagnostics={"reason": exc.reason, "error": str(exc)},
            )
        conditions = with_arm_conditions(dict(run))
        provenance.update({"source_basis": prompt.basis})
        if intervention is not None:
            provenance.update({
                "basis_digest": prompt.basis_digest,
                "intervened_context_digest": prompt.intervened_context_digest,
                "removed_source_ids": list(intervention.source_ids),
            })
        if not callable(getattr(self.substrate, "probe_reference_match", None)):
            return None, Observation(
                **_identity_kwargs(state, evaluator, intervention), status="unavailable",
                execution_provenance=provenance,
                diagnostics={"reason": "exact_probe_unsupported"},
            )
        return {
            "messages": prompt.worker_messages(),
            "reference_token_ids": list(conditions.get("continuation_ids") or []),
            "generation_contract": state.to_dict().get("generation_contract"),
            "explicit_conditions": {
                "block": prompt.block,
                "steer_strengths": deepcopy(conditions.get("steer_strengths") or {}),
            },
            "provenance": provenance,
        }, None

    def _run_probe(self, state: ExecutionState, run: Mapping[str, Any], *, arm_id: str | None,
                   intervention: DeleteSource | None, evaluator: ExactReferenceMatch) -> Observation:
        prepared, unavailable = self._prepare_probe(
            state, run, intervention=intervention, evaluator=evaluator,
        )
        if unavailable is not None:
            return unavailable
        assert prepared is not None
        provenance = dict(prepared["provenance"])
        probe = getattr(self.substrate, "probe_reference_match", None)
        try:
            raw = probe(
                prepared["messages"], prepared["reference_token_ids"],
                generation_contract=prepared["generation_contract"],
                explicit_conditions=prepared["explicit_conditions"],
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

    def execute_many(self, requests: tuple[ArmExecutionRequest, ...] | list[ArmExecutionRequest], *,
                     cancel: Any = None) -> BatchExecutionResult:
        """Execute exact-reference conditions through one generic batch seam.

        Native-many is selected only when the substrate explicitly advertises
        proof-grade semantic parity.  Otherwise this is the scalar reference
        strategy, still returned as arm-addressed outcomes.
        """
        requests = tuple(requests)
        if not requests:
            return BatchExecutionResult((), {"execution_strategy": "none", "batch_count": 0})
        first = requests[0]
        if not isinstance(first.state, ExecutionState) or not isinstance(first.evaluator, ExactReferenceMatch):
            raise TypeError("exact batch requests require ExecutionState and ExactReferenceMatch")
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
            if request.state != first.state or not isinstance(request.evaluator, ExactReferenceMatch):
                raise ExecutionAdapterError("exact batch requests must share one execution state/evaluator")
            if _cancelled(cancel):
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="not_executed", state="cancelled",
                    diagnostics={"reason": "cancelled_before_dispatch"},
                ))
                continue
            payload, unavailable = self._prepare_probe(
                request.state, run, intervention=request.intervention, evaluator=request.evaluator,
            )
            if unavailable is not None:
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, observation=unavailable,
                    execution_disposition="not_executed", state="failed",
                    diagnostics={"reason": "probe_preparation_unavailable"},
                ))
            else:
                assert payload is not None
                prepared.append((request, payload))
        if not prepared:
            return BatchExecutionResult(tuple(outcomes), {"execution_strategy": "preflight", "batch_count": 0})

        shared = self._execute_shared_parent(prepared, run=run, cancel=cancel)
        if shared is not None:
            outcomes.extend(shared.outcomes)
            return BatchExecutionResult(tuple(outcomes), {
                **dict(shared.diagnostics), "preflight_count": len(outcomes) - len(shared.outcomes),
            })

        use_native = bool(
            self.execution_strategy in {"auto", "native_many"}
            and
            getattr(self.substrate, "probe_reference_match_many_proof_grade", False)
            and callable(getattr(self.substrate, "probe_reference_match_many", None))
        )
        raw_by_arm: dict[str, Any] = {}
        strategy = "native_many" if use_native else "scalar"
        if use_native:
            native_arms = [{key: deepcopy(value) for key, value in payload.items() if key != "provenance"}
                           for _request, payload in prepared]
            raw_rows = probe_reference_match_many(
                self.substrate, native_arms, cancel=cancel, proof_grade=True,
            )
            if not isinstance(raw_rows, (list, tuple)) or len(raw_rows) != len(prepared):
                raise ExecutionAdapterError("proof-grade exact batch returned the wrong arm count")
            raw_by_arm = {request.arm_id: raw for (request, _payload), raw in zip(prepared, raw_rows)}
        else:
            for request, payload in prepared:
                if _cancelled(cancel):
                    outcomes.append(ArmExecutionOutcome(
                        arm_id=request.arm_id, execution_disposition="not_executed", state="cancelled",
                        diagnostics={"reason": "cancelled_before_dispatch"},
                    ))
                    continue
                try:
                    raw_by_arm[request.arm_id] = self.substrate.probe_reference_match(
                        payload["messages"], payload["reference_token_ids"],
                        generation_contract=payload["generation_contract"],
                        explicit_conditions=payload["explicit_conditions"],
                    )
                except Exception as exc:
                    raw_by_arm[request.arm_id] = {"status": "failed", "reason": "probe_failed", "error": str(exc)}
        for request, payload in prepared:
            raw = raw_by_arm.get(request.arm_id)
            if not isinstance(raw, Mapping):
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="executed", state="failed",
                    diagnostics={"reason": "probe_malformed", "execution_strategy": strategy},
                ))
                continue
            observation = _observation_from_probe(
                request.state, request.evaluator, request.intervention, raw,
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

    def _shared_parent_supported(self) -> bool:
        if self._shared_parent_disabled or self.execution_strategy == "scalar":
            return False
        if self.execution_strategy == "native_many":
            return False
        return bool(
            getattr(self.substrate, "shared_parent_exact_proof_grade", False)
            and self.engine is not None
            and all(callable(getattr(self.engine, name, None)) for name in (
                "reference_match_persistent_create", "reference_match_persistent_probe",
                "reference_match_persistent_promote", "reference_match_persistent_close",
            ))
        )

    def _render_parent_prompt(self, run: Mapping[str, Any]) -> str:
        prompt = resolve_effective_prompt(run, None)
        messages = prompt.rendered_messages()
        apply_info = getattr(self.engine, "apply_template_info", None)
        if callable(apply_info):
            value = apply_info(messages)
            rendered = value.get("prompt") if isinstance(value, Mapping) else None
        else:
            apply_template = getattr(self.engine, "apply_template", None)
            rendered = apply_template(messages) if callable(apply_template) else None
        if not isinstance(rendered, str) or not rendered:
            raise SharedParentSessionError("the canonical effective prompt could not be rendered", code="parent_prompt_unavailable")
        return rendered

    def _execute_shared_parent(self, prepared: list[tuple[ArmExecutionRequest, dict[str, Any]]], *,
                               run: Mapping[str, Any], cancel: Any = None) -> BatchExecutionResult | None:
        """Run a proof-grade shared-parent round, or return None before dispatch."""
        if not self._shared_parent_supported():
            return None
        # Shared sessions currently support the unchanged exact generation
        # contract only.  A varying block/steering condition must use the
        # ordinary exact adapter rather than being silently grouped.
        condition_values = [payload["explicit_conditions"] for _request, payload in prepared]
        if any((value.get("steer_strengths") or {}) or value.get("block") not in (None, "")
               for value in condition_values):
            return None
        probe_started = False
        try:
            if self._shared_parent is None:
                contract = prepared[0][0].state.to_dict().get("generation_contract") or {}
                token_ids = tuple(prepared[0][1]["reference_token_ids"])
                self._shared_parent = SharedParentSessionClient(self.engine, token_ids, contract)
                self._shared_parent.create(self._render_parent_prompt(run))
            children: list[dict[str, Any]] = []
            for rank, (request, payload) in enumerate(prepared):
                identity = execution_observation_identity(request.state, request.evaluator, request.intervention)
                candidate_id = condition_candidate_id(identity["observation_id"])
                prompt = self.engine.apply_template(payload["messages"])
                children.append({"candidate_id": candidate_id, "candidate_rank": rank, "prompt": prompt})
                self._shared_parent_children[candidate_id] = request.arm_id
            probe_started = True
            response = self._shared_parent.probe_round(children)
        except Exception:
            # Session creation happens before child dispatch and is safe to
            # abandon.  Once probe_round was accepted, this exception is
            # intentionally allowed to propagate so the runner conservatively
            # records every submitted condition as executed/failed.
            if not probe_started:
                if self._shared_parent is not None:
                    try:
                        self._shared_parent.close()
                    except Exception:
                        pass
                self._shared_parent = None
                return None
            self._shared_parent_disabled = True
            raise
        rows = response.get("results") if isinstance(response, Mapping) else None
        by_candidate = {
            row.get("candidate_id"): row for row in rows
            if isinstance(row, Mapping) and isinstance(row.get("candidate_id"), str)
        } if isinstance(rows, list) else {}
        outcomes: list[ArmExecutionOutcome] = []
        for request, payload in prepared:
            identity = execution_observation_identity(request.state, request.evaluator, request.intervention)
            candidate_id = condition_candidate_id(identity["observation_id"])
            raw = by_candidate.get(candidate_id)
            if isinstance(raw, Mapping) and isinstance(raw.get("result"), Mapping):
                raw = raw["result"]
            if not isinstance(raw, Mapping) or "status" not in raw:
                outcomes.append(ArmExecutionOutcome(
                    arm_id=request.arm_id, execution_disposition="executed", state="failed",
                    diagnostics={"execution_strategy": "shared_parent", "reason": "shared_parent_token_evidence_unavailable"},
                ))
                continue
            observation = _observation_from_probe(
                request.state, request.evaluator, request.intervention, raw,
                provenance=payload["provenance"],
            )
            outcomes.append(ArmExecutionOutcome(
                arm_id=request.arm_id, observation=observation,
                execution_disposition="executed", state="completed" if observation.completed else "failed",
                diagnostics={"execution_strategy": "shared_parent", "candidate_id": candidate_id},
            ))
        return BatchExecutionResult(tuple(outcomes), {
            "execution_strategy": "shared_parent", "batch_count": 1,
            "batch_size": len(prepared), "shared_parent_session": self._shared_parent.report(),
            "optimization_diagnostics": deepcopy(self._optimization_diagnostics),
        })

    def on_control_accepted(self, **_kwargs: Any) -> None:
        return None

    def on_candidate_accepted(self, *, evidence: Any, **_kwargs: Any) -> None:
        if self._shared_parent is None or self._shared_parent_disabled:
            return
        if not isinstance(evidence, Mapping):
            return
        if evidence.get("disposition") != "executed" or evidence.get("observation_status") != "exact_preserved":
            return
        observation_id = evidence.get("observation_id")
        if not isinstance(observation_id, str):
            return
        candidate = condition_candidate_id(observation_id)
        try:
            self._shared_parent.promote(candidate, exact_preserved=True)
        except Exception as exc:
            self._shared_parent_disabled = True
            self._optimization_diagnostics.append({
                "kind": "promotion_failed", "error": str(exc),
                "candidate_id": candidate,
            })
            try:
                self._shared_parent.close()
            except Exception:
                pass
            self._shared_parent = None

    def close(self) -> None:
        if self._shared_parent is not None:
            try:
                self._shared_parent.close()
            finally:
                self._shared_parent = None

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
