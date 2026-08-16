"""Exact-reference adapter for :mod:`clozn.runs.budgeted_reduce`.

This layer owns prompt construction, worker token counting, and the existing
multi-arm exact-reference dispatch.  The reducer itself remains independent of
all of those concerns.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from collections.abc import Callable, Iterable, Mapping, Sequence
import os
import time
from typing import Any

from clozn.runs.answer_preservation import (
    is_reference_match_failed,
    is_reference_match_preserving,
)
from clozn.runs.budgeted_reduce import (
    BudgetedReductionResult,
    PreparedCandidate,
    UnitID,
    run_budgeted_reduction,
)
from clozn.runs.multi_arm import probe_reference_match_many
from clozn.runs.persistent_parent import (
    PersistentParentSessionClient,
    PersistentParentSessionError,
    assert_scalar_parity,
    candidate_id,
)


@dataclass
class EngineReferenceMatchAdapter:
    """Prepare and directly probe exact-reference context candidates.

    ``render_messages`` is supplied by the integration harness because the
    adapter should not know how a particular unit universe is rendered.
    ``engine.apply_template_info`` is the worker tokenizer/template seam used
    for the objective cost.
    """

    engine: Any
    substrate: Any
    render_messages: Callable[[tuple[UnitID, ...]], Sequence[Mapping[str, Any]]]
    reference_token_ids: tuple[int, ...]
    generation_contract: Mapping[str, Any]
    explicit_conditions: Mapping[str, Any] | None = None
    _probe_parent_retained_ids: tuple[UnitID, ...] | None = field(default=None, init=False, repr=False)
    native_parent_anchor_metrics: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)
    native_parent_anchor_parity_mismatches: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False,
    )

    def set_probe_context(self, *, stage: str, parent_retained_ids: tuple[UnitID, ...]) -> None:
        """Receive reducer dispatch metadata for the opt-in native parent anchor."""
        del stage
        self._probe_parent_retained_ids = tuple(parent_retained_ids)

    def prepare_candidate(self, retained_ids: tuple[UnitID, ...]) -> PreparedCandidate:
        messages = [dict(message) for message in self.render_messages(retained_ids)]
        rendered = self.engine.apply_template_info(messages)
        cost = rendered.get("prompt_tokens") if isinstance(rendered, Mapping) else None
        if isinstance(cost, bool) or not isinstance(cost, int) or cost < 0:
            raise RuntimeError(
                "worker apply_template_info did not return exact non-negative prompt_tokens"
            )
        payload = {
            "messages": messages,
            "reference_token_ids": list(self.reference_token_ids),
            "generation_contract": deepcopy(dict(self.generation_contract)),
            "explicit_conditions": deepcopy(dict(self.explicit_conditions or {})),
        }
        return PreparedCandidate(
            retained_ids=retained_ids,
            cost=cost,
            probe_payload=payload,
        )

    def probe_many(self, prepared_candidates: Sequence[PreparedCandidate]) -> list[Any]:
        arms = [dict(candidate.probe_payload) for candidate in prepared_candidates]
        parent_anchor_enabled = os.environ.get("CLOZN_ENABLE_NATIVE_PARENT_ANCHOR", "").lower() in {
            "1", "true", "yes", "on",
        }
        if parent_anchor_enabled and self._probe_parent_retained_ids:
            parent_messages = [dict(message) for message in self.render_messages(
                tuple(self._probe_parent_retained_ids)
            )]
            parent_prompt = self.engine.apply_template(parent_messages)
            for arm in arms:
                arm["parent_anchor_prompt"] = parent_prompt
        # Keep scalar evidence authoritative. The opt-in native call is a
        # paired diagnostic on the same arms; it can never decide preservation
        # or certificate state and is unavailable on proof-grade calls.
        scalar = probe_reference_match_many(self.substrate, arms)
        native_enabled = os.environ.get("CLOZN_ENABLE_NATIVE_PARENT_ANCHOR", "").lower() in {
            "1", "true", "yes", "on",
        }
        if not native_enabled or not self._probe_parent_retained_ids:
            return scalar
        native = probe_reference_match_many(self.substrate, arms, proof_grade=False)
        metrics = dict(getattr(self.substrate, "last_native_reference_match_metrics", None) or {})
        self.native_parent_anchor_metrics.append(metrics)
        for index, (native_row, scalar_row) in enumerate(zip(native, scalar)):
            projection_keys = (
                "status", "matched_token_count", "first_divergence_index",
                "expected_token_id", "actual_token_id", "divergence_kind",
                "termination_match",
            )
            if {key: native_row.get(key) for key in projection_keys} != {
                    key: scalar_row.get(key) for key in projection_keys}:
                self.native_parent_anchor_parity_mismatches.append({
                    "arm_index": index,
                    "native": {key: native_row.get(key) for key in projection_keys},
                    "scalar": {key: scalar_row.get(key) for key in projection_keys},
                })
        return scalar

    @staticmethod
    def is_preserving(evidence: Any) -> bool:
        return is_reference_match_preserving(evidence)

    @staticmethod
    def is_failed(evidence: Any) -> bool:
        return is_reference_match_failed(evidence)


class PersistentEngineReferenceMatchAdapter(EngineReferenceMatchAdapter):
    """Opt-in reducer adapter using a worker persistent accepted-parent session.

    Scalar probes are always run first and remain the returned evidence.  The native session is a
    paired runtime diagnostic; a reducer adoption callback promotes only after scalar preservation
    and native/scalar parity have both been established for that exact candidate.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.persistent_session = PersistentParentSessionClient(
            self.engine,
            tuple(self.reference_token_ids),
            dict(self.generation_contract),
        )
        self.persistent_parent_metrics: list[dict[str, Any]] = []
        self.persistent_parent_parity_mismatches: list[dict[str, Any]] = []
        self.persistent_parent_promotion_metrics: list[dict[str, Any]] = []
        self.persistent_parent_final_report: dict[str, Any] | None = None
        self.persistent_parent_scalar_confirmation_wall_seconds = 0.0
        self._last_native_by_candidate: dict[str, dict[str, Any]] = {}

    def _prompt_for(self, prepared: PreparedCandidate) -> str:
        return self.engine.apply_template(list(prepared.probe_payload.get("messages") or []))

    def on_control_accepted(self, _candidate: Any, prepared: PreparedCandidate, _evidence: Any) -> None:
        if self.persistent_session.session_id is None:
            self.persistent_session.create(self._prompt_for(prepared))

    def _classify_native_row(self, row: Mapping[str, Any]) -> dict[str, Any]:
        from clozn.runs.answer_preservation import classify_reference_match

        raw = dict(row.get("result") or {})
        generated = raw.get("generated_token_ids")
        if (not isinstance(generated, list) or any(isinstance(value, bool) or not isinstance(value, int)
                                                    for value in generated)):
            raise PersistentParentSessionError("persistent native row has no integer token trace",
                                               code="malformed_native_result", status=502)
        classified = classify_reference_match(
            list(self.reference_token_ids), generated,
            diverged=raw.get("diverged"), diverged_at=raw.get("diverged_at"),
            termination=raw.get("termination"), finish_reason=raw.get("finish_reason"),
            expected_termination=self.generation_contract.get("expected_termination"),
            max_new=self.generation_contract["max_new"],
        )
        classified.update({
            "generated_token_ids": list(generated),
            "finish_reason": raw.get("finish_reason"),
            "termination": dict(raw.get("termination") or {}),
            "reply": raw.get("reply", ""),
        })
        return classified

    def probe_many(self, prepared_candidates: Sequence[PreparedCandidate]) -> list[Any]:
        arms = [dict(candidate.probe_payload) for candidate in prepared_candidates]
        scalar_started = time.perf_counter()
        scalar = probe_reference_match_many(self.substrate, arms, proof_grade=True)
        self.persistent_parent_scalar_confirmation_wall_seconds += max(
            0.0, time.perf_counter() - scalar_started
        )
        self._last_native_by_candidate = {}
        if self.persistent_session.session_id is None:
            return scalar

        children = [
            {
                "candidate_id": candidate_id(candidate.retained_ids),
                "candidate_rank": index,
                "prompt": self._prompt_for(candidate),
            }
            for index, candidate in enumerate(prepared_candidates)
        ]
        response = self.persistent_session.probe_round(children)
        by_id = {str(row["candidate_id"]): row for row in response["results"]}
        native_rows = [self._classify_native_row(by_id[item["candidate_id"]]) for item in children]
        try:
            assert_scalar_parity(native_rows, scalar)
        except Exception as exc:
            mismatches = getattr(exc, "mismatches", [{"error": str(exc)}])
            self.persistent_parent_parity_mismatches.extend(deepcopy(mismatches))
            try:
                self.persistent_session.close()
            finally:
                self.persistent_parent_final_report = self.persistent_session.report()
                self.persistent_parent_final_report.setdefault("telemetry", {})[
                    "total_scalar_confirmation_wall_seconds"
                ] = round(self.persistent_parent_scalar_confirmation_wall_seconds, 6)
            raise
        self.persistent_parent_metrics.append(deepcopy(dict(response.get("round_metrics") or {})))
        for child, native in zip(children, native_rows):
            self._last_native_by_candidate[child["candidate_id"]] = {
                "native": native,
                "row": by_id[child["candidate_id"]],
            }
        return scalar

    def _scalar_confirmation(self, prepared: PreparedCandidate) -> dict[str, Any]:
        """Run the trusted scalar gate immediately before a persistent promotion."""
        arms = [dict(prepared.probe_payload)]
        scalar_started = time.perf_counter()
        scalar = probe_reference_match_many(self.substrate, arms, proof_grade=True)
        self.persistent_parent_scalar_confirmation_wall_seconds += max(
            0.0, time.perf_counter() - scalar_started
        )
        if len(scalar) != 1:
            raise PersistentParentSessionError(
                "scalar confirmation returned the wrong child count",
                code="malformed_scalar_confirmation", status=502,
            )
        return dict(scalar[0])

    def _probe_uncached_accepted_candidate(self, candidate: Any, prepared: PreparedCandidate) -> dict[str, Any]:
        """Re-run a cached reducer winner against the current parent before promotion."""
        scalar = self._scalar_confirmation(prepared)
        cid = candidate_id(candidate.retained_ids)
        native_response = self.persistent_session.probe_round([{
            "candidate_id": cid, "candidate_rank": 0, "prompt": self._prompt_for(prepared),
        }])
        native = [self._classify_native_row(native_response["results"][0])]
        assert_scalar_parity(native, [scalar])
        self.persistent_parent_metrics.append(deepcopy(dict(native_response.get("round_metrics") or {})))
        self._last_native_by_candidate[cid] = {"native": native[0], "row": native_response["results"][0]}
        return scalar

    def on_candidate_accepted(self, candidate: Any, prepared: PreparedCandidate, evidence: Any) -> None:
        cid = candidate_id(candidate.retained_ids)
        record = self._last_native_by_candidate.get(cid)
        if record is None:
            confirmation = self._probe_uncached_accepted_candidate(candidate, prepared)
            record = self._last_native_by_candidate[cid]
        else:
            confirmation = self._scalar_confirmation(prepared)
            assert_scalar_parity([record["native"]], [confirmation])
        if not self.is_preserving(confirmation):
            raise PersistentParentSessionError(
                "trusted scalar confirmation rejected persistent promotion",
                code="scalar_confirmation_rejected_promotion", status=409,
            )
        if not self.is_preserving(evidence):
            raise PersistentParentSessionError(
                "reducer evidence rejected persistent promotion",
                code="scalar_reducer_evidence_rejected_promotion", status=409,
            )
        promotion = self.persistent_session.promote(
            cid,
            scalar_preserves=True,
            native_preserves=self.is_preserving(record["native"]),
        )
        self.persistent_parent_promotion_metrics.append(deepcopy(dict(promotion.get("telemetry") or {})))
        self._last_native_by_candidate = {}

    def close_persistent_session(self) -> None:
        if self.persistent_session.session_id is not None and not self.persistent_session.closed:
            try:
                self.persistent_session.close()
            finally:
                self.persistent_parent_final_report = self.persistent_session.report()
                telemetry = self.persistent_parent_final_report.setdefault("telemetry", {})
                telemetry["total_scalar_confirmation_wall_seconds"] = round(
                    self.persistent_parent_scalar_confirmation_wall_seconds, 6
                )


def run_engine_reference_match_reduction(
    adapter: EngineReferenceMatchAdapter,
    ordered_unit_ids: Iterable[UnitID],
    max_counterfactual_probes: int,
    *,
    attempt_inclusion_check: bool = True,
    ) -> BudgetedReductionResult:
    """Run the model-free reducer through the conservative exact adapter."""

    return run_budgeted_reduction(
        ordered_unit_ids,
        max_counterfactual_probes,
        adapter.prepare_candidate,
        adapter.probe_many,
        attempt_inclusion_check=attempt_inclusion_check,
        is_preserving=adapter.is_preserving,
        is_failed=adapter.is_failed,
    )


def run_engine_reference_match_persistent_reduction(
    adapter: PersistentEngineReferenceMatchAdapter,
    ordered_unit_ids: Iterable[UnitID],
    max_counterfactual_probes: int,
    *,
    attempt_inclusion_check: bool = True,
) -> BudgetedReductionResult:
    """Run the bounded reducer with explicit persistent-session lifecycle cleanup."""
    try:
        return run_budgeted_reduction(
            ordered_unit_ids,
            max_counterfactual_probes,
            adapter.prepare_candidate,
            adapter.probe_many,
            attempt_inclusion_check=attempt_inclusion_check,
            is_preserving=adapter.is_preserving,
            is_failed=adapter.is_failed,
        )
    finally:
        adapter.close_persistent_session()


__all__ = [
    "EngineReferenceMatchAdapter",
    "PersistentEngineReferenceMatchAdapter",
    "run_engine_reference_match_reduction",
    "run_engine_reference_match_persistent_reduction",
]
