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


__all__ = [
    "EngineReferenceMatchAdapter",
    "run_engine_reference_match_reduction",
]
