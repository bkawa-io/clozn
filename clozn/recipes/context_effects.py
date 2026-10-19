"""Thin Q2 recipe: direct source deletion scores plus answer-span projection."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from clozn.experiments.evaluators import ScoreRecordedContinuation
from clozn.experiments.execution import ExecutionAdapterError, resolve_delete_source
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.projection import AnswerSpanEffect, project_answer_effects
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.scoring import DeleteSourceRecordedContinuationScoreAdapter
from clozn.experiments.selections import AnswerSelection, ContextSelection
from clozn.experiments.state import ExecutionState


_NO_EXACTLY_REMOVABLE_SOURCE = "no automatically discovered sources are exactly removable"
_NO_EXACTLY_REMOVABLE_SOURCE_CODE = "context_investigation_no_exactly_removable_source"
_NO_EXACTLY_REMOVABLE_SOURCE_READER_REASON = (
    "no canonical Context Receipt source is exactly removable"
)


class _NoExactlyRemovableSource(ValueError):
    """The receipt is readable, but its default Q2 universe has no legal arm."""


def _receipt_source_universe(run: Mapping[str, Any]) -> list[str]:
    """Return the same derived default units consumed by Minimal Context."""
    manifest = run.get("context_units") if isinstance(run.get("context_units"), Mapping) else {}
    source_ids = manifest.get("default_source_ids")
    if not isinstance(source_ids, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for source_id in source_ids:
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            continue
        if not (source_id.startswith("src_") or source_id.startswith("seg_")):
            continue
        seen.add(source_id)
        result.append(source_id)
    return result


def _receipt_display_source_universe(run: Mapping[str, Any]) -> list[str]:
    """Return every canonical source that the Context Receipt records as delivered.

    The Q2 measurement universe is intentionally narrower: a source can be
    visible in the received-context reader while being protected or otherwise
    ineligible for an automatic one-source deletion arm.  Keep this display
    catalogue derived from receipt order and never infer sources from prompt
    text or the rendered final prompt.
    """
    receipt = run.get("context_receipt") if isinstance(run.get("context_receipt"), Mapping) else {}
    delivered = receipt.get("delivered")
    if not isinstance(delivered, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for segment in delivered:
        if not isinstance(segment, Mapping):
            continue
        candidates: list[Any] = [segment.get("segment_id")]
        sources = segment.get("sources")
        if isinstance(sources, list):
            candidates.extend(
                item.get("source_id") for item in sources if isinstance(item, Mapping)
            )
        for source_id in candidates:
            if (
                not isinstance(source_id, str)
                or not source_id
                or source_id in seen
                or not (source_id.startswith("src_") or source_id.startswith("seg_"))
            ):
                continue
            seen.add(source_id)
            result.append(source_id)
    return result


def _source_universe(run: Mapping[str, Any], source_ids: Iterable[str] | None) -> tuple[list[str], str]:
    if source_ids is not None:
        if isinstance(source_ids, (str, bytes)):
            raise ValueError("source_ids must be an iterable of canonical source IDs")
        values = list(source_ids)
        if not values:
            raise ValueError("source_ids cannot be empty")
        if len(values) != len(set(values)):
            raise ValueError("source_ids must not contain duplicates")
        # ContextSelection is the one canonical ID validator used by the kernel.
        for value in values:
            ContextSelection([value])
        return values, "explicit"
    values = _receipt_source_universe(run)
    if not values:
        raise _NoExactlyRemovableSource(_NO_EXACTLY_REMOVABLE_SOURCE)
    # Auto discovery is conservative: a source that cannot be deleted exactly,
    # including the protected current request, is not silently displayed as an
    # effect arm.
    removable: list[str] = []
    for source_id in values:
        try:
            resolve_delete_source(run, DeleteSource(ContextSelection([source_id])))
        except Exception:
            continue
        removable.append(source_id)
    if not removable:
        raise _NoExactlyRemovableSource(_NO_EXACTLY_REMOVABLE_SOURCE)
    return removable, "canonical_context_units"


@dataclass(frozen=True)
class ContextEffectsPlan:
    """Model-free Q2 plan shared by measurement and linked read models."""

    run_id: str
    execution_state: ExecutionState
    display_source_ids: tuple[str, ...]
    measurement_source_ids: tuple[str, ...]
    source_universe_basis: str
    experiment: Experiment

    @property
    def experiment_id(self) -> str:
        return self.experiment.experiment_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_state": self.execution_state.to_dict(),
            "display_source_ids": list(self.display_source_ids),
            "measurement_source_ids": list(self.measurement_source_ids),
            "source_universe_basis": self.source_universe_basis,
            "experiment": self.experiment.to_dict(),
        }


@dataclass(frozen=True)
class ContextEffectsUnavailablePlan:
    """Read-only projection plan when canonical sources exist but Q2 cannot execute.

    Deliberately carries no Experiment.  It exists only so the linked reader can
    preserve canonical Context and recorded Answer data without minting a usable
    measurement identity or weakening the strict effects-job planner.
    """

    run_id: str
    execution_state: ExecutionState
    display_source_ids: tuple[str, ...]
    measurement_source_ids: tuple[str, ...]
    source_universe_basis: str
    reason: str
    reason_code: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "execution_state": self.execution_state.to_dict(),
            "display_source_ids": list(self.display_source_ids),
            "measurement_source_ids": [],
            "source_universe_basis": self.source_universe_basis,
            "measurement": {
                "status": "unavailable",
                "reason": self.reason,
                "reason_code": self.reason_code,
            },
        }


def plan_context_effects(run: Mapping[str, Any], source_ids: Iterable[str] | None = None) -> ContextEffectsPlan:
    """Build the canonical, model-free Q2 Experiment plan.

    ``display_source_ids`` is the complete receipt catalogue.  The ordered
    ``measurement_source_ids`` tuple is exactly the one-source arm universe
    used by the existing recipe, so extracting this plan does not change
    Experiment or arm identity for equivalent requests.
    """
    if not isinstance(run, Mapping):
        raise TypeError("plan_context_effects requires a run mapping")
    run_id = run.get("id")
    if not isinstance(run_id, str) or not run_id:
        raise ValueError("plan_context_effects requires a recorded run id")
    state = ExecutionState.from_run(run)
    measurement, universe_basis = _source_universe(run, source_ids)
    experiment = Experiment(
        base=state, evaluator=ScoreRecordedContinuation(),
        arms=[DeleteSource(ContextSelection([source_id])) for source_id in measurement],
    )
    return ContextEffectsPlan(
        run_id=run_id,
        execution_state=state,
        display_source_ids=tuple(_receipt_display_source_universe(run)),
        measurement_source_ids=tuple(measurement),
        source_universe_basis=universe_basis,
        experiment=experiment,
    )


def plan_context_effects_reader(
    run: Mapping[str, Any],
) -> ContextEffectsPlan | ContextEffectsUnavailablePlan:
    """Plan the pure linked reader without weakening executable Q2 planning.

    A default plan whose canonical receipt sources are all protected or
    otherwise not exactly removable still has a faithful read projection.  In
    that one condition, return a typed unavailable plan with no Experiment.
    Every other planning error remains an error, and :func:`plan_context_effects`
    continues to reject an empty executable measurement universe.
    """
    try:
        return plan_context_effects(run)
    except _NoExactlyRemovableSource:
        return ContextEffectsUnavailablePlan(
            run_id=str(run["id"]),
            execution_state=ExecutionState.from_run(run),
            display_source_ids=tuple(_receipt_display_source_universe(run)),
            measurement_source_ids=(),
            source_universe_basis="canonical_context_units",
            reason=_NO_EXACTLY_REMOVABLE_SOURCE_READER_REASON,
            reason_code=_NO_EXACTLY_REMOVABLE_SOURCE_CODE,
        )


def measure_context_effects(run: Mapping[str, Any], source_ids: Iterable[str] | None = None, *,
                            execution_adapter: Any = None, substrate: Any = None,
                            run_loader: Any = None, answer_selection: AnswerSelection | None = None,
                            include_control: bool = True, observation_store: ObservationStore | None = None,
                            store: ObservationStore | None = None,
                            cancel: Any = None) -> ExperimentResult:
    """Measure baseline plus direct leave-one-out deletion score arms.

    ``answer_selection`` is accepted as a convenience validation input but is
    intentionally not stored in or hashed into the experiment.  Use
    :func:`project_context_effects` for the read-side selection.

    Every arm uses the same trusted scalar scoring contract.
    """
    if not isinstance(run, Mapping):
        raise TypeError("measure_context_effects requires a run mapping")
    if answer_selection is not None and not isinstance(answer_selection, AnswerSelection):
        raise TypeError("answer_selection must be an AnswerSelection")
    plan = plan_context_effects(run, source_ids)
    adapter = execution_adapter
    if adapter is None:
        adapter = DeleteSourceRecordedContinuationScoreAdapter(
            substrate, run=run, run_loader=run_loader,
        )
    return run_experiment(
        plan.experiment, adapter, include_control=include_control,
        observation_store=observation_store, store=store,
        cancel=cancel,
        requested_by={"recipe": "context_effects"},
        diagnostics={
        "recipe": "context_effects",
        "source_universe": list(plan.measurement_source_ids),
        "source_universe_basis": plan.source_universe_basis,
        "measurement": "direct_leave_one_out_delete_source",
        },
    )


def project_context_effects(result: ExperimentResult, answer_selection: AnswerSelection,
                            *, ordering: str = "absolute") -> list[AnswerSpanEffect]:
    """Project arbitrary recorded-answer selections with zero execution calls."""
    return project_answer_effects(result, answer_selection, ordering=ordering)


def context_effect_message(effect: AnswerSpanEffect) -> str:
    """Use intervention-specific wording for one projected result."""
    if effect.status != "completed" or effect.delta_nats is None:
        return f"Deleting {', '.join(effect.source_ids)} has no available measured score for the selected recorded answer span."
    if effect.delta_nats > 0:
        return f"Deleting {', '.join(effect.source_ids)} reduced support for the selected recorded answer span by {effect.delta_nats:g} nats."
    if effect.delta_nats < 0:
        return f"Deleting {', '.join(effect.source_ids)} increased support for the selected recorded answer span by {abs(effect.delta_nats):g} nats."
    return f"Deleting {', '.join(effect.source_ids)} produced no measured score difference for the selected recorded answer span."


__all__ = [
    "ContextEffectsPlan", "ContextEffectsUnavailablePlan", "context_effect_message",
    "measure_context_effects", "plan_context_effects", "plan_context_effects_reader",
    "project_context_effects",
]
