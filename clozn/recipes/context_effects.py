"""Thin Q2 recipe: direct source deletion scores plus answer-span projection."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
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


def _receipt_source_universe(run: Mapping[str, Any]) -> list[str]:
    receipt = run.get("context_receipt") if isinstance(run.get("context_receipt"), Mapping) else {}
    delivered = receipt.get("delivered")
    if not isinstance(delivered, list):
        return []
    result: list[str] = []
    seen: set[str] = set()
    for segment in delivered:
        if not isinstance(segment, Mapping):
            continue
        sources = segment.get("sources")
        candidates = [
            item.get("source_id") for item in sources
            if isinstance(sources, list) and isinstance(item, Mapping)
            and isinstance(item.get("source_id"), str) and item.get("source_id")
        ] if isinstance(sources, list) else []
        if not candidates:
            candidates = [segment.get("segment_id")]
        for source_id in candidates:
            if not isinstance(source_id, str) or not source_id or source_id in seen:
                continue
            if not (source_id.startswith("src_") or source_id.startswith("seg_")):
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
        raise ValueError("no canonical Context Receipt sources are available")
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
        raise ValueError("no automatically discovered sources are exactly removable")
    return removable, "canonical_context_receipt"


def measure_context_effects(run: Mapping[str, Any], source_ids: Iterable[str] | None = None, *,
                            execution_adapter: Any = None, substrate: Any = None,
                            run_loader: Any = None, answer_selection: AnswerSelection | None = None,
                            include_control: bool = True, observation_store: ObservationStore | None = None,
                            store: ObservationStore | None = None) -> ExperimentResult:
    """Measure baseline plus direct leave-one-out deletion score arms.

    ``answer_selection`` is accepted as a convenience validation input but is
    intentionally not stored in or hashed into the experiment.  Use
    :func:`project_context_effects` for the read-side selection.
    """
    if not isinstance(run, Mapping):
        raise TypeError("measure_context_effects requires a run mapping")
    if answer_selection is not None and not isinstance(answer_selection, AnswerSelection):
        raise TypeError("answer_selection must be an AnswerSelection")
    state = ExecutionState.from_run(run)
    universe, universe_basis = _source_universe(run, source_ids)
    experiment = Experiment(
        base=state, evaluator=ScoreRecordedContinuation(),
        arms=[DeleteSource(ContextSelection([source_id])) for source_id in universe],
    )
    adapter = execution_adapter
    if adapter is None:
        adapter = DeleteSourceRecordedContinuationScoreAdapter(
            substrate, run=run, run_loader=run_loader,
        )
    return run_experiment(
        experiment, adapter, include_control=include_control,
        observation_store=observation_store, store=store,
        requested_by={"recipe": "context_effects"},
        diagnostics={
        "recipe": "context_effects",
        "source_universe": list(universe),
        "source_universe_basis": universe_basis,
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


__all__ = ["context_effect_message", "measure_context_effects", "project_context_effects"]
