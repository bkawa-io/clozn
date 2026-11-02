"""Thin Q2 recipe: direct source deletion scores plus answer-span projection."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
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
from clozn.runs.context_units import MAX_CONTEXT_ARMS_CEILING


_NO_EXACTLY_REMOVABLE_SOURCE = "no automatically discovered sources are exactly removable"
_NO_EXACTLY_REMOVABLE_SOURCE_CODE = "context_investigation_no_exactly_removable_source"
_NO_EXACTLY_REMOVABLE_SOURCE_READER_REASON = (
    "no canonical Context Receipt source is exactly removable"
)


class _NoExactlyRemovableSource(ValueError):
    """The receipt is readable, but its default Q2 universe has no legal arm."""


def _receipt_source_universe(run: Mapping[str, Any]) -> list[str]:
    """Return the derived units shared by Context Effects and Lean Context."""
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

    The display catalogue includes both whole-message roots and any nested
    automatic chunks. The measurement universe chooses one non-overlapping
    partition from that catalogue.
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


def configure_context_effect_sources(
    run: Mapping[str, Any], *, granularity: str, max_arms: int,
) -> tuple[str, ...]:
    """Choose an exact canonical partition for a bounded Context Effects job.

    ``message`` uses one whole-message receipt root per non-empty message.
    ``adaptive`` starts from every available Markdown-derived child span and
    deterministically collapses the least text-dense split messages until the
    requested arm budget is met. No synthetic or approximate source is minted.
    """
    if granularity not in {"message", "adaptive"}:
        raise ValueError("granularity must be 'message' or 'adaptive'")
    if (
        isinstance(max_arms, bool)
        or not isinstance(max_arms, int)
        or not 1 <= max_arms <= MAX_CONTEXT_ARMS_CEILING
    ):
        raise ValueError(
            f"max_arms must be an integer between 1 and {MAX_CONTEXT_ARMS_CEILING}"
        )
    messages = run.get("messages")
    receipt = run.get("context_receipt")
    delivered = receipt.get("delivered") if isinstance(receipt, Mapping) else None
    if not isinstance(messages, list) or not isinstance(delivered, list):
        raise ValueError("recorded messages and Context Receipt are required")

    partitions: list[dict[str, Any]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, Mapping) or not isinstance(message.get("content"), str):
            continue
        text = message["content"]
        if not text:
            continue
        row = delivered[message_index] if message_index < len(delivered) else None
        if not isinstance(row, Mapping):
            raise ValueError(f"Context Receipt is missing message {message_index}")
        root_id = row.get("segment_id")
        if not isinstance(root_id, str) or not root_id:
            raise ValueError(f"Context Receipt message {message_index} has no canonical root")
        children = [
            source.get("source_id")
            for source in row.get("sources", [])
            if isinstance(source, Mapping)
            and isinstance(source.get("source_id"), str)
            and source.get("source_id")
        ] if isinstance(row.get("sources"), list) else []
        partitions.append({
            "message_index": message_index,
            "text_length": len(text),
            "root_id": root_id,
            "fine_ids": tuple(dict.fromkeys(children)) if len(children) > 1 else (root_id,),
        })

    if not partitions:
        raise _NoExactlyRemovableSource(_NO_EXACTLY_REMOVABLE_SOURCE)
    minimum = len(partitions)
    if max_arms < minimum:
        raise ValueError(
            f"max_arms={max_arms} cannot cover {minimum} non-empty messages exactly; "
            f"use at least {minimum} arms"
        )
    if granularity == "message":
        return tuple(item["root_id"] for item in partitions)

    selected = [list(item["fine_ids"]) for item in partitions]
    selected_count = sum(len(ids) for ids in selected)
    candidates = sorted(
        (
            (
                item["text_length"] / (len(item["fine_ids"]) - 1),
                item["message_index"],
                index,
            )
            for index, item in enumerate(partitions)
            if len(item["fine_ids"]) > 1
        ),
        key=lambda candidate: (candidate[0], candidate[1]),
    )
    for _density, _message_index, index in candidates:
        if selected_count <= max_arms:
            break
        selected_count -= len(selected[index]) - 1
        selected[index] = [partitions[index]["root_id"]]
    if selected_count > max_arms:
        raise ValueError(
            f"the canonical context partition requires at least {selected_count} arms"
        )
    return tuple(source_id for ids in selected for source_id in ids)


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
    # Auto discovery is conservative: a source that cannot be deleted exactly
    # is not silently presented as an executable effect arm.
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


def plan_context_effects(
    run: Mapping[str, Any], source_ids: Iterable[str] | None = None, *,
    scoring_mode: str = "exact",
) -> ContextEffectsPlan:
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
        base=state, evaluator=ScoreRecordedContinuation(scoring_mode=scoring_mode),
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
    source_ids: Iterable[str] | None = None,
    *,
    scoring_mode: str = "exact",
) -> ContextEffectsPlan | ContextEffectsUnavailablePlan:
    """Plan the pure linked reader without weakening executable Q2 planning.

    A default plan with no exactly removable canonical source still has a
    faithful read projection. In that condition, return a typed unavailable
    plan with no Experiment.
    Every other planning error remains an error, and :func:`plan_context_effects`
    continues to reject an empty executable measurement universe.
    """
    try:
        return plan_context_effects(
            run, source_ids=source_ids, scoring_mode=scoring_mode,
        )
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
                            cancel: Any = None, arm_batch_size: int | None = None,
                            batch_control: bool = False,
                            scoring_mode: str = "exact",
                            execution_diagnostics: Mapping[str, Any] | None = None,
                            on_progress: Callable[[Mapping[str, Any]], None] | None = None) -> ExperimentResult:
    """Measure baseline plus direct leave-one-out deletion score arms.

    ``answer_selection`` is accepted as a convenience validation input but is
    intentionally not stored in or hashed into the experiment.  Use
    :func:`project_context_effects` for the read-side selection.

    Every arm uses the requested scoring contract: ``exact`` preserves the
    scalar reference path, while ``fast`` uses the native multi-arm path and
    records its approximate evidence grade (with scalar fallback).
    """
    if not isinstance(run, Mapping):
        raise TypeError("measure_context_effects requires a run mapping")
    if answer_selection is not None and not isinstance(answer_selection, AnswerSelection):
        raise TypeError("answer_selection must be an AnswerSelection")
    plan = plan_context_effects(run, source_ids, scoring_mode=scoring_mode)
    adapter = execution_adapter
    if adapter is None:
        adapter = DeleteSourceRecordedContinuationScoreAdapter(
            substrate, run=run, run_loader=run_loader,
        )
    return run_experiment(
        plan.experiment, adapter, include_control=include_control,
        observation_store=observation_store, store=store,
        cancel=cancel,
        arm_batch_size=arm_batch_size,
        batch_control=batch_control,
        on_progress=on_progress,
        requested_by={"recipe": "context_effects", "scoring_mode": scoring_mode},
        diagnostics={
            "recipe": "context_effects",
            "source_universe": list(plan.measurement_source_ids),
            "source_universe_basis": plan.source_universe_basis,
            "measurement": "direct_leave_one_out_delete_source",
            "scoring_mode": scoring_mode,
            "requested_sequence_batch_size": arm_batch_size,
            "batch_control_requested": batch_control,
            **dict(execution_diagnostics or {}),
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
    "ContextEffectsPlan", "ContextEffectsUnavailablePlan", "configure_context_effect_sources",
    "context_effect_message",
    "measure_context_effects", "plan_context_effects", "plan_context_effects_reader",
    "project_context_effects",
]
