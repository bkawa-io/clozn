"""Optional coarse-to-fine scheduling for context-source score experiments.

This module sits above the experimental kernel.  It changes neither the
default :func:`measure_context_effects` recipe nor inference semantics: callers
must opt in, provide an explicit budget, and may inspect every unmeasured
condition.  Coarse groups are joint ``DeleteSource`` interventions; selected
fine arms are one-source interventions; requested joint arms remain joint.

The built-in adapter follows the ordinary trusted scalar context-effects path
unless the caller explicitly requests native screening.  ``screening_only`` is
always retained as a distinct output and selected second-stage conditions are
scalar-refined before they can become trusted evidence.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

from clozn.experiments.evaluators import ScoreRecordedContinuation
from clozn.experiments.interventions import DeleteSource
from clozn.experiments.kernel import Experiment
from clozn.experiments.observations import (
    TokenScoreDelta,
    TokenScoreObservation,
    execution_observation_identity,
)
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import ExperimentResult, run_experiment
from clozn.experiments.scoring import (
    DeleteSourceRecordedContinuationScoreAdapter,
    ScoreTokensRefinementPolicy,
)
from clozn.experiments.selections import ContextSelection

from .context_effects import plan_context_effects


SCHEMA_VERSION = "clozn.context-effects-coarse-to-fine.v1"
_EVIDENCE_STATES = frozenset({"trusted", "screening_only", "unavailable", "skipped"})


@dataclass(frozen=True)
class ContextEffectsSearchBudget:
    """Hard limits on *new* intervention measurements in each stage.

    A trusted observation already present in ``ObservationStore`` and a
    condition measured earlier in this same search do not consume a new-model
    slot.  The unchanged control is reported separately and is not an
    intervention arm.
    """

    max_coarse_arms: int
    max_fine_arms: int
    max_joint_arms: int = 0

    def __post_init__(self) -> None:
        for name in ("max_coarse_arms", "max_fine_arms", "max_joint_arms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_coarse_arms": self.max_coarse_arms,
            "max_fine_arms": self.max_fine_arms,
            "max_joint_arms": self.max_joint_arms,
        }


@dataclass(frozen=True)
class ContextEffectsScheduledArm:
    """One scheduled or explicitly skipped condition.

    ``execution_state`` describes orchestration. ``evidence_status`` describes
    epistemic use.  In particular, an executed native screen is
    ``screening_only`` rather than trusted, while a budget omission is
    ``skipped`` rather than weak or unavailable evidence.
    """

    phase: str
    source_ids: tuple[str, ...]
    selection_reason: str
    execution_state: str
    execution_disposition: str
    evidence_status: str
    delta_nats: float | None = None
    observation_id: str | None = None
    proof_grade: str | None = None
    reused_from_phase: str | None = None
    numerical_regime: str | None = None
    execution_identity: Mapping[str, Any] | None = None
    diagnostics: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.phase not in {"coarse", "fine", "joint"}:
            raise ValueError("phase must be coarse, fine, or joint")
        if self.evidence_status not in _EVIDENCE_STATES:
            raise ValueError("unsupported evidence_status")
        if not self.source_ids:
            raise ValueError("scheduled arms require source IDs")
        if self.numerical_regime is not None and (
            not isinstance(self.numerical_regime, str) or not self.numerical_regime
        ):
            raise ValueError("numerical_regime must be a non-empty string or None")
        if self.execution_identity is not None and not isinstance(
            self.execution_identity, Mapping
        ):
            raise TypeError("execution_identity must be a mapping or None")

    @property
    def trusted(self) -> bool:
        return self.evidence_status == "trusted"

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "source_ids": list(self.source_ids),
            "selection_reason": self.selection_reason,
            "execution_state": self.execution_state,
            "execution_disposition": self.execution_disposition,
            "evidence_status": self.evidence_status,
            "delta_nats": self.delta_nats,
            "observation_id": self.observation_id,
            "proof_grade": self.proof_grade,
            "reused_from_phase": self.reused_from_phase,
            "numerical_regime": self.numerical_regime,
            "execution_identity": (
                dict(self.execution_identity)
                if self.execution_identity is not None else None
            ),
            "diagnostics": dict(self.diagnostics or {}),
        }


@dataclass(frozen=True)
class ContextEffectsCoarseToFineResult:
    run_id: str
    source_ids: tuple[str, ...]
    refine_threshold_nats: float
    budget: ContextEffectsSearchBudget
    arms: tuple[ContextEffectsScheduledArm, ...]
    coverage: Mapping[str, Any]
    budget_usage: Mapping[str, Any]
    native_screening_requested: bool
    coarse_experiment: ExperimentResult
    refinement_experiment: ExperimentResult | None

    def arms_for(self, phase: str) -> tuple[ContextEffectsScheduledArm, ...]:
        return tuple(arm for arm in self.arms if arm.phase == phase)

    def arm_for(self, phase: str, source_ids: Iterable[str]) -> ContextEffectsScheduledArm:
        key = tuple(sorted(source_ids))
        for arm in self.arms:
            if arm.phase == phase and arm.source_ids == key:
                return arm
        raise KeyError((phase, key))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": self.run_id,
            "source_ids": list(self.source_ids),
            "refine_threshold_nats": self.refine_threshold_nats,
            "budget": self.budget.to_dict(),
            "budget_usage": dict(self.budget_usage),
            "native_screening_requested": self.native_screening_requested,
            "coverage": dict(self.coverage),
            "arms": [arm.to_dict() for arm in self.arms],
            "experiments": {
                "coarse_experiment_id": self.coarse_experiment.experiment_id,
                "refinement_experiment_id": (
                    self.refinement_experiment.experiment_id
                    if self.refinement_experiment is not None else None
                ),
            },
        }


@dataclass(frozen=True)
class _Candidate:
    phase: str
    source_ids: tuple[str, ...]
    selection_reason: str

    @property
    def key(self) -> tuple[str, ...]:
        return self.source_ids

    @property
    def intervention(self) -> DeleteSource:
        return DeleteSource(ContextSelection(self.source_ids))


def _source_set(value: Iterable[str], *, name: str, universe: set[str], minimum: int = 1) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of source IDs")
    result = ContextSelection(value).source_ids
    if len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} distinct source IDs")
    unknown = set(result) - universe
    if unknown:
        raise ValueError(f"{name} contains source IDs outside the measurement universe: {sorted(unknown)!r}")
    return result


def _groups(source_ids: tuple[str, ...], raw_groups: Iterable[Iterable[str]] | None,
            group_size: int) -> tuple[tuple[str, ...], ...]:
    universe = set(source_ids)
    if raw_groups is None:
        return tuple(
            tuple(sorted(source_ids[index:index + group_size]))
            for index in range(0, len(source_ids), group_size)
        )
    if isinstance(raw_groups, (str, bytes)):
        raise ValueError("coarse_groups must be an iterable of source-ID groups")
    groups = tuple(
        _source_set(value, name=f"coarse_groups[{index}]", universe=universe)
        for index, value in enumerate(raw_groups)
    )
    if not groups:
        raise ValueError("coarse_groups cannot be empty")
    flattened = [source_id for group in groups for source_id in group]
    if len(flattened) != len(set(flattened)):
        raise ValueError("coarse_groups must not overlap")
    return groups


def _cached_observation(store: ObservationStore | None, state: Any,
                        intervention: DeleteSource) -> TokenScoreObservation | None:
    if store is None:
        return None
    identity = execution_observation_identity(
        state, ScoreRecordedContinuation(), intervention,
    )
    observation = store.find_observation(identity["observation_key_sha256"])
    if (
        isinstance(observation, TokenScoreObservation)
        and observation.completed
        and observation.trusted
    ):
        return observation
    return None


def _select_with_budget(candidates: Sequence[_Candidate], limit: int, *, state: Any,
                        store: ObservationStore | None,
                        prior: Mapping[tuple[str, ...], ContextEffectsScheduledArm] | None = None,
                        budget_reason: str) -> tuple[list[_Candidate], list[_Candidate], int, int]:
    selected: list[_Candidate] = []
    skipped: list[_Candidate] = []
    new_slots = 0
    cache_hits = 0
    prior = prior or {}
    for candidate in candidates:
        reused = candidate.key in prior and prior[candidate.key].evidence_status == "trusted"
        cached = None if reused else _cached_observation(store, state, candidate.intervention)
        if reused or cached is not None:
            selected.append(candidate)
            cache_hits += 1
        elif new_slots < limit:
            selected.append(candidate)
            new_slots += 1
        else:
            skipped.append(_Candidate(candidate.phase, candidate.source_ids, budget_reason))
    return selected, skipped, new_slots, cache_hits


def _screening_total(row: Any) -> float | None:
    values = []
    if getattr(row, "observation", None) is not None:
        values.append(getattr(row.observation, "diagnostics", {}))
    values.append(getattr(row, "diagnostics", {}))
    for diagnostics in values:
        value = diagnostics.get("screening_total_logprob") if isinstance(diagnostics, Mapping) else None
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
            return float(value)
    return None


def _execution_contract(
    observation: Any, diagnostics: Mapping[str, Any] | None,
) -> tuple[str | None, Mapping[str, Any] | None]:
    """Project stable execution facts out of internal diagnostic envelopes.

    Observation provenance is authoritative when present.  Per-arm diagnostics
    remain a fallback for screening-only and failed rows that do not carry a
    reusable Observation.  Missing facts stay explicit as ``None``; the
    scheduler must never infer a scalar identity from evidence status alone.
    """
    candidates = (
        getattr(observation, "execution_provenance", None),
        getattr(observation, "diagnostics", None),
        diagnostics,
    )
    numerical_regime: str | None = None
    execution_identity: Mapping[str, Any] | None = None
    for value in candidates:
        if not isinstance(value, Mapping):
            continue
        if numerical_regime is None:
            declared_regime = value.get("numerical_regime")
            if isinstance(declared_regime, str) and declared_regime:
                numerical_regime = declared_regime
        if execution_identity is None:
            declared_identity = value.get("execution_identity")
            if isinstance(declared_identity, Mapping):
                execution_identity = dict(declared_identity)
    return numerical_regime, execution_identity


def _measured_arm(*, phase: str, source_ids: tuple[str, ...], selection_reason: str,
                  row: Any, control: TokenScoreObservation | None) -> ContextEffectsScheduledArm:
    observation = getattr(row, "observation", None)
    diagnostics = dict(getattr(row, "diagnostics", {}) or {})
    delta: float | None = None
    evidence_status = "unavailable"
    proof_grade = getattr(observation, "proof_grade", None)
    if (
        isinstance(observation, TokenScoreObservation)
        and observation.completed
        and observation.trusted
    ):
        evidence_status = "trusted"
        if isinstance(control, TokenScoreObservation):
            score_delta = TokenScoreDelta.from_observations(control, observation)
            if score_delta.status == "completed":
                delta = score_delta.total_delta_nats
    else:
        screening_total = _screening_total(row)
        is_screen = proof_grade == "screening" or diagnostics.get("reason") == (
            "screening_not_selected_for_scalar_refinement"
        )
        if is_screen:
            evidence_status = "screening_only"
            if control is not None and control.total_continuation_logprob is not None and screening_total is not None:
                delta = control.total_continuation_logprob - screening_total
    numerical_regime, execution_identity = _execution_contract(
        observation, diagnostics,
    )
    return ContextEffectsScheduledArm(
        phase=phase,
        source_ids=source_ids,
        selection_reason=selection_reason,
        execution_state=str(getattr(row, "state", "failed")),
        execution_disposition=str(diagnostics.get("execution_disposition", "not_executed")),
        evidence_status=evidence_status,
        delta_nats=delta,
        observation_id=getattr(row, "observation_id", None),
        proof_grade=proof_grade,
        numerical_regime=numerical_regime,
        execution_identity=execution_identity,
        diagnostics=diagnostics,
    )


def _skipped(candidate: _Candidate) -> ContextEffectsScheduledArm:
    return ContextEffectsScheduledArm(
        phase=candidate.phase,
        source_ids=candidate.source_ids,
        selection_reason=candidate.selection_reason,
        execution_state="not_executed",
        execution_disposition="not_executed",
        evidence_status="skipped",
        diagnostics={"reason": candidate.selection_reason},
    )


def _reused(candidate: _Candidate, prior: ContextEffectsScheduledArm) -> ContextEffectsScheduledArm:
    return ContextEffectsScheduledArm(
        phase=candidate.phase,
        source_ids=candidate.source_ids,
        selection_reason=candidate.selection_reason,
        execution_state="completed",
        execution_disposition="reused",
        evidence_status=prior.evidence_status,
        delta_nats=prior.delta_nats,
        observation_id=prior.observation_id,
        proof_grade=prior.proof_grade,
        reused_from_phase=prior.phase,
        numerical_regime=prior.numerical_regime,
        execution_identity=prior.execution_identity,
        diagnostics={"reason": "same_condition_measured_in_coarse_phase"},
    )


def _reused_observation(candidate: _Candidate, observation: TokenScoreObservation,
                        control: TokenScoreObservation | None) -> ContextEffectsScheduledArm:
    delta: float | None = None
    if control is not None:
        score_delta = TokenScoreDelta.from_observations(control, observation)
        if score_delta.status == "completed":
            delta = score_delta.total_delta_nats
    numerical_regime, execution_identity = _execution_contract(observation, None)
    return ContextEffectsScheduledArm(
        phase=candidate.phase,
        source_ids=candidate.source_ids,
        selection_reason=candidate.selection_reason,
        execution_state="completed",
        execution_disposition="reused",
        evidence_status="trusted",
        delta_nats=delta,
        observation_id=observation.observation_id,
        proof_grade=observation.proof_grade,
        numerical_regime=numerical_regime,
        execution_identity=execution_identity,
        diagnostics={"reason": "trusted_observation_store_cache_hit"},
    )


def _cancelled(cancel: Any) -> bool:
    if cancel is None:
        return False
    if callable(cancel):
        return bool(cancel())
    method = getattr(cancel, "is_set", None)
    return bool(method()) if callable(method) else bool(cancel)


def _report_progress(progress: Any, *, phase: str, completed: int, total: int) -> None:
    """Emit truthful scheduler phase boundaries when a caller asks for them.

    The experiment runner currently owns each phase as one semantic batch, so
    this callback deliberately reports only boundaries.  It must not invent
    per-arm completion while a native or scalar batch is in flight.
    """
    if progress is None:
        return
    if not callable(progress):
        raise TypeError("progress must be callable")
    progress(phase=phase, completed=completed, total=total)


def measure_context_effects_coarse_to_fine(
    run: Mapping[str, Any],
    *,
    budget: ContextEffectsSearchBudget,
    source_ids: Iterable[str] | None = None,
    coarse_groups: Iterable[Iterable[str]] | None = None,
    coarse_group_size: int = 4,
    refine_threshold_nats: float = 0.25,
    drilldown_source_ids: Iterable[str] = (),
    joint_source_sets: Iterable[Iterable[str]] = (),
    execution_adapter: Any = None,
    substrate: Any = None,
    run_loader: Any = None,
    use_native_screening: bool = False,
    observation_store: ObservationStore | None = None,
    store: ObservationStore | None = None,
    cancel: Any = None,
    progress: Any = None,
) -> ContextEffectsCoarseToFineResult:
    """Measure bounded coarse groups, selected sources, and explicit joints.

    Fine selection is deterministic: caller-requested drilldowns first, then
    sources in coarse groups ordered by descending absolute measured effect.
    Requested drilldowns ignore a weak or unavailable coarse score but still
    respect ``max_fine_arms``.  Joint interventions are never inferred from
    singleton effects; only ``joint_source_sets`` are executed.  The default
    adapter performs trusted scoring unless ``use_native_screening=True``.
    That explicit opt-in composes the native screen/scalar-refine primitive:
    coarse missing conditions remain screening-only, while every selected
    missing fine/joint condition is marked proof-critical and scalar-refined.
    A custom adapter's screening rows are also preserved and can guide
    selection, but never become trusted observations.
    """
    if not isinstance(budget, ContextEffectsSearchBudget):
        raise TypeError("budget must be a ContextEffectsSearchBudget")
    if isinstance(coarse_group_size, bool) or not isinstance(coarse_group_size, int) or coarse_group_size < 1:
        raise ValueError("coarse_group_size must be a positive integer")
    if (
        isinstance(refine_threshold_nats, bool)
        or not isinstance(refine_threshold_nats, (int, float))
        or not math.isfinite(float(refine_threshold_nats))
        or float(refine_threshold_nats) < 0
    ):
        raise ValueError("refine_threshold_nats must be a finite non-negative number")
    if observation_store is not None and store is not None and observation_store is not store:
        raise ValueError("pass only one observation store")
    durable = observation_store or store
    if durable is not None and not isinstance(durable, ObservationStore):
        raise TypeError("observation_store must be an ObservationStore")
    if not isinstance(use_native_screening, bool):
        raise TypeError("use_native_screening must be a boolean")
    if execution_adapter is not None and use_native_screening:
        raise ValueError("use_native_screening cannot be combined with a custom execution_adapter")
    if progress is not None and not callable(progress):
        raise TypeError("progress must be callable")

    base_plan = plan_context_effects(run, source_ids)
    universe = tuple(base_plan.measurement_source_ids)
    universe_set = set(universe)
    groups = _groups(universe, coarse_groups, coarse_group_size)
    requested = tuple(
        _source_set([source_id], name="drilldown_source_ids", universe=universe_set)[0]
        for source_id in drilldown_source_ids
    )
    if len(requested) != len(set(requested)):
        raise ValueError("drilldown_source_ids must not contain duplicates")
    if isinstance(joint_source_sets, (str, bytes)):
        raise ValueError("joint_source_sets must be an iterable of source-ID sets")
    joint_sets = tuple(
        _source_set(value, name=f"joint_source_sets[{index}]", universe=universe_set, minimum=2)
        for index, value in enumerate(joint_source_sets)
    )
    if len(joint_sets) != len(set(joint_sets)):
        raise ValueError("joint_source_sets must not contain duplicate conditions")

    coarse_adapter = execution_adapter
    if coarse_adapter is None:
        coarse_adapter = DeleteSourceRecordedContinuationScoreAdapter(
            substrate, run=run, run_loader=run_loader,
            refinement_policy=(
                ScoreTokensRefinementPolicy() if use_native_screening else None
            ),
        )

    coarse_candidates = [
        _Candidate("coarse", group, "coarse_candidate") for group in groups
    ]
    coarse_selected, coarse_budget_skips, coarse_new, coarse_cache = _select_with_budget(
        coarse_candidates, budget.max_coarse_arms,
        state=base_plan.execution_state, store=durable,
        budget_reason="coarse_budget_exhausted",
    )
    coarse_experiment = Experiment(
        base=base_plan.execution_state,
        evaluator=ScoreRecordedContinuation(),
        arms=[candidate.intervention for candidate in coarse_selected],
    )
    coarse_phase = (
        "coarse_control_and_native_screening"
        if use_native_screening else "coarse_scalar_scoring"
    )
    _report_progress(
        progress, phase=coarse_phase, completed=0, total=len(coarse_selected),
    )
    coarse_result = run_experiment(
        coarse_experiment, coarse_adapter, include_control=True,
        observation_store=durable, cancel=cancel,
        requested_by={"recipe": "context_effects_coarse_to_fine", "phase": "coarse"},
        diagnostics={"recipe": "context_effects_coarse_to_fine", "phase": "coarse"},
    )
    _report_progress(
        progress, phase=coarse_phase,
        completed=len(coarse_selected), total=len(coarse_selected),
    )
    control = coarse_result.control if isinstance(coarse_result.control, TokenScoreObservation) else None
    coarse_arms: list[ContextEffectsScheduledArm] = []
    for candidate, row in zip(coarse_selected, coarse_result.arms):
        coarse_arms.append(_measured_arm(
            phase="coarse", source_ids=candidate.source_ids,
            selection_reason=candidate.selection_reason, row=row, control=control,
        ))
    coarse_arms.extend(_skipped(candidate) for candidate in coarse_budget_skips)
    coarse_by_key = {arm.source_ids: arm for arm in coarse_arms}

    _report_progress(progress, phase="selecting_refinement", completed=0, total=1)
    requested_set = set(requested)
    ranked_groups = sorted(
        (
            arm for arm in coarse_arms
            if arm.delta_nats is not None
            and arm.evidence_status in {"trusted", "screening_only"}
            and abs(arm.delta_nats) >= float(refine_threshold_nats)
        ),
        key=lambda arm: (-abs(float(arm.delta_nats)), groups.index(arm.source_ids)),
    )
    fine_reasons: dict[str, str] = {source_id: "user_requested_drilldown" for source_id in requested}
    for arm in ranked_groups:
        for source_id in arm.source_ids:
            fine_reasons.setdefault(source_id, "coarse_effect_above_threshold")
    fine_candidates = [
        _Candidate("fine", (source_id,), fine_reasons[source_id])
        for source_id in universe if source_id in fine_reasons
    ]
    # Explicit requests lead, while retaining receipt order inside each tier.
    fine_candidates.sort(key=lambda candidate: (
        0 if candidate.source_ids[0] in requested_set else 1,
        universe.index(candidate.source_ids[0]),
    ))
    fine_selected, fine_budget_skips, fine_new, fine_cache = _select_with_budget(
        fine_candidates, budget.max_fine_arms,
        state=base_plan.execution_state, store=durable, prior=coarse_by_key,
        budget_reason="fine_budget_exhausted",
    )
    selected_fine_ids = {candidate.source_ids[0] for candidate in fine_candidates}
    fine_not_selected = [
        _Candidate(
            "fine", (source_id,),
            "not_in_coarse_groups" if not any(source_id in group for group in groups)
            else "coarse_not_measured_or_below_threshold",
        )
        for source_id in universe if source_id not in selected_fine_ids
    ]

    joint_candidates = [
        _Candidate("joint", source_set, "explicit_joint_intervention")
        for source_set in joint_sets
    ]
    joint_selected, joint_budget_skips, joint_new, joint_cache = _select_with_budget(
        joint_candidates, budget.max_joint_arms,
        state=base_plan.execution_state, store=durable, prior=coarse_by_key,
        budget_reason="joint_budget_exhausted",
    )

    prior_reused: list[ContextEffectsScheduledArm] = []
    phase_two: list[_Candidate] = []
    for candidate in [*fine_selected, *joint_selected]:
        prior = coarse_by_key.get(candidate.key)
        if prior is not None and prior.evidence_status == "trusted":
            prior_reused.append(_reused(candidate, prior))
            continue
        cached = _cached_observation(
            durable, base_plan.execution_state, candidate.intervention,
        )
        if cached is not None:
            prior_reused.append(_reused_observation(candidate, cached, control))
            continue
        phase_two.append(candidate)

    cancelled_after_coarse = _cancelled(cancel)
    if cancelled_after_coarse:
        for item in phase_two:
            target = fine_budget_skips if item.phase == "fine" else joint_budget_skips
            target.append(_Candidate(item.phase, item.source_ids, "cancelled_after_coarse"))
        phase_two = []
    _report_progress(progress, phase="selecting_refinement", completed=1, total=1)

    refinement_result: ExperimentResult | None = None
    measured_phase_two: list[ContextEffectsScheduledArm] = []
    if phase_two:
        refinement_phase = (
            "fine_joint_native_screen_then_scalar_refine"
            if use_native_screening else "fine_joint_scalar_scoring"
        )
        _report_progress(
            progress, phase=refinement_phase, completed=0, total=len(phase_two),
        )
        refinement_experiment = Experiment(
            base=base_plan.execution_state,
            evaluator=ScoreRecordedContinuation(),
            arms=[candidate.intervention for candidate in phase_two],
        )
        refinement_adapter = execution_adapter
        if refinement_adapter is None:
            proof_critical_ids = (
                tuple(arm.arm_id for arm in refinement_experiment.arms)
                if use_native_screening else ()
            )
            refinement_adapter = DeleteSourceRecordedContinuationScoreAdapter(
                substrate, run=run, run_loader=run_loader,
                refinement_policy=(
                    ScoreTokensRefinementPolicy(
                        proof_critical_arm_ids=proof_critical_ids,
                    )
                    if use_native_screening else None
                ),
            )
        refinement_result = run_experiment(
            refinement_experiment, refinement_adapter, include_control=False,
            observation_store=durable, cancel=cancel,
            requested_by={"recipe": "context_effects_coarse_to_fine", "phase": "refinement"},
            diagnostics={"recipe": "context_effects_coarse_to_fine", "phase": "refinement"},
        )
        for candidate, row in zip(phase_two, refinement_result.arms):
            measured_phase_two.append(_measured_arm(
                phase=candidate.phase, source_ids=candidate.source_ids,
                selection_reason=candidate.selection_reason, row=row, control=control,
            ))
        _report_progress(
            progress, phase=refinement_phase,
            completed=len(phase_two), total=len(phase_two),
        )

    all_fine = [arm for arm in [*prior_reused, *measured_phase_two] if arm.phase == "fine"]
    all_fine.extend(_skipped(candidate) for candidate in fine_budget_skips)
    all_fine.extend(_skipped(candidate) for candidate in fine_not_selected)
    all_fine.sort(key=lambda arm: universe.index(arm.source_ids[0]))
    all_joint = [arm for arm in [*prior_reused, *measured_phase_two] if arm.phase == "joint"]
    all_joint.extend(_skipped(candidate) for candidate in joint_budget_skips)
    joint_order = {source_set: index for index, source_set in enumerate(joint_sets)}
    all_joint.sort(key=lambda arm: joint_order[arm.source_ids])
    coarse_arms.sort(key=lambda arm: groups.index(arm.source_ids))
    all_arms = tuple([*coarse_arms, *all_fine, *all_joint])

    grouped_ids = {source_id for group in groups for source_id in group}
    coverage = {
        "source_count": len(universe),
        "source_ids": list(universe),
        "coarse_group_count": len(groups),
        "coarse_grouped_source_ids": [source_id for source_id in universe if source_id in grouped_ids],
        "coarse_ungrouped_source_ids": [source_id for source_id in universe if source_id not in grouped_ids],
        "coarse_trusted_group_count": sum(arm.evidence_status == "trusted" for arm in coarse_arms),
        "coarse_screening_only_group_count": sum(arm.evidence_status == "screening_only" for arm in coarse_arms),
        "coarse_skipped_group_count": sum(arm.evidence_status == "skipped" for arm in coarse_arms),
        "fine_trusted_source_ids": [arm.source_ids[0] for arm in all_fine if arm.evidence_status == "trusted"],
        "fine_screening_only_source_ids": [arm.source_ids[0] for arm in all_fine if arm.evidence_status == "screening_only"],
        "fine_skipped_source_ids": [arm.source_ids[0] for arm in all_fine if arm.evidence_status == "skipped"],
        "requested_drilldown_source_ids": list(requested),
        "requested_drilldowns_measured": [
            arm.source_ids[0] for arm in all_fine
            if arm.source_ids[0] in requested_set and arm.evidence_status in {"trusted", "screening_only"}
        ],
        "joint_requested_count": len(joint_sets),
        "joint_trusted_count": sum(arm.evidence_status == "trusted" for arm in all_joint),
        "joint_screening_only_count": sum(arm.evidence_status == "screening_only" for arm in all_joint),
        "joint_skipped_count": sum(arm.evidence_status == "skipped" for arm in all_joint),
    }
    control_disposition = coarse_result.control_disposition
    executed_by_phase = {
        phase: sum(
            arm.execution_disposition == "executed"
            for arm in all_arms if arm.phase == phase
        )
        for phase in ("coarse", "fine", "joint")
    }
    reused_by_phase = {
        phase: sum(
            arm.execution_disposition == "reused"
            for arm in all_arms if arm.phase == phase
        )
        for phase in ("coarse", "fine", "joint")
    }
    skipped_by_phase = {
        phase: sum(
            arm.evidence_status == "skipped"
            for arm in all_arms if arm.phase == phase
        )
        for phase in ("coarse", "fine", "joint")
    }
    control_executed = int(control_disposition == "executed")
    control_reused = int(control_disposition == "reused")
    control_not_executed = int(control_disposition == "not_executed")
    reused_total = sum(reused_by_phase.values()) + control_reused
    skipped_total = sum(skipped_by_phase.values()) + control_not_executed
    budget_usage = {
        "limits": budget.to_dict(),
        # Reservation is reported separately from work actually dispatched so
        # cancellation after coarse cannot masquerade as fine/joint execution.
        "new_slots_reserved_before_cancellation": {
            "coarse": coarse_new,
            "fine": fine_new,
            "joint": joint_new,
        },
        "cache_or_prior_hits": {
            "coarse": coarse_cache,
            "fine": fine_cache,
            "joint": joint_cache,
        },
        "exhausted": {
            "coarse": any(item.selection_reason == "coarse_budget_exhausted" for item in coarse_budget_skips),
            "fine": any(item.selection_reason == "fine_budget_exhausted" for item in fine_budget_skips),
            "joint": any(item.selection_reason == "joint_budget_exhausted" for item in joint_budget_skips),
        },
        "control_outside_intervention_budget": True,
        "actual_work": {
            "control": {
                "executed": control_executed,
                "reused": control_reused,
                "not_executed": control_not_executed,
            },
            "interventions": {
                "executed_by_phase": executed_by_phase,
                "reused_by_phase": reused_by_phase,
                "skipped_by_phase": skipped_by_phase,
            },
            "model_measurements_executed_total": (
                sum(executed_by_phase.values()) + control_executed
            ),
            "model_measurements_reused_total": reused_total,
            "model_measurements_skipped_total": skipped_total,
            "saved_model_measurements_total": reused_total + skipped_total,
            "saved_work_breakdown": {
                "trusted_cache_or_prior_reuse": reused_total,
                "scheduler_or_cancellation_skip": skipped_total,
            },
        },
    }
    return ContextEffectsCoarseToFineResult(
        run_id=base_plan.run_id,
        source_ids=universe,
        refine_threshold_nats=float(refine_threshold_nats),
        budget=budget,
        arms=all_arms,
        coverage=coverage,
        budget_usage=budget_usage,
        native_screening_requested=use_native_screening,
        coarse_experiment=coarse_result,
        refinement_experiment=refinement_result,
    )


__all__ = [
    "ContextEffectsCoarseToFineResult",
    "ContextEffectsScheduledArm",
    "ContextEffectsSearchBudget",
    "SCHEMA_VERSION",
    "measure_context_effects_coarse_to_fine",
]
