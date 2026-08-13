"""Deterministic, model-free worlds for Context Dependence search tests.

These are *evaluation fixtures*, not a search policy and not a substitute for
the production teacher-forced measurement primitive.  Each world defines the
exact removal effect for every subset of its stable source IDs.  A search test
can therefore spend a bounded number of ``SyntheticScorer.measure`` calls and
be evaluated on the source sets it actually measured.

The effect convention matches the Context Dependence study:

    delta_nats = full_context_target_logp - removed_context_target_logp

Thus a positive effect means the recorded target became less likely after the
specified supplied sources were removed.  ``material_source_sets`` is a
benchmark oracle (effect at or above ``MATERIAL_EFFECT_FLOOR_NATS``); it is
not a product-facing semantic label.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Callable, Iterable, Mapping


MATERIAL_EFFECT_FLOOR_NATS = 1.0
"""Fixed fixture threshold used only to classify synthetic expected effects."""


SourceSet = frozenset[str]


class UnknownSyntheticSourceIDError(ValueError):
    """Raised when a synthetic experiment is not over this case's receipt IDs."""


@dataclass(frozen=True)
class SyntheticSource:
    """A source with stable receipt-like identity and deliberately visible origin."""

    source_id: str
    text: str
    origin: str = "rag"
    position: int = 0
    token_count: int = 16


@dataclass(frozen=True)
class SyntheticTarget:
    """The fixed recorded continuation target used by a synthetic world."""

    text: str = "The recorded answer."
    unicode_range: tuple[int, int] = (0, 20)
    recorded_token_range: tuple[int, int] = (0, 3)
    recorded_prefix_range: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class SyntheticMeasurement:
    """One directly measured synthetic source-set experiment."""

    case_id: str
    removed_source_ids: tuple[str, ...]
    baseline_target_logp: float
    target_logp: float
    delta_nats: float
    per_target_token_delta_nats: tuple[float, ...]
    score_passes: int = 1
    provenance: str = "measured"

    @property
    def removed_source_set(self) -> SourceSet:
        return frozenset(self.removed_source_ids)


@dataclass(frozen=True)
class SyntheticContextDependenceCase:
    """A complete, directly scoreable Context Dependence test world.

    ``expected_direct_effects`` intentionally contains every source subset,
    including the empty set.  This makes coalition behavior auditable without
    requiring a model, random seed, external service, or a particular search
    strategy.
    """

    case_id: str
    description: str
    sources: tuple[SyntheticSource, ...]
    target: SyntheticTarget
    expected_direct_effects: Mapping[SourceSet, float]
    material_source_sets: frozenset[SourceSet]
    non_material_source_sets: frozenset[SourceSet]
    baseline_target_logp: float = -12.0
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(source.source_id for source in self.sources)

    def canonical_source_set(self, source_ids: Iterable[str]) -> tuple[str, ...]:
        """Validate a source set and return receipt-order canonical IDs."""
        requested = frozenset(source_ids)
        unknown = requested.difference(self.source_ids)
        if unknown:
            raise UnknownSyntheticSourceIDError(
                f"{self.case_id}: unknown source IDs: {', '.join(sorted(unknown))}")
        return tuple(source_id for source_id in self.source_ids if source_id in requested)

    def expected_effect(self, removed_source_ids: Iterable[str]) -> float:
        source_set = frozenset(self.canonical_source_set(removed_source_ids))
        return self.expected_direct_effects[source_set]

    def scorer(self) -> "SyntheticScorer":
        return SyntheticScorer(self)


class SyntheticScorer:
    """A small fake teacher-forced scorer with explicit score-pass accounting.

    It has no generation API and does not import any model runtime.  Repeating
    an experiment deliberately consumes another pass: cache behavior is a
    property to test in the search/study implementation, not a hidden fixture
    behavior.
    """

    def __init__(self, case: SyntheticContextDependenceCase):
        self.case = case
        self.calls: list[SyntheticMeasurement] = []

    @property
    def passes_consumed(self) -> int:
        return sum(call.score_passes for call in self.calls)

    def measure(self, removed_source_ids: Iterable[str] = ()) -> SyntheticMeasurement:
        canonical_ids = self.case.canonical_source_set(removed_source_ids)
        delta = self.case.expected_direct_effects[frozenset(canonical_ids)]
        token_count = self.case.target.recorded_token_range[1] - self.case.target.recorded_token_range[0]
        if token_count <= 0:
            raise ValueError("synthetic target must contain at least one recorded token")
        # Keep the result deterministic while making the aggregate invariant
        # exact even when binary floating point cannot represent delta / n.
        per_token = tuple(delta / token_count for _ in range(token_count - 1))
        per_token += (delta - sum(per_token),)
        measurement = SyntheticMeasurement(
            case_id=self.case.case_id,
            removed_source_ids=canonical_ids,
            baseline_target_logp=self.case.baseline_target_logp,
            target_logp=self.case.baseline_target_logp - delta,
            delta_nats=delta,
            per_target_token_delta_nats=per_token,
        )
        self.calls.append(measurement)
        return measurement


@dataclass(frozen=True)
class BenchmarkEvaluation:
    """Budget-aware assessment of a search's *directly measured* experiments."""

    case_id: str
    passes_requested: int | None
    passes_consumed: int
    passes_remaining: int | None
    exceeded_budget: bool
    directly_measured_source_sets: frozenset[SourceSet]
    directly_measured_material_sets: frozenset[SourceSet]
    missed_material_source_sets: frozenset[SourceSet]
    low_singleton_source_ids: frozenset[str]
    source_ids_falsely_declared_irrelevant: frozenset[str]

    @property
    def all_material_sets_discovered(self) -> bool:
        return not self.missed_material_source_sets

    @property
    def incorrectly_treated_low_singletons_as_irrelevant(self) -> bool:
        return bool(self.source_ids_falsely_declared_irrelevant)


def evaluate_direct_measurements(
    case: SyntheticContextDependenceCase,
    measurements: Iterable[SyntheticMeasurement],
    *,
    passes_requested: int | None,
    low_effect_floor_nats: float = MATERIAL_EFFECT_FLOOR_NATS,
    declared_irrelevant_source_ids: Iterable[str] = (),
) -> BenchmarkEvaluation:
    """Evaluate a bounded search without inferring anything from unmeasured sets.

    ``declared_irrelevant_source_ids`` is optional on purpose.  Search code
    that never emits an irrelevance conclusion is not penalized merely because
    a singleton is small.  When it does emit one, this helper flags sources
    whose weak singleton participates in a material coalition in the fixture.
    """
    measured = tuple(measurements)
    if any(measurement.case_id != case.case_id for measurement in measured):
        raise ValueError("cannot evaluate measurements from a different synthetic case")
    measured_sets = frozenset(measurement.removed_source_set for measurement in measured)
    passes_consumed = sum(measurement.score_passes for measurement in measured)
    if passes_requested is not None and passes_requested < 0:
        raise ValueError("passes_requested must be non-negative or None")
    source_ids = frozenset(case.source_ids)
    declared = frozenset(declared_irrelevant_source_ids)
    unknown_declared = declared.difference(source_ids)
    if unknown_declared:
        raise UnknownSyntheticSourceIDError(
            f"{case.case_id}: unknown declared source IDs: {', '.join(sorted(unknown_declared))}")
    low_singletons = frozenset(
        source_id for source_id in case.source_ids
        if abs(case.expected_direct_effects[frozenset((source_id,))]) < low_effect_floor_nats
        and any(source_id in source_set for source_set in case.material_source_sets)
    )
    remaining = None if passes_requested is None else max(0, passes_requested - passes_consumed)
    return BenchmarkEvaluation(
        case_id=case.case_id,
        passes_requested=passes_requested,
        passes_consumed=passes_consumed,
        passes_remaining=remaining,
        exceeded_budget=passes_requested is not None and passes_consumed > passes_requested,
        directly_measured_source_sets=measured_sets,
        directly_measured_material_sets=case.material_source_sets.intersection(measured_sets),
        missed_material_source_sets=case.material_source_sets.difference(measured_sets),
        low_singleton_source_ids=low_singletons,
        source_ids_falsely_declared_irrelevant=declared.intersection(low_singletons),
    )


def _all_source_sets(source_ids: tuple[str, ...]) -> tuple[SourceSet, ...]:
    return tuple(
        frozenset(group)
        for size in range(len(source_ids) + 1)
        for group in combinations(source_ids, size)
    )


def _case(
    case_id: str,
    description: str,
    sources: tuple[SyntheticSource, ...],
    effect: Callable[[SourceSet], float],
    *,
    target: SyntheticTarget | None = None,
    metadata: Mapping[str, object] | None = None,
) -> SyntheticContextDependenceCase:
    source_ids = tuple(source.source_id for source in sources)
    if len(source_ids) != len(set(source_ids)):
        raise ValueError(f"{case_id}: source IDs must be unique")
    effects = {source_set: float(effect(source_set)) for source_set in _all_source_sets(source_ids)}
    material = frozenset(
        source_set for source_set, delta in effects.items()
        if delta >= MATERIAL_EFFECT_FLOOR_NATS
    )
    return SyntheticContextDependenceCase(
        case_id=case_id,
        description=description,
        sources=sources,
        target=target or SyntheticTarget(),
        expected_direct_effects=MappingProxyType(effects),
        material_source_sets=material,
        non_material_source_sets=frozenset(set(effects).difference(material)),
        metadata=MappingProxyType(dict(metadata or {})),
    )


def _source(source_id: str, text: str, *, origin: str = "rag", position: int = 0,
            token_count: int = 16) -> SyntheticSource:
    return SyntheticSource(source_id, text, origin, position, token_count)


def _duplicate_effect(
    source_a: str,
    source_b: str,
    single_a: float,
    single_b: float,
    joint: float,
) -> Callable[[SourceSet], float]:
    def effect(removed: SourceSet) -> float:
        if source_a in removed and source_b in removed:
            return joint
        if source_a in removed:
            return single_a
        if source_b in removed:
            return single_b
        return 0.0
    return effect


def _three_way_coalition_effect(removed: SourceSet) -> float:
    coalition = frozenset(("coalition-a", "coalition-b", "coalition-c"))
    removed_count = len(removed.intersection(coalition))
    if removed_count == 3:
        return 9.20
    if removed_count == 2:
        return 0.05
    if removed_count == 1:
        return 0.02
    return 0.0


def _multi_hop_effect(removed: SourceSet) -> float:
    hops = frozenset(("hop-fact", "hop-relation", "hop-bridge"))
    removed_hops = removed.intersection(hops)
    if len(removed_hops) == 3:
        return 9.00
    if len(removed_hops) == 2:
        return 6.50
    if len(removed_hops) == 1:
        return {"hop-fact": 4.00, "hop-relation": 3.80, "hop-bridge": 3.20}[next(iter(removed_hops))]
    return 0.0


def _long_position_case() -> SyntheticContextDependenceCase:
    sources = (
        _source("early-evidence", "The old archive says the answer is Selene.", position=0, token_count=30),
        *tuple(_source(f"middle-filler-{index}", f"Long unrelated note {index}.",
                       position=index, token_count=180) for index in range(1, 9)),
        _source("late-evidence", "Most recent correction: the answer is Selene.", position=9, token_count=30),
    )
    return _case(
        "position_sensitive_long_context",
        "A long receipt gives a late correction a much larger removal effect than early evidence.",
        sources,
        _duplicate_effect("early-evidence", "late-evidence", 0.35, 5.80, 7.00),
        metadata={"long_context": True, "position_sensitive": True, "late_source_id": "late-evidence"},
    )


def _cases() -> tuple[SyntheticContextDependenceCase, ...]:
    return (
        _case(
            "irrelevant_filler",
            "Every supplied source is filler for the fixed recorded target.",
            (_source("filler-a", "Weather notes."), _source("filler-b", "Meeting agenda.")),
            lambda _removed: 0.0,
        ),
        _case(
            "one_necessary_source",
            "A single supplied source is necessary for the recorded target; a sibling is filler.",
            (_source("necessary", "The launch code is 72."), _source("filler", "The lobby has plants.")),
            lambda removed: 6.40 if "necessary" in removed else 0.0,
        ),
        _case(
            "exact_duplicate_evidence",
            "Two byte-identical supplied sources only fail jointly.",
            (_source("duplicate-a", "The capital is Oslo."), _source("duplicate-b", "The capital is Oslo.")),
            _duplicate_effect("duplicate-a", "duplicate-b", 0.03, 0.04, 7.30),
            metadata={"duplicate_kind": "exact"},
        ),
        _case(
            "paraphrased_substitutable_duplicate_evidence",
            "Two differently worded sources provide substitutable evidence.",
            (_source("paraphrase-a", "The meeting moved to Tuesday."),
             _source("paraphrase-b", "The gathering is now scheduled for Tuesday.")),
            _duplicate_effect("paraphrase-a", "paraphrase-b", 0.05, 0.06, 6.90),
            metadata={"duplicate_kind": "paraphrased_substitutable"},
        ),
        _case(
            "either_a_or_b_sufficiency",
            "Either independent evidence path is sufficient, while removing both is costly.",
            (_source("path-a", "The signed ledger reports 18 units."),
             _source("path-b", "The warehouse manifest reports 18 units.")),
            _duplicate_effect("path-a", "path-b", 0.02, 0.02, 5.50),
            metadata={"sufficiency": "either_source"},
        ),
        _case(
            "a_and_b_complementarity",
            "Both separately necessary components support the answer, with a larger joint removal effect.",
            (_source("component-a", "Cipher key: ORBIT."), _source("component-b", "Shift amount: 3.")),
            _duplicate_effect("component-a", "component-b", 4.20, 4.00, 8.60),
            metadata={"dependency": "and"},
        ),
        _case(
            "three_way_coalition",
            "Only deleting all three weakly substitutable sources materially changes the target likelihood.",
            (_source("coalition-a", "Record A confirms the date."),
             _source("coalition-b", "Record B confirms the date."),
             _source("coalition-c", "Record C confirms the date.")),
            _three_way_coalition_effect,
        ),
        _case(
            "multi_hop_evidence",
            "A fact, relation, and bridge form a three-hop answer chain.",
            (_source("hop-fact", "Nera is in Vela."), _source("hop-relation", "Vela is north of Sorn."),
             _source("hop-bridge", "The requested route starts in Sorn.")),
            _multi_hop_effect,
            metadata={"dependency": "multi_hop"},
        ),
        _case(
            "parametric_knowledge_overlap",
            "The answer remains correct after supplied context is removed because the fake model already knows it.",
            (_source("weak-context", "Paris is the capital of France."),),
            lambda removed: 0.08 if "weak-context" in removed else 0.0,
            metadata={"recorded_answer_correct": True, "parametric_knowledge_overlap": True},
        ),
        _case(
            "coreference_broken_by_deletion",
            "Deleting the antecedent breaks a later pronoun reference in the remaining source.",
            (_source("antecedent", "Dr. Morrow founded the clinic."),
             _source("coreferent-fact", "She opened it in 1998.")),
            _duplicate_effect("antecedent", "coreferent-fact", 5.40, 0.35, 5.80),
            metadata={"deletion_breaks_coreference": True},
        ),
        _long_position_case(),
        _case(
            "repeated_user_and_rag_evidence",
            "The same evidence appears once in user text and once in supplied RAG context.",
            (_source("user-evidence", "User: the access code is 4815.", origin="user"),
             _source("rag-evidence", "Retrieved note: the access code is 4815.", origin="rag")),
            _duplicate_effect("user-evidence", "rag-evidence", 0.03, 0.04, 7.30),
            metadata={"duplicate_across_origins": True},
        ),
        _case(
            "later_answer_depends_on_recorded_prefix",
            "A later target region is carried mainly by the conditioned earlier recorded answer prefix.",
            (_source("weak-context", "The answer begins with a name."),
             _source("unrelated-context", "A separate status update.")),
            lambda removed: 0.10 if "weak-context" in removed else 0.0,
            target=SyntheticTarget(
                text="therefore it was approved.", unicode_range=(24, 49),
                recorded_token_range=(4, 8), recorded_prefix_range=(0, 4)),
            metadata={
                "conditioned_prefix_dependency_nats": 8.10,
                "target_depends_primarily_on": "recorded_answer_prefix",
            },
        ),
    )


ALL_CONTEXT_DEPENDENCE_CASES = _cases()
CONTEXT_DEPENDENCE_CASES: Mapping[str, SyntheticContextDependenceCase] = MappingProxyType(
    {case.case_id: case for case in ALL_CONTEXT_DEPENDENCE_CASES})


def case_by_id(case_id: str) -> SyntheticContextDependenceCase:
    """Return one named benchmark world with a useful error for typos."""
    try:
        return CONTEXT_DEPENDENCE_CASES[case_id]
    except KeyError as exc:
        raise KeyError(f"unknown Context Dependence benchmark case: {case_id}") from exc
