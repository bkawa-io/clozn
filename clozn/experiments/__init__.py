"""Experimental kernel exports plus lazy access to the legacy registry.

The registry/dispatcher in :mod:`clozn.experiments.experiment` is retained as a
temporary parity oracle.  It is deliberately lazy: importing a new-kernel
module does not import or initialize that legacy API.
"""
from importlib import import_module


_LEGACY_EXPORTS = frozenset({"REGISTRY", "catalog", "run_experiment", "substrate_ok"})


def __getattr__(name):
    if name in _LEGACY_EXPORTS:
        legacy = import_module(".experiment", __name__)
        return getattr(legacy, name)
    raise AttributeError(name)


from .evaluators import ExactReferenceMatch, ScoreRecordedContinuation, Generate
from .batch import ArmExecutionOutcome, ArmExecutionRequest, BatchExecutionError, BatchExecutionResult
from .shared_parent import (
    SharedParentParityError, SharedParentSessionClient, SharedParentSessionError,
    assert_evidence_parity,
)
from .context_search import ContextSearchDispatcher, ContextSearchUnavailable
from .execution import (
    DeleteSourceExactReferenceAdapter,
    ExecutionAdapter,
    ExactReferenceMatchAdapter,
    resolve_delete_source,
)
from .interventions import DeleteSource
from .interventions import ForceToken, Intervention, intervention_from_dict
from .kernel import Experiment, ExperimentArm
from .materialize import MaterializeBranch, materialize_arm, materialize_generated_observation
from .generation import DeleteSourceGenerateAdapter, GenerateExecutionAdapter, GenerateExecutionError
from .observations import (
    Observation, ObservationError, ObservationIntegrityError, GeneratedObservation, TokenScoreDelta,
    TokenScoreObservation, condition_for_intervention, execution_observation_identity,
    observation_from_dict, observation_identity,
)
from .persistence import (
    ARM_STATES, EXPERIMENT_STATES, EXPERIMENT_STORE_SCHEMA_VERSION,
    ExperimentArmView, ExperimentPersistenceError, ExperimentView,
    ObservationNotFound, ObservationPersistenceError, ObservationStore,
)
from .projection import AnswerSpanEffect, ProjectionError, project_answer_effects, project_answer_selection
from .context_investigation import (
    AnswerSelectionProjectionUnavailable, ContextInvestigationError, ContextInvestigationStale,
    ContextInvestigationUnavailable, DEFAULT_MEASUREMENT_FLOOR_NATS, DISPLAY_COORDINATE_BASIS,
    build_context_investigation_reader, project_locus_details, project_source_loci,
    query_answer_effects,
)
from .runner import ExperimentResult, run_experiment as experimental_run_experiment
from .scoring import (
    DeleteSourceRecordedContinuationScoreAdapter, DeleteSourceScoreAdapter,
    RecordedContinuationScoreAdapter, ScoreRecordedContinuationAdapter,
)
from .selections import (
    AnswerSelection, AnswerSelectionUnavailable, ContextSelection, ResolvedAnswerSelection,
    resolve_answer_selection,
)
from .state import ExecutionState
from .state_ref import (
    AnswerTokenBoundary, RecordedAnswerBoundary, ResolvedState, StateRef, StateRefError,
    enumerate_answer_boundaries, list_answer_token_boundaries, operation_readiness, resolve_state,
)
from .search import (
    BEST_VERIFIED, INCLUSION_MINIMUM, SearchBudget, SearchEvidenceRef, SearchResult,
    SearchTrial, SearchTrajectoryEntry, run_adaptive_search,
)
from .suite import (MANIFEST_SCHEMA, RESULT_SCHEMA, list_result_paths, load_manifest, load_result,
                    results_directory, run_manifest, select_cells, validate_manifest, validate_result)

ExperimentStore = ObservationStore

__all__ = ["REGISTRY", "catalog", "run_experiment", "substrate_ok", "MANIFEST_SCHEMA",
           "RESULT_SCHEMA", "list_result_paths", "load_manifest", "load_result", "results_directory",
           "run_manifest", "select_cells", "validate_manifest", "validate_result",
           "ExecutionState", "StateRef", "AnswerTokenBoundary", "RecordedAnswerBoundary", "ResolvedState", "StateRefError",
           "enumerate_answer_boundaries", "list_answer_token_boundaries", "operation_readiness", "resolve_state", "ContextSelection", "DeleteSource", "ForceToken", "Intervention",
           "intervention_from_dict", "ExactReferenceMatch", "ScoreRecordedContinuation", "Generate",
           "AnswerSelection", "ResolvedAnswerSelection",
           "AnswerSelectionUnavailable", "Experiment", "ExperimentArm", "Observation",
           "ObservationError", "ObservationIntegrityError", "GeneratedObservation", "condition_for_intervention",
           "execution_observation_identity", "observation_from_dict", "observation_identity", "TokenScoreObservation",
           "TokenScoreDelta", "AnswerSpanEffect", "ExperimentResult", "ExperimentArmView",
           "ExperimentView", "ExperimentStore", "ObservationStore", "ObservationNotFound", "ObservationPersistenceError",
           "ExperimentPersistenceError", "ARM_STATES", "EXPERIMENT_STATES",
           "EXPERIMENT_STORE_SCHEMA_VERSION",
           "ProjectionError",
           "ExecutionAdapter", "DeleteSourceExactReferenceAdapter", "ExactReferenceMatchAdapter",
           "DeleteSourceRecordedContinuationScoreAdapter", "ScoreRecordedContinuationAdapter",
           "DeleteSourceScoreAdapter", "RecordedContinuationScoreAdapter",
           "resolve_delete_source", "resolve_answer_selection", "project_answer_effects",
           "project_answer_selection", "experimental_run_experiment", "materialize_arm",
           "materialize_generated_observation", "MaterializeBranch", "GenerateExecutionAdapter",
           "DeleteSourceGenerateAdapter", "GenerateExecutionError", "BEST_VERIFIED", "INCLUSION_MINIMUM", "SearchBudget",
           "SearchEvidenceRef", "SearchResult", "SearchTrial", "SearchTrajectoryEntry",
           "run_adaptive_search", "ContextSearchDispatcher", "ContextSearchUnavailable",
           "ContextInvestigationError", "ContextInvestigationStale", "ContextInvestigationUnavailable",
           "AnswerSelectionProjectionUnavailable", "DEFAULT_MEASUREMENT_FLOOR_NATS",
           "DISPLAY_COORDINATE_BASIS", "build_context_investigation_reader", "project_locus_details",
           "project_source_loci", "query_answer_effects",
           "ArmExecutionOutcome", "ArmExecutionRequest", "BatchExecutionError",
           "BatchExecutionResult", "SharedParentSessionClient", "SharedParentSessionError"]
