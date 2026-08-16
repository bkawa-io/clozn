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


from .evaluators import ExactReferenceMatch, ScoreRecordedContinuation
from .execution import (
    DeleteSourceExactReferenceAdapter,
    ExecutionAdapter,
    ExactReferenceMatchAdapter,
    resolve_delete_source,
)
from .interventions import DeleteSource
from .kernel import Experiment, ExperimentArm
from .materialize import MaterializeBranch, materialize_arm
from .observations import Observation
from .observations import TokenScoreDelta, TokenScoreObservation
from .projection import AnswerSpanEffect, ProjectionError, project_answer_effects, project_answer_selection
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
from .suite import (MANIFEST_SCHEMA, RESULT_SCHEMA, list_result_paths, load_manifest, load_result,
                    results_directory, run_manifest, select_cells, validate_manifest, validate_result)

__all__ = ["REGISTRY", "catalog", "run_experiment", "substrate_ok", "MANIFEST_SCHEMA",
           "RESULT_SCHEMA", "list_result_paths", "load_manifest", "load_result", "results_directory",
           "run_manifest", "select_cells", "validate_manifest", "validate_result",
           "ExecutionState", "ContextSelection", "DeleteSource", "ExactReferenceMatch",
           "ScoreRecordedContinuation", "AnswerSelection", "ResolvedAnswerSelection",
           "AnswerSelectionUnavailable", "Experiment", "ExperimentArm", "Observation",
           "TokenScoreObservation", "TokenScoreDelta", "AnswerSpanEffect", "ExperimentResult",
           "ProjectionError",
           "ExecutionAdapter", "DeleteSourceExactReferenceAdapter", "ExactReferenceMatchAdapter",
           "DeleteSourceRecordedContinuationScoreAdapter", "ScoreRecordedContinuationAdapter",
           "DeleteSourceScoreAdapter", "RecordedContinuationScoreAdapter",
           "resolve_delete_source", "resolve_answer_selection", "project_answer_effects",
           "project_answer_selection", "experimental_run_experiment", "materialize_arm",
           "MaterializeBranch"]
