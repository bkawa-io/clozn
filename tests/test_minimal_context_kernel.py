"""Batch 5 adaptive Minimal Context kernel coverage."""
from __future__ import annotations

import types

from clozn.experiments.search import BEST_VERIFIED, INCLUSION_MINIMUM, run_adaptive_search
from clozn.recipes.minimal_context import run_minimal_context
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.materialize import materialize_arm
from clozn.runs.context_units import build_context_unit_manifest
from clozn.runs import store as run_store


def _search_run():
    return {
        "id": "batch5-model-free",
        "messages": [],
    }


def test_search_policy_never_infers_non_monotone_results_or_global_minimum():
    universe = ("A", "B", "C")
    # Retained C preserves, while retained B,C (remove A) diverges.  The
    # policy may nominate either based on its bounded trajectory, but it must
    # use only the returned direct classifications.
    preserving = {frozenset(universe), frozenset({"C"})}
    calls = []

    def prepare(retained):
        retained = tuple(retained)
        return {"retained_ids": retained, "cost": len(retained), "payload": retained}

    def probe_many(candidates):
        rows = []
        for candidate in candidates:
            retained = frozenset(candidate["retained_ids"])
            calls.append(retained)
            rows.append({
                "status": "matched" if retained in preserving else "diverged",
                "experiment_id": "exp-" + "-".join(candidate["retained_ids"]),
                "arm_id": "arm",
                "observation_id": "obs-" + "-".join(candidate["retained_ids"]),
                "disposition": "executed",
            })
        return rows

    result = run_adaptive_search(universe, 20, prepare, probe_many)
    assert result.certificate in {BEST_VERIFIED, INCLUSION_MINIMUM}
    assert result.certificate != "EXACT_MINIMUM"
    assert all(trial.classification in {"preserves", "diverged", "unknown"} for trial in result.trials)
    # No result serialization contains a direct evidence body; only refs.
    encoded = result.to_dict()
    assert all(set(trial["evidence"]) <= {
        "disposition", "experiment_id", "arm_id", "observation_id", "observation_status",
    } for trial in encoded["trials"] if trial["evidence"])
    assert calls


def test_minimal_context_uses_generic_experiments_and_reuses_store_without_new_calls(tmp_path, monkeypatch):
    # Importing these fixtures here keeps this test independent of the old
    # Minimal Context orchestration and uses only its trusted run fixture.
    from tests.test_experiment_kernel import _run, ProbeSubstrate

    class Engine:
        @staticmethod
        def apply_template_info(messages):
            return {"prompt_tokens": sum(len(str(item.get("content", ""))) for item in messages)}

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _source_ids = _run()
    run["context_units"] = build_context_unit_manifest(run)
    store = ObservationStore()
    substrate = ProbeSubstrate()
    first = run_minimal_context(
        run, substrate=substrate, engine=Engine(), observation_store=store,
        max_new_counterfactual_observations=20,
    )
    first_calls = len(substrate.calls)
    second = run_minimal_context(
        run, substrate=substrate, engine=Engine(), observation_store=store,
        max_new_counterfactual_observations=20,
    )
    assert first.certificate == second.certificate
    assert first.best == second.best
    assert second.budget.used_new_executions == 0
    assert second.budget.reused_observation_count > 0
    assert len(substrate.calls) == first_calls
    assert run_store.list_runs(100) == []


def test_minimal_context_without_faithful_prompt_token_seam_is_typed_unavailable():
    from tests.test_experiment_kernel import _run, ProbeSubstrate

    run, _source_ids = _run()
    run["context_units"] = build_context_unit_manifest(run)
    result = run_minimal_context(
        run, substrate=ProbeSubstrate(), max_new_counterfactual_observations=2,
    )
    assert result.status == "unavailable"
    assert result.reason_code == "rendered_prompt_token_count_unavailable"


def test_minimal_context_winner_points_to_materializable_generic_arm(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import _run, ProbeSubstrate, GenerationSubstrate

    class Engine:
        @staticmethod
        def apply_template_info(messages):
            return {"prompt_tokens": len(messages)}

    run, _source_ids = _run()
    run["context_units"] = build_context_unit_manifest(run)
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    store = ObservationStore()
    result = run_minimal_context(
        run, substrate=ProbeSubstrate(), engine=Engine(), observation_store=store,
        max_new_counterfactual_observations=10,
    )
    assert result.status == "completed"
    assert result.best is not None
    assert result.best.experiment_id is not None
    assert result.best.arm_id is not None
    assert result.best.observation_id is not None
    assert result.to_dict()["best"]["experiment_id"] == result.best.experiment_id

    # The search itself has already finished; generic materialization is the
    # only operation that is allowed to create the child Run.
    materialized = materialize_arm(
        run, result.best.experiment_id, result.best.arm_id,
        substrate=GenerationSubstrate(), observation_id=result.best.observation_id,
        require_preserved=True, observation_store=store,
    )
    assert materialized["state"] == "completed"
    assert materialized["parent_run_id"] == run["id"]
