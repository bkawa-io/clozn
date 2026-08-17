"""Durable Minimal Context v1 result/proof contract coverage."""
from __future__ import annotations

from copy import deepcopy

import clozn.runs.store as run_store
from clozn.experiments.search import (
    Candidate, EXACT_MINIMUM, SearchEvidenceRef, SearchTrial, certify_exact_minimum,
)
from clozn.recipes.minimal_context import run_minimal_context
from clozn.recipes.minimal_context_result_store import MinimalContextResultStore, current_binding
from clozn.runs.context_units import build_context_unit_manifest


def _engine():
    class Engine:
        @staticmethod
        def apply_template_info(messages):
            return {"prompt_tokens": sum(len(str(item.get("content", ""))) for item in messages)}

    return Engine()


def test_exact_minimum_is_a_pure_complete_ledger_certificate():
    universe = ("A", "B")
    trials = []
    classifications = {
        (): ("preserves", "exact_preserved", 0),
        ("A",): ("diverged", "diverged", 1),
        ("B",): ("diverged", "diverged", 1),
        ("A", "B"): ("preserves", "exact_preserved", 2),
    }
    for ordinal, (retained, (classification, status, cost)) in enumerate(classifications.items()):
        trials.append(SearchTrial(
            ordinal=ordinal, stage="fixture", retained_ids=retained, cost=cost,
            classification=classification,
            evidence_ref=SearchEvidenceRef(
                experiment_id="exp", arm_id=f"arm-{ordinal}", observation_id=f"obs-{ordinal}",
                observation_status=status,
            ),
        ))
    proof = certify_exact_minimum(
        universe, trials, original_candidate=Candidate(universe, 2),
        winner=Candidate((), 0),
    )
    assert proof is not None
    assert proof["certificate"] == EXACT_MINIMUM
    assert proof["candidate_space_size"] == 4
    assert proof["directly_classified_subset_count"] == 4

    incomplete = certify_exact_minimum(universe, trials[:-1], original_candidate=Candidate(universe, 2))
    assert incomplete is None


def test_result_store_round_trip_is_immutable_and_binding_is_readable(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import ProbeSubstrate, _run

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _ = _run()
    run["context_units"] = build_context_unit_manifest(run)
    store = MinimalContextResultStore()
    result = run_minimal_context(
        run, substrate=ProbeSubstrate(), engine=_engine(),
        max_new_counterfactual_observations=0,
    )
    # The search above intentionally has no ObservationStore because this
    # fixture exercises the derived result document itself.  Persisting a
    # result with no winner evidence is still safe historical metadata.
    store.put(result)
    assert store.put(result) == result.result_id
    loaded = MinimalContextResultStore().get(result.result_id)
    assert loaded is not None
    assert loaded.to_json() == result.to_json()
    assert current_binding(loaded, run)["status"] == "current"
    assert store.list_for_run(run["id"], limit=10)[0]["result_id"] == result.result_id
    assert store.latest_for_search(result.search_id).result_id == result.result_id

    stale = deepcopy(run)
    stale["context_units"] = deepcopy(run["context_units"])
    stale["context_units"]["universe_mutation"] = True
    # A changed receipt/execution is the binding that governs actions; the
    # historical result remains readable even when it is stale.
    stale["messages"] = deepcopy(run["messages"])
    stale["messages"][0]["content"] += " changed"
    assert current_binding(loaded, stale)["status"] == "stale"


def test_result_has_inspectable_reduction_proof_and_source_records(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import ProbeSubstrate, _run
    from clozn.experiments.persistence import ObservationStore

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _ = _run()
    run["context_units"] = build_context_unit_manifest(run)
    result = run_minimal_context(
        run, substrate=ProbeSubstrate(), engine=_engine(), observation_store=ObservationStore(),
        max_new_counterfactual_observations=4,
    )
    document = result.to_dict()
    assert document["schema_version"] == "clozn.minimal-context-search-result.v2"
    assert document["result_id"].startswith("mcres_")
    assert document["reduction"]["objective"] == "rendered_prompt_tokens.v1"
    assert document["source_inspection"]
    assert {item["disposition"] for item in document["source_inspection"]} <= {"retained", "removed"}
    assert document["proof"]["trajectory"] == document["trajectory"]
    assert document["experiment_accounting"]["candidate_trials"] >= 0


def test_full_context_winner_is_a_typed_noop_for_materialization(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import ProbeSubstrate, _run
    from clozn.experiments.persistence import ObservationStore
    from clozn.server.routes import minimal_context as route

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _ = _run()
    run["context_units"] = build_context_unit_manifest(run)
    result = run_minimal_context(
        run, substrate=ProbeSubstrate(), engine=_engine(), observation_store=ObservationStore(),
        max_new_counterfactual_observations=0,
    )
    result_store = MinimalContextResultStore()
    result_store.put(result)
    monkeypatch.setattr(run_store, "get_run", lambda _run_id: run)

    class Handler:
        status = None
        body = None

        def _json(self, status, body, **_kwargs):
            self.status, self.body = status, body

    handler = Handler()
    assert route.try_post(
        handler, f"/runs/{run['id']}/minimal-context/results/{result.result_id}/materialize", {},
    ) is True
    assert handler.status == 409
    assert handler.body["code"] == "no_reduction_to_materialize"
    assert run_store.list_runs(20) == []
