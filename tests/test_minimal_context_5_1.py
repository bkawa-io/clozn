"""Batch 5.1 semantic closure and product cutover coverage."""
from __future__ import annotations

from copy import deepcopy

import clozn.runs.store as run_store
from clozn.experiments.effective_prompt import inject_block
from clozn.experiments.search import BEST_VERIFIED, run_adaptive_search
from clozn.experiments.persistence import ObservationStore
from clozn.recipes.minimal_context import run_minimal_context
from clozn.runs.context_receipt import build_context_receipt
from clozn.runs.context_units import build_context_unit_manifest


def _engine():
    class Engine:
        def __init__(self):
            self.inputs = []

        def apply_template_info(self, messages):
            copied = [dict(item) for item in messages]
            self.inputs.append(copied)
            return {"prompt": "rendered", "prompt_tokens": sum(len(str(item.get("content", ""))) for item in copied)}

    return Engine()


def test_executed_unavailable_and_failed_arms_each_charge_new_budget():
    statuses = iter(("unavailable", "failed", "diverged"))
    calls = []

    def prepare(retained):
        retained = tuple(retained)
        return {"retained_ids": retained, "cost": len(retained), "payload": retained}

    def probe_many(candidates):
        rows = []
        for candidate in candidates:
            calls.append(tuple(candidate["retained_ids"]))
            status = "matched" if tuple(candidate["retained_ids"]) == ("A", "B", "C") else next(statuses)
            rows.append({"status": status, "disposition": "executed"})
        return rows

    result = run_adaptive_search(("A", "B", "C"), 3, prepare, probe_many)

    assert result.budget.used_new_executions == 3
    assert len(calls[1:]) == 3
    assert [trial.disposition for trial in result.trials[1:]] == ["executed"] * 3
    assert result.certificate == BEST_VERIFIED


def test_zero_budget_reuses_completed_observations_and_derives_same_result(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import _run, ProbeSubstrate

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _source_ids = _run()
    run["context_units"] = build_context_unit_manifest(run)
    store = ObservationStore()
    first_substrate = ProbeSubstrate()
    first = run_minimal_context(
        run, substrate=first_substrate, engine=_engine(), observation_store=store,
        max_new_counterfactual_observations=20,
    )
    second_substrate = ProbeSubstrate()
    second = run_minimal_context(
        run, substrate=second_substrate, engine=_engine(), observation_store=store,
        max_new_counterfactual_observations=0,
    )

    assert first.status == second.status == "completed"
    assert second.best == first.best
    assert second.certificate == first.certificate
    assert second.budget.used_new_executions == 0
    assert second.budget.reused_observation_count > 0
    assert second_substrate.calls == []
    assert run_store.list_runs(100) == []


def test_search_identity_binds_inclusion_policy(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import _run, ProbeSubstrate

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _source_ids = _run()
    run["context_units"] = build_context_unit_manifest(run)
    store = ObservationStore()
    first = run_minimal_context(
        run, substrate=ProbeSubstrate(), engine=_engine(), observation_store=store,
        max_new_counterfactual_observations=4, attempt_inclusion_check=True,
    )
    second = run_minimal_context(
        run, substrate=ProbeSubstrate(), engine=_engine(), observation_store=store,
        max_new_counterfactual_observations=4, attempt_inclusion_check=False,
    )
    assert first.search_id != second.search_id
    assert first.policy["attempt_inclusion_check"] is True
    assert second.policy["attempt_inclusion_check"] is False


def test_prompt_block_cost_is_the_same_effective_prompt_used_by_execution(tmp_path, monkeypatch):
    from tests.test_experiment_kernel import ProbeSubstrate, _run

    class BlockSubstrate(ProbeSubstrate):
        def __init__(self):
            super().__init__()
            self.effective_inputs = []

        def probe_reference_match(self, messages, reference_token_ids, *, generation_contract,
                                  explicit_conditions):
            effective = inject_block(messages, explicit_conditions.get("block"))
            self.effective_inputs.append(deepcopy(effective))
            return super().probe_reference_match(
                effective, reference_token_ids, generation_contract=generation_contract,
                explicit_conditions=explicit_conditions,
            )

    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, _source_ids = _run()
    run.pop("assembled_messages")
    run["memory"] = {"prompt_block": "recorded prompt block"}
    run["context_receipt"] = build_context_receipt(
        messages=run["messages"], assembled_messages=None, final_prompt=None,
        run_id=run["id"], privacy="full",
    )
    run["context_units"] = build_context_unit_manifest(run)
    engine = _engine()
    substrate = BlockSubstrate()
    result = run_minimal_context(
        run, substrate=substrate, engine=engine, observation_store=ObservationStore(),
        max_new_counterfactual_observations=1,
    )

    assert result.status == "completed"
    expected = [
        {"role": "system", "content": "stable context\n\nrecorded prompt block"},
        {"role": "user", "content": "removable source"},
        {"role": "user", "content": "current question"},
    ]
    assert expected in engine.inputs
    assert expected in substrate.effective_inputs
    expected_cost = sum(len(str(item["content"])) for item in expected)
    assert expected_cost == sum(len(str(item["content"])) for item in substrate.effective_inputs[0])


def test_product_job_worker_calls_new_recipe_not_legacy_executor(monkeypatch):
    from clozn.server.routes import minimal_context as route
    import clozn.runs.store as runlog

    run = {"id": "route-run"}
    universe = {"status": "planned", "universe_id": "universe-1"}
    request = {"max_units": 5, "max_new_counterfactual_observations": 2,
               "attempt_inclusion_check": True}
    calls = []

    class Result:
        best = None
        certificate = None

        def to_dict(self):
            return {"schema_version": "clozn.minimal-context-search-result.v1", "status": "completed"}

    def fake_recipe(*args, **kwargs):
        calls.append((args, kwargs))
        return Result()

    class Control:
        def checkpoint(self, **_kwargs):
            return None

        def cancel_requested(self):
            return False

        def attach_result(self, value):
            self.result = value

    monkeypatch.setattr(runlog, "get_run", lambda _run_id: run)
    monkeypatch.setattr(route, "planned_universe", lambda *_args: universe)
    monkeypatch.setattr(route, "run_minimal_context", fake_recipe)
    outcome = route._job_worker(run, object(), object(), request, universe)(Control())

    assert outcome == {"state": "completed"}
    assert calls and calls[0][0][0] == run
    assert calls[0][1]["max_new_counterfactual_observations"] == 2
