from __future__ import annotations

from copy import deepcopy

import pytest

from clozn.experiments.evaluators import Generate
from clozn.experiments.interventions import DeleteSource, ForceToken
from clozn.experiments.kernel import Experiment
from clozn.experiments.materialize import materialize_generated_observation
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.runner import run_experiment
from clozn.experiments.selections import ContextSelection
from clozn.experiments.state import ExecutionState
from clozn.recipes.context_counterfactual import (
    ContextCounterfactualUnavailable,
    generate_without_source,
    plan_context_counterfactual,
)
from clozn.runs import store as run_store

from tests.test_context_effects_kernel import _identity, _run


def _context_run():
    run, source_ids = _run()
    identity = _identity()
    identity["adapter"] = {
        "present": False, "identity_sha256": None,
        "artifact_sha256": None, "scale": None,
    }
    run["identity"] = identity
    return run, source_ids


class _Steer:
    def __init__(self):
        self.strength = {"live": 2.0}
        self._engaged = True


class _ChatSubstrate:
    def __init__(self, runtime, *, fail=False):
        self.runtime_identity = deepcopy(runtime)
        self.steer = _Steer()
        self.fail = fail
        self.calls = []

    def chat(self, messages, *, max_new, sample, trace_out, mem_out, stop):
        self.calls.append({
            "messages": deepcopy(messages), "max_new": max_new,
            "sample": deepcopy(sample), "stop": list(stop),
            "dials": deepcopy(self.steer.strength),
        })
        if self.fail:
            raise RuntimeError("fixture generation failure")
        trace_out.extend([
            {"token_id": 20, "piece": "new"},
            {"token_id": 21, "piece": " answer"},
        ])
        mem_out.update(
            assembled_messages=deepcopy(messages),
            final_prompt="exact counterfactual prompt",
        )
        return "new answer"


@pytest.fixture()
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()


def test_context_generate_matrix_is_narrow_and_plan_is_model_free():
    run, source_ids = _context_run()
    state = ExecutionState.from_run(run)
    evaluator = Generate(max_new=4)
    intervention = DeleteSource(ContextSelection([source_ids[1]]))

    experiment = Experiment(base=state, evaluator=evaluator, arms=[intervention])
    assert experiment.arms[0].intervention.to_dict() == intervention.to_dict()
    with pytest.raises(TypeError):
        Experiment(base=state, evaluator=evaluator, arms=[ForceToken(token_piece="x")])
    with pytest.raises(TypeError):
        Experiment(base=state, evaluator=evaluator, arms=[None])

    plan = plan_context_counterfactual(run, source_ids[1])
    assert plan.experiment_id == experiment.experiment_id
    assert plan.arm_id == experiment.arms[0].arm_id


def test_context_generate_captures_observation_and_restores_live_controls(isolated_runs):
    run, source_ids = _context_run()
    substrate = _ChatSubstrate(run["identity"])
    store = ObservationStore()

    result = generate_without_source(
        run, source_ids[1], substrate=substrate, observation_store=store,
    )

    assert result.state == "completed"
    assert len(substrate.calls) == 1
    assert substrate.calls[0]["dials"] == {}
    assert substrate.steer.strength == {"live": 2.0}
    observation = result.arms[0].observation
    assert observation.status == "completed"
    assert observation.state_ref is None
    assert observation.intervention["target"]["source_ids"] == [source_ids[1]]
    assert observation.generated_suffix_text == "new answer"
    assert observation.generated_token_ids == (20, 21)
    assert observation.input_snapshot["final_prompt"] == "exact counterfactual prompt"
    assert observation.input_snapshot["source_ids"] == [source_ids[1]]
    assert run_store.list_runs(20) == []


def test_context_generate_failure_restores_live_controls_and_is_not_reusable(isolated_runs):
    run, source_ids = _context_run()
    substrate = _ChatSubstrate(run["identity"], fail=True)
    result = generate_without_source(
        run, source_ids[1], substrate=substrate, observation_store=ObservationStore(),
    )

    assert result.state == "failed"
    assert result.arms[0].observation is None
    assert result.arms[0].diagnostics["observation_status"] == "failed"
    assert substrate.steer.strength == {"live": 2.0}
    assert run_store.list_runs(20) == []


def test_context_generate_reuses_persisted_observation_without_a_second_chat(isolated_runs):
    run, source_ids = _context_run()
    store = ObservationStore()
    first_substrate = _ChatSubstrate(run["identity"])
    first = generate_without_source(
        run, source_ids[1], substrate=first_substrate, observation_store=store,
    )
    second_substrate = _ChatSubstrate(run["identity"])
    second = generate_without_source(
        run, source_ids[1], substrate=second_substrate, observation_store=store,
    )

    assert len(first_substrate.calls) == 1
    assert len(second_substrate.calls) == 0
    assert first.arms[0].observation.observation_id == second.arms[0].observation.observation_id
    assert second.arms[0].diagnostics["execution_disposition"] == "reused"


def test_context_generate_plan_rejects_incomplete_sample_contract_before_worker():
    run, source_ids = _context_run()
    run["generation_contract"] = {
        "decode_mode": "sample", "sampling": {
            "temperature": 0.8, "top_p": 0.9, "top_k": 40,
            "repeat_penalty": 1.1,
        }, "max_new": 4, "stop": [],
        "expected_termination": {"reason": "eos"},
    }
    with pytest.raises(ContextCounterfactualUnavailable):
        plan_context_counterfactual(run, source_ids[1])


def test_context_materialization_uses_observed_input_and_never_calls_model(isolated_runs, monkeypatch):
    run, source_ids = _context_run()
    store = ObservationStore()
    substrate = _ChatSubstrate(run["identity"])
    result = generate_without_source(
        run, source_ids[1], substrate=substrate, observation_store=store,
    )
    calls = len(substrate.calls)
    materialized = materialize_generated_observation(
        run, result.experiment_id, result.arms[0].arm_id,
        observation_id=result.arms[0].observation_id,
        observation_store=store,
    )

    assert materialized["state"] == "completed"
    assert len(substrate.calls) == calls
    child = run_store.get_run(materialized["child_run_id"])
    assert child["parent_run_id"] == run["id"]
    assert child["response"] == "new answer"
    assert all(message["content"] != "removable source" for message in child["messages"])
    assert child["final_prompt"] == "exact counterfactual prompt"
    assert child["changes_applied"]["experiment"]["operation"] == "delete_source_generate"
    assert child["changes_applied"]["experiment"]["origin"] == {"kind": "recorded_prompt_boundary"}
    assert child["trace"]["token_ids"] == [20, 21]


def test_context_materialization_rejects_wrong_observation_reference(isolated_runs):
    run, source_ids = _context_run()
    store = ObservationStore()
    result = generate_without_source(
        run, source_ids[1], substrate=_ChatSubstrate(run["identity"]), observation_store=store,
    )
    with pytest.raises(Exception, match="does not match"):
        materialize_generated_observation(
            run, result.experiment_id, result.arms[0].arm_id,
            observation_id="obs_wrong", observation_store=store,
        )
