from __future__ import annotations

from collections import Counter
from itertools import islice
import json

import pytest

from clozn.experiments.coalitions import shapley_pairs
from clozn.experiments.persistence import ObservationStore
from clozn.recipes.context_effects import measure_context_effects
from clozn.recipes.random_coalitions import run_random_coalitions
from clozn.runs import store as run_store
from tests.test_context_effects_kernel import ScoreSubstrate, _run
from tests.test_lean_context import TokenCountingEngine


def execute(substrate=None, **kwargs):
    run, _ = _run()
    return run_random_coalitions(
        run, substrate=substrate or ScoreSubstrate(), engine=TokenCountingEngine(), **kwargs,
    )


def test_sampler_reproducible_canonical_balanced_and_not_half_mask_weighted():
    ids = ("a", "b", "c", "d")
    samples = list(islice(shapley_pairs(ids, seed=29), 16000))
    assert samples == list(islice(shapley_pairs(ids, seed=29), 16000))
    assert samples[:12] != list(islice(shapley_pairs(ids, seed=30), 12))
    for i in range(0, len(samples), len(ids)):
        assert {row[0] for row in samples[i:i+len(ids)]} == set(ids)
    for sid, without, with_source in samples:
        assert sid not in without
        assert set(with_source) == set(without) | {sid}
        assert with_source == tuple(value for value in ids if value in with_source)
    # Size-uniform, not binomially concentrated around n/2 (Banzhaf).
    sizes = Counter(len(row[1]) for row in samples)
    assert all(3700 < sizes[k] < 4300 for k in range(4))


@pytest.mark.parametrize("budget", [0, 1, 2, 3, 4, 5, 7, 20])
def test_budget_counts_control_and_actual_calls_without_persistent_store(budget):
    substrate = ScoreSubstrate()
    result = execute(substrate, max_new_executions=budget, max_pairs=25)
    assert len(substrate.calls) == result["budget"]["used_new_executions"] <= budget
    assert result["budget"]["control_included"] is True
    assert len(result["candidates"]) == len(substrate.calls)
    assert result["claim"]["minimum_proven"] is False
    assert result["claim"]["redundancy_proven"] is False
    if budget == 0:
        assert result["status"] == "unavailable"
        assert result["best"] is None
    if result["best"] is not None:
        assert result["best"]["within_tolerance"] is True


def test_repeated_draws_do_not_repeat_execution_and_zero_budget_can_replay(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    store = ObservationStore()
    substrate = ScoreSubstrate()
    first = execute(substrate, observation_store=store, max_new_executions=8, max_pairs=90, seed=11)
    assert len(substrate.calls) == 8  # every subset of the three-source universe
    replay_substrate = ScoreSubstrate()
    second = execute(replay_substrate, observation_store=store, max_new_executions=0, max_pairs=90, seed=11)
    assert replay_substrate.calls == []
    assert first["samples"] == second["samples"]
    assert first["source_estimates"] == second["source_estimates"]
    assert second["budget"]["reused_observation_count"] == 8


def test_reuses_leave_one_out_and_best_tolerant_candidate_without_new_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    store = ObservationStore()
    run, ids = _run()
    measure_context_effects(run, substrate=ScoreSubstrate(), observation_store=store)
    substrate = ScoreSubstrate()
    result = execute(substrate, observation_store=store, max_new_executions=0)
    assert substrate.calls == []
    assert [row["leave_one_out_nats"] for row in result["source_estimates"]] == [.5, 2., 0.]
    assert result["best"]["retained_source_ids"] == ids[:2]
    assert result["budget"]["reused_observation_count"] == 4


class RedundantSubstrate(ScoreSubstrate):
    """Two independent-looking texts supply interchangeable support; third is null."""

    def score_tokens(self, messages, continuation_ids=None, **kwargs):
        self.calls.append(messages)
        present = {message["content"] for message in messages}
        supported = bool(present & {"stable context", "removable source"})
        return [{"id": token_id, "piece": piece, "logprob": -1. if supported else -3.}
                for token_id, piece in zip(continuation_ids, ["same", " answer"])]


def test_random_coalitions_expose_support_hidden_from_leave_one_out(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    store = ObservationStore()
    run, ids = _run()
    measure_context_effects(run, substrate=RedundantSubstrate(), observation_store=store)
    substrate = RedundantSubstrate()
    result = execute(substrate, observation_store=store, max_new_executions=4, max_pairs=3000, seed=3)
    a, b, irrelevant = result["source_estimates"]
    assert a["leave_one_out_nats"] == b["leave_one_out_nats"] == 0.
    assert a["mean_marginal_nats"] == pytest.approx(2., abs=.2)
    assert b["mean_marginal_nats"] == pytest.approx(2., abs=.2)
    assert irrelevant["mean_marginal_nats"] == 0.
    assert len(result["best"]["retained_source_ids"]) == 1
    assert result["best"]["retained_source_ids"][0] in ids[:2]
    assert len(substrate.calls) == 4
    # Last user message is an ordinary sampled source, not protected.
    assert irrelevant["sample_count"] == 1000


def test_sparse_results_never_turn_unknown_into_zero():
    result = execute(max_new_executions=3, max_pairs=1)
    for row in result["source_estimates"]:
        if row["sample_count"] == 0:
            assert row["coverage"] == "unsampled"
            assert row["mean_marginal_nats"] is None
            assert row["mean_token_marginals_nats"] is None
        assert row["sample_stddev_nats"] is None
    json.dumps(result, allow_nan=False)


def test_signed_token_marginals_and_tolerance_use_different_reductions():
    result = execute(max_new_executions=8, max_pairs=90)
    assert any(value < 0 for sample in result["samples"] for value in sample["token_deltas_nats"])
    for sample in result["samples"]:
        assert sample["delta_nats"] == pytest.approx(sum(sample["token_deltas_nats"]))
    for row in result["candidates"]:
        if row["within_tolerance"]:
            assert row["loss"]["mean_positive_support_loss_nats_per_token"] <= .05
            assert row["loss"]["max_token_support_loss_nats"] <= .7


def test_sample_progress_publishes_each_completed_pair():
    events = []
    result = execute(
        max_new_executions=8,
        max_pairs=3,
        sample_progress=events.append,
    )
    assert len(events) == result["budget"]["completed_pairs"] == 3
    assert [event["completed_pairs"] for event in events] == [1, 2, 3]
    assert all(event["sample"]["source_id"] for event in events)


def test_sampler_rejects_duplicate_sources_and_invalid_seed():
    with pytest.raises(ValueError):
        next(shapley_pairs(["a", "a"], seed=0))
    with pytest.raises(ValueError):
        next(shapley_pairs([], seed=0))
    with pytest.raises(ValueError):
        next(shapley_pairs(["a"], seed=True))
    with pytest.raises(ValueError):
        next(shapley_pairs("abc", seed=0))


def test_bound_and_exact_prompt_cost_fail_before_inference():
    substrate = ScoreSubstrate()
    with pytest.raises(ValueError, match="exceed"):
        execute(substrate, max_units=1)
    with pytest.raises(ValueError, match="exact rendered"):
        execute(substrate, prompt_token_counter=lambda messages: None)
    assert substrate.calls == []


def test_failure_after_control_is_partial_and_charged():
    class FailArm(ScoreSubstrate):
        def score_tokens(self, *args, **kwargs):
            if self.calls:
                self.calls.append("failed")
                raise RuntimeError("fixture arm failed")
            return super().score_tokens(*args, **kwargs)
    substrate = FailArm()
    result = execute(substrate, seed=6, max_new_executions=3)
    assert result["status"] == "partial"
    assert result["budget"]["used_new_executions"] == len(substrate.calls) == 2
    assert result["samples"] == []
    assert result["best"]["retained_source_ids"] == _run()[1]


def test_cancellation_stops_before_execution_and_between_pair_endpoints():
    substrate = ScoreSubstrate()
    result = execute(substrate, cancel=lambda: True)
    assert result["stop_reason"] == "cancelled"
    assert substrate.calls == []
    result = execute(substrate, cancel=lambda: len(substrate.calls) >= 2, seed=6)
    assert result["stop_reason"] == "cancelled"
    assert result["budget"]["used_new_executions"] == len(substrate.calls) == 2
    assert result["samples"] == []


def test_failed_score_consumes_budget_and_is_not_a_zero_effect():
    class Broken(ScoreSubstrate):
        def score_tokens(self, *args, **kwargs):
            self.calls.append("failed")
            raise RuntimeError("fixture worker failed")
    substrate = Broken()
    result = execute(substrate, max_new_executions=1)
    assert result["stop_reason"] == "score_unavailable"
    assert result["budget"]["used_new_executions"] == len(substrate.calls) == 1
    assert result["best"] is None
    assert all(row["mean_marginal_nats"] is None for row in result["source_estimates"])


@pytest.mark.parametrize("kwargs", [
    {"max_new_executions": -1}, {"max_new_executions": True}, {"max_pairs": -1},
    {"max_pairs": 1.5}, {"seed": False}, {"mean_loss_tolerance_nats": float("nan")},
    {"max_token_loss_tolerance_nats": -1}, {"mean_loss_tolerance_nats": True},
])
def test_rejects_invalid_configuration_before_scoring(kwargs):
    substrate = ScoreSubstrate()
    with pytest.raises(ValueError):
        execute(substrate, **kwargs)
    assert substrate.calls == []
