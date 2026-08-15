from __future__ import annotations

from collections import Counter

from clozn.runs.answer_preservation import is_reference_match_preserving
from clozn.runs.budgeted_reduce import (
    BEST_VERIFIED,
    INCLUSION_MINIMUM,
    run_budgeted_reduction,
)
from clozn.runs.budgeted_reduce import PreparedCandidate


def _run(universe, preserving, *, costs=None, budget=100, inclusion=True):
    calls = []
    costs = costs or {}

    def prepare(retained):
        retained = tuple(retained)
        return {
            "retained_ids": retained,
            "cost": costs.get(retained, len(retained)),
            "payload": retained,
        }

    def probe_many(prepared):
        batch = [tuple(item["retained_ids"]) for item in prepared]
        calls.append(batch)
        return [
            {"preserves": frozenset(item["retained_ids"]) in preserving}
            for item in prepared
        ]

    result = run_budgeted_reduction(
        universe,
        budget,
        prepare,
        probe_many,
        attempt_inclusion_check=inclusion,
    )
    return result, calls


def test_control_failure_runs_no_altered_probe():
    calls = []

    def prepare(ids):
        return {"retained_ids": tuple(ids), "cost": len(tuple(ids)), "payload": tuple(ids)}

    def probe_many(candidates):
        calls.append(len(candidates))
        return [{"preserves": False}]

    result = run_budgeted_reduction(("a", "b"), 10, prepare, probe_many)

    assert result.status == "control_failed"
    assert result.certificate_level is None
    assert result.budget.used_counterfactual_probes == 0
    assert calls == [1]
    assert len(result.trials) == 1


def test_zero_budget_returns_controlled_full_context():
    result, calls = _run((0, 1, 2), {frozenset({0, 1, 2})}, budget=0)

    assert result.status == "ok"
    assert result.certificate_level == BEST_VERIFIED
    assert result.best_candidate.retained_ids == (0, 1, 2)
    assert result.budget.used_counterfactual_probes == 0
    assert result.inclusion_check.attempted is False
    assert calls == [[(0, 1, 2)]]


def test_large_deletion_is_adopted_only_with_direct_preservation_evidence():
    result, _calls = _run(
        (0, 1, 2, 3),
        {frozenset({1}), frozenset({0, 1}), frozenset({1, 2}), frozenset({1, 3}),
         frozenset({0, 1, 2, 3})},
        costs={
            (0, 1, 2, 3): 40,
            (0, 1): 20,
            (2, 3): 20,
            (0,): 10,
            (1,): 10,
        },
    )

    assert result.best_candidate.retained_ids == (1,)
    assert result.best_candidate.cost == 10
    assert any(trial.retained_ids == (0, 1) and trial.preserves for trial in result.trials)
    assert result.certificate_level == INCLUSION_MINIMUM


def test_failed_deletion_is_never_adopted():
    result, _calls = _run((0, 1, 2, 3), {frozenset({0, 1, 2, 3})}, budget=100)

    assert result.best_candidate.retained_ids == (0, 1, 2, 3)
    assert all(entry.retained_ids == (0, 1, 2, 3) for entry in result.trajectory) is True


def test_hard_budget_never_exceeds_limit_and_stays_best_verified():
    result, calls = _run((0, 1, 2, 3, 4), {frozenset({0, 1, 2, 3, 4})}, budget=1)

    assert result.budget.used_counterfactual_probes == 1
    assert result.budget.exhausted is True
    assert sum(len(batch) for batch in calls[1:]) == 1
    assert result.certificate_level == BEST_VERIFIED


def test_duplicate_subset_is_directly_probed_once():
    result, _calls = _run((0, 1, 2, 3), {frozenset({0, 1, 2, 3})}, budget=100)

    trial_ids = [trial.retained_ids for trial in result.trials]
    assert len(trial_ids) == len(set(trial_ids))
    assert Counter(trial_ids)[()] <= 1


def test_variable_cost_prefers_lower_cost_even_when_it_retains_more_units():
    universe = (0, 1, 2, 3)
    preserving = {
        frozenset(universe), frozenset({0, 1}), frozenset({2, 3}),
        frozenset({0}), frozenset({1}), frozenset({2}), frozenset({3}),
    }
    result, _calls = _run(
        universe,
        preserving,
        costs={
            universe: 100,
            (0, 1): 60,
            (2, 3): 80,
            (0,): 90,
            (1,): 90,
            (2,): 90,
            (3,): 90,
        },
    )

    assert result.best_candidate.retained_ids == (0, 1)
    assert result.best_candidate.cost == 60
    assert len(result.best_candidate.retained_ids) > 1


def test_search_and_trial_order_are_deterministic():
    preserving = {frozenset({0, 2}), frozenset({0, 2, 3}), frozenset({0, 1, 2, 3})}
    first, _ = _run((0, 1, 2, 3), preserving, budget=20)
    second, _ = _run((0, 1, 2, 3), preserving, budget=20)

    assert first.best_candidate == second.best_candidate
    assert first.trials == second.trials
    assert first.trajectory == second.trajectory


def test_complete_inclusion_sweep_emits_inclusion_minimum():
    result, _calls = _run((0, 1, 2), {frozenset({0, 1, 2})}, budget=100)

    assert result.certificate_level == INCLUSION_MINIMUM
    assert result.inclusion_check.attempted is True
    assert result.inclusion_check.complete is True
    assert result.inclusion_check.tested_child_count == 3


def test_preserving_one_unit_child_blocks_inclusion_certificate_even_if_cost_is_higher():
    universe = (0, 1, 2, 3)
    result, _calls = _run(
        universe,
        {frozenset(universe), frozenset({0, 1}), frozenset({0})},
        costs={universe: 10, (0, 1): 5, (2, 3): 99, (0,): 99, (1,): 1},
        budget=100,
    )

    assert result.best_candidate.retained_ids == (0, 1)
    assert result.certificate_level == BEST_VERIFIED
    assert result.inclusion_check.complete is False
    assert result.inclusion_check.all_children_failed is False


def test_unknown_direct_evidence_cannot_be_used_as_an_inclusion_failure():
    def prepare(ids):
        ids = tuple(ids)
        return {"retained_ids": ids, "cost": len(ids), "payload": ids}

    def probe_many(candidates):
        return [
            {"status": "matched"} if tuple(item["retained_ids"]) == (0, 1, 2)
            else {"status": "unavailable"}
            for item in candidates
        ]

    result = run_budgeted_reduction((0, 1, 2), 100, prepare, probe_many)

    assert result.status == "ok"
    assert result.certificate_level == BEST_VERIFIED
    assert result.inclusion_check.complete is False


def test_inclusion_sweep_interrupted_by_budget_stays_best_verified():
    result, _calls = _run((0, 1, 2, 3), {frozenset({0, 1, 2, 3})}, budget=3)

    assert result.certificate_level == BEST_VERIFIED
    assert result.inclusion_check.attempted is True
    assert result.inclusion_check.complete is False
    assert result.budget.used_counterfactual_probes == 3


def test_candidate_change_during_search_starts_a_new_inclusion_sweep():
    universe = (0, 1, 2, 3)
    preserving = {
        frozenset(universe), frozenset({0, 1}), frozenset({0}),
    }
    result, _calls = _run(universe, preserving, budget=100)

    assert result.best_candidate.retained_ids == (0,)
    assert result.certificate_level == INCLUSION_MINIMUM
    inclusion_trials = [trial for trial in result.trials if trial.stage == "inclusion"]
    assert any(trial.retained_ids == () and trial.preserves is False for trial in result.trials)
    assert result.inclusion_check.total_child_count == 1


def test_explicitly_non_monotone_oracle_uses_only_direct_observations():
    universe = (0, 1, 2)
    preserving = {frozenset(universe), frozenset({1}), frozenset({0, 2})}
    result, _calls = _run(universe, preserving, budget=100)

    assert result.best_candidate.retained_ids in {(1,), (0, 2)}
    assert result.certificate_level != "EXACT_MINIMUM"
    assert result.certificate_level in {BEST_VERIFIED, INCLUSION_MINIMUM}
    assert all(trial.evidence["preserves"] is trial.preserves for trial in result.trials)


def test_batch_accounting_charges_one_probe_per_candidate():
    result, calls = _run((0, 1, 2, 3), {frozenset({0, 1, 2, 3})}, budget=100)

    assert result.budget.used_counterfactual_probes == sum(len(batch) for batch in calls[1:])
    assert all(len(batch) >= 1 for batch in calls)


def test_shared_exact_reference_predicate_is_strict():
    assert is_reference_match_preserving({"status": "matched"}) is True
    assert is_reference_match_preserving({"status": "diverged"}) is False
    assert is_reference_match_preserving({"preserves": True}) is False
    assert is_reference_match_preserving(None) is False


def test_exact_engine_adapter_uses_worker_prompt_token_count_and_shared_probe(monkeypatch):
    import clozn.runs.budgeted_reduce_reference as reference_adapter

    class Engine:
        def apply_template_info(self, messages):
            return {"prompt": "rendered", "prompt_tokens": 123}

    calls = []

    def fake_probe(substrate, arms):
        calls.append((substrate, arms))
        return [{"status": "matched"} for _ in arms]

    monkeypatch.setattr(reference_adapter, "probe_reference_match_many", fake_probe)
    adapter = reference_adapter.EngineReferenceMatchAdapter(
        engine=Engine(),
        substrate=object(),
        render_messages=lambda retained: [{"role": "system", "content": str(retained)}],
        reference_token_ids=(11, 22),
        generation_contract={"decode_mode": "greedy"},
    )

    prepared = adapter.prepare_candidate(("unit",))
    assert isinstance(prepared, PreparedCandidate)
    assert prepared.cost == 123
    assert prepared.probe_payload["reference_token_ids"] == [11, 22]
    assert adapter.probe_many([prepared]) == [{"status": "matched"}]
    assert calls and "proof_grade" not in calls[0][1][0]
    assert adapter.is_preserving({"status": "matched"}) is True
