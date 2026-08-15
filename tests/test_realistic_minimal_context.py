"""Model-free integration checks for the realistic recorded-trace evaluation."""
from __future__ import annotations

from copy import deepcopy
import json

from clozn.runs.budgeted_reduce import (
    BEST_VERIFIED,
    INCLUSION_MINIMUM,
    Budget,
    BudgetedReductionResult,
    Candidate,
    InclusionCheck,
    PreparedCandidate,
    Trial,
)
from clozn.runs.context_search_universe import plan_context_search_universe
from clozn.runs.answer_preservation import assess_exact_eligibility
from clozn.runs.realistic_minimal_context import (
    bind_engine_recorded_run,
    evaluate_recorded_run,
    reconstruct_checkpoint,
    serialize_outcome,
    suite_summary,
)
from scripts.bench.fixtures.minimal_context_realistic import (
    SCENARIO_IDS,
    RealisticScenario,
    build_fixture_run,
    built_in_scenarios,
    get_scenario,
    validate_registry,
    validate_scenario,
)
from scripts.bench.fixtures.minimal_context_scaled import (
    SCALED_SCENARIO_IDS,
    built_in_scaled_scenarios,
    validate_scaled_registry,
)


class _FakeAdapter:
    def __init__(self, preserving=None, control_status="matched"):
        self.preserving = preserving or (lambda ids: bool(ids))
        self.control_status = control_status
        self.calls = 0

    def prepare_candidate(self, retained_ids):
        return PreparedCandidate(tuple(retained_ids), len(retained_ids) * 10, tuple(retained_ids))

    def probe_many(self, candidates):
        out = []
        is_control = self.calls == 0
        self.calls += 1
        for candidate in candidates:
            ids = tuple(candidate.retained_ids)
            if is_control and self.control_status != "matched":
                out.append({"status": self.control_status})
            else:
                out.append({"status": "matched" if self.preserving(ids) else "diverged"})
        return out

    @staticmethod
    def is_preserving(evidence):
        return isinstance(evidence, dict) and evidence.get("status") == "matched"

    @staticmethod
    def is_failed(evidence):
        return isinstance(evidence, dict) and evidence.get("status") == "diverged"


def _eligible():
    return {
        "eligible": True,
        "reason": None,
        "reasons": [],
        "reference_token_count": 1,
        "generation_contract": {
            "decode_mode": "greedy",
            "sampling": None,
            "max_new": 8,
            "stop": [],
            "expected_termination": {"reason": "stop", "reason_raw": "stop"},
        },
    }


def test_a_builtin_registry_has_exact_stable_ids_and_order():
    assert tuple(item.case_id for item in built_in_scenarios()) == SCENARIO_IDS
    assert SCENARIO_IDS == (
        "single_relevant", "distributed", "redundant", "multi_turn", "broad_control",
    )


def test_b_scenarios_have_no_answer_or_reducer_oracles():
    fields = {field for field in RealisticScenario.__dataclass_fields__}
    assert fields == {"case_id", "description", "tags", "messages"}
    for scenario in built_in_scenarios():
        assert not hasattr(scenario, "expected_answer")
        assert not hasattr(scenario, "expected_minimal_ids")
        assert not hasattr(scenario, "expected_reduction")


def test_c_ordinary_context_units_protect_current_user_and_derive_removable_units():
    scenario = get_scenario("single_relevant")
    run = build_fixture_run(scenario)
    manifest = run["context_units"]
    assert manifest["protected_message_indices"] == [1]
    assert manifest["default_source_ids"]
    assert all(unit["message_index"] != 1 for unit in manifest["units"])
    auto_cases = 0
    for item in built_in_scenarios():
        derivations = {unit.get("derivation") for unit in build_fixture_run(item)["context_units"]["units"]}
        auto_cases += "auto_structural" in derivations
    assert auto_cases >= 4


def test_d_search_universe_is_the_real_receipt_partition():
    for scenario in built_in_scenarios():
        run = build_fixture_run(scenario)
        artifact = plan_context_search_universe(run, run["context_units"], max_units=50)
        assert artifact["status"] == "planned"
        assert artifact["source_count"] == len(artifact["source_ids"])
        assert artifact["coverage"]["protected_message_indices"] == run["context_units"]["protected_message_indices"]


def _checkpoint_result():
    original = Candidate(("a", "b", "c"), 30)
    trials = (
        Trial(1, "control", ("a", "b", "c"), 30, True, {"status": "matched"}),
        Trial(2, "coarse", ("a", "b"), 20, True, {"status": "matched"}),
        Trial(3, "inclusion", ("b",), 10, False, {"status": "diverged"}),
        Trial(4, "inclusion", ("a",), 10, False, {"status": "diverged"}),
    )
    return BudgetedReductionResult(
        status="ok",
        certificate_level=INCLUSION_MINIMUM,
        original_candidate=original,
        best_candidate=Candidate(("a", "b"), 20),
        control_evidence={"status": "matched"},
        trials=trials,
        trajectory=(),
        budget=Budget(10, 3, False),
        inclusion_check=InclusionCheck(True, True, 2, 2, True),
    )


def test_e_checkpoint_reconstruction_excludes_future_trials():
    result = _checkpoint_result()
    early = reconstruct_checkpoint(result, 1)
    later = reconstruct_checkpoint(result, 3)
    assert early["best_verified"]["retained_source_ids"] == ["a", "b"]
    assert early["certificate_level"] == BEST_VERIFIED
    assert later["certificate_level"] == INCLUSION_MINIMUM


def test_f_certificate_is_local_to_direct_prefix_evidence():
    result = _checkpoint_result()
    assert reconstruct_checkpoint(result, 1)["certificate_level"] != INCLUSION_MINIMUM
    assert reconstruct_checkpoint(result, 3)["certificate_level"] == INCLUSION_MINIMUM


def test_g_exact_unavailable_and_control_failed_are_typed_and_independent():
    run = build_fixture_run(get_scenario("single_relevant"))
    unavailable = evaluate_recorded_run(
        run, adapter=None, max_counterfactual_probes=0,
        eligibility={"eligible": False, "reason": "runtime_identity_unavailable", "reasons": []},
    )
    failed = evaluate_recorded_run(
        run, adapter=_FakeAdapter(control_status="diverged"), max_counterfactual_probes=0,
        eligibility=_eligible(),
    )
    assert unavailable.status == "exact_unavailable"
    assert failed.status == "control_failed"
    control_unavailable = evaluate_recorded_run(
        run, adapter=_FakeAdapter(control_status="unavailable"), max_counterfactual_probes=0,
        eligibility=_eligible(),
    )
    assert control_unavailable.status == "exact_unavailable"
    assert suite_summary(
        [
            serialize_outcome(case_id="a", description="a", tags=("x",), run=run,
                              outcome=unavailable, max_counterfactual_probes=0),
            serialize_outcome(case_id="b", description="b", tags=("x",), run=run,
                              outcome=failed, max_counterfactual_probes=0),
        ],
        max_counterfactual_probes=0,
    )["case_count"] == 2


def test_h_saved_recorded_run_adapter_reaches_reducer_without_baseline_regeneration():
    run = build_fixture_run(get_scenario("single_relevant"))
    run["response"] = "ok"
    run["trace"] = {"tokens": ["ok"], "token_ids": [7], "steps": [{"piece": "ok", "token_id": 7}]}

    class FakeEngine:
        def __init__(self):
            self.complete_calls = 0

        def apply_template_info(self, messages):
            return {"prompt": "\n".join(item["content"] for item in messages),
                    "prompt_tokens": sum(len(item["content"]) for item in messages)}

        def complete(self, *args, **kwargs):
            self.complete_calls += 1
            raise AssertionError("saved-run evaluation must not regenerate its baseline")

    class FakeSubstrate:
        def probe_reference_match(self, **kwargs):
            return {"status": "matched"}

        def probe_reference_match_many(self, arms, **kwargs):
            return [{"status": "matched"} for _ in arms]

    engine = FakeEngine()
    binding = bind_engine_recorded_run(
        run, engine=engine, substrate=FakeSubstrate(), eligibility=_eligible(),
    )
    outcome = evaluate_recorded_run(
        run, adapter=binding.adapter, max_counterfactual_probes=0,
        eligibility=binding.eligibility,
    )
    assert outcome.status == "ok"
    assert engine.complete_calls == 0


def test_h2_empty_ordinary_output_contract_is_not_structured_output():
    run = build_fixture_run(get_scenario("single_relevant"))
    run["response"] = "ok"
    run["trace"] = {"tokens": ["ok"], "token_ids": [7], "steps": [{"piece": "ok", "token_id": 7}]}
    eligibility = assess_exact_eligibility(run, current_runtime={})
    assert "unsupported_structured_output" not in eligibility["reasons"]


def test_i_serialized_report_is_deterministic_and_does_not_leak_runtime_objects():
    run = build_fixture_run(get_scenario("single_relevant"))
    outcome = evaluate_recorded_run(
        run, adapter=_FakeAdapter(), max_counterfactual_probes=0, eligibility=_eligible(),
    )
    first = serialize_outcome(
        case_id="single_relevant", description="fixture", tags=("document",), run=run,
        outcome=outcome, max_counterfactual_probes=0,
    )
    second = serialize_outcome(
        case_id="single_relevant", description="fixture", tags=("document",), run=run,
        outcome=outcome, max_counterfactual_probes=0,
    )
    encoded_first = json.dumps(first, sort_keys=True, separators=(",", ":"))
    encoded_second = json.dumps(second, sort_keys=True, separators=(",", ":"))
    assert encoded_first == encoded_second
    assert "EngineClient" not in encoded_first
    assert "substrate" not in encoded_first
    assert "Northstar travel policy" not in encoded_first


def test_j_broad_control_is_structural_only_and_has_no_runtime_reduction_claim():
    row = validate_scenario(get_scenario("broad_control"))
    assert "broad-synthesis" in row["case_tags"]
    assert row["removable_unit_count"] >= 10
    assert "reduction_percent" not in row
    assert "response" not in build_fixture_run(get_scenario("broad_control"))


def test_k_scaled_registry_has_six_realistic_cases_without_oracles():
    scenarios = built_in_scaled_scenarios()
    assert tuple(item.case_id for item in scenarios) == SCALED_SCENARIO_IDS
    assert len(scenarios) == 6
    fields = set(RealisticScenario.__dataclass_fields__)
    assert fields == {"case_id", "description", "tags", "messages"}
    rows = validate_scaled_registry()
    assert all(row["raw_context_unit_count"] >= 20 for row in rows)
    assert all(row["bounded_search_universe_count"] <= 50 for row in rows)


def test_l_scaled_shape_is_structural_not_an_exact_token_count():
    for scenario in built_in_scaled_scenarios():
        total_chars = sum(len(message["content"]) for message in scenario.messages)
        assert total_chars >= 9000
        assert not hasattr(scenario, "expected_answer")
        assert not hasattr(scenario, "expected_minimal_ids")
        assert not hasattr(scenario, "expected_reduction")


def test_m_scaled_geometry_reports_raw_bounded_and_message_partitions():
    rows = validate_scaled_registry()
    for row in rows:
        assert row["protected_message_indices"]
        assert row["removable_message_indices"]
        assert row["raw_context_unit_count"] >= row["bounded_search_universe_count"]
    assert any(row["raw_context_unit_count"] > row["bounded_search_universe_count"] for row in rows)


def test_n_geometric_checkpoints_mark_future_points_after_early_termination():
    result = _checkpoint_result()
    point = reconstruct_checkpoint(result, 8)
    assert point["run_already_terminated"] is True
    assert point["termination_probe"] == result.budget.used_counterfactual_probes
    assert point["best_verified"]["retained_source_ids"] == ["a", "b"]


def test_o_improvement_events_start_with_control_and_are_strict():
    run = build_fixture_run(get_scenario("single_relevant"))
    outcome = evaluate_recorded_run(
        run, adapter=_FakeAdapter(), max_counterfactual_probes=8, eligibility=_eligible(),
    )
    report = serialize_outcome(
        case_id="single_relevant", description="fixture", tags=("document",), run=run,
        outcome=outcome, max_counterfactual_probes=8,
    )
    events = report["improvement_events"]
    assert events[0]["probe_count"] == 0
    assert events[0]["stage"] == "control"
    assert all(
        (event["cost"], event["retained_unit_count"]) <
        (previous["cost"], previous["retained_unit_count"])
        for previous, event in zip(events, events[1:])
    )


def test_p_termination_reasons_and_zero_reduction_are_serialized():
    run = build_fixture_run(get_scenario("broad_control"))
    outcome = evaluate_recorded_run(
        run, adapter=_FakeAdapter(), max_counterfactual_probes=0, eligibility=_eligible(),
    )
    report = serialize_outcome(
        case_id="broad_control", description="fixture", tags=("control",), run=run,
        outcome=outcome, max_counterfactual_probes=0,
    )
    assert report["termination"]["reason"] == "budget_exhausted"
    assert report["milestones"]["probe_to_50_percent_eventual_reduction"] is None
    summary = suite_summary([report], max_counterfactual_probes=0)
    assert summary["descriptive_questions"]["probe_to_50_percent_eventual_reduction"]["broad_control"] is None
    assert summary["termination_reason_counts"]["budget_exhausted"] == 1
