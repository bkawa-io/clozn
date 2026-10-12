from __future__ import annotations

from copy import deepcopy
from threading import Event

import pytest

from clozn.experiments.observations import (
    TokenScoreObservation,
    execution_observation_identity,
)
from clozn.experiments.persistence import ObservationStore
from clozn.recipes.context_effects_search import (
    ContextEffectsSearchBudget,
    measure_context_effects_coarse_to_fine,
)
from clozn.runs import store as run_store
from clozn.runs.context_receipt import build_context_receipt


def _run(source_count: int = 6):
    messages = [
        {"role": "system", "content": f"source {index}"}
        for index in range(source_count)
    ] + [{"role": "user", "content": "current question"}]
    run = {
        "id": "run_context_effects_search",
        "model": "fixture-model",
        "substrate": "fixture",
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "final_prompt": "exact fixture prompt",
        "response": "answer",
        "generation_contract": {
            "decode_mode": "greedy",
            "sampling": None,
            "max_new": 1,
            "stop": [],
            "expected_termination": {"reason": "eos", "reason_raw": "eos"},
        },
        "identity": {
            "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16,
            "engine_build": "test-engine",
            "context_size": 4096,
            "backend": "cpu",
            "white_box_flags": {
                "sae": False,
                "jlens": False,
                "attn_knockout": False,
            },
        },
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "behavior": {"active_dials": {}},
        "trace": {
            "tokens": ["answer"],
            "token_ids": [10],
            "steps": [{"token_id": 10, "piece": "answer"}],
        },
    }
    run["context_receipt"] = build_context_receipt(
        messages=messages,
        assembled_messages=messages,
        final_prompt=run["final_prompt"],
        run_id=run["id"],
        privacy="full",
    )
    source_ids = tuple(
        item["segment_id"] for item in run["context_receipt"]["assembled"][:source_count]
    )
    return run, source_ids


class _EffectAdapter:
    """Trusted scalar fixture keyed by the exact jointly deleted source set."""

    def __init__(self, effects):
        self.effects = {
            tuple(sorted(source_ids)): float(delta)
            for source_ids, delta in effects.items()
        }
        self.calls = []

    def execute(self, state, intervention=None, *, evaluator=None, arm_id=None):
        source_ids = tuple(intervention.source_ids) if intervention is not None else ()
        self.calls.append(source_ids)
        delta = self.effects.get(source_ids, 0.0)
        total = -1.0 - delta
        identity = execution_observation_identity(state, evaluator, intervention)
        return TokenScoreObservation(
            **identity,
            run_id=state.run_id,
            status="completed",
            recorded_token_ids=[10],
            token_pieces=["answer"],
            token_spans=[(0, 6)],
            token_logprobs=[total],
            total_continuation_logprob=total,
            evaluator_provenance={"evaluator": "fixture"},
            score_basis={"fixture": True},
            execution_provenance={
                "adapter": "fixture_scalar",
                "execution_strategy": "scalar_serial",
                "numerical_regime": "scalar_reference",
                "execution_identity": {
                    "kind": "clozn.score-tokens-scalar-reference.v1",
                },
            },
            proof_grade="trusted",
            trusted=True,
        )


def _key(*source_ids):
    return tuple(sorted(source_ids))


def test_requested_drilldown_overrides_opposing_effects_hidden_by_coarse_cancellation():
    run, ids = _run(4)
    adapter = _EffectAdapter({
        _key(ids[0], ids[1]): 0.0,
        _key(ids[0]): 2.0,
        _key(ids[1]): -2.0,
    })

    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids[:2], ids[2:]],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=2, max_fine_arms=2, max_joint_arms=0,
        ),
        refine_threshold_nats=0.5,
        drilldown_source_ids=ids[:2],
        execution_adapter=adapter,
    )

    assert result.arm_for("coarse", ids[:2]).delta_nats == 0.0
    first = result.arm_for("fine", [ids[0]])
    second = result.arm_for("fine", [ids[1]])
    assert (first.delta_nats, second.delta_nats) == (2.0, -2.0)
    assert first.selection_reason == second.selection_reason == "user_requested_drilldown"
    assert first.evidence_status == second.evidence_status == "trusted"
    assert result.coverage["requested_drilldowns_measured"] == list(ids[:2])


def test_progress_reports_only_truthful_phase_boundaries():
    run, ids = _run(4)
    events = []
    adapter = _EffectAdapter({
        _key(*ids): 2.0,
        _key(ids[0]): 1.0,
    })

    measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=1, max_fine_arms=1, max_joint_arms=0,
        ),
        refine_threshold_nats=0.5,
        drilldown_source_ids=[ids[0]],
        execution_adapter=adapter,
        progress=lambda **event: events.append(event),
    )

    assert events == [
        {"phase": "coarse_scalar_scoring", "completed": 0, "total": 1},
        {"phase": "coarse_scalar_scoring", "completed": 1, "total": 1},
        {"phase": "selecting_refinement", "completed": 0, "total": 1},
        {"phase": "selecting_refinement", "completed": 1, "total": 1},
        {"phase": "fine_joint_scalar_scoring", "completed": 0, "total": 1},
        {"phase": "fine_joint_scalar_scoring", "completed": 1, "total": 1},
    ]


def test_redundant_sources_are_refined_and_explicit_joint_reuses_the_coarse_observation():
    run, ids = _run(4)
    adapter = _EffectAdapter({
        _key(ids[0], ids[1]): 3.0,
        _key(ids[0]): 0.1,
        _key(ids[1]): 0.1,
    })

    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids[:2], ids[2:]],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=2, max_fine_arms=2, max_joint_arms=0,
        ),
        refine_threshold_nats=1.0,
        joint_source_sets=[ids[:2]],
        execution_adapter=adapter,
    )

    assert [
        result.arm_for("fine", [source_id]).delta_nats for source_id in ids[:2]
    ] == pytest.approx([0.1, 0.1])
    joint = result.arm_for("joint", ids[:2])
    assert joint.delta_nats == 3.0
    assert joint.evidence_status == "trusted"
    assert joint.execution_disposition == "reused"
    assert joint.reused_from_phase == "coarse"
    assert adapter.calls.count(_key(ids[0], ids[1])) == 1
    assert result.budget_usage["new_slots_reserved_before_cancellation"]["joint"] == 0


def test_joint_only_effect_runs_only_when_explicitly_requested():
    run, ids = _run(3)
    adapter = _EffectAdapter({_key(*ids): 4.0})

    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[[source_id] for source_id in ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=3, max_fine_arms=3, max_joint_arms=1,
        ),
        refine_threshold_nats=0.5,
        joint_source_sets=[ids],
        execution_adapter=adapter,
    )

    assert all(arm.evidence_status == "skipped" for arm in result.arms_for("fine"))
    joint = result.arm_for("joint", ids)
    assert joint.evidence_status == "trusted"
    assert joint.delta_nats == 4.0
    assert joint.selection_reason == "explicit_joint_intervention"
    assert result.coverage["joint_trusted_count"] == 1


def test_budget_exhaustion_is_explicit_at_every_stage():
    run, ids = _run(6)
    adapter = _EffectAdapter({
        _key(ids[0], ids[1]): 2.0,
        _key(ids[0]): 1.0,
        _key(ids[0], ids[2]): 3.0,
        _key(ids[1], ids[3]): 3.0,
    })

    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids[0:2], ids[2:4], ids[4:6]],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=1, max_fine_arms=1, max_joint_arms=1,
        ),
        refine_threshold_nats=0.5,
        joint_source_sets=[(ids[0], ids[2]), (ids[1], ids[3])],
        execution_adapter=adapter,
    )

    coarse = result.arms_for("coarse")
    assert [arm.evidence_status for arm in coarse] == ["trusted", "skipped", "skipped"]
    assert coarse[1].selection_reason == coarse[2].selection_reason == "coarse_budget_exhausted"
    assert result.arm_for("fine", [ids[0]]).evidence_status == "trusted"
    assert result.arm_for("fine", [ids[1]]).selection_reason == "fine_budget_exhausted"
    joints = result.arms_for("joint")
    assert [arm.evidence_status for arm in joints] == ["trusted", "skipped"]
    assert joints[1].selection_reason == "joint_budget_exhausted"
    assert result.budget_usage["exhausted"] == {
        "coarse": True, "fine": True, "joint": True,
    }
    assert result.coverage["coarse_skipped_group_count"] == 2
    assert len(adapter.calls) == 4  # one control plus one new arm in each phase


def test_durable_trusted_observations_are_reused_without_spending_new_slots(tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, ids = _run(4)
    effects = {
        _key(ids[0], ids[1]): 2.0,
        _key(ids[0]): 1.0,
        _key(ids[1]): 1.0,
    }
    first_budget = ContextEffectsSearchBudget(
        max_coarse_arms=1, max_fine_arms=2, max_joint_arms=0,
    )
    store = ObservationStore()
    first_adapter = _EffectAdapter(effects)
    measure_context_effects_coarse_to_fine(
        run, source_ids=ids, coarse_groups=[ids[:2], ids[2:]],
        budget=first_budget, refine_threshold_nats=0.5,
        execution_adapter=first_adapter, observation_store=store,
    )
    second_adapter = _EffectAdapter(effects)

    second = measure_context_effects_coarse_to_fine(
        run, source_ids=ids, coarse_groups=[ids[:2], ids[2:]],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=0, max_fine_arms=0, max_joint_arms=0,
        ),
        refine_threshold_nats=0.5,
        execution_adapter=second_adapter, observation_store=store,
    )

    assert second_adapter.calls == []
    assert second.budget_usage["new_slots_reserved_before_cancellation"] == {
        "coarse": 0, "fine": 0, "joint": 0,
    }
    assert second.budget_usage["cache_or_prior_hits"]["coarse"] == 1
    assert [arm.evidence_status for arm in second.arms_for("coarse")] == [
        "trusted", "skipped",
    ]
    reused = second.arms_for("coarse")[0]
    assert reused.numerical_regime == "scalar_reference"
    assert reused.execution_identity == {
        "kind": "clozn.score-tokens-scalar-reference.v1",
    }
    skipped = second.arms_for("coarse")[1]
    assert skipped.numerical_regime is None
    assert skipped.execution_identity is None
    assert second.budget_usage["exhausted"]["coarse"] is True
    actual = second.budget_usage["actual_work"]
    assert actual["model_measurements_executed_total"] == 0
    assert actual["model_measurements_reused_total"] == 4
    assert actual["saved_model_measurements_total"] >= 4


def test_screening_only_is_not_reported_as_trusted_or_skipped():
    run, ids = _run(2)

    class ScreeningAdapter(_EffectAdapter):
        def execute(self, state, intervention=None, *, evaluator=None, arm_id=None):
            if intervention is None:
                return super().execute(
                    state, intervention, evaluator=evaluator, arm_id=arm_id,
                )
            source_ids = tuple(intervention.source_ids)
            self.calls.append(source_ids)
            identity = execution_observation_identity(state, evaluator, intervention)
            return TokenScoreObservation(
                **identity,
                run_id=state.run_id,
                status="unavailable",
                evaluator_provenance={"evaluator": "fixture"},
                score_basis={"fixture": True},
                execution_provenance={"adapter": "fixture_screen"},
                proof_grade="screening",
                trusted=False,
                diagnostics={
                    "reason": "screening_not_selected_for_scalar_refinement",
                    "screening_total_logprob": -3.0,
                },
            )

    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=1, max_fine_arms=0, max_joint_arms=0,
        ),
        refine_threshold_nats=0.5,
        execution_adapter=ScreeningAdapter({}),
    )

    coarse = result.arms_for("coarse")[0]
    assert coarse.evidence_status == "screening_only"
    assert coarse.trusted is False
    assert coarse.delta_nats == 2.0
    assert result.coverage["coarse_screening_only_group_count"] == 1
    assert all(arm.evidence_status == "skipped" for arm in result.arms_for("fine"))


def test_cancellation_after_coarse_reports_reserved_but_no_refinement_work():
    run, ids = _run(3)
    cancelled = Event()

    class CancelAfterCoarse(_EffectAdapter):
        def execute(self, state, intervention=None, *, evaluator=None, arm_id=None):
            observation = super().execute(
                state, intervention, evaluator=evaluator, arm_id=arm_id,
            )
            if intervention is not None and len(intervention.source_ids) == 2:
                cancelled.set()
            return observation

    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids[:2]],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=1, max_fine_arms=2, max_joint_arms=1,
        ),
        refine_threshold_nats=0.5,
        joint_source_sets=[(ids[0], ids[2])],
        execution_adapter=CancelAfterCoarse({_key(*ids[:2]): 2.0}),
        cancel=cancelled,
    )

    assert result.budget_usage["new_slots_reserved_before_cancellation"] == {
        "coarse": 1, "fine": 2, "joint": 1,
    }
    actual = result.budget_usage["actual_work"]
    assert actual["model_measurements_executed_total"] == 2
    assert actual["interventions"]["executed_by_phase"] == {
        "coarse": 1, "fine": 0, "joint": 0,
    }
    assert all(
        arm.evidence_status == "skipped" for arm in result.arms_for("fine")
    )
    assert result.arm_for("joint", (ids[0], ids[2])).evidence_status == "skipped"


def test_explicit_native_screening_refines_every_selected_condition_through_scalar(
        tmp_path, monkeypatch):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, ids = _run(2)

    class NativeScreenSubstrate:
        score_tokens_many_proof_grade = True
        score_tokens_many_execution_strategy = "native_prefix_reuse_step_batch"

        def __init__(self):
            self.scalar_calls = 0
            self.native_calls = 0

        def identity_meta(self):
            return deepcopy(run["identity"])

        def run_meta(self):
            return deepcopy(run["meta"])

        @staticmethod
        def _row(messages):
            contents = {message.get("content") for message in messages}
            missing = sum(f"source {index}" not in contents for index in range(2))
            logprob = -1.0 - missing
            return [{"id": 10, "piece": "answer", "logprob": logprob}]

        def score_tokens(self, messages, continuation_ids=None, **_kwargs):
            self.scalar_calls += 1
            return self._row(messages)

        def score_tokens_many(self, arms, *, cancel=None):
            self.native_calls += 1
            return {
                "results": [self._row(arm["messages"]) for arm in arms],
                "execution_metadata": {
                    "execution_strategy": "native_prefix_reuse_step_batch",
                    "numerical_regime": "screening_unqualified",
                    "evidence_use": "screening",
                    "qualified": False,
                    "execution_identity": {"kind": "fixture-native-screen"},
                },
            }

    substrate = NativeScreenSubstrate()
    result = measure_context_effects_coarse_to_fine(
        run,
        source_ids=ids,
        coarse_groups=[ids],
        budget=ContextEffectsSearchBudget(
            max_coarse_arms=1, max_fine_arms=2, max_joint_arms=0,
        ),
        refine_threshold_nats=0.5,
        substrate=substrate,
        use_native_screening=True,
        observation_store=ObservationStore(),
    )

    assert result.native_screening_requested is True
    coarse = result.arm_for("coarse", ids)
    assert coarse.evidence_status == "screening_only"
    assert coarse.delta_nats == 2.0
    assert coarse.observation_id is None
    assert coarse.numerical_regime == "screening_unqualified"
    assert coarse.execution_identity == {"kind": "fixture-native-screen"}
    fine = result.arms_for("fine")
    assert [arm.evidence_status for arm in fine] == [
        "trusted", "trusted",
    ]
    assert [arm.numerical_regime for arm in fine] == [
        "scalar_reference", "scalar_reference",
    ]
    assert [arm.execution_identity for arm in fine] == [
        {"kind": "clozn.score-tokens-scalar-reference.v1"},
        {"kind": "clozn.score-tokens-scalar-reference.v1"},
    ]
    assert substrate.native_calls == 2  # one grouped screen, one fine screen
    assert substrate.scalar_calls == 3  # one control, two exact refinements
