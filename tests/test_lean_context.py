from __future__ import annotations

from clozn import schemas
from clozn.experiments.persistence import ObservationStore
from clozn.experiments.score_context_search import continuation_support_loss
from clozn.recipes.context_effects import measure_context_effects
from clozn.recipes.lean_context import run_lean_context
from clozn.runs import store as run_store

from tests.test_context_effects_kernel import ScoreSubstrate, _run


class TokenCountingEngine:
    def apply_template_info(self, messages):
        return {
            "prompt": "fixture",
            "prompt_tokens": sum(len(str(message.get("content", ""))) for message in messages),
        }


def test_continuation_support_loss_clamps_improvements_instead_of_offsetting_losses():
    run, source_ids = _run()
    result = measure_context_effects(
        run, source_ids=[source_ids[0]], substrate=ScoreSubstrate(),
    )

    loss = continuation_support_loss(result.control, result.arms[0])

    assert loss == {
        "answer_token_count": 2,
        "positive_loss_token_count": 1,
        "mean_positive_support_loss_nats_per_token": 0.5,
        "max_token_support_loss_nats": 1.0,
    }


def test_lean_context_returns_a_tolerant_best_found_claim_not_a_minimum_claim():
    run, source_ids = _run()

    result = run_lean_context(
        run, substrate=ScoreSubstrate(), engine=TokenCountingEngine(),
    )

    assert result["status"] == "completed"
    assert result["claim"] == {
        "kind": "best_found_within_budget",
        "minimum_proven": False,
    }
    assert result["tolerance"] == {
        "profile": "balanced.v1",
        "mean_positive_support_loss_nats_per_token": 0.05,
        "max_token_support_loss_nats": 0.7,
    }
    assert result["best"]["removed_source_ids"] == [source_ids[2]]
    assert result["best"]["tolerance_status"] == "within_tolerance"
    assert result["budget"]["used_new_executions"] <= 16
    schemas.validate(result)


def test_lean_context_reuses_context_effects_observations_without_new_scoring(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(run_store, "RUNS_DIR", str(tmp_path / "runs"))
    run_store._schema_verified.clear()
    run, source_ids = _run()
    store = ObservationStore()
    measure_context_effects(
        run,
        source_ids=source_ids,
        substrate=ScoreSubstrate(),
        observation_store=store,
    )
    lean_substrate = ScoreSubstrate()

    result = run_lean_context(
        run,
        substrate=lean_substrate,
        engine=TokenCountingEngine(),
        observation_store=store,
        max_new_executions=0,
    )

    assert result["status"] == "completed"
    assert result["best"]["removed_source_ids"] == [source_ids[2]]
    assert result["budget"]["used_new_executions"] == 0
    assert result["budget"]["reused_observation_count"] > 0
    assert lean_substrate.calls == []
    schemas.validate(result)
