"""Focused model-free tests for direct set-valued Context Dependence."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.receipts.context_dependence import (
    ContextDependenceError,
    ContextDependenceSupportIncompatible,
    ContextDependenceStudy,
    INTERVENTION_OPERATOR,
    NEUTRALIZATION_OPERATOR,
    NEUTRALIZATION_PROVENANCE,
    MeasurementUnavailableError,
    PROVENANCE,
    SCHEMA,
    UnknownSourceIdError,
    measure_removal_effect,
)
from clozn.receipts.forced import MATCHED_LENGTH_NEUTRAL_FILLER_RECIPE, matched_length_neutral_filler
from clozn.runs.context_receipt import build_context_receipt


class StepClock:
    def __init__(self, *, start=10.0, step=0.01):
        self.value = start
        self.step = step

    def __call__(self):
        value = self.value
        self.value += self.step
        return value


class FakeScoreSub:
    """Only the teacher-forced scorer is usable; free generation is forbidden."""

    def __init__(self):
        self.calls = []

    def chat(self, *_args, **_kwargs):  # pragma: no cover - invoked only if production violates the contract
        raise AssertionError("Context Dependence must never generate")

    def score_tokens(self, messages, continuation_ids, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        contents = [message.get("content") for message in messages if isinstance(message, dict)]
        self.calls.append({
            "contents": contents,
            "continuation_ids": deepcopy(continuation_ids),
            "continuation": continuation,
            "block": block,
            "steer_strengths": deepcopy(steer_strengths),
        })
        # These source-specific terms make single-, pair-, and N-way deletion
        # distinguishable without a model.  The continuation is always reused.
        penalty = sum({"source A": 0.1, "source B": 0.2, "source C": 0.4}.get(item, 0.0)
                      for item in (set(["source A", "source B", "source C"]) - set(contents)))
        return [
            {"id": 11, "piece": "One", "logprob": -0.10 - penalty},
            {"id": 12, "piece": " two", "logprob": -0.20 - penalty},
            {"id": 13, "piece": " three", "logprob": -0.30 - penalty},
        ]


def _run(*, token_ids=True):
    messages = [
        {"role": "system", "content": "source A", "source_id": "client-A"},
        {"role": "user", "content": "source B", "source_id": "client-B"},
        {"role": "user", "content": "source C", "source_id": "client-C"},
    ]
    receipt = build_context_receipt(
        messages=messages,
        assembled_messages=messages,
        final_prompt="rendered exact prompt",
        run_id="run_cd_1",
        identity={"template_fingerprint": "0123456789abcdef"},
        privacy="full",
    )
    return {
        "id": "run_cd_1",
        "model": "test-model.gguf",
        "substrate": "FakeScoreSub",
        "identity": {"model_sha256": "model-artifact", "template_fingerprint": "0123456789abcdef", "captured_at": 123.0},
        "messages": deepcopy(messages),
        "assembled_messages": deepcopy(messages),
        "context_receipt": receipt,
        "final_prompt": "rendered exact prompt",
        "response": "One two three",
        "behavior": {"active_dials": {"careful": 0.5}},
        "trace": {"token_ids": [11, 12, 13]} if token_ids else {},
    }


def _source_ids(run):
    return [segment["segment_id"] for segment in run["context_receipt"]["assembled"]]


def _experiment(document):
    assert document["schema_version"] == SCHEMA
    assert len(document["experiments"]) == 1
    return document["experiments"][0]


def test_single_source_deletion_is_a_direct_measured_set_experiment():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)
    sub = FakeScoreSub()

    document = measure_removal_effect(run, sub, removed_source_ids=[source_a], clock=StepClock())
    experiment = _experiment(document)

    schemas.validate(document)
    assert experiment["intervention_operator"] == INTERVENTION_OPERATOR == "delete_source"
    assert experiment["removed_source_ids"] == [source_a]
    assert experiment["provenance"] == PROVENANCE == "measured"
    # v2 persists the evidence it actually scored: one delta for every token
    # in the recorded continuation, not one caller-selected target range.
    assert experiment["delta_nats"] == sum(experiment["per_token_delta_nats"])
    assert experiment["token_indices"] == [0, 1, 2]
    assert experiment["delta_nats"] > 0
    assert document["baseline"]["teacher_forced_logp"] == experiment["baseline_logp"]
    assert document["baseline"]["tokens"] == [
        {"index": 0, "token_id": 11, "piece": "One", "unicode_range": [0, 3], "logprob": -0.10},
        {"index": 1, "token_id": 12, "piece": " two", "unicode_range": [3, 7], "logprob": -0.20},
        {"index": 2, "token_id": 13, "piece": " three", "unicode_range": [7, 13], "logprob": -0.30},
    ]
    assert document["budget"] == {"passes_requested": 2, "passes_consumed": 2}
    # baseline then delete-source arm, never chat/generation.
    assert [call["contents"] for call in sub.calls] == [
        ["source A", "source B", "source C"],
        ["source B", "source C"],
    ]
    assert all(call["continuation_ids"] == [11, 12, 13] for call in sub.calls)
    assert all(call["continuation"] is None for call in sub.calls)
    assert all(call["steer_strengths"] == {"careful": 0.5} for call in sub.calls)


def test_two_source_and_arbitrary_n_source_deletion_preserve_remaining_message_order():
    run = _run()
    source_a, source_b, source_c = _source_ids(run)
    sub = FakeScoreSub()
    study = ContextDependenceStudy(run, sub, clock=StepClock())

    two = study.measure_removal_effect([source_a, source_c])
    all_three = study.measure_removal_effect([source_a, source_b, source_c])
    document = study.document()

    assert two["removed_source_ids"] == sorted([source_a, source_c])
    assert [item["message_index"] for item in two["exact_removed_ranges"]] == [0, 2]
    # First baseline, then the pair leaves B in its original relative order,
    # and arbitrary N deletes every selected source without a replacement.
    assert [call["contents"] for call in sub.calls] == [
        ["source A", "source B", "source C"], ["source B"], [],
    ]
    assert all_three["removed_source_ids"] == sorted([source_a, source_b, source_c])
    assert len(all_three["exact_removed_ranges"]) == 3
    assert document["budget"] == {"passes_requested": 3, "passes_consumed": 3}


def test_matched_length_neutralization_is_a_separate_cached_robustness_control():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)
    sub = FakeScoreSub()
    study = ContextDependenceStudy(run, sub, clock=StepClock())

    deletion = study.measure_removal_effect([source_a])
    control = study.measure_neutralization_control([source_a])
    repeat = study.measure_neutralization_control([source_a])
    document = study.document()

    schemas.validate(document)
    assert repeat == control and repeat is not control
    assert len(document["experiments"]) == 1
    assert document["experiments"][0] == deletion
    assert len(document["robustness_controls"]) == 1
    assert control["intervention_operator"] == NEUTRALIZATION_OPERATOR == "neutralize_source"
    assert control["provenance"] == NEUTRALIZATION_PROVENANCE
    assert control["neutralized_source_ids"] == [source_a]
    assert control["neutralization"] == {
        "operator": "neutralize_source",
        "strategy": "matched_length_neutral_filler",
        "recipe": MATCHED_LENGTH_NEUTRAL_FILLER_RECIPE,
        "length_contract": "unicode_code_points_exact",
        "utf8_byte_length_contract": "not_guaranteed",
        "message_structure": "preserved",
    }
    exact = control["exact_neutralized_ranges"][0]
    assert exact["original_unicode_code_points"] == exact["replacement_unicode_code_points"] == len("source A")
    assert exact["replacement_content_sha256"]
    # Baseline, canonical delete arm, and separately scored filler arm.  The
    # duplicate control request reuses its control_id and consumes no pass.
    assert [call["contents"] for call in sub.calls] == [
        ["source A", "source B", "source C"],
        ["source B", "source C"],
        [matched_length_neutral_filler(len("source A")), "source B", "source C"],
    ]
    assert document["budget"] == {"passes_requested": 3, "passes_consumed": 3}


def test_source_set_order_does_not_change_identity_but_set_membership_does():
    run = _run()
    source_a, source_b, source_c = _source_ids(run)

    first = _experiment(measure_removal_effect(
        run, FakeScoreSub(), removed_source_ids=[source_c, source_a], clock=StepClock(start=1),
    ))
    second = _experiment(measure_removal_effect(
        run, FakeScoreSub(), removed_source_ids=[source_a, source_c], clock=StepClock(start=900, step=0.37),
    ))
    different = _experiment(measure_removal_effect(
        run, FakeScoreSub(), removed_source_ids=[source_a, source_b], clock=StepClock(),
    ))

    assert first["experiment_id"] == second["experiment_id"]
    assert first["experiment_id"] != different["experiment_id"]
    assert first["score_ms"] != second["score_ms"]


def test_unknown_source_id_fails_closed_before_any_score_call():
    study = ContextDependenceStudy(_run(), FakeScoreSub())
    with pytest.raises(UnknownSourceIdError, match="unknown canonical Context Receipt source ID"):
        study.measure_removal_effect(["seg_not_a_real_source"])
    assert study._sub.calls == []


def test_stale_or_mismatched_receipt_segment_cannot_delete_a_scoring_message():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)
    # The receipt claims the canonical ID at index zero, while its hash no
    # longer describes the exact message that will be teacher-forced.  The
    # measurement must not use index alone to delete that scoring message.
    run["context_receipt"]["assembled"][0]["content_hash"] = "0" * 16
    sub = FakeScoreSub()
    study = ContextDependenceStudy(run, sub)

    with pytest.raises(UnknownSourceIdError, match="unknown canonical Context Receipt source ID"):
        study.measure_removal_effect([source_a])
    assert sub.calls == []


def test_exact_token_measurement_fails_closed_when_scorer_text_is_not_recorded_response():
    run = _run()
    run["response"] = "A different recorded answer"
    source_a, _source_b, _source_c = _source_ids(run)
    sub = FakeScoreSub()

    with pytest.raises(MeasurementUnavailableError, match="did not decode to the recorded response text"):
        measure_removal_effect(run, sub, removed_source_ids=[source_a], clock=StepClock())
    # Baseline is allowed to establish the mismatch; no deletion arm executes.
    assert len(sub.calls) == 1


def test_intervened_vector_refuses_a_different_continuation_tokenization():
    class DriftingArmSub(FakeScoreSub):
        def score_tokens(self, messages, continuation_ids, **kwargs):
            tokens = super().score_tokens(messages, continuation_ids, **kwargs)
            if len(self.calls) == 2:
                tokens[1] = {**tokens[1], "piece": " TWO", "id": 999}
            return tokens

    run = _run(token_ids=False)
    source_a, _source_b, _source_c = _source_ids(run)

    with pytest.raises(MeasurementUnavailableError, match="token pieces"):
        measure_removal_effect(
            run, DriftingArmSub(), removed_source_ids=[source_a], clock=StepClock(),
        )


def test_target_is_rejected_so_it_cannot_bind_a_run_level_measurement_identity():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)

    sub = FakeScoreSub()
    with pytest.raises(ContextDependenceError, match="target is not accepted"):
        measure_removal_effect(
            run, sub,
            target={"recorded_token_range": [1, 3], "recorded_prefix_range": [0, 1]},
            removed_source_ids=[source_a],
            clock=StepClock(),
        )
    assert sub.calls == []


def test_recorded_text_fallback_is_explicitly_retokenized_and_approximate():
    run = _run(token_ids=False)
    source_a, _source_b, _source_c = _source_ids(run)
    document = measure_removal_effect(run, FakeScoreSub(), removed_source_ids=[source_a], clock=StepClock())

    assert document["continuation"] == {
        "fidelity": "recomputed_from_recorded_response_text",
        "scored_text": "One two three",
        "unicode_offset_basis": "recorded_response_unicode",
        "kind": "recorded_response_text_retokenized",
        "token_ids_exact": False,
        "retokenized": True,
        "recorded_text": "One two three",
    }


def test_run_is_not_mutated_and_cached_baseline_is_reused():
    run = _run()
    before = deepcopy(run)
    source_a, source_b, _source_c = _source_ids(run)
    sub = FakeScoreSub()
    study = ContextDependenceStudy(run, sub, clock=StepClock())

    study.measure_removal_effect([source_a])
    study.measure_removal_effect([source_b])
    document = study.document()

    assert run == before
    assert len(sub.calls) == 3  # one baseline + one arm per source set
    assert document["baseline"]["scored_once"] is True


def test_identical_source_set_reuses_experiment_without_another_arm_or_duplicate_record():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)
    sub = FakeScoreSub()
    study = ContextDependenceStudy(run, sub, clock=StepClock())

    first = study.measure_removal_effect([source_a])
    second = study.measure_removal_effect([source_a])
    document = study.document()

    assert second == first
    assert second is not first
    assert len(sub.calls) == 2  # one baseline + one unique deletion arm
    assert len(document["experiments"]) == 1
    assert document["budget"] == {"passes_requested": 2, "passes_consumed": 2}


def test_compatible_persisted_support_hydrates_baseline_and_experiments_without_scoring_again():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)
    first_sub = FakeScoreSub()
    first = ContextDependenceStudy(run, first_sub, clock=StepClock())
    original = first.measure_removal_effect([source_a])
    document = first.document()

    second_sub = FakeScoreSub()
    second = ContextDependenceStudy(run, second_sub, existing_document=document, clock=StepClock())
    reused = second.measure_removal_effect([source_a])

    assert reused == original
    assert second_sub.calls == []
    assert second.direct_evidence() == [original]


def test_support_bound_to_one_search_universe_is_not_reused_for_another():
    run = _run()
    source_a, _source_b, _source_c = _source_ids(run)
    first = ContextDependenceStudy(
        run,
        FakeScoreSub(),
        search_universe_id="mcu_one",
        runtime_identity={"runtime_key_sha256": "a" * 64},
        clock=StepClock(),
    )
    first.measure_removal_effect([source_a])
    document = first.document()

    with pytest.raises(ContextDependenceSupportIncompatible, match="search universe"):
        ContextDependenceStudy(
            run,
            FakeScoreSub(),
            search_universe_id="mcu_two",
            runtime_identity={"runtime_key_sha256": "a" * 64},
            existing_document=document,
            clock=StepClock(),
        )
