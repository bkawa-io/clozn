#include "clozn/variant_score_checkpoint_reuse.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

clozn::VariantScoreGenerationState state() {
    // The tail represents generated tokens in a checkpoint-on-finish record. Only the declared
    // original prompt [0, prompt_tokens) is eligible as the score-variants baseline.
    return {{1, 2, 3, 4, 90, 91}, 6, 4, true, false};
}

template <typename Fn>
void assert_invalid(Fn&& fn) {
    bool raised = false;
    try {
        fn();
    } catch (const std::invalid_argument&) {
        raised = true;
    }
    assert(raised);
}

void test_late_divergence_reuses_original_generation_prefix() {
    const auto plan = clozn::make_variant_score_checkpoint_reuse_plan(
        state(), {1, 2, 3, 4}, {{1, 2, 3, 8}, {1, 2, 3, 9}}, 3, 64);
    assert(plan.applied);
    assert(plan.reason_code == "applied");
    assert(plan.common_prefix_tokens == 3);
    assert(plan.retained_prefix_tokens == 3);
    assert(plan.decoded_prompt_rows == 2);
    assert(plan.continuation_kv_rows == 4);
    assert(plan.required_kv_rows == 9);
}

void test_terminal_prompt_rewinds_one_row_to_recover_logits() {
    const auto plan = clozn::make_variant_score_checkpoint_reuse_plan(
        state(), {1, 2, 3, 4}, {{1, 2, 3}, {1, 2, 3, 8}}, 1, 64);
    assert(plan.applied);
    assert(plan.common_prefix_tokens == 3);
    assert(plan.retained_prefix_tokens == 2);
    assert(plan.decoded_prompt_rows == 3);  // one row for terminal, two for longer prompt
    assert(plan.continuation_kv_rows == 0);
    assert(plan.required_kv_rows == 5);

    const auto identical = clozn::make_variant_score_checkpoint_reuse_plan(
        state(), {1, 2, 3, 4}, {{1, 2, 3, 4}}, 2, 64);
    assert(identical.applied);
    assert(identical.retained_prefix_tokens == 3);
    assert(identical.decoded_prompt_rows == 1);
}

void test_identity_and_execution_regime_fail_closed() {
    auto checkpoint = state();
    auto plan = clozn::make_variant_score_checkpoint_reuse_plan(
        checkpoint, {1, 2, 7, 4}, {{1, 2, 3, 8}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_baseline_mismatch");

    checkpoint = state();
    checkpoint.has_steer = true;
    plan = clozn::make_variant_score_checkpoint_reuse_plan(
        checkpoint, {1, 2, 3, 4}, {{1, 2, 3, 8}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_has_steer");

    checkpoint = state();
    checkpoint.causal = false;
    plan = clozn::make_variant_score_checkpoint_reuse_plan(
        checkpoint, {1, 2, 3, 4}, {{1, 2, 3, 8}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_not_causal");
}

void test_malformed_history_and_shallow_overlap_fall_back() {
    auto checkpoint = state();
    checkpoint.prompt_tokens = 0;
    auto plan = clozn::make_variant_score_checkpoint_reuse_plan(
        checkpoint, {1, 2, 3, 4}, {{1, 2, 3, 8}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_history_invalid");

    checkpoint = state();
    plan = clozn::make_variant_score_checkpoint_reuse_plan(
        checkpoint, {1, 2, 3, 4}, {{9, 2, 3, 8}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_no_shared_prefix");

    plan = clozn::make_variant_score_checkpoint_reuse_plan(
        checkpoint, {1, 2, 3, 4}, {{1}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_reuse_too_shallow");

    auto long_checkpoint = state();
    long_checkpoint.tokens = {1, 2, 3, 4, 5, 6, 7, 8};
    long_checkpoint.n_past = 8;
    long_checkpoint.prompt_tokens = 8;
    plan = clozn::make_variant_score_checkpoint_reuse_plan(
        long_checkpoint, long_checkpoint.tokens,
        {{1, 2, 3, 4, 90}, {1, 2, 3, 4, 91}}, 2, 64);
    assert(!plan.applied);
    assert(plan.reason_code == "checkpoint_shared_prefix_too_short");
}

void test_capacity_is_checkpoint_shape_specific() {
    const auto exact = clozn::make_variant_score_checkpoint_reuse_plan(
        state(), {1, 2, 3, 4}, {{1, 2, 3, 8}, {1, 2, 3, 9}}, 3, 11, 2);
    assert(exact.applied);
    assert(exact.required_kv_rows == 11);

    const auto exceeded = clozn::make_variant_score_checkpoint_reuse_plan(
        state(), {1, 2, 3, 4}, {{1, 2, 3, 8}, {1, 2, 3, 9}}, 3, 10, 2);
    assert(!exceeded.applied);
    assert(exceeded.reason_code == "checkpoint_reuse_capacity_exceeded");
    assert(exceeded.required_kv_rows == 11);
}

void test_invalid_planner_arguments_throw() {
    assert_invalid([&] {
        clozn::make_variant_score_checkpoint_reuse_plan(
            state(), {1, 2, 3, 4}, {}, 1, 64);
    });
    assert_invalid([&] {
        clozn::make_variant_score_checkpoint_reuse_plan(
            state(), {1, 2, 3, 4}, {{1}}, 0, 64);
    });
    assert_invalid([&] {
        clozn::make_variant_score_checkpoint_reuse_plan(
            state(), {1, 2, 3, 4}, {{}}, 1, 64);
    });
    assert_invalid([&] {
        clozn::make_variant_score_checkpoint_reuse_plan(
            state(), {1, 2, 3, 4}, {{1}}, 1, 2, 2);
    });
}

}  // namespace

int main() {
    test_late_divergence_reuses_original_generation_prefix();
    test_terminal_prompt_rewinds_one_row_to_recover_logits();
    test_identity_and_execution_regime_fail_closed();
    test_malformed_history_and_shallow_overlap_fall_back();
    test_capacity_is_checkpoint_shape_specific();
    test_invalid_planner_arguments_throw();
    return 0;
}
