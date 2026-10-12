#include "clozn/variant_score_plan.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

namespace {

clozn::VariantScoreCapacity capacity(long long kv_rows = 1000, int max_sequences = 16,
                                     int max_sequence_tokens = 1000, long long reserve_rows = 0) {
    return clozn::VariantScoreCapacity{kv_rows, reserve_rows, max_sequences, max_sequence_tokens};
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

template <typename Fn>
void assert_out_of_range(Fn&& fn) {
    bool raised = false;
    try {
        fn();
    } catch (const std::out_of_range&) {
        raised = true;
    }
    assert(raised);
}

void test_shared_prefix_and_no_remerge_accounting() {
    const auto shared = clozn::make_variant_score_plan(
        {{1, 2, 3, 4, 10}, {1, 2, 3, 4, 11}, {1, 2, 3, 5, 12}}, 2, capacity());
    assert(shared.waves.size() == 1);
    assert(shared.waves[0].logical_prompt_rows == 15);
    assert(shared.waves[0].deduplicated_prompt_rows == 15);
    assert(shared.waves[0].physical_prompt_rows == 8);
    assert(shared.waves[0].logical_continuation_kv_rows == 3);
    assert(shared.waves[0].continuation_kv_rows == 3);
    assert(shared.waves[0].required_kv_rows == 11);

    const auto no_remerge = clozn::make_variant_score_plan(
        {{1, 2, 10, 20, 30}, {1, 2, 11, 20, 30}}, 2, capacity());
    assert(no_remerge.waves[0].physical_prompt_rows == 8);
    assert(no_remerge.waves[0].continuation_kv_rows == 2);
}

void test_no_overlap_and_divergence_depths() {
    const auto disjoint = clozn::make_variant_score_plan({{1, 2, 3}, {9, 8}}, 1, capacity());
    assert(disjoint.waves[0].deduplicated_prompt_rows == 5);
    assert(disjoint.waves[0].physical_prompt_rows == 5);

    const auto early = clozn::make_variant_score_plan({{1, 2, 3}, {9, 2, 3}}, 1, capacity());
    const auto middle = clozn::make_variant_score_plan({{1, 2, 3, 4}, {1, 2, 9, 4}}, 1, capacity());
    const auto late = clozn::make_variant_score_plan({{1, 2, 3}, {1, 2, 9}}, 1, capacity());
    assert(early.waves[0].physical_prompt_rows == 6);
    assert(middle.waves[0].physical_prompt_rows == 6);
    assert(late.waves[0].physical_prompt_rows == 4);
}

void test_duplicates_collapse_and_map_back_in_caller_order() {
    const std::vector<std::vector<int>> prompts{{7, 8}, {1, 2}, {7, 8}, {1, 2, 3}};
    const auto plan = clozn::make_variant_score_plan(prompts, 2, capacity());
    assert(plan.unique_prompts.size() == 3);
    assert((plan.unique_prompts[0].tokens == std::vector<int>{1, 2}));
    assert((plan.unique_prompts[1].tokens == std::vector<int>{1, 2, 3}));
    assert((plan.unique_prompts[2].tokens == std::vector<int>{7, 8}));
    assert((plan.unique_prompts[2].caller_indexes == std::vector<int>{0, 2}));
    assert((plan.caller_to_unique == std::vector<int>{2, 0, 2, 1}));
    assert(plan.waves[0].logical_prompt_rows == 9);
    assert(plan.waves[0].deduplicated_prompt_rows == 7);
    assert(plan.waves[0].logical_continuation_kv_rows == 4);
    assert(plan.waves[0].continuation_kv_rows == 3);  // three histories, not four callers
}

void test_duplicate_only_requests_consume_one_sequence_and_allocation() {
    const auto plan = clozn::make_variant_score_plan(
        {{1, 2, 3}, {1, 2, 3}, {1, 2, 3}}, 2, capacity(4, 1));
    assert(plan.unique_prompts.size() == 1);
    assert(plan.waves.size() == 1);
    assert(plan.waves[0].unique_prompt_indexes.size() == 1);
    assert(plan.waves[0].required_kv_rows == 4);
    assert(plan.waves[0].logical_prompt_rows == 9);
    assert(plan.waves[0].deduplicated_prompt_rows == 3);
    assert(plan.waves[0].logical_continuation_kv_rows == 3);
    assert(plan.waves[0].continuation_kv_rows == 1);
}

void test_terminal_prompt_and_continuation_are_separate_branches() {
    const auto plan = clozn::make_variant_score_plan({{1, 2}, {1, 2, 3}}, 2, capacity());
    assert(plan.waves[0].physical_prompt_rows == 3);
    assert(plan.waves[0].continuation_kv_rows == 2);
    assert(plan.waves[0].required_kv_rows == 5);
    const auto& traversal = plan.waves[0].prompt_traversal;
    assert(traversal.nodes[static_cast<size_t>(traversal.root)].terminal_end -
           traversal.nodes[static_cast<size_t>(traversal.root)].terminal_begin == 1);
}

void test_kv_and_sequence_limits_create_deterministic_waves() {
    // The first two prompts need four physical prompt rows plus two continuation KV rows.  Adding the
    // disjoint third prompt would need nine total rows, so a six-row budget splits 2 + 1.
    const std::vector<std::vector<int>> prompts{{9, 8}, {1, 2, 4}, {1, 2, 3}};
    const auto plan = clozn::make_variant_score_plan(prompts, 2, capacity(6));
    assert(plan.waves.size() == 2);
    assert((plan.waves[0].unique_prompt_indexes == std::vector<int>{0, 1}));
    assert(plan.waves[0].required_kv_rows == 6);
    assert((plan.waves[1].unique_prompt_indexes == std::vector<int>{2}));
    assert(plan.waves[1].required_kv_rows == 3);

    const auto sequence_limited = clozn::make_variant_score_plan(prompts, 2, capacity(100, 1));
    assert(sequence_limited.waves.size() == 3);
    for (const auto& wave : sequence_limited.waves)
        assert(wave.unique_prompt_indexes.size() == 1);
}

void test_reserve_is_part_of_every_capacity_decision() {
    const auto plan = clozn::make_variant_score_plan(
        {{1, 2, 3}, {1, 2, 4}}, 2, capacity(8, 16, 1000, 2));
    assert(plan.waves.size() == 1);
    assert(plan.waves[0].physical_prompt_rows == 4);
    assert(plan.waves[0].continuation_kv_rows == 2);
    assert(plan.waves[0].required_kv_rows == 8);

    const auto split = clozn::make_variant_score_plan(
        {{1, 2, 3}, {1, 2, 4}}, 2, capacity(7, 16, 1000, 2));
    assert(split.waves.size() == 2);
}

void test_planning_is_independent_of_caller_order() {
    const auto a = clozn::make_variant_score_plan(
        {{9, 8}, {1, 2, 4}, {1, 2, 3}}, 2, capacity(6));
    const auto b = clozn::make_variant_score_plan(
        {{1, 2, 3}, {9, 8}, {1, 2, 4}}, 2, capacity(6));
    assert(a.unique_prompts.size() == b.unique_prompts.size());
    for (size_t i = 0; i < a.unique_prompts.size(); ++i)
        assert(a.unique_prompts[i].tokens == b.unique_prompts[i].tokens);
    assert(a.waves.size() == b.waves.size());
    for (size_t i = 0; i < a.waves.size(); ++i) {
        assert(a.waves[i].unique_prompt_indexes == b.waves[i].unique_prompt_indexes);
        assert(a.waves[i].required_kv_rows == b.waves[i].required_kv_rows);
    }
}

void test_wave_local_traversal_indexes_compose_to_global_unique_indexes() {
    const auto plan = clozn::make_variant_score_plan(
        {{7, 1}, {1, 1}, {8, 1}, {2, 1}}, 1, capacity(100, 2));
    assert(plan.waves.size() == 2);
    const auto& second = plan.waves[1];
    assert((second.unique_prompt_indexes == std::vector<int>{2, 3}));
    assert(second.prompt_traversal.order.size() == 2);
    assert(second.global_unique_prompt_index(second.prompt_traversal.order[0]) == 2);
    assert(second.global_unique_prompt_index(second.prompt_traversal.order[1]) == 3);
    assert_out_of_range([&] { second.global_unique_prompt_index(2); });
    assert((plan.unique_prompts[2].caller_indexes == std::vector<int>{0}));
    assert((plan.unique_prompts[3].caller_indexes == std::vector<int>{2}));
}

void test_invalid_inputs_and_capacity_boundaries() {
    assert_invalid([&] { clozn::make_variant_score_plan({}, 1, capacity()); });
    assert_invalid([&] { clozn::make_variant_score_plan({{}}, 1, capacity()); });
    assert_invalid([&] { clozn::make_variant_score_plan({{-1}}, 1, capacity()); });
    assert_invalid([&] { clozn::make_variant_score_plan({{1}}, 0, capacity()); });
    assert_invalid([&] { clozn::make_variant_score_plan({{1}}, 1, capacity(0)); });
    assert_invalid([&] { clozn::make_variant_score_plan({{1}}, 1, capacity(10, 0)); });
    assert_invalid([&] { clozn::make_variant_score_plan({{1, 2}}, 2, capacity(100, 16, 3)); });
    assert_invalid([&] { clozn::make_variant_score_plan({{1, 2, 3}}, 2, capacity(3)); });
    assert_invalid([&] { clozn::make_variant_score_plan({{1}}, 1, capacity(10, 16, 100, 10)); });

    const auto exact = clozn::make_variant_score_plan({{1, 2, 3}}, 2, capacity(4));
    assert(exact.waves.size() == 1);
    assert(exact.waves[0].required_kv_rows == 4);

    // A one-token continuation comes entirely from prompt-final logits and adds no KV row.  The
    // declared support limit still includes the target so the request has a scalar /score oracle.
    const auto one_target = clozn::make_variant_score_plan(
        {{1, 2, 3}}, 1, capacity(3, 1, 4));
    assert(one_target.waves[0].continuation_kv_rows == 0);
    assert(one_target.waves[0].required_kv_rows == 3);
    assert_invalid([&] {
        clozn::make_variant_score_plan({{1, 2, 3}}, 1, capacity(3, 1, 3));
    });
}

}  // namespace

int main() {
    test_shared_prefix_and_no_remerge_accounting();
    test_no_overlap_and_divergence_depths();
    test_duplicates_collapse_and_map_back_in_caller_order();
    test_duplicate_only_requests_consume_one_sequence_and_allocation();
    test_terminal_prompt_and_continuation_are_separate_branches();
    test_kv_and_sequence_limits_create_deterministic_waves();
    test_reserve_is_part_of_every_capacity_decision();
    test_planning_is_independent_of_caller_order();
    test_wave_local_traversal_indexes_compose_to_global_unique_indexes();
    test_invalid_inputs_and_capacity_boundaries();
    return 0;
}
