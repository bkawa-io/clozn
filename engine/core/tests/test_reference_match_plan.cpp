#include "clozn/reference_match_plan.hpp"

#include <cassert>
#include <string>
#include <utility>
#include <vector>

using clozn::ReferenceMatchTraversalPlan;

namespace {

void test_shared_prefix() {
    const std::vector<std::vector<int>> prompts{
        {1, 2, 3, 4, 10},
        {1, 2, 3, 4, 11},
        {1, 2, 3, 5, 12},
    };
    const auto plan = clozn::make_reference_match_traversal_plan(prompts);
    assert(clozn::reference_match_plan_physical_rows(plan) == 8);
    assert(15 - clozn::reference_match_plan_physical_rows(plan) == 7);
}

void test_suffixes_do_not_remerge() {
    const std::vector<std::vector<int>> prompts{
        {1, 2, 10, 20, 30},
        {1, 2, 11, 20, 30},
    };
    const auto plan = clozn::make_reference_match_traversal_plan(prompts);
    assert(clozn::reference_match_plan_physical_rows(plan) == 8);

    int kv = 0;
    std::vector<std::pair<int, int>> decodes;
    int probes = 0;
    clozn::walk_reference_match_traversal(
        plan,
        [&](int, int from, int to, bool) {
            assert(kv == from);
            kv = to;
            decodes.emplace_back(from, to);
        },
        [&](int, int, int depth) {
            assert(kv == depth);
            ++probes;
            kv = depth;  // fake probe rollback
        },
        [&](int depth, int) {
            assert(kv >= depth);
            kv = depth;
        });
    assert(kv == 0);
    assert(probes == 2);
    int shared_ab = 0;
    int shared_qr = 0;
    for (const auto& segment : decodes) {
        if (segment == std::make_pair(0, 2)) ++shared_ab;
        if (segment == std::make_pair(2, 5)) ++shared_qr;
    }
    assert(shared_ab == 1);
    assert(shared_qr == 2);  // one Q/R path per divergent history, never re-merged
}

void test_terminal_and_duplicates() {
    {
        const std::vector<std::vector<int>> prompts{{1, 2, 3}, {1, 2, 3, 4}};
        const auto plan = clozn::make_reference_match_traversal_plan(prompts);
        assert(clozn::reference_match_plan_physical_rows(plan) == 4);
        int probe_calls = 0;
        clozn::walk_reference_match_traversal(
            plan, [&](int, int, int, bool) {},
            [&](int begin, int end, int depth) {
                assert(end - begin == 1);
                assert(depth == 3);
                ++probe_calls;
            },
            [&](int, int) {});
        assert(probe_calls == 1);
    }
    {
        const std::vector<std::vector<int>> prompts{{1, 2, 3}, {1, 2, 3}};
        const auto plan = clozn::make_reference_match_traversal_plan(prompts);
        assert(clozn::reference_match_plan_physical_rows(plan) == 3);
        int terminal_arms = 0;
        int probe_calls = 0;
        clozn::walk_reference_match_traversal(
            plan, [&](int, int, int, bool) {},
            [&](int begin, int end, int depth) {
                assert(depth == 3);
                terminal_arms += end - begin;
                ++probe_calls;
            },
            [&](int, int) {});
        assert(probe_calls == 1);
        assert(terminal_arms == 2);
    }
}

}  // namespace

int main() {
    test_shared_prefix();
    test_suffixes_do_not_remerge();
    test_terminal_and_duplicates();
    return 0;
}
