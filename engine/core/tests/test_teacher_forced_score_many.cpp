#include "clozn/teacher_forced_score_many.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    {
        const auto plan = clozn::plan_teacher_forced_score_many(
            {{1, 2, 3, 4}, {1, 2, 8}, {1, 2, 3, 9}}, 3, 32);
        assert(plan.arm_count == 3);
        assert(plan.logical_prompt_rows == 11);
        // [1,2] + [3,4] + [8] + [9]
        assert(plan.physical_prompt_rows == 6);
        assert(plan.prefix_rows_reused == 5);
        assert(plan.continuation_rows == 6);
        assert(plan.total_physical_rows == 12);
    }
    {
        const auto plan = clozn::plan_teacher_forced_score_many(
            {{7, 8}, {7, 8}}, 1, 4);
        assert(plan.logical_prompt_rows == 4);
        assert(plan.physical_prompt_rows == 2);
        assert(plan.prefix_rows_reused == 2);
        assert(plan.continuation_rows == 0);
        assert(plan.total_physical_rows == 2);
    }

    bool rejected = false;
    try {
        (void) clozn::plan_teacher_forced_score_many({{1, 2}, {1, 3}}, 3, 5);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);

    rejected = false;
    try {
        (void) clozn::plan_teacher_forced_score_many({}, 1, 8);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
