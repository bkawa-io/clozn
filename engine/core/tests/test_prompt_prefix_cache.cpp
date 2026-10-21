#include "clozn/prompt_prefix_cache.hpp"

#include <cassert>
#include <stdexcept>
#include <vector>

int main() {
    using clozn::exact_token_prefix_length;
    using clozn::plan_prompt_prefix_reuse;
    using clozn::plan_score_prefix_reuse;

    assert(exact_token_prefix_length({1, 2, 3}, {1, 2, 4}) == 2);
    assert(exact_token_prefix_length({1, 2}, {1, 2, 3}) == 2);
    assert(exact_token_prefix_length({9}, {1}) == 0);

    const auto cold = plan_prompt_prefix_reuse(10, 0);
    assert(cold.decode_from == 0);
    assert(cold.evaluated_tokens == 10);
    assert(cold.cached_tokens == 0);

    const auto appended = plan_prompt_prefix_reuse(10, 7);
    assert(appended.decode_from == 7);
    assert(appended.evaluated_tokens == 3);
    assert(appended.cached_tokens == 7);

    const auto identical = plan_prompt_prefix_reuse(10, 10);
    assert(identical.decode_from == 9);
    assert(identical.evaluated_tokens == 1);
    assert(identical.cached_tokens == 9);

    const auto scored = plan_score_prefix_reuse(14, 8, 9);
    assert(scored.decode_from == 8);
    assert(scored.evaluated_tokens == 6);
    assert(scored.cached_tokens == 8);

    const auto identical_score = plan_score_prefix_reuse(14, 14, 9);
    assert(identical_score.decode_from == 9);
    assert(identical_score.evaluated_tokens == 5);
    assert(identical_score.cached_tokens == 9);

    bool rejected = false;
    try {
        (void) plan_prompt_prefix_reuse(4, 5);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
