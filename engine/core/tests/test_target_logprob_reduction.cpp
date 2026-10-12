#include "clozn/target_logprob_reduction.hpp"

#include <cassert>
#include <stdexcept>
#include <utility>
#include <vector>

int main() {
    using Range = std::pair<size_t, size_t>;

    // Independent sibling prompt terminals can retain one batched decode.
    assert((clozn::target_logprob_decode_ranges(
                {{}, {0}, {}, {1}, {2}}, 16, 3) ==
            std::vector<Range>{{0, 5}}));

    // A short prompt terminal carries both itself and its longer descendant. The descendant's own
    // terminal would be a second output for seq 1, so it begins a new decode.
    assert((clozn::target_logprob_decode_ranges(
                {{}, {0, 1}, {}, {1}}, 16, 2) ==
            std::vector<Range>{{0, 3}, {3, 4}}));

    // The normal batch ceiling still applies when there are no output conflicts.
    assert((clozn::target_logprob_decode_ranges(
                {{}, {}, {}, {}, {}}, 2, 1) ==
            std::vector<Range>{{0, 2}, {2, 4}, {4, 5}}));

    bool rejected = false;
    try {
        (void) clozn::target_logprob_decode_ranges({{2}}, 1, 2);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}
