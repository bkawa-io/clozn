#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clozn {

// Partition trie rows for llama.cpp backend sampling. output_sequence_ids[row] is empty for a
// no-logits row and otherwise lists every sequence carried by that output row. llama.cpp accepts at
// most one output per sequence in a decode, so a prefix prompt's terminal row and a descendant
// prompt's terminal row must be separated even though independent sibling terminals may stay
// batched. Returned ranges are [begin,end), cover every row once, and never exceed max_rows.
inline std::vector<std::pair<size_t, size_t>> target_logprob_decode_ranges(
        const std::vector<std::vector<int>>& output_sequence_ids,
        size_t max_rows,
        int n_sequences) {
    if (max_rows == 0) throw std::invalid_argument("max_rows must be positive");
    if (n_sequences <= 0) throw std::invalid_argument("n_sequences must be positive");

    std::vector<std::pair<size_t, size_t>> ranges;
    for (size_t begin = 0; begin < output_sequence_ids.size();) {
        std::vector<bool> output_seen(static_cast<size_t>(n_sequences), false);
        size_t end = begin;
        while (end < output_sequence_ids.size() && end - begin < max_rows) {
            bool conflict = false;
            for (const int sequence : output_sequence_ids[end]) {
                if (sequence < 0 || sequence >= n_sequences)
                    throw std::invalid_argument("output sequence id is out of range");
                conflict |= output_seen[static_cast<size_t>(sequence)];
            }
            if (conflict) break;
            for (const int sequence : output_sequence_ids[end])
                output_seen[static_cast<size_t>(sequence)] = true;
            ++end;
        }
        // A valid first row can never conflict with an empty seen set.
        if (end == begin) throw std::logic_error("target logprob chunk planner made no progress");
        ranges.emplace_back(begin, end);
        begin = end;
    }
    return ranges;
}

}  // namespace clozn
