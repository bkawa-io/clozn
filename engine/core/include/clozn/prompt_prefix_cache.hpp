// Exact-token prompt-prefix reuse planning for ordinary autoregressive serving.
#pragma once

#include <algorithm>
#include <stdexcept>
#include <vector>

namespace clozn {

inline int exact_token_prefix_length(const std::vector<int>& left,
                                     const std::vector<int>& right) {
    const size_t limit = std::min(left.size(), right.size());
    size_t index = 0;
    while (index < limit && left[index] == right[index]) ++index;
    return static_cast<int>(index);
}

struct PromptPrefixPlan {
    int matched_tokens = 0;
    int decode_from = 0;
    int evaluated_tokens = 0;
    int cached_tokens = 0;
};

inline PromptPrefixPlan plan_prompt_prefix_reuse(int prompt_tokens, int matched_tokens) {
    if (prompt_tokens < 1)
        throw std::invalid_argument("prompt prefix reuse requires a non-empty prompt");
    if (matched_tokens < 0 || matched_tokens > prompt_tokens)
        throw std::invalid_argument("matched prompt prefix is outside the prompt");

    PromptPrefixPlan plan;
    plan.matched_tokens = matched_tokens;
    plan.decode_from = matched_tokens;
    if (matched_tokens == prompt_tokens) {
        // Resident KV has no logits row. Re-evaluate the final prompt token as a one-token bridge.
        plan.decode_from = prompt_tokens - 1;
    }
    plan.evaluated_tokens = prompt_tokens - plan.decode_from;
    plan.cached_tokens = prompt_tokens - plan.evaluated_tokens;
    return plan;
}

struct ScorePrefixPlan {
    int matched_tokens = 0;
    int decode_from = 0;
    int evaluated_tokens = 0;
    int cached_tokens = 0;
};

inline ScorePrefixPlan plan_score_prefix_reuse(int sequence_tokens, int matched_tokens,
                                               int first_logits_position) {
    if (sequence_tokens < 1)
        throw std::invalid_argument("score prefix reuse requires a non-empty sequence");
    if (matched_tokens < 0 || matched_tokens > sequence_tokens)
        throw std::invalid_argument("matched score prefix is outside the sequence");
    if (first_logits_position < 0 || first_logits_position >= sequence_tokens)
        throw std::invalid_argument("first score logits position is outside the sequence");

    ScorePrefixPlan plan;
    plan.matched_tokens = matched_tokens;
    // KV does not retain logits. Even for an identical sequence, the first requested logits row
    // and everything after it must be decoded again.
    plan.decode_from = std::min(matched_tokens, first_logits_position);
    plan.evaluated_tokens = sequence_tokens - plan.decode_from;
    plan.cached_tokens = plan.decode_from;
    return plan;
}

}  // namespace clozn
