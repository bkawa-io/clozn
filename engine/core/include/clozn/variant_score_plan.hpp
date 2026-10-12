// Exact-token planning for scoring one recorded continuation under related prompts.
//
// This layer is deliberately model-free.  It decides which distinct prompt histories may be
// resident together, accounts for prefix-shared KV rows, and preserves the mapping back to every
// caller-supplied variant.  It does not claim that a backend can execute the plan: the eventual
// executor must still check its sequence-copy, cache, and batching capabilities at runtime.
// This first plan bounds KV rows, sequence count, and per-sequence positions only.  Compute/output
// workspace and request-body/input-count limits remain separate executor and route preflights.
#pragma once

#include "clozn/reference_match_plan.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

namespace clozn {

struct VariantScoreCapacity {
    // Unified KV rows available to this request, including the explicit reserve below.
    long long kv_rows = 0;
    // Rows held back for backend/runtime needs.  They are included in Wave::required_kv_rows.
    long long reserve_rows = 0;
    // Maximum simultaneously resident distinct histories.  Duplicate complete requests do not
    // consume another sequence because they are scored once and fanned back out to callers.
    int max_sequences = 0;
    // Conservative per-sequence support limit.  Prompt plus the full recorded continuation must
    // fit so every planned request stays comparable with the current scalar /score oracle, even
    // though the optimized executor only needs N-1 continuation tokens resident in KV.
    int max_sequence_tokens = 0;
};

struct VariantScorePlan {
    struct UniquePrompt {
        std::vector<int> tokens;
        // Original caller indexes in stable caller order.  More than one entry means the complete
        // request was an exact duplicate and can be executed once.
        std::vector<int> caller_indexes;
    };

    struct Wave {
        // Indexes into unique_prompts, in deterministic token-lexicographic execution order.
        // prompt_traversal indexes are LOCAL to this vector: compose
        // unique_prompt_indexes[prompt_traversal.order[i]] before mapping through caller_indexes.
        std::vector<int> unique_prompt_indexes;
        ReferenceMatchTraversalPlan prompt_traversal;
        // Logical rows count every caller request.  Deduplicated rows count each complete prompt
        // once before prefix sharing; physical rows count each exact trie edge once.
        long long logical_prompt_rows = 0;
        long long deduplicated_prompt_rows = 0;
        long long physical_prompt_rows = 0;
        long long logical_continuation_kv_rows = 0;
        // Resident continuation KV rows after exact duplicate requests have been collapsed.  The
        // first target comes from prompt-final logits, so N targets add only N-1 KV rows.
        long long continuation_kv_rows = 0;
        long long required_kv_rows = 0;

        int global_unique_prompt_index(int wave_local_index) const {
            if (wave_local_index < 0 ||
                wave_local_index >= static_cast<int>(unique_prompt_indexes.size()))
                throw std::out_of_range("variant-score wave-local prompt index out of range");
            return unique_prompt_indexes[static_cast<size_t>(wave_local_index)];
        }
    };

    int continuation_tokens = 0;
    std::vector<UniquePrompt> unique_prompts;
    std::vector<int> caller_to_unique;
    std::vector<Wave> waves;
};

namespace detail {

inline long long checked_add_variant_score_rows(long long a, long long b) {
    if (b > 0 && a > std::numeric_limits<long long>::max() - b)
        throw std::overflow_error("variant-score row accounting overflow");
    return a + b;
}

inline long long checked_mul_variant_score_rows(long long a, long long b) {
    if (a < 0 || b < 0)
        throw std::invalid_argument("variant-score row counts must be non-negative");
    if (a != 0 && b > std::numeric_limits<long long>::max() / a)
        throw std::overflow_error("variant-score row accounting overflow");
    return a * b;
}

inline VariantScorePlan::Wave make_variant_score_wave(
    const std::vector<VariantScorePlan::UniquePrompt>& unique_prompts,
    int begin, int end, int continuation_tokens, long long reserve_rows) {
    VariantScorePlan::Wave wave;
    std::vector<std::vector<int>> prompts;
    prompts.reserve(static_cast<size_t>(end - begin));
    for (int index = begin; index < end; ++index) {
        wave.unique_prompt_indexes.push_back(index);
        prompts.push_back(unique_prompts[static_cast<size_t>(index)].tokens);
        const auto& unique = unique_prompts[static_cast<size_t>(index)];
        const long long prompt_rows = static_cast<long long>(unique.tokens.size());
        const long long callers = static_cast<long long>(unique.caller_indexes.size());
        wave.deduplicated_prompt_rows = checked_add_variant_score_rows(
            wave.deduplicated_prompt_rows, prompt_rows);
        wave.logical_prompt_rows = checked_add_variant_score_rows(
            wave.logical_prompt_rows, checked_mul_variant_score_rows(prompt_rows, callers));
        const long long continuation_kv_rows = static_cast<long long>(continuation_tokens - 1);
        wave.logical_continuation_kv_rows = checked_add_variant_score_rows(
            wave.logical_continuation_kv_rows,
            checked_mul_variant_score_rows(continuation_kv_rows, callers));
    }
    wave.prompt_traversal = make_reference_match_traversal_plan(prompts);
    wave.physical_prompt_rows = reference_match_plan_physical_rows(wave.prompt_traversal);
    wave.continuation_kv_rows = checked_mul_variant_score_rows(
        static_cast<long long>(end - begin), static_cast<long long>(continuation_tokens - 1));
    wave.required_kv_rows = checked_add_variant_score_rows(
        checked_add_variant_score_rows(wave.physical_prompt_rows, wave.continuation_kv_rows),
        reserve_rows);
    return wave;
}

}  // namespace detail

// Construct deterministic resident waves.  Unique prompts are token-lexicographically ordered so
// prompts with common prefixes remain adjacent; each wave is the largest contiguous group that fits
// the declared sequence and KV budgets.  This greedy packing is deterministic, not claimed optimal.
//
// KV accounting is physical: the prompt component counts each exact trie edge once, while the
// recorded continuation's KV growth is charged once per distinct prompt history.  The prompt-final
// row scores continuation token 0; continuation tokens [0,N-2] score targets [1,N-1], so N targets
// add exactly N-1 KV rows.  Continuations are never shared after prompt divergence even when their
// token IDs match.  Duplicate complete prompts are collapsed before planning and mapped back to every
// original caller index.  This is intentionally leaner than today's scalar /score implementation,
// which decodes the unused final continuation row; the new executor must follow this plan explicitly.
inline VariantScorePlan make_variant_score_plan(
    const std::vector<std::vector<int>>& prompts,
    int continuation_tokens,
    const VariantScoreCapacity& capacity) {
    if (prompts.empty()) throw std::invalid_argument("variant scoring needs at least one prompt");
    if (continuation_tokens <= 0)
        throw std::invalid_argument("variant scoring needs a non-empty continuation");
    if (capacity.kv_rows <= 0)
        throw std::invalid_argument("variant scoring needs positive KV capacity");
    if (capacity.reserve_rows < 0 || capacity.reserve_rows >= capacity.kv_rows)
        throw std::invalid_argument("variant scoring reserve must be non-negative and below KV capacity");
    if (capacity.max_sequences <= 0)
        throw std::invalid_argument("variant scoring needs at least one sequence");
    if (capacity.max_sequence_tokens <= 0)
        throw std::invalid_argument("variant scoring needs a positive per-sequence limit");
    if (prompts.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("variant scoring has too many prompts");

    std::vector<int> order(prompts.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        const auto& left = prompts[static_cast<size_t>(a)];
        const auto& right = prompts[static_cast<size_t>(b)];
        if (left != right) return left < right;
        return a < b;
    });

    VariantScorePlan plan;
    plan.continuation_tokens = continuation_tokens;
    plan.caller_to_unique.assign(prompts.size(), -1);
    for (int caller_index : order) {
        const auto& prompt = prompts[static_cast<size_t>(caller_index)];
        if (prompt.empty())
            throw std::invalid_argument("variant scoring does not support an empty prompt");
        if (prompt.size() > static_cast<size_t>(std::numeric_limits<int>::max()))
            throw std::invalid_argument("variant scoring prompt is too long");
        if (std::any_of(prompt.begin(), prompt.end(), [](int token) { return token < 0; }))
            throw std::invalid_argument("variant scoring prompt token IDs must be non-negative");
        const long long sequence_tokens = detail::checked_add_variant_score_rows(
            static_cast<long long>(prompt.size()), static_cast<long long>(continuation_tokens));
        if (sequence_tokens > capacity.max_sequence_tokens)
            throw std::invalid_argument("variant scoring request exceeds the per-sequence limit");

        if (plan.unique_prompts.empty() || plan.unique_prompts.back().tokens != prompt) {
            plan.unique_prompts.push_back(VariantScorePlan::UniquePrompt{prompt, {}});
        }
        const int unique_index = static_cast<int>(plan.unique_prompts.size()) - 1;
        plan.unique_prompts.back().caller_indexes.push_back(caller_index);
        plan.caller_to_unique[static_cast<size_t>(caller_index)] = unique_index;
    }
    for (auto& unique : plan.unique_prompts)
        std::sort(unique.caller_indexes.begin(), unique.caller_indexes.end());

    const int unique_count = static_cast<int>(plan.unique_prompts.size());
    for (int begin = 0; begin < unique_count;) {
        int accepted_end = begin;
        VariantScorePlan::Wave accepted;
        const int end_limit = begin + std::min(capacity.max_sequences, unique_count - begin);
        for (int candidate_end = begin + 1; candidate_end <= end_limit; ++candidate_end) {
            auto candidate = detail::make_variant_score_wave(
                plan.unique_prompts, begin, candidate_end,
                continuation_tokens, capacity.reserve_rows);
            if (candidate.required_kv_rows > capacity.kv_rows) break;
            accepted_end = candidate_end;
            accepted = std::move(candidate);
        }
        if (accepted_end == begin)
            throw std::invalid_argument("one variant cannot fit within the declared KV capacity");
        plan.waves.push_back(std::move(accepted));
        begin = accepted_end;
    }
    return plan;
}

}  // namespace clozn
