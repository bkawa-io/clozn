// Model-free admission and capacity accounting for reusing a generation checkpoint while
// scoring related prompt variants.
//
// The runtime owns the opaque KV bytes.  This planner only proves that the checkpoint's declared
// token history is the requested baseline prompt and decides how much of that history may remain
// resident.  The last shared token is deliberately re-decoded when a variant terminates exactly at
// the shared boundary: serialized llama.cpp KV does not contain its logits row, and answer token 0
// must be scored from that row.
#pragma once

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace clozn {

struct VariantScoreGenerationState {
    std::vector<int> tokens;
    int n_past = 0;
    int prompt_tokens = 0;
    bool causal = true;
    bool has_steer = false;
};

struct VariantScoreCheckpointReusePlan {
    bool applied = false;
    std::string reason_code;
    int common_prefix_tokens = 0;
    int retained_prefix_tokens = 0;
    long long decoded_prompt_rows = 0;
    long long continuation_kv_rows = 0;
    long long required_kv_rows = 0;
};

namespace detail {

inline long long checked_checkpoint_reuse_add(long long left, long long right) {
    if (right > 0 && left > std::numeric_limits<long long>::max() - right)
        throw std::invalid_argument("checkpoint reuse row accounting overflow");
    return left + right;
}

inline int common_token_prefix(const std::vector<int>& left, const std::vector<int>& right,
                               int limit) {
    int length = 0;
    const size_t end = std::min(
        static_cast<size_t>(limit), std::min(left.size(), right.size()));
    while (static_cast<size_t>(length) < end &&
           left[static_cast<size_t>(length)] == right[static_cast<size_t>(length)])
        ++length;
    return length;
}

}  // namespace detail

// `wave_prompts` must already contain distinct histories and be bounded by the runtime's resident
// sequence limit. `baseline_prompt` identifies what the generation called its original prompt;
// exact equality with state.tokens[0,prompt_tokens) is required before any opaque KV is trusted.
//
// An unapplied plan is a safe, ordinary-prefill fallback decision, not a request error. Malformed
// capacity/input arguments remain programmer errors and throw.  A runtime that applies the plan:
//   1. copies the bounded checkpoint record out of its worker-local store;
//   2. restores it into one request-owned context lease;
//   3. evicts seq-0 positions [retained_prefix_tokens, inf);
//   4. copies only [0, retained_prefix_tokens) to the other wave sequences;
//   5. decodes every prompt's remaining suffix and then the shared continuation;
//   6. clears all request KV before releasing the lease.
inline VariantScoreCheckpointReusePlan make_variant_score_checkpoint_reuse_plan(
    const VariantScoreGenerationState& state,
    const std::vector<int>& baseline_prompt,
    const std::vector<std::vector<int>>& wave_prompts,
    int continuation_tokens,
    long long kv_rows,
    long long reserve_rows = 0) {
    if (wave_prompts.empty())
        throw std::invalid_argument("checkpoint reuse requires at least one wave prompt");
    if (continuation_tokens <= 0)
        throw std::invalid_argument("checkpoint reuse continuation must be non-empty");
    if (kv_rows <= 0 || reserve_rows < 0 || reserve_rows >= kv_rows)
        throw std::invalid_argument("checkpoint reuse capacity is invalid");
    for (const auto& prompt : wave_prompts)
        if (prompt.empty())
            throw std::invalid_argument("checkpoint reuse does not support an empty prompt");

    VariantScoreCheckpointReusePlan result;
    if (!state.causal) {
        result.reason_code = "checkpoint_not_causal";
        return result;
    }
    // A steered prompt KV is only valid if the identical steering remains armed for every suffix
    // and continuation decode. The v1 scorer has no intervention field, so it must not silently mix
    // steered prefix state with unsteered new rows.
    if (state.has_steer) {
        result.reason_code = "checkpoint_has_steer";
        return result;
    }
    if (state.prompt_tokens <= 0 || state.n_past < state.prompt_tokens || state.n_past < 0 ||
        static_cast<size_t>(state.n_past) > state.tokens.size()) {
        result.reason_code = "checkpoint_history_invalid";
        return result;
    }
    if (baseline_prompt.size() != static_cast<size_t>(state.prompt_tokens) ||
        !std::equal(baseline_prompt.begin(), baseline_prompt.end(), state.tokens.begin())) {
        result.reason_code = "checkpoint_baseline_mismatch";
        return result;
    }

    int common = state.prompt_tokens;
    for (const auto& prompt : wave_prompts)
        common = detail::common_token_prefix(state.tokens, prompt, common);
    result.common_prefix_tokens = common;
    if (common == 0) {
        result.reason_code = "checkpoint_no_shared_prefix";
        return result;
    }
    // Generation KV was produced under the original prompt's decode shape. Splicing a shallow
    // fragment into a much larger freshly decoded suffix is both a weak performance trade and a
    // distinct numerical regime. Keep v1 deliberately conservative: generation-state reuse is a
    // late-divergence optimization only. Earlier counterfactuals use the request-local trie path.
    if (static_cast<long long>(common) * 4LL <
        static_cast<long long>(state.prompt_tokens) * 3LL) {
        result.reason_code = "checkpoint_shared_prefix_too_short";
        return result;
    }

    // If one prompt ends at the common boundary, its final resident KV row has no serialized logits.
    // Rewind that one row so every prompt has at least one suffix token whose decode emits the row
    // needed to score continuation token 0. At a divergence boundary all prompts already have a
    // suffix, so the entire common prefix can remain resident.
    const bool terminal_at_common = std::any_of(
        wave_prompts.begin(), wave_prompts.end(),
        [common](const std::vector<int>& prompt) {
            return static_cast<int>(prompt.size()) == common;
        });
    result.retained_prefix_tokens = common - (terminal_at_common ? 1 : 0);
    if (result.retained_prefix_tokens <= 0) {
        result.reason_code = "checkpoint_reuse_too_shallow";
        return result;
    }

    long long decoded_prompt_rows = 0;
    for (const auto& prompt : wave_prompts) {
        const long long suffix =
            static_cast<long long>(prompt.size()) - result.retained_prefix_tokens;
        decoded_prompt_rows = detail::checked_checkpoint_reuse_add(decoded_prompt_rows, suffix);
    }
    const long long continuation_rows_per_prompt = continuation_tokens - 1LL;
    long long continuation_rows = 0;
    for (size_t i = 0; i < wave_prompts.size(); ++i)
        continuation_rows = detail::checked_checkpoint_reuse_add(
            continuation_rows, continuation_rows_per_prompt);

    long long required = reserve_rows;
    // llama_memory_seq_cp associates the retained cells with more sequence IDs; it does not clone
    // one physical KV row per branch. Divergent suffix and continuation rows are independent.
    required = detail::checked_checkpoint_reuse_add(required, result.retained_prefix_tokens);
    required = detail::checked_checkpoint_reuse_add(required, decoded_prompt_rows);
    required = detail::checked_checkpoint_reuse_add(required, continuation_rows);
    result.decoded_prompt_rows = decoded_prompt_rows;
    result.continuation_kv_rows = continuation_rows;
    result.required_kv_rows = required;
    if (required > kv_rows) {
        result.reason_code = "checkpoint_reuse_capacity_exceeded";
        return result;
    }

    result.applied = true;
    result.reason_code = "applied";
    return result;
}

}  // namespace clozn
