// Request-scoped original-generation KV reuse for /v1/score-variants.
//
// This is intentionally a server helper over existing narrow GgmlAdapter sequence primitives, not
// a persistent cache API. The caller owns one ordinary context lease. On success the returned
// terminal logits and divergent sequence KVs remain valid for immediate continuation scoring; on
// failure this helper clears every restored/copied sequence before propagating the exception.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "clozn/model_ggml.hpp"
#include "clozn/variant_score_checkpoint_reuse.hpp"

namespace clozn {

struct VariantScoreCheckpointPrefillResult {
    std::vector<double> target_logprobs;
    bool full_vocab_host_reduction = false;
    long long host_output_floats = 0;
    long long decode_calls = 0;
};

inline VariantScoreCheckpointPrefillResult prefill_variant_prompts_from_checkpoint(
    GgmlAdapter& adapter,
    const EngineCheckpoint& checkpoint,
    const std::vector<std::vector<int>>& prompts,
    const VariantScoreCheckpointReusePlan& plan,
    int target_token) {
    if (!plan.applied || plan.retained_prefix_tokens <= 0)
        throw std::invalid_argument(
            "checkpoint variant prefill requires an applied reuse plan");
    if (prompts.empty() || prompts.size() > 16)
        throw std::invalid_argument(
            "checkpoint variant prefill requires 1..16 prompts");
    if (!checkpoint.causal || checkpoint.has_steer)
        throw std::invalid_argument(
            "checkpoint variant prefill requires unsteered causal state");

    try {
        adapter.load_checkpoint(checkpoint);  // clears prior request state before restoring seq 0
        adapter.evict_ar_seq_from(0, plan.retained_prefix_tokens);
        if (adapter.ar_seq_size(0) != plan.retained_prefix_tokens)
            throw std::runtime_error(
                "checkpoint variant prefill retained prefix size mismatch");
        for (size_t sequence = 1; sequence < prompts.size(); ++sequence)
            adapter.copy_ar_seq(0, static_cast<int>(sequence), 0,
                                plan.retained_prefix_tokens);

        VariantScoreCheckpointPrefillResult result;
        result.target_logprobs.reserve(prompts.size());
        long long decoded_rows = 0;
        for (size_t sequence = 0; sequence < prompts.size(); ++sequence) {
            const auto& prompt = prompts[sequence];
            if (static_cast<int>(prompt.size()) <= plan.retained_prefix_tokens)
                throw std::runtime_error(
                    "checkpoint variant prefill plan left no terminal suffix row");
            const auto reduced = adapter.ar_forward_seq_segment_target_logprob(
                prompt, plan.retained_prefix_tokens, static_cast<int>(prompt.size()),
                static_cast<int>(sequence), target_token);
            if (reduced.target_logprobs.size() != 1)
                throw std::runtime_error(
                    "checkpoint variant prefill returned the wrong target score count");
            result.target_logprobs.push_back(reduced.target_logprobs[0]);
            result.full_vocab_host_reduction |= reduced.full_vocab_host_reduction;
            result.host_output_floats += reduced.host_output_floats;
            result.decode_calls += adapter.last_decode_call_count();
            decoded_rows += static_cast<long long>(prompt.size()) -
                            plan.retained_prefix_tokens;
        }
        if (decoded_rows != plan.decoded_prompt_rows)
            throw std::runtime_error(
                "checkpoint variant prefill decoded-row accounting mismatch");
        return result;
    } catch (...) {
        adapter.reset_ar_kv();
        throw;
    }
}

}  // namespace clozn
