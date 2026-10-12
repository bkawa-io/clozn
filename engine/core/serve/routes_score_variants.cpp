// Native related-prompt scoring: exact-token prefix reuse plus multi-sequence continuation decode.
//
// This endpoint is deliberately screening-only. It returns only the recorded token logprobs and,
// when the output backend supports llama.cpp's sampler graph, reduces each full-vocabulary row to
// one target probability before device-to-host transfer. Scalar /score remains the evidence oracle
// until this distinct batch shape and backend reduction are qualified.
#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "clozn/score_variants_protocol.hpp"
#include "clozn/variant_score_checkpoint_reuse.hpp"
#include "clozn/variant_score_plan.hpp"
#include "server_context.hpp"
#include "variant_score_checkpoint_executor.hpp"

namespace clozn {

namespace {

constexpr const char* kScoreVariantsAlgorithmId =
    "clozn.score-variants.native-prefix-trie-step-batch-target-reduce-generation-reuse.v3";
constexpr const char* kScoreVariantsStrategy = "native_prefix_reuse_step_batch";
constexpr const char* kScoreVariantsGenerationReuseStrategy =
    "native_generation_prefix_reuse_step_batch";
constexpr const char* kScoreVariantsMixedReuseStrategy =
    "native_generation_prefix_reuse_step_batch_mixed";

ScoreVariantsParsedInteger parsed_integer(const json& value) {
    if (!value.is_number_integer() && !value.is_number_unsigned())
        return ScoreVariantsParsedInteger::non_integer();
    try {
        return ScoreVariantsParsedInteger::integer(value.get<std::int64_t>());
    } catch (...) {
        return ScoreVariantsParsedInteger::non_integer();
    }
}

class TargetLogprobReductionScope {
public:
    TargetLogprobReductionScope(GgmlAdapter& adapter, int n_sequences)
        : adapter_(adapter) {
        adapter_.begin_target_logprob_reduction(n_sequences);
    }
    ~TargetLogprobReductionScope() { adapter_.end_target_logprob_reduction(); }
    TargetLogprobReductionScope(const TargetLogprobReductionScope&) = delete;
    TargetLogprobReductionScope& operator=(const TargetLogprobReductionScope&) = delete;

private:
    GgmlAdapter& adapter_;
};

std::vector<int> wave_caller_indexes(
    const VariantScorePlan& plan, const VariantScorePlan::Wave& wave) {
    std::vector<int> indexes;
    for (const int unique_index : wave.unique_prompt_indexes) {
        const auto& callers = plan.unique_prompts[static_cast<size_t>(unique_index)].caller_indexes;
        indexes.insert(indexes.end(), callers.begin(), callers.end());
    }
    std::sort(indexes.begin(), indexes.end());
    return indexes;
}

bool lower_sha256(const std::string& value) {
    if (value.size() != 64) return false;
    return std::all_of(value.begin(), value.end(), [](const char character) {
        return (character >= '0' && character <= '9') ||
               (character >= 'a' && character <= 'f');
    });
}

json execution_identity_json(const ServerContext& ctx) {
    json identity = {
        {"algorithm_id", kScoreVariantsAlgorithmId},
        {"worker_generation_id", ctx.worker_generation_id},
    };
    // Direct developer-launched workers may not have supervisor-measured hashes. The screen stays
    // usable and unqualified there; when hashes exist, carry them as stronger identity dimensions.
    if (lower_sha256(ctx.model_sha256)) identity["model_sha256"] = ctx.model_sha256;
    if (lower_sha256(ctx.tokenizer_sha256)) identity["tokenizer_sha256"] = ctx.tokenizer_sha256;
    return identity;
}

std::string execution_identity_string(const ServerContext& ctx) {
    return std::string(kScoreVariantsAlgorithmId) + ":" + ctx.worker_generation_id + ":" +
           ctx.model_sha256 + ":" + ctx.tokenizer_sha256;
}

}  // namespace

void register_score_variants_routes(httplib::Server& svr, ServerContext& ctx) {
    svr.Post(std::string(score_variants_contract::ENDPOINT),
             [&](const httplib::Request& req, httplib::Response& res) {
        auto fail = [&](int status, const std::string& code, const std::string& message) {
            res.status = status;
            res.set_content(json{{"error", message}, {"code", code},
                                 {"qualified", score_variants_contract::QUALIFIED},
                                 {"evidence_use",
                                  score_variants_contract::EVIDENCE_USE.data()}}.dump(),
                            "application/json");
        };

        if (!ctx.ar_mode) {
            fail(409, "unsupported_model_mode",
                 "score-variants requires an autoregressive model");
            return;
        }
        const json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            fail(400, "request_invalid", "invalid JSON body");
            return;
        }
        for (const auto& item : body.items()) {
            if (!score_variants_request_key_allowed(item.key())) {
                fail(400, "request_invalid", "unknown top-level field: " + item.key());
                return;
            }
        }
        if (!body.contains("variants") || !body["variants"].is_array() ||
            !body.contains("continuation_ids") || !body["continuation_ids"].is_array()) {
            fail(400, "request_invalid",
                 "request requires exactly variants and continuation_ids arrays");
            return;
        }

        ScoreVariantsRequest parsed;
        parsed.variants.reserve(body["variants"].size());
        for (size_t index = 0; index < body["variants"].size(); ++index) {
            const auto& item = body["variants"][index];
            if (!item.is_object()) {
                fail(400, "request_invalid",
                     "variants[" + std::to_string(index) + "] must be an object");
                return;
            }
            for (const auto& field : item.items()) {
                if (!score_variant_request_row_key_allowed(field.key())) {
                    fail(400, "request_invalid",
                         "unknown variants[" + std::to_string(index) + "] field: " +
                             field.key());
                    return;
                }
            }
            if (!item.contains("id") || !item["id"].is_string() ||
                !item.contains("prompt_ids") || !item["prompt_ids"].is_array()) {
                fail(400, "request_invalid",
                     "each variant requires exactly string id and prompt_ids array");
                return;
            }
            ScoreVariantRequestRow row;
            row.id = item["id"].get<std::string>();
            row.prompt_token_ids.reserve(item["prompt_ids"].size());
            for (const auto& value : item["prompt_ids"])
                row.prompt_token_ids.push_back(parsed_integer(value));
            parsed.variants.push_back(std::move(row));
        }
        parsed.continuation_token_ids.reserve(body["continuation_ids"].size());
        for (const auto& value : body["continuation_ids"])
            parsed.continuation_token_ids.push_back(parsed_integer(value));
        if (body.contains("generation_state")) {
            const auto& state = body["generation_state"];
            if (!state.is_object()) {
                fail(400, "request_invalid", "generation_state must be an object");
                return;
            }
            for (const auto& field : state.items()) {
                if (!score_variants_generation_state_key_allowed(field.key())) {
                    fail(400, "request_invalid",
                         "unknown generation_state field: " + field.key());
                    return;
                }
            }
            if (state.size() != 3 || !state.contains("checkpoint_id") ||
                !state["checkpoint_id"].is_string() ||
                !state.contains("worker_generation_id") ||
                !state["worker_generation_id"].is_string() ||
                !state.contains("baseline_variant_id") ||
                !state["baseline_variant_id"].is_string()) {
                fail(400, "request_invalid",
                     "generation_state requires exactly checkpoint_id, worker_generation_id, "
                     "and baseline_variant_id strings");
                return;
            }
            parsed.has_generation_state = true;
            parsed.generation_state = {
                state["checkpoint_id"].get<std::string>(),
                state["worker_generation_id"].get<std::string>(),
                state["baseline_variant_id"].get<std::string>(),
            };
        }

        const ScoreVariantsRequestValidation validated = validate_score_variants_request(
            parsed, ctx.model->config().vocab_size, ctx.n_ctx);
        if (!validated.ok()) {
            fail(400, score_variants_protocol_code(validated.code), validated.message);
            return;
        }

        VariantScorePlan plan;
        try {
            plan = make_variant_score_plan(
                validated.prompt_token_ids,
                static_cast<int>(validated.continuation_token_ids.size()),
                VariantScoreCapacity{
                    static_cast<long long>(ctx.n_ctx),
                    0,
                    score_variants_contract::MAX_PROMPT_ROWS_PER_CONTEXT,
                    ctx.n_ctx,
                });
        } catch (const std::exception& error) {
            fail(400, "request_limit_exceeded",
                 std::string("score-variants cannot schedule request: ") + error.what());
            return;
        }

        std::optional<EngineCheckpoint> generation_checkpoint;
        std::string generation_state_reason = "not_requested";
        int baseline_caller_index = -1;
        if (parsed.has_generation_state) {
            for (size_t index = 0; index < validated.variant_ids.size(); ++index) {
                if (validated.variant_ids[index] ==
                    parsed.generation_state.baseline_variant_id) {
                    baseline_caller_index = static_cast<int>(index);
                    break;
                }
            }
            if (parsed.generation_state.worker_generation_id != ctx.worker_generation_id) {
                generation_state_reason = "worker_generation_mismatch";
            } else if (!ctx.find_checkpoint) {
                generation_state_reason = "checkpoint_lookup_unavailable";
            } else {
                generation_checkpoint =
                    ctx.find_checkpoint(parsed.generation_state.checkpoint_id);
                generation_state_reason = generation_checkpoint.has_value()
                    ? "candidate"
                    : "checkpoint_unavailable";
            }
        }

        std::vector<VariantScoreCheckpointReusePlan> checkpoint_reuse_plans(plan.waves.size());
        int checkpoint_reuse_wave_count = 0;
        if (generation_checkpoint.has_value()) {
            const VariantScoreGenerationState generation_state{
                generation_checkpoint->tokens,
                generation_checkpoint->n_past,
                generation_checkpoint->prompt_tokens,
                generation_checkpoint->causal,
                generation_checkpoint->has_steer,
            };
            const auto& baseline_prompt = validated.prompt_token_ids[
                static_cast<size_t>(baseline_caller_index)];
            for (size_t wave_index = 0; wave_index < plan.waves.size(); ++wave_index) {
                const auto& wave = plan.waves[wave_index];
                std::vector<std::vector<int>> wave_prompts;
                wave_prompts.reserve(wave.unique_prompt_indexes.size());
                for (const int unique_index : wave.unique_prompt_indexes)
                    wave_prompts.push_back(
                        plan.unique_prompts[static_cast<size_t>(unique_index)].tokens);
                checkpoint_reuse_plans[wave_index] =
                    make_variant_score_checkpoint_reuse_plan(
                        generation_state, baseline_prompt, wave_prompts,
                        static_cast<int>(validated.continuation_token_ids.size()),
                        static_cast<long long>(ctx.n_ctx));
                if (checkpoint_reuse_plans[wave_index].applied)
                    ++checkpoint_reuse_wave_count;
            }
            if (checkpoint_reuse_wave_count == 0 && !checkpoint_reuse_plans.empty())
                generation_state_reason = checkpoint_reuse_plans.front().reason_code;
            else if (checkpoint_reuse_wave_count == static_cast<int>(plan.waves.size()))
                generation_state_reason = "applied";
            else
                generation_state_reason = "partially_applied";
        } else if (parsed.has_generation_state) {
            for (auto& reuse : checkpoint_reuse_plans)
                reuse.reason_code = generation_state_reason;
        }

        const char* execution_strategy = kScoreVariantsStrategy;
        if (checkpoint_reuse_wave_count == static_cast<int>(plan.waves.size()) &&
            checkpoint_reuse_wave_count > 0) {
            execution_strategy = kScoreVariantsGenerationReuseStrategy;
        } else if (checkpoint_reuse_wave_count > 0) {
            execution_strategy = kScoreVariantsMixedReuseStrategy;
        }

        ScoreVariantsResponse protocol_response;
        protocol_response.status = score_variants_contract::TOP_STATUSES[2];
        protocol_response.continuation_token_ids = validated.continuation_token_ids;
        protocol_response.execution = {
            execution_strategy,
            score_variants_contract::NUMERICAL_REGIME.data(),
            score_variants_contract::EVIDENCE_USE.data(),
            score_variants_contract::QUALIFIED,
            execution_identity_string(ctx),
            {},
        };
        protocol_response.results.reserve(validated.variant_ids.size());
        for (size_t index = 0; index < validated.variant_ids.size(); ++index) {
            protocol_response.results.push_back({
                validated.variant_ids[index],
                SCORE_VARIANT_STATUS_NOT_STARTED,
                static_cast<int>(validated.prompt_token_ids[index].size()),
                {},
                "not_started",
            });
        }

        std::vector<std::vector<double>> scores_by_unique(plan.unique_prompts.size());
        std::vector<std::string> wave_states(plan.waves.size(), "not_started");
        long long decoded_rows = 0;
        long long decode_calls = 0;
        std::int64_t prefill_ns = 0;
        std::int64_t continuation_ns = 0;
        bool observed_logprob_output = false;
        bool full_vocab_host_reduction = false;
        long long logprob_host_output_floats = 0;
        size_t executing_wave = 0;
        std::string runtime_error;
        const auto wall_started = std::chrono::steady_clock::now();
        try {
            ContextPool::Lease lease = ctx.pool.acquire();
            GgmlAdapter& adapter = *lease;
            const long long decoded_before = adapter.decoded_tokens();
            adapter.set_causal(true);
            try {
                for (; executing_wave < plan.waves.size(); ++executing_wave) {
                    const auto& wave = plan.waves[executing_wave];
                    const auto& checkpoint_reuse =
                        checkpoint_reuse_plans[executing_wave];
                    const long long wave_decoded_before = adapter.decoded_tokens();
                    std::vector<std::vector<int>> prompts;
                    prompts.reserve(wave.unique_prompt_indexes.size());
                    for (const int unique_index : wave.unique_prompt_indexes)
                        prompts.push_back(plan.unique_prompts[static_cast<size_t>(unique_index)].tokens);
                    auto reduction_scope = std::make_unique<TargetLogprobReductionScope>(
                        adapter, static_cast<int>(prompts.size()));

                    const auto prefill_started = std::chrono::steady_clock::now();
                    TargetLogprobBatchResult target_scores;
                    try {
                        if (checkpoint_reuse.applied) {
                            const auto restored = prefill_variant_prompts_from_checkpoint(
                                adapter, *generation_checkpoint, prompts, checkpoint_reuse,
                                validated.continuation_token_ids[0]);
                            decode_calls += restored.decode_calls;
                            target_scores.target_logprobs = restored.target_logprobs;
                            target_scores.full_vocab_host_reduction =
                                restored.full_vocab_host_reduction;
                            target_scores.host_output_floats = restored.host_output_floats;
                        } else {
                            target_scores =
                                adapter.ar_forward_prefill_batch_target_logprobs(
                                    prompts, validated.continuation_token_ids[0]);
                            decode_calls += adapter.last_decode_call_count();
                        }
                    } catch (...) {
                        prefill_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - prefill_started).count();
                        throw;
                    }
                    prefill_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - prefill_started).count();
                    if (!checkpoint_reuse.applied &&
                        adapter.last_prefill_physical_rows() != wave.physical_prompt_rows)
                        throw std::runtime_error(
                            "planner/runtime physical prompt row accounting disagreed");
                    observed_logprob_output = true;
                    full_vocab_host_reduction |= target_scores.full_vocab_host_reduction;
                    logprob_host_output_floats += target_scores.host_output_floats;

                    for (size_t local = 0; local < target_scores.target_logprobs.size(); ++local) {
                        const int unique_index = wave.global_unique_prompt_index(
                            static_cast<int>(local));
                        auto& scores = scores_by_unique[static_cast<size_t>(unique_index)];
                        scores.reserve(validated.continuation_token_ids.size());
                        scores.push_back(target_scores.target_logprobs[local]);
                    }

                    std::vector<int> tokens(prompts.size());
                    std::vector<int> positions(prompts.size());
                    const std::vector<bool> active(prompts.size(), true);
                    for (size_t step = 0;
                         step + 1 < validated.continuation_token_ids.size(); ++step) {
                        for (size_t local = 0; local < prompts.size(); ++local) {
                            tokens[local] = validated.continuation_token_ids[step];
                            positions[local] = static_cast<int>(prompts[local].size() + step);
                        }
                        const auto continuation_started = std::chrono::steady_clock::now();
                        try {
                            target_scores = adapter.ar_forward_batch_at_target_logprobs(
                                tokens, positions, active,
                                validated.continuation_token_ids[step + 1]);
                        } catch (...) {
                            continuation_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - continuation_started).count();
                            decode_calls += adapter.last_decode_call_count();
                            throw;
                        }
                        continuation_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - continuation_started).count();
                        decode_calls += adapter.last_decode_call_count();
                        observed_logprob_output = true;
                        full_vocab_host_reduction |= target_scores.full_vocab_host_reduction;
                        logprob_host_output_floats += target_scores.host_output_floats;
                        for (size_t local = 0;
                             local < target_scores.target_logprobs.size(); ++local) {
                            const int unique_index = wave.global_unique_prompt_index(
                                static_cast<int>(local));
                            scores_by_unique[static_cast<size_t>(unique_index)].push_back(
                                target_scores.target_logprobs[local]);
                        }
                    }
                    const long long measured_wave_rows =
                        adapter.decoded_tokens() - wave_decoded_before;
                    const long long expected_wave_rows = checkpoint_reuse.applied
                        ? checkpoint_reuse.decoded_prompt_rows + wave.continuation_kv_rows
                        : wave.physical_prompt_rows + wave.continuation_kv_rows;
                    if (measured_wave_rows != expected_wave_rows) {
                        throw std::runtime_error(
                            "planner/runtime total decode row accounting disagreed");
                    }

                    for (const int caller_index : wave_caller_indexes(plan, wave)) {
                        auto& row = protocol_response.results[static_cast<size_t>(caller_index)];
                        row.target_logprobs = scores_by_unique[static_cast<size_t>(
                            plan.caller_to_unique[static_cast<size_t>(caller_index)])];
                        row.status = SCORE_VARIANT_STATUS_COMPLETED;
                        row.reason_code.clear();
                    }
                    wave_states[executing_wave] = "completed";
                }
                adapter.reset_ar_kv();
            } catch (...) {
                decoded_rows = adapter.decoded_tokens() - decoded_before;
                adapter.reset_ar_kv();
                throw;
            }
            decoded_rows = adapter.decoded_tokens() - decoded_before;
        } catch (const std::exception& error) {
            runtime_error = error.what();
            if (executing_wave < plan.waves.size()) {
                wave_states[executing_wave] = "failed";
                for (const int caller_index : wave_caller_indexes(plan, plan.waves[executing_wave])) {
                    auto& row = protocol_response.results[static_cast<size_t>(caller_index)];
                    row.status = SCORE_VARIANT_STATUS_FAILED;
                    row.reason_code = "native_execution_failed";
                    row.target_logprobs.clear();
                }
            }
        }
        const std::int64_t wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_started).count();

        int completed_count = 0;
        for (const auto& row : protocol_response.results)
            if (row.status == SCORE_VARIANT_STATUS_COMPLETED) ++completed_count;
        protocol_response.status =
            completed_count == static_cast<int>(protocol_response.results.size())
                ? "completed"
                : (completed_count > 0 ? "partial" : "failed");
        for (size_t wave_index = 0; wave_index < plan.waves.size(); ++wave_index) {
            const auto& wave = plan.waves[wave_index];
            ScoreVariantsExecutionMetadata::Wave metadata;
            metadata.wave_index = static_cast<int>(wave_index);
            for (const int caller_index : wave_caller_indexes(plan, wave))
                metadata.variant_ids.push_back(
                    validated.variant_ids[static_cast<size_t>(caller_index)]);
            metadata.unique_prompt_count = static_cast<int>(wave.unique_prompt_indexes.size());
            metadata.logical_prompt_rows = wave.logical_prompt_rows;
            metadata.physical_prompt_rows = wave.physical_prompt_rows;
            metadata.continuation_kv_rows = wave.continuation_kv_rows;
            metadata.required_kv_rows = wave.required_kv_rows;
            protocol_response.execution.waves.push_back(std::move(metadata));
        }

        const ScoreVariantsValidation response_validation =
            validate_score_variants_response(protocol_response, validated);
        if (!response_validation.ok()) {
            fail(500, score_variants_protocol_code(response_validation.code),
                 "internal score-variants response validation failed: " +
                     response_validation.message);
            return;
        }

        int completed = 0;
        json rows = json::array();
        for (const auto& row : protocol_response.results) {
            json item = {
                {"id", row.id},
                {"status", row.status},
                {"prompt_token_count", row.prompt_token_count},
            };
            if (row.status == SCORE_VARIANT_STATUS_COMPLETED) {
                ++completed;
                item["target_logprobs"] = row.target_logprobs;
            } else {
                item["reason_code"] = row.reason_code;
                if (row.status == SCORE_VARIANT_STATUS_FAILED)
                    item["error"] = runtime_error.empty() ? "native execution failed" : runtime_error;
            }
            rows.push_back(std::move(item));
        }
        const std::string& top_status = protocol_response.status;
        // No completed backend output means no evidence that target-only transfer actually ran.
        // Report the conservative full-vocabulary regime in that case.
        const bool used_full_vocab_host_reduction =
            !observed_logprob_output || full_vocab_host_reduction;

        json waves = json::array();
        for (size_t wave_index = 0; wave_index < plan.waves.size(); ++wave_index) {
            const auto& wave = plan.waves[wave_index];
            json variant_ids = json::array();
            for (const int caller_index : wave_caller_indexes(plan, wave))
                variant_ids.push_back(validated.variant_ids[static_cast<size_t>(caller_index)]);
            waves.push_back({
                {"wave_index", wave_index},
                {"status", wave_states[wave_index]},
                {"accounting", "planned_capacity"},
                {"variant_ids", std::move(variant_ids)},
                {"unique_prompt_count", wave.unique_prompt_indexes.size()},
                {"logical_prompt_rows", wave.logical_prompt_rows},
                {"deduplicated_prompt_rows", wave.deduplicated_prompt_rows},
                {"physical_prompt_rows", wave.physical_prompt_rows},
                {"logical_continuation_kv_rows", wave.logical_continuation_kv_rows},
                {"continuation_kv_rows", wave.continuation_kv_rows},
                {"required_kv_rows", wave.required_kv_rows},
                {"prefill_strategy", checkpoint_reuse_plans[wave_index].applied
                    ? "generation_checkpoint_prefix"
                    : "request_prefix_trie"},
                {"generation_state_reuse", {
                    {"applied", checkpoint_reuse_plans[wave_index].applied},
                    {"reason_code", checkpoint_reuse_plans[wave_index].reason_code},
                    {"common_prefix_tokens",
                     checkpoint_reuse_plans[wave_index].common_prefix_tokens},
                    {"retained_prefix_tokens",
                     checkpoint_reuse_plans[wave_index].retained_prefix_tokens},
                    {"decoded_prompt_rows",
                     checkpoint_reuse_plans[wave_index].decoded_prompt_rows},
                    {"required_kv_rows",
                     checkpoint_reuse_plans[wave_index].required_kv_rows},
                }},
            });
        }

        long long logical_prompt_rows = 0;
        long long deduplicated_prompt_rows = 0;
        long long physical_prompt_rows = 0;
        long long continuation_kv_rows = 0;
        for (const auto& wave : plan.waves) {
            logical_prompt_rows += wave.logical_prompt_rows;
            deduplicated_prompt_rows += wave.deduplicated_prompt_rows;
            physical_prompt_rows += wave.physical_prompt_rows;
            continuation_kv_rows += wave.continuation_kv_rows;
        }

        json response = {
            {"schema_version", score_variants_contract::SCHEMA_VERSION.data()},
            {"status", top_status},
            {"continuation_ids", validated.continuation_token_ids},
            {"results", std::move(rows)},
            {"execution", {
                {"strategy", execution_strategy},
                {"numerical_regime", score_variants_contract::NUMERICAL_REGIME.data()},
                {"evidence_use", score_variants_contract::EVIDENCE_USE.data()},
                {"qualified", score_variants_contract::QUALIFIED},
                {"execution_identity", execution_identity_json(ctx)},
                {"waves", std::move(waves)},
                {"full_vocab_host_reduction", used_full_vocab_host_reduction},
                {"target_logprob_reduction",
                 used_full_vocab_host_reduction
                     ? "host_log_softmax"
                     : "backend_softmax_target_gather"},
                {"generation_state_reuse", {
                    {"requested", parsed.has_generation_state},
                    {"status", generation_state_reason},
                    {"baseline_variant_id", parsed.has_generation_state
                        ? parsed.generation_state.baseline_variant_id
                        : std::string()},
                    {"applied_wave_count", checkpoint_reuse_wave_count},
                    {"checkpoint_lifecycle", "worker_fifo_borrowed_copy"},
                }},
            }},
            {"metrics", {
                {"variant_count", validated.variant_ids.size()},
                {"unique_prompt_count", plan.unique_prompts.size()},
                {"duplicate_variant_count",
                 validated.variant_ids.size() - plan.unique_prompts.size()},
                {"wave_count", plan.waves.size()},
                {"planned_logical_prompt_rows", logical_prompt_rows},
                {"planned_deduplicated_prompt_rows", deduplicated_prompt_rows},
                {"planned_physical_prompt_rows", physical_prompt_rows},
                {"planned_continuation_kv_rows", continuation_kv_rows},
                {"successful_decode_rows", decoded_rows},
                {"successful_decode_calls", decode_calls},
                {"logprob_host_output_floats", logprob_host_output_floats},
                {"prefill_seconds", static_cast<double>(prefill_ns) / 1.0e9},
                {"continuation_seconds", static_cast<double>(continuation_ns) / 1.0e9},
                {"wall_seconds", static_cast<double>(wall_ns) / 1.0e9},
            }},
        };
        res.set_content(response.dump(), "application/json");
    });
}

}  // namespace clozn
