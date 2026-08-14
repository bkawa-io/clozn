// Private experiment route for native exact recorded-reference multi-arm probes.
// It is deliberately separate from /v1/completions: proof arms are detached
// measurements, never chat requests and never journaled Runs.
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "server_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace clozn {

using json = nlohmann::json;

void register_reference_match_routes(httplib::Server& svr, ServerContext& ctx) {
    svr.Post("/v1/reference-match/arms", [&](const httplib::Request& req,
                                             httplib::Response& res) {
        const json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        auto fail = [&](const std::string& message) {
            res.status = 400;
            res.set_content(json{{"error", message}}.dump(), "application/json");
        };
        if (body.is_discarded() || !body.is_object()) {
            fail("invalid JSON body");
            return;
        }
        if (!ctx.ar_mode) {
            res.status = 409;
            res.set_content(json{{"error", "reference-match arms require autoregressive mode"}}.dump(),
                            "application/json");
            return;
        }
        if (!body.contains("arms") || !body["arms"].is_array() || body["arms"].empty()) {
            fail("arms must be a non-empty array");
            return;
        }
        if (body["arms"].size() > 1276) {
            fail("reference-match arms are bounded to 1276 per request");
            return;
        }
        if (!body.contains("reference_token_ids") || !body["reference_token_ids"].is_array() ||
            body["reference_token_ids"].empty()) {
            fail("reference_token_ids must be a non-empty integer array");
            return;
        }
        std::vector<int> reference;
        try {
            reference = body["reference_token_ids"].get<std::vector<int>>();
        } catch (...) {
            fail("reference_token_ids must contain integers");
            return;
        }
        if (!body.contains("generation_contract") || !body["generation_contract"].is_object()) {
            fail("generation_contract must be an object");
            return;
        }
        const json contract = body["generation_contract"];
        if (contract.value("decode_mode", std::string()) != "greedy") {
            res.status = 409;
            res.set_content(json{{"error", "native reference-match currently supports greedy only"},
                                 {"reason", "sampled_replay_not_proven"}}.dump(),
                            "application/json");
            return;
        }
        const int max_tokens = contract.value("max_new", 0);
        if (max_tokens < 1) {
            fail("generation_contract.max_new must be a positive integer");
            return;
        }
        std::vector<std::string> stops;
        if (contract.contains("stop")) {
            if (!contract["stop"].is_array()) {
                fail("generation_contract.stop must be an array");
                return;
            }
            for (const auto& value : contract["stop"]) {
                if (!value.is_string()) {
                    fail("generation_contract.stop must contain strings");
                    return;
                }
                stops.push_back(value.get<std::string>());
            }
        }

        std::vector<int> arm_ids;
        std::vector<std::vector<int>> prompt_ids;
        arm_ids.reserve(body["arms"].size());
        prompt_ids.reserve(body["arms"].size());
        std::map<int, bool> seen_ids;
        std::int64_t tokenize_ns = 0;
        for (const auto& arm : body["arms"]) {
            if (!arm.is_object() || !arm.contains("arm_id") || !arm["arm_id"].is_number_integer() ||
                !arm.contains("prompt") || !arm["prompt"].is_string()) {
                fail("each arm requires integer arm_id and string prompt");
                return;
            }
            const int arm_id = arm["arm_id"].get<int>();
            if (seen_ids.find(arm_id) != seen_ids.end()) {
                fail("arm_id values must be unique");
                return;
            }
            seen_ids[arm_id] = true;
            const auto started = std::chrono::steady_clock::now();
            std::vector<int> ids;
            try {
                ids = ctx.model->encode(arm["prompt"].get<std::string>());
            } catch (const std::exception& e) {
                fail(std::string("prompt tokenization failed: ") + e.what());
                return;
            }
            tokenize_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
            if (ids.empty()) {
                fail("arm prompt must tokenize to at least one token");
                return;
            }
            if (static_cast<int>(ids.size()) > ctx.n_ctx) {
                fail("arm prompt exceeds the worker context window");
                return;
            }
            arm_ids.push_back(arm_id);
            prompt_ids.push_back(std::move(ids));
        }

        const auto wall_started = std::chrono::steady_clock::now();
        ReferenceMatchBatchMetrics total_metrics;
        total_metrics.peak_resident_sequences = 0;
        total_metrics.prompt_tokenization_ns = tokenize_ns;
        std::vector<GenerateResult> results;
        std::vector<std::vector<int>> raw_tokens;
        results.reserve(prompt_ids.size());
        raw_tokens.reserve(prompt_ids.size());
        long long live_weight = 0;
        long long live_steps = 0;

        // A single HTTP request is internally drained through bounded resident
        // batches. This keeps the native wire primitive one-call while honoring
        // the worker's sequence and physical-KV limits.
        size_t cursor = 0;
        try {
            ContextPool::Lease lease = ctx.pool.acquire();
            while (cursor < prompt_ids.size()) {
                std::vector<std::vector<int>> batch_prompts;
                size_t end = cursor;
                int physical_rows = 0;
                while (end < prompt_ids.size() && batch_prompts.size() < 16) {
                    const int next_rows = static_cast<int>(prompt_ids[end].size());
                    if (!batch_prompts.empty() && physical_rows + next_rows > ctx.n_ctx) break;
                    physical_rows += next_rows;
                    batch_prompts.push_back(prompt_ids[end]);
                    ++end;
                }
                ReferenceMatchBatchResult measured = generate_ar_reference_match_batched(
                    *lease, batch_prompts, reference, max_tokens, stops);
                const auto& m = measured.metrics;
                total_metrics.prefill_ns += m.prefill_ns;
                total_metrics.decode_ns += m.decode_ns;
                total_metrics.physical_prompt_rows += m.physical_prompt_rows;
                total_metrics.output_token_positions_evaluated += m.output_token_positions_evaluated;
                total_metrics.model_forward_decode_calls += m.model_forward_decode_calls;
                live_weight += static_cast<long long>(m.mean_live_sequences * m.decode_steps);
                live_steps += m.decode_steps;
                total_metrics.max_live_sequences = std::max(total_metrics.max_live_sequences,
                                                            m.max_live_sequences);
                total_metrics.peak_resident_sequences = std::max(total_metrics.peak_resident_sequences,
                                                                  m.peak_resident_sequences);
                for (const auto& item : m.first_divergence_histogram)
                    total_metrics.first_divergence_histogram[item.first] += item.second;
                results.insert(results.end(), measured.arms.begin(), measured.arms.end());
                raw_tokens.insert(raw_tokens.end(), measured.generated_token_ids.begin(),
                                  measured.generated_token_ids.end());
                cursor = end;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("native reference-match failed: ") + e.what()},
                                 {"execution_regime", "native_batched_experimental"},
                                 {"proof_grade", false}}.dump(), "application/json");
            return;
        }

        total_metrics.wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - wall_started).count();
        // The weighted mean is the only aggregate whose denominator is not
        // additive; report the conservative observed maximum and a measured
        // request-level mean over all non-empty decode steps below.
        total_metrics.mean_live_sequences = live_steps > 0
            ? static_cast<double>(live_weight) / static_cast<double>(live_steps) : 0.0;

        json rows = json::array();
        const std::string stop_termination = stops.empty() ? "stop" : "user_stop_sequence";
        for (size_t i = 0; i < results.size(); ++i) {
            const GenerateResult& result = results[i];
            const std::string termination_kind = result.reason == "stop"
                                                     ? stop_termination
                                                     : result.reason;
            rows.push_back({
                {"arm_id", arm_ids[i]},
                {"result", {
                    {"generated_token_ids", raw_tokens[i]},
                    {"reply", result.text},
                    {"finish_reason", finish_reason(result.reason)},
                    {"termination", {{"kind", termination_kind}}},
                    {"diverged", result.diverged},
                    {"diverged_at", result.diverged_at},
                }},
            });
        }
        json divergence = json::object();
        for (const auto& item : total_metrics.first_divergence_histogram)
            divergence[std::to_string(item.first)] = item.second;
        res.set_content(dump_json({
            {"results", rows},
            {"execution_regime", "native_batched_experimental"},
            {"proof_grade", false},
            {"metrics", {
                {"total_wall_time_ns", total_metrics.wall_ns},
                {"prompt_rendering_time_ns", 0},
                {"prompt_tokenization_time_ns", total_metrics.prompt_tokenization_ns},
                {"native_prefill_time_ns", total_metrics.prefill_ns},
                {"native_decode_time_ns", total_metrics.decode_ns},
                {"n_ctx", ctx.pool.n_ctx()},
                {"n_batch", ctx.pool.n_batch()},
                {"n_ubatch", ctx.pool.n_ubatch()},
                {"memory", ctx.pool.memory_breakdown()},
                {"model_forward_decode_call_count", total_metrics.model_forward_decode_calls},
                {"physical_prompt_rows_submitted", total_metrics.physical_prompt_rows},
                {"output_token_positions_evaluated", total_metrics.output_token_positions_evaluated},
                {"first_divergence_histogram", divergence},
                {"max_live_sequences_per_decode_step", total_metrics.max_live_sequences},
                {"mean_live_sequences_per_decode_step", total_metrics.mean_live_sequences},
                {"peak_resident_sequences", total_metrics.peak_resident_sequences},
                {"peak_kv_usage", nullptr},
                {"cancellation_point", nullptr},
                {"prefix_reuse", false},
            }},
        }), "application/json");
    });
}

}  // namespace clozn
