// Private experiment route for native exact recorded-reference multi-arm probes.
// It is deliberately separate from /v1/completions: proof arms are detached
// measurements, never chat requests and never journaled Runs.
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "server_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

namespace clozn {

using json = nlohmann::json;

namespace {

enum class ReferenceMatchExecutionStrategy {
    ResidentBatched,
    RequestWideRollback,
    ParentAnchored,
};

ReferenceMatchExecutionStrategy reference_match_strategy() {
    const char* value = std::getenv("CLOZN_REFERENCE_MATCH_STRATEGY");
    if (value != nullptr && std::string(value) == "rollback")
        return ReferenceMatchExecutionStrategy::RequestWideRollback;
    if (value != nullptr && std::string(value) == "parent_anchor")
        return ReferenceMatchExecutionStrategy::ParentAnchored;
    return ReferenceMatchExecutionStrategy::ResidentBatched;
}

}  // namespace

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
        std::vector<int> parent_anchor_prompt;
        bool has_parent_anchor = false;
        if (body.contains("parent_anchor_prompt")) {
            if (!body["parent_anchor_prompt"].is_string()) {
                fail("parent_anchor_prompt must be a string");
                return;
            }
            try {
                parent_anchor_prompt = ctx.model->encode(body["parent_anchor_prompt"].get<std::string>());
            } catch (const std::exception& e) {
                fail(std::string("parent anchor tokenization failed: ") + e.what());
                return;
            }
            if (parent_anchor_prompt.empty()) {
                fail("parent_anchor_prompt must tokenize to at least one token");
                return;
            }
            if (static_cast<int>(parent_anchor_prompt.size()) > ctx.n_ctx) {
                fail("parent_anchor_prompt exceeds the worker context window");
                return;
            }
            has_parent_anchor = true;
        }
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
        const ReferenceMatchExecutionStrategy strategy = reference_match_strategy();
        const bool request_wide_rollback =
            strategy == ReferenceMatchExecutionStrategy::RequestWideRollback;
        // The parent-anchor field is itself the explicit wire opt-in. This
        // keeps the worker default resident path unchanged and avoids
        // requiring a process-wide environment toggle for per-request use.
        const bool parent_anchored = has_parent_anchor;
        const std::string strategy_name = parent_anchored
                                              ? "parent_anchor"
                                              : (request_wide_rollback ? "rollback" : "resident");
        const std::string execution_regime = parent_anchored
                                                  ? "native_reference_match_parent_anchor_experimental"
                                                  : (request_wide_rollback
                                                         ? "native_reference_match_rollback_experimental"
                                                         : "native_batched_experimental");

        auto aggregate_metrics = [&](const ReferenceMatchBatchMetrics& m) {
            total_metrics.prefill_ns += m.prefill_ns;
            total_metrics.decode_ns += m.decode_ns;
            total_metrics.logical_prompt_rows += m.logical_prompt_rows;
            total_metrics.physical_prompt_rows += m.physical_prompt_rows;
            total_metrics.prefix_rows_reused += m.prefix_rows_reused;
            total_metrics.output_token_positions_evaluated += m.output_token_positions_evaluated;
            total_metrics.model_forward_decode_calls += m.model_forward_decode_calls;
            total_metrics.request_wide_prefix_reuse =
                total_metrics.request_wide_prefix_reuse || m.request_wide_prefix_reuse;
            total_metrics.traversal_decode_call_count += m.traversal_decode_call_count;
            total_metrics.probe_decode_call_count += m.probe_decode_call_count;
            total_metrics.rollback_count += m.rollback_count;
            total_metrics.rollback_prompt_rows += m.rollback_prompt_rows;
            total_metrics.unique_terminal_prompts += m.unique_terminal_prompts;
            total_metrics.duplicate_terminal_arms_reused += m.duplicate_terminal_arms_reused;
            total_metrics.max_traversal_depth = std::max(total_metrics.max_traversal_depth,
                                                         m.max_traversal_depth);
            total_metrics.traversal_planning_ns += m.traversal_planning_ns;
            total_metrics.parent_anchor_reuse =
                total_metrics.parent_anchor_reuse || m.parent_anchor_reuse;
            total_metrics.parent_anchor_children += m.parent_anchor_children;
            total_metrics.parent_anchor_prefix_rows += m.parent_anchor_prefix_rows;
            total_metrics.parent_anchor_prompt_rows += m.parent_anchor_prompt_rows;
            total_metrics.parent_anchor_logical_rows += m.parent_anchor_logical_rows;
            total_metrics.parent_anchor_physical_rows += m.parent_anchor_physical_rows;
            live_weight += static_cast<long long>(m.mean_live_sequences * m.decode_steps);
            live_steps += m.decode_steps;
            total_metrics.max_live_sequences = std::max(total_metrics.max_live_sequences,
                                                        m.max_live_sequences);
            total_metrics.peak_resident_sequences = std::max(total_metrics.peak_resident_sequences,
                                                              m.peak_resident_sequences);
            for (const auto& item : m.first_divergence_histogram)
                total_metrics.first_divergence_histogram[item.first] += item.second;
        };

        // A single HTTP request is internally drained through bounded resident
        // batches. This keeps the native wire primitive one-call while honoring
        // the worker's sequence and physical-KV limits.
        size_t cursor = 0;
        try {
            ContextPool::Lease lease = ctx.pool.acquire();
            if (parent_anchored) {
                ReferenceMatchBatchResult measured = generate_ar_reference_match_parent_anchor(
                    *lease, parent_anchor_prompt, prompt_ids, reference, max_tokens, stops);
                aggregate_metrics(measured.metrics);
                results = std::move(measured.arms);
                raw_tokens = std::move(measured.generated_token_ids);
                cursor = prompt_ids.size();
            } else if (request_wide_rollback) {
                ReferenceMatchBatchResult measured = generate_ar_reference_match_rollback(
                    *lease, prompt_ids, reference, max_tokens, stops);
                aggregate_metrics(measured.metrics);
                results.insert(results.end(), measured.arms.begin(), measured.arms.end());
                raw_tokens.insert(raw_tokens.end(), measured.generated_token_ids.begin(),
                                  measured.generated_token_ids.end());
                cursor = prompt_ids.size();
            }
            while (!request_wide_rollback && cursor < prompt_ids.size()) {
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
                aggregate_metrics(measured.metrics);
                results.insert(results.end(), measured.arms.begin(), measured.arms.end());
                raw_tokens.insert(raw_tokens.end(), measured.generated_token_ids.begin(),
                                  measured.generated_token_ids.end());
                cursor = end;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("native reference-match failed: ") + e.what()},
                                 {"execution_regime", execution_regime},
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
            {"execution_regime", execution_regime},
            {"proof_grade", false},
            {"metrics", {
                {"total_wall_time_ns", total_metrics.wall_ns},
                {"prompt_rendering_time_ns", 0},
                {"prompt_tokenization_time_ns", total_metrics.prompt_tokenization_ns},
                {"native_prefill_time_ns", total_metrics.prefill_ns},
                {"native_decode_time_ns", total_metrics.decode_ns},
                {"strategy", strategy_name},
                {"request_wide_prefix_reuse", total_metrics.request_wide_prefix_reuse},
                {"n_ctx", ctx.pool.n_ctx()},
                {"n_batch", ctx.pool.n_batch()},
                {"n_ubatch", ctx.pool.n_ubatch()},
                {"memory", ctx.pool.memory_breakdown()},
                {"model_forward_decode_call_count", total_metrics.model_forward_decode_calls},
                {"logical_prompt_rows", total_metrics.logical_prompt_rows},
                {"physical_prompt_rows_submitted", total_metrics.physical_prompt_rows},
                {"prefix_rows_reused", total_metrics.prefix_rows_reused},
                {"reuse_percent", (total_metrics.parent_anchor_reuse
                                      ? total_metrics.parent_anchor_logical_rows
                                      : total_metrics.logical_prompt_rows) > 0
                                      ? (100.0 * static_cast<double>(total_metrics.prefix_rows_reused) /
                                         static_cast<double>(total_metrics.parent_anchor_reuse
                                             ? total_metrics.parent_anchor_logical_rows
                                             : total_metrics.logical_prompt_rows))
                                      : 0.0},
                {"traversal_decode_call_count", total_metrics.traversal_decode_call_count},
                {"probe_decode_call_count", total_metrics.probe_decode_call_count},
                {"rollback_count", total_metrics.rollback_count},
                {"rollback_prompt_rows", total_metrics.rollback_prompt_rows},
                {"unique_terminal_prompts", total_metrics.unique_terminal_prompts},
                {"duplicate_terminal_arms_reused", total_metrics.duplicate_terminal_arms_reused},
                {"max_traversal_depth", total_metrics.max_traversal_depth},
                {"traversal_planning_time_ns", total_metrics.traversal_planning_ns},
                {"parent_anchor_reuse", total_metrics.parent_anchor_reuse},
                {"parent_anchor_children", total_metrics.parent_anchor_children},
                {"parent_anchor_prefix_rows", total_metrics.parent_anchor_prefix_rows},
                {"parent_anchor_prompt_rows", total_metrics.parent_anchor_prompt_rows},
                {"parent_anchor_logical_rows", total_metrics.parent_anchor_logical_rows},
                {"parent_anchor_physical_rows", total_metrics.parent_anchor_physical_rows},
                {"output_token_positions_evaluated", total_metrics.output_token_positions_evaluated},
                {"first_divergence_histogram", divergence},
                {"max_live_sequences_per_decode_step", total_metrics.max_live_sequences},
                {"mean_live_sequences_per_decode_step", total_metrics.mean_live_sequences},
                {"peak_resident_sequences", total_metrics.peak_resident_sequences},
                {"peak_kv_usage", nullptr},
                {"cancellation_point", nullptr},
                {"prefix_reuse", total_metrics.prefix_rows_reused > 0},
            }},
        }), "application/json");
    });
}

}  // namespace clozn
