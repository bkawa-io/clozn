// Private experiment route for native exact recorded-reference multi-arm probes.
// It is deliberately separate from /v1/completions: proof arms are detached
// measurements, never chat requests and never journaled Runs.
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "checkpoint_codec.hpp"
#include "server_context.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <limits>
#include <string>
#include <unordered_set>
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

std::string digest_ints(const std::vector<int>& values) {
    RunningSha256 hash;
    for (int value : values) hash.update_pod(static_cast<std::int32_t>(value));
    return hash.hex_digest();
}

std::string digest_json(const json& value) {
    const std::string serialized = value.dump();
    RunningSha256 hash;
    hash.update(serialized.data(), serialized.size());
    return hash.hex_digest();
}

bool parse_generation_contract(const json& contract, int& max_tokens,
                               std::vector<std::string>& stops, std::string& error) {
    if (!contract.is_object()) {
        error = "generation_contract must be an object";
        return false;
    }
    if (contract.value("decode_mode", std::string()) != "greedy") {
        error = "native persistent reference-match currently supports greedy only";
        return false;
    }
    max_tokens = contract.value("max_new", 0);
    if (max_tokens < 1) {
        error = "generation_contract.max_new must be a positive integer";
        return false;
    }
    stops.clear();
    if (contract.contains("stop")) {
        if (!contract["stop"].is_array()) {
            error = "generation_contract.stop must be an array";
            return false;
        }
        for (const auto& value : contract["stop"]) {
            if (!value.is_string()) {
                error = "generation_contract.stop must contain strings";
                return false;
            }
            stops.push_back(value.get<std::string>());
        }
    }
    return true;
}

struct PersistentParentSession {
    std::string session_id;
    std::unique_ptr<ContextPool::PersistentLease> lease;
    std::string worker_generation_id;
    std::string model_path;
    std::string model_sha256;
    std::string tokenizer_sha256;
    int n_ctx = 0;
    int n_batch = 0;
    int n_ubatch = 0;
    std::vector<int> reference_token_ids;
    json generation_contract;
    std::string reference_digest;
    std::string generation_digest;
    std::vector<int> parent_prompt;
    std::string parent_prompt_digest;
    std::uint64_t parent_version = 0;
    std::string retained_candidate_id;
    int retained_candidate_rank = std::numeric_limits<int>::max();
    std::uint64_t retained_parent_version = 0;
    std::vector<int> retained_prompt;
    long long parent_prefill_rows = 0;
    long long child_logical_prompt_rows = 0;
    long long parent_prefix_rows_reused = 0;
    long long child_suffix_rows_evaluated = 0;
    long long parent_refill_rows_after_initial_create = 0;
    int promoted_child_count = 0;
    int search_round_count = 0;
    int probe_count = 0;
    std::int64_t total_native_probe_wall_ns = 0;
    std::int64_t total_scalar_confirmation_wall_ns = 0;
    std::int64_t promotion_wall_ns = 0;
};

struct PersistentSessionRegistry {
    std::mutex mutex;
    std::unique_ptr<PersistentParentSession> session;
    std::uint64_t next_session_id = 1;
};

json raw_probe_result(const ReferenceMatchProbeResult& probe, const std::vector<std::string>& stops) {
    const GenerateResult& result = probe.result;
    const std::string stop_termination = stops.empty() ? "stop" : "user_stop_sequence";
    const std::string termination_kind = result.reason == "stop" ? stop_termination : result.reason;
    return {
        {"generated_token_ids", probe.generated_token_ids},
        {"reply", result.text},
        {"finish_reason", finish_reason(result.reason)},
        {"termination", {{"kind", termination_kind}}},
        {"diverged", result.diverged},
        {"diverged_at", result.diverged_at},
    };
}

json persistent_runtime_identity(const PersistentParentSession& session) {
    return {
        {"worker_generation_id", session.worker_generation_id},
        {"model_path", session.model_path},
        {"model_sha256", session.model_sha256},
        {"tokenizer_sha256", session.tokenizer_sha256},
        {"n_ctx", session.n_ctx},
        {"n_batch", session.n_batch},
        {"n_ubatch", session.n_ubatch},
    };
}

json persistent_session_metrics(const PersistentParentSession& session) {
    return {
        {"parent_prefill_rows", session.parent_prefill_rows},
        {"child_logical_prompt_rows", session.child_logical_prompt_rows},
        {"parent_prefix_rows_reused", session.parent_prefix_rows_reused},
        {"child_suffix_rows_evaluated", session.child_suffix_rows_evaluated},
        {"promoted_child_count", session.promoted_child_count},
        {"parent_refill_rows_after_initial_create", session.parent_refill_rows_after_initial_create},
        {"search_round_count", session.search_round_count},
        {"probe_count", session.probe_count},
        {"total_native_probe_wall_seconds",
         static_cast<double>(session.total_native_probe_wall_ns) / 1.0e9},
        {"total_scalar_confirmation_wall_seconds",
         static_cast<double>(session.total_scalar_confirmation_wall_ns) / 1.0e9},
        {"promotion_wall_seconds", static_cast<double>(session.promotion_wall_ns) / 1.0e9},
    };
}

}  // namespace

void register_reference_match_routes(httplib::Server& svr, ServerContext& ctx) {
    // v0 deliberately supports one worker-local persistent session.  The lease stays reserved for
    // the session lifetime, so no other request can accidentally observe or mutate its KV.  The
    // mutex serializes session operations; ordinary stateless routes retain their existing pool
    // discipline and the experimental mode is explicitly non-proof-grade.
    // Captured by value in the route handlers: register_* returns before HTTP requests arrive, so
    // persistent state must outlive this function and be reclaimed with the server's route closures.
    auto persistent_state = std::make_shared<PersistentSessionRegistry>();

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

    // -------------------------------------------------------------------------
    // Experimental persistent accepted-parent session.
    // -------------------------------------------------------------------------
    svr.Post("/v1/reference-match/persistent/create", [&, persistent_state](const httplib::Request& req,
                                                            httplib::Response& res) {
        std::lock_guard<std::mutex> lock(persistent_state->mutex);
        const json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        auto fail = [&](int status, const std::string& code, const std::string& message) {
            res.status = status;
            res.set_content(json{{"error", message}, {"code", code}, {"proof_grade", false}}.dump(),
                            "application/json");
        };
        if (body.is_discarded() || !body.is_object()) {
            fail(400, "invalid_request", "invalid JSON body");
            return;
        }
        if (!ctx.ar_mode) {
            fail(409, "capability_unavailable", "persistent parent sessions require autoregressive mode");
            return;
        }
        if (persistent_state->session) {
            fail(409, "persistent_session_busy", "one persistent parent session is already active on this worker");
            return;
        }
        if (!body.contains("prompt") || !body["prompt"].is_string() || body["prompt"].get<std::string>().empty()) {
            fail(400, "invalid_prompt", "prompt must be a non-empty canonical rendered prompt string");
            return;
        }
        if (!body.contains("reference_token_ids") || !body["reference_token_ids"].is_array() ||
            body["reference_token_ids"].empty()) {
            fail(400, "invalid_reference", "reference_token_ids must be a non-empty integer array");
            return;
        }
        std::vector<int> reference;
        try {
            reference = body["reference_token_ids"].get<std::vector<int>>();
        } catch (...) {
            fail(400, "invalid_reference", "reference_token_ids must contain integers");
            return;
        }
        const json contract = body.value("generation_contract", json::object());
        int max_tokens = 0;
        std::vector<std::string> stops;
        std::string contract_error;
        if (!parse_generation_contract(contract, max_tokens, stops, contract_error)) {
            fail(409, "unsupported_generation_contract", contract_error);
            return;
        }

        auto candidate = std::make_unique<PersistentParentSession>();
        candidate->session_id = "rpps_" + ctx.worker_generation_id + "_" +
                                std::to_string(persistent_state->next_session_id++);
        candidate->worker_generation_id = ctx.worker_generation_id;
        candidate->model_path = ctx.model_path;
        candidate->model_sha256 = ctx.model_sha256;
        candidate->tokenizer_sha256 = ctx.tokenizer_sha256;
        candidate->n_ctx = ctx.pool.n_ctx();
        candidate->n_batch = ctx.pool.n_batch();
        candidate->n_ubatch = ctx.pool.n_ubatch();
        candidate->reference_token_ids = std::move(reference);
        candidate->generation_contract = contract;
        candidate->reference_digest = digest_ints(candidate->reference_token_ids);
        candidate->generation_digest = digest_json(contract);
        try {
            candidate->parent_prompt = ctx.model->encode(body["prompt"].get<std::string>());
        } catch (const std::exception& e) {
            fail(400, "prompt_tokenization_failed", e.what());
            return;
        }
        if (candidate->parent_prompt.empty()) {
            fail(400, "invalid_prompt", "prompt must tokenize to at least one token");
            return;
        }
        if (static_cast<int>(candidate->parent_prompt.size()) > ctx.n_ctx) {
            fail(400, "prompt_too_long", "prompt exceeds the worker context window");
            return;
        }
        candidate->parent_prompt_digest = digest_ints(candidate->parent_prompt);

        try {
            candidate->lease = std::make_unique<ContextPool::PersistentLease>(ctx.pool.acquire_persistent());
            GgmlAdapter& adapter = **candidate->lease;
            adapter.set_causal(true);
            adapter.ar_forward_seq_segment(candidate->parent_prompt, 0,
                                           static_cast<int>(candidate->parent_prompt.size()), true, 0);
            if (adapter.ar_seq_size(0) != static_cast<int>(candidate->parent_prompt.size()))
                throw std::runtime_error("persistent parent prefill did not establish the expected sequence");
            candidate->parent_prefill_rows = static_cast<long long>(candidate->parent_prompt.size());
            persistent_state->session = std::move(candidate);
        } catch (const std::exception& e) {
            if (candidate && candidate->lease) {
                try { (**candidate->lease).reset_ar_kv(); } catch (...) {}
            }
            fail(400, "persistent_session_create_failed",
                 std::string("persistent parent session create failed: ") + e.what());
            return;
        }

        const PersistentParentSession& session = *persistent_state->session;
        res.set_content(dump_json({
            {"session_id", session.session_id},
            {"parent_version", session.parent_version},
            {"parent_prompt_token_count", session.parent_prompt.size()},
            {"parent_prompt_digest", session.parent_prompt_digest},
            {"reference_token_digest", session.reference_digest},
            {"generation_contract_digest", session.generation_digest},
            {"runtime_identity", persistent_runtime_identity(session)},
            {"execution_regime", "native_reference_match_persistent_parent_experimental"},
            {"proof_grade", false},
            {"telemetry", persistent_session_metrics(session)},
        }), "application/json");
    });

    svr.Post("/v1/reference-match/persistent/probe", [&, persistent_state](const httplib::Request& req,
                                                           httplib::Response& res) {
        std::lock_guard<std::mutex> lock(persistent_state->mutex);
        auto fail = [&](int status, const std::string& code, const std::string& message) {
            res.status = status;
            res.set_content(json{{"error", message}, {"code", code}, {"proof_grade", false}}.dump(),
                            "application/json");
        };
        const json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            fail(400, "invalid_request", "invalid JSON body");
            return;
        }
        if (!persistent_state->session) {
            fail(404, "session_not_found", "persistent parent session is not active");
            return;
        }
        PersistentParentSession& session = *persistent_state->session;
        if (!body.contains("session_id") || !body["session_id"].is_string() ||
            body["session_id"].get<std::string>() != session.session_id) {
            fail(409, "stale_session", "session_id does not identify the active persistent session");
            return;
        }
        if (!body.contains("expected_parent_version") || !body["expected_parent_version"].is_number_integer()) {
            fail(400, "invalid_parent_version", "expected_parent_version must be an integer");
            return;
        }
        const std::int64_t expected_version = body["expected_parent_version"].get<std::int64_t>();
        if (expected_version != static_cast<std::int64_t>(session.parent_version)) {
            fail(409, "stale_parent_state", "probe round is bound to an older parent version");
            return;
        }
        if (!body.contains("children") || !body["children"].is_array() || body["children"].empty()) {
            fail(400, "invalid_children", "children must be a non-empty array");
            return;
        }

        struct Child {
            std::string id;
            std::string prompt_text;
            std::vector<int> prompt;
            int rank = 0;
            int lcp = 0;
        };
        std::vector<Child> children;
        children.reserve(body["children"].size());
        std::unordered_set<std::string> seen_ids;
        std::unordered_set<int> seen_ranks;
        for (const auto& item : body["children"]) {
            if (!item.is_object() || !item.contains("candidate_id") ||
                !item["candidate_id"].is_string() || item["candidate_id"].get<std::string>().empty() ||
                !item.contains("prompt") || !item["prompt"].is_string() || item["prompt"].get<std::string>().empty() ||
                !item.contains("candidate_rank") || !item["candidate_rank"].is_number_integer()) {
                fail(400, "invalid_child", "each child requires candidate_id, prompt, and integer candidate_rank");
                return;
            }
            Child child;
            child.id = item["candidate_id"].get<std::string>();
            child.prompt_text = item["prompt"].get<std::string>();
            child.rank = item["candidate_rank"].get<int>();
            if (child.rank < 0 || !seen_ids.insert(child.id).second ||
                !seen_ranks.insert(child.rank).second) {
                fail(400, "invalid_child", "candidate_id and candidate_rank must be unique non-negative values");
                return;
            }
            try {
                child.prompt = ctx.model->encode(child.prompt_text);
            } catch (const std::exception& e) {
                fail(400, "prompt_tokenization_failed", e.what());
                return;
            }
            if (child.prompt.empty()) {
                fail(400, "invalid_child", "child prompt must tokenize to at least one token");
                return;
            }
            if (static_cast<int>(child.prompt.size()) > ctx.n_ctx) {
                fail(400, "prompt_too_long", "child prompt exceeds the worker context window");
                return;
            }
            const int limit = static_cast<int>(std::min(session.parent_prompt.size(), child.prompt.size()));
            while (child.lcp < limit && session.parent_prompt[static_cast<size_t>(child.lcp)] ==
                                             child.prompt[static_cast<size_t>(child.lcp)]) {
                ++child.lcp;
            }
            children.push_back(std::move(child));
        }

        constexpr int kParentSeq = 0;
        constexpr int kRetainedSeq = 1;
        constexpr int kFirstWorkingSeq = 2;
        constexpr int kWorkingWaveSize = 14; // existing llama n_seq_max is 16; parent + reserve consume 2
        const auto started = std::chrono::steady_clock::now();
        std::vector<json> rows(children.size());
        std::vector<int> lcp_values;
        lcp_values.reserve(children.size());
        long long round_logical_rows = 0;
        long long round_theoretical_suffix_rows = 0;
        long long round_actual_rows = 0;
        long long round_reused_rows = 0;
        long long round_suffix_rows = 0;
        int wave_count = 0;
        try {
            GgmlAdapter& adapter = **session.lease;
            if (adapter.ar_seq_size(kParentSeq) != static_cast<int>(session.parent_prompt.size()))
                throw std::runtime_error("persistent parent sequence was mutated before probe round");
            const auto expected_termination = session.generation_contract.value(
                "expected_termination", json::object()).value("reason_raw", std::string());

            for (size_t wave_start = 0; wave_start < children.size(); wave_start += kWorkingWaveSize) {
                const size_t wave_end = std::min(children.size(), wave_start + kWorkingWaveSize);
                ++wave_count;
                for (size_t index = wave_start; index < wave_end; ++index) {
                    Child& child = children[index];
                    const int seq_id = kFirstWorkingSeq + static_cast<int>(index - wave_start);
                    adapter.clear_ar_seq(seq_id);

                    // A child that is a complete parent prefix has no suffix rows to decode, but we
                    // still need logits for its final prompt token. Reuse all but that terminal row
                    // and recompute exactly one row; this is explicit in actual-vs-theoretical telemetry.
                    int copy_to = child.lcp;
                    int decode_from = child.lcp;
                    if (decode_from == static_cast<int>(child.prompt.size())) {
                        copy_to = std::max(0, decode_from - 1);
                        decode_from = copy_to;
                    }
                    if (copy_to > 0)
                        adapter.copy_ar_seq(kParentSeq, seq_id, 0, copy_to);
                    ForwardResult logits = adapter.ar_forward_seq_segment(
                        child.prompt, decode_from, static_cast<int>(child.prompt.size()), true, seq_id);
                    const ReferenceMatchProbeResult probe = generate_ar_reference_match_probe(
                        adapter, seq_id, child.prompt, logits, session.reference_token_ids,
                        session.generation_contract.value("max_new", 0),
                        session.generation_contract.value("stop", std::vector<std::string>{}));
                    if (adapter.ar_seq_size(seq_id) != static_cast<int>(child.prompt.size()))
                        throw std::runtime_error("clean child prompt state was not preserved after probe");

                    const long long logical_rows = static_cast<long long>(child.prompt.size());
                    const long long actual_rows = static_cast<long long>(child.prompt.size() - decode_from);
                    const long long theoretical_rows = logical_rows - static_cast<long long>(child.lcp);
                    round_logical_rows += logical_rows;
                    round_theoretical_suffix_rows += theoretical_rows;
                    round_actual_rows += actual_rows;
                    round_reused_rows += copy_to;
                    round_suffix_rows += actual_rows;
                    lcp_values.push_back(child.lcp);

                    const bool native_preserves = !probe.result.diverged &&
                        (expected_termination.empty() || probe.result.reason == expected_termination);
                    bool retained = false;
                    if (native_preserves && child.rank < session.retained_candidate_rank) {
                        adapter.clear_ar_seq(kRetainedSeq);
                        adapter.copy_ar_seq(seq_id, kRetainedSeq, 0,
                                           static_cast<int>(child.prompt.size()));
                        if (adapter.ar_seq_size(kRetainedSeq) != static_cast<int>(child.prompt.size()))
                            throw std::runtime_error("retained clean child prompt state has the wrong length");
                        session.retained_candidate_id = child.id;
                        session.retained_candidate_rank = child.rank;
                        session.retained_parent_version = session.parent_version;
                        session.retained_prompt = child.prompt;
                        retained = true;
                    }
                    rows[index] = {
                        {"candidate_id", child.id},
                        {"candidate_rank", child.rank},
                        {"lcp_tokens", child.lcp},
                        {"actual_child_prompt_rows_evaluated", actual_rows},
                        {"theoretical_suffix_rows", theoretical_rows},
                        {"native_preserves", native_preserves},
                        {"retained_for_promotion", retained},
                        {"result", raw_probe_result(probe, session.generation_contract.value(
                            "stop", std::vector<std::string>{}))},
                    };
                    adapter.clear_ar_seq(seq_id);
                    if (adapter.ar_seq_size(kParentSeq) != static_cast<int>(session.parent_prompt.size()))
                        throw std::runtime_error("persistent parent sequence changed during child probe");
                }
            }
            session.child_logical_prompt_rows += round_logical_rows;
            session.parent_prefix_rows_reused += round_reused_rows;
            session.child_suffix_rows_evaluated += round_suffix_rows;
            session.search_round_count += 1;
            session.probe_count += static_cast<int>(children.size());
            const auto wall_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
            session.total_native_probe_wall_ns += wall_ns;
            std::sort(lcp_values.begin(), lcp_values.end());
            auto percentile = [&](double fraction) -> int {
                if (lcp_values.empty()) return 0;
                const size_t position = static_cast<size_t>(fraction * static_cast<double>(lcp_values.size() - 1));
                return lcp_values[position];
            };
            json round_metrics = {
                {"parent_version", session.parent_version},
                {"parent_prompt_tokens", session.parent_prompt.size()},
                {"parent_prompt_digest", session.parent_prompt_digest},
                {"child_count", children.size()},
                {"lcp_min", lcp_values.empty() ? 0 : lcp_values.front()},
                {"lcp_p50", percentile(0.50)},
                {"lcp_p90", percentile(0.90)},
                {"lcp_max", lcp_values.empty() ? 0 : lcp_values.back()},
                {"logical_child_prompt_rows", round_logical_rows},
                {"theoretical_suffix_rows", round_theoretical_suffix_rows},
                {"actual_child_prompt_rows_evaluated", round_actual_rows},
                {"reused_parent_prefix_rows", round_reused_rows},
                {"evaluated_child_suffix_rows", round_suffix_rows},
                {"resident_wave_count", wave_count},
                {"native_probe_wall_seconds", static_cast<double>(wall_ns) / 1.0e9},
                {"parent_refill_rows_after_initial_create", session.parent_refill_rows_after_initial_create},
            };
            res.set_content(dump_json({
                {"session_id", session.session_id},
                {"parent_version", session.parent_version},
                {"results", rows},
                {"round_metrics", round_metrics},
                {"telemetry", persistent_session_metrics(session)},
                {"retained_candidate_id", session.retained_candidate_id.empty()
                                             ? json(nullptr) : json(session.retained_candidate_id)},
                {"execution_regime", "native_reference_match_persistent_parent_experimental"},
                {"proof_grade", false},
            }), "application/json");
        } catch (const std::exception& e) {
            // The parent is only safe to retain if our cheap sequence assertion still holds.  On any
            // uncertain native failure, close the session and force the caller to recreate it.
            try {
                if (session.lease && (**session.lease).ar_seq_size(kParentSeq) ==
                    static_cast<int>(session.parent_prompt.size())) {
                    for (int seq = kRetainedSeq; seq < 16; ++seq) (**session.lease).clear_ar_seq(seq);
                } else if (session.lease) {
                    (**session.lease).reset_ar_kv();
                }
            } catch (...) {}
            persistent_state->session.reset();
            fail(400, "persistent_session_invalidated",
                 std::string("persistent probe failed and the session was closed: ") + e.what());
        }
    });

    svr.Post("/v1/reference-match/persistent/promote", [&, persistent_state](const httplib::Request& req,
                                                              httplib::Response& res) {
        std::lock_guard<std::mutex> lock(persistent_state->mutex);
        auto fail = [&](int status, const std::string& code, const std::string& message) {
            res.status = status;
            res.set_content(json{{"error", message}, {"code", code}, {"proof_grade", false}}.dump(),
                            "application/json");
        };
        const json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            fail(400, "invalid_request", "invalid JSON body");
            return;
        }
        if (!persistent_state->session) {
            fail(404, "session_not_found", "persistent parent session is not active");
            return;
        }
        PersistentParentSession& session = *persistent_state->session;
        if (!body.contains("session_id") || !body["session_id"].is_string() ||
            body["session_id"].get<std::string>() != session.session_id) {
            fail(409, "stale_session", "session_id does not identify the active persistent session");
            return;
        }
        if (!body.contains("expected_parent_version") || !body["expected_parent_version"].is_number_integer()) {
            fail(400, "invalid_parent_version", "expected_parent_version must be an integer");
            return;
        }
        if (body["expected_parent_version"].get<std::int64_t>() !=
            static_cast<std::int64_t>(session.parent_version)) {
            fail(409, "stale_parent_state", "promotion is bound to an older parent version");
            return;
        }
        if (!body.contains("candidate_id") || !body["candidate_id"].is_string() ||
            body["candidate_id"].get<std::string>() != session.retained_candidate_id ||
            session.retained_parent_version != session.parent_version) {
            fail(409, "stale_candidate", "candidate is not the retained clean child of the current parent");
            return;
        }
        constexpr int kParentSeq = 0;
        constexpr int kRetainedSeq = 1;
        try {
            GgmlAdapter& adapter = **session.lease;
            if (adapter.ar_seq_size(kParentSeq) != static_cast<int>(session.parent_prompt.size()) ||
                adapter.ar_seq_size(kRetainedSeq) != static_cast<int>(session.retained_prompt.size()))
                throw std::runtime_error("promotion source sequence is not intact");
            const auto started = std::chrono::steady_clock::now();
            adapter.clear_ar_seq(kParentSeq);
            adapter.copy_ar_seq(kRetainedSeq, kParentSeq, 0,
                                static_cast<int>(session.retained_prompt.size()));
            if (adapter.ar_seq_size(kParentSeq) != static_cast<int>(session.retained_prompt.size()))
                throw std::runtime_error("promotion did not establish the new parent sequence");
            for (int seq = kRetainedSeq; seq < 16; ++seq) adapter.clear_ar_seq(seq);
            session.parent_prompt = session.retained_prompt;
            session.parent_prompt_digest = digest_ints(session.parent_prompt);
            session.parent_version += 1;
            session.promoted_child_count += 1;
            session.promotion_wall_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - started).count();
            session.retained_candidate_id.clear();
            session.retained_candidate_rank = std::numeric_limits<int>::max();
            session.retained_parent_version = session.parent_version;
            session.retained_prompt.clear();
            res.set_content(dump_json({
                {"session_id", session.session_id},
                {"parent_version", session.parent_version},
                {"parent_prompt_token_count", session.parent_prompt.size()},
                {"parent_prompt_digest", session.parent_prompt_digest},
                {"promoted_candidate_id", body["candidate_id"]},
                {"parent_refill_rows_after_initial_create", session.parent_refill_rows_after_initial_create},
                {"telemetry", persistent_session_metrics(session)},
                {"execution_regime", "native_reference_match_persistent_parent_experimental"},
                {"proof_grade", false},
            }), "application/json");
        } catch (const std::exception& e) {
            if (session.lease) {
                try { (**session.lease).reset_ar_kv(); } catch (...) {}
            }
            persistent_state->session.reset();
            fail(400, "persistent_session_invalidated",
                 std::string("promotion failed and the session was closed: ") + e.what());
        }
    });

    svr.Post("/v1/reference-match/persistent/close", [&, persistent_state](const httplib::Request& req,
                                                           httplib::Response& res) {
        std::lock_guard<std::mutex> lock(persistent_state->mutex);
        auto fail = [&](int status, const std::string& code, const std::string& message) {
            res.status = status;
            res.set_content(json{{"error", message}, {"code", code}, {"proof_grade", false}}.dump(),
                            "application/json");
        };
        const json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object() || !body.contains("session_id") ||
            !body["session_id"].is_string()) {
            fail(400, "invalid_request", "session_id must be supplied");
            return;
        }
        if (!persistent_state->session) {
            fail(404, "session_not_found", "persistent parent session is not active");
            return;
        }
        if (body["session_id"].get<std::string>() != persistent_state->session->session_id) {
            fail(409, "stale_session", "session_id does not identify the active persistent session");
            return;
        }
        persistent_state->session.reset(); // Lease destructor releases the reserved worker context.
        res.set_content(json{{"closed", true}, {"proof_grade", false}}.dump(), "application/json");
    });
}

}  // namespace clozn
