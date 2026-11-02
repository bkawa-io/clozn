// Private v1 scoring and prompt-template routes.
#include "httplib.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "clozn/teacher_forced_score_many.hpp"
#include "server_context.hpp"

namespace clozn {

namespace {

bool reject_unknown_fields(const json& body, const std::set<std::string>& allowed,
                           httplib::Response& res) {
    for (const auto& item : body.items()) {
        if (allowed.count(item.key()) == 0) {
            res.status = 400;
            res.set_content(
                json{{"error", "unsupported v1 request field: " + item.key()}}.dump(),
                "application/json");
            return true;
        }
    }
    return false;
}

bool exact_token_ids(const json& value, const char* field, int vocab_size,
                     std::vector<int>& out, httplib::Response& res) {
    if (!value.is_array() || value.empty()) {
        res.status = 400;
        res.set_content(json{{"error", std::string(field) + " must be a non-empty integer array"}}.dump(),
                        "application/json");
        return false;
    }
    out.reserve(value.size());
    for (const auto& token : value) {
        if (!token.is_number_integer()) {
            res.status = 400;
            res.set_content(json{{"error", std::string(field) + " must contain only integers"}}.dump(),
                            "application/json");
            return false;
        }
        const int id = token.get<int>();
        if (id < 0 || id >= vocab_size) {
            res.status = 400;
            res.set_content(json{{"error", std::string(field) + " contains an out-of-range token"}}.dump(),
                            "application/json");
            return false;
        }
        out.push_back(id);
    }
    return true;
}

class NativeScoreManyScope {
public:
    NativeScoreManyScope(GgmlAdapter& adapter, int arm_count, bool use_backend_reduction)
        : adapter_(adapter) {
        // The route is valid for any causal GGUF supported by the runtime. No architecture or
        // tokenizer family is inspected here.
        adapter_.set_causal(true);
        adapter_.begin_target_logprob_reduction(arm_count, use_backend_reduction);
        reduction_active_ = true;
    }

    ~NativeScoreManyScope() {
        if (reduction_active_) adapter_.end_target_logprob_reduction();
        // This route intentionally does not populate either scalar prefix cache. Returning a clean
        // adapter keeps a failed/partial native attempt safe to retry through scalar /score.
        adapter_.reset_ar_kv();
    }

    NativeScoreManyScope(const NativeScoreManyScope&) = delete;
    NativeScoreManyScope& operator=(const NativeScoreManyScope&) = delete;

private:
    GgmlAdapter& adapter_;
    bool reduction_active_ = false;
};

}  // namespace

void register_whitebox_routes(httplib::Server& svr, ServerContext& ctx) {
    // Scalar teacher-forced scoring is the sole recorded-answer scorer in v1.
    svr.Post("/score", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "request body must be a JSON object"}}.dump(),
                            "application/json");
            return;
        }
        static const std::set<std::string> allowed{
            "prompt", "prompt_ids", "continuation", "continuation_ids", "topk",
            "use_prefix_cache",
        };
        if (reject_unknown_fields(body, allowed, res)) return;

        const int vocab_size = ctx.model->config().vocab_size;
        std::vector<int> prompt_ids;
        if (body.contains("prompt_ids")) {
            if (!exact_token_ids(body["prompt_ids"], "prompt_ids", vocab_size, prompt_ids, res)) return;
        } else {
            if (!body.contains("prompt") || !body["prompt"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "need prompt string or prompt_ids"}}.dump(),
                                "application/json");
                return;
            }
            prompt_ids = ctx.model->encode(body["prompt"].get<std::string>());
            if (prompt_ids.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "empty prompt"}}.dump(), "application/json");
                return;
            }
        }

        std::vector<int> continuation_ids;
        bool boundary_approximate = false;
        if (body.contains("continuation_ids")) {
            if (!exact_token_ids(body["continuation_ids"], "continuation_ids", vocab_size,
                                 continuation_ids, res)) return;
        } else {
            if (!body.contains("continuation") || !body["continuation"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "need continuation string or continuation_ids"}}.dump(),
                                "application/json");
                return;
            }
            continuation_ids = ctx.model->encode(body["continuation"].get<std::string>());
            boundary_approximate = true;
            if (continuation_ids.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "empty continuation"}}.dump(), "application/json");
                return;
            }
        }

        int topk = 0;
        if (body.contains("topk")) {
            if (!body["topk"].is_number_integer()) {
                res.status = 400;
                res.set_content(json{{"error", "topk must be an integer"}}.dump(), "application/json");
                return;
            }
            topk = body["topk"].get<int>();
            if (topk < 0 || topk > 100) {
                res.status = 400;
                res.set_content(json{{"error", "topk must be between 0 and 100"}}.dump(),
                                "application/json");
                return;
            }
        }
        if (body.contains("use_prefix_cache") && !body["use_prefix_cache"].is_boolean()) {
            res.status = 400;
            res.set_content(json{{"error", "use_prefix_cache must be a boolean"}}.dump(),
                            "application/json");
            return;
        }
        const bool use_prefix_cache = body.value("use_prefix_cache", true);

        const int n_prompt = static_cast<int>(prompt_ids.size());
        const int n_continuation = static_cast<int>(continuation_ids.size());
        if (n_prompt + n_continuation > ctx.n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "prompt + continuation exceeds n_ctx"},
                                 {"n_prompt", n_prompt}, {"n_cont", n_continuation},
                                 {"n_ctx", ctx.n_ctx}}.dump(), "application/json");
            return;
        }

        std::vector<int> tokens = prompt_ids;
        tokens.insert(tokens.end(), continuation_ids.begin(), continuation_ids.end());
        std::vector<int> logits_for(static_cast<std::size_t>(n_continuation));
        for (int index = 0; index < n_continuation; ++index) {
            logits_for[static_cast<std::size_t>(index)] = n_prompt - 1 + index;
        }

        try {
            ForwardResult forward;
            ScorePrefixPlan prefix_plan = plan_score_prefix_reuse(
                static_cast<int>(tokens.size()), 0, logits_for.front());
            {
                ContextPool::Lease lease = use_prefix_cache
                    ? ctx.pool.acquire_score(tokens) : ctx.pool.acquire();
                GgmlAdapter& adapter = *lease;
                adapter.set_emit_activations(false);
                const int cached_prefix = use_prefix_cache ? lease.cached_score_prefix() : 0;
                prefix_plan = plan_score_prefix_reuse(
                    static_cast<int>(tokens.size()), cached_prefix, logits_for.front());
                forward = adapter.ar_forward_score(tokens, logits_for, cached_prefix);
                if (adapter.last_score_cached_tokens() != prefix_plan.cached_tokens) {
                    prefix_plan.cached_tokens = adapter.last_score_cached_tokens();
                    prefix_plan.evaluated_tokens =
                        static_cast<int>(tokens.size()) - prefix_plan.cached_tokens;
                }
                if (use_prefix_cache) lease.commit_score_cache(tokens);
            }

            json scored = json::array();
            double sum_logprob = 0.0;
            for (int row_index = 0; row_index < n_continuation; ++row_index) {
                const float* row = forward.row(row_index);
                float maximum = row[0];
                for (int token = 1; token < forward.vocab; ++token) {
                    maximum = std::max(maximum, row[token]);
                }
                float sum_exp = 0.0f;
                for (int token = 0; token < forward.vocab; ++token) {
                    sum_exp += std::exp(row[token] - maximum);
                }
                const float log_normalizer = maximum + std::log(sum_exp);
                const int actual = continuation_ids[static_cast<std::size_t>(row_index)];
                const float logprob = row[actual] - log_normalizer;
                sum_logprob += logprob;

                json item{{"id", actual}, {"piece", ctx.model->decode({actual})},
                          {"logprob", logprob}};
                if (topk > 0) {
                    std::vector<int> indices(static_cast<std::size_t>(forward.vocab));
                    for (int token = 0; token < forward.vocab; ++token) {
                        indices[static_cast<std::size_t>(token)] = token;
                    }
                    const int count = std::min(topk, forward.vocab);
                    std::partial_sort(indices.begin(), indices.begin() + count, indices.end(),
                                      [&](int left, int right) { return row[left] > row[right]; });
                    json candidates = json::array();
                    for (int rank = 0; rank < count; ++rank) {
                        const int token = indices[static_cast<std::size_t>(rank)];
                        candidates.push_back({{"id", token},
                                              {"piece", ctx.model->decode({token})},
                                              {"logprob", row[token] - log_normalizer}});
                    }
                    item["topk"] = std::move(candidates);
                }
                scored.push_back(std::move(item));
            }

            json response{{"n_prompt", n_prompt}, {"n_cont", n_continuation},
                          {"prompt_ids", prompt_ids}, {"tokens", std::move(scored)},
                          {"sum_logprob", sum_logprob},
                          {"cache", {{"matched_tokens", prefix_plan.matched_tokens},
                                     {"cached_tokens", prefix_plan.cached_tokens},
                                     {"evaluated_tokens", prefix_plan.evaluated_tokens}}},
                          {"numerical_regime", prefix_plan.cached_tokens > 0
                              ? "scalar_prefix_cached_v1" : "scalar_reference"}};
            if (boundary_approximate) response["boundary_approximate"] = true;
            res.set_content(dump_json(response), "application/json");
        } catch (const std::exception& error) {
            res.status = 400;
            res.set_content(json{{"error", error.what()}}.dump(), "application/json");
        }
    });

    // Experimental native-many teacher-forced scoring. Every arm has an independent rendered
    // prompt and all arms score the same recorded continuation. Exact shared token prefixes are
    // decoded once through a trie; subsequent forced continuation tokens advance every arm in one
    // native batch. The scalar /score route remains the proof oracle and safe fallback.
    svr.Post("/v1/teacher-forced-score-many", [&](const httplib::Request& req,
                                                   httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "request body must be a JSON object"}}.dump(),
                            "application/json");
            return;
        }
        static const std::set<std::string> allowed{
            "arms", "continuation_ids", "topk", "reduction",
        };
        if (reject_unknown_fields(body, allowed, res)) return;
        if (!body.contains("arms") || !body["arms"].is_array() || body["arms"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "arms must be a non-empty array"}}.dump(),
                            "application/json");
            return;
        }
        if (body["arms"].size() > 16) {
            res.status = 400;
            res.set_content(json{{"error", "arms exceeds the native limit of 16"}}.dump(),
                            "application/json");
            return;
        }
        if (body.contains("topk") &&
                (!body["topk"].is_number_integer() || body["topk"].get<int>() != 0)) {
            res.status = 400;
            res.set_content(
                json{{"error", "native teacher-forced score-many supports only topk=0"}}.dump(),
                "application/json");
            return;
        }
        const std::string reduction = body.value("reduction", std::string("backend_target"));
        if (reduction != "backend_target" && reduction != "host_full_vocab") {
            res.status = 400;
            res.set_content(
                json{{"error", "reduction must be backend_target or host_full_vocab"}}.dump(),
                "application/json");
            return;
        }
        const bool use_backend_reduction = reduction == "backend_target";

        const int vocab_size = ctx.model->config().vocab_size;
        std::vector<int> continuation_ids;
        if (!body.contains("continuation_ids") ||
                !exact_token_ids(body["continuation_ids"], "continuation_ids", vocab_size,
                                 continuation_ids, res)) {
            if (!body.contains("continuation_ids")) {
                res.status = 400;
                res.set_content(
                    json{{"error", "continuation_ids is required"}}.dump(),
                    "application/json");
            }
            return;
        }

        std::vector<std::vector<int>> prompts;
        prompts.reserve(body["arms"].size());
        for (std::size_t arm_index = 0; arm_index < body["arms"].size(); ++arm_index) {
            const json& arm = body["arms"][arm_index];
            if (!arm.is_object()) {
                res.status = 400;
                res.set_content(json{{"error", "each arm must be an object"},
                                     {"arm_index", arm_index}}.dump(), "application/json");
                return;
            }
            static const std::set<std::string> arm_allowed{"prompt", "prompt_ids"};
            if (reject_unknown_fields(arm, arm_allowed, res)) return;
            if (arm.contains("prompt") == arm.contains("prompt_ids")) {
                res.status = 400;
                res.set_content(
                    json{{"error", "each arm needs exactly one of prompt or prompt_ids"},
                         {"arm_index", arm_index}}.dump(),
                    "application/json");
                return;
            }

            std::vector<int> prompt_ids;
            if (arm.contains("prompt_ids")) {
                if (!exact_token_ids(arm["prompt_ids"], "prompt_ids", vocab_size,
                                     prompt_ids, res)) return;
            } else {
                if (!arm["prompt"].is_string() || arm["prompt"].get<std::string>().empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "arm prompt must be a non-empty string"},
                                         {"arm_index", arm_index}}.dump(),
                                    "application/json");
                    return;
                }
                prompt_ids = ctx.model->encode(arm["prompt"].get<std::string>());
                if (prompt_ids.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "arm prompt encoded to no tokens"},
                                         {"arm_index", arm_index}}.dump(),
                                    "application/json");
                    return;
                }
            }
            prompts.push_back(std::move(prompt_ids));
        }

        TeacherForcedScoreManyPlan plan;
        try {
            plan = plan_teacher_forced_score_many(
                prompts, continuation_ids.size(), ctx.n_ctx);
        } catch (const std::exception& error) {
            res.status = 400;
            res.set_content(json{{"error", error.what()}, {"fallback_safe", true}}.dump(),
                            "application/json");
            return;
        }

        const auto started = std::chrono::steady_clock::now();
        try {
            ContextPool::Lease lease = ctx.pool.acquire();
            GgmlAdapter& adapter = *lease;
            adapter.set_emit_activations(false);
            NativeScoreManyScope native_scope(
                adapter, plan.arm_count, use_backend_reduction);

            std::vector<std::vector<double>> logprobs(
                static_cast<std::size_t>(plan.arm_count),
                std::vector<double>(continuation_ids.size(), 0.0));
            bool full_vocab_host_reduction = false;
            long long host_output_floats = 0;
            long long decode_calls = 0;

            TargetLogprobBatchResult first =
                adapter.ar_forward_prefill_batch_target_logprobs(
                    prompts, continuation_ids.front());
            decode_calls += adapter.last_decode_call_count();
            full_vocab_host_reduction |= first.full_vocab_host_reduction;
            host_output_floats += first.host_output_floats;
            for (int arm = 0; arm < plan.arm_count; ++arm)
                logprobs[static_cast<std::size_t>(arm)][0] =
                    first.target_logprobs[static_cast<std::size_t>(arm)];

            const long long measured_prompt_rows = adapter.last_prefill_physical_rows();
            const long long measured_reused_rows = adapter.last_prefill_reused_rows();
            std::vector<int> tokens_per_arm(
                static_cast<std::size_t>(plan.arm_count), continuation_ids.front());
            std::vector<int> positions(static_cast<std::size_t>(plan.arm_count), 0);
            const std::vector<bool> active(static_cast<std::size_t>(plan.arm_count), true);
            for (std::size_t token_index = 1;
                    token_index < continuation_ids.size(); ++token_index) {
                std::fill(tokens_per_arm.begin(), tokens_per_arm.end(),
                          continuation_ids[token_index - 1]);
                for (int arm = 0; arm < plan.arm_count; ++arm) {
                    positions[static_cast<std::size_t>(arm)] =
                        static_cast<int>(prompts[static_cast<std::size_t>(arm)].size()) +
                        static_cast<int>(token_index) - 1;
                }
                TargetLogprobBatchResult step =
                    adapter.ar_forward_batch_at_target_logprobs(
                        tokens_per_arm, positions, active,
                        continuation_ids[token_index]);
                decode_calls += adapter.last_decode_call_count();
                full_vocab_host_reduction |= step.full_vocab_host_reduction;
                host_output_floats += step.host_output_floats;
                for (int arm = 0; arm < plan.arm_count; ++arm)
                    logprobs[static_cast<std::size_t>(arm)][token_index] =
                        step.target_logprobs[static_cast<std::size_t>(arm)];
            }

            json results = json::array();
            for (int arm = 0; arm < plan.arm_count; ++arm) {
                json scored = json::array();
                double sum_logprob = 0.0;
                for (std::size_t token_index = 0;
                        token_index < continuation_ids.size(); ++token_index) {
                    const int token = continuation_ids[token_index];
                    const double logprob =
                        logprobs[static_cast<std::size_t>(arm)][token_index];
                    if (!std::isfinite(logprob))
                        throw std::runtime_error("native score-many produced a non-finite logprob");
                    sum_logprob += logprob;
                    scored.push_back({
                        {"id", token},
                        {"piece", ctx.model->decode({token})},
                        {"logprob", logprob},
                    });
                }
                results.push_back({
                    {"n_prompt", static_cast<int>(prompts[static_cast<std::size_t>(arm)].size())},
                    {"n_cont", static_cast<int>(continuation_ids.size())},
                    {"tokens", std::move(scored)},
                    {"sum_logprob", sum_logprob},
                });
            }

            const auto elapsed = std::chrono::steady_clock::now() - started;
            const double elapsed_ms =
                std::chrono::duration<double, std::milli>(elapsed).count();
            json execution{
                {"kind", "clozn.teacher-forced-score-many.native-prefix-trie.v1"},
                {"arm_count", plan.arm_count},
                {"logical_prompt_rows", plan.logical_prompt_rows},
                {"physical_prompt_rows", measured_prompt_rows},
                {"prefix_rows_reused", measured_reused_rows},
                {"continuation_rows", plan.continuation_rows},
                {"total_physical_rows", measured_prompt_rows + plan.continuation_rows},
                {"decode_calls", decode_calls},
                {"host_output_floats", host_output_floats},
                {"full_vocab_host_reduction", full_vocab_host_reduction},
                {"target_reduction", reduction},
                {"elapsed_ms", elapsed_ms},
                {"fallback", "/score"},
            };
            res.set_content(dump_json({
                {"results", std::move(results)},
                {"numerical_regime", use_backend_reduction
                    ? "native_prefix_trie_target_logprob_v1"
                    : "native_prefix_trie_host_float32_v1"},
                {"qualified", false},
                {"execution", std::move(execution)},
            }), "application/json");
        } catch (const std::exception& error) {
            // Native execution is an optional optimization. Any unsupported backend/model detail,
            // numerical underflow, or capacity failure is safe for the caller to retry arm-by-arm.
            res.status = 409;
            res.set_content(json{{"error", error.what()},
                                 {"code", "native_score_many_unavailable"},
                                 {"fallback_safe", true}}.dump(),
                            "application/json");
        }
    });

    // Prompt rendering remains model-owned and uses the embedded GGUF chat template.
    svr.Post("/apply_template", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "request body must be a JSON object"}}.dump(),
                            "application/json");
            return;
        }
        static const std::set<std::string> allowed{
            "messages", "add_assistant", "enable_thinking", "include_token_ids",
        };
        if (reject_unknown_fields(body, allowed, res)) return;
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "messages must be a non-empty array"}}.dump(),
                            "application/json");
            return;
        }
        if (!ctx.chat_templates.available()) {
            res.status = 400;
            res.set_content(json{{"error", "model has no embedded chat template"}}.dump(),
                            "application/json");
            return;
        }

        std::vector<std::pair<std::string, std::string>> messages;
        messages.reserve(body["messages"].size());
        for (const auto& message : body["messages"]) {
            if (!message.is_object() || !message.contains("role") || !message["role"].is_string()
                    || message["role"].get<std::string>().empty()
                    || !message.contains("content") || !message["content"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "each message needs non-empty role and string content"}}.dump(),
                                "application/json");
                return;
            }
            messages.emplace_back(message["role"].get<std::string>(),
                                  message["content"].get<std::string>());
        }

        const bool add_assistant = body.value("add_assistant", true);
        const bool enable_thinking = body.value("enable_thinking", false);
        const bool include_token_ids = body.value("include_token_ids", false);
        try {
            const std::string prompt = ctx.chat_templates.apply(
                messages, add_assistant, enable_thinking);
            const std::vector<int> prompt_token_ids = ctx.model->encode(prompt);
            json response{{"prompt", prompt},
                          {"prompt_tokens", static_cast<int>(prompt_token_ids.size())},
                          {"template_source", "model"}, {"renderer", "jinja"}};
            if (include_token_ids) response["prompt_token_ids"] = prompt_token_ids;
            res.set_content(dump_json(response), "application/json");
        } catch (const std::exception& error) {
            res.status = 400;
            res.set_content(json{{"error", error.what()}}.dump(), "application/json");
        }
    });
}

}  // namespace clozn
