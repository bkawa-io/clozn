// Private v1 scoring and prompt-template routes.
#include "httplib.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

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
            {
                ContextPool::Lease lease = ctx.pool.acquire();
                GgmlAdapter& adapter = *lease;
                adapter.set_causal(true);
                forward = adapter.ar_forward_score(tokens, logits_for);
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
                          {"sum_logprob", sum_logprob}};
            if (boundary_approximate) response["boundary_approximate"] = true;
            res.set_content(dump_json(response), "application/json");
        } catch (const std::exception& error) {
            res.status = 400;
            res.set_content(json{{"error", error.what()}}.dump(), "application/json");
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
            "messages", "add_assistant", "include_token_ids",
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
        const bool include_token_ids = body.value("include_token_ids", false);
        try {
            const std::string prompt = ctx.chat_templates.apply(messages, add_assistant);
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
