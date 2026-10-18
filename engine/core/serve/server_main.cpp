// Private v1 worker: autoregressive generation, scalar scoring, template rendering,
// and exact recorded-reference matching for one GGUF.
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "clozn/generate_ar.hpp"
#include "clozn/model_ggml.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "server_context.hpp"
#include "server_shared.hpp"
#include "worker_identity.hpp"

using json = nlohmann::json;
using namespace clozn;

#ifndef CLOZN_ENGINE_VERSION
#define CLOZN_ENGINE_VERSION "development"
#endif
#ifndef CLOZN_ENGINE_BUILD_ID
#define CLOZN_ENGINE_BUILD_ID "development"
#endif
#ifndef CLOZN_ENGINE_BACKEND
#define CLOZN_ENGINE_BACKEND "cpu"
#endif
#ifndef CLOZN_LLAMA_CPP_COMMIT
#define CLOZN_LLAMA_CPP_COMMIT "0000000000000000000000000000000000000000"
#endif

static json engine_build_info() {
    json feature_flags{
        {"generation", true},
        {"scoring", true},
        {"reference_match", true},
    };
    return {
        {"engine_version", CLOZN_ENGINE_VERSION},
        {"build_id", CLOZN_ENGINE_BUILD_ID},
        {"protocol_version", PROTOCOL_VERSION},
        {"backend", CLOZN_ENGINE_BACKEND},
        {"llama_cpp_commit", CLOZN_LLAMA_CPP_COMMIT},
        {"feature_flags", feature_flags},
    };
}

static bool reject_unknown_fields(const json& body, const std::set<std::string>& allowed,
                                  httplib::Response& response) {
    for (const auto& item : body.items()) {
        if (allowed.count(item.key()) == 0) {
            response.status = 400;
            response.set_content(
                json{{"error", "unsupported v1 completion field: " + item.key()}}.dump(),
                "application/json");
            return true;
        }
    }
    return false;
}

static bool parse_stop_sequences(const json& body, GrammarConfig& grammar,
                                 httplib::Response& response) {
    if (!body.contains("stop") || body["stop"].is_null()) return true;
    const json& values = body["stop"];
    if (!values.is_array() || values.empty() || values.size() > 16) {
        response.status = 400;
        response.set_content(json{{"error", "stop must be an array of 1..16 strings"}}.dump(),
                             "application/json");
        return false;
    }
    for (const auto& value : values) {
        if (!value.is_string() || value.get<std::string>().empty()) {
            response.status = 400;
            response.set_content(json{{"error", "stop sequences must be non-empty strings"}}.dump(),
                                 "application/json");
            return false;
        }
        grammar.additional_stops.push_back(value.get<std::string>());
    }
    return true;
}

static bool parse_reference_tokens(const json& body, int vocab_size,
                                   std::vector<int>& reference,
                                   httplib::Response& response) {
    if (!body.contains("reference_tokens")) return true;
    const json& values = body["reference_tokens"];
    if (!values.is_array() || values.empty()) {
        response.status = 400;
        response.set_content(json{{"error", "reference_tokens must be a non-empty integer array"}}.dump(),
                             "application/json");
        return false;
    }
    reference.reserve(values.size());
    for (const auto& value : values) {
        if (!value.is_number_integer()) {
            response.status = 400;
            response.set_content(json{{"error", "reference_tokens must contain only integers"}}.dump(),
                                 "application/json");
            return false;
        }
        const int token = value.get<int>();
        if (token < 0 || token >= vocab_size) {
            response.status = 400;
            response.set_content(json{{"error", "reference_tokens contains an out-of-range token"}}.dump(),
                                 "application/json");
            return false;
        }
        reference.push_back(token);
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--version") {
        const bool as_json = argc >= 3 && std::string(argv[2]) == "--json";
        const json info = engine_build_info();
        if (as_json) {
            std::printf("%s\n", info.dump().c_str());
        } else {
            std::printf("clozn-server %s (build %s, protocol %s, backend %s, llama.cpp %.12s)\n",
                        CLOZN_ENGINE_VERSION, CLOZN_ENGINE_BUILD_ID, PROTOCOL_VERSION,
                        CLOZN_ENGINE_BACKEND, CLOZN_LLAMA_CPP_COMMIT);
        }
        return 0;
    }
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <model.gguf> [--port N] [--host H] [--gpu-layers N] "
                             "[--eos ID] [--ctx N] [--batch N] [--ubatch N] [--workers N] "
                             "[--model-sha256 HEX] [--tokenizer-sha256 HEX]\n", argv[0]);
        return 1;
    }

    const std::string model_path = argv[1];
    int port = 8080;
    int gpu_layers = 0;
    int eos = -1;
    int n_ctx = 4096;
    int n_batch = 0;
    int n_ubatch = 0;
    int workers = 1;
    std::string host = "127.0.0.1";
    std::string model_sha256;
    std::string tokenizer_sha256;
    for (int index = 2; index < argc; ++index) {
        const std::string option = argv[index];
        auto next = [&]() -> const char* {
            if (index + 1 >= argc) return "";
            return argv[++index];
        };
        if (option == "--port") port = std::atoi(next());
        else if (option == "--host") host = next();
        else if (option == "--gpu-layers") gpu_layers = std::atoi(next());
        else if (option == "--eos") eos = std::atoi(next());
        else if (option == "--ctx") n_ctx = std::atoi(next());
        else if (option == "--batch") n_batch = std::atoi(next());
        else if (option == "--ubatch") n_ubatch = std::atoi(next());
        else if (option == "--workers") workers = std::atoi(next());
        else if (option == "--model-sha256") model_sha256 = next();
        else if (option == "--tokenizer-sha256") tokenizer_sha256 = next();
        else {
            std::fprintf(stderr, "unsupported v1 worker option: %s\n", option.c_str());
            return 1;
        }
    }
    if (port < 1 || port > 65535 || n_ctx < 2 || workers < 1) {
        std::fprintf(stderr, "invalid worker port/context/worker count\n");
        return 1;
    }

    llama_log_set(quiet_log, nullptr);
    using PerfClock = std::chrono::steady_clock;
    auto duration_ns = [](PerfClock::time_point started) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            PerfClock::now() - started).count();
    };
    std::vector<PerformancePhase> startup_timing_phases;

    const PerfClock::time_point model_load_started = PerfClock::now();
    auto model = std::make_shared<GgmlModel>(model_path, /*mask_token=*/-1, eos, gpu_layers);
    startup_timing_phases.push_back(PerformancePhase{
        "model_load", "model_load", duration_ns(model_load_started), -1,
        "context_only", "process_startup"});

    const PerfClock::time_point context_create_started = PerfClock::now();
    ContextPool pool(model, workers, n_ctx, /*flash_attn=*/true, n_batch, n_ubatch);
    startup_timing_phases.push_back(PerformancePhase{
        "context_create", "clozn_worker", duration_ns(context_create_started), -1,
        "context_only", "process_startup", {"kv_allocate"}});

    std::unique_ptr<ChatTemplateRenderer> chat_templates;
    try {
        chat_templates = std::make_unique<ChatTemplateRenderer>(model->handle());
    } catch (const std::exception& error) {
        std::fprintf(stderr, "[clozn-server] embedded chat template is invalid: %s\n", error.what());
        return 2;
    }

    ServerContext context{model, pool, *chat_templates, n_ctx, true, model_path};

    const std::string worker_generation_id = make_worker_generation_id();
    context.worker_generation_id = worker_generation_id;
    context.model_sha256 = model_sha256;
    context.tokenizer_sha256 = tokenizer_sha256;

    int model_n_embd = 0;
    int model_n_layer = 0;
    std::string model_architecture;
    {
        ContextPool::Lease lease = pool.acquire();
        model_n_embd = (*lease).n_embd();
        model_n_layer = (*lease).n_layer();
    }
    {
        char value[128]{};
        if (llama_model_meta_val_str(model->handle(), "general.architecture", value, sizeof(value)) >= 0) {
            model_architecture = value;
        }
    }

    httplib::Server server;
    CancelRegistry cancel_registry;
    std::atomic<bool> startup_timing_claimed{false};

    server.Get("/health", [&](const httplib::Request&, httplib::Response& response) {
        json capabilities{
            {"streaming", true},
            {"sampling", true},
            {"reference_match_arms", true},
            {"persistent_parent_session", true},
            {"performance_timing", true},
        };
        const json build = engine_build_info();
        const std::string backend = build.value("backend", std::string("cpu"));
        json health{{"status", "ok"},
                    {"engine_version", build.value("engine_version", std::string())},
                    {"build_id", build.value("build_id", std::string())},
                    {"backend", backend},
                    {"llama_cpp_commit", build.value("llama_cpp_commit", std::string())},
                    {"protocol_version", PROTOCOL_VERSION},
                    {"worker_generation_id", worker_generation_id},
                    {"capabilities", capabilities},
                    {"model", model_path},
                    {"mode", "autoregressive"},
                    {"architecture", model_architecture},
                    {"model_sha256", model_sha256},
                    {"tokenizer_sha256", tokenizer_sha256},
                    {"n_embd", model_n_embd},
                    {"n_layer", model_n_layer},
                    {"vocab_size", model->config().vocab_size},
                    {"n_ctx", pool.n_ctx()},
                    {"n_batch", pool.n_batch()},
                    {"n_ubatch", pool.n_ubatch()},
                    {"memory", pool.memory_breakdown()},
                    {"gpu_layers", gpu_layers},
                    {"device", gpu_layers > 0 ? backend : "cpu"}};
        response.set_content(health.dump(), "application/json");
    });

    server.Post("/cancel", [&](const httplib::Request& request, httplib::Response& response) {
        const json body = json::parse(request.body, nullptr, /*allow_exceptions=*/false);
        const std::string id = (!body.is_discarded() && body.is_object())
            ? body.value("req", std::string()) : std::string();
        if (id.empty()) {
            response.status = 400;
            response.set_content(json{{"error", "need req string"}}.dump(), "application/json");
            return;
        }
        response.set_content(json{{"cancelled", cancel_registry.cancel(id)}, {"req", id}}.dump(),
                             "application/json");
    });

    register_whitebox_routes(server, context);
    register_reference_match_routes(server, context);

    server.Post("/v1/completions", [&](const httplib::Request& request,
                                        httplib::Response& response) {
        json body = json::parse(request.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            response.status = 400;
            response.set_content(json{{"error", "request body must be a JSON object"}}.dump(),
                                 "application/json");
            return;
        }
        static const std::set<std::string> allowed_fields{
            "prompt", "max_tokens", "temperature", "rep_penalty", "top_k", "top_p",
            "seed", "stream", "reference_tokens", "stop",
        };
        if (reject_unknown_fields(body, allowed_fields, response)) return;
        if (!body.contains("prompt") || !body["prompt"].is_string()
                || body["prompt"].get<std::string>().empty()) {
            response.status = 400;
            response.set_content(json{{"error", "prompt must be a non-empty string"}}.dump(),
                                 "application/json");
            return;
        }

        GenerateConfig config;
        if (body.contains("max_tokens") && !body["max_tokens"].is_number_integer()) {
            response.status = 400;
            response.set_content(json{{"error", "max_tokens must be an integer"}}.dump(),
                                 "application/json");
            return;
        }
        config.max_new = body.value("max_tokens", 32);
        if (config.max_new < 1 || config.max_new > n_ctx) {
            response.status = 400;
            response.set_content(json{{"error", "max_tokens must be in [1, n_ctx]"}}.dump(),
                                 "application/json");
            return;
        }

        SampleConfig sample;
        try {
            sample = sample_from(body);
        } catch (const std::exception& error) {
            response.status = 400;
            response.set_content(json{{"error", std::string("invalid sampling field: ") + error.what()}}.dump(),
                                 "application/json");
            return;
        }
        if (!std::isfinite(sample.temperature) || sample.temperature < 0.0
                || !std::isfinite(sample.rep_penalty) || sample.rep_penalty <= 0.0
                || sample.top_k < 0 || !std::isfinite(sample.top_p)
                || sample.top_p <= 0.0 || sample.top_p > 1.0) {
            response.status = 400;
            response.set_content(json{{"error", "sampling fields are out of range"}}.dump(),
                                 "application/json");
            return;
        }
        if (body.contains("stream") && !body["stream"].is_boolean()) {
            response.status = 400;
            response.set_content(json{{"error", "stream must be a boolean"}}.dump(),
                                 "application/json");
            return;
        }
        const bool stream = body.value("stream", false);

        GrammarConfig stop_grammar;
        if (!parse_stop_sequences(body, stop_grammar, response)) return;
        const GrammarConfig* grammar = stop_grammar.additional_stops.empty() ? nullptr : &stop_grammar;

        std::vector<int> reference_tokens;
        if (!parse_reference_tokens(body, model->config().vocab_size, reference_tokens, response)) return;

        const PerfClock::time_point tokenize_started = PerfClock::now();
        const std::vector<int> prompt_ids = model->encode(body["prompt"].get<std::string>());
        if (prompt_ids.empty()) {
            response.status = 400;
            response.set_content(json{{"error", "prompt tokenized to zero tokens"}}.dump(),
                                 "application/json");
            return;
        }
        if (static_cast<int>(prompt_ids.size()) >= n_ctx) {
            response.status = 400;
            response.set_content(json{{"error", "prompt exceeds the worker context window"}}.dump(),
                                 "application/json");
            return;
        }
        std::vector<PerformancePhase> request_phases{
            PerformancePhase{"tokenize", "clozn_worker", duration_ns(tokenize_started)},
        };

        auto run = [&pool, prompt_ids, config, sample, reference_tokens, grammar,
                    request_phases, &startup_timing_phases, &startup_timing_claimed,
                    duration_ns](const std::function<void(const Event&)>& on_event) {
            std::vector<PerformancePhase> phases = request_phases;
            const PerfClock::time_point worker_start_started = PerfClock::now();
            ContextPool::Lease lease = pool.acquire();
            phases.push_back(PerformancePhase{
                "worker_start", "clozn_worker", duration_ns(worker_start_started)});
            if (!startup_timing_claimed.exchange(true, std::memory_order_relaxed)) {
                phases.insert(phases.end(), startup_timing_phases.begin(), startup_timing_phases.end());
            }
            (*lease).set_causal(true);
            (*lease).set_emit_activations(false);
            return generate_ar(
                *lease, prompt_ids, config, on_event, sample, nullptr,
                reference_tokens.empty() ? nullptr : &reference_tokens,
                grammar, nullptr, -1, nullptr, &phases);
        };

        const bool user_stop_active = grammar != nullptr;
        const std::string id = make_id("cmpl-");
        if (stream) {
            response.set_chunked_content_provider(
                "text/event-stream",
                [run, id, model, prompt_ids, &cancel_registry, user_stop_active]
                (size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& value) {
                        return sink.write(value.data(), value.size());
                    };
                    StreamEnvelope envelope{id, write};
                    envelope.cancel = cancel_registry.register_request(id);
                    CancelRegistry::Guard cancel_guard{cancel_registry, id};
                    try {
                        auto on_event = [&](const Event& event) {
                            envelope.check_cancelled();
                            envelope.frame_str(sse_data(event, *model, prompt_ids, {}));
                        };
                        const GenerateResult result = run(on_event);
                        json final_frame{{"id", id}, {"object", "text_completion"},
                                         {"choices", json::array({{{"text", result.text}, {"index", 0},
                                              {"finish_reason", finish_reason(result.reason)}}})},
                                         {"usage", {{"prompt_tokens", static_cast<int>(prompt_ids.size())},
                                                    {"completion_tokens", result.new_tokens}}},
                                         {"termination", {{"kind", user_stop_active && result.reason == "stop"
                                                  ? "user_stop_sequence" : result.reason}}}};
                        if (result.ref_active) {
                            final_frame["diverged"] = result.diverged;
                            final_frame["diverged_at"] = result.diverged_at;
                        }
                        envelope.frame(std::move(final_frame));
                        envelope.done();
                    } catch (const GenerationCancelled&) {
                        envelope.frame(json{{"cancelled", true}, {"id", id}});
                        envelope.done();
                    } catch (const std::exception& error) {
                        envelope.frame(json{{"error", std::string("generation failed: ") + error.what()}});
                        envelope.done();
                    } catch (...) {
                        envelope.frame(json{{"error", "generation failed"}});
                        envelope.done();
                    }
                    sink.done();
                    return true;
                });
            return;
        }

        try {
            const GenerateResult result = run({});
            json payload{{"id", id}, {"object", "text_completion"},
                         {"choices", json::array({{{"text", result.text}, {"index", 0},
                              {"finish_reason", finish_reason(result.reason)}}})},
                         {"usage", {{"prompt_tokens", static_cast<int>(prompt_ids.size())},
                                    {"completion_tokens", result.new_tokens},
                                    {"steps_total", result.steps_total}}},
                         {"termination", {{"kind", user_stop_active && result.reason == "stop"
                                  ? "user_stop_sequence" : result.reason}}}};
            if (result.ref_active) {
                payload["diverged"] = result.diverged;
                payload["diverged_at"] = result.diverged_at;
            }
            response.set_content(dump_json(payload), "application/json");
        } catch (const std::exception& error) {
            response.status = 400;
            response.set_content(
                dump_json(json{{"error", std::string("generation failed: ") + error.what()}}),
                "application/json");
        }
    });

    std::fprintf(stderr, "[clozn-server] autoregressive %s, %d worker(s) on http://%s:%d "
                         "(model: %s)\n",
                 gpu_layers > 0 ? "GPU" : "CPU", pool.size(), host.c_str(), port,
                 model_path.c_str());
    if (!server.listen(host, port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
