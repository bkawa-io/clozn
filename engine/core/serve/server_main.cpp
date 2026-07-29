// clozn_server.cpp — the L2 serving layer: an HTTP server over the clozn runtime. Loads one GGUF
// diffusion LM and serves completions + infill, with SSE streaming that emits the §5.1 event spine
// directly (the native streaming protocol the events were designed for). Uses the single-header
// cpp-httplib + nlohmann/json that llama.cpp already vendors — no new dependencies.
//
//   clozn-server <model.gguf> [--port N] [--host H] [--gpu-layers N] [--diffusion --mask-token ID]
//
// Endpoints:
//   GET  /health           -> {"status":"ok","model":...}
//   POST /v1/completions    {prompt, max_tokens, steps, block_len, topk, cache, stream}
//                          -> OpenAI-ish {choices:[{text,finish_reason}], usage}; stream=true => SSE
//   POST /v1/infill         {prefix, suffix, gap, steps, topk, stream} -> fill-in-the-middle
//
// The GgmlAdapter wraps one stateful llama context, so a mutex serializes generation; HTTP I/O is
// concurrent, generation is one-at-a-time (correct for a single context). cpp-httplib is the split
// build (httplib.cpp compiled into this target by CMake supplies the out-of-line definitions).
#include "httplib.h"
#include "nlohmann/json.hpp"

#include "clozn/events.hpp"
#include "clozn/generate.hpp"
#include "clozn/generate_ar.hpp"
#include "clozn/model_ggml.hpp"
#ifdef CLOZN_SAE
#include "clozn/sae.hpp"  // on-device SAE feature readout (--sae; built with CLOZN_BUILD_SAE)
#endif
#include "readout_plane.hpp"  // Phase 2.3: multi-observer readout plane (includes server_shared.hpp)
#include "viz_html.hpp"

#include "ggml.h"       // J-lens: standalone CPU ggml graph (J_l @ h -> rms_norm -> head)
#include "ggml-cpu.h"   // ggml_graph_compute_with_ctx
#include "gguf.h"       // read the GGUF's own output_norm.weight + output.weight for the J-lens head

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "server_shared.hpp"   // helpers + white-box state structs (was the anonymous namespace)
#include "server_context.hpp"  // ServerContext + register_*_routes declarations

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
        {"jlens", true},
        {"lora", true},
        {"native_chat_io", true},
        {"whitebox", true},
#ifdef CLOZN_SAE
        {"sae", true},
#else
        {"sae", false},
#endif
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

int main(int argc, char** argv) {
    // Build identity is deliberately available before model parsing/loading. `clozn setup` executes
    // this model-free path inside staging and refuses promotion unless it agrees with the release
    // manifest. Human output stays useful for direct diagnostics; --json is the machine contract.
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
                             "[--ar | --diffusion] [--mask-token ID] [--eos ID] [--ctx N] [--workers N] "
                             "[--lora PATH] [--lora-scale F] "
                             "[--sae <dir>] [--sae-k N] [--jlens <dir>] [--model-sha256 HEX] "
                             "[--no-flash-attn]\n", argv[0]);
        return 1;
    }
    const std::string model_path = argv[1];
    int port = 8080, gpu_layers = 0, mask_token = 151665, eos = -1, n_ctx = 4096, workers = 1;
    std::string host = "127.0.0.1";
    std::string sae_dir;  // --sae: exported SAE weight dir (tools/export_sae_weights.py); off by default
    int sae_k = 16;
    std::string jlens_dir;  // --jlens: already validated/model-scoped by the product supervisor
    std::string model_sha256;
    bool force_ar = false;
    bool force_diffusion = false;
    // --no-flash-attn: build contexts with FA disabled so "kq_soft_max-<il>" materializes and
    // /score's attn_knockout can cut individual query->key attention edges. Costs decode speed,
    // so it stays opt-in; the /health capability reports which mode is live.
    bool flash_attn = true;
    // --lora: a fine-tune adapter GGUF applied over the base model's weights. Distinct from the
    // steering control vector (--steer paths / set_steer) in both math and mechanism; see
    // docs/ENGINE_ADAPTER_SUPPORT.md. --lora-scale multiplies the delta; 0.0 attaches the adapter but
    // contributes nothing, which is the identity control for "did the adapter actually change this?"
    std::string lora_path;
    float lora_scale = 1.0f;
    for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--port") port = std::atoi(next());
        else if (a == "--host") host = next();
        else if (a == "--gpu-layers") gpu_layers = std::atoi(next());
        else if (a == "--mask-token") { mask_token = std::atoi(next()); force_diffusion = true; }
        else if (a == "--eos") eos = std::atoi(next());
        else if (a == "--ctx") n_ctx = std::atoi(next());
        else if (a == "--workers") workers = std::atoi(next());
        else if (a == "--sae") sae_dir = next();
        else if (a == "--sae-k") sae_k = std::atoi(next());
        else if (a == "--jlens") jlens_dir = next();
        else if (a == "--model-sha256") model_sha256 = next();
        else if (a == "--ar") force_ar = true;
        else if (a == "--diffusion") force_diffusion = true;
        else if (a == "--no-flash-attn") flash_attn = false;
        else if (a == "--lora") lora_path = next();
        else if (a == "--lora-scale") lora_scale = static_cast<float>(std::atof(next()));
    }
    if (force_ar && force_diffusion) {
        std::fprintf(stderr, "--ar and --diffusion are mutually exclusive\n");
        return 1;
    }
    if (workers < 1) workers = 1;
    if (sae_k < 1) sae_k = 1;
    llama_log_set(quiet_log, nullptr);
    using PerfClock = std::chrono::steady_clock;
    auto duration_ns = [](PerfClock::time_point started) {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            PerfClock::now() - started).count();
    };
    std::vector<PerformancePhase> startup_timing_phases;

    // One copy of the weights, N contexts over it — concurrent requests, one model in (V)RAM.
    const PerfClock::time_point model_load_started = PerfClock::now();
    auto model = std::make_shared<GgmlModel>(model_path, mask_token, eos, gpu_layers);
    startup_timing_phases.push_back(PerformancePhase{
        "model_load", "model_load", duration_ns(model_load_started), -1,
        "context_only", "process_startup"});
    const PerfClock::time_point context_create_started = PerfClock::now();
    ContextPool pool(model, workers, n_ctx, flash_attn);
    startup_timing_phases.push_back(PerformancePhase{
        "context_create", "clozn_worker", duration_ns(context_create_started), -1,
        "context_only", "process_startup", {"kv_allocate"}});

    // A requested adapter that fails to attach ABORTS the worker rather than serving the base model.
    // Roadmap rule 3 (no silent fallback) is at its sharpest here: someone evaluating a fine-tune who
    // silently gets base-model output would draw a conclusion about weights that were never loaded --
    // a wrong answer that looks exactly like a right one, and one no downstream receipt could catch.
    if (!lora_path.empty()) {
        const PerfClock::time_point adapter_attach_started = PerfClock::now();
        std::string lora_err;
        if (!pool.attach_lora(lora_path, lora_scale, &lora_err)) {
            std::fprintf(stderr, "clozn-server: %s\n", lora_err.c_str());
            return 1;
        }
        std::fprintf(stderr, "clozn-server: LoRA attached: %s (scale %.3f)\n",
                     lora_path.c_str(), static_cast<double>(lora_scale));
        startup_timing_phases.push_back(PerformancePhase{
            "adapter_attach", "clozn_worker", duration_ns(adapter_attach_started), -1,
            "context_only", "process_startup"});
    }
    if (!flash_attn)
        std::fprintf(stderr, "[clozn-server] flash attention DISABLED: attention weights "
                             "materialize, /score attn_knockout available (slower decode)\n");

    // --sae: load the on-device SAE encoder BEFORE probe calibration (the read tap must move to the
    // SAE's layer so the probes calibrate in the space they'll project). Refusals are hard errors —
    // a server that silently dropped the requested readout would be lying about what it emits.
    SaeServe sae_serve;
    sae_serve.k = sae_k;
    if (!sae_dir.empty()) {
#ifdef CLOZN_SAE
        if (!sae_serve.enc.load(sae_dir)) {
            std::fprintf(stderr, "[clozn-server] --sae load failed: %s\n", sae_serve.enc.error().c_str());
            return 1;
        }
        sae_serve.layer = sae_serve.enc.layer();
        sae_serve.on = true;  // n_embd-vs-d_in verified against the model below (calibration block)
        std::fprintf(stderr, "[clozn-server] SAE ready: %d features, d_in %d, layer %d, k %d (%.0f MB device)\n",
                     sae_serve.enc.d_sae(), sae_serve.enc.d_in(), sae_serve.layer, sae_serve.k,
                     sae_serve.enc.device_bytes() / 1e6);
#else
        std::fprintf(stderr, "[clozn-server] --sae requires a server built with -DCLOZN_BUILD_SAE=ON "
                             "(this binary has no CUDA SAE encoder)\n");
        return 1;
#endif
    }

    // Mode is explicit. A mask token is not proof of diffusion: ordinary AR checkpoints can carry
    // mask/tool/multimodal control tokens (Gemma 4 does). Product launches mark the bounded known
    // diffusion roster with --diffusion; every other GGUF safely defaults to causal AR.
    const bool ar_mode = force_ar || !force_diffusion;

    // White-box Tier 2: build concept probes once, in the model's own activation space (a temp
    // context with the tap on; ~a few CPU forwards). Passed to generate when a request asks for
    // features. If calibration yields nothing, those requests fall back to the raw "norm" feature.
    // Two probe sets, decoupling READ from WRITE (the read/write tap tension the layer sweep
    // exposed): a diff-in-means direction only steers at the layer it was calibrated in (the
    // residual basis rotates across depth), but per-token concepts read SHARPEST at an early
    // layer (sweep: layer 2 ~18% above 2/3-depth). So concept_probes reads at the adapter's
    // default early tap (display); steer_probes recalibrates at mid-depth (2/3) for an effective
    // control vector. ~2x the startup forwards (a few CPU passes) and one extra K*n_embd buffer.
    ConceptProbes concept_probes;  // READ: sharp early-layer tap, drives the white-box display
    ConceptProbes steer_probes;    // WRITE: mid-depth tap, drives the steering control vector
    {
        std::fprintf(stderr, "[clozn-server] calibrating white-box concept probes (%s)...\n",
                     ar_mode ? "causal/AR" : "bidirectional/diffusion");
        GgmlAdapter cal(model, 512);
        // AR models read activations under CAUSAL attention; calibrate the probe directions in the
        // same regime they'll be projected/steered in, or the diff-in-means directions won't transfer
        // (a token's hidden state differs bidirectional vs causal). forward() under causal mode returns
        // the per-token causal hidden states the AR tap also sees.
        if (ar_mode) cal.set_causal(true);
        cal.set_emit_activations(true);
#ifdef CLOZN_SAE
        if (sae_serve.on) {
            // The SAE only reads the residual space it was trained on: n_embd must equal d_in (the
            // cached andyrdt L15 SAE is Qwen2.5-7B-Instruct's 3584 — a Llama-1B GGUF can't serve it).
            if (cal.n_embd() != sae_serve.enc.d_in()) {
                std::fprintf(stderr, "[clozn-server] --sae mismatch: model n_embd %d != SAE d_in %d "
                                     "(this SAE targets Qwen2.5-7B-Instruct layer %d residuals)\n",
                             cal.n_embd(), sae_serve.enc.d_in(), sae_serve.layer);
                return 1;
            }
            // Featureful requests will tap at the SAE's layer, so the concept probes must calibrate
            // THERE (a diff-in-means direction only reads in the space it was calibrated in). Costs
            // the early-tap sharpness; keeping the display honest matters more than the sharpness.
            cal.set_tap_layer(sae_serve.layer);
            std::fprintf(stderr, "[clozn-server] --sae active: read tap + concept probes move to layer %d\n",
                         sae_serve.layer);
        }
#endif
        const int read_tap = cal.tap_layer();                   // default early tap (SAE layer when --sae)
        concept_probes = calibrate_concept_probes(cal, *model);
        const int steer_tap = cal.n_layer() * 2 / 3;
        cal.set_tap_layer(steer_tap);
        steer_probes = calibrate_concept_probes(cal, *model);    // mid-depth (steer-effective)
        std::fprintf(stderr, "[clozn-server] %d concept probe(s) ready:", concept_probes.size());
        for (const auto& nm : concept_probes.names) std::fprintf(stderr, " %s", nm.c_str());
        std::fprintf(stderr, " (read tap %d, steer tap %d)\n", read_tap, steer_tap);
    }

    // J-lens (JLENS_ENGINE_PLAN.md J2): load the fitted J_l sidecars + the GGUF's own final_norm/head once,
    // so POST /jlens can read each position's "disposed to say" tokens. Off (route 400s) if the dir is
    // absent/incomplete -- a research feature, never required for chat/harvest/score to serve.
    JlensServe jlens;
    if (!jlens_dir.empty()) {
        int jl_n_embd = 0;
        { ContextPool::Lease lease = pool.acquire(); jl_n_embd = (*lease).n_embd(); }
        if (jlens.load(jlens_dir, model_path, jl_n_embd, model->config().vocab_size)) {
            std::fprintf(stderr, "[clozn-server] J-lens ready: d_model %d, vocab %d, eps %.1e, layers",
                         jlens.d_model, jlens.vocab, jlens.eps);
            for (int L : jlens.layers()) std::fprintf(stderr, " %d", L);
            std::fprintf(stderr, " (default %d), compute %s, from %s\n", jlens.default_layer,
                         jlens.dev_backend ? "DEVICE (weights resident)" : "cpu (thread-capped fallback)",
                         jlens_dir.c_str());
        } else {
            std::fprintf(stderr, "[clozn-server] J-lens off: %s\n", jlens.err.c_str());
        }
    }

    // Parse the model's embedded Jinja template once. The old llama_chat_apply_template path only
    // recognizes a fixed family list and rejects newer valid GGUFs such as Gemma 4. llama-common's
    // renderer executes the actual checked-in tokenizer.chat_template and also owns BOS/EOS policy.
    std::unique_ptr<ChatTemplateRenderer> chat_templates;
    try {
        chat_templates = std::make_unique<ChatTemplateRenderer>(model->handle());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "[clozn-server] embedded chat template is invalid: %s\n", e.what());
        return 2;
    }

    // Phase 12.4: the shared state the relocated route families read (instead of [&]-capturing
    // main locals). Reference members alias these locals; the model rides as a shared_ptr.
    ServerContext ctx{model, pool, concept_probes, steer_probes, sae_serve, jlens, *chat_templates,
                       n_ctx, mask_token, ar_mode, gpu_layers, model_path};

    httplib::Server svr;

    // Checkpoint store: named snapshots of the KV cache + token sequence, keyed by a server-assigned
    // id. Saved via POST /v1/checkpoint, restored via POST /v1/restore, forked via POST /v1/branch.
    // In-memory only (no persistence across restarts); a cap prevents unbounded growth.
    std::mutex ckpt_mtx;
    std::map<std::string, EngineCheckpoint> checkpoints;
    static constexpr int kMaxCheckpoints = 16;

    // Cooperative cancellation (see server_shared.hpp's CancelRegistry docstring): shared by every
    // SSE streaming route below so POST /cancel can reach a generation running on another worker
    // thread, and so a dead client socket is noticed the next time that request tries to write a frame.
    CancelRegistry cancel_registry;
    std::atomic<bool> startup_timing_claimed{false};

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
        if (llama_model_meta_val_str(model->handle(), "general.architecture", value, sizeof(value)) >= 0)
            model_architecture = value;
    }

    svr.Get("/health", [&](const httplib::Request&, httplib::Response& res) {
        // capabilities: the stable feature flags a client/supervisor negotiates on. Model-shape fields
        // (n_embd/n_layer/...) stay top-level as repro metadata; the sae/jlens detail blocks below carry
        // dimensions, while capabilities carries only the booleans. Keep the KEYS in lockstep with
        // protocol/fixtures/handshake.json (the golden-fixture test guards this).
        bool sae_on = false;
#ifdef CLOZN_SAE
        sae_on = sae_serve.on;
#endif
        // Hoisted out of the capabilities literal on purpose: tests/test_protocol_handshake.py pins
        // the capability vocabulary by regex-extracting `{"key",` pairs from the literal below, so a
        // nested json::array({"a","b"}) inside it would read its ELEMENTS as capability names. Keep
        // every non-scalar capability value in a named variable so that guard stays honest.
        json execution_fork_regimes = json::array({"live_kv_truncated", "reprefill"});
        json capabilities{
            {"streaming", true},          // SSE event stream on stream:true
            {"state_stream", true},       // protocol:true reshapes frames to StateStep
            {"sampling", true},           // temperature/top_k/top_p/rep_penalty/seed
            {"steering", true},           // steer:{concept,coef,layer} control vector
            {"infill", false},            // /v1/infill (fill-in-the-middle) was diffusion-only; the
                                          // diffusion generator was removed (THE_CUT) -- always off now
            {"revise", false},            // /v1/revise was diffusion-only; the route itself was removed
            {"sae", sae_on},              // on-device SAE feature readout (--sae build)
            {"jlens", jlens.on},          // live Jacobian-lens "disposed to say" readout
            {"readout", true},            // Phase 2.3 multi-observer readout plane (readout:{...})
            {"attn_knockout", !flash_attn},  // /score attn_knockout (needs --no-flash-attn)
            {"lora", true},               // this build can attach a fine-tune adapter (--lora). A
                                          // CAPABILITY, not a state: whether one is currently attached
                                          // is the separate "lora" object below, present only when it
                                          // is. A client needs both facts, and a build that predates
                                          // adapters omits this key entirely rather than reporting
                                          // false -- so absent means "doesn't know about adapters",
                                          // false would have meant "knows, and declines".
            {"score_arms", true},         // /score arms: batched multi-arm scoring -- APPROXIMATE
                                          // regime (screening only; response carries the label)
            {"checkpoint", true},          // POST /v1/checkpoint (save) + /v1/restore + /v1/branch
                                          // (KV-cache save/resume/fork)
            {"execution_fork", true},      // POST /v1/execution-fork: intervention forks from a
                                          // checkpoint at an arbitrary EARLIER point than its own
                                          // n_past (force_token/sampling/steer/residual_write),
                                          // labeled with the regime that made it exact
            {"execution_fork_regimes", execution_fork_regimes},
            {"performance_timing", true}, // versioned optional timing object on gen_finished
        };
        json h{{"status", "ok"},
               {"protocol_version", PROTOCOL_VERSION},        // worker <-> supervisor wire contract
               {"capabilities", capabilities},
               {"model", model_path},
               {"mode", ar_mode ? "autoregressive" : "diffusion"},
               {"architecture", model_architecture},
               {"model_sha256", model_sha256},
               {"n_embd", model_n_embd},
               {"n_layer", model_n_layer},
               {"vocab_size", model->config().vocab_size},
               {"n_ctx", n_ctx},                              // configured context window (repro metadata)
               {"gpu_layers", gpu_layers},                    // layers offloaded to the GPU (0 => CPU-resident)
               {"device", gpu_layers > 0 ? "cuda" : "cpu"}};  // CUDA build; device follows the offload setting
        // The ATTACHED adapter, present only when one actually is -- omitted, never null-padded, so a
        // reader cannot mistake "no adapter" for "adapter whose fields we failed to read". The meta map
        // is whatever the adapter GGUF itself declares (rank, alpha, target modules), read back off the
        // loaded file rather than inferred, so a run receipt records what was really applied.
        if (pool.lora_loaded()) {
            h["lora"] = {{"path", pool.lora_path()},
                         {"scale", pool.lora_scale()},
                         {"meta", pool.lora_meta()}};
        }
        h["native_chat_io"] = {
            {"available", ar_mode && chat_templates->available()},
            {"executor_id", NATIVE_CHAT_EXECUTOR_ID},
            {"renderer_id", NATIVE_CHAT_RENDERER_ID},
            {"grammar_id", NATIVE_CHAT_GRAMMAR_ID},
            {"parser_id", NATIVE_CHAT_PARSER_ID},
            {"atomic", true},
            {"buffered", true},
        };
#ifdef CLOZN_SAE
        if (sae_serve.on)
            h["sae"] = {{"d_sae", sae_serve.enc.d_sae()}, {"layer", sae_serve.layer}, {"k", sae_serve.k}};
#endif
        if (jlens.on)
            h["jlens"] = {{"layers", jlens.layers()}, {"default_layer", jlens.default_layer},
                          {"d_model", jlens.d_model}, {"vocab", jlens.vocab}};
        res.set_content(h.dump(), "application/json");
    });

    // The real-time denoise visualization (a pure consumer of the SSE event stream).
    svr.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_content(VIZ_HTML, "text/html; charset=utf-8");
    });

    // POST /cancel {"req": "<id>"} -- cooperative cancel: flip the named request's cancel flag (if
    // it's still live) so its generation loop stops within one token/pass instead of running to
    // completion. The gateway can call this the instant it notices its client is gone; an ordinary
    // disconnect is usually caught even sooner (see StreamEnvelope::write_raw). Idempotent and never
    // an error: an unknown/already-finished id just reports cancelled:false -- the request is moot
    // either way, since it has either already stopped or there was nothing to stop.
    svr.Post("/cancel", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        const std::string id = (!body.is_discarded() && body.is_object())
                                    ? body.value("req", std::string()) : std::string();
        if (id.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "need {\"req\": \"<request id>\"}"}}.dump(), "application/json");
            return;
        }
        const bool found = cancel_registry.cancel(id);
        res.set_content(json{{"cancelled", found}, {"req", id}}.dump(), "application/json");
    });

    // Shared body of completions + infill: build the runner, then either stream the events as SSE
    // or run once and return JSON.
    auto handle = [&](const httplib::Request& req, httplib::Response& res, bool is_infill) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        if (is_infill) {  // infill (fill-in-the-middle) was structurally diffusion-only; the diffusion
            // generator was removed with the diffusion program (THE_CUT) -- there is no longer any
            // mode this can succeed in.
            res.status = 400;
            res.set_content(json{{"error", "infill is no longer supported -- diffusion generation was "
                                           "removed from this engine build"}}.dump(),
                            "application/json");
            return;
        }

        // Private Phase 2.8 atomic seam: prepare with the currently loaded template, constrain the
        // completion, and parse it below without allowing a client-held descriptor to drift or be
        // modified between those operations. Streaming is deliberately rejected here: the public
        // structured route buffers until validation anyway.
        std::optional<PreparedChat> atomic_prepared_chat;
        std::vector<PerformancePhase> request_timing_phases;
        if (body.contains("chat_request")) {
            if (body.contains("prepared_chat")) {
                res.status = 400;
                res.set_content(json{{"error", "chat_request and prepared_chat are mutually exclusive"}}.dump(),
                                "application/json");
                return;
            }
            if (!ar_mode || is_infill) {
                res.status = 400;
                res.set_content(json{{"error", "chat_request requires an autoregressive completion"}}.dump(),
                                "application/json");
                return;
            }
            if (body.value("stream", false)) {
                res.status = 400;
                res.set_content(json{{"error", "chat_request is buffer-then-validate only; set stream:false"}}.dump(),
                                "application/json");
                return;
            }
            try {
                const json& request_json = body["chat_request"];
                if (!request_json.is_object() || !request_json.contains("messages") ||
                    !request_json["messages"].is_array() || request_json["messages"].empty()) {
                    throw std::invalid_argument("chat_request.messages must be a non-empty array");
                }
                ChatTemplateRequest request;
                request.messages_json = request_json["messages"].dump();
                request.tools_json = request_json.value("tools", json::array()).dump();
                request.tool_choice_json = request_json.value("tool_choice", json("auto")).dump();
                if (request_json.contains("json_schema")) {
                    if (!request_json["json_schema"].is_object()) {
                        throw std::invalid_argument("chat_request.json_schema must be an object");
                    }
                    request.json_schema_json = request_json["json_schema"].dump();
                }
                request.parallel_tool_calls = request_json.value("parallel_tool_calls", false);
                request.add_generation_prompt = request_json.value("add_generation_prompt", true);
                request.enable_thinking = request_json.value("enable_thinking", true);
                request.reasoning_format = request_json.value("reasoning_format", std::string("none"));
                const bool active_tools = request_json.contains("tools") &&
                    request_json["tools"].is_array() && !request_json["tools"].empty() &&
                    !(request_json.contains("tool_choice") &&
                      request_json["tool_choice"].is_string() &&
                      request_json["tool_choice"].get<std::string>() == "none");
                if (!active_tools && !request_json.contains("json_schema")) {
                    throw std::invalid_argument(
                        "chat_request requires active tools or json_schema structured output");
                }
                const PerfClock::time_point prompt_template_started = PerfClock::now();
                atomic_prepared_chat = chat_templates->prepare(request);
                request_timing_phases.push_back(PerformancePhase{
                    "prompt_template", "clozn_worker", duration_ns(prompt_template_started)});
                const PreparedChat& prepared = *atomic_prepared_chat;
                json triggers = json::array();
                for (const ChatGrammarTrigger& trigger : prepared.grammar_triggers) {
                    triggers.push_back({{"type", trigger.type}, {"value", trigger.value},
                                        {"token", trigger.token}});
                }
                body["prepared_chat"] = {
                    {"prompt", prepared.prompt}, {"grammar", prepared.grammar},
                    {"grammar_lazy", prepared.grammar_lazy},
                    {"grammar_triggers", std::move(triggers)},
                    {"preserved_tokens", prepared.preserved_tokens},
                    {"additional_stops", prepared.additional_stops},
                    {"generation_prompt", prepared.generation_prompt},
                    {"thinking_start_tag", prepared.thinking_start_tag},
                    {"thinking_end_tag", prepared.thinking_end_tag},
                };
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", std::string("invalid chat_request: ") + e.what()}}.dump(),
                                "application/json");
                return;
            }
        }

        // A descriptor returned by /prepare_chat is inspection/qualification data, not a generation
        // capability: accepting it back from a client would allow grammar/parser tampering and stale
        // state after a worker restart. Only the server-created atomic descriptor above may enter the
        // sampler. Product activation still requires exact qualification in the Python gateway.
        std::optional<GrammarConfig> chat_grammar;
        if (body.contains("prepared_chat")) {
            if (!atomic_prepared_chat) {
                res.status = 400;
                res.set_content(json{{"error", "client-held prepared_chat is not accepted; use chat_request"}}.dump(),
                                "application/json");
                return;
            }
            if (!ar_mode || is_infill) {
                res.status = 400;
                res.set_content(json{{"error", "prepared_chat requires an autoregressive completion"}}.dump(),
                                "application/json");
                return;
            }
            const json& prepared = body["prepared_chat"];
            try {
                if (!prepared.is_object()) {
                    throw std::invalid_argument("prepared_chat must be an object returned by /prepare_chat");
                }
                const std::string prepared_prompt = prepared.at("prompt").get<std::string>();
                if (prepared_prompt.empty()) {
                    throw std::invalid_argument("prepared_chat.prompt must not be empty");
                }
                if (body.contains("prompt") &&
                    (!body["prompt"].is_string() || body["prompt"].get<std::string>() != prepared_prompt)) {
                    throw std::invalid_argument("prompt must exactly match prepared_chat.prompt");
                }
                body["prompt"] = prepared_prompt;

                GrammarConfig parsed;
                parsed.grammar = prepared.at("grammar").get<std::string>();
                parsed.grammar_lazy = prepared.at("grammar_lazy").get<bool>();
                parsed.preserved_tokens = prepared.at("preserved_tokens").get<std::vector<std::string>>();
                parsed.generation_prompt = prepared.at("generation_prompt").get<std::string>();
                parsed.reasoning_start_tag = prepared.at("thinking_start_tag").get<std::string>();
                parsed.reasoning_end_tag = prepared.at("thinking_end_tag").get<std::string>();
                parsed.additional_stops = prepared.at("additional_stops").get<std::vector<std::string>>();

                const json& triggers = prepared.at("grammar_triggers");
                if (!triggers.is_array()) {
                    throw std::invalid_argument("prepared_chat.grammar_triggers must be an array");
                }
                parsed.grammar_triggers.reserve(triggers.size());
                for (const json& value : triggers) {
                    if (!value.is_object()) {
                        throw std::invalid_argument("each prepared_chat grammar trigger must be an object");
                    }
                    GrammarTrigger trigger;
                    const std::string type = value.at("type").get<std::string>();
                    trigger.value = value.at("value").get<std::string>();
                    trigger.token = value.at("token").get<int>();
                    if (type == "token") trigger.type = GrammarTriggerType::Token;
                    else if (type == "word") trigger.type = GrammarTriggerType::Word;
                    else if (type == "pattern") trigger.type = GrammarTriggerType::Pattern;
                    else if (type == "pattern_full") trigger.type = GrammarTriggerType::PatternFull;
                    else throw std::invalid_argument("unknown prepared_chat grammar trigger type: " + type);
                    parsed.grammar_triggers.push_back(std::move(trigger));
                }
                if (parsed.grammar.empty()) {
                    throw std::invalid_argument("prepared_chat grammar must not be empty");
                }
                chat_grammar = std::move(parsed);
            } catch (const std::exception& e) {
                res.status = 400;
                res.set_content(json{{"error", std::string("invalid prepared_chat: ") + e.what()}}.dump(),
                                "application/json");
                return;
            }
        }
        const GenerateConfig cfg = config_from(body);
        const SampleConfig sample = sample_from(body);
        const bool stream = body.value("stream", false);
        // Phase 1.2 state-stream protocol: protocol:true reshapes the SSE frames to StateStep; state:"full"
        // rides the heavy raw-activation tensor on each frame (the light frame omits it). "full" implies
        // the activation tap (the raw state only exists when the tap is on), so it forces features on.
        const bool protocol = body.value("protocol", false);
        const bool state_full = body.value("state", std::string("light")) == std::string("full");
        const bool features = body.value("features", false) || state_full;  // white-box: emit per-slot activations
        // White-box WRITE: steer:{concept, coef, layer?} pushes a concept direction into the residual
        // stream during the denoise (a control vector). coef 0 / no object => no steering.
        std::string steer_concept; double steer_coef = 0.0; int steer_layer = 0;
        if (body.contains("steer") && body["steer"].is_object()) {
            steer_concept = body["steer"].value("concept", std::string());
            steer_coef = body["steer"].value("coef", 0.0);
            steer_layer = body["steer"].value("layer", 0);
        }

        // F1 live lens ("watch it think"): {"lens": {layer?, topk?, every?}} (or lens:true) streams a
        // jlens_live frame per committed token — the J-lens "disposed to say" readout computed from the
        // tapped residual DURING generation, not post-hoc. Stream-only; exclusive with features/state/
        // protocol (one activation tap per request; the lens owns it). Each readout is one CPU
        // J-transport + unembed (~10-30ms/token) — "every" thins it to every Nth committed token.
        bool lens_on = false; int lens_layer = 0, lens_topk = 5, lens_every = 1;
        if (body.contains("lens") &&
            ((body["lens"].is_boolean() && body["lens"].get<bool>()) || body["lens"].is_object())) {
            const json lb = body["lens"].is_object() ? body["lens"] : json::object();
            if (!jlens.on) {
                res.status = 400;
                res.set_content(json{{"error",
                    "lens requested but J-lens not loaded (start with --jlens <dir> or set CLOZN_JLENS_DIR)"}}.dump(),
                    "application/json");
                return;
            }
            lens_layer = lb.value("layer", jlens.default_layer);
            if (!jlens.has(lens_layer)) {
                res.status = 400;
                res.set_content(json{{"error", "no J-lens sidecar for that layer"},
                                     {"layer", lens_layer}, {"available", jlens.layers()}}.dump(),
                                "application/json");
                return;
            }
            lens_topk = lb.value("topk", 5);
            if (lens_topk < 1) lens_topk = 1;
            if (lens_topk > 20) lens_topk = 20;
            lens_every = lb.value("every", 1);
            if (lens_every < 1) lens_every = 1;
            if (!stream) {
                res.status = 400;
                res.set_content(json{{"error",
                    "lens rides the SSE stream only; set stream:true (post-hoc readout: POST /jlens)"}}.dump(),
                    "application/json");
                return;
            }
            if (features || protocol) {
                res.status = 400;
                res.set_content(json{{"error",
                    "lens is exclusive with features/state/protocol (one activation tap per request)"}}.dump(),
                    "application/json");
                return;
            }
            lens_on = true;
        }

        // Phase 2.3 readout plane: {"readout": {layers?, jlens?, norms?, probes?, topk?, every?,
        // include_prompt?}} captures N layers' residuals in ONE forward and fans them out to every
        // requested observer on a worker thread — the decode thread never runs observer math (the
        // structural fix for jlens_live's decode-thread GEMV, and for the single-owner tap). Emits a
        // `readout` frame per observed token + a final `readout_stats` frame with honest coverage
        // (observed / dropped / skipped — no silent caps). Stream-only.
        std::shared_ptr<ReadoutPlane> plane;
        std::vector<int> readout_layers;
        if (body.contains("readout") &&
            ((body["readout"].is_boolean() && body["readout"].get<bool>()) || body["readout"].is_object())) {
            const json rb = body["readout"].is_object() ? body["readout"] : json::object();
            if (!stream) {
                res.status = 400;
                res.set_content(json{{"error", "readout rides the SSE stream only; set stream:true"}}.dump(),
                                "application/json");
                return;
            }
            if (lens_on) {
                res.status = 400;
                res.set_content(json{{"error",
                    "readout supersedes lens (it computes the same jlens readout, plus more, off the decode "
                    "thread); drop the lens param"}}.dump(), "application/json");
                return;
            }
            ReadoutObserverConfig rc;
            rc.jlens = rb.value("jlens", true);
            rc.norms = rb.value("norms", true);
            rc.probes = rb.value("probes", false);
            rc.topk = std::max(1, std::min(rb.value("topk", 8), 20));
            rc.every = std::max(1, rb.value("every", 1));
            rc.include_prompt = rb.value("include_prompt", false);
            if (rb.contains("layers") && rb["layers"].is_array())
                for (const auto& l : rb["layers"])
                    if (l.is_number_integer()) readout_layers.push_back(l.get<int>());
            // Default layer set: every J-lens sidecar layer when the lens is loaded (the observers
            // people actually want), else the 2/3-depth mid layer (norms-only still works there).
            if (readout_layers.empty()) {
                if (jlens.on) readout_layers = jlens.layers();
                else readout_layers = {model_n_layer * 2 / 3};
            }
            for (int L : readout_layers) {
                if (L < 1 || L >= model_n_layer) {
                    res.status = 400;
                    res.set_content(json{{"error", "readout layer out of range (1..n_layer-1)"},
                                         {"layer", L}, {"n_layer", model_n_layer}}.dump(),
                                    "application/json");
                    return;
                }
            }
            if (rc.jlens && !jlens.on && !rc.norms && !rc.probes) {
                res.status = 400;
                res.set_content(json{{"error",
                    "readout.jlens requested but J-lens not loaded (start with --jlens <dir>), and no "
                    "other observer is on"}}.dump(), "application/json");
                return;
            }
            // Concept probes are calibrated at the 2/3-depth mid layer (the same convention the
            // steering paths use); capture it if the probes observer needs it and it isn't listed.
            rc.probe_layer = rb.value("probe_layer", model_n_layer * 2 / 3);
            if (rc.probes) {
                bool have = false;
                for (int L : readout_layers) if (L == rc.probe_layer) { have = true; break; }
                if (!have && rc.probe_layer >= 1 && rc.probe_layer < model_n_layer)
                    readout_layers.push_back(rc.probe_layer);
            }
            rc.layers = readout_layers;
            plane = std::make_shared<ReadoutPlane>(rc, &jlens, &concept_probes, model.get());
        }

        // Circuit-tracer S4 (notes/CIRCUIT_TRACER_DESIGN.md): GENERATE with an activation patch
        // armed — "disable this node and watch the answer change". `write` is one
        // {layer, positions, values} spec or an array of them (same contract as /score's write;
        // GgmlAdapter::add_write_state per spec, leak-free clear on every exit). Positions are
        // absolute board positions: prompt sites apply during the prefill (the KV then carries the
        // patched state through the whole generation); a generated position's patch applies when
        // that row is decoded. Composes with reference_tokens for the margin-test observation arm
        // (patch + greedy decode + stop at first divergence from the baseline reply).
        struct WriteReq { int layer; std::vector<int> positions; std::vector<float> values; };
        std::vector<WriteReq> write_reqs;
        if (body.contains("write")) {
            json wspecs = json::array();
            if (body["write"].is_object()) wspecs.push_back(body["write"]);
            else if (body["write"].is_array()) wspecs = body["write"];
            for (const json& wb : wspecs) {
                WriteReq w{};
                w.layer = wb.is_object() ? wb.value("layer", 0) : 0;
                if (wb.is_object() && wb.contains("positions") && wb["positions"].is_array())
                    w.positions = wb["positions"].get<std::vector<int>>();
                if (wb.is_object() && wb.contains("values") && wb["values"].is_array())
                    w.values = wb["values"].get<std::vector<float>>();
                if (w.layer < 1 || w.positions.empty() || w.values.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error",
                        "each write spec needs {layer >= 1, positions:[int], values:[float] = positions*n_embd}"}}.dump(),
                        "application/json");
                    return;
                }
                for (int p : w.positions) {
                    if (p < 0 || p >= n_ctx) {
                        res.status = 400;
                        res.set_content(json{{"error", "write position out of range"}, {"position", p},
                                             {"n_ctx", n_ctx}}.dump(), "application/json");
                        return;
                    }
                }
                write_reqs.push_back(std::move(w));
            }
        }

        // Encode inputs via the model's vocab (no context needed — fully concurrent).
        const PerfClock::time_point tokenize_started = PerfClock::now();
        std::vector<int> prompt_ids, suffix_ids;
        int gap = 0;
        if (is_infill) {
            prompt_ids = model->encode(body.value("prefix", std::string()));
            suffix_ids = model->encode(body.value("suffix", std::string()));
            gap = body.value("gap", 8);
        } else {
            prompt_ids = model->encode(body.value("prompt", std::string()));
        }
        request_timing_phases.push_back(PerformancePhase{
            "tokenize", "clozn_worker", duration_ns(tokenize_started)});
        if (prompt_ids.empty() && !(is_infill && !suffix_ids.empty())) {
            res.status = 400;
            res.set_content(json{{"error", "empty prompt"}}.dump(), "application/json");
            return;
        }

        // Optional PyTorch-trained soft prefix (the train-on-HF / serve-on-llama.cpp bridge): a flat
        // prefix_rows x n_embd float array spliced in ahead of the prompt before decoding. AR-only.
        std::vector<float> prefix_embd;
        const int prefix_rows = body.value("prefix_rows", 0);
        if (prefix_rows > 0 && body.contains("prefix_embd") && body["prefix_embd"].is_array()) {
            prefix_embd = body["prefix_embd"].get<std::vector<float>>();   // AR: ar_forward_embd; diffusion: set_diffusion_prefix
        }
        std::vector<float> steer_vec;
        if (body.contains("steer_vec") && body["steer_vec"].is_array()) {
            steer_vec = body["steer_vec"].get<std::vector<float>>();
        }
        // Optional early-stop reference (prove-all ablated arms, AR-only): the baseline reply's committed
        // token ids. generate_ar halts at the first generated token that differs from this list -- so the
        // ablated arm decodes only up to the point the answer provably changed, not the full ~max_tokens.
        // Purely a termination check; greedy output up to the stop is bit-identical to full generation.
        std::vector<int> reference_tokens;
        if (ar_mode && body.contains("reference_tokens") && body["reference_tokens"].is_array()) {
            reference_tokens = body["reference_tokens"].get<std::vector<int>>();
        }

        // checkpoint_on_finish (AR only): save the generation's OWN KV as a checkpoint right
        // after decoding finishes, on the same lease -- ZERO reconstruction, so the saved state
        // is the run's numerics by construction (immune to the batch-shape and observer-regime
        // epsilons that make from-tokens rebuilds only near-exact under strong steering).
        // Sampler + steer provenance are auto-filled from THIS request's own config. The id
        // comes back as "checkpoint_id" on the non-streaming response.
        const bool want_ckpt = ar_mode && body.value("checkpoint_on_finish", false);
        auto ckpt_id_out = std::make_shared<std::string>();

        // One call into the runtime on a POOLED context (acquire blocks until one is free, so N
        // workers run N requests concurrently; the Lease releases it on any exit).
        auto run = [&pool, &concept_probes, &steer_probes, &sae_serve, prompt_ids, suffix_ids, gap, cfg, sample, is_infill, ar_mode, features, steer_concept, steer_coef, steer_layer, prefix_embd, prefix_rows, steer_vec, reference_tokens, lens_on, lens_layer, plane, readout_layers, write_reqs, chat_grammar, want_ckpt, ckpt_id_out, &ckpt_mtx, &checkpoints, request_timing_phases, &startup_timing_phases, &startup_timing_claimed, duration_ns](
                       const std::function<void(const Event&)>& on_event) {
            std::vector<PerformancePhase> phases = request_timing_phases;
            const PerfClock::time_point worker_start_started = PerfClock::now();
            ContextPool::Lease lease = pool.acquire();
            phases.push_back(PerformancePhase{
                "worker_start", "clozn_worker", duration_ns(worker_start_started)});
            if (!startup_timing_claimed.exchange(true, std::memory_order_relaxed)) {
                phases.insert(phases.end(), startup_timing_phases.begin(), startup_timing_phases.end());
            }
            // white-box tap on for this request (off by default); the live lens implies it — the
            // per-token StepActivations events ARE its input (consumed into jlens_live frames downstream).
            (*lease).set_emit_activations(features || lens_on);
            // Phase 2.3 readout plane: arm the multi-layer capture set + sink. The sink body runs on
            // the decode thread but only queues (ReadoutPlane::push is bounded + non-blocking); the
            // observer math runs on the plane's own worker thread.
            if (plane) {
                (*lease).set_capture_layers(readout_layers);
                std::shared_ptr<ReadoutPlane> pl = plane;
                (*lease).set_capture_sink([pl](CaptureFrame&& f) { pl->push(std::move(f)); });
            }
            // Tracer S4: arm the activation patch(es) for this whole generation (cleared below).
            if (!write_reqs.empty()) {
                (*lease).clear_write();
                for (const auto& w : write_reqs) {
                    if (!(*lease).add_write_state(w.layer, w.positions, w.values)) {
                        (*lease).clear_write();
                        throw std::invalid_argument(
                            "write rejected: layer must be in [1, n_layer) and values.size must equal "
                            "positions.size * n_embd (n_embd " + std::to_string((*lease).n_embd()) + ")");
                    }
                }
            }
            // --sae: read at the SAE's own layer and ride each pass's top-k features on the stream.
            const bool sae_on = features && sae_serve.on;
            const int default_tap = (*lease).tap_layer();
            if (sae_on) (*lease).set_tap_layer(sae_serve.layer);
            else if (lens_on) (*lease).set_tap_layer(lens_layer);  // the lens owns the tap (validated exclusive)
            const std::function<void(const Event&)> ev = with_sae_readout(on_event, sae_serve, sae_on);
            const ConceptProbes* probes = (features && concept_probes.ready()) ? &concept_probes : nullptr;
            const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
            const bool raw_steer = !steer_vec.empty();   // a raw tone direction (studio engine dials)
            if (steering || raw_steer) {
                const int nl = (*lease).n_layer();
                int lo, hi;
                if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                else { const int tl = nl * 2 / 3;  // steer at mid-depth, where steer_probes is calibrated
                       lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                if (lo < 1) lo = 1;
                if (steering) {
                    (*lease).set_steer(build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl), lo, hi);
                } else {                          // RAW tone direction: build the same cvec layout straight from it
                    const int ne = static_cast<int>(steer_vec.size());
                    std::vector<float> cvec(static_cast<size_t>(ne) * nl, 0.0f);
                    const double c = steer_coef != 0.0 ? steer_coef : 1.0;
                    for (int L = lo; L <= hi; ++L) {
                        if (L < 1 || L >= nl) continue;
                        float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                        for (int i = 0; i < ne; ++i) slice[i] = static_cast<float>(c * steer_vec[i]);
                    }
                    (*lease).set_steer(cvec, lo, hi);
                }
            }
            // AR model => the causal left-to-right loop (same white-box reads/steering, no scheduler).
            // Diffusion => the denoiser (whole-sequence generate, or infill between prefix/suffix).
            // Diffusion: a soft prefix rides in as a frozen block via set_diffusion_prefix (AR uses the
            // ar_forward_embd arg instead). Either way it's the HF-trained memory, injected into the GGUF.
            const bool diff_prefix = !ar_mode && !prefix_embd.empty() && prefix_rows > 0;
            if (diff_prefix) (*lease).set_diffusion_prefix(prefix_embd, prefix_rows);
            // Restore the pooled context to a clean state (steer/prefix/tap/emit off) on EVERY exit path.
            // Critical: a generator can throw (n_ctx exceeded, llama_decode failure, ...). On the STREAMING
            // path that throw escapes into cpp-httplib's worker thread with no handler -> std::terminate() ->
            // abort(): a silent hard crash (no trace on a Windows Release build). So clean up + rethrow -- the
            // streaming provider below catches the rethrow and emits a clean error frame; the non-streaming
            // caller is already inside httplib's routing try/catch, so it degrades to a 500. Either way the
            // pooled context goes back clean, never dirty.
            auto cleanup = [&]() {
                if (diff_prefix) (*lease).clear_diffusion_prefix();
                if (steering || raw_steer) (*lease).clear_steer();
                if (sae_on || lens_on) (*lease).set_tap_layer(default_tap);
                if (plane) {  // disarm the capture plane before returning the pooled context
                    (*lease).set_capture_sink({});
                    (*lease).set_capture_layers({});
                }
                if (!write_reqs.empty()) (*lease).clear_write();  // never leak a patch into the pool
                (*lease).set_emit_activations(false);  // reset before returning the pooled context
            };
            try {
                // Diffusion generation was removed from this engine build (THE_CUT): is_infill is
                // already rejected above, and ar_mode is the only mode with a generator left. A model
                // loaded with --diffusion/--mask-token has no generator to fall back to any more --
                // fail honestly rather than silently misbehave.
                if (!ar_mode) {
                    throw std::invalid_argument(
                        "diffusion generation was removed from this engine build; only autoregressive "
                        "models are supported");
                }
                GenerateResult r = generate_ar(*lease, prompt_ids, cfg, ev, sample, probes,
                                     prefix_embd.empty() ? nullptr : &prefix_embd, prefix_rows,
                                     reference_tokens.empty() ? nullptr : &reference_tokens,
                                     chat_grammar ? &*chat_grammar : nullptr, nullptr,
                                     // resume_truncate_to / force_first_token: execution-fork only,
                                     // defaults here -- this is ordinary generation, not a fork.
                                     -1, nullptr, &phases);
                if (want_ckpt && ar_mode) {
                    // Top-up: the loop's final committed token may not be decoded into the KV
                    // yet (it breaks before the decode when max_new is reached). Decode it with
                    // the SAME single-token shape the loop itself would have used, so the saved
                    // state is exactly the state an uninterrupted longer run would have had.
                    const int n_board = static_cast<int>(r.board.size());
                    // steer/prefix are still armed here (cleanup runs after) -- intentionally:
                    // the top-up decode must run under the run's own regime.
                    (*lease).evict_from(n_board - 1);
                    (*lease).ar_forward({r.board[static_cast<size_t>(n_board - 1)]}, n_board - 1);
                    EngineCheckpoint ckpt = (*lease).save_checkpoint(r.board, n_board);
                    // The prompt/generated boundary within r.board (execution-fork's regime split):
                    // r.board == prompt_ids ++ generated by construction, so prompt_ids.size() IS
                    // that boundary -- no soft prefix is included (prefix_embd rides in as raw
                    // embeddings, never token ids, so it never enters r.board).
                    ckpt.prompt_tokens = static_cast<int>(prompt_ids.size());
                    if (sample.temperature > 0.0) {
                        ckpt.has_sampler = true;
                        ckpt.temperature = sample.temperature;
                        ckpt.rep_penalty = sample.rep_penalty;
                        ckpt.top_k = sample.top_k;
                        ckpt.top_p = sample.top_p;
                        ckpt.seed = sample.seed;
                        ckpt.rng_draws = static_cast<uint64_t>(r.new_tokens);
                    }
                    if (steering || raw_steer) {
                        // capture the cvec ACTUALLY applied (rebuild it identically to above)
                        const int nl = (*lease).n_layer();
                        int lo, hi;
                        if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                        else { const int tl = nl * 2 / 3;
                               lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                        if (lo < 1) lo = 1;
                        std::vector<float> cvec;
                        if (steering) {
                            cvec = build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl);
                        } else {
                            const int ne = static_cast<int>(steer_vec.size());
                            cvec.assign(static_cast<size_t>(ne) * nl, 0.0f);
                            const double c = steer_coef != 0.0 ? steer_coef : 1.0;
                            for (int L = lo; L <= hi; ++L) {
                                if (L < 1 || L >= nl) continue;
                                float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                                for (int i = 0; i < ne; ++i) slice[i] = static_cast<float>(c * steer_vec[i]);
                            }
                        }
                        ckpt.has_steer = true;
                        ckpt.steer_cvec = std::move(cvec);
                        ckpt.steer_lo = lo;
                        ckpt.steer_hi = hi;
                    }
                    const std::string cid = make_id("ckpt-");
                    {
                        std::lock_guard<std::mutex> lk(ckpt_mtx);
                        if (static_cast<int>(checkpoints.size()) >= kMaxCheckpoints)
                            checkpoints.erase(checkpoints.begin());
                        checkpoints[cid] = std::move(ckpt);
                    }
                    *ckpt_id_out = cid;
                }
                cleanup();
                return r;
            } catch (...) {
                cleanup();
                throw;
            }
        };
        const std::string id = make_id(is_infill ? "infill-" : "cmpl-");
        const char* object = is_infill ? "infill" : "text_completion";

        if (stream) {
            const char* substrate = ar_mode ? "autoregressive" : "diffusion";
            // SSE: each §5.1 event becomes a `data: <json>\n\n` frame — the native streaming wire.
            res.set_chunked_content_provider(
                "text/event-stream",
                [run, id, object, model, prompt_ids, suffix_ids, protocol, state_full, substrate, mask_token,
                 &jlens, lens_on, lens_layer, lens_topk, lens_every, &cancel_registry, plane, ckpt_id_out]
                (size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& s) { return sink.write(s.data(), s.size()); };
                    StreamEnvelope env{id, write};        // stamps req + monotonic seq on every frame
                    env.cancel = cancel_registry.register_request(id);      // cooperative cancel (see
                    CancelRegistry::Guard cancel_guard{cancel_registry, id};  // server_shared.hpp); unregisters on any exit
                    try {                                 // a generator throw here would otherwise escape into
                                                          // httplib's worker thread -> abort(); catch it below.
                    if (protocol) {
                        // State-stream protocol: fold the §5.1 events into StateStep frames.
                        StateStepBuilder builder(*model, substrate, state_full,
                                                 [&](json f) { env.frame(std::move(f)); });
                        auto on_event = [&](const Event& e) {
                            env.check_cancelled();
                            builder.on_event(e);
                            // readout plane: interleave completed observer frames (all sink writes
                            // stay on THIS thread; the worker only computes)
                            if (plane) for (auto& fr : plane->drain()) env.frame(std::move(fr));
                        };
                        GenerateResult r = run(on_event);
                        builder.finish();
                        if (plane) for (auto& fr : plane->finish()) env.frame(std::move(fr));
                        // Final summary frame: the canonical snapshot (board + text + layout) the
                        // consumer restores via /v1/board (meta.kind="final").
                        json final_frame = {{"kind", "final"}, {"id", id}, {"object", object},
                                            {"text", r.text}, {"finish_reason", finish_reason(r.reason)},
                                            {"board", r.board},
                                            {"layout", board_layout_json(*model, r.board, mask_token)}};
                        // checkpoint_on_finish + stream:true used to save a checkpoint the caller could
                        // never address (the id was only ever attached to the NON-streaming response) --
                        // an orphan checkpoint on every streamed generation that asked for one. Fixed by
                        // threading ckpt_id_out (a shared_ptr<string>; `run` writes through its own copy
                        // of the SAME pointee) into this frame too.
                        if (!ckpt_id_out->empty()) final_frame["checkpoint_id"] = *ckpt_id_out;
                        env.frame(final_frame);
                        env.done();
                        sink.done();
                        return true;
                    }
                    int lens_n = 0; bool lens_err_sent = false;
                    auto on_event = [&](const Event& e) {
                        env.check_cancelled();
                        // F1 live lens: consume the raw StepActivations event (heavy, never forwarded in
                        // lens mode) into a compact jlens_live frame — the J-lens "disposed to say" top-k
                        // for the just-committed token, computed mid-generation. Readout shape mirrors
                        // POST /jlens so consumers reuse one parser. A readout failure reports ONCE then
                        // goes quiet (generation itself is never disturbed by a lens hiccup).
                        if (lens_on) {
                            if (const auto* sa = std::get_if<StepActivations>(&e)) {
                                const bool due = (lens_n++ % lens_every) == 0;
                                if (due && !sa->values.empty() && !sa->positions.empty()) {
                                    std::vector<std::vector<std::pair<int, float>>> tk;
                                    std::string lerr;
                                    if (jlens.readout(sa->values.data(),
                                                      static_cast<int>(sa->positions.size()),
                                                      lens_layer, lens_topk, tk, lerr)) {
                                        json rd = json::array();
                                        for (const auto& row : tk) {
                                            json rr = json::array();
                                            for (const auto& pr : row)
                                                rr.push_back({{"id", pr.first},
                                                              {"piece", model->decode({pr.first})},
                                                              {"score", pr.second}});
                                            rd.push_back(rr);
                                        }
                                        json fr = {{"type", "jlens_live"}, {"t", sa->t},
                                                   {"layer", lens_layer}, {"positions", sa->positions},
                                                   {"readouts", rd}};
                                        env.frame(std::move(fr));
                                    } else if (!lens_err_sent) {
                                        lens_err_sent = true;
                                        env.frame(json{{"type", "jlens_live"}, {"error", lerr}});
                                    }
                                }
                                return;
                            }
                        }
                        env.frame_str(sse_data(e, *model, prompt_ids, suffix_ids));
                        // readout plane: interleave completed observer frames (all sink writes stay
                        // on THIS thread; the worker only computes)
                        if (plane) for (auto& fr : plane->drain()) env.frame(std::move(fr));
                    };
                    GenerateResult r = run(on_event);
                    if (plane) for (auto& fr : plane->finish()) env.frame(std::move(fr));
                    // A final OpenAI-style frame carrying the assembled text, then [DONE].
                    json final_frame = {{"id", id}, {"object", object},
                                        {"choices", json::array({{{"text", r.text}, {"index", 0},
                                                     {"finish_reason", finish_reason(r.reason)}}})}};
                    if (r.ref_active) {  // early-stop-on-divergence verdict (prove-all ablated arms)
                        final_frame["diverged"] = r.diverged;
                        final_frame["diverged_at"] = r.diverged_at;
                    }
                    // See the protocol-frame branch above: checkpoint_on_finish + stream:true must not
                    // orphan the checkpoint it saves.
                    if (!ckpt_id_out->empty()) final_frame["checkpoint_id"] = *ckpt_id_out;
                    env.frame(final_frame);
                    env.done();
                    sink.done();
                    return true;
                    } catch (const GenerationCancelled&) {
                        // Cooperative cancel fired (POST /cancel, or the client socket write itself
                        // already failed) -- stop, and say so honestly: this is an intentional stop, not
                        // a "generation failed". run() already restored the pooled context via its own
                        // catch(...); a frame write here may itself no-op silently (write_raw's job) if
                        // the client is already gone -- that's fine, nobody's reading it either way.
                        env.frame(json{{"cancelled", true}, {"id", id}});
                        env.done();
                        sink.done();
                        return true;
                    } catch (const std::exception& e) {
                        // The generator threw (n_ctx exceeded, decode failure, ...). Emit a clean error frame
                        // and close the stream gracefully -- run() already restored the pooled context.
                        env.frame(json{{"error", std::string("generation failed: ") + e.what()}});
                        env.done();
                        sink.done();
                        return true;
                    } catch (...) {
                        env.frame(json{{"error", "generation failed"}});
                        env.done();
                        sink.done();
                        return true;
                    }
                });
            return;
        }

        // Non-streaming: run() can throw on a genuinely exceptional decode state (a prompt that exceeds
        // n_ctx, a llama_decode failure). Catch it here so it becomes a clean 400 JSON error, NEVER an
        // uncaught throw that cpp-httplib turns into an empty-body 500 (the streaming path above has the
        // same guard). A generation that merely reaches the context window is NOT exceptional -- it stops
        // gracefully with finish_reason "length" inside generate_ar; this catch is for the real failures.
        try {
            GenerateResult r = run({});
            json resp = {
                {"id", id}, {"object", object},
                {"choices", json::array({{{"text", r.text}, {"index", 0},
                             {"finish_reason", finish_reason(r.reason)}}})},
                {"board", r.board},  // white-box SNAPSHOT: the full final board (save + POST to /v1/board)
                {"layout", board_layout_json(*model, r.board, mask_token)},
                {"usage", {{"prompt_tokens", static_cast<int>(prompt_ids.size())},
                           {"completion_tokens", r.new_tokens}, {"steps_total", r.steps_total}}},
            };
            if (want_ckpt && !ckpt_id_out->empty())
                resp["checkpoint_id"] = *ckpt_id_out;   // checkpoint_on_finish: the live-KV save
            if (atomic_prepared_chat) {
                json trace = json::array();
                for (const Event& event : r.events) {
                    trace.push_back(json::parse(sse_data(event, *model, prompt_ids, suffix_ids)));
                }
                json chat_io = {
                    {"raw_model_output", r.text},
                    {"rendered_prompt", atomic_prepared_chat->prompt},
                    {"model_sha256", model_sha256},
                    {"format", atomic_prepared_chat->format},
                    {"trace", std::move(trace)},
                    {"pipeline", {
                        {"executor_id", NATIVE_CHAT_EXECUTOR_ID},
                        {"renderer_id", NATIVE_CHAT_RENDERER_ID},
                        {"grammar_id", NATIVE_CHAT_GRAMMAR_ID},
                        {"parser_id", NATIVE_CHAT_PARSER_ID},
                    }},
                };
                try {
                    const ParsedChat parsed = chat_templates->parse(*atomic_prepared_chat, r.text);
                    chat_io["openai_json"] = parsed.openai_json;
                    chat_io["message"] = json::parse(parsed.openai_json);
                } catch (const std::exception& e) {
                    // Generation evidence must survive a native parser failure so the supervisor can
                    // journal the exact raw output and trace before returning a typed public error.
                    chat_io["parse_error"] = {
                        {"code", "native_parse_failed"}, {"message", e.what()},
                    };
                }
                resp["chat_io"] = std::move(chat_io);
            }
            if (r.ref_active) {  // early-stop-on-divergence verdict (prove-all ablated arms)
                resp["diverged"] = r.diverged;
                resp["diverged_at"] = r.diverged_at;
            }
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(dump_json(json{{"error", std::string("generation failed: ") + e.what()}}),
                            "application/json");
        }
    };

    register_whitebox_routes(svr, ctx);

    register_jlens_routes(svr, ctx);


    register_state_routes(svr, ctx);



    // --- Checkpointing + branching (Phase 2.1) -----------------------------------------------
    // POST /v1/checkpoint — save the current KV state for a running or completed generation.
    // {tokens: [int], n_past: int} → {checkpoint_id, n_past, size_bytes}
    svr.Post("/v1/checkpoint", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        if (!body.contains("tokens") || !body["tokens"].is_array() || body["tokens"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "'tokens' must be a non-empty array of token ids"}}.dump(), "application/json");
            return;
        }
        std::vector<int> tokens = body["tokens"].get<std::vector<int>>();
        const int n_past = body.value("n_past", static_cast<int>(tokens.size()));
        // prefill_to: rebuild the KV the way the ORIGINAL run built it. A generation's KV is
        // prompt-prefill (one big batch) + one-token decodes for each generated token; a
        // checkpoint that big-batch-prefills the WHOLE sequence gets a numerically different KV
        // for the generated tail (batch-shape FP -- the same landmine as prefix caching), which
        // is invisible to greedy resume but flips sampled draws near boundaries (measured:
        // ~40% of sampled resumes diverged until this). Callers checkpointing a generation pass
        // prefill_to = n_prompt; default keeps the old whole-batch behavior for KV states that
        // WERE built in one batch (e.g. /score-style teacher-forced states).
        const int prefill_to = body.value("prefill_to", static_cast<int>(tokens.size()));
        if (prefill_to < 1 || prefill_to > static_cast<int>(tokens.size())) {
            res.status = 400;
            res.set_content(json{{"error", "prefill_to out of range [1, tokens.size()]"}}.dump(),
                            "application/json");
            return;
        }
        // Declared steering provenance: {steer_vec: [n_embd floats], steer_coef?, steer_layer?}
        // -- the direction the ORIGINAL run generated under. The rebuild below must run steered:
        // a steered run's KV embeds the steer, so an unsteered rebuild would checkpoint the
        // wrong state (the same shape-true principle as prefill_to, applied to interventions).
        std::vector<float> ck_steer;
        double ck_steer_coef = 1.0;
        int ck_steer_layer = 0;
        if (body.contains("steer_vec") && body["steer_vec"].is_array()) {
            ck_steer = body["steer_vec"].get<std::vector<float>>();
            ck_steer_coef = body.value("steer_coef", 1.0);
            ck_steer_layer = body.value("steer_layer", 0);
        }
        try {
            ContextPool::Lease lease = pool.acquire();
            (*lease).set_causal(true);
            std::vector<float> ck_cvec;
            int ck_lo = 0, ck_hi = 0;
            if (!ck_steer.empty()) {
                const int nl = (*lease).n_layer();
                if (ck_steer_layer >= 1) { ck_lo = ck_hi = (ck_steer_layer < nl ? ck_steer_layer : nl - 1); }
                else { const int tl = nl * 2 / 3;
                       ck_lo = (tl - 2 > 1 ? tl - 2 : 1); ck_hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                if (ck_lo < 1) ck_lo = 1;
                const int ne = static_cast<int>(ck_steer.size());
                ck_cvec.assign(static_cast<size_t>(ne) * nl, 0.0f);
                for (int L = ck_lo; L <= ck_hi; ++L) {
                    if (L < 1 || L >= nl) continue;
                    float* slice = ck_cvec.data() + static_cast<size_t>(L - 1) * ne;
                    for (int i = 0; i < ne; ++i)
                        slice[i] = static_cast<float>(ck_steer_coef * ck_steer[i]);
                }
                (*lease).set_steer(ck_cvec, ck_lo, ck_hi);
            }
            const std::vector<int> head(tokens.begin(), tokens.begin() + prefill_to);
            (*lease).ar_forward(head, 0);
            for (int i = prefill_to; i < static_cast<int>(tokens.size()); ++i)
                (*lease).ar_forward({tokens[static_cast<size_t>(i)]}, i);
            EngineCheckpoint ckpt = (*lease).save_checkpoint(tokens, n_past);
            // prefill_to IS the prompt/generated boundary by this route's own contract ("callers
            // checkpointing a generation pass prefill_to = n_prompt" -- see the doc comment above);
            // its default (tokens.size(), the whole-batch-prefill case, e.g. a /score-style
            // teacher-forced state) correctly reports "no generated tail" too.
            ckpt.prompt_tokens = prefill_to;
            if (!ck_cvec.empty()) {
                (*lease).clear_steer();
                ckpt.has_steer = true;
                ckpt.steer_cvec = std::move(ck_cvec);
                ckpt.steer_lo = ck_lo;
                ckpt.steer_hi = ck_hi;
            }
            // Optional declared sampler provenance: {sampler: {seed, rng_draws, temperature?,
            // top_k?, top_p?, rep_penalty?}}. rng_draws = sampled committed tokens so far in the
            // generation being checkpointed (greedy tokens consume no draw). Stored verbatim;
            // /v1/restore uses it for a bit-exact SAMPLED resume unless the request overrides.
            if (body.contains("sampler") && body["sampler"].is_object()) {
                const json& sb = body["sampler"];
                ckpt.has_sampler = true;
                ckpt.seed = sb.value("seed", 0ULL);
                ckpt.rng_draws = sb.value("rng_draws", 0ULL);
                ckpt.temperature = sb.value("temperature", 0.0);
                ckpt.rep_penalty = sb.value("rep_penalty", 1.0);
                ckpt.top_k = sb.value("top_k", 0);
                ckpt.top_p = sb.value("top_p", 1.0);
            }
            const std::string id = make_id("ckpt-");
            const size_t sz = ckpt.kv_data.size();
            {
                std::lock_guard<std::mutex> lk(ckpt_mtx);
                if (static_cast<int>(checkpoints.size()) >= kMaxCheckpoints) {
                    checkpoints.erase(checkpoints.begin());
                }
                checkpoints[id] = std::move(ckpt);
            }
            res.set_content(json{{"checkpoint_id", id}, {"n_past", n_past},
                                 {"n_tokens", static_cast<int>(tokens.size())},
                                 {"size_bytes", sz}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("checkpoint failed: ") + e.what()}}.dump(), "application/json");
        }
    });

    // POST /v1/restore — restore a saved checkpoint and optionally continue generation.
    // {checkpoint_id, max_tokens?, temperature?, ...} → {text, tokens, finish_reason}
    svr.Post("/v1/restore", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string ckpt_id = body.value("checkpoint_id", std::string());
        if (ckpt_id.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "need checkpoint_id"}}.dump(), "application/json");
            return;
        }
        EngineCheckpoint ckpt;
        {
            std::lock_guard<std::mutex> lk(ckpt_mtx);
            auto it = checkpoints.find(ckpt_id);
            if (it == checkpoints.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown checkpoint_id"}}.dump(), "application/json");
                return;
            }
            ckpt = it->second;
        }
        const int max_tokens = body.value("max_tokens", 64);
        const bool fast = body.value("fast", true);
        // Sampling for the resumed suffix: an explicit request wins; otherwise a checkpoint
        // carrying declared sampler provenance resumes THAT run bit-exactly (same config, RNG
        // fast-forwarded by the declared consumed draws). Neither present => greedy default,
        // same as before this feature.
        SampleConfig sample = sample_from(body);
        std::string sampler_source = "request";
        const bool body_has_sampling = body.contains("temperature") || body.contains("seed") ||
                                       body.contains("top_k") || body.contains("top_p") ||
                                       body.contains("rep_penalty");
        if (!body_has_sampling && ckpt.has_sampler) {
            sample.temperature = ckpt.temperature;
            sample.rep_penalty = ckpt.rep_penalty;
            sample.top_k = ckpt.top_k;
            sample.top_p = ckpt.top_p;
            sample.seed = ckpt.seed;
            sample.rng_discard = ckpt.rng_draws;
            sampler_source = "checkpoint";
        }
        try {
            ContextPool::Lease lease = pool.acquire();
            // KV-blob fast restore (the "Phase 2 optimization" the first slice deferred):
            // load_checkpoint + a one-token bridge decode instead of re-prefilling the whole
            // saved sequence. fast:false keeps the re-prefill path as the escape hatch; the
            // response names which path ran so nothing about the restore is implicit. Greedy
            // suffix equality between the two paths is the acceptance bar.
            // Steering: a checkpoint carrying declared steer provenance resumes STEERED (and on
            // the reprefill path, rebuilds steered too -- the KV must embed it either way).
            std::string steer_source = "none";
            if (ckpt.has_steer && !ckpt.steer_cvec.empty()) {
                // The cvec ACTUALLY APPLIED during the checkpointed run, re-applied verbatim.
                (*lease).set_steer(ckpt.steer_cvec, ckpt.steer_lo, ckpt.steer_hi);
                steer_source = "checkpoint";
            }
            GenerateConfig cfg;
            cfg.max_new = max_tokens < 1 ? 1 : max_tokens;
            GenerateResult r;
            try {
                r = fast
                    ? generate_ar(*lease, ckpt.tokens, cfg, {}, sample, nullptr,
                                  nullptr, 0, nullptr, nullptr, &ckpt)
                    : generate_ar(*lease, ckpt.tokens, cfg, {}, sample, nullptr,
                                  nullptr, 0, nullptr);
            } catch (...) {
                if (steer_source == "checkpoint") (*lease).clear_steer();
                throw;
            }
            if (steer_source == "checkpoint") (*lease).clear_steer();
            json resp{{"checkpoint_id", ckpt_id}, {"text", r.text},
                      {"finish_reason", finish_reason(r.reason)},
                      {"generated_tokens", r.new_tokens},
                      {"total_tokens", static_cast<int>(r.board.size())},
                      {"restore_mode", fast ? "kv_blob" : "reprefill"},
                      {"sampler_source", sampler_source},
                      {"steer_source", steer_source}};
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("restore failed: ") + e.what()}}.dump(), "application/json");
        }
    });

    // POST /v1/branch — fork N independent continuations from a checkpoint.
    // {checkpoint_id, n: int, max_tokens?, temperature?, seed?} → {branches: [{text, finish_reason}]}
    svr.Post("/v1/branch", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string ckpt_id = body.value("checkpoint_id", std::string());
        if (ckpt_id.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "need checkpoint_id"}}.dump(), "application/json");
            return;
        }
        EngineCheckpoint ckpt;
        {
            std::lock_guard<std::mutex> lk(ckpt_mtx);
            auto it = checkpoints.find(ckpt_id);
            if (it == checkpoints.end()) {
                res.status = 404;
                res.set_content(json{{"error", "unknown checkpoint_id"}}.dump(), "application/json");
                return;
            }
            ckpt = it->second;
        }
        const int n = std::min(body.value("n", 4), 16);
        const int max_tokens = body.value("max_tokens", 64);
        SampleConfig base_sample = sample_from(body);
        try {
            ContextPool::Lease lease = pool.acquire();
            auto results = generate_ar_branched(*lease, ckpt.tokens, n,
                                                max_tokens < 1 ? 1 : max_tokens,
                                                base_sample);
            json branches = json::array();
            for (int i = 0; i < static_cast<int>(results.size()); ++i) {
                branches.push_back({{"index", i}, {"text", results[i].text},
                                    {"finish_reason", finish_reason(results[i].reason)},
                                    {"generated_tokens", results[i].new_tokens}});
            }
            res.set_content(json{{"checkpoint_id", ckpt_id}, {"n", n},
                                 {"branches", branches}}.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("branch failed: ") + e.what()}}.dump(), "application/json");
        }
    });

    // POST /v1/execution-fork — an exact-execution intervention fork: restore a checkpoint,
    // truncate it to an EARLIER point than its own n_past, apply exactly one narrow intervention
    // AT that fork point, and continue generation -- all under ONE lease, so no partially-restored
    // state is ever observable between HTTP calls. This is the missing middle between /v1/restore
    // (continues the checkpoint's OWN endpoint unchanged) and /score (teacher-forced, no
    // continuation): a caller gets to ask "what if, starting from position P, the run had gone
    // differently" and get back a receipt for exactly how exact that answer is.
    //
    // THE central correctness rule (the regime split): truncating INTO the generated tail
    // (truncate_to > checkpoint.prompt_tokens) is bit-exact via a live-KV evict + single-token
    // bridge decode, because the ORIGINAL decode of that boundary token was ALSO a single-token
    // batch (generate_ar's steady-state loop decodes one token at a time -- see generate_ar.cpp).
    // Truncating AT OR BEFORE the prompt boundary must NOT use that bridge: the prompt's last token
    // was originally decoded INSIDE the one big prefill batch, and llama.cpp's batched attention
    // kernel is not guaranteed bit-identical to a shape-1 decode of the same token (the documented
    // batch-shape landmine -- it has bitten this project three times already; see /v1/checkpoint's
    // prefill_to doc comment above). That regime re-prefills tokens[0, truncate_to) as ONE batch
    // instead, reproducing the original prefill's own batch shape. EngineCheckpoint::prompt_tokens
    // is the boundary this split is computed from; a checkpoint that never had it populated (0 =
    // unknown) is refused rather than guessed at.
    //
    // {checkpoint_id, truncate_to, max_tokens?, intervention?: {type, ...}, checkpoint_on_finish?}
    // intervention.type in {"none","force_token","sampling","steer","residual_write"} (deliberately
    // narrow); see the field-level comments below for each type's shape.
    svr.Post("/v1/execution-fork", [&](const httplib::Request& req, httplib::Response& res) {
        try {
            json body = json::parse(req.body, nullptr, false);
            if (body.is_discarded() || !body.is_object())
                throw std::invalid_argument("invalid JSON body");

            const std::string ckpt_id = body.value("checkpoint_id", std::string());
            if (ckpt_id.empty()) throw std::invalid_argument("need checkpoint_id");
            if (!body.contains("truncate_to") || !body["truncate_to"].is_number_integer())
                throw std::invalid_argument("need integer truncate_to");
            const int truncate_to = body["truncate_to"].get<int>();
            const int max_tokens_raw = body.value("max_tokens", 64);
            const bool want_ckpt = body.value("checkpoint_on_finish", false);

            EngineCheckpoint ckpt;
            {
                std::lock_guard<std::mutex> lk(ckpt_mtx);
                auto it = checkpoints.find(ckpt_id);
                if (it == checkpoints.end()) {
                    res.status = 404;
                    res.set_content(json{{"error", "unknown checkpoint_id"}}.dump(), "application/json");
                    return;
                }
                ckpt = it->second;
            }
            // Fail closed (no silent fallback): a checkpoint that never had prompt_tokens populated
            // cannot honestly support the regime split -- refuse rather than guess which side of the
            // batch-shape landmine truncate_to falls on.
            if (ckpt.prompt_tokens <= 0)
                throw std::invalid_argument(
                    "checkpoint has no recorded prompt_tokens; execution-fork cannot determine the "
                    "reprefill/live_kv regime for it");
            if (truncate_to < 1 || truncate_to > ckpt.n_past)
                throw std::invalid_argument("truncate_to out of range [1, checkpoint n_past=" +
                                            std::to_string(ckpt.n_past) + "]");

            // --- Parse + validate the intervention up front, before touching the pool -- a typed
            // 400 for an unsupported/malformed request, never a silent downgrade. -----------------
            const json intervention = body.value("intervention", json{{"type", "none"}});
            if (!intervention.is_object() || !intervention.contains("type") ||
                !intervention["type"].is_string())
                throw std::invalid_argument("intervention must be an object with a string 'type'");
            const std::string itype = intervention["type"].get<std::string>();
            static const std::set<std::string> kKnownTypes = {
                "none", "force_token", "sampling", "steer", "residual_write"};
            if (kKnownTypes.find(itype) == kKnownTypes.end())
                throw std::invalid_argument("unknown intervention.type: " + itype);

            // "force_token": {"type":"force_token","token_id":1234} -- commit this token AT the
            // fork point (no sampling, no draw consumed), then continue normally.
            int force_token_id = -1;
            if (itype == "force_token") {
                if (!intervention.contains("token_id") || !intervention["token_id"].is_number_integer())
                    throw std::invalid_argument("force_token requires integer token_id");
                force_token_id = intervention["token_id"].get<int>();
                if (force_token_id < 0 || force_token_id >= model->config().vocab_size)
                    throw std::invalid_argument("force_token token_id outside the model vocabulary "
                                                "(vocab_size " +
                                                std::to_string(model->config().vocab_size) + ")");
            }

            // "steer": {"type":"steer","steer_vec":[...],"steer_layer":N,"steer_coef":X} -- replaces
            // whatever the checkpoint recorded. {"type":"steer","clear":true} removes it instead.
            bool steer_clear = false;
            std::vector<float> req_steer_vec;
            double req_steer_coef = 1.0;
            int req_steer_layer = 0;
            if (itype == "steer") {
                if (intervention.value("clear", false)) {
                    steer_clear = true;
                } else {
                    if (!intervention.contains("steer_vec") || !intervention["steer_vec"].is_array() ||
                        intervention["steer_vec"].empty())
                        throw std::invalid_argument("steer requires a non-empty steer_vec (or clear:true)");
                    req_steer_vec = intervention["steer_vec"].get<std::vector<float>>();
                    req_steer_coef = intervention.value("steer_coef", 1.0);
                    req_steer_layer = intervention.value("steer_layer", 0);
                }
            }

            // "residual_write": {"type":"residual_write","layer":L,"position":P,"values":[...]} --
            // one write, reusing the SAME validation /score's write path uses (add_write_state:
            // layer in [1, n_layer), values.size() == n_embd; n_embd is only known after the lease).
            int write_layer = 0, write_pos = 0;
            std::vector<float> write_values;
            if (itype == "residual_write") {
                if (!intervention.contains("layer") || !intervention.contains("position") ||
                    !intervention.contains("values") || !intervention["values"].is_array())
                    throw std::invalid_argument("residual_write requires layer, position, values");
                write_layer = intervention["layer"].get<int>();
                write_pos = intervention["position"].get<int>();
                write_values = intervention["values"].get<std::vector<float>>();
                if (write_pos < 0 || write_pos >= n_ctx)
                    throw std::invalid_argument("residual_write position out of range [0, n_ctx=" +
                                                std::to_string(n_ctx) + ")");
            }

            // --- Sampler state at the fork point: per-field checkpoint inheritance, request wins
            // per-field on override (mirrors /v1/restore's checkpoint-vs-request precedence). --------
            SampleConfig sample;  // starts greedy
            std::string sampler_source = "checkpoint";
            if (ckpt.has_sampler) {
                sample.temperature = ckpt.temperature;
                sample.rep_penalty = ckpt.rep_penalty;
                sample.top_k = ckpt.top_k;
                sample.top_p = ckpt.top_p;
                sample.seed = ckpt.seed;
            } else {
                sampler_source = "request";  // nothing to inherit; greedy defaults stand unless overridden
            }
            bool sampling_override = false;
            if (itype == "sampling") {
                if (intervention.contains("temperature")) {
                    sample.temperature = intervention["temperature"].get<double>();
                    sampling_override = true;
                }
                if (intervention.contains("rep_penalty")) {
                    sample.rep_penalty = intervention["rep_penalty"].get<double>();
                    sampling_override = true;
                }
                if (intervention.contains("top_k")) {
                    sample.top_k = intervention["top_k"].get<int>();
                    sampling_override = true;
                }
                if (intervention.contains("top_p")) {
                    sample.top_p = intervention["top_p"].get<double>();
                    sampling_override = true;
                }
                if (intervention.contains("seed")) {
                    sample.seed = intervention["seed"].get<uint64_t>();
                    sampling_override = true;
                }
            }
            if (sampling_override) sampler_source = "request";
            // rng_discard = tokens already committed past the prompt boundary at the fork point --
            // the same count of draws a bit-exact continuous run would have consumed to get here
            // (one draw per sampled committed token, zero for greedy; see SampleConfig::rng_discard).
            sample.rng_discard = (sample.temperature > 0.0)
                                      ? static_cast<uint64_t>(truncate_to - ckpt.prompt_tokens)
                                      : 0;

            const bool live_kv = truncate_to > ckpt.prompt_tokens;
            const char* restore_mode = live_kv ? "live_kv_truncated" : "reprefill";

            GenerateConfig cfg;
            cfg.max_new = max_tokens_raw < 1 ? 1 : max_tokens_raw;

            ContextPool::Lease lease = pool.acquire();
            const int nl = (*lease).n_layer();

            // --- Steer resolution: the checkpoint's recorded steer applies UNLESS this intervention
            // IS "steer", which either replaces it (a new cvec, same layer-band convention as every
            // other steer path in this file) or removes it (clear:true). ---------------------------
            std::vector<float> cvec;
            int steer_lo = 0, steer_hi = 0;
            std::string steer_source = "none";
            if (itype == "steer") {
                if (steer_clear) {
                    steer_source = "cleared";
                } else {
                    if (static_cast<int>(req_steer_vec.size()) != (*lease).n_embd())
                        throw std::invalid_argument("steer_vec size must equal n_embd (" +
                                                    std::to_string((*lease).n_embd()) + ")");
                    if (req_steer_layer >= 1) {
                        steer_lo = steer_hi = (req_steer_layer < nl ? req_steer_layer : nl - 1);
                    } else {
                        const int tl = nl * 2 / 3;
                        steer_lo = (tl - 2 > 1 ? tl - 2 : 1);
                        steer_hi = (tl + 2 < nl ? tl + 2 : nl - 1);
                    }
                    if (steer_lo < 1) steer_lo = 1;
                    const int ne = static_cast<int>(req_steer_vec.size());
                    cvec.assign(static_cast<size_t>(ne) * nl, 0.0f);
                    for (int L = steer_lo; L <= steer_hi; ++L) {
                        if (L < 1 || L >= nl) continue;
                        float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                        for (int i = 0; i < ne; ++i)
                            slice[i] = static_cast<float>(req_steer_coef * req_steer_vec[i]);
                    }
                    steer_source = "request";
                }
            } else if (ckpt.has_steer && !ckpt.steer_cvec.empty()) {
                cvec = ckpt.steer_cvec;
                steer_lo = ckpt.steer_lo;
                steer_hi = ckpt.steer_hi;
                steer_source = "checkpoint";
            }

            // Every exit path (success OR exception) leaves the pooled context clean -- steer/write
            // must never leak into the next request on this lease (the same discipline /v1/restore
            // and the completions handler already follow).
            bool steer_applied = false, write_applied = false;
            auto cleanup = [&]() {
                if (steer_applied) (*lease).clear_steer();
                if (write_applied) (*lease).clear_write();
            };
            try {
                if (!cvec.empty()) {
                    (*lease).set_steer(cvec, steer_lo, steer_hi);
                    steer_applied = true;
                }
                if (itype == "residual_write") {
                    if (!(*lease).add_write_state(write_layer, {write_pos}, write_values)) {
                        throw std::invalid_argument(
                            "residual_write rejected: layer must be in [1, n_layer) and values.size "
                            "must equal n_embd (n_embd " + std::to_string((*lease).n_embd()) + ")");
                    }
                    write_applied = true;
                }

                const int* force_ptr = (itype == "force_token") ? &force_token_id : nullptr;
                GenerateResult r;
                if (live_kv) {
                    // Order of operations, all under this one lease: restore checkpoint -> truncate
                    // -> bridge-decode the boundary token -> continue. generate_ar's resume_from +
                    // resume_truncate_to do exactly this (see generate_ar.cpp).
                    r = generate_ar(*lease, ckpt.tokens, cfg, {}, sample, nullptr,
                                    nullptr, 0, nullptr, nullptr, &ckpt, truncate_to, force_ptr);
                } else {
                    // Reprefill: tokens[0, truncate_to) as ONE batch, reproducing the original
                    // prefill's own shape -- never the live-KV bridge (see the route doc above).
                    const std::vector<int> reprefill_prompt(ckpt.tokens.begin(),
                                                             ckpt.tokens.begin() + truncate_to);
                    r = generate_ar(*lease, reprefill_prompt, cfg, {}, sample, nullptr,
                                    nullptr, 0, nullptr, nullptr, nullptr, -1, force_ptr);
                }

                json applied{{"type", itype}};
                if (itype == "force_token") {
                    applied["token_id"] = force_token_id;
                } else if (itype == "sampling") {
                    applied["temperature"] = sample.temperature;
                    applied["rep_penalty"] = sample.rep_penalty;
                    applied["top_k"] = sample.top_k;
                    applied["top_p"] = sample.top_p;
                    applied["seed"] = sample.seed;
                } else if (itype == "steer") {
                    if (steer_clear) {
                        applied["cleared"] = true;
                    } else {
                        applied["steer_layer_lo"] = steer_lo;
                        applied["steer_layer_hi"] = steer_hi;
                        applied["steer_coef"] = req_steer_coef;
                    }
                } else if (itype == "residual_write") {
                    applied["layer"] = write_layer;
                    applied["position"] = write_pos;
                }

                json resp{
                    {"text", r.text},
                    {"tokens", r.generated},
                    {"prompt_len", ckpt.prompt_tokens},
                    {"n_past_restored", truncate_to},
                    {"restore_mode", restore_mode},
                    {"exactness", {{"source", live_kv ? "live_kv" : "reprefill"},
                                  {"truncation_regime", live_kv ? "generated_token" : "prompt_boundary"},
                                  {"boundary_shape_true", true}}},
                    {"sampler_source", sampler_source},
                    {"steer_source", steer_source},
                    {"intervention_applied", applied},
                    {"finish_reason", finish_reason(r.reason)},
                };

                if (want_ckpt) {
                    // Top-up decode, same shape-preserving move checkpoint_on_finish uses above: the
                    // loop's final committed token may not be decoded into the KV yet.
                    const int n_board = static_cast<int>(r.board.size());
                    (*lease).evict_from(n_board - 1);
                    (*lease).ar_forward({r.board[static_cast<size_t>(n_board - 1)]}, n_board - 1);
                    EngineCheckpoint out_ckpt = (*lease).save_checkpoint(r.board, n_board);
                    out_ckpt.prompt_tokens = ckpt.prompt_tokens;  // the fork's own prompt boundary is unchanged
                    if (sample.temperature > 0.0) {
                        out_ckpt.has_sampler = true;
                        out_ckpt.temperature = sample.temperature;
                        out_ckpt.rep_penalty = sample.rep_penalty;
                        out_ckpt.top_k = sample.top_k;
                        out_ckpt.top_p = sample.top_p;
                        out_ckpt.seed = sample.seed;
                        // Cumulative draws a bit-exact continuous run would have consumed to reach
                        // THIS checkpoint: the fast-forward baseline plus this call's own draws,
                        // minus one if force_token supplied the first slot without sampling it.
                        const long long draws_this_call = static_cast<long long>(r.steps_total) -
                                                          (itype == "force_token" ? 1 : 0);
                        out_ckpt.rng_draws = sample.rng_discard +
                            static_cast<uint64_t>(draws_this_call > 0 ? draws_this_call : 0);
                    }
                    if (!cvec.empty()) {
                        out_ckpt.has_steer = true;
                        out_ckpt.steer_cvec = cvec;
                        out_ckpt.steer_lo = steer_lo;
                        out_ckpt.steer_hi = steer_hi;
                    }
                    const std::string cid = make_id("ckpt-");
                    {
                        std::lock_guard<std::mutex> lk(ckpt_mtx);
                        if (static_cast<int>(checkpoints.size()) >= kMaxCheckpoints)
                            checkpoints.erase(checkpoints.begin());
                        checkpoints[cid] = std::move(out_ckpt);
                    }
                    resp["checkpoint_id"] = cid;
                }

                cleanup();
                res.set_content(dump_json(resp), "application/json");
            } catch (...) {
                cleanup();
                throw;
            }
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", std::string("execution-fork failed: ") + e.what()}}.dump(),
                            "application/json");
        }
    });

    svr.Post("/v1/completions",
             [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, false); });
    svr.Post("/v1/infill",
             [&](const httplib::Request& req, httplib::Response& res) { handle(req, res, true); });

    // /v1/revise and /v1/board (the diffusion "resolve a selection"/board-restore routes) were
    // removed with the diffusion generator (THE_CUT): both required full bidirectional re-masking,
    // which no longer exists in this engine build. studio/heavn's Resolve edit mode degrades
    // honestly (it already gates on engine mode == "diffusion", which this build never reports).

    std::fprintf(stderr, "[clozn-server] %s %s, %d worker(s) on http://%s:%d  (model: %s)\n",
                 ar_mode ? "autoregressive" : "diffusion",
                 gpu_layers > 0 ? "GPU" : "CPU", pool.size(), host.c_str(), port, model_path.c_str());
    if (!svr.listen(host, port)) {
        std::fprintf(stderr, "failed to bind %s:%d\n", host.c_str(), port);
        return 1;
    }
    return 0;
}
