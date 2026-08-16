// serve/server_shared.hpp -- shared white-box helpers + state structs for the split clozn-server.
// Phase 12.4 of the repo reorg: the native serve monolith (clozn_server.cpp) split into server_main +
// route-family TUs (routes_*.cpp) that read state through a ServerContext. This header carries the pieces
// EVERY route family shares -- the JSON/SSE helpers (dump_json, sse_data, board_layout_json, tensor_json_f32,
// ...), the config parsers (config_from/sample_from/...), and the white-box state structs (StateStepBuilder,
// SaeServe, JlensServe, ContextPool) + the concept-probe / steer-vector / SAE builders. Header-only (free
// functions are `inline`), so multiple route TUs include it with no cross-TU linkage. Mechanical extraction:
// same code as the old anonymous namespace, just relocated (the map's request_helpers/sse/probes modules
// consolidated -- a further granular header split is a trivial follow-up).
#pragma once

#include "nlohmann/json.hpp"

#include "clozn/events.hpp"
#include "clozn/generate.hpp"
#include "clozn/generate_ar.hpp"
#include "clozn/model_ggml.hpp"
#include "clozn/probe.hpp"
#ifdef CLOZN_SAE
#include "clozn/sae.hpp"  // on-device SAE feature readout (--sae; built with CLOZN_BUILD_SAE)
#endif

#include "ggml.h"       // J-lens: standalone CPU ggml graph (J_l @ h -> rms_norm -> head)
#include "ggml-cpu.h"   // ggml_graph_compute_with_ctx
#include "gguf.h"       // read the GGUF's own output_norm.weight + output.weight for the J-lens head

#include <algorithm>
#include <atomic>
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
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace clozn {

using json = nlohmann::json;

// The worker <-> supervisor wire-contract version (see protocol/SPEC.md + clozn/protocol.py). A MAJOR
// bump is breaking -- the Python supervisor refuses a worker whose major it doesn't support; a MINOR
// bump is additive. Pinned identically in clozn/protocol.py and protocol/fixtures/handshake.json; the
// golden-fixture test (tests/test_protocol_handshake.py) fails the moment the three drift.
inline constexpr const char* PROTOCOL_VERSION = "1.1";

inline std::atomic<uint64_t> g_req_counter{0};
inline std::string make_id(const char* prefix) {
    return std::string(prefix) + std::to_string(g_req_counter.fetch_add(1));
}

// dLLM reason -> OpenAI finish_reason.
inline const char* finish_reason(const std::string& reason) {
    return (reason == "eos" || reason == "stop") ? "stop" : "length";
    // length | steps_exhausted -> "length"
}

inline void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

inline GenerateConfig config_from(const json& body) {
    GenerateConfig cfg;
    int mn = body.value("max_tokens", 32); cfg.max_new = mn < 1 ? 1 : mn;   // clamp >= 1: the generators
    int st = body.value("steps", 8);       cfg.steps   = st < 1 ? 1 : st;   // throw on < 1, and on the
    //                                          STREAMING path an uncaught throw silently abort()s the engine.
    cfg.block_len = body.value("block_len", 0);
    cfg.topk = body.value("topk", -1);
    return cfg;
}

// Sampling: opt-in temperature + repetition penalty. Defaults (T=0, penalty=1) keep greedy decoding,
// so omitting them is byte-identical to before.
inline SampleConfig sample_from(const json& body) {
    SampleConfig sample;
    sample.temperature = body.value("temperature", 0.0);
    sample.rep_penalty = body.value("rep_penalty", 1.0);
    sample.top_k = body.value("top_k", 0);
    sample.top_p = body.value("top_p", 1.0);
    sample.seed = body.value("seed", static_cast<uint64_t>(0));
    return sample;
}

// SSE data payload for one event, enriched for the browser viz (which has no tokenizer):
//   gen_started      += prompt_pieces[] (decoded prompt tokens, so each shows as its own cell)
//   tokens_committed += piece per token (the committed text)
// All other events pass through as the plain §5.1 wire form. Presentation only — the core events
// stay {prompt_tokens,...} / {pos,id,conf}.
// A decoded token piece can be a PARTIAL multi-byte UTF-8 sequence: byte-fallback tokens split a character
// (e.g. "½", an em-dash, an emoji) across token boundaries, so ONE token's piece may end mid-codepoint.
// nlohmann's default (strict) serializer THROWS on the incomplete bytes -- and on the streaming path that
// aborted the whole generation mid-reply (truncated text, and the trailing gen_finished/finish_reason frames
// never sent). Serialize with `replace` so an invalid byte becomes U+FFFD and a split char never kills the
// stream: the reply completes and the final frames arrive. (A split char shows as U+FFFD until the model's
// next multi-byte token; a full buffering fix is a separate refinement.)
inline std::string dump_json(const json& j) {
    return j.dump(-1, ' ', false, json::error_handler_t::replace);
}

// ---- cooperative cancellation ------------------------------------------------------------------
// The Python gateway (clozn/server/sse.py) detects a client disconnect (a failed write to the far
// end) well before the C++ worker would ever notice on its own -- a generation loop has no socket of
// its own to watch; it just calls on_event() and keeps computing until it naturally finishes. Two
// independent signals feed ONE per-request flag so either is enough to stop generation PROMPTLY
// (within one committed token/pass, not at the natural end):
//   (1) POST /cancel {"req": id} -- an explicit "stop this", e.g. once the gateway notices the client
//       is gone.
//   (2) A failed SSE frame write (StreamEnvelope::write_raw below) -- the socket is ALREADY dead (tab
//       closed, curl Ctrl-C'd, the gateway's resp.close()), so this fires with no round-trip at all --
//       an ordinary disconnect is caught even if nothing ever calls /cancel.
// GenerationCancelled unwinds out of the generator through the SAME exception path an n_ctx-exceeded
// or decode-failure throw already takes (server_main.cpp's per-request cleanup()/catch(...) already
// restores the pooled context on ANY throw) -- so this needed no change to that machinery, only a new
// throw site (StreamEnvelope::check_cancelled, called once per emitted event).
struct GenerationCancelled : std::runtime_error {
    GenerationCancelled() : std::runtime_error("cancelled") {}
};

// Maps a live streaming request's id (StreamEnvelope::req -- the same cmpl-/infill-/revise-/board- id
// every frame is already stamped with) to its cancel flag, so POST /cancel (handled on httplib's
// listener thread) can reach a generation running on a DIFFERENT worker thread. Registered when a
// stream starts, erased when it ends (Guard, RAII) -- cancelling an id from a finished/unknown request
// just reports cancelled:false, never an error: cancellation is inherently racy (the generation may
// finish the instant before or after the cancel arrives), and both outcomes are fine.
class CancelRegistry {
public:
    std::shared_ptr<std::atomic<bool>> register_request(const std::string& req_id) {
        auto flag = std::make_shared<std::atomic<bool>>(false);
        std::lock_guard<std::mutex> lk(mtx_);
        flags_[req_id] = flag;
        return flag;
    }
    void unregister_request(const std::string& req_id) {
        std::lock_guard<std::mutex> lk(mtx_);
        flags_.erase(req_id);
    }
    // true => a live request was found and flagged; false => nothing to cancel (done/unknown id).
    bool cancel(const std::string& req_id) {
        std::lock_guard<std::mutex> lk(mtx_);
        auto it = flags_.find(req_id);
        if (it == flags_.end()) return false;
        it->second->store(true, std::memory_order_relaxed);
        return true;
    }
    // RAII unregister on scope exit (every path out of a streaming handler -- normal finish, cancel,
    // or an unrelated throw) so a handler with several exit points never has to remember cleanup.
    struct Guard {
        CancelRegistry& reg;
        std::string req_id;
        ~Guard() { reg.unregister_request(req_id); }
    };

private:
    std::mutex mtx_;
    std::map<std::string, std::shared_ptr<std::atomic<bool>>> flags_;
};

// Per-request SSE frame envelope. Every JSON frame on a native stream is stamped with the request id
// (`req`, the same cmpl-/infill-/revise- id the final frame carries) and a monotonic per-request
// sequence number (`seq`, from 0) -- so a consumer can correlate a frame to its request and detect a
// dropped or reordered frame. `[DONE]` and any non-object frame (a hand-serialized error, say) pass
// through untouched: the envelope never drops a frame it can't stamp. Frames route through ONE counter
// per request, so the per-step frames and the trailing final frame share one contiguous sequence.
//
// Cooperative cancel: `write` reports success/failure (cpp-httplib's DataSink::write already returns
// bool -- a failed write means the socket is gone), and a failure flips `cancel` when one is wired up
// (see CancelRegistry above) -- so a dead client is noticed the very next time ANY frame is sent, no
// polling needed. check_cancelled() is what the generation loop's on_event wrapper calls once per
// emitted event to actually stop (see server_main.cpp's streaming handlers).
struct StreamEnvelope {
    std::string req;
    std::function<bool(const std::string&)> write;   // returns false on a broken/closed send
    uint64_t seq = 0;
    std::shared_ptr<std::atomic<bool>> cancel;        // optional: set by /cancel or a failed write

    void write_raw(const std::string& s) {
        if (!write(s) && cancel) cancel->store(true, std::memory_order_relaxed);
    }
    void frame(json f) {                       // stamp a json frame, then write it
        f["req"] = req;
        f["seq"] = seq++;
        write_raw("data: " + dump_json(f) + "\n\n");
    }
    void frame_str(const std::string& dumped) {  // a pre-serialized frame (sse_data): parse, stamp, write
        json f = json::parse(dumped, nullptr, /*allow_exceptions=*/false);
        if (f.is_discarded() || !f.is_object()) { write_raw("data: " + dumped + "\n\n"); return; }
        frame(std::move(f));
    }
    void done() { write_raw("data: [DONE]\n\n"); }
    // Throws GenerationCancelled if cancellation was signaled -- call once per emitted event from
    // inside the on_event callback (AR mode emits at least one event per committed token, diffusion at
    // least one per pass), so a cancel is noticed within one token/pass, not just at stream end.
    void check_cancelled() const {
        if (cancel && cancel->load(std::memory_order_relaxed)) throw GenerationCancelled();
    }
};

inline std::string sse_data(const Event& e, const GgmlModel& model, const std::vector<int>& prompt_ids,
                     const std::vector<int>& suffix_ids) {
    if (const auto* gs = std::get_if<GenStarted>(&e)) {
        json pieces = json::array();
        for (int id : prompt_ids) pieces.push_back(model.decode({id}));
        json sfx = json::array();
        for (int id : suffix_ids) sfx.push_back(model.decode({id}));  // infill: the fixed right-context
        return dump_json(json{{"t", gs->t}, {"type", "gen_started"}, {"prompt_tokens", gs->prompt_tokens},
                    {"block_len", gs->block_len}, {"max_new", gs->max_new},
                    {"prompt_pieces", pieces}, {"suffix_pieces", sfx}});
    }
    if (const auto* tc = std::get_if<TokensCommitted>(&e)) {
        json items = json::array();
        for (const auto& it : tc->items)
            items.push_back({{"pos", it.pos}, {"id", it.id}, {"conf", it.conf},
                             {"piece", model.decode({it.id})}});
        return dump_json(json{{"t", tc->t}, {"type", "tokens_committed"}, {"block", tc->block},
                    {"items", items}});
    }
    if (const auto* sl = std::get_if<StepLens>(&e)) {
        json pieces = json::array();
        for (int id : sl->ids) pieces.push_back(model.decode({id}));     // decode candidates (viz has no tokenizer)
        return dump_json(json{{"t", sl->t}, {"type", "step_lens"}, {"block", sl->block}, {"k", sl->k},
                    {"positions", sl->positions}, {"ids", sl->ids}, {"probs", sl->probs},
                    {"pieces", pieces}});
    }
    return to_jsonl_line(e);
}


// ============================ state-stream protocol (phase 1.2) ============================
// The inspector-facing wire form (protocol/SPEC.md). The engine's §5.1 events FOLD into canonical
// StateStep frames: {step, token, state, readouts, meta}. One forward pass's events (which all share
// the same `t`) collapse into one StateStep; gen_started/finished become control frames. Gated by the
// request flag protocol:true — without it, the legacy sse_data(...) frames stream unchanged, so the
// existing viz keeps working. See SPEC.md "Engine §5.1 event -> StateStep mapping" + "The wire".

// Base64 (standard alphabet, padded) of raw bytes — for tensor `data` on the wire.
inline std::string base64_encode(const uint8_t* data, size_t len) {
    static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string out;
    out.reserve(((len + 2) / 3) * 4);
    size_t i = 0;
    for (; i + 3 <= len; i += 3) {
        const uint32_t n = (uint32_t(data[i]) << 16) | (uint32_t(data[i + 1]) << 8) | data[i + 2];
        out.push_back(tbl[(n >> 18) & 63]); out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(tbl[(n >> 6) & 63]);  out.push_back(tbl[n & 63]);
    }
    if (i < len) {  // 1 or 2 trailing bytes
        uint32_t n = uint32_t(data[i]) << 16;
        const bool two = (i + 1 < len);
        if (two) n |= uint32_t(data[i + 1]) << 8;
        out.push_back(tbl[(n >> 18) & 63]);
        out.push_back(tbl[(n >> 12) & 63]);
        out.push_back(two ? tbl[(n >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

// A tensor on the wire: {dtype, shape, data} where data = base64 of the little-endian raw bytes
// (SPEC.md "Tensors on the wire"). We tap float32 activations; x86/CUDA are little-endian, so the
// in-memory floats ARE the little-endian bytes — a straight reinterpret, no byte-swizzling.
inline json tensor_json_f32(const std::vector<float>& values, std::vector<int> shape) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(values.data());
    return json{{"dtype", "float32"}, {"shape", std::move(shape)},
                {"data", base64_encode(bytes, values.size() * sizeof(float))}};
}

// Folds the §5.1 event stream into StateStep frames and writes them to an SSE sink. One StateStep
// per forward pass (events sharing a `t`); gen_started/finished + block start/finalize fold into the
// running `meta`. `state_full` controls whether the heavy raw-activation tensor rides each frame
// (state="full") or is omitted (the light default). The builder is stateful across the run: it
// accumulates the current pass, and FLUSHES it (emits one frame) when the next pass begins (a new
// tokens_committed at a higher t) or at gen_finished.
class StateStepBuilder {
public:
    StateStepBuilder(const GgmlModel& model, const char* substrate, bool state_full,
                     std::function<void(json)> emit)
        : model_(model), substrate_(substrate), state_full_(state_full), emit_(std::move(emit)) {}

    void on_event(const Event& e) {
        if (const auto* gs = std::get_if<GenStarted>(&e)) {
            // Control frame: stream begin. prompt/total counts in meta.
            json meta{{"kind", "begin"}, {"substrate", substrate_}, {"block_len", gs->block_len},
                      {"prompt_tokens", gs->prompt_tokens}, {"max_new", gs->max_new}};
            emit_control(gs->t, meta);
            return;
        }
        if (const auto* bs = std::get_if<BlockStarted>(&e)) {
            block_ = bs->block;
            span_ = json::array({bs->span.first, bs->span.second});
            return;
        }
        if (const auto* tr = std::get_if<TokensRevised>(&e)) {
            // A revision is its own StateStep (diffusion-only). Flush any open commit first so the
            // revise frame is distinct, then emit the revise frame immediately.
            flush();
            json items = json::array();
            for (const auto& it : tr->items)
                items.push_back({{"pos", it.pos}, {"old", it.old}, {"id", it.id}, {"conf", it.conf},
                                 {"piece", model_.decode({it.id})}});
            json meta = base_meta();
            meta["kind"] = "revise";
            json frame{{"step", tr->t}, {"token", nullptr}, {"state", nullptr},
                       {"readouts", json::array()}, {"meta", meta}, {"revised", items}};
            write_frame(frame);
            return;
        }
        if (const auto* tc = std::get_if<TokensCommitted>(&e)) {
            flush();             // a new pass begins -> close out the previous StateStep
            open_ = true;
            step_ = tc->t;
            // token = the committed id(s) for this pass (+ pieces for the tokenizer-free consumer).
            token_ = json::array();
            json confs = json::array();
            for (const auto& it : tc->items) {
                token_.push_back({{"pos", it.pos}, {"id", it.id}, {"piece", model_.decode({it.id})}});
                confs.push_back(it.conf);
            }
            commit_confs_ = std::move(confs);
            return;
        }
        if (const auto* ss = std::get_if<StepStats>(&e)) {
            ensure_open(ss->t);
            stats_ = json{{"committed", ss->committed}, {"remaining", ss->remaining},
                          {"step", ss->step}, {"ms", ss->ms}, {"cache_hit", ss->cache_hit}};
            return;
        }
        if (const auto* sf = std::get_if<StepFeatures>(&e)) {
            ensure_open(sf->t);
            // One Readout per concept. value = the per-slot scores for that concept (position-major
            // sliced into a per-concept vector). Honesty invariant: confidence travels; causal_verified
            // is null (probes are correlational until patched — SPEC's wire honesty rule).
            const int K = static_cast<int>(sf->features.size());
            const int rows = K > 0 ? static_cast<int>(sf->scores.size()) / K : 0;
            for (int k = 0; k < K; ++k) {
                json per_slot = json::array();
                double maxabs = 0.0;
                for (int r = 0; r < rows; ++r) {
                    const float v = sf->scores[static_cast<size_t>(r) * K + k];
                    per_slot.push_back(v);
                    if (std::fabs(v) > maxabs) maxabs = std::fabs(v);
                }
                readouts_.push_back({{"name", sf->features[k]},
                                     {"value", {{"positions", sf->positions}, {"scores", per_slot}}},
                                     {"confidence", maxabs}, {"causal_verified", nullptr}});
            }
            return;
        }
        if (const auto* sl = std::get_if<StepLens>(&e)) {
            ensure_open(sl->t);
            // The logit-lens readout: top-k candidates+probs per requested slot. value carries the
            // decoded pieces too (the consumer has no tokenizer). confidence = the top-1 prob seen.
            json per_pos = json::array();
            double top_conf = 0.0;
            const int k = sl->k;
            for (size_t r = 0; r < sl->positions.size(); ++r) {
                json cand = json::array();
                for (int j = 0; j < k; ++j) {
                    const size_t idx = r * static_cast<size_t>(k) + j;
                    if (idx >= sl->ids.size()) break;
                    const float prob = sl->probs[idx];
                    if (j == 0 && prob > top_conf) top_conf = prob;
                    cand.push_back({{"id", sl->ids[idx]}, {"prob", prob},
                                    {"piece", model_.decode({sl->ids[idx]})}});
                }
                per_pos.push_back({{"pos", sl->positions[r]}, {"candidates", cand}});
            }
            readouts_.push_back({{"name", "logit-lens"}, {"value", per_pos},
                                 {"confidence", top_conf}, {"causal_verified", nullptr}});
            return;
        }
        if (const auto* sa = std::get_if<StepActivations>(&e)) {
            ensure_open(sa->t);
            // The heavy state: raw per-position hidden state. Held; attached to the frame on flush()
            // only when state="full" (the light frame omits it). Encoded {dtype,shape,data}.
            if (state_full_) {
                state_ = json::object();
                state_["positions"] = sa->positions;
                state_["hidden"] = tensor_json_f32(
                    sa->values, {static_cast<int>(sa->positions.size()), sa->n_embd});
            }
            return;
        }
        if (const auto* gf = std::get_if<GenFinished>(&e)) {
            flush();  // close the last pass before the end-of-stream control frame
            json meta{{"kind", "end"}, {"substrate", substrate_}, {"reason", gf->reason},
                      {"new_tokens", gf->new_tokens}, {"wall_ms", gf->wall_ms},
                      {"steps_total", gf->steps_total}, {"tok_per_s", gf->tok_per_s}};
            meta["timing"] = json::parse(worker_timing_json(*gf));
            emit_control(gf->t, meta);
            return;
        }
    }

    // Flush any pass still open (defensive; gen_finished normally does this).
    void finish() { flush(); }

private:
    json base_meta() const {
        json m{{"substrate", substrate_}, {"block", block_}};
        if (!span_.is_null()) m["span"] = span_;
        return m;
    }
    void ensure_open(int t) {
        if (!open_) { open_ = true; step_ = t; }
    }
    void emit_control(int step, const json& meta) {
        json frame{{"step", step}, {"token", nullptr}, {"state", nullptr},
                   {"readouts", json::array()}, {"meta", meta}};
        write_frame(frame);
    }
    void write_frame(json frame) { emit_(std::move(frame)); }  // envelope stamps req/seq, then writes

    // Emit the accumulated pass as one StateStep, then reset for the next pass.
    void flush() {
        if (!open_) return;
        json meta = base_meta();
        meta["kind"] = "step";
        if (!stats_.is_null()) meta.update(stats_);
        if (!commit_confs_.is_null()) meta["confidence"] = commit_confs_;
        json frame{{"step", step_},
                   {"token", token_.is_null() ? json(json::array()) : token_},
                   {"state", state_.is_null() ? json(nullptr) : state_},
                   {"readouts", readouts_},
                   {"meta", meta}};
        write_frame(frame);
        // reset pass accumulators
        open_ = false;
        token_ = json();
        state_ = json();
        stats_ = json();
        commit_confs_ = json();
        readouts_ = json::array();
    }

    const GgmlModel& model_;
    const char* substrate_;
    bool state_full_;
    std::function<void(json)> emit_;

    // run-scoped
    int block_ = 0;
    json span_ = json();

    // pass-scoped
    bool open_ = false;
    int step_ = 0;
    json token_ = json();
    json state_ = json();
    json stats_ = json();
    json commit_confs_ = json();
    json readouts_ = json::array();
};

// Per-position board layout for the white-box SNAPSHOT: {pos,id,masked,piece}. Read-only introspection
// on every completion response; there is no restore endpoint any more (/v1/board was diffusion-only
// and was removed with the diffusion program -- an AR board never carries a real mask token anyway, so
// `masked` is honestly false for every position here).
inline json board_layout_json(const GgmlModel& model, const std::vector<int>& board, int mask_token) {
    json layout = json::array();
    for (size_t pos = 0; pos < board.size(); ++pos) {
        const bool masked = board[pos] == mask_token;
        layout.push_back({{"pos", static_cast<int>(pos)}, {"id", board[pos]}, {"masked", masked},
                          {"piece", masked ? std::string() : model.decode({board[pos]})}});
    }
    return layout;
}

// ---- on-device SAE feature readout (--sae <dir>; ROADMAP 3.3 wired into the server) ----
// When active, every featureful request taps the residual at the SAE's OWN layer, encodes the
// tapped rows on the GPU (clozn/sae.hpp: JumpReLU GEMV + the sae_topk kernel) and rides each pass's
// top-k features onto the stream as a SECOND StepFeatures event whose names are raw feature indices
// ("sae:<id>") — the same positions-x-features wire shape the concept probes already use, so
// StateStepBuilder / the inspector parse it unchanged. The Neuronpedia id -> label mapping stays
// host/Python side (research/np_labels_l15.json via brain_readout.py), by design: the engine ships
// indices, never a 131k-entry string table. The holder exists in every build so the run lambdas
// capture it uniformly; without CLOZN_BUILD_SAE it is permanently off and --sae refuses at startup.
struct SaeServe {
#ifdef CLOZN_SAE
    SaeEncoder enc;   // the device-resident encoder weights (loaded once at startup)
#endif
    bool on = false;  // loaded + n_embd == d_in verified; readouts ride featureful requests
    int layer = 0;    // the SAE's residual layer — the read tap moves here when on
    int k = 16;       // features kept per position (--sae-k)
};

#ifdef CLOZN_SAE
// One pass's SAE readout: encode the raw-activation event's rows (chunked so the encoder workspace
// stays bounded at ~16 MB regardless of diffusion block size) and fold every row's top-k into one
// StepFeatures over the UNION of lit features (score 0 where a feature missed a row's top-k).
// nullopt when nothing lit / dims mismatch / CUDA failure — the pass just has no SAE readout.
inline std::optional<StepFeatures> sae_features_from(const StepActivations& sa, SaeEncoder& enc, int k) {
    const int rows = static_cast<int>(sa.positions.size());
    if (rows <= 0 || sa.n_embd != enc.d_in()) return std::nullopt;
    constexpr int kRowChunk = 32;  // 0.5 MB/row of device workspace; 32 caps it while covering blocks
    std::vector<int32_t> idx;
    std::vector<float> val;
    idx.reserve(static_cast<size_t>(rows) * k);
    val.reserve(static_cast<size_t>(rows) * k);
    for (int r0 = 0; r0 < rows; r0 += kRowChunk) {
        const int rn = rows - r0 < kRowChunk ? rows - r0 : kRowChunk;
        std::vector<int32_t> ci;
        std::vector<float> cv;
        if (!enc.encode_topk(sa.values.data() + static_cast<size_t>(r0) * sa.n_embd, rn, k, ci, cv))
            return std::nullopt;
        idx.insert(idx.end(), ci.begin(), ci.end());
        val.insert(val.end(), cv.begin(), cv.end());
    }
    // Union of live features (value > 0; zero-valued slots are the top-k pad), ascending ids.
    std::vector<int32_t> feats;
    for (size_t i = 0; i < idx.size(); ++i)
        if (val[i] > 0.0f) feats.push_back(idx[i]);
    std::sort(feats.begin(), feats.end());
    feats.erase(std::unique(feats.begin(), feats.end()), feats.end());
    if (feats.empty()) return std::nullopt;

    StepFeatures sf;
    sf.t = sa.t;
    sf.block = sa.block;
    sf.positions = sa.positions;
    sf.features.reserve(feats.size());
    for (int32_t f : feats) sf.features.push_back("sae:" + std::to_string(f));
    sf.scores.assign(static_cast<size_t>(rows) * feats.size(), 0.0f);
    for (int r = 0; r < rows; ++r)
        for (int j = 0; j < k; ++j) {
            const size_t i = static_cast<size_t>(r) * k + j;
            if (val[i] <= 0.0f) continue;
            const size_t c = std::lower_bound(feats.begin(), feats.end(), idx[i]) - feats.begin();
            sf.scores[static_cast<size_t>(r) * feats.size() + c] = val[i];
        }
    return sf;
}
#endif  // CLOZN_SAE

// Wrap a run's event sink so each StepActivations (emitted whenever the white-box tap is on) also
// yields its SAE readout, BEFORE the activations event itself — both land in the same StateStep.
// Identity when inactive (or in a CLOZN_SAE-less build), so the plain path is byte-identical.
inline std::function<void(const Event&)> with_sae_readout(const std::function<void(const Event&)>& on_event,
                                                   SaeServe& sae, bool active) {
#ifdef CLOZN_SAE
    if (active && sae.on && on_event) {
        SaeEncoder* enc = &sae.enc;
        const int k = sae.k;
        return [on_event, enc, k](const Event& e) {
            if (const auto* sa = std::get_if<StepActivations>(&e))
                if (auto sf = sae_features_from(*sa, *enc, k)) on_event(Event(*sf));
            on_event(e);
        };
    }
#else
    (void)sae;
    (void)active;
#endif
    return on_event;
}

// ---- white-box concept probe calibration (Tier 2) ----
// Categorize a decoded token piece: "number" (has a digit), "punct" (no alphanumerics), "word"
// (all letters), or nullptr (mixed / unknown). Mirrors lab p4_dream_probe.category. ASCII-only
// (English calibration text); non-ASCII bytes just fall through to nullptr, which is fine here.
inline const char* token_category(const std::string& piece) {
    std::string s, low;
    for (unsigned char c : piece)
        if (!std::isspace(c)) { s.push_back(static_cast<char>(c)); low.push_back(static_cast<char>(std::tolower(c))); }
    if (s.empty()) return nullptr;
    bool any_digit = false, any_alnum = false, all_alpha = true;
    for (unsigned char c : s) {
        if (std::isdigit(c)) any_digit = true;
        if (std::isalnum(c)) any_alnum = true;
        if (!std::isalpha(c)) all_alpha = false;
    }
    if (any_digit) return "number";
    if (!any_alnum) return "punct";
    if (all_alpha) {
        // closed-class function words vs open-class content words — the syntactic-role split.
        static const std::set<std::string> kFunction = {
            "the","a","an","of","and","to","in","is","it","that","this","for","on","with","as","at",
            "by","or","but","not","are","was","were","be","been","being","i","you","he","she","they",
            "we","his","her","its","their","our","my","your","from","up","out","if","then","than","so",
            "no","do","does","did","have","has","had","will","would","can","could","should","may","me"
        };
        return kFunction.count(low) ? "function" : "content";
    }
    return nullptr;
}

// Build training-free diff-in-means category probes in the model's OWN mid-layer activation
// space: run a small labeled corpus through the adapter (which must have emit_activations on),
// standardize, and for each category take (its mean - the pooled mean of the other categories),
// unit-normalized. Self-contained — no lab->core probe transfer (the activation spaces differ).
//
// Two families of probes are built in one pass:
//   (1) PER-TOKEN: punct/number/function/content — each token is categorized individually.
//   (2) CONTRASTIVE (sentence-level): code/question — all tokens in a positive sentence contribute
//       to the positive class, all tokens in a negative sentence to the negative class. This
//       captures sentence-level concepts that don't reduce to per-token labels.
inline ConceptProbes calibrate_concept_probes(GgmlAdapter& ad, const GgmlModel& model) {
    // --- per-token calibration corpus (existing) ---
    static const std::vector<std::string> corpus = {
        "The quick brown fox jumps over the lazy dog near the old stone bridge.",
        "In 2024 the company sold 15 boats, 320 bikes, and 7 cars to 4 buyers.",
        "She said, \"Hello!\" Then he asked: why now? Nobody really knew.",
        "Pi is about 3.14159 and the speed of light is 299792458 meters per second.",
        "Prices fell 12 percent in March, rose 8 percent in April, and held in May.",
        "Wait, stop -- listen carefully; the answer matters more than the question.",
    };
    static const std::vector<std::string> cats = {"punct", "number", "function", "content"};

    // --- contrastive calibration pairs: {positive_sentences, negative_sentences, name} ---
    struct ContrastivePair {
        std::vector<std::string> pos;
        std::vector<std::string> neg;
        std::string name;
    };
    static const std::vector<ContrastivePair> contrastive = {
        {   // code vs prose
            {"def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
             "for i in range(10): result = data[i] * weights[i] + bias",
             "int main(int argc, char** argv) { return process(argc, argv); }"},
            {"The sun set behind the mountains as the birds flew south for the winter.",
             "She walked to the park and sat on the bench near the fountain.",
             "The report concluded that revenue grew steadily throughout the quarter."},
            "code"
        },
        {   // question vs statement
            {"What is the meaning of life and why does it matter?",
             "How many people live in Tokyo and what languages do they speak?",
             "Why did the experiment fail and what should we change next time?"},
            {"The meaning of life is a deeply personal question with many answers.",
             "About fourteen million people live in Tokyo and most speak Japanese.",
             "The experiment failed because the sample size was too small."},
            "question"
        },
    };

    int n_embd = 0;
    long long N = 0;
    std::vector<double> sum, sumsq;
    std::vector<std::vector<double>> cat_sum(cats.size());
    std::vector<long long> cat_cnt(cats.size(), 0);

    // Contrastive accumulators — sentence-level: all tokens in a pos sentence -> pos, neg -> neg.
    std::vector<std::vector<double>> con_pos_sum(contrastive.size());
    std::vector<std::vector<double>> con_neg_sum(contrastive.size());
    std::vector<long long> con_pos_cnt(contrastive.size(), 0);
    std::vector<long long> con_neg_cnt(contrastive.size(), 0);

    auto run_sentence = [&](const std::string& text) -> std::pair<std::vector<float>, size_t> {
        const std::vector<int> toks = model.encode(text);
        if (toks.size() < 2) return {{}, 0};
        const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
        const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
        if (fwd.activations.empty() || fwd.n_embd <= 0) return {{}, 0};
        if (n_embd == 0) {
            n_embd = fwd.n_embd;
            sum.assign(n_embd, 0.0);
            sumsq.assign(n_embd, 0.0);
            for (auto& cs : cat_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_pos_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_neg_sum) cs.assign(n_embd, 0.0);
        }
        return {fwd.activations, fwd.act_rows.size()};
    };

    // Pass 1: per-token corpus — accumulate per-category means + global stats.
    for (const std::string& text : corpus) {
        const std::vector<int> toks = model.encode(text);
        if (toks.size() < 2) continue;
        const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
        const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
        if (fwd.activations.empty() || fwd.n_embd <= 0) continue;
        if (n_embd == 0) {
            n_embd = fwd.n_embd;
            sum.assign(n_embd, 0.0);
            sumsq.assign(n_embd, 0.0);
            for (auto& cs : cat_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_pos_sum) cs.assign(n_embd, 0.0);
            for (auto& cs : con_neg_sum) cs.assign(n_embd, 0.0);
        }
        for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
            const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
            for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
            ++N;
            const char* c = token_category(model.decode({toks[fwd.act_rows[r]]}));
            if (!c) continue;
            for (size_t ci = 0; ci < cats.size(); ++ci)
                if (cats[ci] == c) {
                    for (int i = 0; i < n_embd; ++i) cat_sum[ci][i] += h[i];
                    ++cat_cnt[ci];
                    break;
                }
        }
    }

    // Pass 2: contrastive corpus — accumulate sentence-level pos/neg means (all tokens contribute).
    for (size_t ci = 0; ci < contrastive.size(); ++ci) {
        for (const std::string& text : contrastive[ci].pos) {
            const std::vector<int> toks = model.encode(text);
            if (toks.size() < 2) continue;
            const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
            const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
            if (fwd.activations.empty()) continue;
            for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
                const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
                for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
                ++N;
                for (int i = 0; i < n_embd; ++i) con_pos_sum[ci][i] += h[i];
                ++con_pos_cnt[ci];
            }
        }
        for (const std::string& text : contrastive[ci].neg) {
            const std::vector<int> toks = model.encode(text);
            if (toks.size() < 2) continue;
            const Mask m = attention_mask(static_cast<int>(toks.size()), 0, 0);
            const ForwardResult fwd = ad.forward(toks, m, nullptr, std::nullopt, {});
            if (fwd.activations.empty()) continue;
            for (size_t r = 0; r < fwd.act_rows.size(); ++r) {
                const float* h = fwd.activations.data() + r * static_cast<size_t>(n_embd);
                for (int i = 0; i < n_embd; ++i) { sum[i] += h[i]; sumsq[i] += static_cast<double>(h[i]) * h[i]; }
                ++N;
                for (int i = 0; i < n_embd; ++i) con_neg_sum[ci][i] += h[i];
                ++con_neg_cnt[ci];
            }
        }
    }

    ConceptProbes p;
    if (n_embd == 0 || N == 0) return p;
    p.n_embd = n_embd;
    p.mean.resize(n_embd);
    p.inv_std.resize(n_embd);
    for (int i = 0; i < n_embd; ++i) {
        const double mu = sum[i] / static_cast<double>(N);
        const double var = sumsq[i] / static_cast<double>(N) - mu * mu;
        p.mean[i] = static_cast<float>(mu);
        p.inv_std[i] = static_cast<float>(1.0 / std::sqrt(var > 1e-12 ? var : 1e-12));
    }

    // Per-token probes: diff-in-means (each category vs pooled rest).
    for (size_t ci = 0; ci < cats.size(); ++ci) {
        if (cat_cnt[ci] < 3) continue;
        long long neg_cnt = 0;
        std::vector<double> neg_sum(n_embd, 0.0);
        for (size_t cj = 0; cj < cats.size(); ++cj)
            if (cj != ci && cat_cnt[cj] > 0) {
                neg_cnt += cat_cnt[cj];
                for (int i = 0; i < n_embd; ++i) neg_sum[i] += cat_sum[cj][i];
            }
        if (neg_cnt == 0) continue;
        std::vector<float> dir(n_embd);
        double norm = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const double pos_std = (cat_sum[ci][i] / static_cast<double>(cat_cnt[ci]) - p.mean[i]) * p.inv_std[i];
            const double neg_std = (neg_sum[i] / static_cast<double>(neg_cnt) - p.mean[i]) * p.inv_std[i];
            const double d = pos_std - neg_std;
            dir[i] = static_cast<float>(d);
            norm += d * d;
        }
        norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
        for (int i = 0; i < n_embd; ++i) dir[i] = static_cast<float>(dir[i] / norm);
        p.names.push_back(cats[ci]);
        p.dirs.insert(p.dirs.end(), dir.begin(), dir.end());
    }

    // Contrastive probes: sentence-level diff-in-means (pos vs neg class).
    for (size_t ci = 0; ci < contrastive.size(); ++ci) {
        if (con_pos_cnt[ci] < 3 || con_neg_cnt[ci] < 3) continue;
        std::vector<float> dir(n_embd);
        double norm = 0.0;
        for (int i = 0; i < n_embd; ++i) {
            const double pos_std = (con_pos_sum[ci][i] / static_cast<double>(con_pos_cnt[ci]) - p.mean[i]) * p.inv_std[i];
            const double neg_std = (con_neg_sum[ci][i] / static_cast<double>(con_neg_cnt[ci]) - p.mean[i]) * p.inv_std[i];
            const double d = pos_std - neg_std;
            dir[i] = static_cast<float>(d);
            norm += d * d;
        }
        norm = std::sqrt(norm > 1e-12 ? norm : 1e-12);
        for (int i = 0; i < n_embd; ++i) dir[i] = static_cast<float>(dir[i] / norm);
        p.names.push_back(contrastive[ci].name);
        p.dirs.insert(p.dirs.end(), dir.begin(), dir.end());
    }

    return p;
}

// Build a llama control-vector buffer (n_embd*n_layer, layer-1-indexed) that ADDS coef * a concept's
// raw steering direction at every layer in [lo, hi] (other layers zero). Empty if the concept isn't
// found. NOTE: llama only has cvec tensors for layers 1..n_layer-1 (llama-adapter.cpp — no tensor for
// the last layer), so callers must keep hi <= n_layer-1. Pairs with GgmlAdapter::set_steer.
inline std::vector<float> build_steer_cvec(const ConceptProbes& p, const std::string& concept,
                                    double coef, int lo, int hi, int n_layer) {
    int idx = -1;
    for (int i = 0; i < p.size(); ++i) if (p.names[i] == concept) { idx = i; break; }
    if (idx < 0 || p.n_embd <= 0 || n_layer <= 0) return {};
    const std::vector<float> v = p.steer_vector(idx);
    std::vector<float> data(static_cast<size_t>(p.n_embd) * n_layer, 0.0f);
    for (int L = lo; L <= hi; ++L) {
        if (L < 1 || L >= n_layer) continue;  // valid cvec layers are 1..n_layer-1
        float* slice = data.data() + static_cast<size_t>(L - 1) * p.n_embd;
        for (int i = 0; i < p.n_embd; ++i) slice[i] = static_cast<float>(coef * v[i]);
    }
    return data;
}

// ---- J-lens readout (JLENS_ENGINE_PLAN.md J2) --------------------------------------------------------
// lens_l(h) = output.weight @ rmsnorm(J_l @ h): a fitted Jacobian J_l [d_model x d_model] f16 transports a
// layer-l residual into the FINAL-layer basis, then the model's OWN final RMSNorm + output head unembed it
// -> per-position top-k "disposed to say" tokens (Anthropic's J-lens, transferred to the GGUF engine in
// J0/J1). The J_l matrices are fitted sidecars (config dir, out of git, treated like model weights); the
// final_norm + head are read straight from the loaded GGUF (so the head is the model's own quantized q6_K
// lm_head -> expect drift vs the fp32 numpy oracle, tolerance not bitwise). The apply is a small standalone
// CPU ggml graph (the 7B stays on the GPU); weights are loaded once at startup, mirroring the SAE sidecar.
struct JlensServe {
    bool on = false;
    std::string err;                            // last load failure, human-readable
    int d_model = 0, vocab = 0;
    int default_layer = 0;                      // manifest engine_default_tap_layer (the /jlens default)
    float eps = 1e-6f;                          // final RMSNorm epsilon (read from the GGUF; Qwen2.5 = 1e-6)
    struct ggml_context* wctx = nullptr;        // persistent: output_norm, output.weight, and each J_l
    struct ggml_tensor* out_norm = nullptr;     // F32 [d_model]
    struct ggml_tensor* out_head = nullptr;     // the GGUF's own lm_head, quantized [d_model, vocab]
    std::map<int, struct ggml_tensor*> Jl;      // layer -> F16 [d_model, d_model] (ggml layout Jl(k,m)=J[m,k])
    std::mutex mtx;                             // a readout builds a transient graph; serialize (on-demand)

    // Device-resident compute path (Phase 2.3 readout plane). The CPU readout is a ~2 GFLOP
    // full-vocab GEMV per layer per token; run at every=1 it saturates every core and STARVES the
    // decode thread's CPU side (measured: 48% tok/s loss with the math on a worker thread — the
    // contention is the cores, not the thread). On the GPU the same GEMV is sub-ms and rides
    // beside the decode kernels. Weights are uploaded ONCE here; each readout is a tiny graph on
    // dev_backend + one [vocab x n] logits D2H + a CPU top-k. Null backend => CPU fallback
    // (thread-capped so it can never starve the decode feed).
    ggml_backend_t dev_backend = nullptr;
    struct ggml_context* dev_wctx = nullptr;       // device weight tensor metadata (no_alloc)
    ggml_backend_buffer_t dev_wbuf = nullptr;      // the device weight buffer
    struct ggml_tensor* dev_norm = nullptr;
    struct ggml_tensor* dev_head = nullptr;
    std::map<int, struct ggml_tensor*> dev_Jl;
    ggml_gallocr_t dev_galloc = nullptr;           // compute-buffer allocator (sized by chunking)

    JlensServe() = default;
    JlensServe(const JlensServe&) = delete;
    JlensServe& operator=(const JlensServe&) = delete;
    ~JlensServe() {
        if (dev_galloc) ggml_gallocr_free(dev_galloc);
        if (dev_wbuf) ggml_backend_buffer_free(dev_wbuf);
        if (dev_wctx) ggml_free(dev_wctx);
        if (dev_backend) ggml_backend_free(dev_backend);
        if (wctx) ggml_free(wctx);
    }

    std::vector<int> layers() const {
        std::vector<int> v;
        for (const auto& kv : Jl) v.push_back(kv.first);
        return v;
    }
    bool has(int layer) const { return Jl.count(layer) > 0; }

    // Read `sz` bytes at absolute file offset `off` into `dst` (64-bit seek: the head lives past 2 GB).
    static bool read_at(std::ifstream& f, void* dst, size_t off, size_t sz) {
        f.clear();
        f.seekg(static_cast<std::streamoff>(off), std::ios::beg);
        f.read(static_cast<char*>(dst), static_cast<std::streamsize>(sz));
        return static_cast<size_t>(f.gcount()) == sz;
    }

    // Load the J_l sidecars from `dir` (+ manifest.json) and the GGUF's OWN final_norm/head from
    // `model_path`. n_embd/n_vocab come from the loaded model. Returns false + sets err on any failure.
    bool load(const std::string& dir, const std::string& model_path, int n_embd, int n_vocab) {
        d_model = n_embd;
        vocab = n_vocab;
        std::vector<int> want_layers;
        {   // manifest: which layers + the default tap + a d_model sanity check.
            std::ifstream mf(dir + "/manifest.json", std::ios::binary);
            if (!mf) { err = "no manifest.json in " + dir; return false; }
            json m;
            try { mf >> m; } catch (...) { err = "manifest.json parse failed"; return false; }
            if (m.contains("d_model") && m["d_model"].is_number() && m["d_model"].get<int>() != d_model) {
                err = "manifest d_model " + std::to_string(m["d_model"].get<int>()) +
                      " != model n_embd " + std::to_string(d_model);
                return false;
            }
            if (m.contains("layers") && m["layers"].is_array())
                for (const auto& l : m["layers"]) if (l.is_number_integer()) want_layers.push_back(l.get<int>());
            default_layer = m.value("engine_default_tap_layer", want_layers.empty() ? 0 : want_layers.front());
        }
        if (want_layers.empty()) { err = "manifest lists no layers"; return false; }

        // Read output_norm.weight + output.weight straight from the GGUF (no_alloc meta -> targeted file read;
        // never loads the whole 4.7 GB file). The head is the model's OWN quantized lm_head.
        struct ggml_context* meta = nullptr;
        struct gguf_init_params gp; gp.no_alloc = true; gp.ctx = &meta;
        struct gguf_context* gg = gguf_init_from_file(model_path.c_str(), gp);
        if (!gg || !meta) { err = "gguf_init_from_file failed (head read)"; if (gg) gguf_free(gg); return false; }
        {   // eps from the model's own metadata (matches the oracle's 1e-6 for Qwen2.5); default keeps 1e-6.
            int64_t k = gguf_find_key(gg, "qwen2.attention.layer_norm_rms_epsilon");
            if (k >= 0) eps = gguf_get_val_f32(gg, k);
        }
        // Untied head is "output.weight"; a tied-embedding model has none -> fall back to token_embd.weight.
        std::string head_name = ggml_get_tensor(meta, "output.weight") ? "output.weight" : "token_embd.weight";
        struct ggml_tensor* mo = ggml_get_tensor(meta, head_name.c_str());
        struct ggml_tensor* mn = ggml_get_tensor(meta, "output_norm.weight");
        if (!mo || !mn) { err = "GGUF missing output(.weight)/output_norm.weight"; ggml_free(meta); gguf_free(gg); return false; }
        const int64_t oid = gguf_find_tensor(gg, head_name.c_str());
        const int64_t nid = gguf_find_tensor(gg, "output_norm.weight");
        const size_t data_off = gguf_get_data_offset(gg);
        const size_t o_off = data_off + gguf_get_tensor_offset(gg, oid), o_sz = gguf_get_tensor_size(gg, oid);
        const size_t n_off = data_off + gguf_get_tensor_offset(gg, nid), n_sz = gguf_get_tensor_size(gg, nid);
        const enum ggml_type o_type = mo->type;  const int64_t o_ne0 = mo->ne[0], o_ne1 = mo->ne[1];
        const enum ggml_type n_type = mn->type;  const int64_t n_ne0 = mn->ne[0];
        ggml_free(meta); gguf_free(gg);
        if (o_ne0 != d_model || o_ne1 != vocab) { err = "head shape mismatch vs model"; return false; }

        // Persistent weight context sized to hold the head + norm + every J_l.
        const size_t j_bytes = static_cast<size_t>(d_model) * d_model * sizeof(uint16_t);   // f16
        const size_t mem = o_sz + n_sz + j_bytes * want_layers.size()
                         + (want_layers.size() + 4) * ggml_tensor_overhead() + (1ULL << 20);
        struct ggml_init_params ip; ip.mem_size = mem; ip.mem_buffer = nullptr; ip.no_alloc = false;
        wctx = ggml_init(ip);
        if (!wctx) { err = "ggml_init(jlens weights) failed"; return false; }

        std::ifstream mfile(model_path, std::ios::binary);
        if (!mfile) { err = "cannot reopen model file for head read"; return false; }
        out_head = ggml_new_tensor_2d(wctx, o_type, o_ne0, o_ne1);
        out_norm = ggml_new_tensor_1d(wctx, n_type, n_ne0);
        if (!read_at(mfile, out_head->data, o_off, o_sz) || !read_at(mfile, out_norm->data, n_off, n_sz)) {
            err = "short read on the GGUF head/norm"; return false;
        }
        for (int L : want_layers) {
            const std::string path = dir + "/J_layer" + std::to_string(L) + ".f16";
            std::ifstream jf(path, std::ios::binary);
            if (!jf) { err = "missing sidecar " + path; return false; }
            struct ggml_tensor* J = ggml_new_tensor_2d(wctx, GGML_TYPE_F16, d_model, d_model);
            jf.read(static_cast<char*>(J->data), static_cast<std::streamsize>(j_bytes));
            if (static_cast<size_t>(jf.gcount()) != j_bytes) { err = "short read on " + path; return false; }
            Jl[L] = J;
        }
        init_device();  // best-effort: a failure leaves dev_backend null => CPU fallback, still correct
        on = true;
        return true;
    }

    // Upload the lens weights to the first GPU backend device (once). Every failure path cleans up
    // and leaves dev_backend null — the CPU readout stays fully functional, just slower.
    void init_device() {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (!dev) return;
        dev_backend = ggml_backend_dev_init(dev, nullptr);
        if (!dev_backend) return;
        struct ggml_init_params ip;
        ip.mem_size = (Jl.size() + 4) * ggml_tensor_overhead();
        ip.mem_buffer = nullptr;
        ip.no_alloc = true;  // metadata only; the data lives in the device buffer below
        dev_wctx = ggml_init(ip);
        if (!dev_wctx) { ggml_backend_free(dev_backend); dev_backend = nullptr; return; }
        dev_head = ggml_new_tensor_2d(dev_wctx, out_head->type, out_head->ne[0], out_head->ne[1]);
        dev_norm = ggml_new_tensor_1d(dev_wctx, out_norm->type, out_norm->ne[0]);
        for (const auto& kv : Jl)
            dev_Jl[kv.first] = ggml_new_tensor_2d(dev_wctx, GGML_TYPE_F16, d_model, d_model);
        dev_wbuf = ggml_backend_alloc_ctx_tensors(dev_wctx, dev_backend);
        if (!dev_wbuf) {  // e.g. VRAM exhausted: fall back to CPU
            dev_Jl.clear(); ggml_free(dev_wctx); dev_wctx = nullptr;
            ggml_backend_free(dev_backend); dev_backend = nullptr;
            return;
        }
        ggml_backend_tensor_set(dev_head, out_head->data, 0, ggml_nbytes(out_head));
        ggml_backend_tensor_set(dev_norm, out_norm->data, 0, ggml_nbytes(out_norm));
        for (const auto& kv : Jl)
            ggml_backend_tensor_set(dev_Jl[kv.first], kv.second->data, 0, ggml_nbytes(kv.second));
        dev_galloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(dev_backend));
        if (!dev_galloc) {
            dev_Jl.clear();
            ggml_backend_buffer_free(dev_wbuf); dev_wbuf = nullptr;
            ggml_free(dev_wctx); dev_wctx = nullptr;
            ggml_backend_free(dev_backend); dev_backend = nullptr;
        }
    }

    // Per-row top-k over a [vocab x n] logits block (column p = logits + p*vocab).
    void topk_rows(const float* Lg, int n_rows, int topk,
                   std::vector<std::vector<std::pair<int, float>>>& out, size_t out_base) const {
        const int k = std::min(topk, vocab);
        std::vector<int> idx(static_cast<size_t>(vocab));
        for (int p = 0; p < n_rows; ++p) {
            const float* row = Lg + static_cast<size_t>(p) * vocab;
            for (int v = 0; v < vocab; ++v) idx[static_cast<size_t>(v)] = v;
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return row[a] > row[b]; });
            auto& dst = out[out_base + static_cast<size_t>(p)];
            dst.reserve(static_cast<size_t>(k));
            for (int j = 0; j < k; ++j) dst.emplace_back(idx[static_cast<size_t>(j)], row[idx[static_cast<size_t>(j)]]);
        }
    }

    // Per-position top-k of lens_l(h). h = [n_tokens * d_model] host f32 (harvest layout, position-major).
    // Fills out[pos] = up to `topk` (token_id, logit) descending. Serialized (transient graph per call).
    // Takes the device path when init_device() succeeded (weights already resident; per-call cost is a
    // [n x d_model] H2D, a sub-ms GEMV graph, and a [vocab x n] logits D2H) — chunked so the device
    // compute buffer stays ~20 MB even for whole-prompt readouts. CPU fallback is thread-capped: it
    // must never saturate the cores the decode thread needs (that starvation, not queue blocking, was
    // the measured 48% readout overhead).
    bool readout(const float* h, int n_tokens, int layer, int topk,
                 std::vector<std::vector<std::pair<int, float>>>& out, std::string& e) {
        if (!on) { e = "jlens not loaded"; return false; }
        auto it = Jl.find(layer);
        if (it == Jl.end()) { e = "no J-lens sidecar for layer " + std::to_string(layer); return false; }
        if (n_tokens <= 0) { e = "no tokens"; return false; }
        std::lock_guard<std::mutex> lk(mtx);

        if (dev_backend) {
            out.assign(static_cast<size_t>(n_tokens), {});
            const int chunk = 16;  // caps the gallocr compute buffer at ~chunk*vocab*4 (~16 MB)
            std::vector<float> lg;
            for (int base = 0; base < n_tokens; base += chunk) {
                const int n = std::min(chunk, n_tokens - base);
                struct ggml_init_params gp;
                gp.mem_size = 16 * ggml_tensor_overhead() + ggml_graph_overhead();
                gp.mem_buffer = nullptr;
                gp.no_alloc = true;  // gallocr places every tensor in the device compute buffer
                struct ggml_context* gctx = ggml_init(gp);
                if (!gctx) { e = "ggml_init(jlens device graph) failed"; return false; }
                struct ggml_tensor* h_t = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, n);
                struct ggml_tensor* hJ = ggml_mul_mat(gctx, dev_Jl[layer], h_t);
                struct ggml_tensor* nm = ggml_rms_norm(gctx, hJ, eps);
                nm = ggml_mul(gctx, nm, dev_norm);
                struct ggml_tensor* logits = ggml_mul_mat(gctx, dev_head, nm);
                struct ggml_cgraph* gf = ggml_new_graph(gctx);
                ggml_build_forward_expand(gf, logits);
                bool ok = ggml_gallocr_alloc_graph(dev_galloc, gf);
                if (ok) {
                    ggml_backend_tensor_set(h_t, h + static_cast<size_t>(base) * d_model, 0,
                                            static_cast<size_t>(n) * d_model * sizeof(float));
                    ok = ggml_backend_graph_compute(dev_backend, gf) == GGML_STATUS_SUCCESS;
                }
                if (ok) {
                    lg.resize(static_cast<size_t>(vocab) * n);
                    ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
                }
                ggml_free(gctx);
                if (!ok) { e = "jlens device graph failed"; return false; }
                topk_rows(lg.data(), n, topk, out, static_cast<size_t>(base));
            }
            return true;
        }

        // CPU fallback. Transient graph context: leaves (h) + intermediates (hJ, normed, logits) + work buffer.
        struct ggml_tensor* J = it->second;
        const size_t tens = static_cast<size_t>((3 * d_model) + vocab) * n_tokens * sizeof(float);
        const size_t mem = tens + tens / 2 + ggml_graph_overhead() + 32 * ggml_tensor_overhead() + (128ULL << 20);
        struct ggml_init_params ip; ip.mem_size = mem; ip.mem_buffer = nullptr; ip.no_alloc = false;
        struct ggml_context* ctx = ggml_init(ip);
        if (!ctx) { e = "ggml_init(jlens graph) failed"; return false; }

        struct ggml_tensor* h_t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, d_model, n_tokens);   // h_t(k,n)=h[n][k]
        std::memcpy(h_t->data, h, static_cast<size_t>(d_model) * n_tokens * sizeof(float));
        struct ggml_tensor* hJ = ggml_mul_mat(ctx, J, h_t);           // [d_model, n]: x[n][m]=sum_k J[m,k]*h[n][k]
        struct ggml_tensor* nm = ggml_rms_norm(ctx, hJ, eps);         // final RMSNorm over the feature dim
        nm = ggml_mul(ctx, nm, out_norm);                             // * output_norm.weight (broadcast over n)
        struct ggml_tensor* logits = ggml_mul_mat(ctx, out_head, nm); // [vocab, n]: the model's own head

        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, logits);
        // Half the cores, max 4: a live-readout fallback must leave the decode thread's CPU side
        // (graph feed + sampling) breathing room, or the observer throttles generation itself.
        int nthreads = static_cast<int>(std::thread::hardware_concurrency()) / 2;
        if (nthreads < 1) nthreads = 1;
        if (nthreads > 4) nthreads = 4;
        const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, nthreads);
        if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); e = "jlens graph compute failed"; return false; }

        out.assign(static_cast<size_t>(n_tokens), {});
        topk_rows(static_cast<const float*>(logits->data), n_tokens, topk, out, 0);
        ggml_free(ctx);
        return true;
    }

    // Multi-layer readout: transport EVERY requested layer, then hit the head ONCE for all of them
    // (concat the transported+normed columns into one [d_model, n*L] block, one head mul_mat). The
    // head is the traffic (~vocab*d_model quantized bytes per read, ~25x a J_l); per-layer readout()
    // reads it L times per token, this reads it once — measured as the difference between halving
    // decode throughput and a modest overhead at 2 layers. h_by_layer: (layer, [n_tokens*d_model]
    // host f32); layers without a sidecar are skipped (absent from `out`). Device path only — on the
    // CPU fallback it just loops readout() (the CPU cost is compute-bound, not head-read-bound).
    bool readout_multi(const std::vector<std::pair<int, const float*>>& h_by_layer, int n_tokens,
                       int topk,
                       std::map<int, std::vector<std::vector<std::pair<int, float>>>>& out,
                       std::string& e) {
        if (!on) { e = "jlens not loaded"; return false; }
        if (n_tokens <= 0) { e = "no tokens"; return false; }
        std::vector<std::pair<int, const float*>> want;
        for (const auto& lv : h_by_layer) if (has(lv.first)) want.push_back(lv);
        if (want.empty()) return true;  // nothing observable: not an error, just no output

        if (!dev_backend) {  // CPU fallback: loop the single-layer path (already thread-capped)
            for (const auto& lv : want)
                if (!readout(lv.second, n_tokens, lv.first, topk, out[lv.first], e)) return false;
            return true;
        }

        std::lock_guard<std::mutex> lk(mtx);
        const int L = static_cast<int>(want.size());
        for (const auto& lv : want) out[lv.first].assign(static_cast<size_t>(n_tokens), {});
        const int chunk = 16;  // caps the compute buffer at ~chunk*L*vocab*4
        std::vector<float> lg;
        for (int base = 0; base < n_tokens; base += chunk) {
            const int n = std::min(chunk, n_tokens - base);
            struct ggml_init_params gp;
            gp.mem_size = static_cast<size_t>(8 * L + 8) * ggml_tensor_overhead() + ggml_graph_overhead();
            gp.mem_buffer = nullptr;
            gp.no_alloc = true;
            struct ggml_context* gctx = ggml_init(gp);
            if (!gctx) { e = "ggml_init(jlens multi graph) failed"; return false; }
            std::vector<struct ggml_tensor*> inputs(want.size());
            struct ggml_tensor* stacked = nullptr;  // [d_model, n*L], layer-major columns
            for (int l = 0; l < L; ++l) {
                inputs[l] = ggml_new_tensor_2d(gctx, GGML_TYPE_F32, d_model, n);
                struct ggml_tensor* hJ = ggml_mul_mat(gctx, dev_Jl[want[l].first], inputs[l]);
                struct ggml_tensor* nm = ggml_rms_norm(gctx, hJ, eps);
                nm = ggml_mul(gctx, nm, dev_norm);
                stacked = stacked ? ggml_concat(gctx, stacked, nm, 1) : nm;
            }
            struct ggml_tensor* logits = ggml_mul_mat(gctx, dev_head, stacked);  // ONE head read for all layers
            struct ggml_cgraph* gf = ggml_new_graph(gctx);
            ggml_build_forward_expand(gf, logits);
            bool ok = ggml_gallocr_alloc_graph(dev_galloc, gf);
            if (ok) {
                for (int l = 0; l < L; ++l)
                    ggml_backend_tensor_set(inputs[l],
                                            want[l].second + static_cast<size_t>(base) * d_model, 0,
                                            static_cast<size_t>(n) * d_model * sizeof(float));
                ok = ggml_backend_graph_compute(dev_backend, gf) == GGML_STATUS_SUCCESS;
            }
            if (ok) {
                lg.resize(static_cast<size_t>(vocab) * n * L);
                ggml_backend_tensor_get(logits, lg.data(), 0, lg.size() * sizeof(float));
            }
            ggml_free(gctx);
            if (!ok) { e = "jlens multi device graph failed"; return false; }
            for (int l = 0; l < L; ++l)  // columns [l*n, l*n + n) = layer l's tokens for this chunk
                topk_rows(lg.data() + static_cast<size_t>(l) * n * vocab, n, topk,
                          out[want[l].first], static_cast<size_t>(base));
        }
        return true;
    }

    // Hand back ONE row of the model's own (quantized) unembed/lm_head matrix -- W_U[token_id],
    // a [d_model] fp32 vector -- via ggml_get_rows. This is the missing ingredient the product's
    // dir(c) concept-dial (clozn/behavior/steering/concept_dir.py) needs: dir(c) =
    // normalize(J_l^T @ W_U[c]) already has J_l (the shipped product sidecar, read straight with
    // plain numpy) but had NO in-product source for W_U at all -- it lives only inside this
    // process, read straight out of the loaded GGUF for /jlens (out_head above). ggml_get_rows
    // dequantizes through the tensor's own `to_float` type trait (ggml-cpu/ops.cpp's
    // ggml_compute_forward_get_rows_q/_f16/...), so this works whatever quant type the GGUF's
    // head tensor is, with no engine-side dequant code of our own. out_head is laid out
    // [d_model, vocab] (ne0=d_model contiguous, ne1=vocab -- see load()'s "the model's own head"
    // comment and /jlens's `ggml_mul_mat(out_head, nm)`, which contracts over that same ne0), so
    // "row token_id" (ne1-index=token_id) is exactly W_U[token_id, :]. Hands back ONE row, never
    // the whole [vocab, d_model] matrix (~2 GB fp32) -- that never needs to leave this process.
    // Returns false + sets e on a not-loaded engine or an out-of-range token_id; never returns a
    // wrong-shaped vector.
    bool unembed_row(int token_id, std::vector<float>& out, std::string& e) {
        if (!on) { e = "jlens not loaded"; return false; }
        if (token_id < 0 || token_id >= vocab) {
            e = "token_id " + std::to_string(token_id) + " out of range for vocab " + std::to_string(vocab);
            return false;
        }
        std::lock_guard<std::mutex> lk(mtx);

        const size_t mem = static_cast<size_t>(d_model) * sizeof(float)
                         + ggml_graph_overhead() + 8 * ggml_tensor_overhead() + (1ULL << 20);
        struct ggml_init_params ip; ip.mem_size = mem; ip.mem_buffer = nullptr; ip.no_alloc = false;
        struct ggml_context* ctx = ggml_init(ip);
        if (!ctx) { e = "ggml_init(unembed_row graph) failed"; return false; }

        struct ggml_tensor* idx = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);
        *static_cast<int32_t*>(idx->data) = token_id;
        struct ggml_tensor* row = ggml_get_rows(ctx, out_head, idx);   // [d_model, 1] f32, dequantized

        struct ggml_cgraph* gf = ggml_new_graph(ctx);
        ggml_build_forward_expand(gf, row);
        int nthreads = static_cast<int>(std::thread::hardware_concurrency());
        if (nthreads < 1) nthreads = 4;
        if (nthreads > 16) nthreads = 16;
        const enum ggml_status st = ggml_graph_compute_with_ctx(ctx, gf, nthreads);
        if (st != GGML_STATUS_SUCCESS) { ggml_free(ctx); e = "unembed_row graph compute failed"; return false; }

        const float* data = static_cast<const float*>(row->data);
        out.assign(data, data + d_model);
        ggml_free(ctx);
        return true;
    }
};

// A pool of contexts over ONE shared model. Each request acquires a free context (blocking until
// one is available), runs on it, and releases it — so N workers serve N requests concurrently
// while the weights (the bulk of VRAM) are loaded once. RAII Lease guarantees release on any exit.
class ContextPool {
public:
    // flash_attn=false materializes the attention weights (kq_soft_max) so /score's attn_knockout
    // can cut individual query->key edges; costs some decode speed, hence opt-in via
    // --no-flash-attn rather than the default.
    ContextPool(std::shared_ptr<GgmlModel> model, int workers, int n_ctx, bool flash_attn = true,
                int n_batch = 0, int n_ubatch = 0)
        : model_(std::move(model)), flash_attn_(flash_attn) {
        for (int i = 0; i < workers; ++i) {
            adapters_.push_back(std::make_unique<GgmlAdapter>(model_, n_ctx, flash_attn,
                                                               n_batch, n_ubatch));
            free_.push(adapters_.back().get());
        }
    }
    class Lease {
    public:
        Lease(ContextPool& p, GgmlAdapter* a) : pool_(p), adapter_(a) {}
        ~Lease() { if (adapter_ != nullptr) pool_.release(adapter_); }
        Lease(const Lease&) = delete;
        Lease& operator=(const Lease&) = delete;
        Lease(Lease&& other) noexcept : pool_(other.pool_), adapter_(other.adapter_) {
            other.adapter_ = nullptr;
        }
        Lease& operator=(Lease&&) = delete;
        GgmlAdapter& operator*() const { return *adapter_; }
    private:
        ContextPool& pool_;
        GgmlAdapter* adapter_;
    };

    // Experimental persistent reference-match sessions own a dedicated context rather than holding
    // one ordinary pooled lease forever.  This leaves the regular pool available for the trusted
    // scalar confirmation path; the session registrar still enforces one active persistent context.
    class PersistentLease {
    public:
        PersistentLease(ContextPool& p, std::unique_ptr<GgmlAdapter> a)
            : pool_(p), adapter_(std::move(a)) {}
        ~PersistentLease() {
            if (adapter_ == nullptr) return;
            // Destroy the dedicated context before making the slot available again.  Otherwise a
            // new session could briefly construct a second extra context while this one is still
            // tearing down its KV/state resources.
            adapter_.reset();
            pool_.release_persistent();
        }
        PersistentLease(const PersistentLease&) = delete;
        PersistentLease& operator=(const PersistentLease&) = delete;
        PersistentLease(PersistentLease&& other) noexcept
            : pool_(other.pool_), adapter_(std::move(other.adapter_)) {}
        PersistentLease& operator=(PersistentLease&&) = delete;
        GgmlAdapter& operator*() const { return *adapter_; }
    private:
        ContextPool& pool_;
        std::unique_ptr<GgmlAdapter> adapter_;
    };

    Lease acquire() {
        std::unique_lock<std::mutex> lk(mtx_);
        cv_.wait(lk, [&] { return !free_.empty(); });
        GgmlAdapter* a = free_.front();
        free_.pop();
        return Lease(*this, a);
    }
    PersistentLease acquire_persistent() {
        std::lock_guard<std::mutex> lk(persistent_mtx_);
        if (persistent_active_)
            throw std::runtime_error("persistent reference-match context is busy");
        persistent_active_ = true;
        try {
            auto adapter = std::make_unique<GgmlAdapter>(
                model_, n_ctx(), flash_attn_, n_batch(), n_ubatch());
            if (lora_loaded()) {
                std::string lora_err;
                if (!adapter->set_lora(lora_path(), lora_scale(), &lora_err)) {
                    throw std::runtime_error("persistent reference-match context LoRA attach failed: "
                                             + lora_err);
                }
            }
            return PersistentLease(*this, std::move(adapter));
        } catch (...) {
            persistent_active_ = false;
            throw;
        }
    }
    int size() const { return static_cast<int>(adapters_.size()); }
    int n_ctx() const { return adapters_.empty() ? 0 : adapters_.front()->n_ctx(); }
    int n_batch() const { return adapters_.empty() ? 0 : adapters_.front()->n_batch(); }
    int n_ubatch() const { return adapters_.empty() ? 0 : adapters_.front()->n_ubatch(); }
    std::map<std::string, std::size_t> memory_breakdown() {
        if (adapters_.empty()) return {};
        auto lease = acquire();
        return (*lease).memory_breakdown();
    }

    // Attach a fine-tune adapter to EVERY pooled context, or to none of them. All-or-nothing is the
    // whole point: a pool where only some contexts carry the LoRA would serve adapted or unadapted
    // output depending on which lease a request happened to acquire -- nondeterminism no run identity
    // could explain, and the worst possible failure mode for a tool whose product is receipts. On any
    // failure the already-attached contexts are rolled back before returning false.
    bool attach_lora(const std::string& path, float scale, std::string* err) {
        for (size_t i = 0; i < adapters_.size(); ++i) {
            if (!adapters_[i]->set_lora(path, scale, err)) {
                for (size_t j = 0; j < i; ++j) adapters_[j]->clear_lora();
                return false;
            }
        }
        return true;
    }
    // Pool-wide adapter state. attach_lora's all-or-nothing contract is what makes reading these off
    // the first context correct for the whole pool.
    bool lora_loaded() const { return !adapters_.empty() && adapters_.front()->lora_loaded(); }
    std::string lora_path() const {
        return adapters_.empty() ? std::string() : adapters_.front()->lora_path();
    }
    float lora_scale() const { return adapters_.empty() ? 0.0f : adapters_.front()->lora_scale(); }
    std::map<std::string, std::string> lora_meta() const {
        return adapters_.empty() ? std::map<std::string, std::string>() : adapters_.front()->lora_meta();
    }

private:
    void release(GgmlAdapter* a) {
        { std::lock_guard<std::mutex> lk(mtx_); free_.push(a); }
        cv_.notify_one();
    }
    void release_persistent() {
        std::lock_guard<std::mutex> lk(persistent_mtx_);
        persistent_active_ = false;
    }
    std::shared_ptr<GgmlModel> model_;
    std::vector<std::unique_ptr<GgmlAdapter>> adapters_;
    std::queue<GgmlAdapter*> free_;
    std::mutex mtx_;
    std::condition_variable cv_;
    bool persistent_active_ = false;
    std::mutex persistent_mtx_;
    bool flash_attn_ = true;
};


}  // namespace clozn
