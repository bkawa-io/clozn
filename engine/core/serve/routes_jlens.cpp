// serve/routes_jlens.cpp -- the /jlens route family (J-lens readout, JLENS_ENGINE_PLAN.md J2) (Phase 12.4 split of the serve monolith).
// Moved VERBATIM from the monolith; shared state is read through ServerContext -- the register
// fn re-binds local aliases so each handler body is byte-identical to the original.
#include "httplib.h"

#include "server_context.hpp"

namespace clozn {

void register_jlens_routes(httplib::Server& svr, ServerContext& ctx) {
    auto& model = ctx.model;
    auto& pool = ctx.pool;
    auto& concept_probes = ctx.concept_probes;
    auto& steer_probes = ctx.steer_probes;
    auto& sae_serve = ctx.sae_serve;
    auto& jlens = ctx.jlens;
    const int& n_ctx = ctx.n_ctx;
    const int& mask_token = ctx.mask_token;
    const bool& ar_mode = ctx.ar_mode;
    (void)model; (void)pool; (void)concept_probes; (void)steer_probes; (void)sae_serve;
    (void)jlens; (void)n_ctx; (void)mask_token; (void)ar_mode;

    // POST /jlens — the J-lens readout (JLENS_ENGINE_PLAN.md J2). Body {text, layer?, topk?=5}: tap the
    // residual `h` at `layer` (the /harvest machinery), transport it into the final-layer basis with the
    // fitted J_l, then unembed through the model's OWN final RMSNorm + output head -> each position's top-k
    // "disposed to say" tokens. NOT generation, NO sampling: a deterministic linear read. Response
    // {layer, n_tokens, tokens:[piece...], readouts:[[{id,piece,score}...k]...n]} (score = the raw lens
    // logit). Guards mirror /harvest: missing sidecar -> 400 + available layers; empty/over-n_ctx -> 400;
    // malformed JSON -> 400; the pooled context's tap/emit flags are restored on EVERY exit.
    svr.Post("/jlens", [&](const httplib::Request& req, httplib::Response& res) {
        if (!jlens.on) {
            res.status = 400;
            res.set_content(json{{"error", "J-lens not loaded (start with --jlens <dir> or set CLOZN_JLENS_DIR)"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string text = body.value("text", std::string());
        if (text.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty text"}}.dump(), "application/json");
            return;
        }
        int topk = body.value("topk", 5);
        if (topk < 1) topk = 1;
        const int layer = body.value("layer", jlens.default_layer);
        if (!jlens.has(layer)) {
            res.status = 400;
            res.set_content(json{{"error", "no J-lens sidecar for that layer"},
                                 {"layer", layer}, {"available", jlens.layers()}}.dump(), "application/json");
            return;
        }
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "text tokenized to zero tokens"}}.dump(), "application/json");
            return;
        }
        if (static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text too long for one forward"},
                                 {"n_tokens", static_cast<int>(tokens.size())}, {"n_ctx", n_ctx}}.dump(),
                            "application/json");
            return;
        }
        try {
            ForwardResult fwd;
            {   // tap `h` at `layer` via the harvest path; restore the pooled context's tap/emit on all exits.
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const int default_tap = ad.tap_layer();
                ad.set_causal(true);            // harvest under causal attention (also clears the KV)
                ad.set_emit_activations(true);
                ad.set_tap_layer(layer);
                try {
                    fwd = ad.harvest(tokens);
                    ad.set_emit_activations(false);
                    ad.set_tap_layer(default_tap);
                } catch (...) {
                    ad.set_emit_activations(false);
                    ad.set_tap_layer(default_tap);
                    throw;
                }
            }
            if (fwd.n_embd != jlens.d_model ||
                fwd.activations.size() != static_cast<size_t>(tokens.size()) * jlens.d_model) {
                res.status = 400;
                res.set_content(json{{"error", "harvest returned no/mismatched activations for this layer"}}.dump(),
                                "application/json");
                return;
            }
            std::vector<std::vector<std::pair<int, float>>> tk;
            std::string e;
            if (!jlens.readout(fwd.activations.data(), static_cast<int>(tokens.size()), layer, topk, tk, e)) {
                res.status = 400;
                res.set_content(json{{"error", e}}.dump(), "application/json");
                return;
            }
            json pieces = json::array();
            for (int id : tokens) pieces.push_back(model->decode({id}));
            json readouts = json::array();
            for (const auto& row : tk) {
                json rr = json::array();
                for (const auto& pr : row)
                    rr.push_back({{"id", pr.first}, {"piece", model->decode({pr.first})}, {"score", pr.second}});
                readouts.push_back(rr);
            }
            json resp = {{"layer", layer}, {"n_tokens", static_cast<int>(tokens.size())},
                         {"tokens", pieces}, {"readouts", readouts}};
            // dump_json (replace handler): token/readout pieces can be a PARTIAL multi-byte UTF-8 sequence
            // (byte-fallback tokens split a codepoint) -- strict dump() would throw on those; replace emits
            // U+FFFD instead, same as the streaming path. Determinism holds (identical inputs -> identical bytes).
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /jlens/unembed_row — hands back ONE row of the model's own (quantized) unembed/lm_head
    // matrix, W_U[token_id] ([d_model] fp32) — the ingredient the product's dir(c) concept-dial
    // (clozn/analysis/concept_dir.py: dir(c) = normalize(J_l^T @ W_U[c])) was missing:
    // J_l already ships in the product J-lens sidecar (read directly with plain numpy); W_U lived
    // ONLY inside this process (read straight out of the loaded GGUF for /jlens above — see
    // JlensServe::load's out_head) with no route back to it at all (concept_dir.py's
    // BLOCKER_NOTE). Body {token_id}. Response {token_id, piece, d_model, vector:[d_model
    // floats]}. Hands back exactly one row (via ggml_get_rows, which dequantizes whatever quant
    // type the GGUF head is), never the whole [vocab, d_model] matrix — no full-precision unembed
    // copy ever needs to leave this process. Guards mirror /jlens: J-lens not loaded -> 400;
    // missing/out-of-range token_id -> 400.
    svr.Post("/jlens/unembed_row", [&](const httplib::Request& req, httplib::Response& res) {
        if (!jlens.on) {
            res.status = 400;
            res.set_content(json{{"error", "J-lens not loaded (start with --jlens <dir> or set CLOZN_JLENS_DIR)"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        try {
            const int token_id = body.value("token_id", -1);
            std::vector<float> row;
            std::string e;
            if (!jlens.unembed_row(token_id, row, e)) {
                res.status = 400;
                res.set_content(json{{"error", e}}.dump(), "application/json");
                return;
            }
            json vec = json::array();
            for (float v : row) vec.push_back(v);
            json resp = {{"token_id", token_id}, {"piece", model->decode({token_id})},
                        {"d_model", jlens.d_model}, {"vector", vec}};
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });
}

}  // namespace clozn
