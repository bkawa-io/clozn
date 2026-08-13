// serve/routes_state.cpp -- the write/steer routes: /state (activation patch) + /intervene (steered generation) (Phase 12.4 split of the serve monolith).
// Moved VERBATIM from the monolith; shared state is read through ServerContext -- the register
// fn re-binds local aliases so each handler body is byte-identical to the original.
#include "httplib.h"

#include "server_context.hpp"

namespace clozn {

void register_state_routes(httplib::Server& svr, ServerContext& ctx) {
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

    // POST /state — GAP #1 (task #43): the WRITE-and-observe inverse of /harvest, over HTTP. Body
    // {text, layer, positions:[int], values:[float] = positions.size()*n_embd (the EDITED residual rows a
    // client read via /harvest and changed)}: run a baseline causal forward, OVERWRITE those positions'
    // residual at `layer` (GgmlAdapter::write_state — the eval-callback activation override), run
    // again, then clear — and report how the model's next-token prediction moved. /harvest (read) + this
    // (write) close the read->edit->write->observe loop on the LIVE model over HTTP. Leak-free: the write
    // is cleared before the pooled context is released.
    svr.Post("/state", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, false);
        if (body.is_discarded() || !body.contains("text") || !body.contains("layer") ||
            !body.contains("positions") || !body.contains("values")) {
            res.status = 400;
            res.set_content(json{{"error", "need {text, layer, positions:[int], values:[float]}"}}.dump(),
                            "application/json");
            return;
        }
        const std::string text = body["text"].get<std::string>();
        const int layer = body["layer"].get<int>();
        const std::vector<int> positions = body["positions"].get<std::vector<int>>();
        const std::vector<float> values = body["values"].get<std::vector<float>>();
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty() || static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text empty or too long for one forward"}}.dump(),
                            "application/json");
            return;
        }
        auto top3 = [&](const std::vector<float>& lg, int vocab) {
            json arr = json::array();
            if (lg.empty() || vocab <= 0) return arr;
            float mx = lg[0];
            for (int v = 1; v < vocab; ++v) mx = std::max(mx, lg[v]);
            double Z = 0.0;
            for (int v = 0; v < vocab; ++v) Z += std::exp(double(lg[v]) - double(mx));
            std::vector<int> idx(static_cast<size_t>(vocab));
            for (int v = 0; v < vocab; ++v) idx[static_cast<size_t>(v)] = v;
            const int k = std::min(3, vocab);
            std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                              [&](int a, int b) { return lg[a] > lg[b]; });
            for (int j = 0; j < k; ++j) {
                const int v = idx[static_cast<size_t>(j)];
                arr.push_back({{"token", model->decode({v})},
                               {"prob", std::exp(double(lg[v]) - double(mx)) / Z}});
            }
            return arr;
        };
        try {
            std::vector<float> base_lg, edit_lg;
            int vocab = 0;
            bool applied = false;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                try {
                    ad.set_causal(true);                       // causal forward (also clears the KV)
                    ForwardResult b = ad.ar_forward(tokens, 0);
                    base_lg = b.logits; vocab = b.vocab;
                    applied = ad.write_state(layer, positions, values);
                    if (applied) {
                        ad.set_causal(true);                   // fresh KV, with the write now armed
                        ForwardResult e = ad.ar_forward(tokens, 0);
                        edit_lg = e.logits;
                    }
                    ad.clear_write();                          // leak-free: never sticks on the pooled ctx
                } catch (...) {
                    ad.clear_write();
                    throw;
                }
            }
            double moved = 0.0;
            if (applied && base_lg.size() == edit_lg.size()) {
                for (size_t i = 0; i < base_lg.size(); ++i) {
                    const double d = double(base_lg[i]) - double(edit_lg[i]);
                    moved += d * d;
                }
                moved = std::sqrt(moved);
            }
            json resp = {
                {"applied", applied}, {"layer", layer},
                {"n_positions", static_cast<int>(positions.size())},
                {"n_values", static_cast<int>(values.size())},
                {"moved_l2", moved},
                {"baseline_top", top3(base_lg, vocab)},
                {"edited_top", applied ? top3(edit_lg, vocab) : json::array()},
            };
            if (!applied)
                resp["error"] = "write_state rejected (layer must be in [1, n_layer); "
                                "values.size must equal positions.size * n_embd)";
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /intervene — the state-stream protocol's WRITE channel (SPEC.md "Intervene"). Accepts an
    // Intervention {kind, target, vector?, coef} and applies it, then runs a generation so the effect
    // "shows up in the next steps" (the engine is stateless-per-request, so an intervention is realized
    // ON a concrete generation rather than parked on a context). kind:"steer" wraps the existing
    // control-vector path: either a NAMED concept (target.concept -> build_steer_cvec over the
    // calibrated steer_probes) or a RAW direction (vector:[n_embd] -> a control vector built directly).
    // target may carry the usual generation params (prompt, max_tokens, steps, ...) + stream/protocol/
    // state; the response is the steered result (board+text), or the StateStep SSE stream when stream:true.
    // (edit/restore are a board op: snapshot = the `board` in any response, restore = POST /v1/board.)
    svr.Post("/intervene", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        const std::string kind = body.value("kind", std::string("steer"));
        if (kind != "steer") {
            // edit/restore/patch are realized through /v1/board (snapshot+restore), not here.
            res.status = 400;
            res.set_content(json{{"error", "/intervene currently supports kind:'steer' only; "
                                  "use /v1/board for edit/restore (snapshot+restore)"},
                                 {"kind", kind}}.dump(), "application/json");
            return;
        }
        const json target = body.value("target", json::object());
        const double coef = body.value("coef", target.value("coef", 1.0));
        const std::string concept = target.value("concept", body.value("concept", std::string()));
        const int req_layer = target.value("layer", body.value("layer", 0));
        // Raw direction (optional): an explicit [n_embd] steering vector the inspector supplies.
        std::vector<float> raw_vec;
        if (body.contains("vector") && body["vector"].is_array())
            for (const auto& v : body["vector"]) raw_vec.push_back(v.get<float>());
        if (concept.empty() && raw_vec.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "steer needs target.concept or a raw 'vector'"}}.dump(),
                            "application/json");
            return;
        }
        if (!concept.empty() && !steer_probes.ready()) {
            res.status = 400;
            res.set_content(json{{"error", "no concept probes calibrated; pass a raw 'vector' instead"}}.dump(),
                            "application/json");
            return;
        }
        // Resolve a concept name against the calibrated set up-front (clear 404-style error).
        if (!concept.empty()) {
            bool found = false;
            for (const auto& nm : steer_probes.names) if (nm == concept) { found = true; break; }
            if (!found) {
                res.status = 400;
                res.set_content(json{{"error", "unknown concept"}, {"concept", concept},
                                     {"available", steer_probes.names}}.dump(), "application/json");
                return;
            }
        }

        // The generation the intervention is applied to. Reuses the completions request shape; the
        // generation params (prompt, max_tokens, steps, ...) may sit at top level OR nested under
        // `target` — merge them (target wins) so both spellings work.
        json gen = body;
        if (target.is_object())
            for (auto it = target.begin(); it != target.end(); ++it) gen[it.key()] = it.value();
        const GenerateConfig cfg = config_from(gen);
        const SampleConfig sample = sample_from(gen);
        const bool stream = body.value("stream", false);
        const bool protocol = body.value("protocol", false);
        const bool state_full = body.value("state", std::string("light")) == std::string("full");
        const bool features = body.value("features", false) || state_full;
        std::vector<int> prompt_ids = model->encode(target.value("prompt", body.value("prompt", std::string())));
        if (prompt_ids.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "intervene needs a target.prompt to apply the steer to"}}.dump(),
                            "application/json");
            return;
        }

        // The runner: acquire a context, BUILD + SET the control vector (the wrapped set_steer path),
        // generate, then clear. The layer window matches the per-request steer path (mid-depth, where
        // steer_probes is calibrated) unless target.layer pins one.
        // Written inside run() below, read by the streaming content-provider AFTER this handler returns
        // (set_chunked_content_provider only registers the callback; it fires later, from httplib's worker
        // thread, once this handler's stack frame is long gone). A plain handler-local `json` captured by
        // reference here would dangle the moment the handler returns -- shared_ptr keeps it alive for both
        // lambdas (same fix shape as raw_vec below, which only needs to be read, so a by-value copy suffices).
        auto applied_layers = std::make_shared<json>(json::array());
        auto run = [&pool, &steer_probes, &concept_probes, &sae_serve, raw_vec, concept, coef, req_layer, prompt_ids,
                    cfg, sample, features, ar_mode, applied_layers](
                       const std::function<void(const Event&)>& on_event) {
            ContextPool::Lease lease = pool.acquire();
            (*lease).set_emit_activations(features);
            const bool sae_on = features && sae_serve.on;
            const int default_tap = (*lease).tap_layer();
            if (sae_on) (*lease).set_tap_layer(sae_serve.layer);
            const std::function<void(const Event&)> ev = with_sae_readout(on_event, sae_serve, sae_on);
            const ConceptProbes* probes = (features && concept_probes.ready()) ? &concept_probes : nullptr;
            const int nl = (*lease).n_layer();
            int lo, hi;
            if (req_layer >= 1) { lo = hi = (req_layer < nl ? req_layer : nl - 1); }
            else { const int tl = nl * 2 / 3; lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
            if (lo < 1) lo = 1;
            // Build the n_embd*n_layer control-vector buffer: from the named concept (build_steer_cvec)
            // or straight from the raw direction (same layout, applied over [lo,hi]).
            std::vector<float> cvec;
            if (!concept.empty()) {
                cvec = build_steer_cvec(steer_probes, concept, coef, lo, hi, nl);
            } else {
                const int ne = steer_probes.n_embd > 0 ? steer_probes.n_embd : static_cast<int>(raw_vec.size());
                cvec.assign(static_cast<size_t>(ne) * nl, 0.0f);
                const int m = ne < static_cast<int>(raw_vec.size()) ? ne : static_cast<int>(raw_vec.size());
                for (int L = lo; L <= hi; ++L) {
                    if (L < 1 || L >= nl) continue;
                    float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                    for (int i = 0; i < m; ++i) slice[i] = static_cast<float>(coef * raw_vec[i]);
                }
            }
            *applied_layers = json::array({lo, hi});
            (*lease).set_steer(cvec, lo, hi);
            // Diffusion generation was removed from this engine build (THE_CUT); ar_mode is the only
            // mode with a generator left. Fail honestly rather than call a function that no longer exists.
            if (!ar_mode) {
                (*lease).clear_steer();
                throw std::invalid_argument(
                    "diffusion generation was removed from this engine build; only autoregressive "
                    "models are supported");
            }
            auto r = generate_ar(*lease, prompt_ids, cfg, ev, sample, probes);
            (*lease).clear_steer();
            if (sae_on) (*lease).set_tap_layer(default_tap);
            (*lease).set_emit_activations(false);
            return r;
        };
        const std::string id = make_id("interv-");
        const char* substrate = ar_mode ? "autoregressive" : "diffusion";

        if (stream) {
            res.set_chunked_content_provider(
                "text/event-stream",
                [run, id, model, protocol, state_full, substrate, mask_token, applied_layers, kind, concept, coef]
                (size_t, httplib::DataSink& sink) {
                    auto write = [&](const std::string& s) { sink.write(s.data(), s.size()); };
                    try {                                 // a generator throw here would otherwise escape into
                                                          // httplib's worker thread -> abort(); catch it below.
                    GenerateResult r;
                    if (protocol) {
                        StateStepBuilder builder(*model, substrate, state_full, write);
                        r = run([&](const Event& e) { builder.on_event(e); });
                        builder.finish();
                    } else {
                        r = run([&](const Event& e) {
                            write("data: " + to_jsonl_line(e) + "\n\n");
                        });
                    }
                    json final_frame = {{"kind", "final"}, {"id", id}, {"object", "intervene"},
                                        {"applied", true},
                                        {"intervention", {{"kind", kind}, {"concept", concept}, {"coef", coef},
                                                          {"layers", *applied_layers}}},
                                        {"text", r.text}, {"finish_reason", finish_reason(r.reason)},
                                        {"board", r.board},
                                        {"layout", board_layout_json(*model, r.board, mask_token)}};
                    write("data: " + dump_json(final_frame) + "\n\n");
                    write("data: [DONE]\n\n");
                    sink.done();
                    return true;
                    } catch (const std::exception& e) {
                        // The generator threw (n_ctx exceeded, decode failure, ...). Emit a clean error frame
                        // and close the stream gracefully, mirroring /v1/completions' streaming guard.
                        json err = {{"error", std::string("generation failed: ") + e.what()}};
                        write("data: " + err.dump() + "\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    } catch (...) {
                        write("data: {\"error\":\"generation failed\"}\n\n");
                        write("data: [DONE]\n\n");
                        sink.done();
                        return true;
                    }
                });
            return;
        }

        GenerateResult r = run({});
        json resp = {
            {"id", id}, {"object", "intervene"}, {"applied", true},
            {"intervention", {{"kind", kind}, {"concept", concept}, {"coef", coef}, {"layers", *applied_layers}}},
            {"choices", json::array({{{"text", r.text}, {"index", 0},
                         {"finish_reason", finish_reason(r.reason)}}})},
            {"board", r.board},
            {"layout", board_layout_json(*model, r.board, mask_token)},
            {"usage", {{"completion_tokens", r.new_tokens}, {"steps_total", r.steps_total}}},
        };
        res.set_content(dump_json(resp), "application/json");
    });
}

}  // namespace clozn
