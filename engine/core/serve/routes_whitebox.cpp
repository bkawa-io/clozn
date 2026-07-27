// serve/routes_whitebox.cpp -- white-box read and private template routes (Phase 12.4 split of the serve monolith).
// Moved VERBATIM from the monolith; shared state is read through ServerContext -- the register
// fn re-binds local aliases so each handler body is byte-identical to the original.
#include "httplib.h"

#include <map>

#include "server_context.hpp"

namespace clozn {

namespace {

json prepared_chat_json(const PreparedChat& prepared) {
    json triggers = json::array();
    for (const auto& trigger : prepared.grammar_triggers) {
        triggers.push_back({
            {"type", trigger.type},
            {"value", trigger.value},
            {"token", trigger.token},
        });
    }
    return {
        {"prompt", prepared.prompt},
        {"grammar", prepared.grammar},
        {"grammar_lazy", prepared.grammar_lazy},
        {"grammar_triggers", std::move(triggers)},
        {"preserved_tokens", prepared.preserved_tokens},
        {"additional_stops", prepared.additional_stops},
        {"generation_prompt", prepared.generation_prompt},
        {"parser", prepared.parser},
        {"format", prepared.format},
        {"capabilities", prepared.capabilities},
        {"supports_thinking", prepared.supports_thinking},
        {"thinking_start_tag", prepared.thinking_start_tag},
        {"thinking_end_tag", prepared.thinking_end_tag},
        {"reasoning_format", prepared.reasoning_format},
        {"parse_tool_calls", prepared.parse_tool_calls},
    };
}

PreparedChat prepared_chat_from_json(const json& value) {
    if (!value.is_object()) {
        throw std::invalid_argument("'prepared' must be an object returned by /prepare_chat");
    }

    PreparedChat prepared;
    prepared.prompt = value.at("prompt").get<std::string>();
    prepared.grammar = value.at("grammar").get<std::string>();
    prepared.grammar_lazy = value.at("grammar_lazy").get<bool>();
    prepared.preserved_tokens = value.at("preserved_tokens").get<std::vector<std::string>>();
    prepared.additional_stops = value.at("additional_stops").get<std::vector<std::string>>();
    prepared.generation_prompt = value.at("generation_prompt").get<std::string>();
    prepared.parser = value.at("parser").get<std::string>();
    prepared.format = value.at("format").get<std::string>();
    prepared.capabilities = value.at("capabilities").get<std::map<std::string, bool>>();
    prepared.supports_thinking = value.at("supports_thinking").get<bool>();
    prepared.thinking_start_tag = value.at("thinking_start_tag").get<std::string>();
    prepared.thinking_end_tag = value.at("thinking_end_tag").get<std::string>();
    prepared.reasoning_format = value.at("reasoning_format").get<std::string>();
    prepared.parse_tool_calls = value.at("parse_tool_calls").get<bool>();

    const auto& triggers = value.at("grammar_triggers");
    if (!triggers.is_array()) {
        throw std::invalid_argument("prepared.grammar_triggers must be an array");
    }
    prepared.grammar_triggers.reserve(triggers.size());
    for (const auto& trigger : triggers) {
        if (!trigger.is_object()) {
            throw std::invalid_argument("each prepared grammar trigger must be an object");
        }
        prepared.grammar_triggers.push_back(ChatGrammarTrigger{
            trigger.at("type").get<std::string>(),
            trigger.at("value").get<std::string>(),
            trigger.at("token").get<std::int32_t>(),
        });
    }
    return prepared;
}

json parsed_chat_json(const ParsedChat& parsed) {
    json tool_calls = json::array();
    for (const auto& call : parsed.tool_calls) {
        tool_calls.push_back({
            {"id", call.id},
            {"name", call.name},
            {"arguments", call.arguments},
        });
    }
    json message = json::parse(parsed.openai_json);
    return {
        {"role", parsed.role},
        {"content", parsed.content},
        {"reasoning_content", parsed.reasoning_content},
        {"tool_name", parsed.tool_name},
        {"tool_call_id", parsed.tool_call_id},
        {"tool_calls", std::move(tool_calls)},
        {"openai_json", parsed.openai_json},
        {"message", std::move(message)},
    };
}

}  // namespace

void register_whitebox_routes(httplib::Server& svr, ServerContext& ctx) {
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

    // POST /harvest — the §3.1 "activation harvesting at scale" READ endpoint. Body {text, layer?}:
    // tokenize `text`, run ONE causal forward over all its tokens with the white-box tap on, and
    // return every token's residual at the tap layer (NOT a generation — no sampling, no streaming).
    // Response: {tokens:[piece,...], layer:N, activations:{dtype:"float32", shape:[n_tokens,n_embd],
    // data: base64-LE}} — the matrix a discovery harness (SAE/PCA) trains on. `layer` (optional)
    // overrides the adapter's default tap (else the calibrated early tap, layer 2 for Qwen-0.5B);
    // an out-of-range value falls back to the final layer. One forward per text => efficient, and it
    // captures NATURAL-text activations (every input token), sidestepping the sustained-generation
    // crash the generation-based harvest hit. Works in either mode (it forces causal locally); the
    // pooled context is restored (tap default + emit off) before release.
    svr.Post("/harvest", [&](const httplib::Request& req, httplib::Response& res) {
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
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "text tokenized to zero tokens"}}.dump(), "application/json");
            return;
        }
        // One forward decodes all tokens in a single ubatch (n_ubatch == n_ctx), so a passage longer
        // than the context can't be harvested in one pass — reject it cleanly (400) BEFORE acquiring a
        // context, so an over-length passage is a skippable client error, never a 500. The harvester
        // chunks its corpus under this; the explicit guard keeps the contract honest at the edge.
        if (static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text too long for one forward"},
                                 {"n_tokens", static_cast<int>(tokens.size())},
                                 {"n_ctx", n_ctx}}.dump(), "application/json");
            return;
        }
        const bool has_layer = body.contains("layer") && body["layer"].is_number();
        const int req_layer = has_layer ? body["layer"].get<int>() : -1;

        try {
            ForwardResult fwd;
            int used_layer = 0;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const int default_tap = ad.tap_layer();
                ad.set_causal(true);            // harvest under causal attention (also clears the KV)
                ad.set_emit_activations(true);  // white-box tap on for this call
                if (has_layer) ad.set_tap_layer(req_layer);  // 0 / out-of-range => final-layer fallback
                used_layer = ad.tap_layer();
                try {
                    fwd = ad.harvest(tokens);
                    ad.set_emit_activations(false);   // restore the pooled context for the next request
                    ad.set_tap_layer(default_tap);
                } catch (...) {
                    ad.set_emit_activations(false);   // restore even on failure, then rethrow
                    ad.set_tap_layer(default_tap);
                    throw;
                }
            }

            json pieces = json::array();
            for (int id : tokens) pieces.push_back(model->decode({id}));
            json resp = {
                {"tokens", pieces},
                {"layer", used_layer},
                {"n_tokens", static_cast<int>(tokens.size())},
                {"n_embd", fwd.n_embd},
                {"activations", tensor_json_f32(
                    fwd.activations,
                    {static_cast<int>(tokens.size()), fwd.n_embd})},
            };
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /harvest/layers — the per-layer activation SUMMARY: one causal forward, and every layer's
    // residual is reduced to the L2 norm of each token's hidden state (GgmlAdapter::layer_summary). Returns
    // the depth x position "MRI" map ([n_layer][n_tokens] norms) + a per-layer mean, in ONE forward — the
    // cheap cross-depth view /harvest (single-layer, full tensor) can't give without n_layer separate calls.
    svr.Post("/harvest/layers", [&](const httplib::Request& req, httplib::Response& res) {
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
        const std::vector<int> tokens = model->encode(text);
        if (tokens.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "text tokenized to zero tokens"}}.dump(), "application/json");
            return;
        }
        if (static_cast<int>(tokens.size()) > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "text too long for one forward"},
                                 {"n_tokens", static_cast<int>(tokens.size())},
                                 {"n_ctx", n_ctx}}.dump(), "application/json");
            return;
        }
        try {
            LayerSummary ls;
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                ad.set_causal(true);              // summary under causal attention (also clears the KV)
                ls = ad.layer_summary(tokens);    // one forward, all layers (the method restores its own flag)
            }
            json pieces = json::array();
            for (int id : tokens) pieces.push_back(model->decode({id}));
            json norms = json::array();
            json layer_mean = json::array();
            for (const std::vector<float>& layer : ls.norms) {
                json row = json::array();
                double sum = 0.0;
                for (float v : layer) { row.push_back(v); sum += v; }
                norms.push_back(row);
                layer_mean.push_back(layer.empty() ? 0.0 : sum / static_cast<double>(layer.size()));
            }
            json resp = {
                {"tokens", pieces},
                {"n_tokens", ls.n_tokens},
                {"n_layer", ls.n_layer},
                {"norms", norms},           // [n_layer][n_tokens]: |residual| per token per layer
                {"layer_mean", layer_mean}, // [n_layer]: mean token norm at each layer
            };
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /score — teacher-forced per-token logprob of a continuation, AR-only (the
    // reproduce-and-prove foundation). No sampling: ONE causal decode of prompt++continuation
    // (GgmlAdapter::ar_forward_score) reads back, for each continuation token, what the model
    // actually thought of the token it was FORCED to see next -- the log-softmax probability the
    // model assigned to A[i] at the position that predicts it. Logits at absolute sequence index j
    // predict token j+1, so A[i] (absolute index n_p+i) reads from the log-softmax of logits at
    // n_p+i-1: logits_for = {n_p-1 .. n_p+n_a-2} (n_a rows; no logits needed for earlier prompt
    // positions). Continuation ids are the PRIMARY form (exact -- from a stored trace); a raw
    // `continuation` string is a fallback that retokenizes independently and can drift at the
    // prompt/continuation BPE boundary -- flagged `boundary_approximate` in the response.
    // Request: {prompt|prompt_ids, continuation_ids|continuation, topk?, steer?:{concept?,coef,
    // layer}, steer_vec?}. Response: {n_prompt, n_cont, tokens:[{id,piece,logprob,topk?}], sum_logprob}.
    svr.Post("/score", [&](const httplib::Request& req, httplib::Response& res) {
        if (!ar_mode) {  // the AR next-token factorization; diffusion has no equivalent one-shot read
            res.status = 400;
            res.set_content(json{{"error", "score requires an autoregressive model"}}.dump(),
                            "application/json");
            return;
        }
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }

        const int vocab_n = model->config().vocab_size;
        auto ids_in_range = [&](const std::vector<int>& ids) {
            for (int id : ids) if (id < 0 || id >= vocab_n) return false;
            return true;
        };

        // Prompt: token ids take precedence (exact -- what a stored trace/generation actually saw);
        // else tokenize `prompt` with the SAME model->encode /v1/completions uses, so n_p matches.
        std::vector<int> prompt_ids;
        if (body.contains("prompt_ids") && body["prompt_ids"].is_array()) {
            for (const auto& v : body["prompt_ids"]) prompt_ids.push_back(v.get<int>());
        } else {
            prompt_ids = model->encode(body.value("prompt", std::string()));
        }
        if (prompt_ids.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty prompt"}}.dump(), "application/json");
            return;
        }
        if (!ids_in_range(prompt_ids)) {
            res.status = 400;
            res.set_content(json{{"error", "prompt_ids out of vocab range"}}.dump(), "application/json");
            return;
        }

        // Continuation: token ids are PRIMARY; text is a documented-approximate fallback (BPE
        // boundary merges mean tokenizing prompt+continuation separately can differ from tokenizing
        // their concatenation).
        std::vector<int> cont_ids;
        bool boundary_approximate = false;
        if (body.contains("continuation_ids") && body["continuation_ids"].is_array()) {
            for (const auto& v : body["continuation_ids"]) cont_ids.push_back(v.get<int>());
        } else if (body.contains("continuation")) {
            cont_ids = model->encode(body.value("continuation", std::string()));
            boundary_approximate = true;
        }
        if (cont_ids.empty()) {
            res.status = 400;
            res.set_content(json{{"error", "empty continuation (need continuation_ids or continuation)"}}.dump(),
                            "application/json");
            return;
        }
        if (!ids_in_range(cont_ids)) {
            res.status = 400;
            res.set_content(json{{"error", "continuation_ids out of vocab range"}}.dump(), "application/json");
            return;
        }

        const int n_p = static_cast<int>(prompt_ids.size());
        const int n_a = static_cast<int>(cont_ids.size());
        if (n_p + n_a > n_ctx) {
            res.status = 400;
            res.set_content(json{{"error", "prompt + continuation exceeds n_ctx"},
                                 {"n_prompt", n_p}, {"n_cont", n_a}, {"n_ctx", n_ctx}}.dump(),
                            "application/json");
            return;
        }

        const int topk = body.value("topk", 0);

        // Same steer parsing/lease discipline as /v1/completions: a NAMED concept (steer_probes) or
        // a RAW direction (steer_vec -- how the studio's tone dials arrive) as a control vector over
        // [lo,hi] layers. Cleared on every exit (including a throw) so a scored request never leaks
        // a steered lease back into the pool.
        std::string steer_concept; double steer_coef = 0.0; int steer_layer = 0;
        if (body.contains("steer") && body["steer"].is_object()) {
            steer_concept = body["steer"].value("concept", std::string());
            steer_coef = body["steer"].value("coef", 0.0);
            steer_layer = body["steer"].value("layer", 0);
        }
        std::vector<float> steer_vec;
        if (body.contains("steer_vec") && body["steer_vec"].is_array()) {
            steer_vec = body["steer_vec"].get<std::vector<float>>();
        }

        std::vector<int> tokens = prompt_ids;
        tokens.insert(tokens.end(), cont_ids.begin(), cont_ids.end());
        std::vector<int> logits_for(static_cast<size_t>(n_a));
        for (int i = 0; i < n_a; ++i) logits_for[static_cast<size_t>(i)] = n_p - 1 + i;
        const int n_total = n_p + n_a;

        // Circuit-tracer slice 1 (notes/CIRCUIT_TRACER_DESIGN.md): teacher-forced score UNDER an
        // intervention. write:{layer, positions:[absolute seq pos], values:[positions*n_embd rows]}
        // overwrites those residual rows at `layer` during this ONE forward (GgmlAdapter::write_state
        // — the /state activation patch, leak-free clear on every exit). capture:{layers, positions}
        // returns those positions' residual rows from the multi-layer capture plane, read off the
        // SAME forward — under the patch, if one is armed (at the write layer itself the captured row
        // is the PRE-edit state, same convention as the read tap). Together they are the tracer's two
        // primitives: "score with node ablated" and "patch A, capture at B" in one call each.
        // `write` may be ONE spec {layer, positions, values} or an ARRAY of them — the array form is
        // the tracer's joint arm (all candidate nodes ablated simultaneously, across layers, in one
        // forward — GgmlAdapter::add_write_state per spec).
        struct WriteReq { int layer; std::vector<int> positions; std::vector<float> values; };
        // Shared write-spec parser: one spec {layer, positions, values} or an array of them.
        // Returns an error string ("" = ok) so both the top-level `write` field and each
        // batched arm's `write` field parse IDENTICALLY -- no drift between the two shapes.
        auto parse_write_specs = [](const json& field, std::vector<WriteReq>& out) -> std::string {
            json specs = json::array();
            if (field.is_object()) specs.push_back(field);
            else if (field.is_array()) specs = field;
            else return "write must be an object or array";
            for (const json& wb : specs) {
                WriteReq w{};
                w.layer = wb.is_object() ? wb.value("layer", 0) : 0;
                if (wb.is_object() && wb.contains("positions") && wb["positions"].is_array())
                    w.positions = wb["positions"].get<std::vector<int>>();
                if (wb.is_object() && wb.contains("values") && wb["values"].is_array())
                    w.values = wb["values"].get<std::vector<float>>();
                if (w.layer < 1 || w.positions.empty() || w.values.empty())
                    return "each write spec needs {layer >= 1, positions:[int], values:[float]}";
                out.push_back(std::move(w));
            }
            return "";
        };
        std::vector<WriteReq> write_reqs;
        if (body.contains("write")) {
            json specs = json::array();
            if (body["write"].is_object()) specs.push_back(body["write"]);
            else if (body["write"].is_array()) specs = body["write"];
            for (const json& wb : specs) {
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
                    if (p < 0 || p >= n_total) {
                        res.status = 400;
                        res.set_content(json{{"error", "write position out of range"}, {"position", p},
                                             {"n_tokens", n_total}}.dump(), "application/json");
                        return;
                    }
                }
                write_reqs.push_back(std::move(w));
            }
            if (write_reqs.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "write must be an object or non-empty array of specs"}}.dump(),
                                "application/json");
                return;
            }
        }
        // attn_knockout: [{layer, head?, queries:[int], keys:[int], renormalize?}] — sever
        // "query position reads key position" at a layer/head. This is the cross-position
        // primitive residual patching could not provide (notes/CIRCUIT_TRACER_DESIGN.md §5f):
        // cutting the EDGE dodges both the re-supply problem and the unpatchable last layer.
        // Needs the server started with --no-flash-attn (else the softmax is fused and never
        // materializes) — refused cleanly rather than silently ignored.
        // attn_capture: {"query": <board pos>} — READ (not modify) that query position's
        // post-softmax attention row at every layer, head-averaged. The correlational heatmap the
        // R1 head-to-head compares against knockout's causal ranking. Same --no-flash-attn
        // constraint as knockout, same clean refusal.
        int attn_capture_query = -1;
        if (body.contains("attn_capture")) {
            if (body["attn_capture"].is_object())
                attn_capture_query = body["attn_capture"].value("query", -1);
            if (attn_capture_query < 0) {
                res.status = 400;
                res.set_content(json{{"error", "attn_capture must be {query: <position >= 0>}"}}.dump(),
                                "application/json");
                return;
            }
        }
        std::vector<GgmlAdapter::AttnKnockout> knockouts;
        if (body.contains("attn_knockout")) {
            json ks = json::array();
            if (body["attn_knockout"].is_object()) ks.push_back(body["attn_knockout"]);
            else if (body["attn_knockout"].is_array()) ks = body["attn_knockout"];
            for (const json& kb : ks) {
                GgmlAdapter::AttnKnockout k;
                if (!kb.is_object()) continue;
                k.layer = kb.value("layer", -1);
                k.head = kb.value("head", -1);
                k.renormalize = kb.value("renormalize", false);
                if (kb.contains("queries") && kb["queries"].is_array())
                    k.queries = kb["queries"].get<std::vector<int>>();
                if (kb.contains("keys") && kb["keys"].is_array())
                    k.keys = kb["keys"].get<std::vector<int>>();
                if (k.layer < 0 || k.queries.empty() || k.keys.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "each attn_knockout needs {layer >= 0, "
                                                   "queries:[int], keys:[int]}"}}.dump(),
                                    "application/json");
                    return;
                }
                knockouts.push_back(std::move(k));
            }
            if (knockouts.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "attn_knockout must be an object or non-empty array"}}.dump(),
                                "application/json");
                return;
            }
        }

        std::vector<int> capture_layers, capture_positions;
        if (body.contains("capture") && body["capture"].is_object()) {
            const json& cb = body["capture"];
            if (cb.contains("layers") && cb["layers"].is_array())
                capture_layers = cb["layers"].get<std::vector<int>>();
            if (cb.contains("positions") && cb["positions"].is_array())
                capture_positions = cb["positions"].get<std::vector<int>>();
            if (capture_layers.empty() || capture_positions.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "capture needs {layers:[int], positions:[int]}"}}.dump(),
                                "application/json");
                return;
            }
            for (int p : capture_positions) {
                if (p < 0 || p >= n_total) {
                    res.status = 400;
                    res.set_content(json{{"error", "capture position out of range"}, {"position", p},
                                         {"n_tokens", n_total}}.dump(), "application/json");
                    return;
                }
            }
        }

        // arms: [{write: <spec|array>}, ...] -- batched multi-arm teacher-forced scoring (the
        // engine-debt per-branch-interventions item). ONE forward carries every arm as its own
        // sequence; each arm's writes apply only to its own copy. An arm with no write is a
        // baseline arm. Mutually exclusive with top-level write/capture/attn_knockout/
        // attn_capture (unvalidated tensor layouts under multi-seq batching are REFUSED, not
        // silently risked). Positions are validated here so an out-of-range arm write can never
        // silently no-op.
        std::vector<std::vector<WriteReq>> arm_writes;
        if (body.contains("arms")) {
            if (!body["arms"].is_array() || body["arms"].empty() || body["arms"].size() > 16) {
                res.status = 400;
                res.set_content(json{{"error", "arms must be a non-empty array of at most 16 "
                                               "objects"}}.dump(), "application/json");
                return;
            }
            if (!write_reqs.empty() || !capture_layers.empty() || !knockouts.empty()
                || attn_capture_query >= 0) {
                res.status = 400;
                res.set_content(json{{"error", "arms cannot be combined with top-level write/"
                                               "capture/attn_knockout/attn_capture -- put each "
                                               "arm's write inside the arm; the other surfaces "
                                               "are unvalidated under multi-seq batching and are "
                                               "refused rather than silently risked"}}.dump(),
                                "application/json");
                return;
            }
            if (static_cast<long long>(body["arms"].size()) * n_total > n_ctx) {
                res.status = 400;
                res.set_content(json{{"error", "arms * sequence length exceeds n_ctx (the arms "
                                               "share one kv_unified pool)"},
                                     {"n_arms", body["arms"].size()}, {"seq_len", n_total},
                                     {"n_ctx", n_ctx}}.dump(), "application/json");
                return;
            }
            for (const json& ab : body["arms"]) {
                std::vector<WriteReq> ws;
                if (ab.is_object() && ab.contains("write")) {
                    const std::string err = parse_write_specs(ab["write"], ws);
                    if (!err.empty()) {
                        res.status = 400;
                        res.set_content(json{{"error", "arm " + std::to_string(arm_writes.size())
                                                       + ": " + err}}.dump(), "application/json");
                        return;
                    }
                    for (const WriteReq& w : ws)
                        for (int p : w.positions)
                            if (p < 0 || p >= n_total) {
                                res.status = 400;
                                res.set_content(json{{"error", "arm write position out of range"},
                                                     {"arm", arm_writes.size()}, {"position", p},
                                                     {"n_tokens", n_total}}.dump(),
                                                "application/json");
                                return;
                            }
                } else if (!ab.is_object()) {
                    res.status = 400;
                    res.set_content(json{{"error", "each arm must be an object (use {} for a "
                                                   "baseline arm)"}}.dump(), "application/json");
                    return;
                }
                arm_writes.push_back(std::move(ws));
            }
        }
        const bool has_arms = !arm_writes.empty();

        // Head-output units (R5, notes/HEAD_UNITS_DESIGN.md): head_capture reads per-Q-head
        // slice L2 norms (+ full merged rows with rows:true) at kqv_out-<il>; head_write
        // overwrites head slices before W_o consumes them. Both work with flash attention ON
        // (the head OUTPUT materializes regardless of the softmax fusion). Refused alongside
        // arms (unvalidated multi-seq layout, same rule as the other hooks).
        std::vector<int> head_cap_layers, head_cap_positions;
        bool head_cap_rows = false;
        if (body.contains("head_capture")) {
            const json& hb = body["head_capture"];
            if (hb.is_object()) {
                if (hb.contains("layers") && hb["layers"].is_array())
                    head_cap_layers = hb["layers"].get<std::vector<int>>();
                if (hb.contains("positions") && hb["positions"].is_array())
                    head_cap_positions = hb["positions"].get<std::vector<int>>();
                head_cap_rows = hb.value("rows", false);
            }
            if (head_cap_layers.empty() || head_cap_positions.empty()) {
                res.status = 400;
                res.set_content(json{{"error", "head_capture needs {layers:[int], positions:[int]}"}}.dump(),
                                "application/json");
                return;
            }
            for (int p : head_cap_positions)
                if (p < 0 || p >= n_total) {
                    res.status = 400;
                    res.set_content(json{{"error", "head_capture position out of range"},
                                         {"position", p}, {"n_tokens", n_total}}.dump(),
                                    "application/json");
                    return;
                }
        }
        std::vector<GgmlAdapter::HeadWrite> head_writes;
        if (body.contains("head_write")) {
            json specs = json::array();
            if (body["head_write"].is_object()) specs.push_back(body["head_write"]);
            else if (body["head_write"].is_array()) specs = body["head_write"];
            for (const json& wb : specs) {
                GgmlAdapter::HeadWrite w;
                w.layer = wb.is_object() ? wb.value("layer", -1) : -1;
                w.head = wb.is_object() ? wb.value("head", -1) : -1;
                if (wb.is_object() && wb.contains("positions") && wb["positions"].is_array())
                    w.positions = wb["positions"].get<std::vector<int>>();
                if (wb.is_object() && wb.contains("values") && wb["values"].is_array())
                    w.values = wb["values"].get<std::vector<float>>();
                if (w.layer < 0 || w.head < 0 || w.positions.empty() || w.values.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "each head_write needs {layer >= 0, head >= 0, "
                                                   "positions:[int], values:[float]}"}}.dump(),
                                    "application/json");
                    return;
                }
                for (int p : w.positions)
                    if (p < 0 || p >= n_total) {
                        res.status = 400;
                        res.set_content(json{{"error", "head_write position out of range"},
                                             {"position", p}, {"n_tokens", n_total}}.dump(),
                                        "application/json");
                        return;
                    }
                head_writes.push_back(std::move(w));
            }
        }
        if ((!head_cap_layers.empty() || !head_writes.empty()) && has_arms) {
            res.status = 400;
            res.set_content(json{{"error", "head_capture/head_write cannot be combined with arms "
                                           "(unvalidated under multi-seq batching)"}}.dump(),
                            "application/json");
            return;
        }

        try {
            ForwardResult fwd;
            CaptureFrame cap_frame;             // filled synchronously by the capture sink (if armed)
            std::map<int, std::vector<float>> attn_rows_out;   // attn_capture rows, copied pre-cleanup
            std::map<int, std::map<int, std::vector<float>>> head_norms_out, head_rows_out;
            std::map<std::string, int> head_dims_out;
            std::vector<int> cap_layers_armed;  // layers that survived set_capture_layers validation
            {
                ContextPool::Lease lease = pool.acquire();
                GgmlAdapter& ad = *lease;
                const bool steering = !steer_concept.empty() && steer_coef != 0.0 && steer_probes.ready();
                const bool raw_steer = !steer_vec.empty();
                const bool writing = !write_reqs.empty();
                const bool capturing = !capture_layers.empty();
                const bool knocking = !knockouts.empty();
                const bool head_capturing = !head_cap_layers.empty();
                const bool head_writing = !head_writes.empty();
                auto cleanup = [&]() {
                    if (steering || raw_steer) ad.clear_steer();
                    if (writing || has_arms) ad.clear_write();
                    if (capturing) { ad.set_capture_sink({}); ad.set_capture_layers({}); }
                    if (knocking) ad.clear_attn_knockouts();
                    if (head_capturing) ad.clear_head_capture();
                    if (head_writing) ad.clear_head_writes();
                    if (attn_capture_query >= 0) {
                        // rows are copied into the response BEFORE cleanup runs on the success
                        // path; on the throw path there is nothing to keep.
                        ad.clear_attn_capture();
                    }
                };
                try {
                    ad.set_causal(true);  // the AR forward needs causal attention (also => a clean KV)
                    if (steering || raw_steer) {
                        const int nl = ad.n_layer();
                        int lo, hi;
                        if (steer_layer >= 1) { lo = hi = (steer_layer < nl ? steer_layer : nl - 1); }
                        else { const int tl = nl * 2 / 3;
                               lo = (tl - 2 > 1 ? tl - 2 : 1); hi = (tl + 2 < nl ? tl + 2 : nl - 1); }
                        if (lo < 1) lo = 1;
                        if (steering) {
                            ad.set_steer(build_steer_cvec(steer_probes, steer_concept, steer_coef, lo, hi, nl), lo, hi);
                        } else {
                            const int ne = static_cast<int>(steer_vec.size());
                            std::vector<float> cvec(static_cast<size_t>(ne) * nl, 0.0f);
                            const double c = steer_coef != 0.0 ? steer_coef : 1.0;
                            for (int L = lo; L <= hi; ++L) {
                                if (L < 1 || L >= nl) continue;
                                float* slice = cvec.data() + static_cast<size_t>(L - 1) * ne;
                                for (int i = 0; i < ne; ++i) slice[i] = static_cast<float>(c * steer_vec[i]);
                            }
                            ad.set_steer(cvec, lo, hi);
                        }
                    }
                    if (writing) {
                        ad.clear_write();
                        for (const WriteReq& w : write_reqs) {
                            if (!ad.add_write_state(w.layer, w.positions, w.values)) {
                                throw std::invalid_argument(
                                    "write rejected: layer must be in [1, n_layer) and values.size must "
                                    "equal positions.size * n_embd (n_embd " + std::to_string(ad.n_embd()) +
                                    ", layer " + std::to_string(w.layer) + ")");
                            }
                        }
                    }
                    if (knocking) {
                        // Refuse rather than silently no-op: with flash attention the softmax is
                        // fused and the weights we would zero never exist as a tensor.
                        if (!ad.knockout_available())
                            throw std::invalid_argument(
                                "attn_knockout requires the server to be started with --no-flash-attn "
                                "(flash attention fuses the softmax, so the attention weights never "
                                "materialize and the knockout would be silently ignored)");
                        for (const auto& k : knockouts) {
                            if (k.layer >= ad.n_layer())
                                throw std::invalid_argument("attn_knockout layer out of range [0, n_layer)");
                            if (k.head >= ad.n_head())
                                throw std::invalid_argument("attn_knockout head out of range [0, n_head)");
                        }
                        ad.set_attn_knockouts(knockouts);
                    }
                    if (capturing) {
                        ad.set_capture_layers(capture_layers);
                        cap_layers_armed = ad.capture_layers();  // invalid layers were dropped
                        if (cap_layers_armed.empty())
                            throw std::invalid_argument("capture layers all out of range (1..n_layer-1)");
                        ad.set_capture_sink([&cap_frame](CaptureFrame&& f) { cap_frame = std::move(f); });
                    }
                    if (attn_capture_query >= 0) {
                        if (!ad.knockout_available())
                            throw std::invalid_argument(
                                "attn_capture requires the server to be started with --no-flash-attn "
                                "(flash attention fuses the softmax, so the attention weights never "
                                "materialize and there would be nothing to capture)");
                        ad.set_attn_capture(attn_capture_query);
                    }
                    if (head_capturing)
                        ad.set_head_capture(head_cap_layers, head_cap_positions, head_cap_rows);
                    if (head_writing) ad.set_head_writes(head_writes);
                    if (has_arms) {
                        // Per-arm writes ride the standard write path: arm a's position p is
                        // batch row a*n_total + p (ar_forward_score_arms runs with
                        // write_from_ == 0, so rows ARE positions) -- eval_cb unchanged.
                        ad.clear_write();
                        for (size_t a = 0; a < arm_writes.size(); ++a) {
                            for (const WriteReq& w : arm_writes[a]) {
                                std::vector<int> rows = w.positions;
                                for (int& p : rows) p += static_cast<int>(a) * n_total;
                                if (!ad.add_write_state(w.layer, rows, w.values)) {
                                    throw std::invalid_argument(
                                        "arm " + std::to_string(a) + " write rejected: layer must "
                                        "be in [1, n_layer) and values.size must equal "
                                        "positions.size * n_embd");
                                }
                            }
                        }
                        fwd = ad.ar_forward_score_arms(tokens, logits_for,
                                                       static_cast<int>(arm_writes.size()));
                    } else {
                        fwd = ad.ar_forward_score(tokens, logits_for);
                    }
                    if (attn_capture_query >= 0) attn_rows_out = ad.attn_rows();  // copy BEFORE cleanup
                    if (head_capturing || head_writing) {
                        head_norms_out = ad.head_norms();
                        head_rows_out = ad.head_rows();
                        head_dims_out = ad.head_dims();
                    }
                    cleanup();
                } catch (...) {
                    cleanup();
                    throw;
                }
            }

            const int vocab = fwd.vocab;
            // One arm's token rows starting at result-row `row_base` (0 for the single-arm path;
            // arm_i * n_a for batched arms) -> (tokens json, sum_logprob).
            auto build_tokens = [&](int row_base) -> std::pair<json, double> {
                json tj = json::array();
                double sum = 0.0;
                for (int r = 0; r < n_a; ++r) {
                    const float* row = fwd.row(row_base + r);
                    // log-softmax over the vocab (max-subtract + logsumexp, float32).
                    float mx = row[0];
                    for (int t = 1; t < vocab; ++t) if (row[t] > mx) mx = row[t];
                    float sumexp = 0.0f;
                    for (int t = 0; t < vocab; ++t) sumexp += std::exp(row[t] - mx);
                    const float logZ = mx + std::log(sumexp);

                    const int actual = cont_ids[static_cast<size_t>(r)];
                    const float logprob = row[actual] - logZ;
                    sum += logprob;

                    json item{{"id", actual}, {"piece", model->decode({actual})}, {"logprob", logprob}};
                    if (topk > 0) {
                        std::vector<int> idx(static_cast<size_t>(vocab));
                        for (int t = 0; t < vocab; ++t) idx[static_cast<size_t>(t)] = t;
                        const int k = std::min(topk, vocab);
                        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                                          [&](int a, int b) { return row[a] > row[b]; });
                        json tk = json::array();
                        for (int j = 0; j < k; ++j) {
                            const int t = idx[static_cast<size_t>(j)];
                            tk.push_back({{"id", t}, {"piece", model->decode({t})}, {"logprob", row[t] - logZ}});
                        }
                        item["topk"] = tk;
                    }
                    tj.push_back(item);
                }
                return {tj, sum};
            };

            json resp = {{"n_prompt", n_p}, {"n_cont", n_a},
                         // The exact prompt token ids this forward used. Additive; closes the
                         // "no id-level tokenize surface" gap -- /v1/checkpoint takes ids, and a
                         // client that only has text had no honest way to produce them (piece
                         // strings from /harvest do not round-trip through BPE).
                         {"prompt_ids", prompt_ids}};
            if (has_arms) {
                // Per-arm blocks, arm-major, same token shape as the single path -- and NO
                // top-level tokens/sum_logprob (a client that sent arms reads arms; a defaulted
                // top-level copy of arm 0 would invite silent misreads).
                json arms_json = json::array();
                for (size_t a = 0; a < arm_writes.size(); ++a) {
                    auto [tj, sum] = build_tokens(static_cast<int>(a) * n_a);
                    arms_json.push_back({{"tokens", tj}, {"sum_logprob", sum},
                                         {"n_writes", arm_writes[a].size()}});
                }
                resp["arms"] = arms_json;
                resp["n_arms"] = arm_writes.size();
                // MEASURED, not hypothetical (2026-07-22, 7B/CUDA): batched arms are NOT
                // bit-exact vs sequential /score (abs logprobs drift up to ~1.5e-1 nats with
                // batch shape) and arm-minus-baseline DELTAS are not regime-consistent either
                // (up to ~1.9e-1 nats -- a perturbed forward diverges non-common-mode). The
                // batched regime IS deterministic across repeats. Contract: arms are for
                // SCREENING (rank candidates, then re-measure survivors sequentially) -- never
                // for receipts. The label ships on every response so no client can miss it.
                resp["numerical_regime"] = "batched_approximate";
                resp["regime_note"] = "arm logprobs/deltas differ from sequential /score by "
                                      "batch-shape FP (measured up to ~0.19 nats); deterministic "
                                      "within this regime; use for screening, re-measure "
                                      "anything you intend to claim";
            } else {
                auto [tok_json, sum_logprob] = build_tokens(0);
                resp["tokens"] = tok_json;
                resp["sum_logprob"] = sum_logprob;
            }
            if (boundary_approximate) resp["boundary_approximate"] = true;
            if (!write_reqs.empty()) {
                resp["write_applied"] = true;
                resp["n_writes"] = static_cast<int>(write_reqs.size());
            }
            if (!knockouts.empty()) {
                resp["knockout_applied"] = true;
                resp["n_knockouts"] = static_cast<int>(knockouts.size());
            }
            if (attn_capture_query >= 0) {
                // attn_rows: {"<layer>": [n_kv floats], ...} — the head-mean post-softmax row for
                // the requested query position. Empty object (never fabricated zeros) if the
                // position fell outside the decoded segment at every layer.
                json ar = json::object();
                for (const auto& lv : attn_rows_out)
                    if (!lv.second.empty()) ar[std::to_string(lv.first)] = lv.second;
                resp["attn_rows"] = ar;
                resp["attn_capture_query"] = attn_capture_query;
            }
            if (!head_cap_layers.empty() || !head_writes.empty()) {
                // head_dims doubles as the slice-0 shape probe: d_head == 0 means ne0 did not
                // divide by n_head on this architecture -- nothing was captured or written, and
                // the caller can see exactly why instead of getting silently-wrong slices.
                resp["head_dims"] = head_dims_out;
                if (!head_cap_layers.empty()) {
                    json hn = json::object();
                    for (const auto& ln : head_norms_out) {
                        json per_pos = json::object();
                        for (const auto& pv : ln.second) per_pos[std::to_string(pv.first)] = pv.second;
                        hn[std::to_string(ln.first)] = per_pos;
                    }
                    resp["head_norms"] = hn;
                    if (head_cap_rows) {
                        json hr = json::object();
                        for (const auto& ln : head_rows_out) {
                            json per_pos = json::object();
                            for (const auto& pv : ln.second) per_pos[std::to_string(pv.first)] = pv.second;
                            hr[std::to_string(ln.first)] = per_pos;
                        }
                        resp["head_rows"] = hr;
                    }
                }
                if (!head_writes.empty()) {
                    resp["head_write_applied"] = true;
                    resp["n_head_writes"] = head_writes.size();
                }
            }
            if (!capture_layers.empty()) {
                // captured: {"<layer>": {"<pos>": [n_embd floats], ...}, ...} — the residual rows the
                // tracer keeps for computing edited values (mean-/directional-ablate) client-side.
                json cap_json = json::object();
                for (const auto& lv : cap_frame.layers) {
                    json rows = json::object();
                    for (int p : capture_positions) {
                        if (p < 0 || p >= cap_frame.rows) continue;
                        const float* row = lv.second.data() + static_cast<size_t>(p) * cap_frame.n_embd;
                        rows[std::to_string(p)] = std::vector<float>(row, row + cap_frame.n_embd);
                    }
                    cap_json[std::to_string(lv.first)] = std::move(rows);
                }
                // NO SILENT EMPTIES. A layer can pass the [1, n_layer) range check and still yield
                // nothing. The known case is the LAST layer (n_layer-1): llama.cpp applies the
                // inp_out_ids optimization there, materializing ONLY the rows that produce logits
                // (one row for a single-target /score) instead of all positions -- so `l_out-<last>`
                // fires but is the wrong shape for a whole-sequence capture, and fire_capture drops
                // it. Returning {} would read as "captured, all zeros" to a caller. Report the gap;
                // a request where NOTHING landed is an outright 400.
                //
                // This also BOUNDS cross-position path patching: the last layer cannot be held, so a
                // source whose influence is re-imported by final-layer attention cannot be measured
                // by patching a destination column (measured: 0% routed at every held depth for a
                // late-layer source -- notes/CIRCUIT_TRACER_DESIGN.md 5f).
                json missing = json::array();
                for (int L : cap_layers_armed)
                    if (!cap_json.contains(std::to_string(L))) missing.push_back(L);
                if (cap_json.empty()) {
                    res.status = 400;
                    res.set_content(json{{"error", "capture produced no rows for any requested layer; "
                                                   "the last layer (n_layer-1) materializes only the "
                                                   "logit rows (inp_out_ids), so whole-sequence capture "
                                                   "needs layer <= n_layer-2"},
                                         {"requested", capture_layers},
                                         {"armed", cap_layers_armed}}.dump(), "application/json");
                    return;
                }
                resp["captured"] = std::move(cap_json);
                resp["n_embd"] = cap_frame.n_embd;
                if (!missing.empty()) resp["capture_missing"] = std::move(missing);
            }
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /apply_template — render chat messages into a prompt string using THE MODEL'S OWN embedded
    // chat template (from the GGUF's tokenizer.chat_template metadata), so every model gets its correct
    // format instead of a hardcoded ChatML. This is the seam that makes clozn model-agnostic: the Python
    // substrate sends {messages:[{role,content}]}, the engine templates per-model (Qwen -> ChatML
    // <|im_start|>, Llama-3 -> <|start_header_id|>, Gemma -> <start_of_turn>, ...) via the
    // pinned backend's full Jinja renderer. No context/KV
    // and no sampling: pure model-metadata + string work, so NO pool lease is taken (nothing to leak,
    // fully concurrent with generation). No-embedded-template is surfaced as a clean 400, never silently
    // mis-formatted. The prompt is tokenized by the same GgmlModel::encode seam generation uses, so
    // prompt_tokens is exact worker-tokenizer evidence rather than a Python-side estimate. Body:
    // {messages:[{role,content}], add_assistant?:bool=true} -> {prompt, prompt_tokens, template_source}.
    svr.Post("/apply_template", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON body"}}.dump(), "application/json");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "'messages' must be a non-empty array of {role, content}"}}.dump(),
                            "application/json");
            return;
        }
        if (!ctx.chat_templates.available()) {
            res.status = 400;
            res.set_content(json{{"error", "model has no embedded chat template; cannot format messages "
                                           "per-model (re-convert the GGUF with its tokenizer.chat_template, "
                                           "or send a pre-rendered prompt to /v1/completions)"}}.dump(),
                            "application/json");
            return;
        }
        std::vector<std::pair<std::string, std::string>> messages;
        messages.reserve(body["messages"].size());
        for (const auto& m : body["messages"]) {
            if (!m.is_object()) {
                res.status = 400;
                res.set_content(json{{"error", "each message must be an object with role + content"}}.dump(),
                                "application/json");
                return;
            }
            const std::string role = m.value("role", std::string());
            if (role.empty() || !m.contains("content") || !m["content"].is_string()) {
                res.status = 400;
                res.set_content(json{{"error", "each message must have a non-empty role and string content"}}.dump(),
                                "application/json");
                return;
            }
            messages.emplace_back(role, m["content"].get<std::string>());
        }
        const bool add_assistant = body.value("add_assistant", true);
        try {
            const std::string prompt = ctx.chat_templates.apply(messages, add_assistant);
            const int prompt_tokens = static_cast<int>(ctx.model->encode(prompt).size());
            json resp = {{"prompt", prompt}, {"prompt_tokens", prompt_tokens},
                         {"template_source", "model"},
                         {"renderer", "jinja"}};
            res.set_content(dump_json(resp), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /prepare_chat -- PRIVATE worker seam. It executes llama-common's OpenAI-compatible
    // message/tool parsing and the loaded model's embedded Jinja template, returning the portable
    // prompt + grammar + parser descriptor. This does not generate, qualify, or publish support for
    // structured I/O; the supervisor remains responsible for exact model/template qualification.
    svr.Post("/prepare_chat", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON object body"}}.dump(), "application/json");
            return;
        }
        if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty()) {
            res.status = 400;
            res.set_content(json{{"error", "'messages' must be a non-empty array"}}.dump(),
                            "application/json");
            return;
        }
        if (body.contains("tools") && !body["tools"].is_array()) {
            res.status = 400;
            res.set_content(json{{"error", "'tools' must be an array"}}.dump(), "application/json");
            return;
        }
        if (body.contains("tool_choice") &&
            !body["tool_choice"].is_string() && !body["tool_choice"].is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "'tool_choice' must be a string or object"}}.dump(),
                            "application/json");
            return;
        }
        if (body.contains("json_schema") && !body["json_schema"].is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "'json_schema' must be an object"}}.dump(),
                            "application/json");
            return;
        }
        if (!ctx.chat_templates.available()) {
            res.status = 400;
            res.set_content(json{{"error", "model has no embedded chat template; cannot prepare chat I/O"}}.dump(),
                            "application/json");
            return;
        }

        try {
            ChatTemplateRequest request;
            request.messages_json = body.at("messages").dump();
            request.tools_json = body.value("tools", json::array()).dump();
            request.tool_choice_json = body.value("tool_choice", json("auto")).dump();
            if (body.contains("json_schema")) request.json_schema_json = body["json_schema"].dump();
            request.parallel_tool_calls = body.value("parallel_tool_calls", false);
            request.add_generation_prompt = body.value("add_generation_prompt", true);
            request.enable_thinking = body.value("enable_thinking", true);
            request.reasoning_format = body.value("reasoning_format", std::string("none"));

            json response = prepared_chat_json(ctx.chat_templates.prepare(request));
            response["template_source"] = "model";
            response["renderer"] = "llama-common";
            res.set_content(dump_json(response), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });

    // POST /parse_chat -- PRIVATE inverse seam. The complete descriptor from /prepare_chat must be
    // returned with the raw model output so parser format, generation prefix, and PEG state cannot
    // silently drift between preparation and parsing.
    svr.Post("/parse_chat", [&](const httplib::Request& req, httplib::Response& res) {
        json body = json::parse(req.body, nullptr, /*allow_exceptions=*/false);
        if (body.is_discarded() || !body.is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "invalid JSON object body"}}.dump(), "application/json");
            return;
        }
        if (!body.contains("prepared") || !body["prepared"].is_object()) {
            res.status = 400;
            res.set_content(json{{"error", "'prepared' must be an object returned by /prepare_chat"}}.dump(),
                            "application/json");
            return;
        }
        if (!body.contains("model_output") || !body["model_output"].is_string()) {
            res.status = 400;
            res.set_content(json{{"error", "'model_output' must be a string"}}.dump(),
                            "application/json");
            return;
        }

        try {
            const PreparedChat prepared = prepared_chat_from_json(body["prepared"]);
            const ParsedChat parsed = ctx.chat_templates.parse(
                prepared,
                body["model_output"].get<std::string>(),
                body.value("is_partial", false));
            res.set_content(dump_json(parsed_chat_json(parsed)), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json{{"error", e.what()}}.dump(), "application/json");
        }
    });
}

}  // namespace clozn
