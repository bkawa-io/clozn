// serve/server_context.hpp -- the shared-state seam for the split clozn-server (Phase 12.4).
//
// Every route lambda in the old monolith captured main()'s locals by [&] (the model, the ContextPool,
// the white-box state structs, n_ctx/mask/config). To relocate handlers into their own translation units
// (routes_*.cpp), main() now gathers that state into ONE ServerContext, and each routes file exposes a
// register_<family>_routes(svr, ctx) that reads state through `ctx`. server_main.cpp builds the context
// once and calls each register_*. Mechanical: the handler bodies are byte-identical, they just read
// ctx.pool / ctx.jlens / ... (each register fn re-binds local aliases so the moved body stays verbatim).
#pragma once

#include "server_shared.hpp"            // GgmlModel, ContextPool, ConceptProbes, SaeServe, JlensServe, json
#include "chat_template_renderer.hpp"   // isolated full-GGUF Jinja renderer

namespace httplib { class Server; }  // fwd-decl: the header needn't pull in the single-header httplib

namespace clozn {

// The state the route families share. Reference members alias main()'s locals (which outlive the server
// for the whole svr.listen()), so nothing is copied; the model is a shared_ptr (used as *model / model->).
struct ServerContext {
    std::shared_ptr<GgmlModel> model;
    ContextPool& pool;
    ConceptProbes& concept_probes;   // READ: sharp early-layer tap (white-box display)
    ConceptProbes& steer_probes;     // WRITE: mid-depth tap (steering control vector)
    SaeServe& sae_serve;
    JlensServe& jlens;
    ChatTemplateRenderer& chat_templates;
    int n_ctx;
    int mask_token;
    bool ar_mode;
    int gpu_layers;
    std::string model_path;
};

// Route-family registrars (one per routes_*.cpp). server_main.cpp calls each after building the context.
void register_jlens_routes(httplib::Server& svr, ServerContext& ctx);
void register_whitebox_routes(httplib::Server& svr, ServerContext& ctx);  // /harvest, /harvest/layers, /score, /apply_template
void register_state_routes(httplib::Server& svr, ServerContext& ctx);     // /state, /intervene
void register_reference_match_routes(httplib::Server& svr, ServerContext& ctx); // private native-many probe

}  // namespace clozn
