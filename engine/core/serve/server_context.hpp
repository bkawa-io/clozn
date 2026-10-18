// serve/server_context.hpp -- the shared-state seam for the split clozn-server (Phase 12.4).
//
// Every route lambda in the old monolith captured main()'s locals by [&] (the model, the ContextPool,
// the white-box state structs, n_ctx/mask/config). To relocate handlers into their own translation units
// (routes_*.cpp), main() now gathers that state into ONE ServerContext, and each routes file exposes a
// register_<family>_routes(svr, ctx) that reads state through `ctx`. server_main.cpp builds the context
// once and calls each register_*. Mechanical: the handler bodies are byte-identical, they just read
// ctx.pool / ctx.jlens / ... (each register fn re-binds local aliases so the moved body stays verbatim).
#pragma once

#include "server_shared.hpp"            // GgmlModel, ContextPool, json
#include "chat_template_renderer.hpp"   // isolated full-GGUF Jinja renderer

namespace httplib { class Server; }  // fwd-decl: the header needn't pull in the single-header httplib

namespace clozn {

// The state the route families share. Reference members alias main()'s locals (which outlive the server
// for the whole svr.listen()), so nothing is copied; the model is a shared_ptr (used as *model / model->).
struct ServerContext {
    std::shared_ptr<GgmlModel> model;
    ContextPool& pool;
    ChatTemplateRenderer& chat_templates;
    int n_ctx;
    bool ar_mode;
    std::string model_path;
    // Runtime identity copied from server_main after startup.  Persistent reference-match
    // sessions are worker-local and bind to this generation; they never accept caller identities.
    std::string worker_generation_id;
    std::string model_sha256;
    std::string tokenizer_sha256;
};

// Route-family registrars linked into the private v1 worker.
void register_whitebox_routes(httplib::Server& svr, ServerContext& ctx);  // /score, /apply_template
void register_reference_match_routes(httplib::Server& svr, ServerContext& ctx); // private native-many probe

}  // namespace clozn
