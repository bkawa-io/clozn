// test_device_logits.cpp — the resolving experiment for the zero-copy §4.3 path (Phase 3).
// Gates the whole device-logits integration: with GPU offload (GGML_CUDA), does
// llama_get_logits_tensor (the CLOZN PATCH accessor) return a tensor that (1) lives in DEVICE
// memory (ggml_backend_buffer_is_host == false) so tensor->data is a usable CUDA pointer, and
// (2) IS the logits — a row D2H'd from it byte-matches the same row of llama_get_logits? If both
// hold, the kernel can read the logits in place with no full-vocab D2H. If (1) fails, the logits
// are host-resident and the honest story is the standalone cs_bench win only.
//
// usage: test_device_logits <model.gguf>   (offloads all layers; needs a CUDA-capable llama)
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: test_device_logits <model.gguf>\n");
        return 1;
    }
    const char* prompt = "def add(a, b): return a +";
    const int n_mask = 4;

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 999;  // offload everything so the logits projection runs on the GPU
    llama_model* model = llama_model_load_from_file(argv[1], mp);
    if (!model) { std::fprintf(stderr, "load failed\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 64; cp.n_batch = 64; cp.n_ubatch = 64;
    llama_context* ctx = llama_init_from_model(model, cp);
    if (!ctx) { std::fprintf(stderr, "ctx failed\n"); return 1; }
    llama_set_causal_attn(ctx, false);

    const llama_vocab* vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token MASK = llama_vocab_mask(vocab);
    if (MASK < 0) { std::fprintf(stderr, "model has no mask token\n"); return 1; }
    std::printf("mask token: %d\n", MASK);

    std::vector<llama_token> toks(128);
    int n = llama_tokenize(vocab, prompt, (int)std::strlen(prompt), toks.data(),
                           (int)toks.size(), false, true);
    toks.resize(n);
    const int prompt_len = n;
    std::vector<llama_token> board = toks;
    for (int i = 0; i < n_mask; ++i) board.push_back(MASK);
    const int n_board = (int)board.size();

    llama_batch batch = llama_batch_init(n_board, 0, 1);
    batch.n_tokens = n_board;
    for (int i = 0; i < n_board; ++i) {
        batch.token[i] = board[i]; batch.pos[i] = i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = 1;
    }
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }

    // The CLOZN PATCH accessor: the device-resident logits graph-output tensor.
    ggml_tensor* t = llama_get_logits_tensor(ctx);
    if (!t) { std::fprintf(stderr, "llama_get_logits_tensor returned null\n"); return 2; }
    const bool is_host = ggml_backend_buffer_is_host(t->buffer);
    std::printf("logits tensor: type=%d ne=[%lld,%lld] buffer_is_host=%s\n",
                (int)t->type, (long long)t->ne[0], (long long)t->ne[1], is_host ? "true" : "false");
    std::printf("CHECK A (device-resident under GPU offload): %s\n",
                is_host ? "HOST (zero-copy unavailable)" : "DEVICE (zero-copy unblocked)");

    // CHECK B: a row pulled from the tensor (D2H of just that row) equals the host logits row.
    // This also proves the raw graph order == llama_get_logits' reordered order for our
    // all-positions-requested decode (so the adapter can index tensor rows by batch position).
    const float* host = llama_get_logits(ctx);
    std::vector<float> devrow(n_vocab);
    int rows_ok = 0, rows_checked = 0;
    double max_abs_diff = 0.0;
    const int probe_rows[] = {prompt_len - 1, prompt_len, n_board - 1};
    for (int r : probe_rows) {
        if (r < 0 || r >= n_board) continue;
        ++rows_checked;
        ggml_backend_tensor_get(t, devrow.data(),
                                (size_t)r * n_vocab * sizeof(float),
                                (size_t)n_vocab * sizeof(float));
        const float* hrow = host + (size_t)r * n_vocab;
        double rmax = 0.0;
        for (int v = 0; v < n_vocab; ++v)
            rmax = std::max(rmax, (double)std::fabs(devrow[v] - hrow[v]));
        max_abs_diff = std::max(max_abs_diff, rmax);
        if (rmax == 0.0) ++rows_ok;
        std::printf("  row %d: max|dev-host| = %.3g\n", r, rmax);
    }
    const bool b_ok = (rows_ok == rows_checked && rows_checked > 0);
    std::printf("CHECK B (tensor IS the logits, byte-exact, same order): %d/%d rows -> %s (max diff %.3g)\n",
                rows_ok, rows_checked, b_ok ? "PASS" : "MISMATCH", max_abs_diff);

    // The zero-copy path is unblocked iff the tensor is device-resident AND byte-identical.
    const bool unblocked = !is_host && b_ok;
    std::printf("\nRESULT: %s\n", unblocked
        ? "ZERO-COPY UNBLOCKED (device-resident logits, exact)"
        : (is_host ? "host-resident logits: zero-copy needs offload/another path"
                   : "ordering mismatch: device path needs output_ids mapping"));

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return unblocked ? 0 : 2;
}
