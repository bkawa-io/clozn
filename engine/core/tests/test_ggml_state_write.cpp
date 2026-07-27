// test_ggml_state_write.cpp — GAP #1 (task #43) on a REAL model: the read -> edit -> write -> observe
// loop through the ggml/llama backend. Proves GgmlAdapter::write_state (the patch-free activation patch
// via the eval callback) actually moves the model's output, and that clear_write() returns it to baseline.
//
// Needs a GGUF (pass as argv[1]); without one it SKIPS (returns 0) so the backend-free CI stays green.
//   build-ggml-cpu/.../test_ggml_state_write  <model.gguf>
// (open-dcoder-0.5b-f16.gguf is the small diffusion model the lab goldens use.)
#include <cassert>
#include <cmath>
#include <cstdio>
#include <optional>
#include <string>
#include <vector>

#include "clozn/model_ggml.hpp"

using namespace clozn;

static double l2diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30;
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) { const double d = double(a[i]) - double(b[i]); s += d * d; }
    return std::sqrt(s);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("test_ggml_state_write: SKIPPED (no GGUF arg; pass a model path to run the real loop)\n");
        return 0;
    }
    const std::string path = argv[1];
    const int mask_tok = 151665;  // open-dCoder <M>; harmless for a forward over real tokens

    GgmlAdapter adp(path, mask_tok, /*eos*/ -1, /*n_ctx*/ 512, /*n_gpu_layers*/ 0, /*passthrough*/ false);
    const int mid = std::max(1, adp.n_layer() * 2 / 3);   // a mid residual layer (l_out-<mid>)
    adp.set_tap_layer(mid);
    adp.set_emit_activations(true);

    std::vector<int> board = adp.encode("The capital of France is Paris");
    const int n = static_cast<int>(board.size());
    assert(n >= 3);
    Mask mask; mask.n = n; mask.data.assign(static_cast<size_t>(n) * n, 1);  // fully bidirectional
    std::vector<int> logits_for;
    for (int i = 0; i < n; ++i) logits_for.push_back(i);

    // 1) READ: baseline forward; capture logits + the tap (mid-layer residual, row j == position j).
    ForwardResult a = adp.forward(board, mask, nullptr, std::nullopt, logits_for);
    assert(!a.logits.empty());
    assert(a.n_embd > 0 && static_cast<int>(a.activations.size()) == n * a.n_embd);
    const std::vector<float> base_logits = a.logits;

    // 2) EDIT: take a middle position's residual row from the tap and perturb it strongly + unmistakably.
    const int pos = n / 2;
    const int d = a.n_embd;
    std::vector<float> row(a.activations.begin() + static_cast<long>(pos) * d,
                           a.activations.begin() + static_cast<long>(pos + 1) * d);
    for (float& x : row) x = x * 3.0f + 5.0f;

    // 3) WRITE the edited row back at the tap layer, position pos.
    const bool ok = adp.write_state(mid, {pos}, row);
    assert(ok);

    // 4) OBSERVE: forward again; the output must change (the edited residual propagated through the forward).
    ForwardResult b = adp.forward(board, mask, nullptr, std::nullopt, logits_for);
    const double moved = l2diff(base_logits, b.logits);
    std::printf("  write_state moved logits by L2 = %.4f\n", moved);
    assert(moved > 1e-3);

    // 5) CLEAR: with the write removed, the forward returns to baseline (the write is truly off).
    adp.clear_write();
    ForwardResult c = adp.forward(board, mask, nullptr, std::nullopt, logits_for);
    const double reverted = l2diff(base_logits, c.logits);
    std::printf("  after clear_write, L2 from baseline = %.6f\n", reverted);
    assert(reverted < 1e-3);

    std::printf("test_ggml_state_write: OK (real GGUF: read -> edit -> write -> observe -> clear)\n");
    return 0;
}
