// clozn/sae.hpp — the on-device SAE feature readout (ROADMAP 3.3 wired into the engine).
// =============================================================================
// Runs the andyrdt-style JumpReLU SAE ENCODER on the GPU over tapped residuals and returns each
// row's top-k (feature id, activation) pairs — the sparse code, on-device, using the validated
// kernels/sae_topk kernel for the top-k middle step. Encoder-only by design: the decoder GEMM and
// the Neuronpedia feature-name mapping stay host/Python side (research/brain_readout.py); the
// engine emits raw feature indices ("sae:<id>") on the event stream.
//
// Where it sits (the seam this class fills, mirroring sae_topk.cuh's diagram):
//
//   ForwardResult.activations [rows x d_in]   (host fp32 — the existing white-box tap)
//       --( upload; xh = fp16(x) - b_dec )-->
//   xh [rows x d_in] fp16
//       --( encoder GEMV: hp = fp16(xh · W_enc_t[f]) + b_enc[f]; JumpReLU: hp > threshold[f] )-->
//   gated pre-acts [rows x d_sae] fp32        (device; dense)
//       --( kernels/sae_topk: per-row top-k over the feature dim )-->
//   sparse code [rows x k] ids + values       (D2H: the only bytes that cross back)
//
// Numerics contract: matches research/sae7b.py GpuSAE.encode (the oracle, itself verified
// bit-exact vs sae_lens): torch computes the encoder matmul in fp16 (fp32 accumulate, fp16
// result), adds b_enc in fp16, widens to fp32, then gates relu(hp) * (hp > threshold). The CUDA
// path accumulates in fp32 and reproduces the fp16 roundings at the same places, so values agree
// to fp16 epsilon (see tests/test_sae_encoder.cpp for the measured tolerances).
//
// Weights come from engine/core/tools/export_sae_weights.py: a directory of raw blobs + meta.txt
// (shapes + L2-norm receipts; load() recomputes the norms and refuses corrupt weights). W_enc is
// stored TRANSPOSED [d_sae, d_in] fp16 so each feature's weights are one contiguous GEMV row.
//
// VRAM: the encoder for 131072 features x 3584 dims is ~0.94 GB fp16 (+ <2 MB of biases/thresholds
// + a grow-only per-call workspace of ~0.6 MB/row). device_bytes() reports the honest total.
//
// The header is CUDA-free (pimpl) so clozn_server.cpp compiles against it untouched by nvcc; the
// implementation (src/sae_encoder.cu) is CUDA and links the vendored clozn_sae_topk kernel.
// Thread-safe: encode calls are serialized on an internal mutex (readout cadence, not a hot loop).
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace clozn {

class SaeEncoder {
public:
    SaeEncoder();
    ~SaeEncoder();
    SaeEncoder(const SaeEncoder&) = delete;
    SaeEncoder& operator=(const SaeEncoder&) = delete;

    // Load an exported weight directory (meta.txt + blobs) onto the CUDA device. Verifies shapes
    // and the meta.txt L2-norm receipts (rel 1e-3) so a truncated/corrupt blob never serves.
    // Returns false and sets error() on any failure; ready() stays false.
    bool load(const std::string& dir);

    bool ready() const;
    const std::string& error() const;  // last load/encode failure, human-readable

    int d_in() const;    // encoder input width (the tapped model's n_embd must equal this)
    int d_sae() const;   // dictionary size (feature count)
    int layer() const;   // the residual layer the SAE was trained on (the tap layer to serve it at)
    size_t device_bytes() const;  // VRAM held by weights + workspace (the honest budget number)

    // Encode `rows` host fp32 activation rows ([rows * d_in], position-major — exactly
    // ForwardResult.activations) and return each row's top-k features. out_indices/out_values are
    // resized to [rows * k], row-major; indices ASCENDING per row, values aligned (a slot past the
    // row's live-feature count carries value 0.0 — the sae_topk pad contract). Returns false (and
    // sets error()) on CUDA failure or if !ready().
    bool encode_topk(const float* x, int rows, int k,
                     std::vector<int32_t>& out_indices, std::vector<float>& out_values);

    // The dense gated pre-activations ([rows * d_sae] fp32) — the full JumpReLU code before the
    // top-k. Heavy (0.5 MB/row D2H); exists for the parity test, which diffs the ENTIRE feature
    // space against the torch reference rather than only the selected k.
    bool encode_dense(const float* x, int rows, std::vector<float>& gated);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace clozn
