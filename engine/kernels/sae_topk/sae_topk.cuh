// sae_topk.cuh — Clozn SAE-sparsify top-k kernel (ROADMAP 3.3).
// =============================================================================
//  The on-device primitive for SAE / transcoder inference at scale: a sparse
//  top-k over the FEATURE dimension. Given an SAE pre-activation matrix
//  [rows x n_features] (rows = token positions), keep for each row the top-k
//  features (indices + values), zeroing the rest — the SAE's sparse code.
//
//  This is the confidence-select kernel's top-k (../confidence_select) REPOINTED
//  at the feature dimension (ARCHITECTURE.md). Same block-per-row structure, same
//  dtype conventions (device float32 in; int32 indices + float32 values out),
//  same tie rule (toward the LOWER index), same "exact picks / epsilon values"
//  parity contract. reference.py is the oracle — any divergence is a bug here.
//
//  COMPILED + VALIDATED on RTX 5080 / CUDA 13.3 (sm_120): the kernel's selected
//  indices match reference.sae_topk EXACTLY; values within float32-vs-float64
//  epsilon (kernel reduces in float32, the reference in float64). See validate.py.
// =============================================================================
//
// Where it sits in SAE inference (the seam this op fills):
//
//   activations [rows x d_model]
//       --( encoder GEMM: W_enc^T·acts + b_enc )-->
//   pre_acts    [rows x n_features]              (dense; this kernel's input)
//       --( THIS kernel: per-row top-k over the feature dim )-->
//   sparse code [rows x k]  (out_indices + out_values)
//       --( decoder GEMM: sum_j val_j · W_dec[:, idx_j] + b_dec )-->
//   reconstruction [rows x d_model]
//
// The encoder/decoder matmuls are dense cuBLAS GEMMs; this is the sparse middle
// step. n_features is large (4k-130k) and k small (~16-128) — exactly the regime
// the confidence-select block-per-row reduction targets, over features not vocab.
//
// Contract (identical to reference.sae_topk):
//   inputs : pre_acts [rows, n_features] device-resident row-major float32; k.
//   outputs: per row -> the k selected feature indices (ASCENDING) and their
//            values, aligned. Others are implicitly zero (the sparse code).
//   tie rule: equal values resolve toward the LOWER feature index.
//   relu (default on): rank over max(value, 0); a selected feature whose
//            pre-activation is <= 0 emits a 0.0 value (a "dead" feature adds
//            nothing to the reconstruction). Fixed [rows, k] output either way.

#ifndef CLOZN_SAE_TOPK_CUH
#define CLOZN_SAE_TOPK_CUH

#include <cstdint>

namespace clozn {

// All scalar knobs for one fused call. Mirrors reference.sae_topk's arguments.
struct SaeTopKParams {
    int  rows;        // token positions (one thread-block each)
    int  n_features;  // dictionary size (columns of pre_acts)
    int  k;           // features to keep per row; k_eff = min(k, n_features)
    bool relu;        // rank over max(v,0) and gate non-positive picks to 0.0
};

// Device output buffers (caller-allocated), each [rows * k] row-major. After the
// call the host copies back only these (rows*k ints + rows*k floats) — the sparse
// code, ~ n_features/k smaller than the dense [rows, n_features] pre-activations.
struct SaeTopKOutputs {
    int32_t* out_indices;  // [rows * k]  selected feature indices, ascending per row
    float*   out_values;   // [rows * k]  value at each selected index (0.0 if gated)
};

// Host-callable entry point: launches one block per row on `stream`. `pre_acts`
// is device-resident [rows, n_features] row-major float32. Async; the caller
// synchronizes before reading the outputs. Signature/behavior match
// reference.sae_topk.
//
// `picked_scratch`: device buffer of >= rows*n_features bytes for the per-row
// "already selected" mask (see sae_topk.cu). Pass nullptr (the default) to have
// the call cudaMalloc/cudaFree its own scratch, as before -- fine for one-off
// callers (validate.cu, the parity test). A caller with a persistent workspace
// (clozn/sae.hpp's grow-only reserve()) should allocate picked_scratch itself
// ONCE, sized to its own [rows, n_features] ceiling, and pass it here every call
// to avoid paying a cudaMalloc + forced cudaStreamSynchronize + cudaFree on
// every readout -- exactly the fix this parameter exists for.
void sae_topk(
    const float*           pre_acts,  // device [rows, n_features]
    const SaeTopKParams&   params,
    const SaeTopKOutputs&  outputs,
    void*                  stream,    // cudaStream_t (void* keeps the header
                                      // CUDA-runtime-agnostic, as confidence_select.cuh)
    char*                  picked_scratch = nullptr);

// Bytes crossing GPU->host for the sparse code vs the dense pre-activations: the
// kernel ships rows*k (int index + float value) instead of rows*n_features
// floats — the sparsity win that makes large dictionaries tractable on the bus.
inline int sparse_code_bytes(int rows, int k) {
    constexpr int bytes_per_value = 4;  // int32 index + float32 value
    return 2 * rows * k * bytes_per_value;
}

}  // namespace clozn

#endif  // CLOZN_SAE_TOPK_CUH
