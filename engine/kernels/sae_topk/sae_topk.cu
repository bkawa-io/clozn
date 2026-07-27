// sae_topk.cu — Clozn SAE-sparsify top-k kernel (ROADMAP 3.3).
// =============================================================================
//  COMPILED + VALIDATED on RTX 5080 / CUDA 13.3 (sm_120). Per-row top-k over the
//  FEATURE dimension — the SAE sparse code. Selected indices match
//  reference.sae_topk EXACTLY; values within float32-vs-float64 epsilon
//  (validate.py). reference.py is the oracle; wherever this disagrees with it,
//  this file is the bug.
//
//  Structure mirrors ../confidence_select/confidence_select.cu: one thread-block
//  per row, a block-parallel argmax reduction (max value; ties -> LOWER index,
//  matching numpy argmax / the reference's stable argsort). The confidence kernel
//  takes ONE argmax (the greedy pick) over `vocab`; this takes k_eff argmaxes over
//  `n_features`, masking each winner out before the next round — an iterative
//  top-k. k is small (~16-128) so k rounds of an O(n_features) reduction is the
//  right shape; swap for a block radix-select if k ever grows large.
// =============================================================================

#include "sae_topk.cuh"

#include <cfloat>
#include <cstdint>

#if defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

namespace clozn {

#if defined(__CUDACC__)

namespace {

constexpr int kThreadsPerRow = 256;  // one block per row, as confidence_select

// Block-parallel argmax over the row's RANKED values (max value; ties -> the
// LOWER feature index). Identical reduction to confidence_select::block_argmax,
// but ranking the value an already-picked feature is masked to -FLT_MAX so it
// can't win again. Uses two shared arrays of blockDim.x: values + indices.
// `relu` ranks over max(v,0) so a strongly-negative feature never outranks a
// positive one (the reference ranks the same quantity).
__device__ inline int block_argmax_masked(const float* row, int n_features,
                                           bool relu, const char* picked,
                                           float* sval, int* sidx) {
    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int bidx = 0;
    for (int v = tid; v < n_features; v += blockDim.x) {
        if (picked[v]) continue;  // already selected in a previous round
        float rv = row[v];
        if (relu && rv < 0.0f) rv = 0.0f;  // rank over the ReLU'd value
        // strictly-greater keeps the FIRST (lowest v) on per-thread ties
        if (rv > best) { best = rv; bidx = v; }
    }
    sval[tid] = best;
    sidx[tid] = bidx;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float ov = sval[tid + stride];
            const int oi = sidx[tid + stride];
            if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) {
                sval[tid] = ov;
                sidx[tid] = oi;
            }
        }
        __syncthreads();
    }
    return sidx[0];
}

// One block per row. Selects the row's top-k_eff features by iterated masked
// argmax (tie -> lower index), then writes them ASCENDING with aligned values.
//
// Shared layout: [blockDim.x] floats (sval) ++ [blockDim.x] ints (sidx) ++
// [k] ints (the winners, in selection order, before the ascending sort).
__global__ void sae_topk_kernel(
    const float* __restrict__ pre_acts,   // [rows, n_features]
    int                       n_features,
    int                       k,
    bool                      relu,
    int32_t* __restrict__     out_indices, // [rows, k]
    float* __restrict__       out_values,  // [rows, k]
    char* __restrict__        picked_scratch) {  // [rows, n_features] mask scratch
    extern __shared__ float smem[];
    float* sval = smem;                                       // [blockDim.x]
    int*   sidx = reinterpret_cast<int*>(smem + blockDim.x);  // [blockDim.x]
    int*   winners = sidx + blockDim.x;                       // [k]

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_acts = pre_acts + static_cast<size_t>(row) * n_features;
    char* picked = picked_scratch + static_cast<size_t>(row) * n_features;

    const int k_eff = k < n_features ? k : n_features;

    // Clear this row's picked mask cooperatively.
    for (int v = tid; v < n_features; v += blockDim.x) picked[v] = 0;
    __syncthreads();

    // k_eff rounds of masked argmax. Thread 0 records each winner and marks it
    // picked; a barrier publishes the mask before the next round reads it.
    for (int j = 0; j < k_eff; ++j) {
        const int w = block_argmax_masked(row_acts, n_features, relu, picked, sval, sidx);
        if (tid == 0) {
            winners[j] = w;
            picked[w] = 1;
        }
        __syncthreads();
    }

    // Emit ascending (mirrors reference._select_topk's sorted(...)). k is small,
    // so thread 0 does an insertion-style ordered scatter: for each output slot
    // pick the smallest not-yet-emitted winner. O(k^2), k ~ 16-128.
    if (tid == 0) {
        // Mark emitted within `winners` by setting to -1 as we consume them.
        for (int slot = 0; slot < k_eff; ++slot) {
            int best_pos = -1;
            int best_idx = 0x7fffffff;
            for (int j = 0; j < k_eff; ++j) {
                if (winners[j] >= 0 && winners[j] < best_idx) {
                    best_idx = winners[j];
                    best_pos = j;
                }
            }
            const int feat = winners[best_pos];
            winners[best_pos] = -1;  // consumed

            float val = row_acts[feat];
            if (relu && val < 0.0f) val = 0.0f;  // gate non-positive picks to 0.0
            out_indices[static_cast<size_t>(row) * k + slot] = feat;
            out_values[static_cast<size_t>(row) * k + slot] = val;
        }
        // Pad columns (only when k > n_features): repeat the last index, 0 value.
        for (int slot = k_eff; slot < k; ++slot) {
            const int pad = k_eff > 0 ? out_indices[static_cast<size_t>(row) * k + k_eff - 1] : 0;
            out_indices[static_cast<size_t>(row) * k + slot] = pad;
            out_values[static_cast<size_t>(row) * k + slot] = 0.0f;
        }
    }
}

}  // namespace

void sae_topk(
    const float*           pre_acts,
    const SaeTopKParams&   params,
    const SaeTopKOutputs&  outputs,
    void*                  stream,
    char*                  picked_scratch) {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    const int threads = kThreadsPerRow;
    const int k_eff = params.k < params.n_features ? params.k : params.n_features;
    // Shared: two reduction arrays of `threads` + the k_eff winners buffer.
    const size_t shmem =
        static_cast<size_t>(threads) * (sizeof(float) + sizeof(int)) +
        static_cast<size_t>(k_eff > 0 ? k_eff : 1) * sizeof(int);

    // Per-row "already picked" mask scratch. If the caller didn't bring its own
    // persistent buffer (picked_scratch == nullptr -- one-off callers: validate.cu,
    // the parity test), fall back to the original per-call alloc/free so those
    // callers are untouched. A caller with a workspace (clozn/sae.hpp) passes its
    // own buffer and this path skips the cudaMalloc + forced sync + cudaFree
    // entirely -- exactly the hoist the caller-owned workspace calls for.
    char* d_picked = picked_scratch;
    const bool owns_scratch = (d_picked == nullptr);
    if (owns_scratch)
        cudaMalloc(&d_picked, static_cast<size_t>(params.rows) * params.n_features * sizeof(char));

    sae_topk_kernel<<<params.rows, threads, shmem, cu_stream>>>(
        pre_acts, params.n_features, params.k, params.relu,
        outputs.out_indices, outputs.out_values, d_picked);

    // Only the self-allocated path needs an explicit sync here (it must outlive
    // the kernel before cudaFree). A caller-owned buffer needs no sync: the
    // kernel launch is stream-ordered, so any later work the caller issues on
    // the SAME stream (e.g. sae_encoder.cu's blocking D2H cudaMemcpy) already
    // waits for it -- this call can return without blocking the host.
    if (owns_scratch) {
        cudaStreamSynchronize(cu_stream);
        cudaFree(d_picked);
    }
}

#else  // !__CUDACC__

// Host-only translation unit (no nvcc): a linkable stub so a CUDA-less build
// configures. reference.py is the correctness oracle until a toolchain exists.
void sae_topk(
    const float*,
    const SaeTopKParams&,
    const SaeTopKOutputs&,
    void*,
    char*) {
    // Intentionally empty: the CUDA path is unavailable on a host-only build.
}

#endif  // __CUDACC__

}  // namespace clozn
