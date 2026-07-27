// confidence_select.cu — Clozn confidence-select kernel (DESIGN.md §4.3)
// =============================================================================
//  GREEDY PATH COMPILED + VALIDATED on RTX 5080 / CUDA 13.3 (sm_120).
//  The deterministic surface (argmax pick; max_prob/margin/neg_entropy confidence;
//  top-k and threshold selection) compiles and matches reference.py exactly via
//  validate.py (picks + selected indices exact; confidences within f32-vs-f64 eps).
//  STILL SCAFFOLD / UNVERIFIED: the sampled path (curand cannot bit-match numpy's
//  RNG) and the top_p nucleus filter (the TODO stub below). reference.py is the
//  oracle; wherever this file disagrees with it, this file is the bug.
// =============================================================================
//
// Design sketch (matches the reference's data flow):
//
//   Stage 1 — per-position sample + confidence (one thread-block per row):
//     A thread-block cooperatively reduces over the `vocab` logits of one
//     masked position:
//       1. (optional) divide logits by temperature.
//       2. (optional) top_p nucleus filter: a block-wide sort + cumulative
//          softmax, dropping tokens whose predecessors already reached top_p
//          (the kept-set always includes the top-1 and the token that first
//          crosses top_p — mirrors open-dCoder top_p_logits / reference._top_p_filter).
//       3. softmax (block reduction for max then sum).
//       4. pick: argmax (greedy) or curand categorical draw (sampled).
//       5. confidence: MaxProb gathers the pick prob; Margin reduces top1/top2;
//          NegEntropy reduces sum p*log p with a 1e-10 clamp.
//     Writes out_token_ids[row] and out_confidences[row].
//
//   Stage 2 — selection over the per-row confidences (small array, n_masked):
//     TopK     : argsort-by-(-conf, row) and take the first k_commit, ties
//                toward the lower row index; emit ascending.
//     Threshold: gather rows with conf >= tau; if fewer than min_commit, fall
//                back to the top min_commit by confidence (the rail).
//     n_masked is tiny relative to vocab, so a single-block selection (or even
//     a host-side selection over the already-transferred confidences) is fine;
//     a device top-k keeps the host transfer at the §4.3 minimum.
//
// This file is intentionally light on micro-optimization: it is a correctness
// scaffold to be profiled and tuned once it compiles.

#include "confidence_select.cuh"

#include <cfloat>
#include <cstdint>

// NOTE: <curand_kernel.h> and <cuda_runtime.h> are included only when compiled
// with nvcc. Guarded so the file at least parses under a host-only C++ lint.
#if defined(__CUDACC__)
#include <cuda_runtime.h>
#include <curand_kernel.h>
#endif

namespace clozn {

#if defined(__CUDACC__)

namespace {

constexpr int kThreadsPerRow = 256;  // tune once profilable

// Block-wide reduction helpers (max / sum) over per-thread partials in shared
// memory. Standard tree reduction; replace with cub::BlockReduce when wiring up
// the real build for speed.
__device__ inline float block_reduce_max(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] = fmaxf(shared[tid], shared[tid + stride]);
        __syncthreads();
    }
    return shared[0];
}

__device__ inline float block_reduce_sum(float val, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    return shared[0];
}

// Block-parallel argmax over the row (max temperature-scaled logit; ties -> the
// LOWER index, matching numpy argmax / the reference). Uses two shared arrays of
// blockDim.x: values (sval) and indices (sidx). Returns the winning index.
__device__ inline int block_argmax(const float* row, int vocab, float inv_temp,
                                    float* sval, int* sidx) {
    const int tid = threadIdx.x;
    float best = -FLT_MAX;
    int bidx = 0;
    for (int v = tid; v < vocab; v += blockDim.x) {
        float lv = row[v] * inv_temp;
        if (lv > best) { best = lv; bidx = v; }  // first (lowest v) wins ties per thread
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

// Stage 1: one block per masked position. Computes the softmax over this row's
// (optionally temperature-scaled, top_p-filtered) logits, samples a token, and
// writes the requested confidence variant.
//
// SCAFFOLD: the top_p path below is the simplified "filter by absolute prob"
// shortcut and is NOT yet the exact sorted-cumulative rule the reference uses;
// it must be replaced with a block sort + cumulative-mass mask before this is
// considered correct. Marked clearly so it is not mistaken for verified.
__global__ void sample_and_confidence_kernel(
    const float* __restrict__ logits,        // [n_masked, vocab]
    int                       vocab,
    float                     temperature,
    float                     top_p,
    ConfidenceKind            confidence,
    uint64_t                  rng_seed,
    int32_t* __restrict__     out_token_ids,
    float* __restrict__       out_confidences) {
    extern __shared__ float smem[];  // [blockDim.x] floats, then [blockDim.x] ints
    float* sval = smem;                                     // reused by the block reductions
    int* sidx = reinterpret_cast<int*>(smem + blockDim.x);  // index scratch for argmax

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* row_logits = logits + static_cast<size_t>(row) * vocab;

    const bool greedy = (temperature <= 0.0f);
    const float inv_temp = greedy ? 1.0f : (1.0f / temperature);

    // --- row max (for numerically stable softmax) ---
    float local_max = -FLT_MAX;
    for (int v = tid; v < vocab; v += blockDim.x) {
        local_max = fmaxf(local_max, row_logits[v] * inv_temp);
    }
    float row_max = block_reduce_max(local_max, smem);

    // --- exp + sum ---
    float local_sum = 0.0f;
    for (int v = tid; v < vocab; v += blockDim.x) {
        local_sum += expf(row_logits[v] * inv_temp - row_max);
    }
    float row_sum = block_reduce_sum(local_sum, smem);

    // TODO(pending-toolchain): apply the exact top_p nucleus filter here
    // (block sort + cumulative softmax mask) so support matches the reference
    // before softmax normalization. Omitted in this scaffold.
    (void)top_p;

    const float inv_sum = 1.0f / row_sum;

    // --- pick token (thread 0 writes out_token_ids[row]) ---
    if (greedy) {
        // Block-parallel argmax (max logit; ties -> lower index).
        const int gidx = block_argmax(row_logits, vocab, inv_temp, sval, sidx);
        if (tid == 0) out_token_ids[row] = gidx;
    } else {
        // Sampled: one curand draw from the categorical, done by thread 0. NOTE:
        // curand cannot reproduce numpy's rng.choice sequence, so this path is NOT
        // bit-parity with the reference and is left unvalidated (see the banner).
        if (tid == 0) {
            curandStatePhilox4_32_10_t state;
            curand_init(rng_seed, static_cast<uint64_t>(row), 0, &state);
            float u = curand_uniform(&state);  // (0,1]
            float cdf = 0.0f;
            int chosen = vocab - 1;
            for (int v = 0; v < vocab; ++v) {
                cdf += expf(row_logits[v] * inv_temp - row_max) * inv_sum;
                if (u <= cdf) { chosen = v; break; }
            }
            out_token_ids[row] = chosen;
        }
    }
    __syncthreads();

    // --- confidence variant (thread 0 emits) ---
    if (tid == 0) {
        int chosen = out_token_ids[row];
        if (confidence == ConfidenceKind::MaxProb) {
            out_confidences[row] =
                expf(row_logits[chosen] * inv_temp - row_max) * inv_sum;
        } else if (confidence == ConfidenceKind::Margin) {
            float top1 = -FLT_MAX, top2 = -FLT_MAX;
            for (int v = 0; v < vocab; ++v) {
                float p = expf(row_logits[v] * inv_temp - row_max) * inv_sum;
                if (p > top1) { top2 = top1; top1 = p; }
                else if (p > top2) { top2 = p; }
            }
            out_confidences[row] = top1 - top2;
        } else {  // NegEntropy: sum p * log p, clamp p at 1e-10
            float acc = 0.0f;
            for (int v = 0; v < vocab; ++v) {
                float p = expf(row_logits[v] * inv_temp - row_max) * inv_sum;
                float pc = p < 1e-10f ? 1e-10f : p;
                acc += p * logf(pc);
            }
            out_confidences[row] = acc;
        }
    }
}

// Stage 2: selection over the per-row confidences. n_masked is small, so a
// single block does the whole thing. SCAFFOLD: a straightforward O(n * k) /
// O(n) pass; swap for a device radix/top-k if n_masked ever grows large.
__global__ void select_kernel(
    const float* __restrict__ confidences,
    int                       n_masked,
    SelectMode                mode,
    int                       k_commit,
    float                     tau,
    int                       min_commit,
    int32_t* __restrict__     out_selected,
    int32_t* __restrict__     out_n_selected) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;  // single-thread scaffold

    auto emit_top = [&](int count) {
        // Select `count` highest-confidence rows, ties toward the lower index,
        // and write them ascending. O(count * n) selection (count is small).
        int want = count < n_masked ? count : n_masked;
        // chosen[] tracks already-picked rows.
        // (Bounded by n_masked; a real impl uses a heap or partial radix sort.)
        bool picked[1024] = {false};  // SCAFFOLD cap; size properly when wired.
        int n_out = 0;
        for (int r = 0; r < want; ++r) {
            float best = -FLT_MAX;
            int best_idx = -1;
            for (int i = 0; i < n_masked; ++i) {
                if (picked[i]) continue;
                // strictly-greater keeps the FIRST (lower) index on ties.
                if (confidences[i] > best) { best = confidences[i]; best_idx = i; }
            }
            picked[best_idx] = true;
            ++n_out;
        }
        // Emit picked rows in ascending index order.
        int w = 0;
        for (int i = 0; i < n_masked; ++i) {
            if (picked[i]) out_selected[w++] = i;
        }
        *out_n_selected = w;
    };

    if (mode == SelectMode::TopK) {
        emit_top(k_commit);
        return;
    }

    // Threshold: commit conf >= tau; if too few, fall back to top min_commit.
    int count_above = 0;
    for (int i = 0; i < n_masked; ++i) {
        if (confidences[i] >= tau) ++count_above;
    }
    if (count_above >= min_commit) {
        int w = 0;
        for (int i = 0; i < n_masked; ++i) {
            if (confidences[i] >= tau) out_selected[w++] = i;  // already ascending
        }
        *out_n_selected = w;
    } else {
        emit_top(min_commit);  // the min-one-commit progress rail
    }
}

}  // namespace

void confidence_select(
    const float*                   logits,
    const ConfidenceSelectParams&  params,
    const ConfidenceSelectOutputs& outputs,
    void*                          stream) {
    cudaStream_t cu_stream = static_cast<cudaStream_t>(stream);

    // Stage 1: one block per masked position, kThreadsPerRow threads, shared
    // memory sized for the block reduction.
    const int threads = kThreadsPerRow;
    const size_t shmem = static_cast<size_t>(threads) * (sizeof(float) + sizeof(int));
    sample_and_confidence_kernel<<<params.n_masked, threads, shmem, cu_stream>>>(
        logits, params.vocab, params.temperature, params.top_p, params.confidence,
        params.rng_seed, outputs.out_token_ids, outputs.out_confidences);

    // Stage 2: single-block selection over the n_masked confidences.
    select_kernel<<<1, 1, 0, cu_stream>>>(
        outputs.out_confidences, params.n_masked, params.mode, params.k_commit,
        params.tau, params.min_commit, outputs.out_selected, outputs.out_n_selected);
}

#else  // !__CUDACC__

// Host-only translation unit (no nvcc): provide a linkable stub so CMake's
// optional C++ target builds and can print the "pending toolchain" banner.
void confidence_select(
    const float*,
    const ConfidenceSelectParams&,
    const ConfidenceSelectOutputs&,
    void*) {
    // Intentionally empty: the CUDA path is unverified and unavailable here.
    // reference.py is the correctness oracle until a toolchain exists.
}

#endif  // __CUDACC__

}  // namespace clozn
