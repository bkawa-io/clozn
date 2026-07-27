// confidence_select.cuh — Clozn confidence-select kernel (DESIGN.md §4.3)
// =============================================================================
//  GREEDY PATH COMPILED + VALIDATED on RTX 5080 / CUDA 13.3 (sm_120).
//  The deterministic surface — argmax pick, all three confidence variants
//  (max_prob / margin / neg_entropy), and both selection modes (top-k, threshold)
//  — is compiled and checked against reference.py by validate.py: token picks and
//  selected indices match EXACTLY; confidences within float32-vs-float64 epsilon.
//  STILL SCAFFOLD / UNVERIFIED: the sampled path (curand cannot bit-match numpy's
//  RNG) and the top_p nucleus filter (a TODO stub). reference.py is the oracle —
//  any divergence between this code and reference.py is a bug in THIS code.
// =============================================================================
//
// Contract (DESIGN.md §4.3), identical to reference.confidence_select:
//
//   inputs : logits [n_masked, vocab] (device-resident),
//            temperature, top_p, k_commit (or threshold tau), rng state
//   outputs: per masked position -> (sampled_token_id, confidence) ;
//            plus the indices of the selected positions (top-k_commit by
//            confidence, OR conf >= tau with a min_commit rail)
//   transfer to host: 2 * n_masked ints + floats   (~10,000x smaller than the
//            naive [n_masked, vocab] full-logits copy, which the upstream
//            llama.cpp diffusion PR measured at ~87% of GPU wall time)
//
// Confidence variants (mirror open-dCoder sample_tokens / reference._confidences):
//   MAX_PROB    : probability of the sampled token
//   MARGIN      : top1 - top2 probability
//   NEG_ENTROPY : sum_v p_v * log p_v   (already negative; higher = more peaked)
//
// Selection variants (mirror clozn_lab.scheduler.policies):
//   TOP_K     : commit the min(k_commit, n_masked) highest-confidence positions,
//               ties broken toward the LOWER position index.
//   THRESHOLD : commit positions with conf >= tau; if fewer than min_commit
//               clear tau, commit the top min_commit by confidence (the rail).

#ifndef CLOZN_CONFIDENCE_SELECT_CUH
#define CLOZN_CONFIDENCE_SELECT_CUH

#include <cstdint>

namespace clozn {

// Selectable confidence definition (DESIGN open question #3).
enum class ConfidenceKind : int {
    MaxProb = 0,
    Margin = 1,
    NegEntropy = 2,
};

// Which selection rule the host asked for. Exactly one of k_commit / tau is
// meaningful, picked by `mode`.
enum class SelectMode : int {
    TopK = 0,       // use params.k_commit
    Threshold = 1,  // use params.tau and params.min_commit
};

// All scalar knobs for one fused call. Mirrors reference.confidence_select's
// keyword arguments one-for-one.
struct ConfidenceSelectParams {
    int            n_masked;     // rows of the logits buffer
    int            vocab;        // columns of the logits buffer
    float          temperature;  // 0 => greedy (argmax); >0 => sample
    float          top_p;        // nucleus cutoff in (0,1]; >=1 or <=0 disables
    ConfidenceKind confidence;   // MAX_PROB | MARGIN | NEG_ENTROPY
    SelectMode     mode;         // TOP_K | THRESHOLD
    int            k_commit;     // used when mode == TopK
    float          tau;          // used when mode == Threshold
    int            min_commit;   // rail for the Threshold path (>= 1)
    uint64_t       rng_seed;     // seeds the per-row curand state when sampling
};

// Device output buffers (caller-allocated). After the call the host copies back
// only `out_token_ids` and `out_confidences` (the 2 * n_masked transfer) plus
// `out_selected` / `out_n_selected` — never the full logits.
struct ConfidenceSelectOutputs {
    int32_t* out_token_ids;    // [n_masked]   sampled token per position
    float*   out_confidences;  // [n_masked]   selected confidence variant
    int32_t* out_selected;     // [n_masked]   first out_n_selected entries valid,
                               //              ascending row indices to commit
    int32_t* out_n_selected;   // [1]          count of valid out_selected entries
};

// Host-callable entry point: launches the per-position sample/confidence kernel
// followed by the selection kernel(s) on `stream`. `logits` is device-resident
// [n_masked, vocab] row-major float. Returns once the launches are queued
// (async); the caller synchronizes before reading the outputs.
//
// SCAFFOLD: signature and behavior must match reference.confidence_select.
void confidence_select(
    const float*                   logits,   // device [n_masked, vocab]
    const ConfidenceSelectParams&  params,
    const ConfidenceSelectOutputs& outputs,
    void*                          stream);  // cudaStream_t (void* to keep the
                                             // header CUDA-runtime-agnostic)

// Bytes crossing GPU->host per step under the fused kernel (matches
// reference.host_transfer_bytes): an int token id + a float confidence per
// masked position.
inline int host_transfer_bytes(int n_masked) {
    constexpr int bytes_per_value = 4;  // int32 token id + float32 confidence
    return 2 * n_masked * bytes_per_value;
}

}  // namespace clozn

#endif  // CLOZN_CONFIDENCE_SELECT_CUH
