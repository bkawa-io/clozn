// clozn/model_ggml.hpp — the L0 ggml/llama.cpp ModelAdapter. This is the ONE place a model
// backend lives on the C++ side (DESIGN invariant 1, the C++ analogue of torch living
// only under lab/clozn_lab/models/). The scheduler/runtime never include this header.
//
// Diffusion forward (slice 1): load a GGUF, decode the board bidirectionally
// (llama_set_causal_attn false), apply the Dream-family shift — logits for board position p
// are read from output row p-1 (the adapter owns the shift).
//
// KV reuse under the one-way law (slice 2): llama.cpp's high-level API exposes only causal
// on/off, never an arbitrary attention mask — so the block-causal one-way law cannot be a
// mask. Instead we get it from DECODE ORDER. Each segment (prompt, then each block) is
// decoded while later positions don't yet exist, so its frozen K/V never attends forward —
// exactly Tier A/B exactness. The active block is re-decoded every step (Tier C = full
// active recompute, exact). The shifted head's source row for a block's first slot
// (position active_start-1, which is frozen and not re-emitted) is captured as a "boundary
// row" when that prefix block is frozen. This reproduces the lab's block-causal goldens
// without a custom ggml mask. The active-block start is read from the mask:
// active_start = min{q : mask(q, n-1)} (0 when whole-sequence / fully bidirectional).
#pragma once

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "clozn/model.hpp"
#include "llama.h"

namespace clozn {

// The opaque KV handle threaded by the scheduler (invariant 4). The llama context holds one
// stateful KV cache, so this is just a marker of how many board positions it now covers;
// the adapter keeps the real bookkeeping (frozen boundary + boundary row) internally. A
// non-null kv passed back into forward() is the "reuse" signal (vs null = cold start).
struct GgmlKV : KVState {
    int n;
    explicit GgmlKV(int n_) : n(n_) {}
    int seq_len() const override { return n; }
};

// The loaded model weights + vocab + ModelConfig, shareable across contexts. llama_model is
// read-only after load, so N contexts can share ONE GgmlModel — essential for serving
// concurrency: the weights (the bulk of VRAM) are loaded once; only the per-context KV/compute
// buffers replicate. Hold it by shared_ptr and hand it to GgmlAdapter(model, n_ctx, ...).
class GgmlModel {
public:
    // mask_token_id is checkpoint-specific (open-dCoder <M> = 151665). eos_token_id < 0 => take
    // the model's own EOS from the vocab. n_gpu_layers > 0 offloads weights to the GPU.
    GgmlModel(const std::string& model_path, int mask_token_id,
              int eos_token_id = -1, int n_gpu_layers = 0);
    ~GgmlModel();
    GgmlModel(const GgmlModel&) = delete;
    GgmlModel& operator=(const GgmlModel&) = delete;

    const ModelConfig& config() const { return cfg_; }
    llama_model* handle() const { return model_; }
    const llama_vocab* vocab() const { return vocab_; }
    // Diffusion head convention (GGUF `diffusion.shift_logits`): true = Dream-family SHIFTED head
    // (logits for position p come from row p-1); false = LLaDA-family IN-PLACE head (row p). Defaults
    // true when the key is absent (e.g. a Dream GGUF converted as Qwen2) — matches llama's diffusion-cli.
    bool shift_logits() const { return shift_logits_; }

    // Tokenization lives on the model (vocab-only, no context) so it's usable without — and
    // concurrently with — any context (a server tokenizes here, then acquires a context to run).
    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

private:
    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    ModelConfig cfg_{};
    bool shift_logits_ = true;  // see shift_logits(): default to the Dream-family shifted head
};

// Per-layer activation summary from ONE forward: the L2 norm of every token's residual at every layer
// (the "Model MRI" depth x position map). Captured by the eval callback prefix-matching "l_out-<il>"
// across all layers in a single decode (cf. llama.cpp's cvector-generator) -- one forward, not n_layer.
struct LayerSummary {
    int n_layer = 0;
    int n_tokens = 0;
    std::vector<std::vector<float>> norms;   // [n_layer][n_tokens]: |residual| per token per layer
};

// Serializable generation state: the KV cache for sequence 0 + enough metadata to resume
// bit-exactly. checkpoint() captures it; restore() reinstates it. The KV blob is opaque
// (llama_state_seq_get/set_data); the token list + n_past let the caller resume generate_ar
// from the right position. Greedy suffix after save->restore is the correctness bar.
struct EngineCheckpoint {
    std::vector<uint8_t> kv_data;   // serialized KV cache (seq 0) via llama_state_seq_get_data
    std::vector<int> tokens;         // full token sequence fed into the KV (prompt + generated so far)
    int n_past = 0;                  // positions covered by the KV cache
    // The prompt/generated boundary: tokens[0, prompt_tokens) is the ORIGINAL prefill batch,
    // tokens[prompt_tokens, n_past) is the one-token-at-a-time generated tail. Populated by the
    // CALLER at checkpoint-creation time (save_checkpoint itself doesn't know the boundary), same
    // as has_sampler/has_steer below. 0 = unknown (a checkpoint whose caller never declared it) --
    // execution-fork treats 0 as "cannot determine the regime split" and refuses rather than guess
    // which side of the batch-shape landmine (docs: the reprefill-vs-live-KV split) truncate_to
    // falls on.
    int prompt_tokens = 0;
    bool causal = true;              // attention mode at checkpoint time
    // Declared sampler provenance (engine debt: sampler state in checkpoints). The checkpoint
    // flow is caller-reconstructive (POST tokens -> prefill -> save), so there is no live RNG to
    // capture; instead the CALLER declares the sampling config + how many draws the original
    // generation had consumed (== its sampled committed tokens), and restore fast-forwards a
    // fresh RNG by that many draws (mt19937_64::discard) -- making a SAMPLED resume bit-exact
    // against the uninterrupted run, not just statistically equivalent. has_sampler=false means
    // none was declared; restore then requires explicit sampling in the request (never guesses).
    bool has_sampler = false;
    double temperature = 0.0;
    double rep_penalty = 1.0;
    int top_k = 0;
    double top_p = 1.0;
    uint64_t seed = 0;
    uint64_t rng_draws = 0;
    // Steering provenance (the intervention half of the sampler-state debt item): the BUILT
    // control vector actually applied during the checkpointed run (n_embd*n_layer cvec + the
    // [lo, hi] band), not a declared direction -- storing what was applied covers raw-direction
    // AND named-concept steering alike, and restore re-applies it verbatim so the resumed
    // suffix continues the same intervention. Load-bearing on the reprefill path too: a steered
    // run's KV embeds the steer, so any rebuild must run under the same cvec.
    bool has_steer = false;
    std::vector<float> steer_cvec;  // the applied cvec, n_embd * n_layer
    int steer_lo = 0;
    int steer_hi = 0;
};

// One decode's multi-layer residual snapshot (Phase 2.3 readout plane). Produced by ar_forward
// when the capture set is armed; consumed by the serve-side ReadoutPlane's worker thread.
struct CaptureFrame {
    int from = 0;    // first board position of the decoded segment
    int rows = 0;    // segment length (1 per token during AR generation; prompt length on prefill)
    int n_embd = 0;
    std::vector<std::pair<int, std::vector<float>>> layers;  // (layer, [rows*n_embd] residuals, row-major)
};

class GgmlAdapter : public ModelAdapter {
public:
    // --- Attention knockout (Phase 2.4b) -----------------------------------------------------
    // Zero individual attention weights A[head, query, key] at a layer, i.e. STOP one position
    // from reading another. This is the primitive residual-site path patching could not provide:
    // §5f measured 0.0% routed at every depth for cross-position edges because patching a
    // destination site leaves the SOURCE free to re-supply the information downstream, and the
    // last layer is unpatchable (inp_out_ids). Cutting the edge itself sidesteps both.
    //
    // Requires flash attention OFF (`--no-flash-attn`): with FA the softmax is fused inside the
    // kernel and `kq_soft_max-<il>` never materializes. knockout_available() reports this, so a
    // caller gets a clean refusal instead of a silently-ignored intervention.
    struct AttnKnockout {
        int layer = 0;
        int head = -1;                 // -1 = every head at this layer
        std::vector<int> queries;      // reading positions
        std::vector<int> keys;         // positions being read (zeroed for each query)
        bool renormalize = false;      // rescale the surviving row to sum 1 (else mass is dropped)
    };
    void set_attn_knockouts(const std::vector<AttnKnockout>& ks);
    void clear_attn_knockouts();
    bool knockout_available() const { return !flash_attn_; }
    int n_head() const { return n_head_; }

    // Attention-row CAPTURE (R1 head-to-head): read (not modify) one query position's post-softmax
    // attention row at every layer, averaged over heads -- the correlational signal the causal
    // knockout ranking is compared against. Same materialization constraint as knockout (needs
    // --no-flash-attn); same eval_cb interception point, read-only. Armed per-forward: set the
    // query board position, run one /score, collect rows, cleared by the caller. The head-mean is
    // the standard "attention heatmap" a viewer would see; per-head capture is deliberately NOT
    // exposed until a product question needs it (32x the payload for no current consumer).
    void set_attn_capture(int query_pos);           // board position whose row to capture
    void clear_attn_capture();
    // per-layer head-averaged rows, [n_layer][n_kv_at_capture]; empty vector at layers not seen
    const std::map<int, std::vector<float>>& attn_rows() const { return attn_rows_; }

    // --- Head-output units (R5: head-level node units; notes/HEAD_UNITS_DESIGN.md) ------------
    // The FOURTH eval_cb hook: "kqv_out-<il>" is the merged attention output BEFORE the W_o
    // projection (verified against llama-graph.cpp: cb(cur,"kqv_out",il) precedes the wo
    // matmul), so per-Q-head h occupies the contiguous rows [h*d_head, (h+1)*d_head) of ne0.
    // Materialized at every position (a mid-graph tensor like l_out -- no inp_out_ids blocker)
    // and exists with flash attention ON (unlike kq_soft_max). GQA note: these are Q heads
    // (kqv_out rows are per-Q-head even when KV heads are fewer); every label says per-Q-head.
    //
    // head_capture: L2 norm of each head's slice at the requested positions -- the cheap
    // screening signal ([n_head] floats per position, 32x smaller than the slices).
    // head_write: overwrite head h's slice at the requested positions before W_o consumes it
    // (values = positions.size()*d_head floats, row-major per position). The causal primitive.
    struct HeadWrite {
        int layer = 0;
        int head = 0;
        std::vector<int> positions;    // board positions (row = pos - write_from_)
        std::vector<float> values;     // positions.size() * d_head floats
    };
    void set_head_capture(const std::vector<int>& layers, const std::vector<int>& positions,
                          bool rows = false);
    void clear_head_capture();
    void set_head_writes(const std::vector<HeadWrite>& ws);
    void clear_head_writes();
    // layer -> pos -> [n_head] slice norms (from the last forward with capture armed)
    const std::map<int, std::map<int, std::vector<float>>>& head_norms() const { return head_norms_; }
    // layer -> pos -> [ne0] full merged rows (rows=true capture): all heads' slices concatenated;
    // the client slices/means them locally (same client-side-means convention as the residual
    // tracer). Requested only for surviving sites, so the payload stays small.
    const std::map<int, std::map<int, std::vector<float>>>& head_rows() const { return head_rows_; }
    // dims observed at the last head hook: {ne0, d_head, n_head} -- the slice-0 shape probe.
    // d_head = ne0 / n_head_ when divisible, else 0 (probe FAILED -- architecture unsupported).
    const std::map<std::string, int>& head_dims() const { return head_dims_; }
    // Per-spec landing (2026-07-28 honesty fix): true at index i iff the i-th spec passed to
    // set_head_writes (in request order) actually wrote EVERY one of its requested positions during
    // the last forward. False for an out-of-range layer/head, a values.size() that didn't match the
    // PROBED d_head (only knowable once the tensor has actually been seen), or a position outside
    // the decoded segment. set_head_writes no longer pre-filters invalid specs -- validity is decided
    // exactly once, here, so a caller can never mistake a request that merely PARSED for one that
    // actually landed. A route/caller must check this (or refuse the whole response unless every
    // entry is true) rather than assume a structurally well-formed request succeeded.
    const std::vector<bool>& head_write_landed() const { return head_write_landed_; }

    // --- FFN (MLP) contribution hook (the additive analogue of kqv_out, for the feed-forward block) ---
    // The FIFTH eval_cb hook: "ffn_out-<il>" is the feed-forward block's contribution BEFORE it is
    // added into the residual stream (verified against models/qwen2.cpp: `cb(cur, "ffn_out", il);`
    // immediately precedes `cur = ggml_add(ctx0, cur, ffn_inp);` -- the residual add happens AFTER
    // this callback fires) -- the exact additive-not-overwrite analogue of kqv_out, but full n_embd
    // width (no per-head split; the MLP has no head structure to slice). Verified present, by direct
    // file read, for every architecture this repo's ~/.clozn/models set actually uses: models/qwen2.cpp
    // (Qwen2.5), models/llama.cpp (Llama-3.x), models/gemma4.cpp (gemma-4), models/qwen35.cpp
    // (Qwen3.5), models/dream.cpp (dream-v0), and models/deepseek2.cpp (Ministral/mistral4 --
    // llama_model_mistral4 inherits llama_model_deepseek2's graph verbatim per models/models.h:1108,
    // and deepseek2's own `cb(cur,"ffn_out",il)` fires on BOTH its dense-layer branch (models/
    // deepseek2.cpp:384) and its MoE branch (:413, after moe_out+ffn_shexp are combined) -- so a
    // MoE-vs-dense layer split inside one model still names the tensor uniformly).
    //
    // UNLIKE kqv_out (emitted from ONE shared helper in llama-graph.cpp, so present for essentially
    // every attention architecture) ffn_out is emitted PER-ARCHITECTURE, directly inside each model's
    // own .cpp file -- a repo-wide grep for the literal `cb(cur, "ffn_out", il)` found it MISSING from
    // 35 of this tree's 133 model files: state-space/no-conventional-FFN architectures (mamba*,
    // rwkv*), several MoE variants that route through a differently-named combine step (qwen2moe,
    // qwen3moe, olmoe, phimoe, openai-moe, smallthinker), and several encoder-only bert variants. So
    // "the tensor never fires for this model" is a REAL, architecture-dependent outcome here, not
    // just a shape-probe edge case the way kqv_out's d_head==0 is -- ffn_capture/ffn_write track
    // whether the tensor was actually SEEN per request (ffn_write_landed_ below; the route mirrors
    // residual `capture`'s capture_missing diagnostic for reads) so an unsupported architecture gets
    // an honest empty/refused result, never a silent no-op.
    //
    // ffn_capture: read the raw [n_embd] MLP-contribution row at requested (layer, position) pairs --
    // no norm summary (unlike head_capture there is no per-head split to compress away). ffn_write:
    // overwrite the MLP contribution at requested positions before the residual add consumes it
    // (values = positions.size()*n_embd floats, row-major per position -- the SAME shape convention
    // as residual `write`). UNLIKE residual `write` (layer 0 RESERVED as the read tap's final-layer
    // sentinel, so writes start at layer 1) every layer 0..n_layer-1 has its own FFN block, so layer 0
    // IS a valid ffn_write/ffn_capture target.
    void set_ffn_capture(const std::vector<int>& layers, const std::vector<int>& positions);
    void clear_ffn_capture();
    // layer -> pos -> [n_embd] raw MLP-contribution row (from the last forward with capture armed)
    const std::map<int, std::map<int, std::vector<float>>>& ffn_rows() const { return ffn_rows_; }
    // Layers that survived range validation [0, n_layer) -- mirrors GgmlAdapter::capture_layers().
    const std::vector<int>& ffn_capture_layers() const { return ffn_cap_layers_; }
    // REPLACE-per-call-site semantics mirror add_write_state: returns false (and adds nothing) when
    // layer is out of [0, n_layer) or values.size() != positions.size()*n_embd -- BOTH are known
    // STATICALLY (n_embd never needs a runtime probe, unlike kqv_out's d_head), so this is the fully
    // fail-closed path: a caller/route sees `false` and can refuse the whole request BEFORE any
    // forward ever runs, never a silently-dropped spec discovered after the fact.
    bool add_ffn_write(int il, const std::vector<int>& positions, const std::vector<float>& values);
    void clear_ffn_write();
    // Per-spec landing, aligned with the order add_ffn_write was called (1:1 -- add_ffn_write never
    // partially adds): true iff eval_cb actually saw "ffn_out-<il>" for that spec's layer AND wrote
    // every one of its requested positions. False for an architecture whose graph never names the
    // tensor "ffn_out" (see the class comment above) -- the runtime existence check add_ffn_write's
    // static shape validation cannot perform by itself.
    const std::vector<bool>& ffn_write_landed() const { return ffn_write_landed_; }

    // Standalone: load a fresh model + create a context over it (the original API).
    GgmlAdapter(const std::string& model_path, int mask_token_id,
                int eos_token_id = -1, int n_ctx = 4096,
                int n_gpu_layers = 0, bool flash_attn = true,
                int n_batch = 0, int n_ubatch = 0);
    // Shared model: create a private context over an already-loaded GgmlModel (for a server pool
    // of concurrent contexts that share one copy of the weights). flash_attn=false materializes
    // the attention weights so knockout works (costs some decode speed).
    explicit GgmlAdapter(std::shared_ptr<GgmlModel> model, int n_ctx = 4096,
                         bool flash_attn = true, int n_batch = 0,
                         int n_ubatch = 0);
    ~GgmlAdapter() override;

    GgmlAdapter(const GgmlAdapter&) = delete;
    GgmlAdapter& operator=(const GgmlAdapter&) = delete;

    const ModelConfig& config() const override { return cfg_; }
    const llama_vocab* vocab() const { return vocab_; }  // read-only native sampler/token-piece seam

    ForwardResult forward(const std::vector<int>& board,
                          const Mask& mask,
                          const std::shared_ptr<KVState>& kv,
                          const std::optional<std::vector<int>>& recompute_kv,
                          const std::vector<int>& logits_for) override;

    std::vector<int> encode(const std::string& text) const override;
    std::string decode(const std::vector<int>& ids) const override;

    // Total token-positions pushed through llama_decode since the last reset. The
    // hardware-independent measure of forward work: with cache off it grows by the full
    // board each pass; with Tier A/B reuse it grows only by the active block (+ one-time
    // freezes). The documented work metric (DESIGN invariant 5) — a wall-clock number ships
    // alongside it, never instead of it.
    long long decoded_tokens() const { return decoded_tokens_; }
    void reset_decoded_tokens() { decoded_tokens_ = 0; }
    int last_decode_call_count() const { return last_decode_call_count_; }
    long long last_prefill_logical_rows() const { return last_prefill_logical_rows_; }
    long long last_prefill_physical_rows() const { return last_prefill_physical_rows_; }
    long long last_prefill_reused_rows() const { return last_prefill_reused_rows_; }

    // White-box activation tap (Tier 2): when on, forward() fills ForwardResult.activations with the
    // per-position hidden state (llama_get_embeddings_ith) for the active block. Default off => empty
    // activations, zero overhead, the 8 scheduler goldens untouched. The white-box projection (concept
    // probes) is a separate consumer.
    void set_emit_activations(bool on);
    bool emit_activations() const { return emit_activations_; }
    int tap_layer() const { return tap_layer_; }  // residual layer the white-box tap reads (0 = final via embeddings)
    void set_tap_layer(int il);  // change which layer the cb_eval callback captures (0 = final via embeddings)
    // White-box WRITE side (Tier 2): activation steering via a llama control vector. `data` is the
    // n_embd*n_layer buffer (layer-1-indexed) applied to the residual stream over [il_start, il_end]
    // (inclusive, 1-based); clear_steer() removes it. The causal counterpart to the read tap — push a
    // concept direction in and watch the denoiser's commits move. NOT a golden path (off by default).
    void set_steer(const std::vector<float>& data, int il_start, int il_end);
    void clear_steer();
    // Fine-tune adapter (LoRA) attach/detach. A DIFFERENT MECHANISM from set_steer above, despite
    // llama.cpp placing llama_set_adapters_lora directly beside llama_set_adapter_cvec in its header --
    // which is exactly why the two get conflated on a keyword search (see docs/ENGINE_ADAPTER_SUPPORT.md).
    // A control vector adds ONE vector to the residual stream at inference time, uniformly across a layer
    // range. A LoRA is a pair of low-rank matrices whose product reconstructs a per-weight-matrix delta,
    // multiplied into specific attention/MLP projections. Having one gets you no part of the other.
    //
    // set_lora returns false and fills `err` when the adapter cannot be attached to THIS model --
    // rank/architecture mismatch, unreadable file, a GGUF that is not a LoRA -- rather than aborting the
    // worker: a bad adapter is a user error to report cleanly, not a crash. An empty path is a valid
    // request meaning "no adapter", not a failure. `scale` multiplies the delta; 0.0 attaches the adapter
    // while contributing nothing, which is the identity control that proves an observed output change
    // came from the adapter's weights rather than from the act of loading it.
    bool set_lora(const std::string& path, float scale, std::string* err = nullptr);
    void clear_lora();
    bool lora_loaded() const { return lora_ != nullptr; }
    const std::string& lora_path() const { return lora_path_; }
    float lora_scale() const { return lora_scale_; }
    // The adapter's OWN GGUF metadata (rank, alpha, target modules, ...) read back from the loaded file,
    // for honest run identity. Empty when no adapter is attached -- absent, never invented.
    std::map<std::string, std::string> lora_meta() const;
    // White-box STATE-WRITE (GAP #1, task #43): overwrite the residual at board `positions` with
    // `values` ([positions.size()*n_embd]) at the l_out-<il> mid layer (same naming as the read tap),
    // applied on EVERY subsequent forward via the SAME eval callback as the tap (ggml_backend_tensor_set,
    // no llama patch) until clear_write(). The causal inverse of the tap: read state out
    // (ForwardResult.activations) -> edit -> write_state -> observe the next forward. Returns true if armed.
    // write_state REPLACES the write set with this one spec (the original single-write semantics);
    // add_write_state APPENDS, so a joint intervention can patch several layers in ONE forward — the
    // circuit tracer's Δ_total arm ("all candidate nodes jointly ablated") needs exactly that.
    bool write_state(int il, const std::vector<int>& positions,
                     const std::vector<float>& values) override;
    bool add_write_state(int il, const std::vector<int>& positions,
                         const std::vector<float>& values);
    void clear_write();

    // Batched multi-ARM teacher-forced scoring (engine-debt: per-branch interventions). ONE
    // llama_decode carrying n_arms copies of the SAME token sequence (arm i = seq_id i, batch
    // rows [i*len, (i+1)*len)), so N intervention arms that differ only in their write specs
    // share one forward. The caller pre-translates each arm's write positions to batch rows
    // (arm_i*len + pos) via add_write_state -- the eval_cb write path applies unchanged, since
    // this forward runs with write_from_ == 0. logits_for positions are PER-ARM (0..len);
    // the result holds n_arms * logits_for.size() rows, arm-major. Requires
    // n_arms*len <= n_ctx (the kv_unified shared pool) and n_arms <= n_seq_max. Knockout,
    // attn_capture, and the capture plane are REFUSED alongside arms (their tensor layouts
    // under multi-seq batching are unvalidated -- refuse loudly rather than return silently
    // wrong evidence; the FP-landmine rule: batched results must be proven bit-exact against
    // the sequential path before anything trusts them).
    ForwardResult ar_forward_score_arms(const std::vector<int>& tokens,
                                        const std::vector<int>& logits_for, int n_arms);
    int n_layer() const { return n_layer_; }
    int n_embd() const { return n_embd_; }  // hidden size (e.g. the --sae dim check at startup)
    int n_ctx() const { return n_ctx_; }    // hard context window: absolute positions [0, n_ctx) fit the KV
    int n_batch() const { return n_batch_; }  // maximum logical rows in one llama_decode call
    int n_ubatch() const { return n_ubatch_; }  // accelerator execution microbatch
    std::map<std::string, std::size_t> memory_breakdown() const;

    // Diffusion soft-PREFIX injection (train-on-HF / serve-on-engine hybrid, diffusion side): lay a
    // continuous prefix (m x n_embd, row-major) as a FROZEN block at positions [0,m) that the whole board
    // attends to; the board's KV shifts to [m, m+n). Contained to forward() via pos_offset_ -- the scheduler
    // stays oblivious (the C++ analogue of the lab's PrefixAdapter). Clear before reusing the pooled context.
    void set_diffusion_prefix(const std::vector<float>& embd, int m);
    void clear_diffusion_prefix();

    // --- Autoregressive (causal) decode: the white-box read/steer harness over a standard
    // left-to-right LLM (any llama.cpp AR GGUF — Llama/Qwen/Mistral/...), as opposed to the
    // diffusion denoiser. The interpretability primitives (tap, probes, steering) are identical;
    // only the GENERATION paradigm differs. set_causal(true) flips the context to causal attention
    // and clears the KV (KV computed under the other attn mode is invalid); ar_forward decodes the
    // next tokens incrementally (KV-cached) and returns the LAST token's next-token logits + its
    // hidden state (tap-aware) as a ForwardResult the white-box helpers consume. No head shift
    // (the AR head is in-place: row i predicts token i+1). Not a golden path (diffusion is the oracle).
    void set_causal(bool on);
    bool causal() const { return causal_; }
    // Clear all request-local AR KV and bookkeeping.  The request-wide reference matcher uses this
    // at both boundaries because its seq-0 traversal deliberately survives sibling probes only
    // through explicit evict_from() calls.
    void reset_ar_kv();
    ForwardResult ar_forward(const std::vector<int>& tokens, int n_past);

    // Like ar_forward, but the inputs are RAW embeddings (n_rows x n_embd, row-major) spliced in at
    // [n_past, n_past+n_rows) via the llama_batch.embd path (the same one multimodal models use to inject
    // image vectors) instead of token ids. The bridge that lets a PyTorch-trained soft prefix ride into
    // the ggml runtime: train-on-HF, serve-on-llama.cpp. Returns the last row's next-token logits.
    ForwardResult ar_forward_embd(const std::vector<float>& embd, int n_rows, int n_past);

    // Teacher-forced SCORE forward (AR only; /score route): ONE causal decode of the whole sequence
    // `tokens` from a clean KV cache (absolute position 0), with per-token logits enabled ONLY at
    // `logits_for` (0-based positions within `tokens`) -- so a long prompt prefix that isn't being
    // scored costs no unembedding work, only the requested rows do. `logits_for` values are BATCH
    // token indices (not compacted output-row indices); llama_get_logits_ith translates each via the
    // context's output_ids, so scoring a scattered/non-contiguous set of positions is still correct
    // (though /score's own caller always requests a contiguous run). No head shift is applied (the
    // in-place causal head, same convention as ar_forward: row i's logits already predict token i+1).
    // Long sequences are split into logical batches of at most n_batch rows. n_batch defaults to
    // n_ctx for compatibility, but it is independent from n_ctx and n_ubatch when configured.
    // Always scores from a cold KV (never reuses another request's cache): teacher-forcing is a
    // one-shot, stateless read, and the pooled context is shared across requests.
    ForwardResult ar_forward_score(const std::vector<int>& tokens, const std::vector<int>& logits_for);

    // Forward-HARVEST (the §3.1 "activation harvesting at scale" path): one causal forward over a
    // whole text and ALL its per-token residuals at the tap layer — NOT just the last row like
    // ar_forward. Decodes `tokens` at absolute positions [0, n) under causal attention with the tap
    // on, then returns the full [n x n_embd] tap_buf_ (row r = the residual the model computed for
    // token r, having read tokens 0..r). One forward per text => natural-text activations, cheaply,
    // and no sustained-generation streaming (which crashed §3.6). Returns activations + act_rows
    // (= [0, n)) + n_embd in a ForwardResult; logits are left empty (harvest wants the state, not a
    // distribution). The caller must have set_causal(true) + set_emit_activations(true) + a mid tap
    // (tap_layer_ > 0); on the final-layer fallback (tap_layer_ == 0) it returns the per-token
    // embeddings instead (one D2H per token — slower but still correct). Not a golden path.
    ForwardResult harvest(const std::vector<int>& tokens);
    // Per-layer activation SUMMARY: one causal forward over `tokens` with the eval callback capturing
    // EVERY layer's residual (prefix-match "l_out-"), reduced to the L2 norm of each token's hidden state
    // at each layer — the depth x position map, in ONE forward (vs n_layer separate harvests). Caller sets
    // set_causal(true) first (as /harvest does); the read/steer tap state is left untouched.
    LayerSummary layer_summary(const std::vector<int>& tokens);
    // --- Checkpointing + branching (Phase 2.1) -----------------------------------------------
    // Save the current KV cache state for seq 0 as a serializable blob. `tokens` is the full
    // sequence fed so far (prompt + generated); `n_past` is how many positions the KV covers.
    // The caller owns these (generate_ar tracks them); the adapter just serializes the KV.
    EngineCheckpoint save_checkpoint(const std::vector<int>& tokens, int n_past) const;
    // Restore a previously saved checkpoint: clear the KV, load the blob, reset internal
    // bookkeeping (frozen_end_, boundary_row_). After this, ar_forward({next_tok}, ckpt.n_past)
    // resumes generation bit-exactly (greedy). Throws on size mismatch or restore failure.
    void load_checkpoint(const EngineCheckpoint& ckpt);

    // FORK-PIN-01: produce a NEW checkpoint representing an EARLIER n_past than `ckpt`'s own,
    // without generating any new tokens -- the "pin an earlier fork point" primitive (a durable pin
    // does not have to be the tip of a run). Reuses the exact same regime split
    // POST /v1/execution-fork documents: truncate_to > ckpt.prompt_tokens is a live-KV evict (the
    // boundary token was ORIGINALLY a single-token decode, so slicing the already-loaded blob at
    // that position reproduces it bit-exactly -- no bridge decode needed here since, unlike
    // execution-fork, nothing continues generating from a fresh logits row); truncate_to <=
    // ckpt.prompt_tokens must instead re-prefill tokens[0, truncate_to) as ONE fresh batch,
    // because those positions were originally computed inside ckpt's own single big prefill batch
    // and llama.cpp's batched attention kernel is not guaranteed bit-identical across batch shapes
    // (the documented batch-shape landmine -- see save_checkpoint's prefill_to doc comment).
    // Throws if ckpt.prompt_tokens is unpopulated (0 = unknown; refuses to guess the regime) or
    // truncate_to is out of [1, ckpt.n_past]. Must be called under the SAME lease discipline as
    // every other checkpoint route (acquire, call, done -- no state survives across HTTP calls).
    EngineCheckpoint truncate_checkpoint(const EngineCheckpoint& ckpt, int truncate_to);

    // --- Batched multi-sequence decode (Phase 2.2) -------------------------------------------
    // Copy seq 0's KV to seq 1..n-1 (shared-prefix branching). Requires the KV to be populated
    // for seq 0 first (via ar_forward). After this, each seq_id has an independent copy of the
    // KV entries and can diverge freely.
    void branch_kv(int n_branches);
    // Decode one token per sequence, chunking active sequences when n_batch is smaller than the
    // active set. `tokens_per_seq[i]` is the
    // token for seq i, decoded at position `n_past` with seq_id = i. Returns per-sequence logits
    // as a vector of ForwardResult (one per active sequence). The logits[i] field has the
    // next-token distribution for sequence i. `active` marks which sequences are still live
    // (not yet EOS/length-stopped); inactive sequences are skipped in the batch.
    std::vector<ForwardResult> ar_forward_batch(
        const std::vector<int>& tokens_per_seq, int n_past,
        const std::vector<bool>& active);
    // Prefill independent prompt token sequences, reusing their literal common token prefix through
    // a seq-0 KV fan-out, then chunking the remaining per-sequence suffix rows at n_batch. Each
    // sequence has its own llama sequence id and only its final prompt row emits logits. The caller
    // must keep the logical prompt rows within the existing n_ctx admission rule and the number of
    // resident sequences within the worker's sequence capacity.
    std::vector<ForwardResult> ar_forward_prefill_batch(
        const std::vector<std::vector<int>>& prompts);
    // Decode board[from,to) into seq 0, preserving the caller's existing [0,from) KV.  Only the
    // final row emits logits when requested; all rows use absolute positions and one sequence ID.
    ForwardResult ar_forward_seq0_segment(const std::vector<int>& board,
                                          int from, int to, bool need_last_logits);
    // Decode one token per sequence at independently advancing positions. This
    // is the survivor-only primitive used by native reference matching; inactive
    // sequences are omitted from the physical batch and their logits stay empty.
    std::vector<ForwardResult> ar_forward_batch_at(
        const std::vector<int>& tokens_per_seq,
        const std::vector<int>& positions,
        const std::vector<bool>& active);
    // Remove seq 1..n-1 from the KV, keeping only seq 0. Call after batched decode is done to
    // return the context to single-sequence state.
    void cleanup_seqs(int n_branches);

    // --- Multi-observer capture plane (Phase 2.3) --------------------------------------------
    // One forward, N observers: when the capture set is non-empty, eval_cb snapshots EVERY listed
    // layer's residual ("l_out-<il>", one D2H per layer per decode) and ar_forward hands the
    // completed CaptureFrame to the capture sink. The sink runs on the DECODE thread — it must
    // only queue the frame (observer compute belongs on the consumer's own worker thread; see
    // serve/readout_plane.hpp). Independent of the single-layer tap (tap_layer_/emit_activations),
    // which stays untouched for the existing /jlens + SAE + probes paths. Layers outside
    // (0, n_layer) are dropped (the "l_out-<il>" residual names exist only for mid layers).
    void set_capture_layers(const std::vector<int>& layers);
    const std::vector<int>& capture_layers() const { return capture_layers_; }
    void set_capture_sink(std::function<void(CaptureFrame&&)> sink);

    // Logits floats copied into the host logits buffer during forward(). Structural data movement,
    // not wall-clock.
    long long logits_d2h_floats() const { return logits_d2h_floats_; }
    void reset_logits_d2h_floats() { logits_d2h_floats_ = 0; }

    // Drop KV for board positions [pos, inf). Public because the KV-blob resume path
    // (generate_ar with resume_from) needs it for the one-token bridge decode: evict the last
    // saved position, re-decode its token there to recover the sampling row the blob doesn't
    // carry. Well-defined at any time; positions below `pos` are untouched.
    void evict_from(int pos);

private:
    // Decode board[from, to) at absolute positions [from, to), reusing whatever KV currently
    // covers [0, from). decode_only just runs llama_decode; decode_segment additionally returns
    // the host logits: row j == position from+j, valid until the next decode.
    void decode_only(const std::vector<int>& board, int from, int to,
                     std::vector<float>* logits_out = nullptr);
    const float* decode_segment(const std::vector<int>& board, int from, int to);
    void decode_prefix_embd();   // lay the diffusion soft prefix as embd at [0, diff_m_) (frozen context)

    // Decode [from, to) and freeze it: capture the boundary row (logits for position to-1,
    // the shifted-head source for the next block's first slot) and advance frozen_end_.
    void freeze_segment(const std::vector<int>& board, int from, int to);

    // active_start = min{q : mask(q, n-1)}; 0 when the mask is fully bidirectional.
    static int active_start_from_mask(const Mask& mask, int n);

    void init_context(int n_ctx, int n_batch, int n_ubatch);  // create + configure the llama context over model_

    std::shared_ptr<GgmlModel> model_owner_;  // keeps the (possibly shared) model alive
    llama_model* model_ = nullptr;            // == model_owner_->handle()
    llama_context* ctx_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
    ModelConfig cfg_{};
    int n_ctx_ = 0;
    int n_batch_ = 0;
    int n_ubatch_ = 0;
    bool emit_activations_ = false;  // white-box tap: fill ForwardResult.activations (forces host path)
    bool causal_ = false;            // attention mode: false = diffusion (bidirectional), true = AR (causal)
    int n_embd_ = 0;                 // hidden size, cached from the model for the activation tap
    int n_layer_ = 0;                // layer count, cached for control-vector (steering) sizing
    // Fine-tune adapter state. OWNED: detached from the context and freed by clear_lora() and by the
    // destructor, which must run while ctx_ is still alive. llama.cpp would also free a leaked adapter
    // when the model dies, but the model here is shared (model_owner_) and may outlive this adapter.
    llama_adapter_lora* lora_ = nullptr;
    std::string lora_path_;          // the path it was loaded from -- recorded, never guessed
    float lora_scale_ = 0.0f;
    int tap_layer_ = 0;              // residual layer to tap (~2/3 depth); 0 => final layer via embeddings
    std::string tap_name_;           // "l_out-<tap_layer_>": the residual tensor the eval callback captures
    std::vector<float> tap_buf_;     // last-decode residual at tap_layer_ [rows*n_embd], filled by eval_cb
    int tap_rows_ = 0;               // token rows in tap_buf_ (= the last decode segment's length)
    bool emit_layer_summary_ = false;              // when on, eval_cb folds EVERY l_out-<il> into layer_norms_
    std::vector<std::vector<float>> layer_norms_;  // [n_layer][rows]: per-token |residual| per layer (summary)
    // White-box state-WRITE (GAP #1): the mirror of the tap — eval_cb overwrites each spec's
    // positions' rows at its layer during decode (ggml_backend_tensor_set), applied until
    // clear_write(). A vector so a joint intervention can hit several layers in one forward.
    struct WriteSpec {
        int layer = 0;
        std::string name;             // "l_out-<layer>"
        std::vector<int> positions;   // board positions to overwrite
        std::vector<float> buf;       // [positions.size()*n_embd] new residual rows
    };
    std::vector<WriteSpec> writes_;
    int write_from_ = 0;                 // current decode segment's `from` (board position -> tensor row)
    // Attention knockout: eval_cb zeroes the listed A[head, query, key] entries of
    // "kq_soft_max-<il>" (shape [n_kv, n_tokens, n_head]) before kqv consumes them.
    std::vector<AttnKnockout> knockouts_;
    bool flash_attn_ = true;             // false => kq_soft_max materializes => knockout possible
    int n_head_ = 0;
    // Attention-row capture (read-only sibling of knockouts_): -1 = disarmed.
    int attn_capture_query_ = -1;
    std::map<int, std::vector<float>> attn_rows_;   // layer -> head-mean row [n_kv]
    // Head-output hook state (kqv_out-<il>): capture set + write specs + observed dims.
    std::vector<int> head_cap_layers_;
    std::vector<int> head_cap_positions_;
    bool head_cap_rows_ = false;
    std::vector<HeadWrite> head_writes_;
    std::vector<bool> head_write_landed_;   // parallel to head_writes_ -- see head_write_landed()
    std::map<int, std::map<int, std::vector<float>>> head_norms_;
    std::map<int, std::map<int, std::vector<float>>> head_rows_;
    std::map<std::string, int> head_dims_;
    // FFN contribution hook state (ffn_out-<il>): capture set + write specs + per-spec landing.
    struct FfnWrite {
        int layer = 0;
        std::string name;              // "ffn_out-<layer>"
        std::vector<int> positions;
        std::vector<float> buf;        // [positions.size()*n_embd] new MLP-contribution rows
    };
    std::vector<FfnWrite> ffn_writes_;
    std::vector<bool> ffn_write_landed_;               // parallel to ffn_writes_
    std::vector<int> ffn_cap_layers_;
    std::vector<int> ffn_cap_positions_;
    std::map<int, std::map<int, std::vector<float>>> ffn_rows_;   // layer -> pos -> [n_embd] row
    std::vector<float> diff_prefix_;     // [diff_m_ * n_embd] diffusion soft prefix, laid as a frozen block [0,diff_m_)
    int diff_m_ = 0;                     // diffusion prefix length (0 = none)
    // Multi-observer capture plane (Phase 2.3): eval_cb fills cap_bufs_ for every layer in the
    // capture set; fire_capture (end of ar_forward) moves them into a CaptureFrame for the sink.
    std::vector<int> capture_layers_;                 // layers to snapshot (empty = off)
    std::map<int, std::vector<float>> cap_bufs_;      // layer -> [rows*n_embd] last-decode residuals
    int cap_rows_ = 0;                                // rows in the last capture (= segment length)
    std::function<void(CaptureFrame&&)> capture_sink_;
    void fire_capture(int from, int rows);            // hand the completed frame to the sink (if armed)
    int pos_offset_ = 0;                 // = diff_m_ when a prefix is active: shifts the board's physical KV positions
    // ggml scheduler eval callback: grabs the mid-layer residual during llama_decode (no source patch).
    static bool eval_cb_thunk(struct ggml_tensor* t, bool ask, void* user_data);
    bool eval_cb(struct ggml_tensor* t, bool ask);

    // KV reuse state. [0, frozen_end_) is laid into the llama cache and frozen-exact.
    int frozen_end_ = 0;
    std::vector<float> boundary_row_;  // logits for position frozen_end_-1 (empty if none)
    std::vector<float> segment_logits_;  // host logits gathered across a chunked decode_segment call
    long long decoded_tokens_ = 0;     // token-positions sent through llama_decode
    long long logits_d2h_floats_ = 0;  // logits floats copied into the host logits buffer
    int last_decode_call_count_ = 0;   // llama_decode calls made by the most recent public forward
    long long last_prefill_logical_rows_ = 0;
    long long last_prefill_physical_rows_ = 0;
    long long last_prefill_reused_rows_ = 0;
};

}  // namespace clozn
