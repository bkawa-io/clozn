#include "clozn/model_ggml.hpp"

#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "ggml-backend.h"
#include "../third_party/llama.cpp/src/llama-ext.h"

namespace clozn {

namespace {
// llama_backend_init/free are process-global and refcount-free; init once on first
// adapter, never free (cheap, and freeing while another adapter lives would crash).
bool g_backend_inited = false;

struct PrefixTrieNode {
    int token = -1;
    int position = -1;
    std::vector<int> seq_ids;
    std::vector<int> terminal_arms;
    std::map<int, std::unique_ptr<PrefixTrieNode>> children;
};

struct TriePrefillRow {
    int token = -1;
    int position = -1;
    std::vector<int> seq_ids;
    std::vector<int> terminal_arms;
};
}  // namespace

GgmlModel::GgmlModel(const std::string& model_path, int mask_token_id,
                     int eos_token_id, int n_gpu_layers) {
    if (!g_backend_inited) {
        llama_backend_init();
        g_backend_inited = true;
    }
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers;  // > 0 offloads model layers to the GPU
    model_ = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model_) throw std::runtime_error("failed to load model: " + model_path);

    vocab_ = llama_model_get_vocab(model_);
    cfg_.vocab_size = llama_vocab_n_tokens(vocab_);
    cfg_.mask_token_id = mask_token_id;
    if (eos_token_id >= 0) {
        cfg_.eos_token_id = eos_token_id;
    } else {
        const llama_token eos = llama_vocab_eos(vocab_);
        cfg_.eos_token_id = eos < 0 ? -1 : static_cast<int>(eos);  // -1 == none
    }
    // Head convention from GGUF metadata (same key + default as llama's diffusion-cli): "true"/absent
    // => Dream-family shifted head, "false" => LLaDA-family in-place head.
    char shift_buf[16];
    if (llama_model_meta_val_str(model_, "diffusion.shift_logits", shift_buf, sizeof(shift_buf)) >= 0) {
        shift_logits_ = (std::strcmp(shift_buf, "true") == 0);
    }
}

GgmlModel::~GgmlModel() {
    if (model_) llama_model_free(model_);
}

void GgmlAdapter::init_context(int n_ctx, int n_batch, int n_ubatch) {
    if (n_ctx <= 0) throw std::invalid_argument("GgmlAdapter: n_ctx must be positive");
    n_ctx_ = n_ctx;
    n_batch_ = n_batch > 0 ? n_batch : n_ctx_;
    n_ubatch_ = n_ubatch > 0 ? n_ubatch : std::min(n_batch_, 512);
    if (n_batch_ <= 0 || n_batch_ > n_ctx_)
        throw std::invalid_argument("GgmlAdapter: n_batch must be in [1, n_ctx]");
    if (n_ubatch_ <= 0 || n_ubatch_ > n_batch_)
        throw std::invalid_argument("GgmlAdapter: n_ubatch must be in [1, n_batch]");
    model_ = model_owner_->handle();
    vocab_ = model_owner_->vocab();
    cfg_ = model_owner_->config();
    n_embd_ = llama_model_n_embd(model_);    // hidden size for the white-box activation tap
    n_layer_ = llama_model_n_layer(model_);  // layer count for control-vector steering
    n_head_ = llama_model_n_head(model_);    // attention heads (knockout indexes A[head, q, k])
    tap_layer_ = n_layer_ > 3 ? 2 : 0;  // layer 2: best per-token probe separation (sweep-validated)
    tap_name_ = "l_out-" + std::to_string(tap_layer_);   // per-layer residual name (llama-context.cpp "%s-%d")

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_ctx_;
    cp.n_batch = n_batch_;
    cp.n_ubatch = n_ubatch_;
    cp.n_seq_max = 16;     // Phase 2.2: batched multi-sequence decode (up to 16 branches)
    // UNIFIED KV (2026-07-22, found by the 2.8 tool-loop repro): without this, llama.cpp SPLITS
    // n_ctx across n_seq_max -- 4096/16 = a 256-token cap PER SEQUENCE ("n_ctx_seq (256)" at
    // boot), and any single-sequence request crossing ~256 effective tokens dies inside
    // llama_decode with "failed to find a memory slot" (observed: ~20% of tool-call
    // continuations; the first workload long enough to cross it). ar_forward's n_ctx_ guard and
    // the context receipts all reason about the FULL window, which unified restores: one shared
    // 4096-cell pool across sequences -- a lone request may use all of it, and the 16 branch
    // sequences share it (their TOTAL, not each, is bounded by n_ctx -- the honest budget for
    // batched branching anyway).
    cp.kv_unified = true;
    // Flash attention fuses the softmax inside the kernel, so "kq_soft_max-<il>" never
    // materializes and attention KNOCKOUT is impossible. Default stays AUTO (fast); a server
    // started with --no-flash-attn gets the explicit materialized path instead.
    cp.flash_attn_type = flash_attn_ ? LLAMA_FLASH_ATTN_TYPE_AUTO
                                     : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = &GgmlAdapter::eval_cb_thunk;  // observe the mid-layer residual (white-box tap, no patch)
    cp.cb_eval_user_data = this;
    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) throw std::runtime_error("failed to create llama context");
    llama_set_causal_attn(ctx_, false);  // the diffusion forward: fully bidirectional
}

GgmlAdapter::GgmlAdapter(const std::string& model_path, int mask_token_id,
                         int eos_token_id, int n_ctx,
                         int n_gpu_layers, bool flash_attn,
                         int n_batch, int n_ubatch)
    : model_owner_(std::make_shared<GgmlModel>(model_path, mask_token_id, eos_token_id, n_gpu_layers)),
      flash_attn_(flash_attn) {
    init_context(n_ctx, n_batch, n_ubatch);
}

GgmlAdapter::GgmlAdapter(std::shared_ptr<GgmlModel> model, int n_ctx, bool flash_attn,
                         int n_batch, int n_ubatch)
    : model_owner_(std::move(model)), flash_attn_(flash_attn) {
    if (!model_owner_) throw std::invalid_argument("GgmlAdapter: null GgmlModel");
    init_context(n_ctx, n_batch, n_ubatch);
}

std::map<std::string, std::size_t> GgmlAdapter::memory_breakdown() const {
    std::map<std::string, std::size_t> out{
        {"model_bytes", 0}, {"context_bytes", 0}, {"compute_bytes", 0},
    };
    if (!ctx_) return out;
    for (const auto& item : llama_get_memory_breakdown(ctx_)) {
        out["model_bytes"] += item.second.model;
        out["context_bytes"] += item.second.context;
        out["compute_bytes"] += item.second.compute;
    }
    return out;
}

void GgmlAdapter::set_attn_knockouts(const std::vector<AttnKnockout>& ks) {
    knockouts_.clear();
    for (const AttnKnockout& k : ks)
        if (k.layer >= 0 && k.layer < n_layer_ && !k.queries.empty() && !k.keys.empty())
            knockouts_.push_back(k);
}

void GgmlAdapter::clear_attn_knockouts() { knockouts_.clear(); }

void GgmlAdapter::set_attn_capture(int query_pos) {
    attn_capture_query_ = query_pos;
    attn_rows_.clear();
}

void GgmlAdapter::clear_attn_capture() {
    attn_capture_query_ = -1;
    attn_rows_.clear();
}

void GgmlAdapter::set_head_capture(const std::vector<int>& layers, const std::vector<int>& positions,
                                   bool rows) {
    head_cap_layers_ = layers;
    head_cap_positions_ = positions;
    head_cap_rows_ = rows;
    head_norms_.clear();
    head_rows_.clear();
    head_dims_.clear();
}

void GgmlAdapter::clear_head_capture() {
    head_cap_layers_.clear();
    head_cap_positions_.clear();
    head_cap_rows_ = false;
    head_norms_.clear();
    head_rows_.clear();
    head_dims_.clear();
}

void GgmlAdapter::set_head_writes(const std::vector<HeadWrite>& ws) {
    // NO silent pre-filter (2026-07-28 honesty fix): every requested spec is kept verbatim, indexed
    // 1:1 with head_write_landed_, so an out-of-range layer/head or a values.size() that doesn't
    // match the PROBED d_head shows up as landed=false rather than vanishing before eval_cb ever
    // runs. All validation now happens in exactly one place (eval_cb below) instead of being split
    // between here (silently dropping some specs) and there (silently skipping others) with no
    // record of which happened -- the split was the root cause of head_write_applied:true lying.
    head_writes_ = ws;
    head_write_landed_.assign(ws.size(), false);
}

void GgmlAdapter::clear_head_writes() {
    head_writes_.clear();
    head_write_landed_.clear();
}

void GgmlAdapter::set_ffn_capture(const std::vector<int>& layers, const std::vector<int>& positions) {
    ffn_cap_layers_.clear();
    for (int il : layers)
        if (il >= 0 && il < n_layer_) ffn_cap_layers_.push_back(il);  // every layer has its own FFN (layer 0 included)
    ffn_cap_positions_ = positions;
    ffn_rows_.clear();
}

void GgmlAdapter::clear_ffn_capture() {
    ffn_cap_layers_.clear();
    ffn_cap_positions_.clear();
    ffn_rows_.clear();
}

bool GgmlAdapter::add_ffn_write(int il, const std::vector<int>& positions, const std::vector<float>& values) {
    // Fully fail-closed: unlike head_write's d_head (only knowable from a live probe of kqv_out),
    // ffn_out is always n_embd-wide (every build_ffn(...) call site down-projects back to n_embd,
    // dense or MoE alike -- see models/deepseek2.cpp:379-413), so BOTH the layer range and the
    // values shape are known STATICALLY. A bad spec returns false here, before any forward runs --
    // never a silently-dropped write discovered only after the fact.
    if (il < 0 || il >= n_layer_) return false;   // every layer 0..n_layer-1 has its own FFN block
    if (n_embd_ <= 0) return false;
    if (values.size() != positions.size() * static_cast<size_t>(n_embd_)) return false;
    FfnWrite w;
    w.layer = il;
    w.name = "ffn_out-" + std::to_string(il);
    w.positions = positions;
    w.buf = values;
    ffn_writes_.push_back(std::move(w));
    ffn_write_landed_.push_back(false);   // becomes true in eval_cb iff "ffn_out-<il>" actually fires
                                           // for this architecture AND every position lands
    return true;
}

void GgmlAdapter::clear_ffn_write() {
    ffn_writes_.clear();
    ffn_write_landed_.clear();
}

GgmlAdapter::~GgmlAdapter() {
    clear_lora();                // must run while ctx_ is alive: it detaches from the context first
    if (ctx_) llama_free(ctx_);  // the model is freed by GgmlModel when the last adapter releases it
    // backend intentionally left initialized (see note above).
}

void GgmlAdapter::evict_from(int pos) {
    // Drop KV for board positions [pos, inf); pos_offset_ maps to physical positions so a diffusion prefix
    // laid at [0, pos_offset_) is never evicted (board pos >= 0 => physical >= pos_offset_).
    llama_memory_seq_rm(llama_get_memory(ctx_), 0, pos_offset_ + pos, -1);
}

void GgmlAdapter::set_emit_activations(bool on) {
    emit_activations_ = on;
    // Final-layer embeddings stay on as the fallback (cheap); the mid-layer tap_layer_ path is captured
    // by eval_cb during decode. Additive: per-token hidden states alongside logits (pooling NONE).
    llama_set_embeddings(ctx_, on);
}

void GgmlAdapter::set_tap_layer(int il) {
    tap_layer_ = (il > 0 && il < n_layer_) ? il : 0;
    tap_name_ = tap_layer_ > 0 ? ("l_out-" + std::to_string(tap_layer_)) : "";
}

bool GgmlAdapter::eval_cb_thunk(struct ggml_tensor* t, bool ask, void* user_data) {
    return static_cast<GgmlAdapter*>(user_data)->eval_cb(t, ask);
}

bool GgmlAdapter::eval_cb(struct ggml_tensor* t, bool ask) {
    const char* nm = ggml_get_name(t);
    // Attention knockout: "kq_soft_max-<il>" is the post-softmax weight matrix, laid out
    // [n_kv, n_tokens, n_head] (ne0 = keys, ne1 = queries, ne2 = heads). Zeroing A[h, q, k]
    // severs "query position q reads key position k at head h" -- the edge itself, rather than
    // the residual either end of it. Only exists with flash attention DISABLED.
    int ko_il = -1;
    if ((!knockouts_.empty() || attn_capture_query_ >= 0) &&
        std::strncmp(nm, "kq_soft_max-", 12) == 0)
        ko_il = std::atoi(nm + 12);
    if (ko_il >= 0) {
        bool want = attn_capture_query_ >= 0;   // capture wants EVERY layer's tensor
        for (const AttnKnockout& k : knockouts_) if (k.layer == ko_il) { want = true; break; }
        if (want) {
            if (ask) return true;
            const int n_kv = static_cast<int>(t->ne[0]);
            const int n_q  = static_cast<int>(t->ne[1]);
            const int n_h  = static_cast<int>(t->ne[2]);
            std::vector<float> row(static_cast<size_t>(n_kv));
            // CAPTURE first (read-only, head-mean), so a same-pass knockout never contaminates
            // the recorded heatmap: the row read here is the model's own pre-intervention
            // attention. Query index is relative to this decode segment (row = board pos - from),
            // same mapping the residual write path uses below.
            if (attn_capture_query_ >= 0) {
                const int q = attn_capture_query_ - write_from_;
                if (q >= 0 && q < n_q) {
                    std::vector<float>& mean = attn_rows_[ko_il];
                    mean.assign(static_cast<size_t>(n_kv), 0.0f);
                    for (int h = 0; h < n_h; ++h) {
                        const size_t off = ((static_cast<size_t>(h) * n_q) + q) * n_kv;
                        ggml_backend_tensor_get(t, row.data(), off * sizeof(float),
                                                row.size() * sizeof(float));
                        for (int i = 0; i < n_kv; ++i) mean[static_cast<size_t>(i)] += row[i];
                    }
                    const float inv = 1.0f / static_cast<float>(n_h > 0 ? n_h : 1);
                    for (float& v : mean) v *= inv;
                }
            }
            for (const AttnKnockout& k : knockouts_) {
                if (k.layer != ko_il) continue;
                for (int h = 0; h < n_h; ++h) {
                    if (k.head >= 0 && h != k.head) continue;
                    for (int q : k.queries) {
                        if (q < 0 || q >= n_q) continue;
                        const size_t off = ((static_cast<size_t>(h) * n_q) + q) * n_kv;
                        ggml_backend_tensor_get(t, row.data(), off * sizeof(float),
                                                row.size() * sizeof(float));
                        double removed = 0.0;
                        for (int key : k.keys) {
                            if (key < 0 || key >= n_kv) continue;
                            removed += row[static_cast<size_t>(key)];
                            row[static_cast<size_t>(key)] = 0.0f;
                        }
                        if (k.renormalize && removed > 0.0 && removed < 1.0) {
                            const float s = static_cast<float>(1.0 / (1.0 - removed));
                            for (int i = 0; i < n_kv; ++i) row[static_cast<size_t>(i)] *= s;
                        }
                        ggml_backend_tensor_set(t, row.data(), off * sizeof(float),
                                                row.size() * sizeof(float));
                    }
                }
            }
            return true;
        }
    }
    // Head-output hook (R5 head units): "kqv_out-<il>" = merged per-Q-head attention output,
    // PRE-W_o (verified: llama-graph.cpp names it before the wo matmul), rows [h*d_h,(h+1)*d_h)
    // of ne0 = head h. Capture (slice L2 norms, the screening signal) runs BEFORE writes, so a
    // same-pass intervention never contaminates the recorded norms -- the convention every hook
    // here follows. d_head is derived ne0/n_head_ and PROBED (head_dims_): an architecture where
    // that division fails records d_head=0 and applies nothing (absence visible, never wrong
    // slices).
    int hu_il = -1;
    if ((!head_writes_.empty() || !head_cap_layers_.empty()) &&
        std::strncmp(nm, "kqv_out-", 8) == 0)
        hu_il = std::atoi(nm + 8);
    if (hu_il >= 0) {
        bool want_cap = false, want_write = false;
        for (int L : head_cap_layers_) if (L == hu_il) { want_cap = true; break; }
        for (const HeadWrite& w : head_writes_) if (w.layer == hu_il) { want_write = true; break; }
        if (want_cap || want_write) {
            if (ask) return true;
            const int ne0 = static_cast<int>(t->ne[0]);
            const int ne1 = static_cast<int>(t->ne[1]);
            const int d_h = (n_head_ > 0 && ne0 % n_head_ == 0) ? ne0 / n_head_ : 0;
            head_dims_["ne0"] = ne0;
            head_dims_["n_head"] = n_head_;
            head_dims_["d_head"] = d_h;
            if (d_h > 0) {
                std::vector<float> row(static_cast<size_t>(ne0));
                if (want_cap) {
                    for (int pos : head_cap_positions_) {
                        const int r = pos - write_from_;
                        if (r < 0 || r >= ne1) continue;
                        ggml_backend_tensor_get(t, row.data(),
                                                static_cast<size_t>(r) * ne0 * sizeof(float),
                                                row.size() * sizeof(float));
                        std::vector<float>& norms = head_norms_[hu_il][pos];
                        norms.assign(static_cast<size_t>(n_head_), 0.0f);
                        for (int h = 0; h < n_head_; ++h) {
                            double ss = 0.0;
                            const float* s = row.data() + static_cast<size_t>(h) * d_h;
                            for (int i = 0; i < d_h; ++i) ss += static_cast<double>(s[i]) * s[i];
                            norms[static_cast<size_t>(h)] = static_cast<float>(std::sqrt(ss));
                        }
                        if (head_cap_rows_) head_rows_[hu_il][pos] = row;
                    }
                }
                if (want_write) {
                    // Honesty fix (2026-07-28): every spec's success is recorded in
                    // head_write_landed_[si] rather than assumed. "continue" below means "this spec
                    // did not land" -- landed_[si] simply stays false (its initial value from
                    // set_head_writes), never flipped to true on a partial or invalid write. The
                    // route's own post-forward check refuses the whole request unless every requested
                    // spec landed, so a caller can never mistake a dropped spec for a genuine
                    // negative causal result.
                    for (size_t si = 0; si < head_writes_.size(); ++si) {
                        const HeadWrite& w = head_writes_[si];
                        if (w.layer != hu_il) continue;
                        if (w.head < 0 || w.head >= n_head_) continue;   // out-of-range head: never applied
                        if (w.positions.empty()) continue;
                        if (w.values.size() != w.positions.size() * static_cast<size_t>(d_h))
                            continue;   // values.size() must match the PROBED d_head -- runtime-only fact
                        bool all_ok = true;
                        for (size_t pi = 0; pi < w.positions.size(); ++pi) {
                            const int r = w.positions[pi] - write_from_;
                            if (r < 0 || r >= ne1) { all_ok = false; continue; }  // outside this decode segment
                            const size_t off = (static_cast<size_t>(r) * ne0 +
                                                static_cast<size_t>(w.head) * d_h) * sizeof(float);
                            ggml_backend_tensor_set(t, w.values.data() + pi * d_h, off,
                                                    static_cast<size_t>(d_h) * sizeof(float));
                        }
                        if (all_ok) head_write_landed_[si] = true;
                    }
                }
            }
            return true;
        }
    }
    // FFN (MLP) contribution hook: "ffn_out-<il>" = the feed-forward block's output BEFORE it is
    // added into the residual stream (verified: models/qwen2.cpp `cb(cur, "ffn_out", il); cur =
    // ggml_add(ctx0, cur, ffn_inp);` -- the add happens AFTER this callback), full n_embd width like
    // l_out (no per-head split). UNLIKE kqv_out (one shared llama-graph.cpp call site, present for
    // virtually every architecture) ffn_out is emitted PER-ARCHITECTURE inside each model's own .cpp
    // file and is genuinely ABSENT for some (mamba/rwkv/several MoE variants -- see the header
    // comment) -- so this tensor simply never firing for an unsupported model is a real outcome,
    // tracked honestly below via ffn_write_landed_ (write) and the route's capture_missing-style
    // diagnostic (capture), never silently swallowed.
    int ffn_il = -1;
    if ((!ffn_writes_.empty() || !ffn_cap_layers_.empty()) &&
        std::strncmp(nm, "ffn_out-", 8) == 0)
        ffn_il = std::atoi(nm + 8);
    if (ffn_il >= 0) {
        bool want_ffn_cap = false, want_ffn_write = false;
        for (int L : ffn_cap_layers_) if (L == ffn_il) { want_ffn_cap = true; break; }
        for (const FfnWrite& w : ffn_writes_) if (w.layer == ffn_il) { want_ffn_write = true; break; }
        if (want_ffn_cap || want_ffn_write) {
            if (ask) return true;
            const int ne0 = static_cast<int>(t->ne[0]);
            const int ne1 = static_cast<int>(t->ne[1]);
            if (ne0 == n_embd_ && ne1 > 0) {
                if (want_ffn_cap) {
                    std::vector<float> row(static_cast<size_t>(ne0));
                    for (int pos : ffn_cap_positions_) {
                        const int r = pos - write_from_;
                        if (r < 0 || r >= ne1) continue;
                        ggml_backend_tensor_get(t, row.data(),
                                                static_cast<size_t>(r) * ne0 * sizeof(float),
                                                row.size() * sizeof(float));
                        ffn_rows_[ffn_il][pos] = row;
                    }
                }
                if (want_ffn_write) {
                    for (size_t si = 0; si < ffn_writes_.size(); ++si) {
                        const FfnWrite& w = ffn_writes_[si];
                        if (w.layer != ffn_il) continue;
                        if (w.buf.size() != w.positions.size() * static_cast<size_t>(n_embd_)) continue;
                        bool all_ok = !w.positions.empty();
                        for (size_t pi = 0; pi < w.positions.size(); ++pi) {
                            const int r = w.positions[pi] - write_from_;
                            if (r < 0 || r >= ne1) { all_ok = false; continue; }  // outside this decode segment
                            ggml_backend_tensor_set(t, w.buf.data() + pi * static_cast<size_t>(n_embd_),
                                                    static_cast<size_t>(r) * n_embd_ * sizeof(float),
                                                    static_cast<size_t>(n_embd_) * sizeof(float));
                        }
                        if (all_ok) ffn_write_landed_[si] = true;
                    }
                }
            }
            return true;
        }
    }
    const bool lout = std::strncmp(nm, "l_out-", 6) == 0;    // a per-layer residual tensor
    const bool read = emit_activations_ && tap_layer_ > 0 && tap_name_ == nm;   // the read tap
    bool write = false;                                       // the state-WRITE (any spec at this layer)
    if (lout && !writes_.empty())
        for (const WriteSpec& w : writes_) if (w.name == nm) { write = true; break; }
    // layer-summary mode: match EVERY per-layer residual "l_out-<il>" (prefix) for per-layer norms in one pass
    const bool summary = emit_layer_summary_ && lout;
    // capture plane (Phase 2.3): snapshot every layer in the capture set, one D2H per layer
    int cap_il = -1;
    if (lout && !capture_layers_.empty()) {
        const int il = std::atoi(nm + 6);
        for (int want : capture_layers_) if (want == il) { cap_il = il; break; }
    }
    if (!read && !write && !summary && cap_il < 0) return false;  // not a tensor we read or write
    if (ask) return true;                                     // yes — hand me its data after it computes
    const int ne0 = static_cast<int>(t->ne[0]);  // n_embd
    const int ne1 = static_cast<int>(t->ne[1]);  // token rows (the decode segment)
    if (read) {
        std::vector<float> rows(static_cast<size_t>(ne0) * ne1);
        ggml_backend_tensor_get(t, rows.data(), 0, rows.size() * sizeof(float));
        tap_buf_.insert(tap_buf_.end(), rows.begin(), rows.end());
        tap_rows_ += ne1;
    }
    if (cap_il >= 0 && ne0 == n_embd_ && ne1 > 0) {
        std::vector<float>& buf = cap_bufs_[cap_il];
        const size_t old = buf.size();
        buf.resize(old + static_cast<size_t>(ne0) * ne1);
        ggml_backend_tensor_get(t, buf.data() + old, 0,
                                static_cast<size_t>(ne0) * ne1 * sizeof(float));
        cap_rows_ += ne1;
    }
    if (summary && ne0 == n_embd_ && ne1 > 0) {
        const int il = std::atoi(nm + 6);                    // "l_out-<il>" -> il
        if (il >= 0 && il < n_layer_) {
            std::vector<float> rows(static_cast<size_t>(ne0) * ne1);
            ggml_backend_tensor_get(t, rows.data(), 0, rows.size() * sizeof(float));
            std::vector<float>& dst = layer_norms_[il];
            const size_t old = dst.size();
            dst.resize(old + static_cast<size_t>(ne1), 0.0f);
            for (int r = 0; r < ne1; ++r) {
                const float* h = rows.data() + static_cast<size_t>(r) * ne0;
                double ss = 0.0;
                for (int i = 0; i < ne0; ++i) ss += static_cast<double>(h[i]) * h[i];
                dst[old + static_cast<size_t>(r)] = static_cast<float>(std::sqrt(ss));
            }
        }
    }
    // WRITE side (GAP #1): overwrite each marked position's row (row = board position - this segment's
    // `from`), AFTER the read (so the tap reports the PRE-edit state) and before downstream layers consume
    // t — the activation-patch propagates forward. Rows outside [0, ne1) (e.g. a frozen-prefix decode) are
    // skipped, so the write lands only on the active block's decode. Every spec matching this layer
    // applies (a joint intervention may carry several specs, possibly across layers).
    if (write && ne0 == n_embd_) {
        for (const WriteSpec& w : writes_) {
            if (w.name != nm) continue;
            if (w.buf.size() != w.positions.size() * static_cast<size_t>(n_embd_)) continue;
            for (size_t i = 0; i < w.positions.size(); ++i) {
                const int row = w.positions[i] - write_from_;
                if (row >= 0 && row < ne1) {
                    ggml_backend_tensor_set(t, w.buf.data() + i * static_cast<size_t>(n_embd_),
                                            static_cast<size_t>(row) * n_embd_ * sizeof(float),
                                            static_cast<size_t>(n_embd_) * sizeof(float));
                }
            }
        }
    }
    return true;
}

void GgmlAdapter::set_steer(const std::vector<float>& data, int il_start, int il_end) {
    // Apply a control vector to the residual stream (the white-box WRITE). data is n_embd*n_layer,
    // layer-1-indexed; only [il_start, il_end] are applied. Empty data clears.
    llama_set_adapter_cvec(ctx_, data.empty() ? nullptr : data.data(),
                           data.size(), n_embd_, il_start, il_end);
}

void GgmlAdapter::clear_steer() {
    llama_set_adapter_cvec(ctx_, nullptr, 0, n_embd_, 0, n_layer_);
}

bool GgmlAdapter::set_lora(const std::string& path, float scale, std::string* err) {
    // NOT the control-vector path above -- see the header comment on why these two get confused.
    if (!model_ || !ctx_) {
        if (err) *err = "no model/context to attach a LoRA adapter to";
        return false;
    }
    clear_lora();                       // replace semantics: one adapter at a time in this build
    if (path.empty()) return true;      // "no adapter" is a valid request, not a failure

    // llama_adapter_lora_init does the rank/architecture validation for us: it returns null when the
    // adapter's tensors do not line up with THIS model's shapes. That is the check that has to exist
    // before attaching -- a mismatched adapter must be a clean refusal, not a wrong answer.
    llama_adapter_lora* adapter = llama_adapter_lora_init(model_, path.c_str());
    if (!adapter) {
        if (err) {
            *err = "could not load LoRA adapter '" + path + "' against this model: rank/architecture "
                   "mismatch, unreadable file, or not a LoRA GGUF";
        }
        return false;
    }

    float applied = scale;
    const int32_t rc = llama_set_adapters_lora(ctx_, &adapter, 1, &applied);
    if (rc != 0) {
        llama_adapter_lora_free(adapter);
        if (err) *err = "llama_set_adapters_lora failed (rc=" + std::to_string(rc) + ")";
        return false;
    }

    lora_ = adapter;
    lora_path_ = path;
    lora_scale_ = scale;
    return true;
}

void GgmlAdapter::clear_lora() {
    if (!lora_) return;
    // Detach from the context BEFORE freeing, and only while ctx_ is alive -- the destructor relies on
    // this ordering (it calls clear_lora() ahead of llama_free(ctx_)).
    if (ctx_) llama_set_adapters_lora(ctx_, nullptr, 0, nullptr);
    llama_adapter_lora_free(lora_);
    lora_ = nullptr;
    lora_path_.clear();
    lora_scale_ = 0.0f;
}

std::map<std::string, std::string> GgmlAdapter::lora_meta() const {
    // Whatever the adapter file itself declares -- rank, alpha, target modules. Read back rather than
    // inferred, so run identity records what was actually loaded. Keys that fail to read are skipped,
    // never defaulted: an unreadable field is absent, not zero.
    std::map<std::string, std::string> out;
    if (!lora_) return out;
    const int32_t n = llama_adapter_meta_count(lora_);
    for (int32_t i = 0; i < n; ++i) {
        char key[256] = {0};
        if (llama_adapter_meta_key_by_index(lora_, i, key, sizeof(key)) < 0) continue;
        char val[1024] = {0};
        if (llama_adapter_meta_val_str_by_index(lora_, i, val, sizeof(val)) < 0) continue;
        out[key] = val;
    }
    return out;
}

bool GgmlAdapter::write_state(int il, const std::vector<int>& positions,
                              const std::vector<float>& values) {
    writes_.clear();  // REPLACE semantics (the original single-write contract)
    return add_write_state(il, positions, values);
}

bool GgmlAdapter::add_write_state(int il, const std::vector<int>& positions,
                                  const std::vector<float>& values) {
    if (il <= 0 || il >= n_layer_) return false;   // 0 = final (no l_out name); writable mids are [1, n_layer)
    if (n_embd_ <= 0) return false;
    if (values.size() != positions.size() * static_cast<size_t>(n_embd_)) return false;
    WriteSpec w;
    w.layer = il;
    w.name = "l_out-" + std::to_string(il);
    w.positions = positions;
    w.buf = values;
    writes_.push_back(std::move(w));
    return true;
}

void GgmlAdapter::clear_write() {
    writes_.clear();
}

void GgmlAdapter::set_causal(bool on) {
    causal_ = on;
    llama_set_causal_attn(ctx_, on);
    // The attention mode changed, so any KV laid down under the other mode is now invalid
    // (a token's K/V depends on what it was allowed to attend to). Reset to a clean cache.
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
}

ForwardResult GgmlAdapter::ar_forward(const std::vector<int>& tokens, int n_past) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("ar_forward: empty tokens");
    if (n_past < 0) throw std::invalid_argument("ar_forward: n_past < 0");
    if (n_past + len > n_ctx_) throw std::invalid_argument("ar_forward: exceeds n_ctx");
    // Incremental causal decode: place `tokens` at absolute positions [n_past, n_past+len),
    // reusing whatever KV already covers [0, n_past). The logical batch may be smaller than this
    // request, so each chunk keeps its absolute positions and the same sequence id.
    decoded_tokens_ += len;
    const bool aggregate_capture = capture_sink_ && !capture_layers_.empty();
    std::map<int, std::vector<float>> captured;
    if (aggregate_capture) cap_bufs_.clear();
    last_decode_call_count_ = 0;
    int last_chunk_rows = 0;
    const bool callback_sensitive = (emit_activations_ && tap_layer_ > 0) ||
        aggregate_capture || !capture_layers_.empty() || !writes_.empty() || !knockouts_.empty() ||
        attn_capture_query_ >= 0 || !head_writes_.empty() || !head_cap_layers_.empty() ||
        !ffn_writes_.empty() || !ffn_cap_layers_.empty();
    const int decode_limit = callback_sensitive ? std::min(n_batch_, n_ubatch_) : n_batch_;
    for (int chunk_from = n_past; chunk_from < n_past + len; chunk_from += decode_limit) {
        const int chunk_rows = std::min(decode_limit, n_past + len - chunk_from);
        last_chunk_rows = chunk_rows;
        write_from_ = chunk_from;
        tap_buf_.clear();
        tap_rows_ = 0;
        if (aggregate_capture) {
            cap_bufs_.clear();
            cap_rows_ = 0;
        }
        llama_batch batch = llama_batch_init(chunk_rows, 0, 1);
        batch.n_tokens = chunk_rows;
        for (int i = 0; i < chunk_rows; ++i) {
            batch.token[i] = static_cast<llama_token>(tokens[chunk_from - n_past + i]);
            batch.pos[i] = chunk_from + i;          // absolute position: RoPE + KV slot
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (chunk_from + i == n_past + len - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("ar_forward: llama_decode failed");
        ++last_decode_call_count_;
        if (aggregate_capture) {
            for (int il : capture_layers_) {
                auto it = cap_bufs_.find(il);
                if (it == cap_bufs_.end() ||
                    it->second.size() != static_cast<size_t>(chunk_rows) * n_embd_)
                    continue;
                auto& dst = captured[il];
                dst.insert(dst.end(), it->second.begin(), it->second.end());
                it->second.clear();
            }
        }
    }
    if (aggregate_capture) {
        cap_bufs_ = std::move(captured);
        cap_rows_ = len;
        fire_capture(n_past, len);
    }

    const int vocab = cfg_.vocab_size;
    ForwardResult out;
    out.n_requested = 1;
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(n_past + len);

    // Next-token logits for the last position. In-place AR head — NO Dream-family shift: row i's
    // logits already predict token i+1 (standard causal LM), unlike the diffusion forward.
    const float* logits = llama_get_logits_ith(ctx_, -1);
    if (logits) out.logits.assign(logits, logits + vocab);

    // White-box activation tap (Tier 2): the hidden state at the last decoded position — "the
    // model's state having just read this token". One row; act_rows = the absolute position.
    if (emit_activations_ && n_embd_ > 0) {
        out.n_embd = n_embd_;
        out.act_rows = {n_past + len - 1};
        out.activations.assign(static_cast<size_t>(n_embd_), 0.0f);
        if (tap_layer_ > 0 && tap_rows_ == last_chunk_rows &&
            tap_buf_.size() == static_cast<size_t>(last_chunk_rows) * n_embd_) {
            // mid-layer residual l_out-<tap_layer_>: last row = last token
            std::memcpy(out.activations.data(),
                        tap_buf_.data() + static_cast<size_t>(last_chunk_rows - 1) * n_embd_,
                        static_cast<size_t>(n_embd_) * sizeof(float));
        } else {
            // final-layer fallback: the single output row (logits set only on the last token) is index 0
            const float* e = llama_get_embeddings_ith(ctx_, 0);
            if (e) std::memcpy(out.activations.data(), e, static_cast<size_t>(n_embd_) * sizeof(float));
        }
    }
    return out;
}

ForwardResult GgmlAdapter::ar_forward_embd(const std::vector<float>& embd, int n_rows, int n_past) {
    if (n_rows <= 0) throw std::invalid_argument("ar_forward_embd: empty");
    if (n_embd_ <= 0) throw std::runtime_error("ar_forward_embd: n_embd unknown");
    if (static_cast<int>(embd.size()) != n_rows * n_embd_)
        throw std::invalid_argument("ar_forward_embd: embd size != n_rows*n_embd");
    if (n_past < 0) throw std::invalid_argument("ar_forward_embd: n_past < 0");
    if (n_past + n_rows > n_ctx_) throw std::invalid_argument("ar_forward_embd: exceeds n_ctx");
    decoded_tokens_ += n_rows;
    last_decode_call_count_ = 0;
    for (int chunk_from = 0; chunk_from < n_rows; chunk_from += n_batch_) {
        const int chunk_rows = std::min(n_batch_, n_rows - chunk_from);
        write_from_ = n_past + chunk_from;
        llama_batch batch = llama_batch_init(chunk_rows, n_embd_, 1);
        batch.n_tokens = chunk_rows;
        for (int i = 0; i < chunk_rows; ++i) {
            std::memcpy(batch.embd + static_cast<size_t>(i) * n_embd_,
                        embd.data() + static_cast<size_t>(chunk_from + i) * n_embd_,
                        static_cast<size_t>(n_embd_) * sizeof(float));
            batch.pos[i] = n_past + chunk_from + i;  // absolute position: RoPE + KV slot
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (chunk_from + i == n_rows - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("ar_forward_embd: llama_decode failed");
        ++last_decode_call_count_;
    }

    const int vocab = cfg_.vocab_size;
    ForwardResult out;
    out.n_requested = 1;
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(n_past + n_rows);
    const float* logits = llama_get_logits_ith(ctx_, -1);   // last prefix row (unused if a prompt follows)
    if (logits) out.logits.assign(logits, logits + vocab);
    return out;
}

ForwardResult GgmlAdapter::ar_forward_score(const std::vector<int>& tokens,
                                            const std::vector<int>& logits_for) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("ar_forward_score: empty tokens");
    if (len > n_ctx_) throw std::invalid_argument("ar_forward_score: exceeds n_ctx");
    for (int p : logits_for)
        if (p < 0 || p >= len)
            throw std::invalid_argument("ar_forward_score: logits_for position out of range");

    // Score from a clean KV (position 0): teacher-forcing is a one-shot stateless read over the WHOLE
    // sequence, never incremental, so nothing from a prior request's cache may leak in.
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    write_from_ = 0;
    decoded_tokens_ += len;

    // Mark ONLY the requested positions for logits output -- everything else (typically most of the
    // prompt) costs no unembedding matmul, just the KV it contributes.
    std::vector<char> want(static_cast<size_t>(len), 0);
    for (int p : logits_for) want[static_cast<size_t>(p)] = 1;

    const int vocab = cfg_.vocab_size;
    ForwardResult out;
    out.n_requested = static_cast<int>(logits_for.size());
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(len);
    out.logits.resize(static_cast<size_t>(out.n_requested) * vocab);
    const bool aggregate_capture = capture_sink_ && !capture_layers_.empty();
    std::map<int, std::vector<float>> captured;
    if (aggregate_capture) cap_bufs_.clear();
    last_decode_call_count_ = 0;
    const bool callback_sensitive = aggregate_capture || !capture_layers_.empty() || !writes_.empty() || !knockouts_.empty() ||
        attn_capture_query_ >= 0 || !head_writes_.empty() || !head_cap_layers_.empty() ||
        !ffn_writes_.empty() || !ffn_cap_layers_.empty();
    const int decode_limit = callback_sensitive ? std::min(n_batch_, n_ubatch_) : n_batch_;
    for (int chunk_from = 0; chunk_from < len; chunk_from += decode_limit) {
        const int chunk_rows = std::min(decode_limit, len - chunk_from);
        write_from_ = chunk_from;
        tap_buf_.clear();
        tap_rows_ = 0;
        if (aggregate_capture) {
            cap_bufs_.clear();
            cap_rows_ = 0;
        }
        llama_batch batch = llama_batch_init(chunk_rows, 0, 1);
        batch.n_tokens = chunk_rows;
        for (int i = 0; i < chunk_rows; ++i) {
            const int pos = chunk_from + i;
            batch.token[i] = static_cast<llama_token>(tokens[pos]);
            batch.pos[i] = pos;                 // absolute positions from a clean cache: pos == index
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = want[static_cast<size_t>(pos)];
        }
        const int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("ar_forward_score: llama_decode failed");
        ++last_decode_call_count_;

        // Copy requested rows before the next chunk replaces llama.cpp's output buffer. The
        // public order remains logits_for order, even when positions are scattered.
        for (int r = 0; r < out.n_requested; ++r) {
            const int pos = logits_for[static_cast<size_t>(r)];
            if (pos < chunk_from || pos >= chunk_from + chunk_rows) continue;
            const float* row = llama_get_logits_ith(ctx_, pos - chunk_from);
            if (!row) throw std::runtime_error("ar_forward_score: missing logits row");
            std::memcpy(out.logits.data() + static_cast<size_t>(r) * vocab, row,
                        static_cast<size_t>(vocab) * sizeof(float));
        }
        if (aggregate_capture) {
            for (int il : capture_layers_) {
                auto it = cap_bufs_.find(il);
                if (it == cap_bufs_.end() ||
                    it->second.size() != static_cast<size_t>(chunk_rows) * n_embd_)
                    continue;
                auto& dst = captured[il];
                dst.insert(dst.end(), it->second.begin(), it->second.end());
                it->second.clear();
            }
        }
    }
    if (aggregate_capture) {
        cap_bufs_ = std::move(captured);
        cap_rows_ = len;
        fire_capture(0, len);
    }
    return out;
}

ForwardResult GgmlAdapter::ar_forward_score_arms(const std::vector<int>& tokens,
                                                 const std::vector<int>& logits_for, int n_arms) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("score_arms: empty tokens");
    if (n_arms < 1) throw std::invalid_argument("score_arms: n_arms must be >= 1");
    if (n_arms > 16) throw std::invalid_argument("score_arms: n_arms exceeds n_seq_max (16)");
    if (static_cast<long long>(n_arms) * len > n_ctx_)
        throw std::invalid_argument("score_arms: n_arms * len exceeds n_ctx (the arms share one "
                                    "kv_unified pool; fewer arms or a shorter sequence)");
    for (int p : logits_for)
        if (p < 0 || p >= len)
            throw std::invalid_argument("score_arms: logits_for position out of range");
    // Refuse un-validated tensor consumers rather than silently corrupt them: capture/knockout/
    // attn_capture row layouts under multi-seq batching are unproven (see header comment).
    if (!capture_layers_.empty() || !knockouts_.empty() || attn_capture_query_ >= 0)
        throw std::invalid_argument("score_arms: capture/knockout/attn_capture cannot be armed "
                                    "alongside batched arms (unvalidated under multi-seq layout)");
    if (!head_writes_.empty() || !ffn_writes_.empty() || !head_cap_layers_.empty() ||
        !ffn_cap_layers_.empty())
        throw std::invalid_argument("score_arms: head/ffn hooks cannot be armed alongside batched arms");

    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    write_from_ = 0;                          // batch rows ARE the write positions (pre-translated)
    decoded_tokens_ += n_arms * len;

    const int vocab = cfg_.vocab_size;
    const int per_arm = static_cast<int>(logits_for.size());
    ForwardResult out;
    out.n_requested = n_arms * per_arm;
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(len);
    out.logits.resize(static_cast<size_t>(out.n_requested) * vocab);
    const std::vector<WriteSpec> original_writes = writes_;
    last_decode_call_count_ = 0;
    try {
        // A chunk is a rectangular slice of a group of arms. Keeping each arm's positions
        // contiguous makes the local row mapping explicit, while grouping arms also handles
        // n_batch values smaller than n_arms.
        const int score_limit = original_writes.empty() ? n_batch_ : std::min(n_batch_, n_ubatch_);
        for (int arm_from = 0; arm_from < n_arms;) {
            const int arm_count = std::min(n_arms - arm_from, score_limit);
            const int rows_per_arm = std::max(1, score_limit / arm_count);
            for (int pos_from = 0; pos_from < len; pos_from += rows_per_arm) {
                const int pos_count = std::min(rows_per_arm, len - pos_from);
                const int total = arm_count * pos_count;
                write_from_ = 0;  // the temporary WriteSpec positions below are local batch rows
                writes_.clear();
                for (const WriteSpec& source : original_writes) {
                    WriteSpec mapped;
                    mapped.layer = source.layer;
                    mapped.name = source.name;
                    for (size_t pi = 0; pi < source.positions.size(); ++pi) {
                        const int global = source.positions[pi];
                        const int arm = global / len;
                        const int pos = global % len;
                        if (arm < arm_from || arm >= arm_from + arm_count ||
                            pos < pos_from || pos >= pos_from + pos_count)
                            continue;
                        mapped.positions.push_back((arm - arm_from) * pos_count + (pos - pos_from));
                        const size_t value_from = pi * static_cast<size_t>(n_embd_);
                        mapped.buf.insert(mapped.buf.end(),
                                          source.buf.begin() + static_cast<std::ptrdiff_t>(value_from),
                                          source.buf.begin() + static_cast<std::ptrdiff_t>(value_from + n_embd_));
                    }
                    if (!mapped.positions.empty()) writes_.push_back(std::move(mapped));
                }

                llama_batch batch = llama_batch_init(total, 0, 1);
                batch.n_tokens = total;
                for (int a = 0; a < arm_count; ++a) {
                    const int arm = arm_from + a;
                    for (int i = 0; i < pos_count; ++i) {
                        const int local = a * pos_count + i;
                        const int pos = pos_from + i;
                        batch.token[local] = static_cast<llama_token>(tokens[pos]);
                        batch.pos[local] = pos;  // per-seq absolute position
                        batch.n_seq_id[local] = 1;
                        batch.seq_id[local][0] = arm;
                        bool requested = false;
                        for (int requested_pos : logits_for)
                            if (requested_pos == pos) { requested = true; break; }
                        batch.logits[local] = requested;
                    }
                }
                const int rc = llama_decode(ctx_, batch);
                llama_batch_free(batch);
                if (rc != 0) throw std::runtime_error("score_arms: llama_decode failed");
                ++last_decode_call_count_;

                for (int a = 0; a < arm_count; ++a) {
                    const int arm = arm_from + a;
                    for (int r = 0; r < per_arm; ++r) {
                        const int pos = logits_for[static_cast<size_t>(r)];
                        if (pos < pos_from || pos >= pos_from + pos_count) continue;
                        const int local = a * pos_count + (pos - pos_from);
                        const float* row = llama_get_logits_ith(ctx_, local);
                        if (!row) throw std::runtime_error("score_arms: missing logits row");
                        std::memcpy(out.logits.data() +
                                        (static_cast<size_t>(arm) * per_arm + r) * vocab,
                                    row, static_cast<size_t>(vocab) * sizeof(float));
                    }
                }
            }
            arm_from += arm_count;
        }
    } catch (...) {
        writes_ = original_writes;
        cleanup_seqs(n_arms);
        throw;
    }
    writes_ = original_writes;
    // Leave no arm sequences behind for the next (single-seq) request.
    cleanup_seqs(n_arms);
    return out;
}

ForwardResult GgmlAdapter::harvest(const std::vector<int>& tokens) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("harvest: empty tokens");
    if (len > n_ctx_) throw std::invalid_argument("harvest: exceeds n_ctx");

    // One causal forward over the whole text from a clean cache: positions [0, len). We need ALL
    // rows' logits=1 so the tap captures every row (decode_only sets logits=1 everywhere). Reset the
    // KV first so this text's K/V never sees a prior text's positions (each /harvest is independent).
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    decode_only(tokens, 0, len);   // logits=1 at every position; tap_buf_ fills with all `len` rows

    const int vocab = cfg_.vocab_size;
    ForwardResult out;
    out.n_requested = 0;           // harvest returns state, not a distribution (logits left empty)
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(len);

    if (n_embd_ <= 0) return out;  // model has no hidden size? nothing to harvest
    out.n_embd = n_embd_;
    out.act_rows.resize(len);
    for (int r = 0; r < len; ++r) out.act_rows[r] = r;  // row r = token r's residual

    if (emit_activations_ && tap_layer_ > 0 && tap_rows_ == len &&
        tap_buf_.size() == static_cast<size_t>(len) * n_embd_) {
        out.activations = tap_buf_;  // mid-layer residual l_out-<tap_layer_>, all `len` rows in order
    } else {
        // Final-layer fallback (tap_layer_ == 0, or the cb didn't fire): pull every row's embedding.
        // decode_only set logits=1 for all positions, so output index r == position r.
        out.activations.assign(static_cast<size_t>(len) * n_embd_, 0.0f);
        for (int r = 0; r < len; ++r) {
            const float* e = llama_get_embeddings_ith(ctx_, r);
            if (e) std::memcpy(out.activations.data() + static_cast<size_t>(r) * n_embd_, e,
                               static_cast<size_t>(n_embd_) * sizeof(float));
        }
    }
    return out;
}

LayerSummary GgmlAdapter::layer_summary(const std::vector<int>& tokens) {
    const int len = static_cast<int>(tokens.size());
    if (len <= 0) throw std::invalid_argument("layer_summary: empty tokens");
    if (len > n_ctx_) throw std::invalid_argument("layer_summary: exceeds n_ctx");

    // One causal forward from a clean cache; the eval callback folds EVERY layer's l_out-<il> into
    // layer_norms_ (per-token L2 norm) as the graph runs -- all layers in this single pass.
    layer_norms_.assign(n_layer_, {});
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    const bool prev = emit_layer_summary_;
    emit_layer_summary_ = true;
    try {
        decode_only(tokens, 0, len);
    } catch (...) {
        emit_layer_summary_ = prev;
        throw;
    }
    emit_layer_summary_ = prev;

    LayerSummary out;
    out.n_layer = n_layer_;
    out.n_tokens = len;
    out.norms = layer_norms_;
    return out;
}

void GgmlAdapter::decode_only(const std::vector<int>& board, int from, int to,
                              std::vector<float>* logits_out) {
    const int len = to - from;
    if (len <= 0) throw std::invalid_argument("empty decode segment");
    if (from < 0 || to > static_cast<int>(board.size()))
        throw std::invalid_argument("decode segment outside board");
    decoded_tokens_ += len;
    const int vocab = cfg_.vocab_size;
    if (logits_out) logits_out->assign(static_cast<size_t>(len) * vocab, 0.0f);
    const bool collect_tap = emit_activations_ && tap_layer_ > 0;
    std::vector<float> all_tap;
    if (collect_tap) {
        all_tap.reserve(static_cast<size_t>(len) * n_embd_);
        tap_buf_.clear();
        tap_rows_ = 0;
    }
    std::vector<std::vector<float>> all_norms;
    last_decode_call_count_ = 0;
    if (emit_layer_summary_) {
        all_norms.resize(static_cast<size_t>(n_layer_));
        layer_norms_.assign(static_cast<size_t>(n_layer_), {});
    }
    // llama.cpp's non-causal graph requires one complete attention window in a physical ubatch.
    // Causal decode may submit n_batch rows and let the scheduler split them into n_ubatch pieces;
    // diffusion decode must keep each logical call within both limits.
    const bool callback_sensitive = collect_tap || emit_layer_summary_ || !capture_layers_.empty() ||
        !writes_.empty() || !knockouts_.empty() || attn_capture_query_ >= 0 || !head_writes_.empty() ||
        !head_cap_layers_.empty() || !ffn_writes_.empty() || !ffn_cap_layers_.empty();
    const int decode_limit = (!causal_ || callback_sensitive)
        ? std::min(n_batch_, n_ubatch_) : n_batch_;
    for (int chunk_from = from; chunk_from < to; chunk_from += decode_limit) {
        const int chunk_rows = std::min(decode_limit, to - chunk_from);
        write_from_ = chunk_from;   // board position -> tensor row mapping for the white-box state-WRITE
        tap_buf_.clear();
        tap_rows_ = 0;
        if (emit_layer_summary_)
            for (auto& rows : layer_norms_) rows.clear();
        llama_batch batch = llama_batch_init(chunk_rows, 0, 1);
        batch.n_tokens = chunk_rows;
        for (int i = 0; i < chunk_rows; ++i) {
            batch.token[i] = static_cast<llama_token>(board[chunk_from + i]);
            batch.pos[i] = pos_offset_ + chunk_from + i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = 1;                // logits at every position of the segment
        }
        const int rc = llama_decode(ctx_, batch);
        if (rc == 0 && logits_out) {
            for (int i = 0; i < chunk_rows; ++i) {
                const float* row = llama_get_logits_ith(ctx_, i);
                if (!row) { llama_batch_free(batch); throw std::runtime_error("llama_decode: missing logits row"); }
                std::memcpy(logits_out->data() + static_cast<size_t>(chunk_from - from + i) * vocab,
                            row, static_cast<size_t>(vocab) * sizeof(float));
            }
        }
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("llama_decode failed");
        ++last_decode_call_count_;
        if (collect_tap && tap_buf_.size() == static_cast<size_t>(chunk_rows) * n_embd_)
            all_tap.insert(all_tap.end(), tap_buf_.begin(), tap_buf_.end());
        if (emit_layer_summary_) {
            for (int il = 0; il < n_layer_; ++il) {
                const auto& rows = layer_norms_[static_cast<size_t>(il)];
                if (rows.size() == static_cast<size_t>(chunk_rows)) {
                    auto& dst = all_norms[static_cast<size_t>(il)];
                    dst.insert(dst.end(), rows.begin(), rows.end());
                }
            }
        }
    }
    if (collect_tap && all_tap.size() == static_cast<size_t>(len) * n_embd_) {
        tap_buf_ = std::move(all_tap);
        tap_rows_ = len;
    }
    if (emit_layer_summary_) layer_norms_ = std::move(all_norms);
}

const float* GgmlAdapter::decode_segment(const std::vector<int>& board, int from, int to) {
    segment_logits_.clear();
    decode_only(board, from, to, &segment_logits_);
    return segment_logits_.data();  // row j == position from+j, valid until next decode
}

void GgmlAdapter::freeze_segment(const std::vector<int>& board, int from, int to) {
    // Decode [from, to) reusing the frozen [0, from); its K/V never sees forward (nothing
    // beyond `to` is in the cache yet), so it is frozen-exact under the one-way law.
    evict_from(from);
    const float* rows = decode_segment(board, from, to);
    // Boundary row = logits for position to-1 (the shifted-head source for position `to`).
    const int vocab = cfg_.vocab_size;
    const float* src = rows + static_cast<size_t>((to - 1) - from) * vocab;
    boundary_row_.assign(src, src + vocab);
    frozen_end_ = to;
}

void GgmlAdapter::decode_prefix_embd() {
    // Lay the soft prefix as raw embeddings at PHYSICAL positions [0, diff_m_) (NOT offset -- the prefix IS
    // the offset). In the current bidirectional/diffusion attention mode it attends only to itself (nothing
    // after it exists yet), so it's frozen-exact under the one-way law; the board then attends to it.
    if (diff_m_ <= 0 || n_embd_ <= 0) return;
    decoded_tokens_ += diff_m_;
    last_decode_call_count_ = 0;
    const int decode_limit = std::min(n_batch_, n_ubatch_);
    for (int chunk_from = 0; chunk_from < diff_m_; chunk_from += decode_limit) {
        const int chunk_rows = std::min(decode_limit, diff_m_ - chunk_from);
        write_from_ = chunk_from;
        llama_batch batch = llama_batch_init(chunk_rows, n_embd_, 1);
        batch.n_tokens = chunk_rows;
        for (int i = 0; i < chunk_rows; ++i) {
            const int pos = chunk_from + i;
            std::memcpy(batch.embd + static_cast<size_t>(i) * n_embd_,
                        diff_prefix_.data() + static_cast<size_t>(pos) * n_embd_,
                        static_cast<size_t>(n_embd_) * sizeof(float));
            batch.pos[i] = pos;                   // physical [0, diff_m_): the frozen prefix block
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = (pos == diff_m_ - 1) ? 1 : 0;
        }
        const int rc = llama_decode(ctx_, batch);
        llama_batch_free(batch);
        if (rc != 0) throw std::runtime_error("decode_prefix_embd: llama_decode failed");
        ++last_decode_call_count_;
    }
}

void GgmlAdapter::set_diffusion_prefix(const std::vector<float>& embd, int m) {
    if (m <= 0 || n_embd_ <= 0 || static_cast<int>(embd.size()) != m * n_embd_)
        throw std::invalid_argument("set_diffusion_prefix: embd size != m*n_embd");
    diff_prefix_ = embd;
    diff_m_ = m;
    pos_offset_ = m;
}

void GgmlAdapter::clear_diffusion_prefix() {
    diff_prefix_.clear();
    diff_m_ = 0;
    pos_offset_ = 0;
}

int GgmlAdapter::active_start_from_mask(const Mask& mask, int n) {
    // The active (last) block = {q : block_id(q) == max} = {q : mask(q, n-1) == 1}, since
    // mask(q, k) = block_id(k) <= block_id(q) and block_id(n-1) is the max. Its start is the
    // smallest such q. A fully-bidirectional (all-ones) mask => 0 (whole-sequence).
    if (mask.n != n) throw std::invalid_argument("mask size != board size");
    for (int q = 0; q < n; ++q)
        if (mask.at(q, n - 1)) return q;
    return n;  // no position attends the last key — degenerate; treated as all-frozen
}

ForwardResult GgmlAdapter::forward(const std::vector<int>& board,
                                   const Mask& mask,
                                   const std::shared_ptr<KVState>& kv,
                                   const std::optional<std::vector<int>>& recompute_kv,
                                   const std::vector<int>& logits_for) {
    const int n = static_cast<int>(board.size());
    if (n == 0) throw std::invalid_argument("empty board");
    if (n > n_ctx_) throw std::invalid_argument("board exceeds n_ctx");

    // Everything before the active block is frozen-exact; the active block is recomputed.
    const int active_start = active_start_from_mask(mask, n);
    const bool reuse = (kv != nullptr);

    // recompute_kv, when given, must be a contiguous suffix [s, n) and its start must not
    // precede the active block (we never recompute a frozen block's interior — that scattered
    // Tier C path isn't expressible incrementally; the lab raises the same way).
    if (recompute_kv.has_value()) {
        const auto& r = *recompute_kv;
        if (!r.empty()) {
            const int s = r.front();
            for (int i = 0; i < static_cast<int>(r.size()); ++i)
                if (r[i] != s + i || r.back() != n - 1)
                    throw std::runtime_error("GgmlAdapter: recompute_kv must be a contiguous "
                                             "suffix [s, n) (Tier A/B prefix reuse)");
        }
    }

    if (!reuse) {
        // Cold start: rebuild the frozen prefix from scratch.
        llama_memory_clear(llama_get_memory(ctx_), true);
        frozen_end_ = 0;
        boundary_row_.clear();
        if (diff_m_ > 0) decode_prefix_embd();   // lay the soft prefix as a frozen block [0, diff_m_)
    }
    // Lay down + freeze the just-finalized block(s) so [0, active_start) is frozen-exact.
    // The gap is always exactly one block (prompt, or the block that just finalized), so a
    // single segment decode suffices and its boundary row feeds the active block's first slot.
    if (frozen_end_ < active_start) {
        freeze_segment(board, frozen_end_, active_start);
    } else if (frozen_end_ > active_start) {
        // Active block moved backward — only happens if a caller reuses across an
        // incompatible board; rebuild from cold to stay exact.
        llama_memory_clear(llama_get_memory(ctx_), true);
        frozen_end_ = 0;
        boundary_row_.clear();
        if (diff_m_ > 0) decode_prefix_embd();   // re-lay the prefix after the cold reset
        if (active_start > 0) freeze_segment(board, 0, active_start);
    }

    // Decode the active block [active_start, n), reusing the frozen prefix [0, active_start).
    evict_from(active_start);
    const int vocab = cfg_.vocab_size;

    ForwardResult out;
    out.n_requested = static_cast<int>(logits_for.size());
    out.vocab = vocab;
    out.kv = std::make_shared<GgmlKV>(n);

    if (active_start >= n) return out;  // no active rows (degenerate); logits_for must be empty

    // The Dream-family shift: logits for position m come from source row m-1 (position 0 uses its
    // own row 0 as filler, matching the lab's max(m-1, 0) — needed for suffix-only infill). A row
    // is in the active decode iff its source row >= active_start; source == active_start-1 is the
    // frozen boundary row, served from the host boundary_row_ (one row, at most one per pass).
    const bool shift = model_owner_->shift_logits();  // GGUF diffusion.shift_logits: Dream=true, LLaDA=false (in-place)
    auto src_of = [shift](int m) { return shift ? (m >= 1 ? m - 1 : 0) : m; };
    for (int m : logits_for) {
        if (m < 0 || m >= n) throw std::invalid_argument("logits_for position out of range");
    }
    decode_only(board, active_start, n);
    // llama copied every active output row to the host during decode.
    logits_d2h_floats_ += static_cast<long long>(n - active_start) * vocab;

    // White-box activation tap (Tier 2): pull the per-position hidden state for the active block.
    // Embeddings were enabled in set_emit_activations(); decode_only set logits=1 for every active
    // position, so output index j == active row j == board position active_start+j (batch order).
    if (emit_activations_ && n_embd_ > 0) {
        const int n_active = n - active_start;
        out.n_embd = n_embd_;
        out.act_rows.resize(n_active);
        for (int j = 0; j < n_active; ++j) out.act_rows[j] = active_start + j;
        if (tap_layer_ > 0 && tap_rows_ == n_active &&
            tap_buf_.size() == static_cast<size_t>(n_active) * n_embd_) {
            out.activations = tap_buf_;  // mid-layer residual l_out-<tap_layer_>, row j = position active_start+j
        } else {
            // final-layer fallback via the embeddings API (output index j == active row j == position+j)
            out.activations.assign(static_cast<size_t>(n_active) * n_embd_, 0.0f);
            for (int j = 0; j < n_active; ++j) {
                const float* e = llama_get_embeddings_ith(ctx_, j);
                if (e) std::memcpy(out.activations.data() + static_cast<size_t>(j) * n_embd_, e,
                                   static_cast<size_t>(n_embd_) * sizeof(float));
            }
        }
    }

    // Pull the host logits and apply the model's head shift.
    const float* rows = llama_get_logits(ctx_);  // row j == position active_start+j
    out.logits.resize(static_cast<size_t>(out.n_requested) * vocab);
    for (int r = 0; r < out.n_requested; ++r) {
        const int m = logits_for[r];
        const int src = src_of(m);
        const float* row;
        if (src >= active_start) {
            row = rows + static_cast<size_t>(src - active_start) * vocab;
        } else if (src == active_start - 1 && !boundary_row_.empty()) {
            row = boundary_row_.data();
        } else {
            throw std::runtime_error("GgmlAdapter: shifted-head source row is frozen and "
                                     "uncaptured (logits_for not at the active block front)");
        }
        std::memcpy(out.logits.data() + static_cast<size_t>(r) * vocab, row,
                    static_cast<size_t>(vocab) * sizeof(float));
    }
    return out;
}

EngineCheckpoint GgmlAdapter::save_checkpoint(const std::vector<int>& tokens, int n_past) const {
    EngineCheckpoint ckpt;
    ckpt.tokens = tokens;
    ckpt.n_past = n_past;
    ckpt.causal = causal_;
    const size_t sz = llama_state_seq_get_size(ctx_, 0);
    ckpt.kv_data.resize(sz);
    const size_t written = llama_state_seq_get_data(ctx_, ckpt.kv_data.data(), ckpt.kv_data.size(), 0);
    if (written == 0)
        throw std::runtime_error("save_checkpoint: llama_state_seq_get_data returned 0");
    ckpt.kv_data.resize(written);
    return ckpt;
}

void GgmlAdapter::load_checkpoint(const EngineCheckpoint& ckpt) {
    if (causal_ != ckpt.causal) set_causal(ckpt.causal);
    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    const size_t read = llama_state_seq_set_data(ctx_, ckpt.kv_data.data(), ckpt.kv_data.size(), 0);
    if (read == 0)
        throw std::runtime_error("load_checkpoint: llama_state_seq_set_data returned 0");
}

EngineCheckpoint GgmlAdapter::truncate_checkpoint(const EngineCheckpoint& ckpt, int truncate_to) {
    // Fail closed (no silent fallback): mirrors /v1/execution-fork's own refusal -- a checkpoint
    // that never had prompt_tokens populated cannot honestly support the regime split.
    if (ckpt.prompt_tokens <= 0)
        throw std::invalid_argument(
            "checkpoint has no recorded prompt_tokens; truncate cannot determine the "
            "reprefill/live_kv regime for it");
    if (truncate_to < 1 || truncate_to > ckpt.n_past)
        throw std::invalid_argument("truncate_to out of range [1, checkpoint n_past=" +
                                    std::to_string(ckpt.n_past) + "]");

    const std::vector<int> new_tokens(ckpt.tokens.begin(), ckpt.tokens.begin() + truncate_to);
    const bool live_kv = truncate_to > ckpt.prompt_tokens;

    // The checkpoint's declared steer (if any) was active for the ENTIRE run it came from, so a
    // reprefill of its earlier tokens must run under the same cvec (the same shape-true principle
    // /v1/checkpoint's own prefill applies). For the live-KV branch this is a harmless no-op: no
    // forward pass runs there, evict_from only drops already-computed KV entries.
    bool steer_applied = false;
    if (ckpt.has_steer && !ckpt.steer_cvec.empty()) {
        set_steer(ckpt.steer_cvec, ckpt.steer_lo, ckpt.steer_hi);
        steer_applied = true;
    }
    try {
        if (live_kv) {
            // load_checkpoint installs the full saved KV + resets bookkeeping; evict_from then
            // drops everything at/after truncate_to. What remains for [0, truncate_to) is BYTE-
            // IDENTICAL to what the original run's own KV held at that n_past -- those positions
            // were appended one at a time (generate_ar's steady-state loop) and are never rewritten
            // by later decodes, so slicing the blob is exact; no bridge decode is needed because,
            // unlike execution-fork, nothing here continues generating from a fresh logits row.
            load_checkpoint(ckpt);
            evict_from(truncate_to);
        } else {
            // Reprefill tokens[0, truncate_to) as ONE fresh batch -- reproducing the shape the
            // original prefill itself used for a prompt of exactly this length (never the live-KV
            // slice: those positions were computed inside ckpt's OWN, differently-shaped batch, and
            // llama.cpp's batched attention kernel is not guaranteed bit-identical across batch
            // sizes). Mirrors generate_ar.cpp's own non-resume prefill exactly (no explicit KV clear
            // beforehand either): ar_forward's writes at positions [0, truncate_to) fully overwrite
            // whatever this pooled context held before, and causal decode never reads a position
            // beyond its own n_past, so nothing at or past truncate_to can leak in.
            set_causal(true);
            ar_forward(new_tokens, 0);
        }
    } catch (...) {
        if (steer_applied) clear_steer();
        throw;
    }

    EngineCheckpoint out = save_checkpoint(new_tokens, truncate_to);
    if (steer_applied) clear_steer();

    out.prompt_tokens = std::min(ckpt.prompt_tokens, truncate_to);
    if (ckpt.has_sampler) {
        out.has_sampler = true;
        out.seed = ckpt.seed;
        out.temperature = ckpt.temperature;
        out.rep_penalty = ckpt.rep_penalty;
        out.top_k = ckpt.top_k;
        out.top_p = ckpt.top_p;
        // Draws a bit-exact continuous run would have consumed to reach THIS earlier point (mirrors
        // /v1/execution-fork's own rng_discard formula): one draw per sampled committed token past
        // the prompt boundary, zero for a position still inside the prompt or under greedy decode.
        out.rng_draws = (ckpt.temperature > 0.0 && truncate_to > out.prompt_tokens)
                             ? static_cast<uint64_t>(truncate_to - out.prompt_tokens)
                             : 0;
    }
    if (ckpt.has_steer) {
        out.has_steer = true;
        out.steer_cvec = ckpt.steer_cvec;
        out.steer_lo = ckpt.steer_lo;
        out.steer_hi = ckpt.steer_hi;
    }
    return out;
}

// --- Batched multi-sequence decode (Phase 2.2) -------------------------------------------

void GgmlAdapter::branch_kv(int n_branches) {
    if (n_branches < 2) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    for (int i = 1; i < n_branches; ++i) {
        llama_memory_seq_cp(mem, 0, static_cast<llama_seq_id>(i), -1, -1);
    }
}

std::vector<ForwardResult> GgmlAdapter::ar_forward_batch(
        const std::vector<int>& tokens_per_seq, int n_past,
        const std::vector<bool>& active) {
    std::vector<int> positions(tokens_per_seq.size(), n_past);
    return ar_forward_batch_at(tokens_per_seq, positions, active);
}

std::vector<ForwardResult> GgmlAdapter::ar_forward_batch_at(
        const std::vector<int>& tokens_per_seq,
        const std::vector<int>& positions,
        const std::vector<bool>& active) {
    const int n = static_cast<int>(tokens_per_seq.size());
    if (n <= 0) throw std::invalid_argument("ar_forward_batch: empty tokens_per_seq");
    if (static_cast<int>(positions.size()) != n || static_cast<int>(active.size()) != n)
        throw std::invalid_argument("ar_forward_batch: active size mismatch");

    int n_active = 0;
    for (int i = 0; i < n; ++i) if (active[i]) ++n_active;
    if (n_active == 0) return std::vector<ForwardResult>(n);
    for (int i = 0; i < n; ++i) {
        if (active[i] && (positions[i] < 0 || positions[i] >= n_ctx_))
            throw std::invalid_argument("ar_forward_batch: position outside context window");
    }

    decoded_tokens_ += n_active;
    last_decode_call_count_ = 0;

    const int vocab = cfg_.vocab_size;
    std::vector<ForwardResult> results(n);
    std::vector<int> active_indices;
    active_indices.reserve(n_active);
    for (int i = 0; i < n; ++i) if (active[i]) active_indices.push_back(i);
    for (int begin = 0; begin < n_active; begin += n_batch_) {
        const int chunk_rows = std::min(n_batch_, n_active - begin);
        write_from_ = 0;
        llama_batch batch = llama_batch_init(chunk_rows, 0, 1);
        batch.n_tokens = chunk_rows;
        for (int slot = 0; slot < chunk_rows; ++slot) {
            const int seq_i = active_indices[static_cast<size_t>(begin + slot)];
            batch.token[slot] = static_cast<llama_token>(tokens_per_seq[seq_i]);
            batch.pos[slot] = positions[seq_i];
            batch.n_seq_id[slot] = 1;
            batch.seq_id[slot][0] = static_cast<llama_seq_id>(seq_i);
            batch.logits[slot] = 1;
        }
        const int rc = llama_decode(ctx_, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("ar_forward_batch: llama_decode failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        ++last_decode_call_count_;
        for (int slot = 0; slot < chunk_rows; ++slot) {
            const float* logits = llama_get_logits_ith(ctx_, slot);
            if (!logits) {
                llama_batch_free(batch);
                throw std::runtime_error("ar_forward_batch: null logits for slot " +
                                         std::to_string(slot));
            }
            const int seq_i = active_indices[static_cast<size_t>(begin + slot)];
            ForwardResult& out = results[seq_i];
            out.n_requested = 1;
            out.vocab = vocab;
            out.kv = std::make_shared<GgmlKV>(positions[seq_i] + 1);
            out.logits.assign(logits, logits + vocab);
        }
        llama_batch_free(batch);
    }
    return results;
}

std::vector<ForwardResult> GgmlAdapter::ar_forward_prefill_batch(
        const std::vector<std::vector<int>>& prompts) {
    const int n = static_cast<int>(prompts.size());
    if (n <= 0) throw std::invalid_argument("ar_forward_prefill_batch: empty prompts");
    if (n > 16) throw std::invalid_argument("ar_forward_prefill_batch: too many sequences");

    int total = 0;
    for (int arm = 0; arm < n; ++arm) {
        if (prompts[static_cast<size_t>(arm)].empty())
            throw std::invalid_argument("ar_forward_prefill_batch: empty prompt");
        if (static_cast<int>(prompts[static_cast<size_t>(arm)].size()) > n_ctx_)
            throw std::invalid_argument("ar_forward_prefill_batch: prompt exceeds context window");
        total += static_cast<int>(prompts[static_cast<size_t>(arm)].size());
        if (total > n_ctx_)
            throw std::invalid_argument("ar_forward_prefill_batch: physical prompt rows exceed n_ctx");
    }

    llama_memory_clear(llama_get_memory(ctx_), true);
    frozen_end_ = 0;
    boundary_row_.clear();
    write_from_ = 0;
    last_decode_call_count_ = 0;
    last_prefill_logical_rows_ = total;
    last_prefill_physical_rows_ = 0;
    last_prefill_reused_rows_ = 0;

    PrefixTrieNode root;
    for (int arm = 0; arm < n; ++arm) {
        PrefixTrieNode* node = &root;
        const auto& prompt = prompts[static_cast<size_t>(arm)];
        for (int pos = 0; pos < static_cast<int>(prompt.size()); ++pos) {
            const int token = prompt[static_cast<size_t>(pos)];
            auto& child = node->children[token];
            if (!child) {
                child = std::make_unique<PrefixTrieNode>();
                child->token = token;
                child->position = pos;
            }
            node = child.get();
            node->seq_ids.push_back(arm);
        }
        node->terminal_arms.push_back(arm);
    }

    std::vector<TriePrefillRow> rows;
    rows.reserve(static_cast<size_t>(total));
    auto emit_rows = [&](auto&& self, const PrefixTrieNode& node) -> void {
        for (const auto& entry : node.children) {
            const PrefixTrieNode& child = *entry.second;
            rows.push_back(TriePrefillRow{
                child.token, child.position, child.seq_ids, child.terminal_arms});
            self(self, child);
        }
    };
    emit_rows(emit_rows, root);
    if (rows.empty())
        throw std::runtime_error("ar_forward_prefill_batch: trie produced no rows");

    const long long physical_rows = static_cast<long long>(rows.size());
    last_prefill_physical_rows_ = 0;
    last_prefill_reused_rows_ = last_prefill_logical_rows_ - physical_rows;
    decoded_tokens_ += physical_rows;

    const int vocab = cfg_.vocab_size;
    std::vector<std::vector<float>> final_logits(static_cast<size_t>(n));
    std::vector<ForwardResult> results(static_cast<size_t>(n));

    for (int chunk_from = 0; chunk_from < static_cast<int>(rows.size()); chunk_from += n_batch_) {
        const int chunk_rows = std::min(n_batch_, static_cast<int>(rows.size()) - chunk_from);
        last_prefill_physical_rows_ += chunk_rows;
        write_from_ = 0;
        llama_batch batch = llama_batch_init(chunk_rows, 0, n);
        batch.n_tokens = chunk_rows;
        for (int i = 0; i < chunk_rows; ++i) {
            const TriePrefillRow& row = rows[static_cast<size_t>(chunk_from + i)];
            batch.token[i] = static_cast<llama_token>(row.token);
            batch.pos[i] = row.position;
            batch.n_seq_id[i] = static_cast<int32_t>(row.seq_ids.size());
            for (int j = 0; j < batch.n_seq_id[i]; ++j)
                batch.seq_id[i][j] = static_cast<llama_seq_id>(row.seq_ids[static_cast<size_t>(j)]);
            batch.logits[i] = row.terminal_arms.empty() ? 0 : 1;
        }
        const int rc = llama_decode(ctx_, batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error("ar_forward_prefill_batch: trie llama_decode failed (rc=" +
                                     std::to_string(rc) + ")");
        }
        ++last_decode_call_count_;
        for (int i = 0; i < chunk_rows; ++i) {
            const TriePrefillRow& row = rows[static_cast<size_t>(chunk_from + i)];
            if (row.terminal_arms.empty()) continue;
            const float* logits = llama_get_logits_ith(ctx_, i);
            if (!logits) {
                llama_batch_free(batch);
                throw std::runtime_error("ar_forward_prefill_batch: missing trie terminal logits");
            }
            for (int arm : row.terminal_arms)
                final_logits[static_cast<size_t>(arm)].assign(logits, logits + vocab);
        }
        llama_batch_free(batch);
    }

    for (int arm = 0; arm < n; ++arm) {
        if (final_logits[static_cast<size_t>(arm)].empty())
            throw std::runtime_error("ar_forward_prefill_batch: missing terminal logits for sequence " +
                                     std::to_string(arm));
        ForwardResult& out = results[static_cast<size_t>(arm)];
        out.n_requested = 1;
        out.vocab = vocab;
        out.kv = std::make_shared<GgmlKV>(static_cast<int>(prompts[static_cast<size_t>(arm)].size()));
        out.logits = std::move(final_logits[static_cast<size_t>(arm)]);
    }
    return results;
}

void GgmlAdapter::cleanup_seqs(int n_branches) {
    if (n_branches < 2) return;
    llama_memory_t mem = llama_get_memory(ctx_);
    for (int i = 1; i < n_branches; ++i) {
        llama_memory_seq_rm(mem, static_cast<llama_seq_id>(i), -1, -1);
    }
}

void GgmlAdapter::set_capture_layers(const std::vector<int>& layers) {
    capture_layers_.clear();
    for (int il : layers)
        if (il > 0 && il < n_layer_) capture_layers_.push_back(il);  // "l_out-<il>" mid layers only
    cap_bufs_.clear();
    cap_rows_ = 0;
}

void GgmlAdapter::set_capture_sink(std::function<void(CaptureFrame&&)> sink) {
    capture_sink_ = std::move(sink);
}

void GgmlAdapter::fire_capture(int from, int rows) {
    if (!capture_sink_ || capture_layers_.empty()) return;
    CaptureFrame f;
    f.from = from;
    f.rows = rows;
    f.n_embd = n_embd_;
    for (int il : capture_layers_) {
        auto it = cap_bufs_.find(il);
        if (it == cap_bufs_.end() ||
            it->second.size() != static_cast<size_t>(rows) * n_embd_) continue;  // stale/missing capture
        f.layers.emplace_back(il, std::move(it->second));
        it->second.clear();  // moved-from: force a fresh alloc + fresh D2H next decode
    }
    if (!f.layers.empty()) capture_sink_(std::move(f));
}

std::vector<int> GgmlModel::encode(const std::string& text) const {
    // No BOS, parse special tokens — matches the lab's raw tok.encode for Qwen2 and the
    // forward test. Two-pass: probe the needed size, then tokenize.
    int need = -llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()),
                               nullptr, 0, /*add_special=*/false, /*parse_special=*/true);
    if (need <= 0) return {};
    std::vector<llama_token> toks(need);
    int n = llama_tokenize(vocab_, text.c_str(), static_cast<int>(text.size()), toks.data(),
                           need, /*add_special=*/false, /*parse_special=*/true);
    if (n < 0) throw std::runtime_error("tokenize failed");
    return std::vector<int>(toks.begin(), toks.begin() + n);
}

std::string GgmlModel::decode(const std::vector<int>& ids) const {
    std::string out;
    char piece[512];
    for (int id : ids) {
        int np = llama_token_to_piece(vocab_, static_cast<llama_token>(id), piece,
                                      sizeof(piece), /*lstrip=*/0, /*special=*/false);
        if (np < 0) np = 0;
        out.append(piece, static_cast<size_t>(np));
    }
    return out;
}

std::vector<int> GgmlAdapter::encode(const std::string& text) const {
    return model_owner_->encode(text);
}

std::string GgmlAdapter::decode(const std::vector<int>& ids) const {
    return model_owner_->decode(ids);
}

}  // namespace clozn
