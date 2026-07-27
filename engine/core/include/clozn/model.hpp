// clozn/model.hpp — the ModelAdapter seam in C++ (DESIGN invariant 1), the one
// boundary the model lives behind. Mirrors lab/clozn_lab/models/base.py. The scheduler
// (policies/stepper/blocks/cache) is pure logic against this interface; only the adapter
// implementations (e.g. the ggml one) touch a model backend.
#pragma once

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clozn/blocks.hpp"  // Mask
#include "clozn/kv.hpp"      // KVState

namespace clozn {

struct ModelConfig {
    int vocab_size;
    int mask_token_id;
    int eos_token_id;  // -1 == none
};

// Logits for the requested positions, row-major [n_requested, vocab], plus the KV the
// scheduler threads back on the next forward (invariant 4: model writes KV).
//
// Zero-copy device path (DESIGN §4.3): when an adapter can keep the logits on the GPU, it sets
// device_resident=true and leaves the host `logits` vector EMPTY (skipping the full-vocab D2H).
// `device_logits` is the device-resident logits tensor base (row r at device_logits + r*vocab,
// r in [0, device_n_rows)); `device_src_rows[i]` is the tensor row to read for requested
// position i (the model's head shift already applied). A device-side selector (the kernel)
// reads those rows in place; a host selector requires the host `logits` path (device_resident
// false). The const float* carries no CUDA types, so the seam header stays backend-free — only
// the CUDA selector TU dereferences it on-device.
struct ForwardResult {
    std::vector<float> logits;  // size == n_requested * vocab, row r = positions[r] (host path)
    int n_requested = 0;
    int vocab = 0;
    std::shared_ptr<KVState> kv;

    bool device_resident = false;
    const float * device_logits = nullptr;   // device tensor base; null unless device_resident
    int device_n_rows = 0;                    // rows in the device tensor
    std::vector<int> device_src_rows;         // [n_requested] source row per requested position;
                                              // -1 => the row is `boundary_row` (host), not in the
                                              // device tensor (a block's first slot under the shift)
    std::vector<float> boundary_row;          // the one frozen boundary row (vocab floats) for a
                                              // device_src_rows == -1 entry; empty if none needed

    // White-box activation tap (Tier 2): the hidden state per active-block position, filled only when
    // the adapter has emit_activations on (default off => empty, zero overhead). A model OUTPUT like
    // logits; the concept-probe projection is a separate white-box consumer, not part of the seam.
    std::vector<float> activations;  // [act_rows.size() * n_embd], row r = board position act_rows[r]
    int n_embd = 0;
    std::vector<int> act_rows;       // board positions for each activation row (the active block)

    const float * row(int r) const { return logits.data() + static_cast<size_t>(r) * vocab; }
};

class ModelAdapter {
public:
    virtual ~ModelAdapter() = default;
    virtual const ModelConfig & config() const = 0;

    // board: token ids [n]; mask: [n,n] attention mask; kv/recompute_kv: KV reuse
    // (kv==null => cold start, recompute all; recompute_kv==nullopt => recompute all);
    // logits_for: positions whose logits to return. Returns logits (per the model's
    // head convention — the adapter owns any shift) + the KV covering the board.
    virtual ForwardResult forward(const std::vector<int> & board,
                                  const Mask & mask,
                                  const std::shared_ptr<KVState> & kv,
                                  const std::optional<std::vector<int>> & recompute_kv,
                                  const std::vector<int> & logits_for) = 0;

    virtual std::vector<int> encode(const std::string & text) const = 0;
    virtual std::string decode(const std::vector<int> & ids) const = 0;

    // --- White-box state WRITE (Tier 2 / GAP #1): the inverse of the activation tap. ----------------
    // Overwrite the hidden state at board `positions` with `values` ([positions.size() * n_embd],
    // row-major, the SAME layout as ForwardResult::activations) at residual `layer`, taking effect on
    // the NEXT forward. This is the missing half of the read -> inspect -> edit -> write -> observe loop:
    // ForwardResult::activations reads state OUT; write_state writes edited state back IN. Default is a
    // no-op returning false, so existing adapters are unaffected and opt in explicitly; the ggml L0
    // adapter implements it against the live llama context (a thin additive llama patch, like the
    // device-logits accessor). Returns true if applied.
    virtual bool write_state(int /*layer*/, const std::vector<int> & /*positions*/,
                             const std::vector<float> & /*values*/) { return false; }
};

}  // namespace clozn
