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
struct ForwardResult {
    std::vector<float> logits;  // size == n_requested * vocab, row r = positions[r] (host path)
    int n_requested = 0;
    int vocab = 0;
    std::shared_ptr<KVState> kv;

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
    // adapter implements it against the live llama context through its eval callback. Returns true
    // if applied.
    virtual bool write_state(int /*layer*/, const std::vector<int> & /*positions*/,
                             const std::vector<float> & /*values*/) { return false; }
};

}  // namespace clozn
