// test_state_write.cpp — GAP #1 (task #43): the read -> edit -> write -> observe loop, proven against a
// writable test double (CPU, backend-free, no ggml/CUDA). Gates the ModelAdapter::write_state seam:
//   (1) forward READS the hidden state out (ForwardResult::activations),
//   (2) we EDIT a position's state row,
//   (3) write_state WRITES it back,
//   (4) the NEXT forward's argmax at that position follows the edit; others are untouched.
// Also asserts the seam DEFAULT is a safe no-op: a non-writable adapter (the existing FakeAdapter)
// returns false from write_state and never mutates — so adding the seam method breaks nothing.
//
// The ggml L0 adapter implements write_state against the live llama context (a later slice + a thin
// additive llama patch, like the device-logits accessor); this CPU test fixes the contract first.
#include <cassert>
#include <cstdio>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "clozn/model.hpp"
#include "fake_adapter.hpp"   // the existing non-writable FakeAdapter (default write_state == false)

using namespace clozn;

// A minimal WRITABLE adapter: n_embd == vocab with an identity unembedding, so a hidden-state row IS a
// logit row. Baseline state at position p is one-hot at (p % (vocab-1)) (same picks as FakeAdapter);
// write_state overrides chosen positions, and the next forward emits those overrides as both the
// activations (read side) and the logits (output) -> the argmax follows the written state. The smallest
// fully-observable read/edit/write loop.
class WritableFakeAdapter : public ModelAdapter {
public:
    explicit WritableFakeAdapter(int vocab) {
        cfg_.vocab_size = vocab;
        cfg_.mask_token_id = vocab - 1;
        cfg_.eos_token_id = -1;
    }
    const ModelConfig & config() const override { return cfg_; }
    std::vector<int> encode(const std::string &) const override { return {}; }
    std::string decode(const std::vector<int> & ids) const override { return std::string(ids.size(), '.'); }

    int baseline_target(int p) const { return p % (cfg_.vocab_size - 1); }

    ForwardResult forward(const std::vector<int> &, const Mask &, const std::shared_ptr<KVState> &,
                          const std::optional<std::vector<int>> &,
                          const std::vector<int> & logits_for) override {
        ForwardResult out;
        out.n_requested = static_cast<int>(logits_for.size());
        out.vocab = cfg_.vocab_size;
        out.n_embd = cfg_.vocab_size;                       // identity unembedding: state row == logit row
        out.logits.assign(static_cast<size_t>(out.n_requested) * out.vocab, 0.0f);
        out.activations.assign(static_cast<size_t>(out.n_requested) * out.n_embd, 0.0f);
        out.act_rows = logits_for;
        for (int r = 0; r < out.n_requested; ++r) {
            const int p = logits_for[r];
            std::vector<float> row(static_cast<size_t>(out.n_embd), 0.0f);
            auto it = written_.find(p);
            if (it != written_.end()) {
                row = it->second;                           // edited state takes over
            } else {
                row[static_cast<size_t>(baseline_target(p))] = 8.0f;  // else the baseline one-hot
            }
            for (int v = 0; v < out.vocab; ++v) {
                out.activations[static_cast<size_t>(r) * out.n_embd + v] = row[static_cast<size_t>(v)];
                out.logits[static_cast<size_t>(r) * out.vocab + v] = row[static_cast<size_t>(v)];
            }
        }
        out.kv = std::make_shared<FakeKV>(0);
        return out;
    }

    bool write_state(int /*layer*/, const std::vector<int> & positions,
                     const std::vector<float> & values) override {
        const int d = cfg_.vocab_size;
        if (static_cast<int>(values.size()) != static_cast<int>(positions.size()) * d) return false;
        for (size_t i = 0; i < positions.size(); ++i) {
            written_[positions[i]] = std::vector<float>(values.begin() + static_cast<long>(i) * d,
                                                        values.begin() + static_cast<long>(i + 1) * d);
        }
        return true;
    }

private:
    ModelConfig cfg_{};
    std::map<int, std::vector<float>> written_;
};

static int argmax_row(const ForwardResult & r, int row) {
    const float * p = r.row(row);
    int best = 0; float bv = p[0];
    for (int v = 1; v < r.vocab; ++v) {
        if (p[v] > bv) { bv = p[v]; best = v; }
    }
    return best;
}

int main() {
    const int vocab = 16;
    WritableFakeAdapter adp(vocab);
    const std::vector<int> positions = {0, 1, 2, 3};

    // 1) READ: forward; capture the activation state + the baseline argmax per position.
    ForwardResult a = adp.forward({}, Mask{}, nullptr, std::nullopt, positions);
    assert(a.n_embd == vocab);
    assert(static_cast<int>(a.activations.size()) == static_cast<int>(positions.size()) * vocab);
    for (int r = 0; r < static_cast<int>(positions.size()); ++r) {
        assert(argmax_row(a, r) == adp.baseline_target(positions[r]));
    }

    // 2) EDIT position 2's state row: move its peak to a DIFFERENT token.
    const int pos = 2;
    const int old_tok = adp.baseline_target(pos);
    const int new_tok = (old_tok + 5) % (vocab - 1);
    assert(new_tok != old_tok);
    std::vector<float> edited(static_cast<size_t>(vocab), 0.0f);
    edited[static_cast<size_t>(new_tok)] = 8.0f;

    // 3) WRITE the edited state back.
    const bool ok = adp.write_state(/*layer*/ 12, {pos}, edited);
    assert(ok);

    // 4) OBSERVE: the next forward's argmax at pos follows the written state; other positions unchanged.
    ForwardResult b = adp.forward({}, Mask{}, nullptr, std::nullopt, positions);
    assert(argmax_row(b, 2) == new_tok);                      // the write took effect
    assert(argmax_row(b, 0) == adp.baseline_target(0));       // untouched positions unchanged
    assert(argmax_row(b, 1) == adp.baseline_target(1));
    assert(argmax_row(b, 3) == adp.baseline_target(3));

    // 5) DEFAULT seam is a safe no-op: a non-writable adapter returns false and never mutates.
    FakeAdapter fake(vocab, /*eos*/ -1, /*eos_at*/ -1);
    assert(fake.write_state(0, {0}, std::vector<float>(static_cast<size_t>(vocab), 1.0f)) == false);

    std::printf("test_state_write: OK (read -> edit -> write -> observe; default no-op honored)\n");
    return 0;
}
