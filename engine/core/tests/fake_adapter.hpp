// fake_adapter.hpp — a deterministic, model-free ModelAdapter for backend-free tests. argmax at
// position p is f(p) = p % (vocab-1), peaked so every committed slot is near-equally confident;
// confidence ties break toward the lower position (confidence_topk), so quota fills strictly
// left-to-right and the final board is exactly f(p) per slot. One slot can be redirected to EOS.
// Independent of the board contents and the attention mask, so block/whole-sequence/cached runs
// all yield the same picks — what lets tests assert the loop's control flow + event stream.
#pragma once

#include <string>
#include <vector>

#include "clozn/model.hpp"

namespace clozn {

struct FakeKV : KVState {
    int n;
    explicit FakeKV(int n_) : n(n_) {}
    int seq_len() const override { return n; }
};

class FakeAdapter : public ModelAdapter {
public:
    FakeAdapter(int vocab, int eos, int eos_at) : eos_at_(eos_at) {
        cfg_.vocab_size = vocab;
        cfg_.mask_token_id = vocab - 1;  // tokens 0..vocab-2 are never the mask
        cfg_.eos_token_id = eos;         // -1 == none
    }

    const ModelConfig& config() const override { return cfg_; }

    int target(int p) const {
        if (p == eos_at_) return cfg_.eos_token_id;
        return p % (cfg_.vocab_size - 1);  // in [0, vocab-2], never the mask token
    }

    ForwardResult forward(const std::vector<int>& /*board*/, const Mask& /*mask*/,
                          const std::shared_ptr<KVState>& /*kv*/,
                          const std::optional<std::vector<int>>& /*recompute_kv*/,
                          const std::vector<int>& logits_for) override {
        ForwardResult out;
        out.n_requested = static_cast<int>(logits_for.size());
        out.vocab = cfg_.vocab_size;
        out.logits.assign(static_cast<size_t>(out.n_requested) * cfg_.vocab_size, 0.0f);
        for (int r = 0; r < out.n_requested; ++r) {
            const int t = target(logits_for[r]);
            out.logits[static_cast<size_t>(r) * cfg_.vocab_size + t] = 8.0f;  // peaked argmax
        }
        out.kv = std::make_shared<FakeKV>(0);  // non-null so the delta-cache path activates
        return out;
    }

    std::vector<int> encode(const std::string&) const override { return {}; }
    std::string decode(const std::vector<int>& ids) const override {
        return std::string(ids.size(), '.');  // length only; text isn't asserted
    }

private:
    ModelConfig cfg_{};
    int eos_at_;
};

}  // namespace clozn
