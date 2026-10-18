// clozn/blocks.hpp — the shared attention-mask type and builder used by the model seam.
// A block length of zero produces a fully bidirectional mask.
#pragma once

#include <vector>

namespace clozn {

// [n, n] row-major boolean mask; at(q, k) true means query q may attend to key k.
struct Mask {
    int n = 0;
    std::vector<unsigned char> data;  // n*n, 1 = attend
    bool at(int q, int k) const { return data[static_cast<size_t>(q) * n + k] != 0; }
};

// -1 for prompt positions; 0, 1, 2, ... for successive output blocks. Vestigial outside
// attention_mask's own block_len != 0 branch now that nothing calls it with block_len != 0.
int block_id(int pos, int prompt_len, int block_len);

// whole-sequence (block_len=0): fully bidirectional -- the only case any surviving caller uses
// (the concept-probe calibration's one-shot sentence read). Block mode (block_len > 0): M[q,k] =
// block_id(k) <= block_id(q), kept only because it's cheap to keep and nothing depends on its
// absence; no surviving caller passes a non-zero block_len.
Mask attention_mask(int working_len, int prompt_len, int block_len);

}  // namespace clozn
