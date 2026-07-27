// blocks.cpp — implementation of clozn/blocks.hpp. BlockPlan::blocks() (the diffusion semi-AR block
// partitioner) was removed with the diffusion program (THE_CUT); block_id/attention_mask survive
// because they're still called (always with block_len=0, fully bidirectional) by the concept-probe
// calibration's one-shot sentence read (server_shared.hpp) and clozn-probe-sweep.
#include "clozn/blocks.hpp"

#include <algorithm>

namespace clozn {

int block_id(int pos, int prompt_len, int block_len) {
    if (pos < prompt_len) return -1;
    return (pos - prompt_len) / block_len;
}

Mask attention_mask(int working_len, int prompt_len, int block_len) {
    Mask m;
    m.n = working_len;
    m.data.assign(static_cast<size_t>(working_len) * working_len, 0);
    if (block_len == 0) {
        // fully bidirectional: all attend.
        std::fill(m.data.begin(), m.data.end(), static_cast<unsigned char>(1));
        return m;
    }
    std::vector<int> ids(working_len);
    for (int p = 0; p < working_len; ++p) ids[p] = block_id(p, prompt_len, block_len);
    // M[q, k] = ids[k] <= ids[q]  (the one-way law).
    for (int q = 0; q < working_len; ++q)
        for (int k = 0; k < working_len; ++k)
            m.data[static_cast<size_t>(q) * working_len + k] = (ids[k] <= ids[q]) ? 1 : 0;
    return m;
}

}  // namespace clozn
