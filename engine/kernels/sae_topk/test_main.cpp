// test_main.cpp — tiny smoke-test driver for the SAE-sparsify top-k kernel.
// =============================================================================
//  Compiles host-side against the scaffold stub (or the CUDA lib); exercises the
//  pure host helper and links the kernel symbol. The REAL correctness check is
//  the Python suite (test_reference.py) + the CUDA parity harness (validate.py).
// =============================================================================

#include "sae_topk.cuh"

#include <cstdint>
#include <cstdio>

int main() {
    using namespace clozn;

    // sparse_code_bytes is a pure host helper — exercisable with no GPU.
    const int rows = 8, k = 16;
    const int bytes = sparse_code_bytes(rows, k);
    const int expect = 2 * rows * k * 4;
    std::printf("sparse_code_bytes(%d, %d) = %d (expect %d)\n", rows, k, bytes, expect);
    if (bytes != expect) {
        std::printf("FAIL: sparse_code_bytes mismatch\n");
        return 1;
    }

    std::printf(
        "sae_topk CUDA path validated by validate.py (indices exact, values within eps).\n"
        "Correctness oracle: reference.py / test_reference.py.\n");

    (void)&sae_topk;  // reference the symbol so it links.
    return 0;
}
