// test_main.cpp — tiny smoke-test driver for the confidence-select CUDA kernel.
// =============================================================================
//  PENDING CUDA TOOLCHAIN. This compiles host-side against the scaffold stub;
//  with nvcc + a GPU it would allocate device buffers, run a known logits
//  fixture, and compare against values produced by reference.py. The REAL
//  correctness check is the Python suite (test_reference.py); this exists only
//  so the optional CMake target links and so a GPU bring-up has a starting point.
// =============================================================================

#include "confidence_select.cuh"

#include <cstdint>
#include <cstdio>

int main() {
    using namespace clozn;

    // host_transfer_bytes is a pure host helper — exercisable with no GPU.
    const int n_masked = 12;
    const int bytes = host_transfer_bytes(n_masked);
    std::printf("host_transfer_bytes(%d) = %d (expect %d)\n",
                n_masked, bytes, 2 * n_masked * 4);
    if (bytes != 2 * n_masked * 4) {
        std::printf("FAIL: host_transfer_bytes mismatch\n");
        return 1;
    }

    std::printf(
        "confidence_select CUDA path is UNVERIFIED (pending nvcc + GPU).\n"
        "Correctness oracle: reference.py / test_reference.py.\n");

    // With a CUDA toolchain, this is where a device round-trip against a
    // reference.py-generated fixture would run and assert parity.
    (void)&confidence_select;  // reference the symbol so it links.
    return 0;
}
