#!/usr/bin/env bash
# CUDA-only kernel parity test build -- POSIX mirror of build_cuda.bat's INTENT (CLOZN_BUILD_CUDA=ON),
# not its literal target name.
#
# FINDING, not fixed here: build_cuda.bat on main today still targets `test_kernel_selector` / runs
# `ctest -R kernel_selector`. That target does not exist in the current engine/core/CMakeLists.txt --
# it lived under kernels/confidence_select, which no longer exists in this tree (removed with the
# diffusion scheduler/CommitSelector; see CMakeLists.txt's CLOZN_BUILD_CUDA option comment: "the
# CommitSelector wrapper this option used to also build were diffusion-only and were removed with the
# scheduler"). What CLOZN_BUILD_CUDA actually builds now is `test_sae_topk` (ctest name: sae_topk) --
# this script targets THAT, matching what CLOZN_BUILD_CUDA's own option docstring says it builds and
# matching build_sae.bat's already-current convention for the same option. build_cuda.bat itself was
# left untouched (out of this port's scope) but is very likely presently broken on Windows too; reported
# here, not silently mirrored into a script that would just be broken on POSIX as well.
#
# Requires an NVIDIA CUDA toolchain (nvcc) on PATH -- there is no macOS/Metal equivalent for this target.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== configure (sae_topk CUDA parity test, CLOZN_BUILD_CUDA=ON) ==="
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_CUDA=ON \
    || { echo CONFIGURE_FAILED; exit 2; }

echo "=== build + run the parity test (kernel vs CPU reference) ==="
cmake --build build-cuda --target test_sae_topk || { echo BUILD_FAILED; exit 3; }
ctest --test-dir build-cuda -R sae_topk --output-on-failure || { echo TESTS_FAILED; exit 4; }
echo "=== DONE ==="
