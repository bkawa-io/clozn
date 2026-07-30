#!/usr/bin/env bash
# SAE CUDA build + parity tests (kernel vs CPU reference; encoder vs torch oracle) -- POSIX mirror of
# build_sae.bat. Unlike build_cuda.bat (see build_cuda.sh), build_sae.bat's target names already match
# the current CMakeLists.txt, so this is a direct, literal mirror.
#
# Requires an NVIDIA CUDA toolchain (nvcc/CUDAToolkit) on PATH -- no macOS/Metal equivalent; CLOZN_BUILD_SAE
# hard-requires CUDA (find_package(CUDAToolkit REQUIRED), see CMakeLists.txt).
set -euo pipefail
cd "$(dirname "$0")"

echo "=== configure (CLOZN_BUILD_CUDA=ON CLOZN_BUILD_SAE=ON) ==="
cmake -S . -B build-cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DCLOZN_BUILD_CUDA=ON -DCLOZN_BUILD_SAE=ON \
    || { echo CONFIGURE_FAILED; exit 2; }

echo "=== build + run the sae parity tests (kernel vs CPU ref; encoder vs torch oracle) ==="
cmake --build build-cuda --target test_sae_topk test_sae_encoder || { echo BUILD_FAILED; exit 3; }
ctest --test-dir build-cuda -R "sae_topk|sae_encoder" --output-on-failure || { echo TESTS_FAILED; exit 4; }
echo "=== DONE ==="
