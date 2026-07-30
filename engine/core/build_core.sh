#!/usr/bin/env bash
# CPU-only base runtime build + unit tests -- POSIX mirror of build_core.bat. No CUDA, no ggml/llama.cpp:
# just clozn_runtime + clozn_sha256 and their model-free unit tests (state_write, checkpoint_store).
#
# This is the one build_*.sh in this set with a real precedent already running elsewhere: it is exactly
# what .github/workflows/ci.yml's `cpp` job configures/builds/tests on ubuntu-latest, on every push --
# see that job for the actual, continuously-green Linux evidence this script's flags are proven against.
#
# `set -euo pipefail`: unlike build_core.bat (which stops at CONFIGURE_FAILED/BUILD_FAILED but never
# gates its own exit code on ctest), this script propagates a ctest failure as a nonzero script exit too.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== tools ==="
command -v c++   >/dev/null || { echo CXX_MISSING;   exit 1; }
command -v cmake >/dev/null || { echo CMAKE_MISSING; exit 1; }
command -v ninja >/dev/null || { echo NINJA_MISSING; exit 1; }

echo "=== configure ==="
cmake -S . -B build-cpu -G Ninja -DCMAKE_BUILD_TYPE=Release || { echo CONFIGURE_FAILED; exit 2; }

echo "=== build ==="
cmake --build build-cpu || { echo BUILD_FAILED; exit 3; }

echo "=== ctest ==="
ctest --test-dir build-cpu --output-on-failure || { echo TESTS_FAILED; exit 4; }
echo "=== DONE ==="
