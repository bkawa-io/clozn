#!/usr/bin/env bash
# CPU-only ggml+serve build (works everywhere) -- POSIX mirror of build_serve.bat.
#
# Prerequisite (same on every platform, not repeated by this script): reconstruct the pinned llama.cpp
# source once before the first configure --
#   python engine/core/third_party/bootstrap_llama.py
# build_serve.bat does not do this either; it is documented as its own step in docs/DEVELOPMENT.md.
#
# `set -euo pipefail`: build_serve.bat prints "=== BUILD_DONE exit=%errorlevel% ===" but never calls
# `exit /b` after it, so a FAILED build still leaves the .bat's own process exit code at 0 -- a caller
# chaining on success (`build_serve.bat && start_server`) would not notice. That is worse than no script
# at all, so this mirror deliberately does not reproduce it: a configure or build failure here aborts
# the script with a nonzero exit, every time.
set -euo pipefail
cd "$(dirname "$0")"

echo "=== configure (CPU ggml+serve) ==="
cmake -S . -B build-serve -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON -DGGML_CUDA=OFF || { echo CONFIGURE_FAILED; exit 2; }

echo "=== build ==="
cmake --build build-serve --target clozn-server || { echo BUILD_FAILED; exit 3; }
echo "=== BUILD_DONE ==="
