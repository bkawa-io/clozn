#!/usr/bin/env bash
# GPU build with platform detection -- POSIX mirror of build_gpu.bat's INTENT, not its literal flags:
# that script hardcodes CUDA (this dev box's only GPU backend) plus a CMAKE_CUDA_ARCHITECTURES=120 pin
# for its one RTX 5080 (Blackwell, sm_120 -- see its own comment). This script instead chooses the
# backend per platform -- Metal on macOS, CUDA on Linux (if nvcc is on PATH), else a CPU fallback with a
# warning -- because "GPU" means different hardware/APIs on different platforms; there is no POSIX
# analogue of "assume CUDA" that would be honest on macOS.
#
# Deliberately dropped: -DCMAKE_CUDA_ARCHITECTURES=120. That pin is specific to the one GPU this repo
# has actually built for; hardcoding it here would silently mis-target any other Linux user's GPU.
# Leaving it unset lets CMakeLists.txt's own portable default take over (currently "80;89;120" --
# Ampere/Ada/Blackwell; see its CLOZN_BUILD_CUDA/CLOZN_BUILD_SAE block). Set CLOZN_CUDA_ARCH yourself
# (e.g. CLOZN_CUDA_ARCH=86 for a single Ampere card) if you want a faster, narrower build.
#
# `set -euo pipefail`, and every step below is explicitly `||`-guarded rather than relying on it alone:
# build_gpu.bat prints a final status line but never gates its own exit code on the build actually
# succeeding (see build_serve.sh for the same note) -- this script aborts loudly instead, on purpose.
#
# Note for macOS readers: the system /bin/bash there is 3.2 (Apple has never shipped a GPL3 bash), which
# has a known bug where `"${empty_array[@]}"` under `set -u` raises "unbound variable" even though the
# array itself IS set (just to zero elements). CUDA_ARCH_ARG below is a plain string, not an array, for
# exactly this reason -- do not "clean this up" into an array without checking bash --version on macOS.
#
# Prerequisite (not repeated here, same as build_serve.sh): `python engine/core/third_party/bootstrap_llama.py`.
set -euo pipefail
cd "$(dirname "$0")"

OS="$(uname -s)"
GGML_METAL=OFF
GGML_CUDA=OFF
CLOZN_BUILD_SAE=OFF
CUDA_ARCH_ARG=""

case "$OS" in
    Darwin)
        echo "=== macOS detected: using Metal ==="
        GGML_METAL=ON
        # CLOZN_BUILD_SAE needs CUDAToolkit (see CMakeLists.txt) -- there is no Metal equivalent, so it
        # stays OFF here. clozn-server still builds and serves; only the on-device SAE readout is absent.
        ;;
    Linux)
        if command -v nvcc >/dev/null 2>&1; then
            echo "=== Linux detected, nvcc found: using CUDA (+ SAE) ==="
            GGML_CUDA=ON
            CLOZN_BUILD_SAE=ON
            if [ -n "${CLOZN_CUDA_ARCH:-}" ]; then
                CUDA_ARCH_ARG="-DCMAKE_CUDA_ARCHITECTURES=$CLOZN_CUDA_ARCH"
            fi
        else
            echo "WARNING: Linux detected but no nvcc on PATH -- falling back to a CPU-only build." >&2
            echo "         Install the CUDA toolkit and re-run for GPU acceleration." >&2
        fi
        ;;
    *)
        echo "WARNING: unrecognized platform '$OS' -- falling back to a CPU-only build." >&2
        ;;
esac

echo "=== configure (GPU ggml+serve, GGML_METAL=$GGML_METAL GGML_CUDA=$GGML_CUDA CLOZN_BUILD_SAE=$CLOZN_BUILD_SAE) ==="
# $CUDA_ARCH_ARG is deliberately unquoted: empty, it word-splits away to nothing; set, it is always a
# single well-formed "-DFOO=bar" token with no embedded whitespace.
cmake -S . -B build-gpu -G Ninja -DCMAKE_BUILD_TYPE=Release \
    -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON \
    -DGGML_METAL=$GGML_METAL -DGGML_CUDA=$GGML_CUDA -DCLOZN_BUILD_SAE=$CLOZN_BUILD_SAE \
    $CUDA_ARCH_ARG || { echo CONFIGURE_FAILED; exit 2; }

echo "=== build ==="
cmake --build build-gpu || { echo BUILD_FAILED; exit 3; }
echo "=== BUILD_DONE ==="
