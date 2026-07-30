# Development bring-up

The product runtime has one release gate: a real GGUF must pass `clozn smoke`. The command has a
non-mutating prerequisite mode, so start there:

```bash
python -m clozn smoke qwen --preflight
```

It reports all missing inputs together. A source checkout needs four things:

1. the Studio files from this repository;
2. the pinned and patched `llama.cpp` source;
3. a C++17 toolchain plus CMake 3.18 or newer; and
4. at least one GGUF known to `clozn models`.

The reproducible reference model, immutable download URL, SHA-256, license, clean Linux commands, and
manual/nightly workflow are documented in [`REAL_RUNTIME_SMOKE.md`](REAL_RUNTIME_SMOKE.md).

## Reconstruct and build the CPU worker

The upstream runtime is intentionally not copied into this repository. Reconstruct the exact pin and
apply Clozn's tracked patch:

```bash
python engine/core/third_party/bootstrap_llama.py
```

On Linux or macOS, `engine/core/build_serve.sh` performs a portable CPU build — the same commands as
below, just already scripted:

```bash
cmake -S engine/core -B engine/core/build-serve \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLOZN_BUILD_GGML=ON \
  -DCLOZN_BUILD_SERVE=ON \
  -DCLOZN_ENGINE_BUILD_ID=development \
  -DGGML_CUDA=OFF
cmake --build engine/core/build-serve --target clozn-server -j
```

For GPU acceleration on Linux/macOS, `engine/core/build_gpu.sh` picks the right backend by platform:
CUDA on Linux when `nvcc` is on `PATH` (falling back to CPU with a warning otherwise), Metal on macOS.
See [Platform support](../README.md#platform-support) for exactly what is and is not independently
verified on each platform — in short: Windows CPU/CUDA and Linux CPU are proven (this repo's own dev
box, and `real-runtime-smoke.yml`'s nightly Linux CI build, respectively); Linux CUDA and macOS/Metal
are scripted and unit-tested (`tests/test_env_with_dlls_platform.py`) but not build-verified from this
repository.

On Windows, `engine/core/build_serve.bat` performs the equivalent build. For CUDA, use
`engine/core/build_gpu.bat` or configure the same CMake target with `GGML_CUDA=ON` in a CUDA-capable
toolchain. `clozn` discovers `build-serve`, `build-gpu`, and the other supported build directories
automatically, on every platform — `find_engine()` in `clozn/cli/engine_process.py` only looks for a
`clozn-server`/`clozn-server.exe` binary under those names, not a platform-specific one.

`engine/core/build_core.sh` (CPU-only backend-free runtime + its unit tests, no CUDA/ggml/llama.cpp) and
`engine/core/build_cuda.sh` / `build_sae.sh` (CUDA kernel/SAE parity tests; require `nvcc`, no macOS
equivalent) mirror the remaining `.bat` scripts for completeness — see each script's own header comment
for what, if anything, diverges from its `.bat` counterpart.

Confirm the model-free embedded identity before loading a GGUF:

```bash
engine/core/build-serve/clozn-server --version --json
```

Published artifacts use a release-unique build ID and are checked against their release manifest before
`clozn setup` promotes them. See [NATIVE_DISTRIBUTION.md](NATIVE_DISTRIBUTION.md) for that contract and
the current release status.

For a nonstandard or test build, `CLOZN_ENGINE_BIN=/absolute/path/to/clozn-server` selects the worker
explicitly; set `CLOZN_ENGINE_GPU=1` when that binary should receive GPU offload flags. A packaged Studio
outside the repository can be selected with `CLOZN_STUDIO_DIR=/absolute/path/to/studio`.

The gateway accepts browser requests from loopback origins by default and rejects other browser origins
before route dispatch. Add trusted, exact origins with a comma-separated `CLOZN_ORIGINS` value;
`CLOZN_ORIGINS=*` is an explicit unsafe-development opt-in.
JSON bodies are capped at 8 MiB by default and can be adjusted with `CLOZN_MAX_REQUEST_BYTES`.
Stateful POST operations are serialized with up to 32 admitted requests; tune the bound with
`CLOZN_MAX_PENDING_REQUESTS` and the wait ceiling with `CLOZN_QUEUE_TIMEOUT` (seconds).

Put a GGUF in `~/.clozn/models`, point `CLOZN_MODELS` at its directory, or use `clozn pull`. Then run:

```bash
python -m clozn smoke qwen
python -m clozn smoke qwen --deep
```

Managed smoke chooses a free public port, launches the exact `clozn serve` command, checks the gateway,
restarts its private worker, and cleans up. To validate a gateway you started yourself without touching
its worker:

```bash
python -m clozn smoke --url http://127.0.0.1:8080
```

Use `--restart-worker` with attach mode only when you explicitly want the smoke command to terminate the
worker registered for that gateway.

## Required result

A release candidate is not runtime-validated until the managed smoke report passes with a real model.
At minimum it must prove:

- Studio and `/readyz` share the one public port;
- `/v1/completions` contains only standard completion chunks;
- `/api/clozn/generate` retains typed native events;
- chat produces a resolvable SQLite run and content-addressed trace;
- the worker PID changes while the gateway PID does not; and
- generation succeeds after that replacement.
