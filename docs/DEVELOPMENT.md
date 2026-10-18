# Development bring-up

## Build the worker

Bootstrap the pinned llama.cpp source and build the private worker:

```sh
python engine/core/third_party/bootstrap_llama.py
cmake -S engine/core -B engine/core/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLOZN_BUILD_GGML=ON \
  -DCLOZN_BUILD_SERVE=ON \
  -DCLOZN_ENGINE_BUILD_ID=development \
  -DGGML_METAL=ON
cmake --build engine/core/build --target clozn-server
ctest --test-dir engine/core/build --output-on-failure
```

Use `GGML_CUDA=ON` on a CUDA host or disable both GPU backends for CPU. The helper scripts
`engine/core/build_serve.*`, `build_gpu.*`, and `build_core.*` provide common configurations. Set
`CLOZN_ENGINE=/absolute/path/to/clozn-server` to select a nonstandard build.

## Start the product

```sh
export CLOZN_DATA_ROOT="$(mktemp -d)/clozn-v1"
python -m clozn serve /absolute/path/to/model.gguf --port 8080
```

In another terminal:

```sh
python -m clozn studio --port 8080 --open
python scripts/smoke/v1_acceptance.py --base-url http://127.0.0.1:8080
```

The acceptance harness creates real runs and analysis results in the selected v1 data root. Use an
isolated root for development so qualification does not mix with personal data.

## Test gates

```sh
.venv/bin/pytest -q tests
.venv/bin/python scripts/docs/check_drift.py .

cd studio-frontend
npm test -- --run
npm run build
```

For a fresh native verification, configure a new build directory instead of relying only on an existing
incremental build.

## Runtime constraints

- The installed Python product remains stdlib-only.
- The gateway and worker are loopback services; only the gateway port is public to clients.
- `CLOZN_ORIGINS` adds exact trusted browser origins; `*` is an explicit unsafe development opt-in.
- `CLOZN_MAX_REQUEST_BYTES` controls the JSON body cap.
- `CLOZN_MAX_PENDING_REQUESTS` and `CLOZN_QUEUE_TIMEOUT` bound admitted stateful requests.
- The v1 runtime supports one generation slot and one loaded model.

No public native binary matrix is currently claimed; see
[NATIVE_DISTRIBUTION.md](NATIVE_DISTRIBUTION.md).
