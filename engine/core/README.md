# Clozn native engine

This directory contains Clozn's private v1 GGUF worker. It uses the pinned llama.cpp source under
`third_party/` and supports one autoregressive model per worker.

The retained native responsibilities are deliberately narrow:

- text generation, streaming, and cancellation;
- scalar teacher-forced scoring of a recorded continuation;
- model-owned chat-template rendering;
- exact recorded-reference matching, including a worker-local persistent parent session; and
- truthful model, tokenizer, engine, backend, and performance identity.

Research routes, diffusion generation, state editing, user checkpoints, J-lens, SAE readouts,
adapter loading, and standalone research tools are not part of the v1 worker.

## Build

Bootstrap the pinned llama.cpp checkout once, then configure the worker:

```sh
python engine/core/third_party/bootstrap_llama.py
cmake -S engine/core -B engine/core/build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLOZN_BUILD_GGML=ON \
  -DCLOZN_BUILD_SERVE=ON \
  -DGGML_METAL=ON
cmake --build engine/core/build --target clozn-server
ctest --test-dir engine/core/build --output-on-failure
```

Use `-DGGML_CUDA=ON` instead of Metal on a CUDA host, or disable both for CPU. The helper scripts
`build_serve.*`, `build_gpu.*`, and `build_core.*` provide the corresponding common configurations.

`clozn-server` is private to the Python gateway. See [serve/README.md](serve/README.md) for its
worker-only HTTP contract.
