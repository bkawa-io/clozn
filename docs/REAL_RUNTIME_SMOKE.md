# Real-runtime acceptance

The production topology is not release-validated by a fake worker. The P0 gate builds the pinned C++
worker and runs both managed smoke modes against a real autoregressive GGUF on CPU. The smoke owns the
complete process tree, replaces the private worker behind the unchanged gateway, checks generation after
replacement, validates the public and native stream contracts, verifies SQLite and the trace-blob digest,
and reports whether every PID and listening port was cleaned up.

## Reference model

The manual/nightly CPU gate uses the official Qwen GGUF below. The model is downloaded at an immutable
repository revision and its bytes are verified before every run (or after restoration from cache).

| Field | Value |
|---|---|
| Source | `Qwen/Qwen2.5-0.5B-Instruct-GGUF` on Hugging Face |
| Revision | `df5bf01389a39c743ab467d734bf501681e041c5` |
| File | `qwen2.5-0.5b-instruct-q4_k_m.gguf` |
| License | Apache-2.0 |
| SHA-256 | `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db` |
| Size | 491,400,032 bytes |
| Quantization | Q4_K_M |
| Model family | Qwen2 autoregressive instruct model |
| GGUF contract | 24 layers, hidden dimension 896, context length 32,768 |
| Chat template | Embedded Qwen ChatML/Jinja template (`<|im_start|>...<|im_end|>`) |

Model page and license: <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF>

## Clean Linux CPU sequence

From a full checkout:

```bash
set -euo pipefail

REV=df5bf01389a39c743ab467d734bf501681e041c5
FILE=qwen2.5-0.5b-instruct-q4_k_m.gguf
SHA=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db
MODEL="$HOME/.clozn/models/$FILE"

mkdir -p "$(dirname "$MODEL")"
curl --fail --location --retry 3 \
  --output "$MODEL" \
  "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/$REV/$FILE?download=true"
echo "$SHA  $MODEL" | sha256sum --check

python engine/core/third_party/bootstrap_llama.py
cmake -S engine/core -B engine/core/build-serve -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCLOZN_BUILD_GGML=ON \
  -DCLOZN_BUILD_SERVE=ON \
  -DGGML_CUDA=OFF
cmake --build engine/core/build-serve --target clozn-server -j 2

python -m unittest -v tests.test_runtime_architecture tests.test_product_smoke
python -m clozn smoke "$MODEL" --cpu --json --timeout 300 --startup-timeout 240
python -m clozn smoke "$MODEL" --cpu --deep --json --timeout 300 --startup-timeout 240
python -m clozn ps
```

`--cpu` is strict: it refuses to run when only a GPU worker exists. A successful report must name a CPU
worker and end with `managed runtime cleanup` showing a clear registry, a stopped supervisor, no live
PIDs, and no open runtime ports.

## Automation

`.github/workflows/real-runtime-smoke.yml` runs the same sequence manually and nightly on Linux CPU. It
caches only the hash-pinned model, reconstructs the pinned and patched `llama.cpp`, uploads logs on
failure, and runs an unconditional residual-process check.

The model-free CI lane remains useful for fast feedback, but it is not evidence that the real runtime is
ready.
