# clozn-server — the white-box local-LM runtime

A local HTTP server with white-box read **and** write: read named concepts + token candidates off
each position live; steer concepts into the residual stream. Two generation paradigms, one harness —
the interpretability is model-agnostic (it sits on a hidden state, not on the decoder):

- **Diffusion** dLLMs (LLaDA-8B / Dream-family) — watch the model *denoise*; plus the uniquely
  diffusion views: in-place revise ("change its mind"), parallel pre-commit confidence, infill.
- **Autoregressive** LLMs (Llama / Qwen / Mistral / ... — the whole llama.cpp zoo) — watch it
  generate *left-to-right*; the same concept reads + logit-lens + steering, per token.

**Mode follows the model** (a diffusion GGUF carries a mask token; an AR one doesn't), reported by
`/health` as `"mode"`. The diffusion-only endpoints (`/v1/infill`, `/v1/revise`, `/v1/board`) return
400 on an AR model.

## Build (Windows)

**GPU (recommended):**
```
core\build_gpu.bat          # vcvars64 + cmake -G Ninja -DGGML_CUDA=ON + ggml/serve
```
→ `build-gpu\clozn-server.exe` (+ `ggml-cuda.dll` and friends in `build-gpu\bin\`).

**CPU only:**
```
core\build_serve.bat        # vcvars64 + cmake -G Ninja -DCLOZN_BUILD_GGML=ON -DCLOZN_BUILD_SERVE=ON
```
→ `build-serve\clozn-server.exe` (+ its `ggml`/`llama` DLLs in `build-serve\bin\`).

## Get a model (GGUF)

LLaDA-8B is natively supported by the vendored llama.cpp (`conversion/llada.py`). Convert a cached
HF checkpoint:
```
.venv\Scripts\python core\third_party\llama.cpp\convert_hf_to_gguf.py <llada-snapshot-dir> ^
    --outfile core\models\LLaDA-8B-Instruct-q8_0.gguf --outtype q8_0
```
(The converter sets `diffusion.shift_logits=false` + non-causal attention; mask token is **126336**.)

## Run

**GPU** (LLaDA-8B Q8_0, all layers offloaded — ~333 ms/short completion, ~20 tok/s):
```
set PATH=%CD%\build-gpu\bin;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\x64;%PATH%
build-gpu\clozn-server.exe core\models\LLaDA-8B-Instruct-q8_0.gguf ^
    --mask-token 126336 --eos 126081 --port 8080 --gpu-layers 99
```

**CPU** (~10 s/short completion — proves the stack):
```
set PATH=%CD%\build-serve\bin;%PATH%
build-serve\clozn-server.exe core\models\LLaDA-8B-Instruct-q8_0.gguf ^
    --mask-token 126336 --eos 126081 --port 8080
```
Open **http://127.0.0.1:8080/** — type a prompt, hit generate, watch it denoise.

**Autoregressive** (any AR GGUF — no `--mask-token`; mode auto-detects, EOS from the vocab):
```
build-gpu\clozn-server.exe core\models\Qwen2.5-0.5B-Instruct-q8_0.gguf --gpu-layers 99 --port 8080
```
Convert one first via the vendored converter, e.g. Qwen (native Qwen2 support):
`.venv\Scripts\python core\third_party\llama.cpp\convert_hf_to_gguf.py <qwen-snapshot> --outfile core\models\Qwen2.5-0.5B-Instruct-q8_0.gguf --outtype q8_0`.
The viz switches to a left-to-right token stream; `clozn-ar <model.gguf>` is the AR CLI (per-token lens + read).

## Endpoints

| | |
|---|---|
| `GET  /`              | the real-time denoise visualization (a pure SSE consumer) |
| `GET  /health`        | `{status, model}` |
| `POST /v1/completions`| `{prompt, max_tokens, steps, block_len, topk, temperature, stream, features, steer}` |
| `POST /v1/infill`     | `{prefix, suffix, gap, ..., features, steer}` — fill the middle (native dLLM infilling) |
| `POST /v1/revise`     | `{text, spans:[[start,end]], grow, ..., features, steer}` — re-mask + re-predict a span in place |
| `POST /v1/board`      | `{board:[ids], steps, ..., features, steer}` — restore/branch a raw board (a `mask_token` id = a hole) |

Every non-stream response carries `board` + `layout` (per-position `{pos,id,masked,piece}`) — the
white-box **snapshot**: save it, edit a slot (set it to the mask id to re-open), POST to `/v1/board`.

## White-box (set `"features": true` to turn the read taps on)

- **concepts** — per-slot category probes, training-free diff-in-means directions in the model's
  *own* mid-layer activation space (tapped via `cb_eval`, no llama patch). Streamed as `step_features`
  events; the viz underlines each slot by its concept. 6 concepts:
  - Per-token: `punct` / `number` / `function` / `content`
  - Contrastive (sentence-level): `code` / `question`
- **logit-lens** — per-slot top-k token candidates (`step_lens` events, decoded to pieces server-side;
  hover a slot in the viz).
- **steering (write)** — `"steer": {"concept":"number","coef":20}` pushes a concept into the residual
  stream (a llama control vector at a mid-depth *steer* tap, `2/3 · n_layer`). Steering uses a **second**
  probe set calibrated at that depth, not the sharp early **read** tap — a diff-in-means direction only
  steers in the layer it was calibrated in (the residual basis rotates with depth). Override the band with
  `"layer": N`. *Slippery*: high `coef` garbles the output — the literature's non-surjective caveat, not a bug.

## Gotchas

- The exe needs its `bin\` dir (ggml/llama DLLs) on `PATH`. GPU build also needs the
  CUDA runtime DLLs at `CUDA\v13.3\bin\x64` (not plain `bin`).
- Concept probes calibrate once at startup (a few CPU forwards) — first boot is a touch slower.
- `cb_eval` mid-layer tap may slightly slow *all* decodes (the graph fuses less when a callback is set).
- llama control vectors have **no tensor for the last layer** (`llama-adapter.cpp`) — steer layers
  `1 .. n_layer-1` only; steering the final layer is a silent no-op.
- CPU build: 8B at Q8_0 is unhurried (~10 s for a short completion). The GPU path (`--gpu-layers`)
  is the speed lane; the CPU build proves the stack end-to-end.
