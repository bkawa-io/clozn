# Architecture — Clozn, the unified runtime

**One product: view and steer a model's evolving internal state — its memory — on the models
you run yourself.** Watch it think; read named concepts and token candidates off each
position; steer concepts into the residual stream; carry memory as readable cards; and hold
answers to account with causal receipts. Ollama's structural opposite: not a black box you
prompt, a glass box you inspect.

*(Naming: `clozn` = `cloze` — the engine inside — + "cozen", to deceive: the illusion it
reveals. The engine began life as a standalone diffusion-LM runtime; diffusion was the birth
story, not the identity — it's one substrate the engine runs.)*

## The layers (one repo, strict downward dependencies)

```
clozn/ + studio/   →   protocol/   →   engine/
  (the product)        (the seam)      (the runtime = "clozn")
```

- **`clozn/`** — the Python product package: the OpenAI-compatible serving layer, the run
  journal, receipts/replay/experiments, memory (cards, anchored bags, facts), behavior
  (tone dials, steering), readouts, eval, and the CLI (`clozn run/serve/studio/trace/...`).
  Owns *all* product surface. Drives the engine over HTTP; delegates hot paths down (it
  never does heavy compute itself).
- **`studio/`** — the white-box UI: the app shell + Run Inspector pages and the deep
  glass-box surfaces (brain, denoise, runtime, J-lens), served by the backend. No build
  step.
- **`protocol/`** — **the keystone.** One state-stream vocabulary shared by the engine and
  the studio (see below). This is what makes it *one* system instead of two inspectors.
- **`engine/`** — the runtime: the C++/ggml core (`engine/core/`) that runs real models
  (GGUF, CUDA or CPU), **emits the state-stream, applies steers, and hosts the interp
  primitives that must scale** — `/harvest`, `/score`, `/jlens`, steer taps; plus the
  Python reference scheduler + adapters + golden fixtures (`engine/lab/`) — its correctness
  oracle and CPU/iteration path — and GPU kernels (`engine/kernels/`: confidence-select
  today, interp kernels next), validated against CPU references. `engine/client/` is the
  thin Python SDK the product package drives it with.

**The rule that fixes drift:** the engine never owns product opinions (no competing
inspector); the product package owns all view/steer/memory surface; they meet only at the
protocol. The engine's C++ viz survives as a *reference* view; the canonical UI is the
studio, consuming the same stream.

## Process model

`clozn serve MODEL --port 8080` owns exactly two processes:

1. a public, Torch-free Python gateway on the requested port; and
2. a C++ model worker on a random loopback-only port.

The worker is an implementation detail, not another serving product. The gateway is the only public
URL and owns OpenAI compatibility, Studio, memory, receipts, runs, and readouts. It receives the worker
port through `CLOZN_ENGINE_PORT`; there is no second engine variable or selectable product substrate.
The CLI supervisor starts the worker first, waits for health, starts the gateway, waits for `/readyz`,
and restarts the worker after an unexpected exit. `clozn studio` only attaches to this gateway.
POST operations pass through a bounded serialized queue because generation evidence, memory, and steering
are currently shared runtime state; health, Studio assets, and GET-only run inspection remain concurrent.

PyTorch is a lab dependency for training and calibration. Lab jobs write forward-only artifacts that the
product can apply. The retained Qwen/Dream visual workbench is launched explicitly with `clozn lab`; its
HTTP process rejects all `/v1/*` and `/api/clozn/*` routes, so it is not a competing chat API.

## Public and native APIs

- `/v1/models`, `/v1/chat/completions`, and `/v1/completions` are the client-facing OpenAI subset; the
  exact endpoint/field contract is [OPENAI_COMPATIBILITY.md](OPENAI_COMPATIBILITY.md).
- `/api/clozn/generate` is the namespaced native stream consumed by Clozn tooling. It carries typed
  state events such as token commits and lens frames.
- The gateway translates worker events for `/v1/completions`; native event frames never appear on
  `/v1/*`. Opt-in chat extensions such as live lens data stay inside a standard
  `chat.completion.chunk` envelope.
- `/steer/*` is the canonical product-owned tone-dial surface: it carries live values, calibration,
  custom/library axes, and persistence. `/engine/steer/*` reaches the same `EngineSteer` in the product
  process but remains only as a deprecated compatibility facade for pre-heavn clients;
  new UI and clients must not use it. Raw runtime inspection stays distinct under `/engine/harvest`,
  `/engine/layers`, and `/engine/observe`.

## Persistence

SQLite is authoritative for queryable run metadata, lineage, status, and the complete run document.
Large normalized traces live as immutable SHA-256-addressed JSON blobs. The old per-run JSON journal has
an explicit one-shot importer (`clozn migrate-runs`) and is never read or dual-written during normal use.

## Product acceptance gate

`clozn smoke MODEL` launches the real `clozn serve` command on a free port and treats the complete
process tree as a black box. It verifies liveness/readiness, Studio delivery, model discovery, OpenAI
chat, the normalized OpenAI chat and completion streams, the native Clozn event stream, HTTP run lookup, run
explanation, the SQLite row, and its SHA-256 trace blob. It then terminates only the registered private
worker, waits for a replacement PID behind the unchanged gateway PID, proves generation still works,
and stops the stack. `--deep` adds forced receipts and replay; `--preflight` reports missing build,
model, vendored-source, compiler, CMake, and Studio-asset prerequisites without starting anything.

This is the release boundary test for the topology. Unit tests may replace the worker, but they cannot
replace a successful smoke run against a real GGUF.

## The keystone — `protocol/` (the state-stream)

- **`StateStep`** — one frame of the model's evolving state: `{t, substrate, slots, values,
  kind}`. A diffusion *board pass* and an AR *token + residual* are both `StateStep`s,
  differing only in `substrate`.
- **`Intervention`** — the write-back: `{target (layer/slot), vector, coef}`. Steering and
  edits flow up this channel.
- **`StateSource`** — anything producing the stream: the engine (any substrate over
  HTTP/SSE) or a lightweight Python source.
- **Memory ops** — snapshot / restore / persist / associate operate on accumulated
  `StateStep`s. *That is "the model's memory, made legible and editable."*

One spine, every substrate. "View" = read the stream; "steer" = push an `Intervention`;
"memory" = persist and recall the state.

## Where the depth lives — the interp maturity ladder

The heavy primitives belong DOWN in the engine; the Python side is orchestration + method
library + UX.

| Capability | Status | Home |
|---|---|---|
| Activation tap (mid-layer), logit-lens, per-token trace | **real** | engine (C++) |
| Teacher-forced `/score` (the receipts + verification keystone) | **real** | engine |
| J-lens ("disposed to say", per token/layer) — fit offline, apply forward | **real** | fit in lab (autograd) → served by the engine |
| Training-free concept probes + steering (tone dials, concept dirs) | **real** | engine + product |
| Causal receipts (leave-one-out over memory/context/dials) | **real** | product, over engine `/score` |
| SAE feature readout at the tap | **real** (per-model dictionaries) | engine |
| Confidence-select top-k kernel, zero-copy device logits | **real** (aimed at diffusion sampling) | kernels / engine |
| Scaled transcoder inference; pretrained dictionaries as plug-ins | **unbuilt** | engine + interp kernels |
| Fast-weights / in-model editable memory (surprise-gated) | **unbuilt — the frontier** | engine-deep + kernels |
| Circuit / attribution-graph tracing | **unbuilt** | product method over engine taps |

Honest caveat: much of "interp at scale" is GPU-batched cuBLAS in the engine, not exotic
kernels — the genuinely kernel-novel frontier is fast-weights (per-token weight deltas
during inference) and very-high-feature-count sparse top-k.

## Substrates & "memory"

The unifying object is the model's in-flight state; "memory" is its sharpest framing.

- **Autoregressive (Llama / Qwen / Mistral / Gemma, any GGUF)** — residual stream + KV cache
  as working memory. The daily driver: the whole white-box stack runs here, model-agnostic.
- **Diffusion (LLaDA / Dream)** — the denoising board as a state canvas; the engine's
  original substrate and the denoise visualization.
- **Recurrent (RWKV-class)** — an explicit fixed-size state vector, the cleanest "memory"
  object; explored in the research lineage, not in the shipped product today.

Persistence turns a captured state into a *cross-session* memory — the bridge from "inspect
activations" to "a model with a legible, editable memory."

## Invariants (non-negotiable)

1. **Honesty-first.** Every speed number ships with its quality column; every "readable"
   claim (a probe decodes X) ships with the causal test (steering X moves behavior) and its
   dose-response. Decodability ≠ use.
2. **The model/seam boundary.** Model backends live only behind the adapter seam; the
   scheduler and product logic are pure against it.
3. **Tests are the oracle.** Goldens pin (prompt, seed, policy) → exact picks; the same
   fixtures validate the engine and the lab.
4. **Substrate-agnostic by construction.** New model families are new `StateSource`s, not
   forks of the product.
