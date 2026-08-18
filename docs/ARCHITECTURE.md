# Architecture — Clozn's current product runtime

Clozn is one local product: an autoregressive GGUF worker behind a Torch-free gateway, with every
generation recorded as an inspectable run. The product is built to compare behavior, expose delivered
context and runtime evidence, apply bounded interventions, and fail closed when evidence or a qualified
artifact is missing.

The engine began as a diffusion-language-model project. That lineage explains some protocol vocabulary
and the archived design documents, but diffusion and the former PyTorch workbench are not current
product surfaces. The [capability matrix](CAPABILITIES.md) is authoritative for user-visible status.

## Layers and dependency direction

```text
clozn/ + studio/  →  protocol/  →  engine/
product/API/UI       shared seam    C++ runtime
```

- `clozn/` owns the CLI, OpenAI-compatible gateway, run journal, context receipts, comparisons,
  experiments, corrective actions, steering, readouts, and setup logic. The installed product remains
  stdlib-only.
- `studio/` is the UI served by the gateway. It reads the same run and evidence APIs as the CLI.
- `protocol/` defines the worker/gateway state-event and handshake contract.
- `engine/core/` owns GGUF loading, chat-template rendering, generation, sampling, scoring, activation
  taps, steering writes, and optional qualified readout application.

Product code never imports Torch or Transformers. Offline fitting, calibration, and research scripts
may use heavier dependencies, but they communicate with the product only through validated artifacts.

## Process model

`clozn serve MODEL --port 8080` supervises exactly two processes:

1. a public loopback Python gateway on the requested port; and
2. a private C++ model worker on a random loopback port.

The gateway is the only public URL. It owns API compatibility, Studio, journaling, receipts, run
comparison, and action orchestration. The worker is replaceable: the gateway negotiates protocol
compatibility, reports the exact worker identity, and restarts it after an unexpected exit.

State-changing POST operations pass through a bounded serialized gate because generation evidence and
steering state are not yet safe for unrestricted concurrent mutation. Health, static Studio assets, and
read-only run inspection remain concurrent.

## Public and native APIs

- `/v1/models` and `/v1/chat/completions` form the strict client-facing OpenAI subset documented in
  [OPENAI_COMPATIBILITY.md](OPENAI_COMPATIBILITY.md).
- `/api/clozn/generate` carries Clozn's typed native state stream. Native event frames never leak into
  an OpenAI completion stream.
- `/runs/*`, `/experiment-results/*`, and other namespaced routes expose recorded evidence and derived
  views. A derived view does not upgrade the evidence in its source run.
- Raw vector steering (`steer_vec`) is an engine-level primitive reached through `/intervene`,
  execution-fork steer arms, and the receipts/analysis machinery, not a dedicated product route. Raw
  engine inspection remains separate under `/engine/*`.

## Runs, identity, and persistence

SQLite is authoritative for queryable run metadata, lineage, status, and the complete run document.
Large normalized traces are immutable SHA-256-addressed blobs. Schema migrations run transactionally
whenever the store opens; there is no user migration command.

A run records the messages delivered to the renderer, exact rendered prompt when available, response,
trace, finish or failure state, behavior settings, model/engine identity, and additive identity facets.
Current runs do not apply or record prompt-card memory. Readers can still tolerate old optional fields
for compatibility, but historical shape is not current capability.

Every persisted artifact gets a versioned schema before it is written. Unknown values are omitted, not
null-padded, and optional identity providers may lose only their own namespace when unavailable.

## Evidence paths

The engine provides measurements; the Python layer composes them into bounded claims:

| Evidence | Source | Claim boundary |
|---|---|---|
| Token confidence and alternatives | native generation trace | commitment, not correctness |
| Delivered context receipt | recorded pre-template messages and truncation metadata | what was delivered/survived, not what caused output |
| Teacher-forced score differences | `/score` interventions | local effect under the stated counterfactual |
| Source influence map | bounded source ablations and thresholds | measured spans only; omissions remain visible |
| Run comparison | stored run identity/context/settings/output | structural difference, not causal attribution |
| Performance diagnosis | versioned gateway/worker monotonic spans, explicit clock owners, known/unaccounted aggregation, and explicit rules | a cause only when the rule's evidence is present; never subtract unrelated process clocks or double-count overlapping spans |
| J-lens | qualified fitted artifact applied to engine activations | disposition readout, not literal thought or causal proof |

## Product acceptance gate

`clozn smoke MODEL` launches the real supervised stack, checks Studio and the public/native protocol surfaces,
verifies journaling and trace integrity, replaces the private worker behind the stable gateway, proves
generation still works, and cleans up the complete process tree. `--deep` adds forced scoring/receipt
paths; `--preflight` reports prerequisites without starting anything.

Unit tests may replace the worker, but they do not substitute for a successful real-GGUF smoke gate.
Likewise, merged installer code is not a released install path until public archives and their manifest
pass the clean-machine release lanes.

## Hard invariants

1. **Evidence before narration.** Observed, eliminated, reproduced, correlated, and causally supported
   are distinct states.
2. **No silent fallback.** Unsupported capabilities fail or return an explicit unavailable state.
3. **Exact identity travels with comparisons.** Model, template, engine, adapter, and artifact
   differences must not disappear behind a friendly label.
4. **Model-free by default.** Contracts, schemas, planners, renderers, and failure paths are testable
   without a model or network.
5. **Qualification is scoped.** Core support never implies that a model-specific lens, SAE, or
   structured-I/O artifact is qualified.
