# Clozn v1 capability boundary

This page is the user-facing boundary for `0.1.0`.

| Area | v1 status | Boundary |
|---|---|---|
| Local generation | Included | One GGUF, one private worker, text streaming, cancellation, durable recording |
| Studio | Included | Conversation index, transcript reader, linked Context Effects, pane heatmaps, Lean Context, and an optional budgeted Stability test |
| Context Effects | Included | Recorded-answer scoring under source removal; compatible cached joint removals surface as non-additivity evidence; influence is not correctness and non-additivity is not redundancy |
| Lean Context | Included | Best context found within fixed probability-loss tolerances and a configurable 1–64 execution budget (default 16); fresh full/reduced answers are compared |
| Run storage | Included | SQLite metadata, content-addressed traces, isolated v1 root, read/list/delete |
| Text API | Included | `/v1/models` and text-only `/v1/chat/completions` |
| Recorded demo | Included | Static, interactive evidence bundle; no model download or inference |
| Native release | Included | Tagged CPU/Metal worker archives plus Python sdist/wheel |

Not in v1:

- tools, multimodal input, structured output, Responses, or Ollama-compatible endpoints;
- multiple loaded models, preload/eviction, or model switching;
- exact-minimum context proofs, saved reduced child runs, or general run comparison;
- claims, diagnosis, corrective retries, guards, calibrated trust, or steering;
- token workbenches, branching, snapshots, checkpoints, or time travel;
- mechanistic tracing, J-lens, SAE readouts, state editing, or a research SDK.

Unsupported public request fields fail visibly. Command and route manifests are explicit and covered
by boundary tests.

The 64-unit context ceiling is an exact safety bound, not lossy compression, and is separate from Lean
Context's model-execution budget. Clozn may merge adjacent spans inside one message, but never merges
text across message boundaries. If the exact message-scoped universe exceeds the ceiling, bounded
search reports that condition instead of dropping context.
