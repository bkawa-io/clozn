# Clozn v1 capability boundary

This page is the current user-facing capability summary. “Implemented” means present in this source
tree with model-free coverage; it does not imply that a packaged native release exists.

| Area | v1 status | Boundary |
|---|---|---|
| Local generation | Implemented | One GGUF, one private worker, text streaming, cancellation, durable recording |
| Studio | Implemented | Run list/composer and one workspace with linked Context Effects, pane heatmaps, Efficiency, Lean Context, and Minimal Context |
| Context Effects | Implemented | Recorded-answer scoring under source removal; measured ranges only; influence is not correctness |
| Lean Context | Implemented | Best context found within fixed probability-loss tolerances and at most 16 new executions; compatible measurements are reused |
| Minimal Context | Implemented | Bounded search with exact answer and stop verification; durable result and idempotent child save |
| Comparison | Implemented | Original versus its saved reduced child only |
| Run storage | Implemented | SQLite metadata, content-addressed traces, isolated v1 root, read/list/delete |
| Text API | Implemented | `/v1/models` and text-only `/v1/chat/completions` |
| Worker internals | Implemented | generation, template rendering, scalar score, exact-reference matching, cancellation |
| Native packaged install | Not released | Build from source; no public artifact matrix is claimed |
| Browser live qualification | Passed | Exact-reduction gate and long-conversation Context Effects/Lean Context surfaces passed on the documented local configurations |

Not in v1:

- tools or structured output;
- Responses or Ollama-compatible endpoints;
- multiple loaded models, preload/eviction, or model switching;
- general run comparison, session navigation, export workbenches, or Model CI;
- claims, diagnosis, corrective retries, guards, calibrated trust, or steering;
- token workbenches, branching, snapshots, checkpoints, or time travel;
- mechanistic tracing, J-lens, SAE readouts, state editing, or a research SDK.

Unsupported public request fields fail visibly. Removed source files do not become public merely by
appearing in a package directory: command and route manifests are explicit and covered by boundary tests.

The configured context-arm cap is exact, not lossy. Clozn can merge adjacent spans inside one message,
but it does not merge text across chat-message boundaries. If the number of non-empty messages alone
exceeds the cap, bounded Lean/Minimal Context search reports that condition instead of dropping input.
Direct Context Effects remains available, but its latency grows with the number and length of sources.
