# docs/ — architecture & technical

The design docs, indexed. Read top-down: the synthesis first, then the per-layer deep dives.

## Start here

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the synthesis: one product, the layers, the
  state-stream protocol, the interp maturity ladder. *How the whole thing fits.*
- **[CAPABILITIES.md](CAPABILITIES.md)** — the single current matrix for merged, released,
  qualified, and limited capabilities.
- **[CLOZN_FEATURE_MAP.md](CLOZN_FEATURE_MAP.md)** — the comprehensive code-derived map of product
  features, available information, actions, evidence types, and capability boundaries.
- **[STUDIO_RELEASE_CHECKLIST.md](STUDIO_RELEASE_CHECKLIST.md)** — the acceptance-gated Studio and
  backend release checklist.
- **[STUDIO_SURFACE_CONTRACT.md](STUDIO_SURFACE_CONTRACT.md)** — canonical Studio routes, evidence
  boundaries, and explicit-operation rules.
- **[STUDIO_LIVE_QUALIFICATION.md](STUDIO_LIVE_QUALIFICATION.md)** — the
  repeatable real-GGUF gateway smoke and hands-on Studio browser evidence.

## Per-layer design

- **[STUDIO.md](STUDIO.md)** — the studio UI: pages, panels, and what each surface shows.
- **[design/UX_FOUNDATION.md](design/UX_FOUNDATION.md)** — the capability-driven information
  architecture, workspace model, evidence hierarchy, and single-run investigation frame.
- **[MODEL_SUPPORT.md](MODEL_SUPPORT.md)** — which model families run, and on which paths. Use
  `clozn qualify MODEL --plan` for a model-free qualification readiness report; it does not qualify
  or install an artifact.
- **[MANAGED_MODELS.md](MANAGED_MODELS.md)** — the single- and multi-model runtime, exact identities,
  routing, preloading, cold loading, and verified-idle eviction.
- **[WORKSPACE_LENS.md](WORKSPACE_LENS.md)** — the J-lens: how it's fitted, what it can and
  cannot claim, and the trace fixture format.
- **[RUNTIME_SPLIT.md](RUNTIME_SPLIT.md)** — how the Python package splits between the pure
  library and the served runtime.

## Protocol

- **[../protocol/README.md](../protocol/README.md)** — the state-stream contract the engine
  emits and the studio consumes.

The four non-negotiable invariants (honesty-first, the seam, tests-as-oracle,
substrate-agnostic) hold across all of the above — see
[ARCHITECTURE.md](ARCHITECTURE.md).
