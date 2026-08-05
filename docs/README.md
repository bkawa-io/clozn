# docs/ — architecture & technical

The design docs, indexed. Read top-down: the synthesis first, then the per-layer deep dives.

## Start here

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — the synthesis: one product, the layers, the
  state-stream protocol, the interp maturity ladder. *How the whole thing fits.*
- **[CAPABILITIES.md](CAPABILITIES.md)** — the single current matrix for merged, released,
  qualified, and limited capabilities.
- **[CLOZN_FEATURE_MAP.md](CLOZN_FEATURE_MAP.md)** — the comprehensive code-derived map of product
  features, available information, actions, evidence types, and capability boundaries.
- **[ROADMAP.md](ROADMAP.md)** — the consolidated map: what's done, the v1 cut, what's next.

## Per-layer design

- **[DESIGN.md](DESIGN.md)** — historical diffusion-engine design. It explains old scheduler and
  kernel references but is not a current-user product contract.
- **[TECHNICAL.md](TECHNICAL.md)** — archived diffusion research measurements and engineering notes.
- **[STUDIO.md](STUDIO.md)** — the studio UI: pages, panels, and what each surface shows.
- **[design/UX_FOUNDATION.md](design/UX_FOUNDATION.md)** — the capability-driven information
  architecture, workspace model, evidence hierarchy, and single-run investigation frame.
- **[MODEL_SUPPORT.md](MODEL_SUPPORT.md)** — which model families run, and on which paths. Use
  `clozn qualify MODEL --plan` for a model-free qualification readiness report; it does not qualify
  or install an artifact.
- **[MANAGED_MODELS.md](MANAGED_MODELS.md)** — the managed multi-model runtime: a copyable
  `clozn.managed-models.v1` manifest, qualification/identity rules, `clozn serve --models-config`, and
  its current preloaded-only limitations.
- **[WORKSPACE_LENS.md](WORKSPACE_LENS.md)** — the J-lens: how it's fitted, what it can and
  cannot claim, and the trace fixture format.
- **[EXPLAIN_THIS_ANSWER_SPEC.md](EXPLAIN_THIS_ANSWER_SPEC.md)** — the explain/receipts spec
  (M1 assembly, causal receipts, the honesty rules the endpoints enforce).
- **[RUNTIME_SPLIT.md](RUNTIME_SPLIT.md)** — how the Python package splits between the pure
  library and the served runtime.

## Studio UI review

- **[STUDIO_UI_AUDIT_2026-08-04.md](STUDIO_UI_AUDIT_2026-08-04.md)** — hands-on audit of every
  primary Studio workspace, Run section, Behavior module, and reachable compatibility surface.
- **[STUDIO_UI_REDESIGN_2026-08-04.md](STUDIO_UI_REDESIGN_2026-08-04.md)** — proposed five-workspace
  information architecture, page consolidation, layouts, visualization choices, and delivery sequence.

## Protocol

- **[../protocol/README.md](../protocol/README.md)** — the state-stream contract the engine
  emits and the studio consumes.

The four non-negotiable invariants (honesty-first, the seam, tests-as-oracle,
substrate-agnostic) hold across all of the above — see
[ARCHITECTURE.md](ARCHITECTURE.md).
