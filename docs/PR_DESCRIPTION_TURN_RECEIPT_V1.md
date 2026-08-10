# Turn Receipt v1 + Ambient Footer Signal Policy

Turn Receipt is Clozn's everyday read-side summary: a compact projection for ordinary local-model chat,
with Studio remaining the escalation surface for inspection, rewind, and forks.

The existing evidence artifacts remain authoritative.  This change composes Context Receipt, Context
Utilization, Context Tension, run identity, timing, hard signals, First Divergence, and Rewind Fidelity;
it does not replace or rename any of them.

Reading a Turn Receipt triggers zero new analysis: no model call, token scoring, influence measurement,
generation, worker start/wake, checkpoint creation, live rewind, or live execution-fork planning.

The footer consumes the same structured signal list as the Turn Receipt.  It now supports `off`,
`exceptions`, and `always`; the existing `receipt_footer: true` behavior remains `exceptions` when no
mode is configured.  `always` emits at least the receipt link, while exception phrases are capped at two
and use code-owned, injection-safe templates.

Context-window occupancy is a literal `prompt_tokens / context_window_tokens` runtime quantity.  It is
not context utilization and is never described as a percentage of context used.  Influence coverage is
explicit: measured sources, below-measured-floor sources, and not-measured sources are kept distinct;
partial coverage never implies low effect for the unmeasured remainder.

Performance is raw recorded timing and throughput.  No generic slow-generation warning is introduced
without a defensible local baseline for comparable model/runtime runs.

Turn Receipt privacy remains metadata-only by construction.  It does not restore labels when the Context
Receipt is `hashes_only`, acknowledges unavailable provenance when the receipt is `off`, and contains no
full prompt or response text.

The existing `/r/<id>` behavior, receipt card, export routes, and Studio frontend remain untouched.
Tests cover JSON/Markdown receipt composition, deterministic read-only behavior, privacy, partial
measurement, footer modes and stripping, and no-model/no-worker/no-scoring/no-checkpoint execution.
No Studio frontend files were changed by this feature branch.
