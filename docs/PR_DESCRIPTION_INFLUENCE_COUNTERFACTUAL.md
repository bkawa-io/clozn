# Influence → Counterfactual Confirmation

This backend-only change adds a read-only plan and an explicit execution route for testing one
persisted source-span → answer-span influence link under controlled free generation.

- The persisted Influence Map remains authoritative for measured effect, delta, floor, and evidence state.
- The counterfactual is additional free-generation evidence; it never replaces or reinterprets the measurement.
- `neutralize` reuses `clozn.matched_length_neutral_filler.v1`; `remove` is explicitly a different intervention.
- Planning performs zero model, worker, scoring, influence, checkpoint, or filesystem work.
- Execution resolves the immutable parent model/runtime, regenerates sibling control/treatment arms through
  the existing replay path, optionally adds one deterministic same-length specificity-control arm, and uses
  the existing model diff / First Divergence projections.
- There is no experiment object, counterfactual store, ambient configuration mutation, arbitrary replacement,
  or automatic source pruning.
- Responses are metadata-oriented: stable span IDs, measured values, child run IDs, decode/runtime labels,
  and compact comparisons; no raw prompt, source, answer, or filler text is returned.
- Universal Test This now dispatches `context_span` selections to this feature without duplicating span
  surgery or generation logic.
- Existing specialized APIs and evidence artifacts remain unchanged; no Studio frontend files were modified.

Focused tests cover measurement binding, fresh span resolution and drift refusal, intervention semantics,
controlled arms, direct-child lineage, deterministic IDs, metadata privacy, route status behavior, and the
read-only/no-worker planning boundary. The backend test suite passes.
