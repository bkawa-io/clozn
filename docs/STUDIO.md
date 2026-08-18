# Clozn Studio — inspect runs, evidence, and changes

Studio is the browser UI served by the same gateway that records Clozn runs. It does not load a second
model or maintain a separate evidence store.

## Run it

```bash
clozn serve MODEL --port 8080
clozn studio --open
```

Point OpenAI-compatible clients at `http://127.0.0.1:8080/v1`. Their completed requests enter the same
SQLite journal and become available in Studio.

## Current surfaces

- **Runs** lists recorded requests and opens their response, trace, identity, and lineage.
- **Minimal Context** is the primary run investigation: it reduces the declared Context Units under
  an explicit exact-output or teacher-forced-likelihood criterion, shows retained versus omitted units,
  exposes direct proof coverage, and offers explicit source-bound branch actions. `EXACT MINIMUM` is
  shown only when every smaller cardinality is directly ruled out; `BEST VERIFIED` and unmeasured space
  remain visibly incomplete.
- **Lens** shows delivered-context receipts, measured source links, token confidence, model readouts,
  and performance evidence. Its Performance view separates measured phases from overlapping/context-only
  spans, shows known versus unaccounted wall time, names the phase clock owner, and labels derived
  end-to-end throughput separately from worker-measured decode throughput. Missing or unmeasured data
  stays visibly unavailable.
- **Compare** aligns two stored runs and separates model/artifact identity, instructions, documents,
  history, settings, tools, and output changes.
- **Experiments** renders the case × variant × seed matrix, summaries, filters, and cell detail from
  versioned experiment results.
- **Behavior** exposes one-shot corrective retries and runtime sampling defaults. A retry previews a
  bounded action, confirms it to run a matched baseline/corrected comparison, and optionally keeps the
  corrected child as that run's own revision (with an explicit undo). Durable, auto-applying corrections
  ("Teach Once") were retired — nothing kept here shapes a later, unrelated request; see
  [CAPABILITIES.md](CAPABILITIES.md).
- **Model** reports the loaded worker and optional artifact state.
- **Scope** explores recorded token and layer evidence without upgrading post-hoc readouts to causal
  claims.
- **Snapshots** previews and pins durable checkpoints, lists manifest metadata, and requires an
  explicit confirmation before unpinning. Checkpoint bytes remain in the local blob store.
- **Answer Time Machine** reports per-run replay fidelity and which turns can branch through
  `GET /runs/<id>/time-machine`. The current branch action is explicitly structural transcript replay;
  retained KV snapshots do not become an exact-replay claim until a restore-and-verify path is enabled.
  The latest turn can now request an explicit prompt-boundary proof through
  `POST /runs/<id>/time-machine/verify`; earlier conversational turns can use an exact matching
  organic session run, while missing or ambiguous session history remains unavailable.
  The eligibility receipt preserves requested/source run IDs, and the card shows that provenance when
  an earlier turn was verified from its historical session run. If a durable checkpoint pin exists
  for that source, verification hydrates the pin after worker restart before running the same proof.
  POST /runs/<id>/time-machine/branch is the explicit same-prompt exact-child action; it restores
  the source checkpoint, runs the unchanged control, and persists a child only after that control
  matches. It rejects alternate questions because changing the prompt is not exact.

The run reader's default hierarchy is Answer → Minimal Context → Context. Compare/Branch is reached
from an explicit Minimal Context source action and opens the existing parent/child comparison surface.
Mechanism, claim, correction, time-machine, and other diagnostic instruments remain available under
the reader as advanced evidence; they are not alternate interpretations of source importance.

### Minimal Context failure states

The product keeps these conditions distinct: exact mode unavailable, no preserving set within budget,
best verified with incomplete coverage, certification budget exhausted, a search universe above its
declared bound, stale run/runtime identity, unavailable worker, and cancellation. Likelihood is never a
silent fallback for an exact request. Unmeasured candidates are not rendered as failed candidates.

The panel registry is additive: a failed optional panel is reported without taking down the other
surfaces.

## Evidence rules

- A context segment marked delivered or survived is not automatically influential.
- The Sources view labels measured, below-threshold, omitted, unavailable, and failed states
  independently; it never invents links for missing measurements.
- Token confidence describes commitment, not correctness.
- Run comparison reports what changed; only a controlled intervention can support a causal claim.
- Time Machine distinguishes structural replay from exact replay. Snapshot presence alone is not proof of
  exactness, and branch execution still requires a ready compatible worker. A successful prompt-boundary
  verification is an ephemeral worker-generation-scoped proof, not a durable checkpoint guarantee; a
  session-turn proof records the requested run separately from its historical source run.
- J-lens and other latent readouts carry method, artifact, and qualification provenance in the view.
- Provenance labels use “measured context dependence” language; they do not claim an answer literally
  came from a document. The raw `CONTEXT_CARRIED`/`MIXED`/`PARAMETRIC` enum remains available in the
  receipt alongside the attention-knockout method and control caveat.
- Performance diagnoses render only rules supported by recorded phases and metrics. A phase measured on
  the gateway monotonic clock is never offset-aligned with one measured on the worker steady clock;
  only non-overlapping measured durations contribute to the known-time total.

## Removed surfaces

Prompt-card and learned-prefix memory are not current Studio features. The former standalone PyTorch
workbench and diffusion UI were retired with that path. Tone dials, dial calibration, user preferences,
feedback signals, memory cards, and concept steering as a user-facing control were removed along with the
rest of the personalization layer; raw-vector steering (`steer_vec`, `POST /intervene`) is unaffected.
Old screenshots, design specs, and handoff notes remain useful historical records, but they are labeled
as archives and are not instructions for running the current product.
