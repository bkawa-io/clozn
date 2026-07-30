# Feature delivery scope — 2026-07-29

Repository baseline: `main` at `99e85d8`.

This is an implementation handoff, not a replacement for `PRODUCT_ROADMAP.md`. It records what the
current repository already provides, the narrow gaps between those pieces and a usable product path,
and the dependency order for coding-agent tickets.

## Boundaries that every ticket keeps

- The public gateway and CLI remain stdlib-only Python. Heavy qualification dependencies stay outside
  the product package.
- C++ GGUF workers remain private loopback processes; the gateway is the only public endpoint.
- A request never silently falls back to another model or reuses evidence from another runtime
  identity.
- Runs remain immutable. Forks, repairs, experiments, and replays create children with lineage.
- Delivery, measured effect, below-floor results, unavailable evidence, and failure remain distinct.
  A UI display mapping may unify presentation, but must retain the artifact-native method and state.
- Structural differences do not become causal explanations without an applicable intervention.

## Current reality and first usable gap

| Workstream | Already present | Foundation added in these waves | First remaining product gap |
|---|---|---|---|
| Exact execution fork | Bit-exact C++ `/v1/execution-fork`; Python client call; legacy gateway text-splice fork | Restart-safe checkpoint identity, fail-closed planning, exact-only plan/execute/result routes, mandatory unchanged control, and immutable child/terminal receipts | Create a public checkpoint reference from an eligible recorded parent, then make the legacy route a compatibility wrapper and replace the Studio text splice |
| Mechanistic comparison | Pair compatibility, transplant, mechanistic diff, restoration metrics, causal bisect | Per-site deterministic random-control seeds; old shared-seed debt closed for new artifacts | Resolve a failed experiment/diff cell into one target, then expose a CLI path before adding async HTTP jobs |
| Evidence investigation | Context receipts, stored influence, diagnosis, run diff, corrective registry | Read-only investigation synthesis plus metadata-only `GET /runs/<id>/span-addresses`, with persisted influence validation and stable privacy-safe IDs | Consume the investigation/span APIs in the first Studio receipt panel |
| Multi-model runtime | One supervised gateway/worker and one global state-changing gate | Reusable worker lifecycle, exact preloaded registry, and request-scoped routing/receipts across native, OpenAI, and Ollama adapters | Make managed `clozn serve` construct and transport the registry/router configuration; cold loading follows only after that bootstrap is proven |
| Scope workbench | React Scope/Observatory views and explicit fork action | Vitest/RTL harness; canonical view/token/reference/layer URL state | Exact-fork status and receipt UI after public checkpoint capture exists |
| Sessions/history | Session ID is lookup metadata and runs already contain most trace facts | None | First-class session persistence and an ordered, paginated run query |
| Claim verification | Source influence and answer text are persisted | Investigation advertises support measurement honestly | Deterministic claim spans, then supplied-source mapping; do not start with a truth verdict |
| Qualification | Model support, batteries, and individual lab scripts exist | None | Qualification manifest/lifecycle schema and a model-free `--plan` command |

## Completed agent-sized slices

### `EV-00` — run investigation synthesis

- Added `clozn.run-investigation.v1`.
- Added read-only `GET /runs/<id>/investigation`.
- Composes recorded context, influence, diagnosis, structural comparison, and corrective availability.
- Does not execute scoring, NLI, replay, or generation.
- Uses metadata-only influence projection and does not duplicate the rendered private prompt.

### `RT-01` — reusable worker lifecycle

- Extracted one worker's spawn, handshake result, restart budget, registry fields, and shutdown into a
  reusable handle.
- Preserved current single-worker behavior and existing test seams.

### `CB-SEED-01` — causal-bisect random-control contract

- Treats the caller seed as a base seed.
- Derives an order-independent uint64 seed for every source/hook/layer/head site.
- Records the derivation strategy while retaining the actual seed in each transplant artifact.

### `UI-TEST-00` and `UI-SCOPE-01` — Studio component foundation

- Added Vitest, React Testing Library, jsdom, shared render/fetch helpers, and `pnpm test`.
- Canonicalized Scope state as
  `#/runs/<run-id>/scope?view=...&token=...&reference=...&layer=...`.
- Preserved token-only links, clamped run-specific bounds, aborted stale reference fetches, and proved
  selection cannot execute a fork.

### `RT-00` — multi-model public contract

- Added ADR 004 and `clozn.model-routing.v1`.
- Closed lifecycle and adapter unions and fixed a 14-code error/status/retry/phase matrix.
- Defined immutable runtime identity, load coalescing, queue/cancellation behavior, idle LRU eviction,
  protocol lowering, and one-model compatibility.
- Added no routing behavior; the current runtime remains single-model.

### `CKPT-ID-01` — restart-safe checkpoint references

- Added one opaque worker-generation identity per process.
- Made every new checkpoint ID generation-scoped and centralized issuance/lookup/true FIFO eviction.
- Added an optional generation precondition to checkpoint consumers without breaking legacy
  in-process callers.
- Advanced the additive private-worker protocol to 1.1.

### `FORK-00` — execution-fork artifact and eligibility planner

- Added `clozn.execution-fork.v1`, fixtures, and a pure planner/classifier returning exactly one of:
  `exact_execution_fork`, `reconstructed_replay`, or `unavailable`.
- Fails stale worker generations, missing token/prompt boundaries, expired checkpoints, and unsupported
  interventions before generation.
- Uses the exact ADR 004 runtime-key digest and the registry's explicit worker-generation identity.
- Treats a supplied exact checkpoint as binding: failure never silently falls back to reconstruction.
- Keeps prompt-boundary reprefill distinct from generated-token live-KV truncation.

### `EV-SPAN-00` — stable text-span addresses

- Added a versioned span-address schema for delivered messages, rendered prompt segments, source spans,
  answer spans, and claims.
- Uses half-open Unicode code-point offsets and exact UTF-8 SHA-256 basis/span hashes.
- Defines full, metadata-only, redacted, unavailable, and drifted resolution without leaking disputed
  text.
- Maps parent/child spans only when kind, offsets, basis hash, and span hash remain exact.
- Projects existing context and influence identifiers without rewriting runs or copying measurements
  into a false universal score.

### `RT-02` — preloaded worker registry

- Added a registry keyed by the exact ADR 004 runtime identity.
- Added independent spawn, handshake qualification, health, restart, stop, and failed-start recovery
  with fake-worker coverage.
- Supports multiple preloaded handles while retaining one configured default and the existing
  single-model launch path.
- Rejects behavior-bearing launch flags that cannot be represented exactly in the v1 runtime key.
- Exposes explicit worker process-generation identity in its status projection.

### `FORK-01` — exact gateway execution and immutable child receipt

- Added exact-only plan, execute, and terminal-result routes without changing the legacy `/fork`
  route.
- Revalidates the parent fingerprint, runtime, worker generation, checkpoint, and exactness regime
  immediately before execution.
- Runs the unchanged control first. Divergence, worker failure, cancellation, or stale evidence
  creates no child generation.
- Stores every terminal receipt immutably; a successful intervention is also embedded atomically in
  its real child run with lineage.
- Cancellation is cooperative before and between the two synchronous worker calls. The private
  worker protocol has no request ID for interrupting an in-flight call.
- The public executor currently requires a caller-supplied, still-live checkpoint reference. It does
  not yet create, pin, export, or import that reference from a recorded parent.

### `FORK-CKPT-01` — recorded-parent checkpoint capture

- Added `clozn.checkpoint-reference.v1` with explicit available, unavailable, and failed states,
  actual worker-reported size, generation scope, bounded-FIFO expiry, and hard-coded
  `ephemeral`/`pinned:false` lifecycle truth.
- Added `POST /runs/<id>/execution-fork/checkpoint` without changing the legacy `/fork` route.
- Reconstructs prompt IDs through the identity-matched worker's `/score` evidence while keeping
  recorded continuation IDs separate at the BPE boundary; the result must match the original
  recorded prompt-token count.
- Restores exact fixed-seed sampler provenance and accepts steering only from a recorded raw vector,
  layer, coefficient, and dial digest. Dial names alone fail closed.
- Captures with the original prompt boundary and proves the checkpoint through the same unchanged
  exact-fork control used by execution, without fabricating a child run or mutating the parent.
- Pinning, export/import, truncation, durable inventory, and deletion remain `FORK-PIN-01`.

### `EV-SPAN-01` — investigation span projection

- Added metadata-only `GET /runs/<id>/span-addresses` and an additive investigation section/action.
- Loads only an already-persisted influence artifact; it never starts a measurement or calls a
  worker.
- Distinguishes not recorded, unavailable/corrupt blob, failed native validation, exact metadata,
  drift, and redaction while retaining usable context addresses.
- Redaction tombstones override stale imported duplicate text, and contract failures do not echo
  exception text from malformed private artifacts.

### `RT-03` — route selection over preloaded workers

- Added an exact preloaded router consumable from the `RT-02` registry projection.
- Routes native generation, OpenAI chat and surviving text completions, and Ollama chat/generate
  through one request-scoped substrate/engine; discovery lists the configured canonical IDs.
- Requalifies live model, context, backend, adapter, capabilities, build/template when available,
  and worker generation before dispatch.
- Journals the full `clozn.model-routing.v1` receipt from the selected worker and clears all routing
  overrides after dispatch. Unknown, unready, failed, and drifted selections never fall back.
- Managed `clozn serve` still launches one worker and does not construct this router. Multi-model
  configuration/process transport remains a required bootstrap ticket, not a shipped CLI feature.

## Next implementation wave

The remaining tickets can run in parallel. They convert the completed seams into user-invokable paths
without starting cold loading, broad workbench actions, or general mechanistic jobs.

### `EV-UI-01` — “What did the model receive?” Studio panel

- Build the first receipt panel on `GET /runs/<id>/investigation` and `/span-addresses`.
- Keep delivered, assembled, omitted, redacted, unavailable, and failed states visually distinct.
- Show token/byte cost and link visible entries through stable span IDs.
- Keep exact rendered prompt text behind the existing authorized context-receipt disclosure rather
  than copying it into metadata-only APIs.

### `RT-BOOT-01` — managed preloaded multi-model bootstrap

- Add explicit CLI/config inputs for canonical model definitions, default, preload set, and resident
  limit without changing the one-model invocation.
- Let the supervisor own an `RT-02` registry, start preloads independently, and transport its exact
  routing projection into the gateway without exposing private worker ports publicly.
- Construct `RT-03` in the gateway, report configured/resident lifecycle state, and shut down or
  recover workers independently.
- Add managed two-model smoke coverage before load-on-demand or eviction is enabled.

## Following wave

1. `FORK-02`: make `/runs/<id>/fork` the compatibility wrapper and expose exact/reconstructed/
   unavailable states in Studio after `FORK-CKPT-01`. Remove the old text splice only after parity
   coverage.
2. `FORK-PIN-01`: add transactional checkpoint export/import/truncate, pin inventory, child-aware
   deletion, rollback, and CLI `snapshot pin|unpin|list`.
3. `MECH-CASE-00`: resolve an experiment cell or diff-model changed token into a versioned behavioral
   target.
4. `MECH-CLI-01`: wire `diff-model --mechanistic` and `experiment explain-cell`; keep artifacts
   file-addressable before adding server storage/jobs.
5. `RT-04`: load coalescing and idle LRU eviction.
6. `RT-05`: per-worker generation semaphore, mutation lock, cancellation, and queue timing.
7. `SESSION-00`: session table/migration and ordered run query, followed by paginated trace synthesis.

Durable checkpoint pin/export/import/truncate work begins after `FORK-CKPT-01` proves public capture
and before Studio presents checkpoints as durable. Model second opinions wait for `RT-BOOT-01`
through `RT-05`; arbitrary “Did this matter?” experiments wait for executable child-run repair/fork
infrastructure; the unified workbench action surface waits for the new component harness and the
relevant backend action, both of which must be real before a button is offered.

## Validation baseline

- Python/model-free product suites: `4199 passed, 18 skipped, 5 subtests passed`.
- Python engine-client contract: `18 passed`; focused cross-slice integration: `575 passed, 2 skipped`.
- Studio: copy lint, TypeScript build, 13 Vitest tests, and SSR smoke over 7 panels/15 routes.
- C++: model-free checkpoint-store CTest passed; `clozn-server` is build-clean.
