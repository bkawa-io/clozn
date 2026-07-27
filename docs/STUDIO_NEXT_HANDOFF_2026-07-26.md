# Clozn Studio Next — Agent Handoff

Repository: `bkawa-io/clozn`
Branch: `codex/studio-next`
Date: 2026-07-26

## 1. Objective

Continue the new Clozn Studio frontend in `studio-frontend/`. The built app is written to
`studio/next/` and is currently available at:

```text
http://127.0.0.1:8080/next/index.html
```

The primary product workflow is:

```text
completed run → inspect → fork → compare
```

The next milestone should add useful performance readouts to Model and a per-run explanation of
latency. Much of the backend evidence already exists; use it before adding new instrumentation.

## 2. Product and visual rules

These are product requirements, not suggestions:

1. Visible copy must identify a control, state, measurement, object, or consequence.
2. Do not spend interface space on slogans, ambient reassurance, or AI-sounding filler.
3. Banned examples include “Local Only,” “It doesn’t phone home,” and “the river, not the
   molecules.”
4. Use square corners. The owner explicitly rejected rounded cards and panels.
5. Missing values stay missing. Never invent plausible metrics, layer values, provenance, or causal
   explanations.
6. Keep measured, derived, estimated, illustrative, and unavailable values distinguishable.
7. Use progressive disclosure: the first view should answer the page's primary question; details
   belong in selection-aware panels.
8. Light theme direction: opal and mother-of-pearl.
9. Dark theme direction: freshwater black pearls.
10. Cyan, mint, violet, pink, and peach may be strong in the casting and data visualizations, but
    should tint rather than flood the rest of the workspace.
11. The interface should feel like a dense technical instrument, not a marketing site.

`pnpm check:copy` enforces known copy violations. Extend the audit when a repeated failure mode
appears.

## 3. Current frontend

The new frontend is a React 19 and Vite application:

```text
studio-frontend/
  src/
    app/App.tsx
    components/
    data/
    features/
      behavior/
      compare/
      lens/
      model/
      observatory/
      runs/
    styles/
  scripts/check-copy.mjs
  docs/SURFACES.md
```

The compiled application is committed under:

```text
studio/next/
```

Current routes:

| Route | Surface | Primary use |
| --- | --- | --- |
| `#/runs` | Runs | Find and open recorded runs |
| `#/runs/<id>` | Lens | Read a response and inspect marked spans |
| `#/runs/<id>/scope` | Model Scope | Inspect one run at token and layer depth |
| `#/compare/<a>/<b>` | Compare | Compare two aligned completed runs |
| `#/behavior` | Behavior | Configure and apply supported interventions |
| `#/model` | Model | Inspect the serving model, capabilities, and active configuration |

Read `studio-frontend/docs/SURFACES.md` before changing page ownership.

## 4. Work completed in this frontend

### Runs

- Dense run ledger with filtering and lineage indicators.
- Opens completed runs into Lens.
- Uses real run summaries and preserves unavailable data.

### Lens

- Response-first reading surface.
- Context and response are independently usable.
- Selection-aware evidence for confidence, sources, concepts, and claim structure.
- Progressive Performance panel wired to `GET /runs/<id>/diagnosis`.
- Shows measured wall time, worker-measured decode throughput when recorded, a labeled derived
  end-to-end fallback, token counts, phase status, and raw evidence paths.
- Lens owns reading and claim structure. It does not duplicate layer navigation or full provenance.

### Model Scope

- Output tokens are rendered as text rather than anonymous circles.
- Clicking an output token shows the context that influenced it.
- Clicking a context span shows the output tokens it influenced.
- Large records use a bounded overview/detail layout:
  - the context list is fixed-row virtualized;
  - long output is grouped into deterministic text regions;
  - the selected token keeps a 73-token local neighborhood;
  - the shared token tape renders at most 120 token buttons;
  - only the active provenance links are drawn, capped at 12 paths.
- Short records retain direct context-span-to-output-token threads. The threaded view is selected
  only when the context, output, source count, link count, and viewport all fit fixed limits.
- Large-record context browsing opens on a quick view of the most and least influential measured
  spans. Ranking is aggregate `Σ |Δ nats|`; unmeasured and unlinked spans are excluded.
- The short-record threaded view exposes the same ranking behind `RANKED SPANS` so the thread field
  remains clear by default.
- Recorded context omitted from the influence calculation remains visible as `NOT MEASURED`.
- Coverage reports recorded sources, measured sources, output tokens, and prompt tokens when the
  latter was recorded.
- Token selection, candidate readout, confidence plot, trace, and fork entry are synchronized.
- Layers no longer uses the synthetic demo planes or invented activation, energy, stability, and
  feature counts for real runs.
- `POST /engine/layers` drives a post-hoc residual-norm layer × worker-token map. The first 300
  response characters are re-tokenized by the current worker and labeled as such.
- A compatible J-lens adds candidate trajectories across up to six sampled fitted layers. The panel
  remains explicitly unavailable when no lens is loaded.
- Stored SAE/concept features are parsed from `trace.workspace_readouts`; no readout is synthesized
  from engine capability alone.
- `POST /runs/<id>/causal-trace` remains an explicit, on-demand action for the selected recorded
  token because it performs controlled interventions.
- Variant view aligns a reference and current run using structural token identity.
- Adapter and LoRA identity is shown only when recorded.
- The variant view labels structural alignment as structural evidence; it does not call it causal.

### Compare

- Response alignment uses LCS-style sequence alignment instead of naive token-index pairing.
- Shared text is preserved even when it occurs at different positions in the two responses.
- Token differences and synchronized A/B inspection are available.

### Behavior

- Tone dials, concepts, memory, runtime, and profiles use real routes.
- Draft, apply, failure, and revert states are explicit.
- Unsupported actions are disabled instead of simulated.

### Model

- Configuration Stack is the default view.
- The stack includes base model, adapter metadata, active steering, concept readout, and memory.
- A compact architecture strip reports factual layer, embedding, context, and vocabulary values.
- The former standalone topology view was removed because it implied inspectable layer evidence that
  the backend does not currently expose.
- Capabilities and read-only Model Inventory remain separate views.

### Workspace

- Full-window square-corner instrument layout.
- Halo and Cathedral themes.
- Compact rail and inspector controls.
- Responsive layouts were checked across all six routes.

## 5. Performance work: use existing evidence first

The owner wants common local-model performance information on Model and a per-run answer to “why was
this response slow?”

Do not begin by creating a new latency model. Clozn already has an evidence-only diagnosis endpoint:

```text
GET /runs/<id>/diagnosis
```

It returns schema `clozn.run_diagnosis.v1` with:

- `why_slow.summary`
- `why_slow.findings`
- `why_cut_off`
- `client_auxiliary_calls`

The implementation is in `clozn/runs/diagnosis.py`. It already distinguishes observed,
not-observed, and unavailable evidence for:

- end-to-end wall time
- model load time
- prompt prefill or prompt evaluation
- generation time
- per-token decode intervals
- context pressure
- context/KV allocation
- CPU spill
- output cutoff
- nearby calls associated with the same client or session

The runtime also records worker-measured generation metadata when the worker supplies it:

```text
meta.generation_duration_ms
meta.generation_tokens
meta.generation_steps
meta.generation_tokens_per_second
```

See `clozn/runs/trace.py::generation_timing_from_frames`.

Run records may also contain:

```text
timing.duration_ms
context_receipt.limits.prompt_tokens
context_receipt.limits.generated_tokens
context_receipt.limits.context_window_tokens
context_receipt.limits.requested_max_tokens
meta.device
meta.gpu_layers
trace.steps[*].dt_ms
```

Availability varies by substrate and run. The frontend must not replace missing worker measurements
with a derived value under the same label.

### Implemented performance slice

The frontend now:

1. Parses `GET /runs/<id>/diagnosis`.
2. Loads recorded performance facts for the selected Lens run.
3. Shows end-to-end duration, prompt and generated token counts, worker-measured generation tok/s,
   generation duration, and finish reason when available.
4. Labels a generated-tokens/end-to-end-duration fallback as `DERIVED END TO END`; it is never
   presented as decode throughput.
5. Renders observed, not-observed, and unavailable phase statuses.
6. Keeps the diagnosis progressively disclosed so the response remains the Lens hero.
7. Shows one selected phase explanation and its raw evidence paths instead of repeating every
   explanation at once.

The next performance step is Compare. A useful first comparison is:
   - total duration A/B
   - measured decode tok/s A/B
   - prompt/generated token counts A/B
   - phase availability A/B

If `generation_tokens_per_second` is absent, it is acceptable to calculate:

```text
generated tokens / end-to-end duration
```

only when both operands are recorded, and only under an explicit label such as:

```text
derived end-to-end generated tokens/s
```

That value includes prompt processing and gateway work. It is not decode throughput.

### Model performance view

Add a Performance view to Model after the per-run contract is wired. Useful values include:

- current serving model
- quantization
- device
- GPU-offloaded layers
- context capacity
- recent measured decode tok/s
- recent end-to-end duration
- recent prompt and generated token counts
- distribution across recent comparable runs

Use recorded recent runs for historical performance. Do not mix runs with different models,
substrates, context sizes, sampling regimes, or active interventions into a single benchmark without
making those dimensions visible.

The current `/engine/health` response provides serving identity and placement fields such as:

- architecture
- model path and hash
- device
- GPU layers
- context size
- embedding width
- layer count
- vocabulary size
- capability flags

It does not currently provide reliable live VRAM, RAM, GPU utilization, KV-cache occupancy, queue
depth, or thermal information. Those must remain unavailable until a real runtime contract exists.

### Backend work that may follow

Only add fields the worker or gateway can directly measure. Candidate contracts:

- first-token latency
- queue duration
- prompt-evaluation duration
- context/KV allocation duration
- current and peak VRAM
- current and peak process RAM
- KV-cache bytes and occupancy
- CPU spill bytes
- GPU utilization sampled during generation

Document whether each value is per request, current runtime state, peak during the run, or a sampled
estimate. Add schema and route tests before drawing it in Studio.

## 6. Known limitations

1. `clozn studio --open` still opens the legacy app.
   - `clozn/server/static.py` sets `APP_INDEX = "/app/index.html"`.
   - `clozn/cli/commands/studio.py` opens the server root.
   - The new app is explicitly opened at `/next/index.html`.
   - Promote the new app only after packaging and static-serving tests cover the compiled assets.
2. The old framework-free frontend under `studio/app/` is still present and remains the default.
   Do not delete it as part of an unrelated UI change.
3. Model Inventory depends on `GET /models/local`. Some running gateways do not expose the route; the
   UI reports it unavailable and does not fabricate load or switch controls.
4. There is no model load/switch route, so Model Inventory is read-only.
5. Adapter/LoRA identity is not consistently recorded in run or health data. The UI shows
   `UNREPORTED`.
6. Model Scope Variants is token-identity alignment, not causal parameter attribution.
7. A base-versus-tuned provenance view requires comparable runs and recorded adapter identity.
8. Residual summaries can run on any compatible current worker. J-lens and SAE panels remain
   unavailable unless their compatible artifact is loaded or a run stored its feature readouts.
   Post-hoc reads are not the original generation trajectory.
9. Live memory and utilization telemetry are not exposed by the current engine health contract.
10. The current context-answer influence producer selects at most eight prompt sources. Scope shows
    the remaining recorded context, but it cannot draw links that were never measured.

## 7. LoRA and steering provenance direction

The owner wants the provenance interaction to support comparisons where the “source” is a base model
versus a LoRA-tuned or steered model.

A valid first version belongs in Compare or Model Scope Variants and requires:

1. the same prompt and decode controls;
2. recorded base model identity;
3. recorded adapter identity and scale, or recorded steering configuration;
4. aligned committed tokens;
5. explicit labels for identity, confidence, entropy, and source-link changes.

Do not label a token as “caused by LoRA” from a normal A/B run alone. Causal attribution requires a
dedicated intervention measurement.

## 8. Local development

Attach Studio to a running Clozn gateway:

```bash
clozn serve <model>
clozn studio
```

Install and build the new frontend:

```bash
cd studio-frontend
pnpm install --frozen-lockfile
pnpm check
pnpm build
```

Open:

```text
http://127.0.0.1:8080/next/index.html
```

The Vite development server can also be used:

```bash
cd studio-frontend
pnpm dev
```

It proxies product API requests to `http://127.0.0.1:8080`.

## 9. Files to read first

1. `docs/PRODUCT_STRATEGY_USER_NEEDS_2026-07-20.md`
2. `studio-frontend/README.md`
3. `studio-frontend/docs/SURFACES.md`
4. `studio-frontend/src/app/App.tsx`
5. `studio-frontend/src/data/types.ts`
6. `studio-frontend/src/data/api.ts`
7. `studio-frontend/src/features/lens/Lens.tsx`
8. `studio-frontend/src/features/model/Model.tsx`
9. `studio-frontend/src/features/model/api.ts`
10. `studio-frontend/src/features/observatory/Observatory.tsx`
11. `studio-frontend/src/features/observatory/TraceScope.tsx`
12. `studio-frontend/src/features/observatory/VariantScope.tsx`
13. `studio-frontend/src/data/stress.ts`
14. `studio-frontend/src/features/compare/alignment.ts`
15. `studio-frontend/src/styles/tokens.css`
16. `studio-frontend/src/styles/workspace.css`
17. `clozn/runs/diagnosis.py`
18. `clozn/runs/trace.py`
19. `clozn/server/routes/runs.py`
20. `clozn/server/static.py`
21. `clozn/cli/commands/studio.py`

## 10. Verification

Run:

```bash
cd studio-frontend
pnpm check
pnpm build

cd ..
.venv/bin/python tests/test_studio.py
node --check studio/app/*.mjs
git diff --check
```

For performance work, also run:

```bash
.venv/bin/pytest tests/test_run_diagnosis.py tests/test_run_diagnosis_server.py \
  tests/test_trace_capture.py -q
```

Browser-check all routes at desktop and compact widths, both themes, 200% zoom, and reduced motion.
At minimum assert:

- no horizontal overflow;
- no console errors;
- every primary surface remains usable;
- unavailable evidence stays visibly unavailable;
- token selection and provenance remain synchronized;
- route navigation does not leave stale selection state.

Scope has deterministic browser-only QA fixtures:

```text
/next/index.html#/scope?fixture=code
/next/index.html#/scope?fixture=rag
/next/index.html#/scope?fixture=agent
/next/index.html#/scope?fixture=thread
```

They exercise long code output, retrieved-document context, multi-step tool traffic, and the compact
word-threading path. They are not listed in product navigation and are labeled as demo data.

## 11. Suggested next sequence

### PR 1 — Run performance

- Wire `GET /runs/<id>/diagnosis`.
- Add recorded and derived performance types.
- Add compact Performance to Lens.
- Preserve evidence status and raw paths.
- Add frontend parser tests and browser checks.

### PR 2 — Performance comparison and Model history

- Add A/B performance facts to Compare.
- Add Model Performance using recent compatible runs.
- Disclose run grouping and measurement source.

### PR 3 — New frontend as default

- Package `studio/next` assets.
- Update static serving and `clozn studio --open`.
- Add clean-room wheel/install coverage.
- Keep a deliberate rollback path for the legacy app during the transition.

### PR 4 — Runtime resource telemetry

- Add worker/gateway contracts only for directly measurable memory and latency phases.
- Add schema tests.
- Add Model live-resource readouts after the contracts are stable.

## 12. Definition of done for the performance milestone

- A run can answer “why was this slow?” using `clozn.run_diagnosis.v1`.
- Measured decode throughput is distinct from derived end-to-end tokens per second.
- Missing phases display as unavailable.
- Model shows recent performance only for clearly identified runs and configurations.
- Memory or utilization numbers are not shown until backed by a runtime contract.
- The page remains useful without slogans or decorative explanatory copy.
- Both themes, compact layouts, 200% zoom, and reduced motion pass browser checks.
- The new frontend still builds into `studio/next/` and the committed bundle matches the source.
