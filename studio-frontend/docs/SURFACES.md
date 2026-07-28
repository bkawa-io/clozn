# Studio surface boundaries

## Runs

**Primary use:** find and open recorded runs.

**Owns:** filtering, identity, status, timestamps, lineage indicators, and entry into inspection.

**Does not own:** token analysis, model-interior views, interventions, or A/B conclusions.

## Lens

**Primary use:** read the response alongside the context and inspect marked spans.

**Owns:** response composition, context-to-answer highlighting, confidence spans, concepts,
selection-aware evidence, and evidence-only performance diagnosis for the selected run.

**Does not own:** layer navigation, replay transport, intervention configuration, full A/B comparison,
or live machine-resource telemetry.

## Model Scope

**Primary use:** inspect one completed run at token and layer depth.

**Owns:** token distributions, top-k entropy, source links, long-context navigation, measurement
coverage, layer readouts when available, token selection, and the launch point for a token fork.

**Does not own:** long-form response reading, behavior configuration, experiment management, or the
final A/B verdict.

The Layers view is evidence-driven:

- `POST /engine/layers` supplies the residual-norm layer × token map for the current worker;
- J-lens supplies top candidates at up to six sampled fitted layers when the worker has a compatible
  lens;
- SAE and concept features appear only when `trace.workspace_readouts` contains stored feature
  readouts;
- causal sites are computed on demand through `POST /runs/<id>/causal-trace`.

Residual and J-lens reads are post-hoc current-worker analyses, not a replay of the original forward
pass. The UI labels them accordingly and labels their worker re-tokenization. The residual endpoint
reads the first 300 response characters; the J-lens view reads the first 600. Recorded confidence and
alternatives remain separate from those post-hoc reads.

When a layer capability or stored artifact is unavailable, Model Scope renders its exact unavailable
state. It does not substitute demo activation, energy, stability, feature, or trajectory values into a
live run.

For large records, Scope uses two levels:

- an overview of context records and output regions;
- a bounded detail view for the selected context record, output region, and nearby tokens.

The context list is virtualized. Long outputs are grouped into deterministic text regions, while the
selected token neighborhood remains directly navigable. The overview draws only links for the active
context record or output token. It does not render the complete provenance graph.

Short traces retain the direct word-to-word thread view when all of these limits are met:

- at most 7 context spans;
- at most 220 context words;
- at most 90 output words and 140 output tokens;
- at most 240 measured links;
- viewport wider than 650 CSS pixels.

The context quick view ranks measured spans by aggregate `Σ |Δ nats|` across their recorded links.
It shows the strongest and weakest distinct spans. Unmeasured spans and measured spans without links
are excluded from the ranking.

Measurement coverage is explicit. Context that was recorded but omitted from the influence
calculation remains readable and is labeled `NOT MEASURED`; it has no inferred links.

## Compare

**Primary use:** compare two completed runs after a fork, replay, model change, or intervention.

**Owns:** aligned responses, committed-token differences, latent divergence, identity differences,
and synchronized A/B inspection.

**Does not own:** creating interventions or explaining a single run in depth.

Matched base/tuned or base/steered comparison belongs here as delta provenance. It requires the same
prompt and decode controls on both runs. The UI may show changed token identity, confidence, entropy,
and source-link structure; it must not label those differences as causal parameter attribution without
a dedicated intervention measurement.

## Behavior

**Primary use:** configure and apply supported interventions.

**Owns:** dials, concepts, memory operations, pending/applied/failed/reverted state, and consequence
previews backed by real routes.

**Does not own:** run history, general diagnosis, or comparison conclusions.

## Primary workflow

```text
Runs → Lens or Model Scope → select token → fork → Compare
                                  │
                                  └─ Behavior, when the change is an intervention rather than a fork
```

---

# Adding a surface

The sections above define what each surface *owns*. This section is the mechanics of adding one.

Two seams, matching the pattern in the repo-root `docs/SEAMS.md`: discover by walking the filesystem,
opt in by an explicit export, keep failures visible. `App.tsx` is not edited for either.

## Seam A — a new top-level surface

Create `src/panels/<id>.tsx` with a default export:

```tsx
import type { PanelContext, StudioPanel } from "./types";

const panel: StudioPanel = {
  id: "experiments",           // MUST equal the filename
  navLabel: "Experiments",
  order: 70,                   // nav position; the existing surfaces are 10..60
  icon: () => <svg viewBox="0 0 24 24">…</svg>,
  match: (hash) => (/^#\/experiments\/?$/.test(hash) ? {} : null),
  routeName: () => "EXPERIMENTS",
  Component: ({ runtime, inspectorOpen, params }: PanelContext) => <Experiments … />,
};

export default panel;
```

**`id` must equal the filename.** The registry rejects a mismatch, for the same reason `clozn/schemas/`
requires a schema's filename to equal its `schema_version`: otherwise a deep link and a nav entry can
disagree about which panel owns a route.

**`match()` returns `{}` on a bare match, never a falsy value.** `null` means "not my route"; returning
something falsy on a real match makes the panel unreachable.

**Anchor your patterns.** Panels are tried in `order`, but do not lean on it — `lens` sorts before
`scope` and would happily swallow `#/runs/<id>/scope` if its regex were not end-anchored.

An unknown hash falls back to the lowest-ordered panel (Runs), preserving the old router's behavior.

A panel whose module fails to load, or whose default export is malformed, appears in the rail as a
visible `<id> failed to load` placeholder rather than silently vanishing.

### Topbar content

Content derivable from `PanelContext` goes in the optional `topStats` / `modeChip` fields.

Content derived from the panel's **own internal state** cannot — App cannot see inside a panel, and the
whole point of the seam is that it no longer tries. Use the hook, from inside your component:

```tsx
useTopbar(() => ({ stats: <span className="top-stat"><b>RUN</b>{id}</span>, modeChip: "LOADING" }),
          [id, status]);
```

The factory-plus-deps shape is deliberate: JSX allocates a new element object every render, so a hook
taking nodes directly would set state on every render forever. Published content wins over the static
fields and is cleared on unmount. `src/panels/scope.tsx` is the worked example — it is the surface whose
run id, model, and fork/load status used to live in `App.tsx` as `route.kind === "scope" && …` branches.

## Seam B — a sub-panel inside someone else's surface

Create `src/slots/<slot>/<id>.tsx`:

```tsx
const panel: SlotPanel<LensData> = {
  id: "context-receipt",
  slot: "lens.evidence",
  title: "What the model saw",
  order: 20,
  Component: ({ data }) => <ContextReceipt run={data.run} />,
};

export default panel;
```

The host renders `<SlotHost slot="lens.evidence" data={…} />`. Each panel gets an error boundary, so a
throw costs that card and nothing else — which is what makes it safe for a host to expose a slot to code
it does not own.

**Slot names and their `data` shape are the host's contract.** Whoever owns the host page documents them
in the table below. `SlotHost` owns the mechanism, never the vocabulary.

### Registered slots

| slot | host | `data` | notes |
|---|---|---|---|
| _(none yet)_ | | | the first host to expose one documents it here |

## Verification

```
pnpm check      # copy check + tsc + smoke render
pnpm smoke      # just the smoke render
```

`scripts/smoke-render.mjs` server-renders every route under Node and asserts each one resolves to the
intended panel and mounts without throwing. It exists because `tsc` and `vite build` both pass happily
on an app that crashes on mount, and opening a browser was previously the only way to find out.

**It is not a substitute for looking at the thing.** Effects do not run under `renderToString`, so data
loading, `useTopbar` publication, and everything after first paint are not covered, and it says nothing
about layout, CSS, or theming. It covers the module graph, the registry, route resolution, and first
render — most of what a routing change can break, and none of what a design change can.

Add your route to `ROUTES` in that file when you add a panel. That list is hardcoded on purpose:
deriving it from the registry would make the assertion vacuous.
