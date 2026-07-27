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
