# Studio surface boundaries

## Runs

**Primary use:** find and open recorded runs.

**Owns:** filtering, identity, status, timestamps, lineage indicators, and entry into inspection.

**Does not own:** token analysis, model-interior views, interventions, or A/B conclusions.

## Lens

**Primary use:** read the response alongside the context and inspect marked spans.

**Owns:** response composition, context-to-answer highlighting, confidence spans, concepts, and
selection-aware evidence.

**Does not own:** layer navigation, replay transport, intervention configuration, or full A/B comparison.

## Model Scope

**Primary use:** inspect one completed run at token and layer depth.

**Owns:** token distributions, top-k entropy, source links, layer readouts when available, token
selection, and the launch point for a token fork.

**Does not own:** long-form response reading, behavior configuration, experiment management, or the
final A/B verdict.

When J-lens data is unavailable, Model Scope shows the recorded token/source trace. It does not
substitute demo layer values into a live run.

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
