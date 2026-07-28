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
- **Lens** shows delivered-context receipts, measured source links, token confidence, model readouts,
  and performance evidence. Its Performance view separates measured phases from overlapping/context-only
  spans, shows known versus unaccounted wall time, names the phase clock owner, and labels derived
  end-to-end throughput separately from worker-measured decode throughput. Missing or unmeasured data
  stays visibly unavailable.
- **Compare** aligns two stored runs and separates model/artifact identity, instructions, documents,
  history, settings, tools, and output changes.
- **Experiments** renders the case × variant × seed matrix, summaries, filters, and cell detail from
  versioned experiment results.
- **Behavior** exposes qualified steering controls and the corrective-action state.
- **Model** reports the loaded worker and optional artifact state.
- **Scope** explores recorded token and layer evidence without upgrading post-hoc readouts to causal
  claims.

The panel registry is additive: a failed optional panel is reported without taking down the other
surfaces.

## Evidence rules

- A context segment marked delivered or survived is not automatically influential.
- The Sources view labels measured, below-threshold, omitted, unavailable, and failed states
  independently; it never invents links for missing measurements.
- Token confidence describes commitment, not correctness.
- Run comparison reports what changed; only a controlled intervention can support a causal claim.
- J-lens and other latent readouts carry method, artifact, and qualification provenance in the view.
- Performance diagnoses render only rules supported by recorded phases and metrics. A phase measured on
  the gateway monotonic clock is never offset-aligned with one measured on the worker steady clock;
  only non-overlapping measured durations contribute to the known-time total.

## Removed surfaces

Prompt-card and learned-prefix memory are not current Studio features. The former standalone PyTorch
workbench and diffusion UI were retired with that path. Old screenshots, design specs, and handoff notes
remain useful historical records, but they are labeled as archives and are not instructions for running
the current product.
