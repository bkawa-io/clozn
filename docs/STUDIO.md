# Clozn Studio

Studio is the browser interface served by the same gateway that records runs.

```sh
clozn serve MODEL --port 8080
clozn studio --port 8080 --open
```

## Run list and composer

The landing view lists durable runs and offers New run. The composer accepts one Markdown or plain-text
prompt plus optional instructions. Clozn uses headings, fenced blocks, document markers, and paragraph
boundaries to derive a bounded flat source partition automatically. The working example populates those
fields and executes a real run; it is not a static mock.

The default arm limit is 16 and can be set from 1 through 64 with
`CLOZN_MAX_CONTEXT_ARMS`. For total removable length `L` and arm limit `A`, the POC uses a target of
`ceil(L/A)`, a preferred minimum of `max(40, ceil(target/2))`, and a preferred maximum of
`max(240, target*2)`. It packs adjacent Markdown pieces toward those sizes, then merges the smallest
adjacent chunks until it reaches the cap. Every non-empty input message is eligible, regardless of role or
turn: system instructions, conversation history, assistant prefills, and the latest user message all belong
to the same analysis universe. Chat-template syntax remains structural and is never treated as source text.

## Run workspace

One run workspace has two modes:

- **Context Effects** opens on the linked input/output reader. Expanding either pane turns that same
  prose into a full-width influence heatmap; hover previews linked text and click pins the existing
  evidence inspector. Its Efficiency view summarizes the highest- and lowest-influence context and
  answer spans. Its Lean Context view runs a 16-execution teacher-forced loss search, reusing compatible
  Context Effects measurements. Lean Context reports only the best candidate found within the budget;
  it never claims a proven minimum.
- **Minimal Context** runs the bounded exact-preservation search, shows progress and cancellation, and
  saves the best verified result. Its comparison is limited to the original and that saved child.

Received context, provenance, model/runtime identity, and termination details live in disclosure panels
inside this workspace. There is no separate session navigator, runtime dashboard, experiment matrix, or
general comparison workbench in v1.

## Evidence rules

- Delivered context is not automatically influential.
- Context Effects scoring is not the success criterion for Minimal Context.
- Lean Context uses Context Effects scoring with fixed balanced loss tolerances; unlike Minimal Context,
  it permits bounded probability loss and does not create a reduced child.
- Low effect, unmeasured, unavailable, failed, and cancelled are distinct.
- “Best verified within budget” is the normal bounded-search result.
- Saving creates a child and leaves the original immutable.
- Deletion is explicit and refuses unsafe parent removal while children remain.

The backend routes used by these views are listed in
[STUDIO_SURFACE_CONTRACT.md](STUDIO_SURFACE_CONTRACT.md).
