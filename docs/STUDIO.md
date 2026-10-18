# Clozn Studio

Studio is the browser interface served by the same gateway that records runs.

```sh
clozn serve MODEL --port 8080
clozn studio --port 8080 --open
```

## Run list and composer

The landing view lists durable runs and offers New run. The composer accepts a question, optional
instructions, and plain-text context blocks. The working example populates those fields and executes a
real run; it is not a static mock.

## Run workspace

One run workspace has two modes:

- **Context Effects** shows recorded input and output side by side. Selecting answer text or a context
  source reveals measured relationships in the corresponding direction. Whole-source evidence remains
  whole-source evidence; the UI does not invent finer attribution.
- **Minimal Context** runs the bounded exact-preservation search, shows progress and cancellation, and
  saves the best verified result. Its comparison is limited to the original and that saved child.

Received context, provenance, model/runtime identity, and termination details live in disclosure panels
inside this workspace. There is no separate session navigator, runtime dashboard, experiment matrix, or
general comparison workbench in v1.

## Evidence rules

- Delivered context is not automatically influential.
- Context Effects scoring is not the success criterion for Minimal Context.
- Low effect, unmeasured, unavailable, failed, and cancelled are distinct.
- “Best verified within budget” is the normal bounded-search result.
- Saving creates a child and leaves the original immutable.
- Deletion is explicit and refuses unsafe parent removal while children remain.

The backend routes used by these views are listed in
[STUDIO_SURFACE_CONTRACT.md](STUDIO_SURFACE_CONTRACT.md).
