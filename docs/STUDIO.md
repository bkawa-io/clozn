# Clozn Studio

Studio is the browser interface served by the same gateway that records Clozn Runs. It reads the same
SQLite journal and evidence artifacts as the API; it does not maintain a separate evidence store.

## Run it

```bash
clozn serve MODEL --port 8080
clozn studio --open
```

Point OpenAI-compatible clients at `http://127.0.0.1:8080/v1`. Completed requests are recorded and
appear in Studio.

## Sessions

Sessions lists conversation records and standalone Runs. A session trace preserves turn order, branch
markers, failures, cumulative facts, and explicit Run selection. Selecting a turn opens its immutable
Run in Inspect.

## Inspect

Inspect is the single-Run evidence workspace:

- **Read / Context effects** shows the recorded answer, supplied context, measured answer-source
  relationships, and explicit context-measurement controls.
- **Received Context** shows what was delivered, assembled, or omitted without treating delivery as
  influence.
- **Context Receipt** shows the metadata-only delivery, rendering, capacity, and termination record.
  Prompt content remains part of the Run and is never duplicated into the receipt.
- **Minimal Context** searches for a directly verified preserving source set. Exact minimum, inclusion
  minimum, best verified, incomplete search, no reduction, failure, cancellation, and unavailability
  remain distinct.
- **Actions** shows diagnosis evidence and the bounded corrective workflow: preview, confirm or cancel,
  optional keep for this Run, and undo.
- **Claims** binds metadata-only claim ranges to the exact recorded answer and preserves supported,
  weakly supported, contradicted, unsupported by supplied materials, unverifiable, and measurement
  unavailable states.

Navigation is read-only. Measurement, search, correction, cancellation, materialization, keep, and undo
run only from explicit controls.

## Runtime

Runtime reports direct-worker or managed-router health and local GGUF inventory. Missing facts remain
unavailable rather than being inferred. The surface does not load, switch, download, or remove models.

## Compare

Compare aligns two immutable Runs and reports recorded identity, context, settings, tools, output, and
classified finding differences. The experiment matrix reads persisted case × variant × seed results and
loads cell evidence on demand. A difference is an observation, not a causal claim.

## Evidence rules

- Delivered context is not automatically influential.
- Token confidence describes commitment, not correctness.
- Screening results identify candidates; exact scoring establishes evidence.
- Missing or failed measurement is not negative evidence.
- “Unsupported by supplied materials” does not mean false.
- Generated previews become evidence only after an explicit materialization records a child Run.
- The source Run remains immutable; run-scoped keep and undo change only its selected revision.

The canonical routes and backend ownership are listed in [STUDIO_SURFACE_CONTRACT.md](STUDIO_SURFACE_CONTRACT.md).
