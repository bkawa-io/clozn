# Studio release checklist

Studio has one source tree (`studio-frontend/`), one compiled application (`studio/app/`), and one
server entry point (`/`, which opens `/app/index.html`).

## Product gates

- [x] Sessions, standalone Runs, Inspect, Runtime, Compare, and experiment-matrix routes are closed
  and typed.
- [x] Inspect provides Context Investigation, Received Context, Context Receipt, Minimal Context,
  Diagnosis and corrective actions, and Claim Support.
- [x] Ordinary navigation is read-only. Model work, cancellation, materialization, keep, and undo are
  explicit operations with separate controls.
- [x] Context evidence preserves measured, unmeasured, unavailable, failed, cancelled, stale, and
  screening-only states without promoting one into another.
- [x] The production bundle is reproducible from the frozen frontend lockfile.
- [x] Run the complete canonical-route browser suite against the packaged gateway.
- [x] Pass the clean wheel/install smoke test, including packaged Studio assets and CLI entry points.
- [x] Launch the packaged gateway with `clozn studio --open` and complete its browser smoke check.
- [x] Complete unavailable-worker and acknowledged-cancellation browser checks for model-touching
  actions.
- [x] Consolidate duplicated engine/client protocol constants and conformance fixtures.
- [x] Keep automatic variant screening behind an explicit default-off configuration, with exact
  refinement retained for evidence-bearing results.

## Current verification

- Studio: 228 component and contract tests across 24 test files, plus TypeScript, copy, raw-color,
  and production-build checks.
- Python product suite: 5,344 passed, 27 skipped, and 7 subtests passed.
- Native engine suite: 12 passed, including variant planning, checkpoint reuse, compact target
  log-probability reduction, and the scoring protocol.
- Live Qwen GPU checks cover Runtime, Run comparison, Context Receipt, Received Context, Minimal
  Context, Diagnosis/corrective actions, and Claim Support.

## Release rules

- Do not add an unavailable navigation destination as a placeholder.
- Do not infer evidence from missing fields or failed measurements.
- Do not start model work on page load.
- Do not publish screening-only measurements as exact evidence.
- Do not keep more than one Studio source tree or compiled application.
