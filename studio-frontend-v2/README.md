# CLOZN Studio v2

This is the greenfield Studio implementation derived from the implementation handoff. It is isolated
from `studio-frontend/`, and its build writes to `studio/v2/`; the current `studio/next/` application
remains the default served Studio until an explicit cutover.

## Current vertical slice

- Lunar Nacre and Black Pearl semantic themes with Blue Ion interaction accents.
- Closed product routing for Runs, Inspect, Time Travel, Compare, Model MRI, and Runtime.
- A chronological Runs journal with separate session and lineage context.
- A readable Context ↔ Answer investigation surface backed by stable span addresses and the read-only
  influence query. It preserves Unicode code-point coordinates and typed unavailable/not-measured/error
  states. The matrix is not the hero surface.
- A Models / Runtime surface that keeps liveness, readiness, model residency, lifecycle, capability,
  and telemetry facts separate. Unreported data stays unreported.
- A recorded Compare surface backed by `clozn.run-diff.v1`, with exact first-divergence coordinates
  when retained and structural observations kept separate from causal evidence.
- A read-only token-boundary Time Travel surface backed by `clozn.rewind-fidelity.v1`. It keeps
  reconstructed replay, current exact-fork planning, and historical exact proof as independent states.
- A token × layer Model MRI instrument shell. Until a typed internal-artifact client is integrated,
  every J-lens/SAE/attention channel remains explicitly `not reported`; no readings are synthesized.
- Shared typed investigation loci, evidence states, registration rails, provenance captions, and
  action-shaped Test This launchers for the remaining surfaces.

Time Travel's execution-fork POST workflow remains deliberately disabled until its plan/unchanged-
control/execute responses have a strict typed client. Model MRI likewise does not call measurement
POST routes merely because a user selects a coordinate.

## Develop and verify

```sh
pnpm install
pnpm dev
pnpm check
pnpm run build
```

Vite proxies the Studio read routes to `http://127.0.0.1:8080`. The production build is directly
previewable from a running CLOZN server at `/v2/index.html`; `/`, `/next`, and the CLI-opened Studio
continue to resolve to the existing application.

## Cutover boundary

A future cutover should happen only after the remaining primary surfaces and browser visual QA are
complete. At that point, change `clozn/server/static.py::APP_INDEX` from `/next/index.html` to
`/v2/index.html` and update the static-serving smoke tests in the same change. Do not edit generated
files in `studio/v2/assets/` by hand.
