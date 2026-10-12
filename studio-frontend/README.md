# CLOZN Studio

Studio is the Sessions, Inspect, Runtime, and Compare application. Its production bundle is built into
`studio/app/` and served by the Clozn gateway at `/`.

```sh
pnpm install --frozen-lockfile
pnpm check
pnpm build
```

## Surfaces

- Sessions reads persisted session identity, paginated conversation traces, and standalone Runs.
- Inspect reads canonical Context Investigation evidence and provides Received Context, Context Receipt,
  Minimal Context, Diagnosis/corrective actions, and Claim Support views.
- Runtime reports worker or managed-model health and local GGUF inventory.
- Compare reads immutable Run differences and experiment-result matrices.

The closed route vocabulary is documented in `docs/STUDIO_SURFACE_CONTRACT.md`. Unknown routes return
to Sessions without selecting a Run or starting model work.

## Safety boundaries

Ordinary navigation is read-only. Context measurement, exact search, cancellation, corrective retries,
child materialization, keep, and undo start only through explicit controls. Generated text remains a
preview until a confirmed materialization records an immutable child Run.

Evidence states remain typed rather than inferred from absence. In particular:

- delivered context does not imply influence;
- screening-only results are not exact evidence;
- missing measurement is not unsupported evidence;
- “unsupported by supplied materials” is not “false”;
- a comparison reports change, not causation.

Wire Unicode ranges use half-open Python code-point coordinates. The UI validates hashes and range
identity at its data boundary before rendering browser strings.

## Build integrity

The production bundle must be generated from `pnpm-lock.yaml`. Do not edit `studio/app/assets/*`
directly; content hashes and `index.html` references are produced together by Vite.
