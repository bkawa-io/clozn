# Clozn Studio

Studio is the conversation-first reader for Context Effects and Lean Context. Its production bundle is
built into `studio/app/` and served by the Clozn gateway at `/app/`.

```sh
pnpm install --frozen-lockfile
pnpm check
pnpm build
pnpm build:demo
```

The primary visual is always the recorded context and answer. Context Effects links measured support
and suppression across those texts; Lean Context presents the same context as a redline or kept-only
document and compares fresh full/lean generations.

The closed route vocabulary is documented in `docs/STUDIO_SURFACE_CONTRACT.md`. Navigation is read-only;
measurement, Lean Context search, cancellation, and run deletion require explicit controls. Evidence
states remain typed rather than inferred from absence, and Unicode ranges use half-open Python
code-point coordinates.

Do not edit `studio/app/assets/*` directly. Vite generates content hashes and `index.html` together.
