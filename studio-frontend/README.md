# Clozn Studio frontend

Parallel frontend source for the next Studio workspace. The production build is written to
`../studio/next/` and is served by the Clozn gateway at `/next/index.html`.

## Commands

```bash
pnpm install --frozen-lockfile
pnpm dev
pnpm check
pnpm build
```

The development server proxies product API requests to `http://127.0.0.1:8080`.

## UI copy

Visible text must identify a control, state, measurement, object, or consequence. Product slogans,
ambient reassurance, and decorative explanations do not belong in the workspace. `pnpm check:copy`
rejects known examples.

## Data labels

Demo values must remain marked `DEMO`. Live surfaces must preserve the distinction between measured,
derived, estimated, illustrative, and unavailable values.
