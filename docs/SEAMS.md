# Extension seams

v1 public surfaces use explicit allowlists.

## CLI

`clozn/cli/main.py::PUBLIC_COMMANDS` is the exact public command sequence. Optional command modules can
be listed in `clozn/cli/commands/_autoload.py::COMMAND_MODULES`, but an unlisted file or marker cannot
expose a command. Update the product-boundary tests whenever the public set intentionally changes.

## HTTP routes

Core health, OpenAI, static, and run routes are hand-wired. Feature route modules must appear in
`clozn/server/routes/_autoload.py::ROUTE_MODULES` and set `CLOZN_ROUTE_AUTOLOAD = True`. Modules may
provide `try_get` and/or `try_post`; returning false means “not handled.” The generic `GET /runs/<id>`
fallback remains last.

## Stored artifacts

Persisted analysis documents require a versioned schema under `clozn/schemas/defs/` and valid/invalid
fixtures under `tests/fixtures/schemas/`. Schemas fail closed on unsupported validator keywords.

## Constraints

- Keep the product Python runtime stdlib-only.
- Unsupported behavior must fail visibly.
- Omit unavailable evidence instead of fabricating values.
- Bind analysis evidence to exact recorded model/template/engine identity.
- Prefer model-free contract tests; use the live v1 acceptance gate for claims that require a GGUF.
