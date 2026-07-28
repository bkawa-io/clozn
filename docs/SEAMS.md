# Extension seams — how to add a command, an artifact, or an identity facet

Three parts of clozn are designed to be extended by *adding a file* rather than by editing a shared
registry. This document is the contract for all three.

They exist because the alternative does not scale past one author. A new subcommand used to mean two
lines in `clozn/cli/main.py`; a new identity field meant editing one function body in
`clozn/runs/identity.py`. With several features in flight at once, every one of them rewrites the same
handful of lines and every merge is a conflict in code nobody meant to change.

**If you are working from a feature spec:** this file wins on *mechanics* (where code goes, how it
registers); your spec wins on *behavior* (what the feature does).

## The rule that matters most

**Do not edit a file another feature also needs to edit.** Three seams exist specifically so you never
have to. If you find yourself adding a line to a shared registry, stop — there is almost certainly a
seam for it, and if there genuinely isn't, say so in your plan instead of editing the file.

Known shared files, all of which you should expect to leave untouched:

| File | Why you don't edit it | Use instead |
|---|---|---|
| `clozn/cli/main.py` | 7 features add commands | `CLOZN_AUTOLOAD` (below) |
| `clozn/runs/identity.py` | 5 features add identity fields | `identity_providers/` (below) |
| `setup.py`, `pyproject.toml` | packaging is feature 01's | flag it in your plan |
| `tests/test_schema_contracts.py` | discovers by filesystem walk | add fixture files |

## Hard architectural constraints

1. **stdlib-only.** `pyproject.toml` declares `dependencies = []` deliberately — the product CLI and
   gateway never import Torch, transformers, or anything outside the stdlib, and the `product-minimal`
   CI lane asserts `clozn.server.app` imports with none of them installed. **Do not add a dependency.**
   Anything an optional command needs beyond the stdlib is imported lazily *inside the function body*,
   never at module scope. This is why there is a hand-written schema validator instead of `jsonschema`.
2. **Omit, never null-pad.** Throughout `clozn.runs`, a value that cannot be honestly measured is an
   absent key, not `null`. Absence must stay visible and must never be mistaken for a measurement.
3. **No silent fallback** (roadmap rule 3). An unsupported capability fails clearly or downgrades with
   an explicit marker. It never appears to work while doing nothing.
4. **Evidence before narration** (roadmap rule 1). Never label a cause proven unless an intervention or
   deterministic comparison proves it. Use the explicit states: `observed`, `eliminated`, `reproduced`,
   `correlated`, `causally_supported`.

## Seam 1 — new subcommands (`CLOZN_AUTOLOAD`)

Create `clozn/cli/commands/<your_feature>.py`:

```python
CLOZN_AUTOLOAD = True

def add_subparser(sub):
    p = sub.add_parser("setup", help="install and verify a matching native engine")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_setup)
```

That is the whole registration. `clozn/cli/commands/_autoload.py` finds it at the end of
`build_parser()`. **Do not add an import line to `main.py`.**

Both `--json` and human output are part of the contract, not a nice-to-have: the shared definition of
done requires both be stable enough for automation.

## Seam 2 — new stored artifacts (`clozn/schemas/`)

Every artifact you persist needs a versioned schema *first* (roadmap rule 7).

1. Write `clozn/schemas/defs/<schema_version>.json` — the filename **is** the `schema_version` string
   its documents carry (`clozn.context-receipt.v1.json`), which is what lets `validate()` infer a schema
   from an untrusted document with no registry to keep in sync.
2. Add fixtures at `tests/fixtures/schemas/<schema_version>/valid__<case>.json` and
   `invalid__<case>.json`. `tests/test_schema_contracts.py` discovers them by walking the filesystem, so
   you get round-trip tests without editing a shared test module. Name each `invalid__` case for the
   defect it encodes.
3. Call `clozn.schemas.validate(doc)` wherever you read or write the artifact.

The validator is a deliberate JSON Schema *subset* (see `clozn/schemas/_validator.py`). It raises
`SchemaError` on any keyword it does not implement rather than ignoring it — if you need `multipleOf`,
add it there and say so in your PR. Unknown *document* fields are tolerated by default for forward
compatibility; set `"additionalProperties": false` on a schema that must stay closed.

Schemas are immutable once released. Compatible change → edit `.v1`. Breaking change → add `.v2` and
leave `.v1` in place. Adding a required property to a `.vN` that users already have on disk silently
invalidates their data; don't.

`clozn.run-identity.v1` already exists — read it as the worked example.

## Seam 3 — new run-identity facets (`identity_providers/`)

Create `clozn/runs/identity_providers/<facet>.py`:

```python
NAME = "engine_artifact"

def identity(context):
    health = (context or {}).get("engine_health") or {}
    out = {}
    if health.get("protocol_version") is not None:
        out["protocol_version"] = health["protocol_version"]
    return out            # {} or None -> namespace omitted entirely
```

Your fields land at `identity["ext"]["engine_artifact"]`. Providers run on the path that records every
real run: keep them cheap (read already-computed values out of `context`; never hash a multi-GB file or
make a network call), and never let one raise — a broken facet must cost its own namespace, not the run.

## Seam 4 — new HTTP routes (`CLOZN_ROUTE_AUTOLOAD`)

`clozn/server/app.py` hand-wires ~25 route modules into `_GET_ROUTES` and `_POST_ROUTES`. Create
`clozn/server/routes/<family>.py` instead:

```python
CLOZN_ROUTE_AUTOLOAD = True

def try_get(h, p):
    if p == "/experiments":
        h._json(200, {"experiments": [...]})
        return True
    return False            # not mine -- keep looking
```

`try_get(h, p)` and `try_post(h, p, body)` are both optional; you are added to whichever lists you have
a handler for. Truthy return means "handled". Setting the marker with neither handler is an error, not
a no-op.

**Order is semantic here.** `_GET_ROUTES` ends with the generic `GET /runs/<id>` fallback, deliberately,
so every more-specific `/runs/<id>/<suffix>` family gets first refusal. Autoloaded GET modules are
spliced in *before* that fallback. If you are adding a `/runs/<id>/<something>` route this is what makes
it reachable at all — appended after the fallback it would be shadowed, and shadowed as a wrong-shaped
200 rather than a 404.

Within the autoloaded group, dispatch order is by module name.

## Survey before you build

**A great deal of this roadmap is already partly implemented.** The pack was written against the product
vision, not against a file-by-file audit of the tree. Before designing anything, find out what exists:

```
clozn/receipts/          24 modules
clozn/behavior/          steering, preferences, corrective_retries, feedback
clozn/replay/            exact replay
clozn/runs/              identity, journal
clozn/cli/commands/      context.py  diagnose.py  doctor.py  explain.py  connect.py
                         regression_suite.py  retry.py  runs_privacy.py  provenance.py
tests/                   153 files, incl. test_context_receipt.py, test_connect_cli.py
```

An agent that greenfields a parallel implementation of something already in the tree has produced work
that will be thrown away. Extending existing code is the expected outcome; replacing it needs a stated
reason.

## Testing

- `python -m pytest tests/ -q` — full suite, ~60s, **2338 passed / 17 skipped is the green baseline.**
  It must still be green when you finish.
- Model-free by default (roadmap rule 8). No GPU, no model download in unit or contract tests.
- Prefer fixture-driven tests that a later feature extends by adding a file.

## Adding a seam

If you need an extension point that isn't here, add it in the same shape the other three use — discovery
by filesystem walk, opt-in by an explicit marker, failures recorded rather than swallowed — and document
it above. The pattern's whole value is that it is the same pattern every time.
