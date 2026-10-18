# Runtime split

The product has two processes with one public boundary.

| Owner | Responsibilities |
|---|---|
| Python gateway | CLI, public API validation, Studio, durable runs, source mapping, jobs, Context Effects, Minimal Context, comparison |
| Native worker | one GGUF, chat template, generation, scalar score, exact-reference matching, cancellation, runtime identity |

The gateway is stdlib-only and never loads model weights. The worker is private and loopback-only.
Public clients use the gateway's Chat Completions subset; the gateway uses its own narrow worker client.

Private worker routes are limited to health, cancellation, generation, scoring, template rendering, and
exact-reference matching. They are not an alternate user-facing API.

The shared invariants are:

1. no successful generation without durable run recording;
2. no silent fallback for unsupported behavior;
3. exact model/template/engine identity travels with analysis evidence;
4. removed research capabilities are neither advertised nor linked into the worker; and
5. Context Effects and Minimal Context reuse canonical recorded source identities but make different
   claims.
