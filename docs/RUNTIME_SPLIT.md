# Runtime split

The product has two processes with one public boundary.

| Owner | Responsibilities |
|---|---|
| Python gateway | CLI, public API validation, Studio, durable runs, source mapping, jobs, Context Effects, Lean Context |
| Native worker | one GGUF, chat template, generation, teacher-forced scoring, cancellation, runtime identity |

The gateway is stdlib-only and never loads model weights. The worker is private and loopback-only.
Public clients use the gateway's Chat Completions subset; the gateway uses its own narrow worker client.

Shared invariants:

1. no successful generation without durable recording;
2. no silent fallback for unsupported behavior;
3. exact model/template/engine identity travels with evidence;
4. removed research capabilities are neither advertised nor routed; and
5. Context Effects and Lean Context reuse canonical recorded source identities while making distinct
   claims.
