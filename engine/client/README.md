# Private worker client

`clozn_engine.EngineClient` is the stdlib-only Python client used by the public gateway to talk to one
private `clozn-server` worker.

Its v1 surface is intentionally narrow:

| Method | Worker operation |
|---|---|
| `health` | worker/model identity and readiness |
| `complete` | plain autoregressive completion |
| `apply_template`, `apply_template_info` | model-owned chat rendering |
| `score` | scalar recorded-continuation score, with optional top-k evidence |
| `reference_match_arms` | batch exact-reference checks |
| exact-reference create/probe/promote/close | worker-local persistent parent session |
| `cancel` | active request cancellation |

The client is private infrastructure, not a research SDK or public compatibility API. It has no NumPy,
Torch, or `requests` dependency and exposes no state editing, steering, harvesting, checkpoints,
J-lens, SAE, or time-travel operations.
