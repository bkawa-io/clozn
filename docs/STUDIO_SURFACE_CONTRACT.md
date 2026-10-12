# Studio surface contract

Studio is a hash-routed application served by the Clozn gateway. Unknown or malformed routes resolve
to Sessions without selecting a Run or starting model work.

## Canonical routes

| Route | Surface | Reads | Explicit operations |
|---|---|---|---|
| `#/sessions` | Session and standalone-Run index | `GET /sessions`, `GET /runs` | None |
| `#/sessions/<session-id>[?run=<run-id>]` | Conversation trace | Session detail and paginated trace | None |
| `#/inspect/<run-id>` | Context Investigation | Canonical reader and locked answer/source query | Start/cancel context measurement |
| `#/inspect/<run-id>?view=context-effects` | Context effects | Same canonical investigation contracts | Start/cancel context measurement |
| `#/inspect/<run-id>?view=context` | Received Context | Run investigation and metadata-only span addresses | None |
| `#/inspect/<run-id>?view=receipt` | Context Receipt | Metadata-only receipt read | None |
| `#/inspect/<run-id>?view=minimal-context` | Minimal Context | Durable result list/detail | Start/cancel exact search; confirm child materialization |
| `#/inspect/<run-id>?view=actions` | Diagnosis and corrective actions | Diagnosis and run action registry | Preview, confirm/cancel, keep, undo |
| `#/inspect/<run-id>?view=claims` | Claim Support | Claim-support projection and exact recorded answer | None; corrective work remains in Actions |
| `#/runtime` | Runtime | Engine health and local model inventory | None |
| `#/compare` | Run comparison picker | None until two Run IDs are supplied | None |
| `#/compare/<run-a>/<run-b>` | Run comparison | Recorded Run-difference projection | None |
| `#/compare/matrix[/<experiment-id>]` | Experiment matrix | Result list/detail and on-demand cell evidence | None |

## Evidence boundaries

- Delivered context is not automatically influential.
- A comparison describes what changed; only a controlled intervention can support a causal claim.
- Screening identifies candidates; exact scoring establishes evidence.
- “Unsupported by supplied materials” is not “false.”
- Missing or failed measurement is “measurement unavailable,” not “unsupported.”
- Generated previews are not Run evidence until an explicit materialization records a child Run.

## Mutation boundaries

Navigation performs reads only. Every model-touching or state-changing operation is attached to an
explicit control, retains its request and Run identity, and leaves the source Run immutable unless a
separate run-scoped keep transaction is confirmed. Undo restores the exact prior selection.
