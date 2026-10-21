# Studio surface contract

Studio consumes one public gateway. The supported backend families are:

| Family | Purpose |
|---|---|
| `GET /runs` | run list |
| `GET /runs/<id>` | one immutable run |
| `DELETE /runs/<id>` | transactional basic deletion |
| `GET /runs/<id>/context-receipt` | recorded context metadata |
| `/runs/<id>/context-investigation...` | Context Effects reader/query, measurement jobs, and bounded Lean Context jobs |
| `/runs/<id>/minimal-context...` | Minimal Context jobs, durable results, save, and cancellation |
| `GET /runs/compare?a=<parent>&b=<child>` | parent/reduced-child comparison |
| `POST /v1/chat/completions` | composer generation |
| `POST /cancel` | active worker request cancellation |

Unknown run suffixes return `unknown_run_subroute`; they do not fall through to a generic run-shaped
response.

Navigation performs reads only. Measurement, reduction, cancellation, save, and delete require explicit
controls. Every model-touching operation remains bound to the recorded run and current worker identity.

Lean Context uses `POST /runs/<id>/context-investigation/lean-context/jobs`, followed by the Run-scoped
job read or cancellation endpoint. The v1 request body is empty: the balanced tolerances and 16-new-
execution budget are fixed product policy, not public tuning fields. Its result claim is
`best_found_within_budget` with `minimum_proven: false`.
