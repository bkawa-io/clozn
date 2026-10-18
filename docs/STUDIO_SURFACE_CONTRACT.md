# Studio surface contract

Studio consumes one public gateway. The supported backend families are:

| Family | Purpose |
|---|---|
| `GET /runs` | run list |
| `GET /runs/<id>` | one immutable run |
| `DELETE /runs/<id>` | transactional basic deletion |
| `GET /runs/<id>/context-receipt` | recorded context metadata |
| `/runs/<id>/context-investigation...` | Context Effects reader, query, jobs, and cancellation |
| `/runs/<id>/minimal-context...` | Minimal Context jobs, durable results, save, and cancellation |
| `GET /runs/compare?a=<parent>&b=<child>` | parent/reduced-child comparison |
| `POST /v1/chat/completions` | composer generation |
| `POST /cancel` | active worker request cancellation |

Unknown run suffixes return `unknown_run_subroute`; they do not fall through to a generic run-shaped
response.

Navigation performs reads only. Measurement, reduction, cancellation, save, and delete require explicit
controls. Every model-touching operation remains bound to the recorded run and current worker identity.
