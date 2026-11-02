# Studio surface contract

Studio consumes one public gateway.

| Family | Purpose |
|---|---|
| `GET /runs` | run and conversation index |
| `GET /runs/<id>` | one immutable recorded run |
| `DELETE /runs/<id>` | transactional deletion |
| `GET /runs/<id>/context-receipt` | recorded context metadata |
| `/runs/<id>/context-investigation...` | Context Effects reader/query, measurement jobs, Lean Context jobs, and optional interaction sampling |
| `POST /v1/chat/completions` | composer generation |
| `POST /cancel` | active worker request cancellation |

Unknown run suffixes return `unknown_run_subroute`; they never fall through to a generic run-shaped
response. Navigation performs reads only. Measurement, reduction, cancellation, and deletion require
explicit controls. Every model-touching operation stays bound to the recorded run and current worker
identity.

Lean Context starts at `POST /runs/<id>/context-investigation/lean-context/jobs`, followed by the
run-scoped job read or cancellation endpoint. The request accepts only `max_new_executions` from 1 to
64 (default 16); answer-change tolerances are fixed product policy. Its result claim is
`best_found_within_budget` with `minimum_proven: false`.
