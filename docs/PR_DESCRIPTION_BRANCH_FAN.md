# Branch Fan / Road Not Taken

Branch Fan is a bounded backend orchestration action over a recorded run. It selects the first recorded alternatives at one response-token boundary and executes each one as a canonical ForceToken experiment through the shared time-travel kernel: `StateRef` → `resolve_state` → `Generate` → `GenerateExecutionAdapter` → `GeneratedObservation`.

Fanning N alternatives produces **N GeneratedObservations and zero Runs**. A child Run appears only when a caller explicitly materializes one selected observation, which is a separate operation (`materialize_generated_observation`) that promotes already-recorded evidence and never re-runs generation.

The parent remains the immutable baseline. There is no Branch Fan store, experiment object of its own, baseline child, candidate rescoring, sampler probe, attribution measurement, or context intervention.

## What Branch Fan owns

Candidate discovery over the parent's recorded alternatives, the bounded fan size, sequential ordering, cancellation, the shared exact-checkpoint capture, and the composed summary. It does **not** own exactness, reconstruction, generation, comparison, or child creation — those are kernel seams.

Exact execution is attempted first. One checkpoint is captured through the canonical capture seam and reused across every exact candidate, and the unchanged exact control is proven once for the whole fan by the shared execution adapter. A candidate without a numeric recorded token ID resolves under `reconstructed_only`, so exact and reconstructed branches can coexist; each branch reports the `resolution_policy` it asked for, so nothing degrades silently. A supplied or captured checkpoint that turns out stale, expired, or bound to another parent produces a typed refusal — never a quiet reconstructed fallback — and stops scheduling the remaining branches, since they share that precondition.

## Wire contract (`clozn.branch-fan.v2`)

`POST /runs/<id>/branch-fan` accepts only `position` and bounded `limit` (default 3, maximum 4). Candidates come only from the parent's recorded alternatives and retain their recorded order.

Each branch identifies its evidence by `experiment_id` / `arm_id` / `observation_id` and the compact `state_ref` identity payload, with `outcome: exact | reconstructed | unavailable`. There is no `child_run_id`: a completed fan has created nothing to link to. The summary counts observations (`observations_completed`, `exact_observations`, `reconstructed_observations`, `unavailable`, `not_attempted`), not children. A completed fan is **200**, not 201, because no resource was created; cancellation with nothing completed is 409 and an unavailable fan is 422.

Comparison is projected from the observation against the parent's recorded suffix (`clozn.observation-comparison.v1`). No temporary Run is materialized to reach the two-Run `diff_runs`, which would hide child creation behind an implementation detail. Where token evidence is unavailable the projection degrades to the surface comparison rather than guessing a divergence index.

Existing run lineage, historical execution-fork receipts, First Divergence, and model diff remain authoritative for the features that still use them. The Branch Fan response itself is not persisted, and the fan writes no `execution_fork_results` receipts.

## Tests

Candidate filtering and bounded selection, the N-observations/zero-Runs invariant, exact and reconstructed and mixed fidelity, checkpoint reuse and single control proof, typed refusal of stale/contradictory exact state without reconstructed fallback, contradictory recorded token evidence, diverged-control stop-scheduling, cancellation, observation-aware comparison, single-child materialization with correct provenance and no re-generation, route status/error behavior, schema validation, and source-level guards that canonical product code never reaches the retired child-creating executor.
