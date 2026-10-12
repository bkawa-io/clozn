# Coarse-to-fine context-effects scheduling

`measure_context_effects_coarse_to_fine` is an optional orchestration layer on
top of the existing context-effects experiment kernel. It does not change the
default `measure_context_effects` behavior.

```python
from clozn.recipes import (
    ContextEffectsSearchBudget,
    measure_context_effects_coarse_to_fine,
)

result = measure_context_effects_coarse_to_fine(
    run,
    source_ids=source_ids,
    coarse_group_size=4,
    refine_threshold_nats=0.5,
    drilldown_source_ids=[source_ids[7]],
    joint_source_sets=[(source_ids[1], source_ids[3])],
    use_native_screening=True,
    budget=ContextEffectsSearchBudget(
        max_coarse_arms=8,
        max_fine_arms=6,
        max_joint_arms=2,
    ),
    substrate=substrate,
    observation_store=store,
)
```

The scheduler first measures grouped source deletions. It then measures
individual sources inside groups whose absolute effect clears the configured
threshold. Explicit `drilldown_source_ids` enter the fine queue even when their
coarse group is weak, unavailable, or not measured. Explicit
`joint_source_sets` are measured as joint deletions; the scheduler never
pretends singleton effects establish a joint effect.

Budgets limit new intervention measurements, not cache hits. Trusted direct
observations already in `ObservationStore`, or the same condition already
measured during the coarse phase, can be reused without consuming a slot.
`budget_usage.actual_work` reports executed, reused, and skipped work by phase,
includes the unchanged control separately, and gives a total for model
measurements saved by reuse or bounded selection. Reservations made before a
cancellation are reported separately from actual dispatched work.

Every arm separates orchestration from evidence:

- `execution_state` and `execution_disposition` say whether work ran or was
  reused.
- `evidence_status=trusted` requires a completed trusted observation.
- `evidence_status=screening_only` remains untrusted and cannot occupy the
  trusted observation store identity.
- `evidence_status=skipped` means no measurement was made. It is never a weak
  effect or evidence that a source was irrelevant.
- `evidence_status=unavailable` means execution did not produce usable
  evidence. It is distinct from both screening and budget omission.

The built-in adapter uses the ordinary trusted context-effects scoring path by
default. `use_native_screening=True` is an explicit opt-in: missing coarse arms
are screened natively, then every selected missing fine or joint arm is marked
proof-critical and remeasured through the scalar-reference path. Cache hits
remain trusted and free. Unselected coarse screens remain `screening_only` and
never enter the trusted observation store. A caller-supplied adapter cannot be
combined with this flag; its own screening rows are preserved but never
promoted by the scheduler.

## Server integration

This recipe is intentionally not folded into the existing
`/runs/<id>/context-investigation/effects/jobs` request. That endpoint has a
closed `source_ids`-only body and a result schema whose counts mean direct
one-source measurements. Reinterpreting those fields as coarse groups,
screening rows, skipped candidates, or joint arms would silently change its
contract. The current job progress shape also assumes one fixed scoring phase,
while this scheduler discovers its fine phase after coarse results arrive.

The additive product endpoint is now:

```text
POST /runs/<run_id>/context-investigation/effects-search/jobs
GET  /runs/<run_id>/context-investigation/effects-search/jobs/<job_id>
POST /runs/<run_id>/context-investigation/effects-search/jobs/<job_id>/cancel
```

Its job kind is `context_investigation_effects_search`. The closed request is
validated by `clozn.context-investigation-effects-search-job-request.v1` and
requires an explicit budget and one of these policies:

- `trusted_scalar_only`
- `native_screen_then_scalar_refine`

The native policy is an explicit request, not automatic rollout. It requires a
worker that advertises the native related-prompt capability. Native coarse rows
remain screening-only; every selected fine or joint row must be scalar-refined
or the job fails closed. The ordinary scoring path remains scalar by default.

The terminal result is validated by
`clozn.context-investigation-effects-search-job-result.v1`. It carries the
resolved search spec, complete arm states, coverage, experiments, and
`budget_usage.actual_work`. Every arm also exposes `numerical_regime` and
`execution_identity` as stable fields rather than requiring callers to parse
free-form diagnostics. Screening rows retain their native identity, selected
exact refinements carry their scalar-reference identity, and skipped work uses
explicit nulls. Reused rows preserve the identity of the original measurement.
At publication, the job associates only completed,
trusted control and singleton observations with the canonical Context
Investigation experiment. Group, joint, screening-only, unavailable, and
skipped rows stay out of the reader projection. This lets Studio refresh the
existing reader without teaching it to treat search rows as direct evidence.

Progress uses real scheduler boundaries: planning, coarse scoring, refinement
selection, fine/joint scoring, and reader publication. It does not synthesize
per-arm percentages while one semantic model batch is in flight.
