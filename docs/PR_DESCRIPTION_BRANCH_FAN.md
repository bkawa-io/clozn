# Branch Fan / Road Not Taken

Branch Fan is a bounded backend orchestration action over a recorded run. It selects the first recorded alternatives at one response-token boundary, creates ordinary direct child runs sequentially, and projects each child’s existing execution-fork proof and `diff_runs()` comparison.

The parent remains the immutable baseline. There is no Branch Fan store, experiment object, baseline child, candidate rescoring, sampler probe, attribution measurement, context intervention, or frontend work.

Exact execution is attempted first using the same shared checkpoint/plan/execute policy as `compat_fork()`. One captured checkpoint is reused across exact candidates, while unchanged-control proof remains per child. A candidate without a numeric recorded token ID may independently fall back to the existing reconstructed replay; exact and reconstructed children can therefore coexist. Failed exact executions are not silently reconstructed.

`POST /runs/<id>/branch-fan` accepts only `position` and bounded `limit` (default 3, maximum 4). Candidates come only from the parent’s recorded alternatives and retain their recorded order. Existing run lineage, execution-fork receipts, reconstructed-fork evidence, First Divergence, and model diff remain authoritative. The response itself is not persisted.

Cancellation and branch failures are reported per branch, preserving children already created. Exact child traces are persisted only from worker-supplied generation evidence through the shared trace normalizer; missing evidence degrades comparison to `trace_unavailable` without changing exactness.

Tests cover candidate filtering, bounded selection, exact/reconstructed/mixed fidelity, checkpoint reuse, cancellation, lineage, comparison projection, exact trace handoff, route status/error behavior, schema validation, and the absence of unrelated model/worker/analysis work. Studio frontend files remain untouched.
