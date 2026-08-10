# Sampler Sensitivity Probe

This backend-only change adds a bounded, explicit Sampler Sensitivity Probe for recorded runs.

The probe is a controlled debugging observation, not a benchmark, semantic evaluator, optimization
surface, robustness score, or experiment matrix. Planning is deterministic and read-only. Execution
uses the existing exact Execution Fork planner/executor, captures the shared parent checkpoint once,
and runs at most four `nearby_v1` parameter probes plus two optional seed probes sequentially.

`nearby_v1` changes only one parameter per child: temperature down/up or top-p down/up. Parameter
probes preserve the recorded seed. Optional seed probes change only the seed, and parameter and seed
sensitivity remain separate in the result. Greedy or sampler-provenance-incomplete parents fail
closed rather than receiving invented defaults or arbitrary sampling activation.

The new plan and result schemas and POST routes report probe counts, child run IDs, resolved sampler
verification, existing model diffs, First Divergence projections, and offsets from the selected
Execution Fork token boundary. They never introduce a scalar sensitivity/fragility/robustness score
or a semantic conclusion claim.

Recorded sampler provenance is read through the shared public `recorded_sampling_config()` helper;
the existing private spelling remains only as a compatibility alias. Exactness, checkpoint capture,
unchanged-control proof, runtime identity, sampler application, child persistence, and comparison
remain authoritative in the existing Execution Fork and run-diff artifacts. Sampling probes never
fall back to reconstructed replay, scoring, influence measurement, Branch Fan, or automatic
execution.

No sampler-sensitivity store, experiment object, matrix, ambient configuration mutation, or frontend
work was added. Universal Test This gains only an additive `probe_sensitivity` dispatcher when that
backend is available; it does not duplicate recipe generation or execution.
