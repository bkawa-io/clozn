"""Assembles a complete, schema-validated ``clozn.triage.v1`` artifact from a baseline/candidate run pair.

This is the one place that (a) knows the full triage order from
notes/agent_roadmap/05-automatic-regression-triage.md, (b) knows which of its steps this build actually
implements (``clozn.triage.steps.STEP_FAMILIES`` -- identity diff and context diff, both model-free), and
(c) makes every OTHER step in that order an explicit, named ``not_run`` placeholder rather than a silent
omission. Spec: "Unsupported steps are visible" / "report unexecuted steps explicitly" -- this module is
where that promise is kept, not just documented.

COST CONTROLS (spec: --max-runs, --max-seconds, --steps)
-----------------------------------------------------------
No step this build executes ever spends a model run (`cost.model_runs` is always 0 for the identity/
context families). `max_runs`/`max_seconds` therefore do not gate anything real yet -- they are accepted,
validated, and threaded into the `not_run` placeholders' `inputs` so a reader can see the budget that
WOULD have applied once a GPU-touching family (replay/quant_export/...) is implemented, rather than
those flags silently doing nothing with no trace at all.
"""
from __future__ import annotations

from datetime import datetime, timezone

from clozn import schemas
from clozn.triage import rules
from clozn.triage.steps import STEP_FAMILIES

SCHEMA_VERSION = "clozn.triage.v1"

# Step families this build's triage order names but does not implement. Each maps to the exact `kind`
# value(s) that spec step would produce once built, so `--steps replay` (say) yields a named, reasoned
# not_run entry instead of nothing. Ordering mirrors the spec's "Triage order" section items 3-6;
# items 1-2 are `clozn.triage.steps.STEP_FAMILIES`, fully implemented and model-free.
_UNIMPLEMENTED_FAMILIES = {
    "replay": ("template_swap", "context_swap", "sampling_swap"),
    "quant_export": ("quant_export_diff",),
    "tool_contract": ("tool_contract_diff",),
    "internal_localization": ("internal_localization",),
}

_NOT_IMPLEMENTED_REASON = (
    "not implemented in this build (roadmap feature 05: controlled-replay, quant/export, tool-contract, "
    "and internal-localization steps need new GPU-touching engine plumbing that was deliberately deferred "
    "-- see the feature's plan for why)"
)

ALL_FAMILIES = tuple(sorted(set(STEP_FAMILIES) | set(_UNIMPLEMENTED_FAMILIES)))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _not_run_step(kind: str, *, deep_requested: bool, max_runs=None, max_seconds=None) -> dict:
    reason = _NOT_IMPLEMENTED_REASON
    inputs: dict = {}
    if deep_requested:
        inputs["deep_requested"] = True
        reason = reason + "; --deep was requested but this build still cannot run it"
    if max_runs is not None:
        inputs["max_runs"] = max_runs
    if max_seconds is not None:
        inputs["max_seconds"] = max_seconds
    return {
        "kind": kind, "status": "not_run", "inputs": inputs,
        "observations": [{"note": reason}], "artifact_refs": [], "cost": {"model_runs": 0},
        "reason": reason,
    }


def unimplemented_steps(family: str, *, deep_requested: bool = False, max_runs=None,
                        max_seconds=None) -> list[dict]:
    """Explicit `not_run` placeholder step(s) for a step family this build does not implement.

    Raises ValueError for an unknown family name -- a typo in `--steps` must fail loudly, never vanish
    into an empty (and therefore falsely reassuring) step list."""
    try:
        kinds = _UNIMPLEMENTED_FAMILIES[family]
    except KeyError:
        raise ValueError(f"unknown step family {family!r}; know: {list(ALL_FAMILIES)}") from None
    return [_not_run_step(kind, deep_requested=deep_requested, max_runs=max_runs, max_seconds=max_seconds)
            for kind in kinds]


def build_triage_artifact(*, baseline_run: dict, candidate_run: dict, baseline_run_id: str | None = None,
                          candidate_run_id: str | None = None, source_experiment_id: str | None = None,
                          case_id: str | None = None, families=None, deep: bool = False,
                          max_runs: int | None = None, max_seconds: float | None = None,
                          validate: bool = True) -> dict:
    """Assemble and (by default) schema-validate one ``clozn.triage.v1`` document.

    `baseline_run_id`/`candidate_run_id` default to the run dicts' own `"id"` field; pass them explicitly
    when a caller has an id the run dict itself might not carry. Raises ValueError if neither source
    yields an id (an artifact must never claim an empty run id) or if `families` names an unknown family.

    `families` selects which step families run/appear, default `ALL_FAMILIES` (identity + context,
    executed for real; the rest, always reported as explicit `not_run`). This is the CLI's `--steps`
    filter's direct backing -- requesting only `["identity", "context"]` yields a report with zero
    `not_run` placeholders at all, e.g. for a fast CI-only pass.
    """
    resolved_baseline_id = baseline_run_id or (baseline_run or {}).get("id")
    resolved_candidate_id = candidate_run_id or (candidate_run or {}).get("id")
    if not resolved_baseline_id:
        raise ValueError("baseline run has no id; pass baseline_run_id explicitly")
    if not resolved_candidate_id:
        raise ValueError("candidate run has no id; pass candidate_run_id explicitly")

    families = list(families) if families is not None else list(ALL_FAMILIES)
    unknown = sorted(set(families) - set(ALL_FAMILIES))
    if unknown:
        raise ValueError(f"unknown step family/families: {unknown}; know: {list(ALL_FAMILIES)}")

    steps: list[dict] = []
    for family in families:
        if family in STEP_FAMILIES:
            steps.extend(STEP_FAMILIES[family](baseline_run, candidate_run))
        else:
            steps.extend(unimplemented_steps(
                family, deep_requested=deep, max_runs=max_runs, max_seconds=max_seconds))

    summary = rules.classify(steps)

    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "baseline_run_id": resolved_baseline_id,
        "candidate_run_id": resolved_candidate_id,
        "steps": steps,
        "summary": summary,
    }
    if source_experiment_id:
        document["source_experiment_id"] = source_experiment_id
    if case_id:
        document["case_id"] = case_id

    if validate:
        schemas.validate(document, SCHEMA_VERSION)
    return document
