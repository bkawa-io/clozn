"""Assembles a complete, schema-validated ``clozn.triage.v1`` artifact from a baseline/candidate run pair.

This is the one place that (a) knows the full triage order from
notes/agent_roadmap/05-automatic-regression-triage.md, (b) knows which of its steps this build actually
implements (``clozn.triage.steps.STEP_FAMILIES`` -- the model-free observation families), and
(c) makes every OTHER step in that order an explicit, named ``not_run`` placeholder rather than a silent
omission. Spec: "Unsupported steps are visible" / "report unexecuted steps explicitly" -- this module is
where that promise is kept, not just documented.

COST CONTROLS (spec: --max-runs, --max-seconds, --steps)
-----------------------------------------------------------
Model-free observation steps cost zero runs. A supplied ``clozn.run-change-test.v1`` artifact contributes
its actual child-run count, duration, evidence ids, per-step budget and stop state. Remaining expensive
tiers are explicit ``not_run: budget_exhausted`` when that artifact consumed the budget.
"""
from __future__ import annotations

from datetime import datetime, timezone

from clozn import schemas
from clozn.triage import rules
from clozn.triage.steps import STEP_FAMILIES

SCHEMA_VERSION = "clozn.triage.v1"

# Families with no artifact/runner in this invocation. Replay entries are replaced by supplied controlled
# tests; internal localization remains gated on a future qualified bounded runner.
_UNIMPLEMENTED_FAMILIES = {
    "replay": ("template_swap", "context_swap", "sampling_swap"),
    "internal_localization": ("internal_localization",),
}

_NOT_IMPLEMENTED_REASON = (
    "not executed in this invocation: controlled replay needs a supplied clozn.run-change-test.v1 "
    "artifact (the CLI creates one with --deep or --plan), while qualified internal localization still "
    "has no bounded product runner"
)

ALL_FAMILIES = (
    "identity", "context", "sampling", "replay",
    "quant_export", "tool_contract", "internal_localization",
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _budget_doc(max_runs=None, max_seconds=None) -> dict:
    out = {}
    if max_runs is not None:
        out["max_runs"] = max_runs
    if max_seconds is not None:
        out["max_seconds"] = max_seconds
    return out


def _not_run_step(kind: str, *, deep_requested: bool, max_runs=None, max_seconds=None,
                  stop_reason: str | None = None) -> dict:
    reason = _NOT_IMPLEMENTED_REASON
    inputs: dict = {}
    if deep_requested:
        inputs["deep_requested"] = True
        reason = reason + "; --deep was requested but this build still cannot run it"
    if max_runs is not None:
        inputs["max_runs"] = max_runs
    if max_seconds is not None:
        inputs["max_seconds"] = max_seconds
    if stop_reason == "budget_exhausted":
        reason = "budget_exhausted before this step could start"
    return {
        "kind": kind, "status": "not_run", "inputs": inputs,
        "observations": [{"note": reason}], "artifact_refs": [], "cost": {"model_runs": 0},
        "ran": False, "runs_used": 0, "duration_ms": 0, "evidence": [],
        "budget": _budget_doc(max_runs, max_seconds),
        "stop_reason": stop_reason or "not_implemented", "reason": reason,
    }


def unimplemented_steps(family: str, *, deep_requested: bool = False, max_runs=None,
                        max_seconds=None, stop_reason: str | None = None) -> list[dict]:
    """Explicit `not_run` placeholder step(s) for a step family this build does not implement.

    Raises ValueError for an unknown family name -- a typo in `--steps` must fail loudly, never vanish
    into an empty (and therefore falsely reassuring) step list."""
    try:
        kinds = _UNIMPLEMENTED_FAMILIES[family]
    except KeyError:
        raise ValueError(f"unknown step family {family!r}; know: {list(ALL_FAMILIES)}") from None
    return [_not_run_step(kind, deep_requested=deep_requested, max_runs=max_runs, max_seconds=max_seconds,
                          stop_reason=stop_reason)
            for kind in kinds]


_CONTROLLED_KIND = {
    "context": "context_swap",
    "template": "template_swap",
    "sampling": "sampling_swap",
}


def _controlled_steps(document: dict) -> list[dict]:
    """Translate a run-change-test artifact into triage's common step shape."""
    out = []
    for test in document.get("tests") or []:
        if not isinstance(test, dict):
            continue
        kind = _CONTROLLED_KIND.get(test.get("kind"), f"{test.get('kind')}_swap")
        evidence = [
            dict(item) for item in test.get("evidence") or [] if isinstance(item, dict)
        ]
        inputs = {"controlled_test": test.get("kind")}
        match_kind = _dict(document.get("match_criterion")).get("kind")
        if match_kind:
            inputs["match_criterion"] = match_kind
        step = {
            "kind": kind,
            "status": test.get("status", "inconclusive"),
            "inputs": inputs,
            "observations": [
                {"arms": test.get("arms") or [], "qualification": test.get("qualification") or {}}
            ],
            "artifact_refs": [item["run_id"] for item in evidence if item.get("run_id")],
            "cost": {"model_runs": int(test.get("runs_used") or 0)},
            "ran": bool(test.get("ran")),
            "runs_used": int(test.get("runs_used") or 0),
            "duration_ms": int(test.get("duration_ms") or 0),
            "budget": dict(test.get("budget") or {}),
            "evidence": evidence,
            "reason": str(test.get("reason") or "controlled test supplied no reason"),
        }
        if test.get("stop_reason"):
            step["stop_reason"] = test["stop_reason"]
        out.append(step)
    return out


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def build_triage_artifact(*, baseline_run: dict, candidate_run: dict, baseline_run_id: str | None = None,
                          candidate_run_id: str | None = None, source_experiment_id: str | None = None,
                          case_id: str | None = None, families=None, deep: bool = False,
                          max_runs: int | None = None, max_seconds: float | None = None,
                          controlled_test_artifact: dict | None = None,
                          dry_run: bool = False,
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
    budget_exhausted = (
        isinstance(controlled_test_artifact, dict)
        and controlled_test_artifact.get("status") == "budget_exhausted"
    )
    if controlled_test_artifact is not None:
        schemas.validate(controlled_test_artifact, "clozn.run-change-test.v1")
    for family in families:
        if family in STEP_FAMILIES:
            family_steps = STEP_FAMILIES[family](baseline_run, candidate_run)
            for step in family_steps:
                step.setdefault("ran", step.get("status") not in ("not_run", "unsupported"))
                step.setdefault("runs_used", 0)
                step.setdefault("duration_ms", 0)
                step.setdefault("budget", _budget_doc(max_runs, max_seconds))
                step.setdefault("evidence", [])
            steps.extend(family_steps)
        elif family == "replay" and isinstance(controlled_test_artifact, dict):
            steps.extend(_controlled_steps(controlled_test_artifact))
        else:
            steps.extend(unimplemented_steps(
                family, deep_requested=deep, max_runs=max_runs, max_seconds=max_seconds,
                stop_reason="budget_exhausted" if budget_exhausted else None))

    summary = rules.classify(steps)

    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now_iso(),
        "baseline_run_id": resolved_baseline_id,
        "candidate_run_id": resolved_candidate_id,
        "steps": steps,
        "summary": summary,
    }
    if max_runs is not None or max_seconds is not None:
        document["budget"] = _budget_doc(max_runs, max_seconds)
    if controlled_test_artifact is not None:
        document["controlled_tests"] = controlled_test_artifact
        document["execution_status"] = controlled_test_artifact.get(
            "status", "planned" if dry_run else "inconclusive")
    elif dry_run:
        document["execution_status"] = "planned"
    if source_experiment_id:
        document["source_experiment_id"] = source_experiment_id
    if case_id:
        document["case_id"] = case_id

    if validate:
        schemas.validate(document, SCHEMA_VERSION)
    return document
