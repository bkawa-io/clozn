"""The rule engine: `classify(steps) -> summary`, a pure recomputation, never an independent narrative.

Roadmap rule 1 (evidence before narration) and this feature's spec are explicit: `summary` must be
RECOMPUTABLE from `steps[]`. This module is that recomputation and nothing else -- it reads step
`kind`/`status`/`caveats`, buckets them, and applies one fixed precedence rule. It never inspects
`observations` content, never invents a label, and never calls anything a cause without a
`causally_supported` step naming it.

THE ONE RULE THAT MATTERS (confirmed with the feature owner, not a private judgment call)
-------------------------------------------------------------------------------------------
`summary.classification` is `"undetermined"` UNLESS exactly one step reached `causally_supported`, in
which case classification is that step's own `kind`, verbatim -- never a hand-authored label like
"template_drift". Anything weaker than a unique causal finding -- an unexplained mismatch, a correlation,
several entangled causal findings -- reads as `"undetermined"` with the supporting (or limiting) evidence
named in `summary.caveats`. This is deliberately conservative: it can never overclaim, only under-claim,
and under-claiming is the failure mode this whole feature exists to prevent the opposite of.
"""
from __future__ import annotations

from clozn.triage.status import STATES

# Strongest to weakest evidentiary support for a classification (or its absence). Bucket names here are
# the DERIVED reading, not the raw step status -- a raw `matched` step reads as `eliminated` support, a
# raw `mismatched` step reads as `observed` support, matching clozn.triage.status.hypothesis_for exactly.
_PRECEDENCE = (
    "causally_supported", "reproduced", "correlated", "observed", "eliminated",
    "inconclusive", "not_run", "unsupported",
)

_RAW_TO_BUCKET = {"matched": "eliminated", "mismatched": "observed"}


def _bucket_for(status: str) -> str:
    """The summary bucket a step's raw status contributes to. Raw matched/mismatched are translated via
    the same table `clozn.triage.status.hypothesis_for` uses; every other valid status is its own
    bucket (a future intervention step reporting `causally_supported` directly needs no translation)."""
    return _RAW_TO_BUCKET.get(status, status)


def classify(steps: list) -> dict:
    """Recompute a `clozn.triage.v1` `summary` from `steps[]`. Never raises on well-formed input;
    raises ValueError if a step's status is not a member of the controlled enum (garbage in is refused,
    not silently classified)."""
    steps = list(steps or [])
    buckets: dict[str, list[str]] = {name: [] for name in _PRECEDENCE}
    step_caveats: list[str] = []

    for step in steps:
        status = step.get("status")
        if status not in STATES:
            raise ValueError(f"step {step.get('kind')!r} has an invalid status: {status!r}")
        kind = str(step.get("kind") or "?")
        bucket = _bucket_for(status)
        buckets.setdefault(bucket, []).append(kind)
        for caveat in step.get("caveats") or []:
            if caveat not in step_caveats:
                step_caveats.append(caveat)

    caveats = list(step_caveats)

    causal_kinds = sorted(buckets.get("causally_supported") or [])
    if len(causal_kinds) == 1:
        classification = causal_kinds[0]
    elif len(causal_kinds) > 1:
        classification = "undetermined"
        caveats.append(
            f"multiple steps reached causally_supported ({', '.join(causal_kinds)}); controlled "
            "isolation across them was not achieved, so no single classification can be named")
    else:
        classification = "undetermined"

    if classification == "undetermined":
        if buckets.get("observed"):
            caveats.append(
                "observed but not causally isolated: " + ", ".join(sorted(buckets["observed"])) +
                " -- a live candidate cause, not a proven one")
        if buckets.get("correlated"):
            caveats.append(
                "correlated but not causally isolated: " + ", ".join(sorted(buckets["correlated"])))
        if buckets.get("reproduced"):
            caveats.append(
                "reproduced but not isolated to one changed dimension: " +
                ", ".join(sorted(buckets["reproduced"])))

    if not steps:
        confidence_basis = "not_run"
        caveats.append("no steps were executed")
    else:
        confidence_basis = next(
            (name for name in _PRECEDENCE if buckets.get(name)), "not_run")

    summary = {
        "classification": classification,
        "confidence_basis": confidence_basis,
        "caveats": caveats,
        "entangled": len(causal_kinds) > 1,
    }
    for name in ("eliminated", "observed", "correlated", "reproduced", "causally_supported", "inconclusive",
                "not_run", "unsupported"):
        summary[name] = sorted(buckets.get(name) or [])
    return summary
