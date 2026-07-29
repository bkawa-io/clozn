"""restoration_metrics.py -- slice 3.6: HOW WE MEASURE whether an intervention moved a candidate
model's behavior toward a reference model's.

This module is pure measurement over already-collected scoring results: no engine calls, no search, and
no verdicts. Every function here takes plain numbers, plain token-id lists, or already-recorded
`clozn.experiments.suite` cell dicts that some OTHER slice collected, and reports how far a quantity
moved between a "baseline" arm (the candidate model before an intervention) and a "treated" arm (the
candidate model after one), relative to a "reference" arm's own value for that same quantity when one is
available.

THE CENTRAL DESIGN REQUIREMENT
-------------------------------
Next-token argmax is not the only, or even the default, success condition. A regression in JSON validity
must be judged by JSON validity (`structured_output_validity_restoration`), not by whether one token
flipped (`greedy_suffix_match`) -- the two can and do disagree (see this module's own tests). Six
metrics are implemented here, each independently composable, each reporting the same underlying idea
(movement of a candidate quantity toward a reference quantity) in the shape suited to that quantity:

  1. reference_token_logprob_recovery       -- how much of the gap in the reference token's logprob closed.
  2. candidate_token_suppression            -- how much the (wrong) candidate token's logprob dropped.
  3. sequence_nll_movement                  -- the same idea aggregated over a teacher-forced continuation.
  4. assertion_restoration                  -- did a failing experiment-suite assertion start passing.
  5. structured_output_validity_restoration -- did invalid structured output become valid.
  6. greedy_suffix_match                    -- the strictest: exact token-for-token reproduction.

No function here picks "the" primary metric for a comparison. `METRIC_KINDS` names the six, and
`select_primary()` requires the CALLER to say which one is primary, sourced from the case under test
(mirroring `clozn.experiments.suite`'s own predeclared `primary_metric`) -- never a hardcoded default,
and never silently substituted for a metric that was not actually computed for this comparison.

NORMALIZATION IS HONEST, NOT PLAUSIBLE
----------------------------------------
The continuous metrics (1, 2, 3, and the match-fraction folded into 6) share one core primitive,
`_movement()`. When the reference and baseline values already sit within noise of each other, "percent
of gap closed" is not a stable number -- dividing by a near-zero gap explodes or is dominated by float
error. `_movement()` never fabricates that ratio: a near-zero gap reports `state: "degenerate_gap"`, the
raw values, and an explicit note in `omitted` instead of a number. The same discipline applies when the
reference value cannot be observed at all (`state: "reference_unknown"`) or when the arm being measured
never produced a usable baseline/treated value in the first place (`state: "not_computable"`). Only
`state: "measurable"` carries a `gap_closed_fraction`.

DIRECTION, MAGNITUDE, AND NOISE
----------------------------------
Every `_movement()` result reports both a raw, always-comparable `movement` (treated minus baseline, in
the metric's own units -- directly comparable between two arms scored against the identical baseline,
with no re-derivation) and, when the reference is known, a `direction_vs_reference` of
`"toward_reference"`, `"away_from_reference"`, or `"unmoved"` -- computed from DISTANCE to the reference
value, so it stays correct past an overshoot too. `beat_control()` reads two arms' already-computed
results side by side (one typically the real intervention, the other a random or no-op control) and
answers "did the first one move further toward the reference" without recomputing a single gap.

OMIT, NEVER NULL-PAD
----------------------
Every function below follows the same rule as the rest of `clozn.analysis`: a quantity that could not be
honestly computed is an ABSENT key plus a reason recorded in `omitted`, never a `null` and never a
fabricated `0.0` -- a `0.0` here would read as "no movement", a different and false claim from "not
measurable" (see `docs/SEAMS.md` rule 2).

NO CAUSAL VOCABULARY
-----------------------
Like `clozn.analysis.mechanistic_diff`, this module describes MOVEMENT, not a claim about what produced
it. None of its output, and no string anywhere else in this file, may use "caused", "because",
"responsible for", or "localized" -- see `tests/test_restoration_metrics.py`'s own source-level guard,
which enforces this the same way `clozn.analysis.mechanistic_diff`'s own test does.

STDLIB ONLY
-------------
`pyproject.toml` declares `dependencies = []` (docs/SEAMS.md rule 1). Nothing here needs more than plain
Python arithmetic over already-materialized floats and lists, so nothing beyond the standard library is
imported anywhere in this module, at module scope or otherwise.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

METRIC_KINDS = frozenset({
    "reference_token_logprob_recovery",
    "candidate_token_suppression",
    "sequence_nll_movement",
    "assertion_restoration",
    "structured_output_validity_restoration",
    "greedy_suffix_match",
})

_EPS = 1e-9
_ASSERTION_STATUSES = ("pass", "fail", "error", "unscored")


def _is_number(value: Any) -> bool:
    """True for a plain finite int/float -- never for a bool (a bool passed where a logprob or a token
    count is expected is almost certainly a caller mistake, not a legitimate `0`/`1` value)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


# =========================================================================================== core primitive
# Shared machinery behind every continuous metric below. See the module docstring's NORMALIZATION and
# DIRECTION sections for exactly what each reported `state` means and when a field is omitted instead of
# filled with a fabricated number.

def _movement(metric: str, *, reference_value: Any, baseline_value: Any, treated_value: Any) -> dict:
    out: dict = {"metric": metric}
    omitted: list = []

    have_baseline = _is_number(baseline_value)
    have_treated = _is_number(treated_value)
    have_reference = _is_number(reference_value)

    if have_baseline:
        out["baseline_value"] = float(baseline_value)
    else:
        omitted.append({"field": "baseline_value", "reason": "no finite baseline value was provided"})
    if have_treated:
        out["treated_value"] = float(treated_value)
    else:
        omitted.append({"field": "treated_value", "reason": "no finite treated value was provided"})
    if have_reference:
        out["reference_value"] = float(reference_value)
    else:
        omitted.append({"field": "reference_value",
                        "reason": "no finite reference value was provided or observable (e.g. a token "
                                 "that never appears in the reference model's own returned top-k)"})

    if not (have_baseline and have_treated):
        out["state"] = "not_computable"
        out["omitted"] = omitted
        return out

    baseline_value, treated_value = float(baseline_value), float(treated_value)
    movement = treated_value - baseline_value
    out["movement"] = movement
    out["movement_sign"] = (
        "unchanged" if abs(movement) < _EPS else ("increased" if movement > 0 else "decreased"))

    if not have_reference:
        out["state"] = "reference_unknown"
        omitted.append({"field": "direction_vs_reference", "reason": "reference_value is not known"})
        omitted.append({"field": "gap", "reason": "reference_value is not known"})
        omitted.append({"field": "gap_closed_fraction", "reason": "reference_value is not known"})
        out["omitted"] = omitted
        return out

    reference_value = float(reference_value)
    dist_baseline = abs(baseline_value - reference_value)
    dist_treated = abs(treated_value - reference_value)
    if abs(dist_treated - dist_baseline) < _EPS:
        out["direction_vs_reference"] = "unmoved"
    elif dist_treated < dist_baseline:
        out["direction_vs_reference"] = "toward_reference"
    else:
        out["direction_vs_reference"] = "away_from_reference"

    gap = reference_value - baseline_value
    out["gap"] = gap
    if abs(gap) < _EPS:
        out["state"] = "degenerate_gap"
        omitted.append({"field": "gap_closed_fraction",
                        "reason": f"reference and baseline already sit within {_EPS:.0e} of each other; "
                                 "a percent-of-gap-closed ratio would divide by ~zero and is refused "
                                 "rather than reported as a fabricated number"})
    else:
        out["state"] = "measurable"
        out["gap_closed_fraction"] = movement / gap

    out["omitted"] = omitted
    return out


# =============================================================================================== metric 1

def reference_token_logprob_recovery(*, reference_logprob: Any, baseline_logprob: Any,
                                     treated_logprob: Any) -> dict:
    """How much of the gap between the reference model's own logprob for the token it actually produced
    and the CANDIDATE model's logprob for that identical token id (before an intervention) has closed
    after the intervention. `reference_logprob`/`baseline_logprob`/`treated_logprob` are plain
    log-softmax floats read off an already-completed teacher-forced score -- e.g. one position's
    `tokens[i].logprob`, or the matching value read from the other model's `topk` for that token id (see
    `clozn.analysis.mechanistic_diff`'s own `_logprob_of` for exactly that lookup).
    """
    return _movement("reference_token_logprob_recovery", reference_value=reference_logprob,
                     baseline_value=baseline_logprob, treated_value=treated_logprob)


# =============================================================================================== metric 2

def candidate_token_suppression(*, baseline_logprob: Any, treated_logprob: Any,
                                reference_logprob: Any = None) -> dict:
    """How much the candidate model's own (wrong) top-1 token's logprob moved between the baseline and
    treated arms -- a drop (`movement_sign: "decreased"`) is suppression. `reference_logprob`, the
    reference model's logprob for that SAME token id, is optional and often unobservable in practice (the
    token may simply not appear in the reference model's own returned top-k); suppression is still fully
    measurable from the two candidate-side values alone when `reference_logprob` is absent.
    """
    return _movement("candidate_token_suppression", reference_value=reference_logprob,
                     baseline_value=baseline_logprob, treated_value=treated_logprob)


# =============================================================================================== metric 3

def sequence_nll_movement(*, reference_logprobs: Sequence[Any], baseline_logprobs: Sequence[Any],
                          treated_logprobs: Sequence[Any]) -> dict:
    """The same recovery idea as metric 1, aggregated as mean per-token negative log-likelihood over a
    whole teacher-forced continuation instead of a single position. The three sequences must already be
    position-aligned over the identical continuation (same length) -- this function trusts that alignment
    and only checks length and per-position finiteness; it never sees token ids itself. A position
    missing a finite value on any one of the three sides cannot contribute to any of the three sums (the
    aggregate would stop being an apples-to-apples comparison otherwise), so it is dropped from all three
    and counted in `positions_skipped`, never treated as a `0.0` log-probability.
    """
    ref_list = [] if reference_logprobs is None else list(reference_logprobs)
    base_list = [] if baseline_logprobs is None else list(baseline_logprobs)
    treat_list = [] if treated_logprobs is None else list(treated_logprobs)

    if not ref_list or not (len(ref_list) == len(base_list) == len(treat_list)):
        return {
            "metric": "sequence_nll_movement", "state": "not_computable",
            "positions_total": max(len(ref_list), len(base_list), len(treat_list)),
            "omitted": [{"field": "movement",
                        "reason": "reference/baseline/treated logprob sequences must all be the same "
                                 f"non-zero length (a shared teacher-forced continuation); got lengths "
                                 f"{len(ref_list)}, {len(base_list)}, {len(treat_list)}"}],
        }

    n = len(ref_list)
    used_ref, used_base, used_treat, skipped = [], [], [], 0
    for r, b, t in zip(ref_list, base_list, treat_list):
        if _is_number(r) and _is_number(b) and _is_number(t):
            used_ref.append(float(r))
            used_base.append(float(b))
            used_treat.append(float(t))
        else:
            skipped += 1

    if not used_ref:
        return {
            "metric": "sequence_nll_movement", "state": "not_computable",
            "positions_total": n, "positions_used": 0, "positions_skipped": skipped,
            "omitted": [{"field": "movement",
                        "reason": "no position had a finite log-probability on all three sides at once"}],
        }

    reference_nll = -sum(used_ref) / len(used_ref)
    baseline_nll = -sum(used_base) / len(used_base)
    treated_nll = -sum(used_treat) / len(used_treat)
    result = _movement("sequence_nll_movement", reference_value=reference_nll,
                       baseline_value=baseline_nll, treated_value=treated_nll)
    result["positions_total"] = n
    result["positions_used"] = len(used_ref)
    if skipped:
        result["positions_skipped"] = skipped
    return result


# =============================================================================================== metric 4

def _cell_status(cell: Any) -> Any:
    if not isinstance(cell, Mapping):
        return None
    status = cell.get("status")
    return status if status in _ASSERTION_STATUSES else None


def _cell_coordinate(cell: Any) -> Any:
    """The `(suite, case, variant, seed)` identity `clozn.experiments.suite._coordinate()` uses -- or
    `None` when `cell` is not even a mapping, so a malformed cell is reported rather than crashing here."""
    if not isinstance(cell, Mapping):
        return None
    return (cell.get("suite"), cell.get("case"), cell.get("variant"), cell.get("seed"))


def assertion_restoration(*, baseline_cell: Mapping[str, Any], treated_cell: Mapping[str, Any]) -> dict:
    """Did a failing `clozn.experiments.suite` cell start passing. `baseline_cell`/`treated_cell` are
    whole cell dicts in that module's own shape -- `status` in `{"pass", "fail", "error", "unscored"}`,
    identified by the `(suite, case, variant, seed)` coordinate. The two cells are not required to share
    every coordinate field (a baseline arm and a treated arm are usually two different `variant`s of the
    same `suite`/`case`/`seed`); both coordinates are carried through on the result for the caller to
    check.

    `status` is the WORST of a cell's individual assertions (`clozn.testkit.runner`'s own convention), so
    this reports restoration at that same granularity. A caller that wants restoration of one specific,
    predeclared assertion inside a multi-assertion cell should pick that entry out of
    `cell["assertions"]` and drive this same function's `pass`/`fail` vocabulary from its own status
    instead of the whole cell's.
    """
    out: dict = {"metric": "assertion_restoration",
                 "coordinates": {"baseline": _cell_coordinate(baseline_cell),
                                 "treated": _cell_coordinate(treated_cell)}}

    baseline_status = _cell_status(baseline_cell)
    treated_status = _cell_status(treated_cell)
    if baseline_status is None or treated_status is None:
        out["state"] = "not_computable"
        out["omitted"] = [{"field": "restoration_state",
                           "reason": "one or both cells have no recognized status "
                                    f"(expected one of {_ASSERTION_STATUSES})"}]
        return out

    out["baseline_status"] = baseline_status
    out["treated_status"] = treated_status

    if "error" in (baseline_status, treated_status) or "unscored" in (baseline_status, treated_status):
        out["state"] = "not_measurable"
        out["restoration_state"] = "not_measurable"
        out["omitted"] = [{"field": "restoration_state",
                           "reason": f"status went {baseline_status!r} -> {treated_status!r}; an "
                                    "'error' or 'unscored' side means the underlying pass/fail behavior "
                                    "was never actually observed there"}]
        return out

    if baseline_status == "fail" and treated_status == "pass":
        restoration_state = "restored"
    elif baseline_status == "pass" and treated_status == "fail":
        restoration_state = "regressed"
    elif baseline_status == "pass":
        restoration_state = "unchanged_pass"
    else:
        restoration_state = "unchanged_fail"

    out["state"] = "measurable"
    out["restoration_state"] = restoration_state
    out["omitted"] = []
    return out


# =============================================================================================== metric 5

def _json_validity(text: Any, normalized_schema: Any) -> dict:
    from clozn.server import structured_io   # stdlib-only itself; imported lazily to keep this module's
                                              # own import footprint minimal (docs/SEAMS.md rule 1 spirit)
    if not isinstance(text, str):
        return {"valid": False, "reason": f"output is not text (got {type(text).__name__})"}
    if normalized_schema is not None:
        contract = {"mode": "json_schema",
                    "response_format": {"json_schema": {"schema": normalized_schema}}}
    else:
        contract = {"mode": "json_object"}
    try:
        structured_io.parse_output(text, contract)
        return {"valid": True, "reason": None}
    except structured_io.StructuredIOError as exc:
        return {"valid": False, "reason": str(exc)}


def structured_output_validity_restoration(*, baseline_output: Any, treated_output: Any,
                                           json_schema: Any = None) -> dict:
    """Did an invalid structured-output completion become valid. Reuses `clozn.server.structured_io`'s
    own strict JSON parsing and (when `json_schema` is given) its own JSON-Schema-subset validator --
    exactly the notion of "valid" the gateway itself enforces on a live `response_format: json_schema`
    request -- rather than re-deciding what counts as valid JSON here. Without `json_schema`, validity
    means "exactly one strict JSON object", `structured_io`'s own `json_object` mode.
    """
    normalized_schema = None
    if json_schema is not None:
        from clozn.server import structured_io
        try:
            normalized_schema = structured_io.normalize_json_schema(json_schema)
        except structured_io.StructuredIOError as exc:
            return {"metric": "structured_output_validity_restoration", "state": "not_computable",
                    "omitted": [{"field": "restoration_state",
                                "reason": f"json_schema is not a valid clozn structured-I/O schema: {exc}"}]}

    baseline = _json_validity(baseline_output, normalized_schema)
    treated = _json_validity(treated_output, normalized_schema)

    out: dict = {"metric": "structured_output_validity_restoration",
                 "baseline_valid": baseline["valid"], "treated_valid": treated["valid"]}
    if not baseline["valid"]:
        out["baseline_invalid_reason"] = baseline["reason"]
    if not treated["valid"]:
        out["treated_invalid_reason"] = treated["reason"]

    if baseline["valid"] and not treated["valid"]:
        restoration_state = "regressed"
    elif not baseline["valid"] and treated["valid"]:
        restoration_state = "restored"
    elif baseline["valid"]:
        restoration_state = "unchanged_valid"
    else:
        restoration_state = "unchanged_invalid"

    out["state"] = "measurable"
    out["restoration_state"] = restoration_state
    out["omitted"] = []
    return out


# =============================================================================================== metric 6

def _prefix_match(reference_ids: list, candidate_ids: Any) -> Any:
    if candidate_ids is None:
        return None
    candidate_ids = list(candidate_ids)
    match_length = 0
    for a, b in zip(reference_ids, candidate_ids):
        if a != b:
            break
        match_length += 1
    exact_match = match_length == len(reference_ids) and len(candidate_ids) == len(reference_ids)
    return {"length": len(candidate_ids), "match_length": match_length, "exact_match": exact_match,
           "match_fraction": match_length / len(reference_ids)}


def greedy_suffix_match(*, reference_ids: Sequence[int], treated_candidate_ids: Sequence[int],
                        baseline_candidate_ids: Any = None) -> dict:
    """The strictest of the six: does the candidate's own greedy continuation now reproduce the
    reference's continuation token-for-token, not merely assign it a higher log-probability.
    `reference_ids` is the reference model's actual continuation; `treated_candidate_ids`/
    `baseline_candidate_ids` are the candidate model's own greedy-decoded continuation ids after and
    before the intervention. `baseline_candidate_ids` is optional -- when it is absent, only the treated
    arm's own match is reported (`state: "baseline_unknown"`).
    """
    reference_list = [] if reference_ids is None else list(reference_ids)
    if not reference_list:
        return {"metric": "greedy_suffix_match", "state": "not_computable",
                "omitted": [{"field": "restoration_state",
                            "reason": "reference_ids is empty; there is no reference continuation to "
                                     "match against"}]}

    out: dict = {"metric": "greedy_suffix_match", "reference_length": len(reference_list)}

    treated = _prefix_match(reference_list, treated_candidate_ids)
    if treated is None:
        out["state"] = "not_computable"
        out["omitted"] = [{"field": "restoration_state", "reason": "treated_candidate_ids was not provided"}]
        return out
    out["treated"] = treated

    baseline = _prefix_match(reference_list, baseline_candidate_ids)
    if baseline is None:
        out["state"] = "baseline_unknown"
        out["restoration_state"] = "baseline_unknown"
        out["omitted"] = [{"field": "restoration_state",
                           "reason": "baseline_candidate_ids was not provided; only the treated arm's own "
                                    "match is known"}]
        return out
    out["baseline"] = baseline

    if not baseline["exact_match"] and treated["exact_match"]:
        restoration_state = "restored"
    elif baseline["exact_match"] and not treated["exact_match"]:
        restoration_state = "regressed"
    elif baseline["exact_match"]:
        restoration_state = "unchanged_match"
    else:
        restoration_state = "unchanged_no_match"

    out["state"] = "measurable"
    out["restoration_state"] = restoration_state
    out["omitted"] = []
    out["movement"] = _movement("greedy_suffix_match_fraction", reference_value=1.0,
                                baseline_value=baseline["match_fraction"],
                                treated_value=treated["match_fraction"])
    return out


# ======================================================================================== primary metric
# The central design requirement (see module docstring): nothing in this file decides which metric
# matters most for a given case. That choice is always supplied by the caller.

def select_primary(results: Mapping[str, Mapping[str, Any]], *, primary_metric: str) -> dict:
    """Pick ONE already-computed metric result as the primary comparison for this case. `primary_metric`
    must come from the caller -- normally a field on the originating experiment case, not a value picked
    here -- and must both name one of `METRIC_KINDS` and be a key already present in `results` (a dict of
    `{metric_kind: metric_result}` this module's own functions produced). Never falls back to a different
    metric and never raises on a caller mistake; an unresolvable request is reported the same way every
    other refusal in this module is, as a typed `state`.
    """
    if primary_metric not in METRIC_KINDS:
        return {"state": "unknown_primary_metric", "primary_metric": primary_metric,
                "reason": f"{primary_metric!r} is not one of this module's metrics: {sorted(METRIC_KINDS)}"}
    if primary_metric not in results:
        return {"state": "primary_metric_not_computed", "primary_metric": primary_metric,
                "reason": f"{primary_metric!r} was not among the computed results for this comparison: "
                         f"{sorted(results)}"}
    return {"state": "selected", "primary_metric": primary_metric, "result": results[primary_metric],
            "other_metrics": {k: v for k, v in results.items() if k != primary_metric}}


# ============================================================================================ arm compare

def beat_control(arm: Mapping[str, Any], control: Mapping[str, Any]) -> dict:
    """Compare two already-computed `_movement()`-shaped results for the SAME metric -- typically the
    real intervention arm and a random/no-op control arm, both scored against the identical baseline and
    reference -- and report whether `arm` moved further toward the reference than `control` did, reading
    fields both results already carry rather than recomputing a gap or a fraction.
    """
    arm_metric, control_metric = arm.get("metric"), control.get("metric")
    if arm_metric != control_metric:
        return {"state": "incomparable_arms",
                "reason": f"arm metric {arm_metric!r} does not match control metric {control_metric!r}"}

    out: dict = {"metric": arm_metric}
    omitted: list = []

    arm_movement, control_movement = arm.get("movement"), control.get("movement")
    have_movement = _is_number(arm_movement) and _is_number(control_movement)
    if have_movement:
        out["arm_movement"] = arm_movement
        out["control_movement"] = control_movement
        out["movement_margin"] = arm_movement - control_movement
    else:
        omitted.append({"field": "movement_margin",
                        "reason": "one or both arms have no computable raw movement"})

    arm_fraction, control_fraction = arm.get("gap_closed_fraction"), control.get("gap_closed_fraction")
    if _is_number(arm_fraction) and _is_number(control_fraction):
        out["arm_gap_closed_fraction"] = arm_fraction
        out["control_gap_closed_fraction"] = control_fraction
        out["gap_closed_fraction_margin"] = arm_fraction - control_fraction
        out["arm_beat_control"] = arm_fraction > control_fraction
        out["state"] = "comparable"
    else:
        omitted.append({"field": "arm_beat_control",
                        "reason": "one or both arms have no computable gap_closed_fraction (a degenerate "
                                 "gap or an unknown reference on at least one side)"})
        out["state"] = "movement_only" if have_movement else "not_comparable"

    out["omitted"] = omitted
    return out
