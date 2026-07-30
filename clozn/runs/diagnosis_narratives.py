"""clozn/runs/diagnosis_narratives.py -- D2: plain-language "Why?" narratives over D1's own
`clozn.diagnosis-findings.v1` artifact (`clozn.runs.diagnosis_rules`) plus a STRUCTURAL run comparison
(`clozn.analysis.run_diff.compare_runs`, the same module D1's own R12 already composes).

THE BOUNDARY THAT RULES EVERYTHING HERE
-------------------------------------------
Run comparison is STRUCTURAL DIFFERENCE, never causal attribution. `clozn.analysis.run_diff.compare_runs()`
answers "what differs between these two recorded runs" -- a mechanical fact, always true regardless of
whether that difference explains anything. This module may NEVER upgrade a structural difference into a
claim that it explains an output change. "Temperature changed from 0 to 0.8 AND the output diverged" is
TWO observed facts, reported as two separate `registers.observed_changes` entries; asserting that the
temperature change EXPLAINS the divergence is a claim this module is not entitled to make from a diff
alone -- making it would need an actual controlled intervention (the kind `clozn.analysis.transplant`/
`causal_bisect` run, which this module does not have access to and does not fabricate). No quantization or
temperature narrative built from `run_diff` evidence alone ever states or implies causation; a co-occurring
setting change and output change are simply two adjacent observed facts, left for a reader to weigh --
this module does not synthesize a bridge between them that neither the diff nor a D1 finding actually
supports.

THREE STRICTLY SEPARATED REGISTERS
--------------------------------------
`registers.observed_changes` -- every `clozn.analysis.run_diff.compare_runs()` difference, rendered as a
  plain fact sentence. Never ranked. Never causal.
`registers.measured_effects` -- a RANKED list, but "no finding, no factor": every entry names the exact
  `clozn.diagnosis-findings.v1` `rule_id` (a `status == "finding"` entry) it is backed by. `basis` is
  `"intervention"` for the two rules built on an actual measured intervention (R08/R09, `context_answer_
  influence`'s forced_score_intervention) and `"rule_finding"` for the other ten -- still real, evidenced,
  deterministic detections, never a guess, just not literally an intervention. A structural difference with
  NO corresponding D1 finding never appears here, no matter how suggestive it looks next to an output
  change -- see the boundary above.
`registers.plausible_but_unproven` -- a hypothesis explicitly labeled unproven, structurally forbidden
  from carrying a `rank`/`severity`/`confidence`/`rule_id` (the schema has no such properties on this
  entry shape at all) so it can never be ranked above -- or confused with -- a measured effect. Today this
  register has exactly one trigger: identity and generation settings matched the comparison run exactly,
  yet the output still differs. That specific shape -- "same everything we can compare, different answer"
  -- is the textbook signature of sampling sensitivity (an unpinned seed, backend nondeterminism), so the
  narrative NAMES that possibility rather than staying silent, but labels it what it is: unproven, not a
  finding, never promoted.

WHY THIS IS NOT A STORED ARTIFACT, AND WHY IT HAS A SCHEMA ANYWAY
----------------------------------------------------------------------
`narrate()` is a pure function of `(run, comparison_run, findings)` -- the same inputs always reproduce
the same narrative, so nothing here is ever written back to a run record; every call recomputes it fresh,
exactly like `clozn.runs.diagnosis_rules.evaluate()` (D1) and `clozn.runs.perf_diagnosis.
build_performance_report()` already do. Despite that, `clozn.diagnosis-narrative.v1` IS a registered,
validated schema: `clozn.performance-trace.v1` is the exact same "recomputed, never persisted, still
schema-first" precedent already in this codebase, and this document's shape (three closed-vocabulary
registers, an evidence-reference contract, a hard schema-level ban on ranking an unproven entry) is
exactly the kind of contract roadmap rule 7 exists to protect, whether or not a copy of it ever touches
disk.

EVERY SENTENCE LINKS BACK TO EVIDENCE
------------------------------------------
Every entry in every register carries a non-empty `evidence` array of `{"kind": "finding", "rule_id"}` /
`{"kind": "diff_field", "dimension"}` / `{"kind": "text_span", "address_id"}` references -- never a bare,
unlinked sentence. `text_span` references are propagated verbatim from a D1 finding's OWN `evidence`
entries (never recomputed here -- this module does not read `clozn.runs.text_span_addresses` directly;
D1 already resolved those addresses when it built the finding).

NO CAUSAL VOCABULARY
------------------------
Every sentence this module can emit must stay free of causal vocabulary ("because", "caused", "causes",
"causing", "due to", "responsible for", "leads to", "results in", "the reason") -- that is the whole point
of the boundary above. `tests/test_diagnosis_narratives.py` scans this file's own source for exactly
those words, the same guard `clozn/analysis/mechanistic_diff.py` and `clozn/analysis/mech_target.py` use
for their own banned-vocabulary lists.

STDLIB ONLY, DETERMINISTIC, COMPOSES READ-ONLY
----------------------------------------------------
No imports beyond the standard library plus `clozn.analysis.run_diff` and `clozn.runs.diagnosis_rules`
(both composed, never reimplemented, never modified). `narrate()` is byte-deterministic for the same
inputs and the same `generated_at` override -- every set/dict-derived list used in output is sorted
before it is ever placed there, exactly the discipline `diagnosis_rules.evaluate()` already established.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from clozn import schemas
from clozn.analysis import run_diff
from clozn.runs import diagnosis_rules

SCHEMA_VERSION = "clozn.diagnosis-narrative.v1"
FINDINGS_SCHEMA_VERSION = diagnosis_rules.SCHEMA_VERSION

_INTERVENTION_BACKED_RULES = frozenset({"R08", "R09"})
_SEVERITY_RANK = {name: index for index, name in enumerate(diagnosis_rules.SEVERITY_VALUES)}
_CONFIDENCE_RANK = {"pattern_match": 0, "derived": 1, "exact": 2}

_DIMENSION_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("identity.model_sha256", "the model file changed ({value_a} -> {value_b})"),
    ("identity.model_path", "the model path changed"),
    ("identity.template_fingerprint", "the chat template/tokenizer rendering fingerprint changed"),
    ("identity.engine_build", "the engine build changed ({value_a} -> {value_b})"),
    ("identity.clozn_version", "the clozn version changed ({value_a} -> {value_b})"),
    ("generation.", "{key} changed ({value_a} -> {value_b})"),
    ("context.limits.", "{key} changed ({value_a} -> {value_b})"),
    ("context.rendered_prompt_sha256", "the rendered prompt hash changed"),
    ("context.delivered.messages.count", "the delivered message count changed ({value_a} -> {value_b})"),
    ("context.output_cut_off", "output_cut_off changed ({value_a} -> {value_b})"),
    ("context.", "{key} changed"),
    ("output.finish_reason", "finish_reason changed ({value_a} -> {value_b})"),
    ("output.tool_call_status", "tool-call parse status changed ({value_a} -> {value_b})"),
    ("output.response_length_words", "response length changed ({value_a} words -> {value_b} words)"),
    ("output.text", "the response text differs"),
    ("identity.ext.", "{key} changed"),
)


# =============================================================================================== helpers

def _str(value: Any) -> "str | None":
    return value if isinstance(value, str) and value else None


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _finding_ref(rule_id: str) -> dict:
    return {"kind": "finding", "rule_id": rule_id}


def _diff_field_ref(dimension: str) -> dict:
    return {"kind": "diff_field", "dimension": dimension}


def _span_ref(address_id: str) -> dict:
    return {"kind": "text_span", "address_id": address_id}


# ========================================================================= register 1: observed_changes

def _dimension_key(dimension: str) -> str:
    return dimension.rsplit(".", 1)[-1]


def _observed_text(difference: Mapping[str, Any]) -> str:
    dimension = difference.get("dimension") or ""
    kind = difference.get("kind")
    if kind == "unavailable":
        return f"{dimension} could not be compared (evidence unavailable on at least one side)"
    if kind == "diff_failed":
        return f"{dimension} comparison did not complete"
    if kind == "added":
        return f"{dimension} is present only on the compared run"
    if kind == "removed":
        return f"{dimension} is present only on the reference run"
    template = None
    for prefix, candidate in _DIMENSION_TEMPLATES:
        if dimension == prefix or (prefix.endswith(".") and dimension.startswith(prefix)):
            template = candidate
            break
    if template is None:
        template = "{key} changed"
    return template.format(key=_dimension_key(dimension), value_a=difference.get("value_a"),
                           value_b=difference.get("value_b"))


def _build_observed_changes(diff_result: "Mapping[str, Any] | None") -> list:
    if not diff_result or not diff_result.get("ok"):
        return []
    entries = []
    for difference in diff_result.get("differences") or []:
        if not isinstance(difference, Mapping) or not isinstance(difference.get("dimension"), str):
            continue
        dimension = difference["dimension"]
        entries.append({
            "dimension": dimension,
            "kind": difference.get("kind") if difference.get("kind") in
                   ("changed", "added", "removed", "unavailable", "diff_failed") else "changed",
            "text": _observed_text(difference) + ".",
            "evidence": [_diff_field_ref(dimension)],
        })
    return entries


# ========================================================================= register 2: measured_effects

def _measured_rank_key(finding: Mapping[str, Any]) -> tuple:
    severity_rank = _SEVERITY_RANK.get(finding.get("severity"), -1)
    confidence_rank = _CONFIDENCE_RANK.get(finding.get("confidence"), -1)
    return (-severity_rank, -confidence_rank, finding.get("rule_id") or "")


def _measured_text(finding: Mapping[str, Any]) -> str:
    rule_name = finding.get("rule_name") or "finding"
    summary = _str(finding.get("summary")) or "a rule finding was recorded with no summary text"
    return f"{rule_name} ({finding.get('rule_id')}): {summary}"


def _measured_evidence(finding: Mapping[str, Any]) -> list:
    refs = [_finding_ref(finding["rule_id"])]
    for item in finding.get("evidence") or []:
        if isinstance(item, Mapping) and item.get("kind") == "text_span" and isinstance(
                item.get("address_id"), str):
            refs.append(_span_ref(item["address_id"]))
    return refs


def _build_measured_effects(findings_doc: "Mapping[str, Any] | None") -> list:
    if not findings_doc:
        return []
    real_findings = [f for f in findings_doc.get("findings") or []
                     if isinstance(f, Mapping) and f.get("status") == "finding"
                     and isinstance(f.get("rule_id"), str)]
    ranked = sorted(real_findings, key=_measured_rank_key)
    entries = []
    for index, finding in enumerate(ranked, start=1):
        rule_id = finding["rule_id"]
        entries.append({
            "rank": index,
            "basis": "intervention" if rule_id in _INTERVENTION_BACKED_RULES else "rule_finding",
            "rule_id": rule_id,
            "severity": finding.get("severity"),
            "confidence": finding.get("confidence"),
            "text": _measured_text(finding),
            "evidence": _measured_evidence(finding),
        })
    return entries


# =================================================================== register 3: plausible_but_unproven

_SAMPLING_NOTE = ("not backed by a clozn.diagnosis-findings.v1 rule finding -- named as a plausible "
                  "possibility only, never ranked above registers.measured_effects")


def _build_plausible_but_unproven(diff_result: "Mapping[str, Any] | None") -> list:
    if not diff_result or not diff_result.get("ok"):
        return []
    differences = [d for d in diff_result.get("differences") or [] if isinstance(d, Mapping)]
    identity_or_setting_changed = any(
        isinstance(d.get("dimension"), str)
        and (d["dimension"].startswith("identity.") or d["dimension"].startswith("generation."))
        and d.get("kind") not in ("unavailable", "diff_failed")
        for d in differences)
    output_differences = [
        d for d in differences
        if isinstance(d.get("dimension"), str) and d["dimension"].startswith("output.")
        and d.get("kind") == "changed"]
    if identity_or_setting_changed or not output_differences:
        return []
    dimensions = sorted({d["dimension"] for d in output_differences})
    return [{
        "text": "identity and generation settings matched the comparison run exactly, but the output "
               "still differs; this may be sampling-sensitive (an unpinned seed, or backend "
               "nondeterminism) rather than a measured factor.",
        "note": _SAMPLING_NOTE,
        "evidence": [_diff_field_ref(dimension) for dimension in dimensions],
    }]


# ==================================================================================================== headline

def _headline(observed: Sequence[dict], measured: Sequence[dict], plausible: Sequence[dict],
              comparison_available: bool) -> str:
    parts = []
    if comparison_available:
        parts.append(f"{len(observed)} structural difference(s) against the comparison run")
    ranked_text = f"{len(measured)} ranked finding(s)" if measured else "no ranked findings"
    parts.append(ranked_text)
    if plausible:
        parts.append(f"{len(plausible)} plausible-but-unproven note(s)")
    return "; ".join(parts) + "."


# =========================================================================================== public API

def narrate(run: Mapping[str, Any], *, comparison_run: "Mapping[str, Any] | None" = None,
           findings: "Mapping[str, Any] | None" = None, generated_at: "str | None" = None,
           validate: bool = True) -> dict:
    """Build one `clozn.diagnosis-narrative.v1` document over `run` (and, when supplied, `comparison_run`).

    `findings` -- an already-computed `clozn.diagnosis-findings.v1` document -- is used AS GIVEN when
    supplied (so a caller who already ran `diagnosis_rules.evaluate()` once, e.g. this module's own HTTP
    route, never pays for it twice and both halves of a response are guaranteed to agree); otherwise this
    function computes one itself via `diagnosis_rules.evaluate(run, comparison_run=comparison_run)`.

    Independently of `findings`, this function ALSO calls `clozn.analysis.run_diff.compare_runs()` itself
    when `comparison_run` is supplied -- disclosed duplication with D1's own R12 (which only reads the
    identity/generation subset of that same comparison): `registers.observed_changes` needs the FULL
    difference set (context/output included), and both calls are pure, in-memory, and cheap, so recomputing
    is simpler and safer than trying to recover the untruncated result from R12's own filtered finding.

    Pure and deterministic: the same inputs (and the same `generated_at` override) always produce
    byte-identical output. Never raises for a malformed run/comparison_run/findings -- everything here
    degrades to an empty register or an early return, exactly as `diagnosis_rules.evaluate()` already does
    for its own inputs.
    """
    run = run if isinstance(run, Mapping) else {}
    comparison_run = comparison_run if isinstance(comparison_run, Mapping) and comparison_run else None

    findings_doc = findings if isinstance(findings, Mapping) and findings.get(
        "schema_version") == FINDINGS_SCHEMA_VERSION else None
    if findings_doc is None:
        findings_doc = diagnosis_rules.evaluate(run, comparison_run=comparison_run, generated_at=generated_at)

    diff_result = run_diff.compare_runs(dict(comparison_run), dict(run)) if comparison_run is not None else None

    observed = _build_observed_changes(diff_result)
    measured = _build_measured_effects(findings_doc)
    plausible = _build_plausible_but_unproven(diff_result)

    run_id = _str(run.get("id")) or "?"
    document: dict = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at if generated_at is not None else _now_iso(),
        "run_id": run_id,
        "comparison_available": comparison_run is not None,
        "findings_schema_version": FINDINGS_SCHEMA_VERSION,
        "headline": _headline(observed, measured, plausible, comparison_run is not None),
        "registers": {
            "observed_changes": observed,
            "measured_effects": measured,
            "plausible_but_unproven": plausible,
        },
        "summary": {"counts": {"observed_changes": len(observed), "measured_effects": len(measured),
                               "plausible_but_unproven": len(plausible)}},
    }
    if comparison_run is not None:
        comparison_id = _str(comparison_run.get("id"))
        if comparison_id:
            document["comparison_run_id"] = comparison_id
    if validate:
        schemas.validate(document)
    return document
