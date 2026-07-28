"""analysis/run_diff.py -- RUN-DIFF: "what changed between run A and run B, and why might the answer have
changed?" (agent roadmap feature 10, "What changed"). Pure, run-record-in / dict-out, never raises, no
model/GPU/network -- the same discipline clozn.analysis.model_diff already follows, whose diff_runs() this
module EMBEDS for the output/token dimension rather than reimplementing (see _output_diff).

Produces the clozn.run-diff.v1 artifact (clozn/schemas/defs/clozn.run-diff.v1.json): a `differences` list
covering identity, generation configuration, context, and output, plus a `findings` list of small,
evidence-capped classifications derived FROM those differences. Comparison, not generation: every value
compared here already lives on the stored run record (clozn.runs.identity, clozn.runs.context_receipt,
clozn.receipts.bundle.REPRO_META_KEYS, clozn.analysis.model_diff) -- this module's whole job is comparing
and labeling data that already exists, not capturing anything new.

WHAT THIS COMPARES
-------------------
- identity.*   clozn.runs.identity's fixed fields (model_path, model_sha256, template_fingerprint,
               engine_build, clozn_version) via _identity_diff, plus a FORWARD-COMPATIBLE walk of
               identity["ext"] via _ext_diff (see below) -- the part of the design the roadmap task
               called central.
- generation.* clozn.receipts.bundle.REPRO_META_KEYS' sampling-relevant subset, read off run["meta"].
- context.*    clozn.runs.context_receipt's CURRENT (partial) shape: delivered message count, a
               compare-time SHA-256 of the survived final_prompt text (feature 06 has not shipped a
               stored rendered-prompt hash yet -- this hashes what the run already carries rather than
               waiting on that), and the limits block (prompt/context/max/generated token counts).
- output.*     finish_reason, response length in words, output_contract.outcome.status (tool-call parse
               status -- see _tool_call_status for the shape this reads, verified against
               tests/test_openai_structured_client_compat.py rather than guessed), and an EMBEDDED
               clozn.analysis.model_diff.diff_runs() result for token-level detail when the two replies
               are not textually identical.

FORWARD-COMPATIBLE identity["ext"] DIFFING
--------------------------------------------
_ext_diff never hardcodes a facet name (no special-casing of "engine_artifact"/"adapter"/"machine" --
features 01/03/09's respective identity_providers/ facets). It walks `sorted(set(ext_a) | set(ext_b))` --
every namespace present on EITHER run -- and for each:

  * present on only one side -> one "added"/"removed" difference, the whole sub-dict as its value.
  * present on both, unequal -> recurses one dict-key level at a time (bounded to _EXT_MAX_DEPTH) so a
    namespace's OWN fields get their own dimension strings (identity.ext.adapter.strength, not just
    identity.ext.adapter as one opaque blob) without this module knowing what "strength" means.
  * present on both, equal -> no entry (silence is the correct "unchanged" signal here, same omission
    discipline clozn.runs.identity_ext already uses).
  * anything that raises while being compared (unhashable/malformed facet content) -> one "diff_failed"
    entry for THAT namespace only, never an exception that kills the whole comparison -- mirrors
    identity_ext.collect()'s own "a broken provider costs its own facet, never the run" rule.

Every namespace in the union produces an entry or an asserted-equal silence: nothing is dropped, matching
clozn.run-diff.v1.json's own contract ("A reader that does not recognise a namespace must preserve it, not
drop it," carried over verbatim from clozn.run-identity.v1.json). Unknown-namespace differences are never
read by _findings_from -- a differ with no semantic model of a facet has no business classifying what it
means; they surface only in the raw `differences` list. No renderer-registry seam is built for this
(considered and rejected): the generic walk above already satisfies every acceptance criterion with zero
knowledge of facets that do not exist yet.

RANKING IS PRESENTATION, NEVER EVIDENCE
------------------------------------------
_RANK_ORDER is a static, named, inspectable tuple of dimension-prefixes (a model change ranks above a
sampling change ranks above a context omission -- domain priors about which axis usually dominates
behavior, NOT a score computed from this run pair's actual behavior). `rank` is assigned only on
`differences[]` entries; `findings[]` entries carry no rank field AT ALL -- there is no field on a finding
a caller could mistake for evidence strength beyond `status`, and `status` is set independently by
_findings_from, capped at "observed"/"correlated" without a replay. Rank and evidence-status are
structurally different lists' fields, not two properties of the same object, which is what keeps a
low-ranked identity.ext.* difference from ever being laundered into looking causally proven merely by
sort position, and a top-ranked identity.model_sha256 difference from being auto-promoted past "observed"
either.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from clozn.analysis import model_diff

SCHEMA_VERSION = "clozn.run-diff.v1"

_EXT_MAX_DEPTH = 3   # beyond this, a nested ext value is compared wholesale (opaque) rather than recursed

_GENERATION_KEYS = (
    "temperature", "top_p", "top_k", "repetition_penalty", "no_repeat_ngram_size",
    "max_tokens", "seed", "n_ctx", "sampler_mode", "sampling", "stop",
)   # top_k/stop are speculative -- not in clozn.receipts.bundle.REPRO_META_KEYS today; harmless to check
    # for (a fixed field missing on both sides is simply omitted, never a false claim).

_RANK_ORDER = (
    "identity.model_sha256",
    "identity.model_path",
    "identity.template_fingerprint",
    "identity.engine_build",
    "identity.clozn_version",
    "generation.",
    "context.",
    "output.finish_reason",
    "output.tool_call_status",
    "output.response_length_words",
    "output.text",
    "identity.ext.",
)

_RANKING_NOTE = (
    "presentation-only sort key over dimension CATEGORIES (a model swap is definitionally the largest "
    "lever available; a sampling change usually swamps a context omission) -- not a score computed from "
    "this run pair's actual behavior, and it never touches findings[].status. An identity.ext.<namespace> "
    "difference ranks last by construction: this differ has no semantic model of a facet it has never "
    "seen, so ranking it confidently would overclaim."
)


# ------------------------------------------------------------------------------------------ tiny coercers

def _dict(x) -> dict:
    return x if isinstance(x, dict) else {}


# --------------------------------------------------------------------------------------- generic scalar diff

def _scalar_diff(dimension: str, a: dict, b: dict, key) -> dict | None:
    """Presence-based diff of one key: omitted (None) when absent on BOTH sides -- there is nothing to
    report, and reporting one would clutter every comparison with fields that are simply never populated
    (e.g. identity.engine_build today). Presence is checked with `key in mapping`, never truthiness, so a
    legitimate falsy value (temperature=0.0, seed=0) is never misread as absent."""
    a_has, b_has = key in a, key in b
    if not a_has and not b_has:
        return None
    if a_has and b_has:
        if a[key] == b[key]:
            return None
        return {"dimension": dimension, "kind": "changed", "value_a": a[key], "value_b": b[key]}
    if b_has:
        return {"dimension": dimension, "kind": "added", "value_b": b[key]}
    return {"dimension": dimension, "kind": "removed", "value_a": a[key]}


def _content_availability_diff(dimension: str, value_a, value_b, *, note: str) -> list[dict]:
    """Like _scalar_diff, but for a value that may be genuinely UNCAPTURABLE (a legacy/light-tier run, or
    -- once feature 06 ships graduated privacy modes -- a restrictive retention policy) rather than merely
    absent-because-irrelevant. `None` on exactly one side is 'unavailable', never silently read as 0/equal
    -- the spec's own privacy rule: compare what evidence exists and label the limitation, never invent."""
    if value_a is None and value_b is None:
        return []
    if value_a is None or value_b is None:
        entry = {"dimension": dimension, "kind": "unavailable", "note": note}
        if value_a is not None:
            entry["value_a"] = value_a
        if value_b is not None:
            entry["value_b"] = value_b
        return [entry]
    if value_a == value_b:
        return []
    return [{"dimension": dimension, "kind": "changed", "value_a": value_a, "value_b": value_b}]


# ---------------------------------------------------------------------------------------- identity + ext

def _identity_diff(run_a: dict, run_b: dict) -> list[dict]:
    id_a, id_b = _dict(run_a.get("identity")), _dict(run_b.get("identity"))
    out = []
    for key in ("model_path", "model_sha256", "template_fingerprint", "engine_build", "clozn_version"):
        entry = _scalar_diff(f"identity.{key}", id_a, id_b, key)
        if entry:
            out.append(entry)
    out += _ext_diff(_dict(id_a.get("ext")), _dict(id_b.get("ext")))
    return out


def _ext_diff(ext_a: dict, ext_b: dict) -> list[dict]:
    """The forward-compatible walk described in the module docstring. Never hardcodes a namespace name;
    isolates a broken namespace's failure to one 'diff_failed' entry rather than raising."""
    out = []
    for namespace in sorted(set(ext_a) | set(ext_b)):
        dimension = f"identity.ext.{namespace}"
        try:
            out.extend(_diff_ext_value(dimension, ext_a.get(namespace), ext_b.get(namespace), depth=0))
        except Exception as exc:      # noqa: BLE001 -- one broken namespace must not cost the whole diff
            out.append({"dimension": dimension, "kind": "diff_failed",
                        "note": f"{type(exc).__name__}: {exc}"})
    return out


def _diff_ext_value(dimension: str, a, b, *, depth: int) -> list[dict]:
    a_present, b_present = a is not None, b is not None
    if not a_present and not b_present:
        return []
    if a_present and not b_present:
        return [{"dimension": dimension, "kind": "removed", "value_a": a}]
    if b_present and not a_present:
        return [{"dimension": dimension, "kind": "added", "value_b": b}]
    if a == b:
        return []
    if depth >= _EXT_MAX_DEPTH or not (isinstance(a, dict) and isinstance(b, dict)):
        # opaque compare: a scalar/list, or a dict nested deeper than we'll recurse into -- still fully
        # reported (nothing dropped), just not decomposed key-by-key past this point.
        return [{"dimension": dimension, "kind": "changed", "value_a": a, "value_b": b}]
    out = []
    for key in sorted(set(a) | set(b)):
        out.extend(_diff_ext_value(f"{dimension}.{key}", a.get(key), b.get(key), depth=depth + 1))
    return out


# ------------------------------------------------------------------------------------- generation config

def _generation_diff(run_a: dict, run_b: dict) -> list[dict]:
    meta_a, meta_b = _dict(run_a.get("meta")), _dict(run_b.get("meta"))
    out = []
    for key in _GENERATION_KEYS:
        entry = _scalar_diff(f"generation.{key}", meta_a, meta_b, key)
        if entry:
            out.append(entry)
    return out


# ------------------------------------------------------------------------------------------------ context

def _msg_count(context_receipt: dict):
    messages = _dict(context_receipt.get("delivered")).get("messages")
    return len(messages) if isinstance(messages, list) else None


def _rendered_prompt_hash(context_receipt: dict):
    """SHA-256 of the SURVIVED final_prompt text, computed HERE at compare time from data the run already
    carries (clozn.runs.context_receipt.build_context_receipt's own `survived.final_prompt`). Feature 06's
    spec wants a stored `rendered.sha256`; that field does not exist on disk yet, so this hashes the text
    that IS captured rather than waiting on it -- never a claim that the artifact itself stores a hash."""
    final_prompt = _dict(context_receipt.get("survived")).get("final_prompt")
    if not isinstance(final_prompt, str) or not final_prompt:
        return None
    return hashlib.sha256(final_prompt.encode("utf-8")).hexdigest()


def _context_diff(run_a: dict, run_b: dict) -> list[dict]:
    cr_a, cr_b = _dict(run_a.get("context_receipt")), _dict(run_b.get("context_receipt"))
    out = []

    out += _content_availability_diff(
        "context.delivered.messages.count", _msg_count(cr_a), _msg_count(cr_b),
        note="delivered message content was not captured for this run -- comparing what evidence "
             "exists rather than assuming an empty conversation")

    out += _content_availability_diff(
        "context.rendered_prompt_sha256", _rendered_prompt_hash(cr_a), _rendered_prompt_hash(cr_b),
        note="the rendered prompt text was not captured for this run -- comparing hashes/counts only")

    limits_a, limits_b = _dict(cr_a.get("limits")), _dict(cr_b.get("limits"))
    for key in ("prompt_tokens", "context_window_tokens", "requested_max_tokens", "generated_tokens"):
        entry = _scalar_diff(f"context.limits.{key}", limits_a, limits_b, key)
        if entry:
            out.append(entry)

    entry = _scalar_diff("context.output_cut_off", cr_a, cr_b, "output_cut_off")
    if entry:
        out.append(entry)

    return out


# ------------------------------------------------------------------------------------------------- output

def _tool_call_status(run: dict) -> str | None:
    """output_contract.outcome.status ('parsed' | 'error'), the shape
    tests/test_openai_structured_client_compat.py exercises against a real gateway path (verified
    empirically before writing this, not guessed) -- clozn.server.structured_io's evidence dict, journaled
    verbatim onto the run at clozn/runs/store.py's `record(output_contract=...)`. `None` when the run
    carried no output_contract at all (no structured/tool call was requested) -- absence here is a
    legitimate 'not applicable', not a missing measurement, so it is never surfaced as 'unavailable'."""
    status = _dict(_dict(run.get("output_contract")).get("outcome")).get("status")
    return status if isinstance(status, str) and status else None


def _output_diff(run_a: dict, run_b: dict) -> list[dict]:
    out = []

    entry = _scalar_diff("output.finish_reason", run_a, run_b, "finish_reason")
    if entry:
        out.append(entry)

    status_a, status_b = _tool_call_status(run_a), _tool_call_status(run_b)
    if (status_a is not None or status_b is not None) and status_a != status_b:
        entry = {"dimension": "output.tool_call_status", "kind": "changed"}
        if status_a is not None:
            entry["value_a"] = status_a
        if status_b is not None:
            entry["value_b"] = status_b
        out.append(entry)

    words_a = len(str(run_a.get("response") or "").split())
    words_b = len(str(run_b.get("response") or "").split())
    if words_a != words_b:
        out.append({"dimension": "output.response_length_words", "kind": "changed",
                    "value_a": words_a, "value_b": words_b})

    # Embed, never reimplement: clozn.analysis.model_diff.diff_runs() already does the honesty-labeled
    # token-level work (common prefix, first divergence, b_was_alternative_in_a, char_similarity). Only
    # surfaced as a difference when the two replies are not textually identical -- an identical pair has
    # nothing to add here beyond what output.response_length_words already showed as unchanged.
    try:
        token_diff = model_diff.diff_runs(run_a, run_b)
    except Exception as exc:      # noqa: BLE001 -- isolate to this one dimension, never the whole compare
        out.append({"dimension": "output.text", "kind": "diff_failed",
                    "note": f"model_diff.diff_runs failed: {type(exc).__name__}: {exc}"})
        return out
    if token_diff.get("ok") and _dict(token_diff.get("summary")).get("identical") is False:
        out.append({"dimension": "output.text", "kind": "changed", "evidence": [token_diff]})

    return out


# ---------------------------------------------------------------------------------------------- findings

def _findings_from(differences: list[dict]) -> list[dict]:
    """Small, explicit classification rules over `differences` -- never a learned/aggregate score. Every
    finding is independently derived from its OWN named dimension(s); multiple simultaneous findings are
    listed separately, never collapsed into one root cause (spec non-goal). `status` is always "observed"
    here -- upgrading to "causally_supported" requires a replay actually demonstrating it (see
    plan_replay), which this pure function cannot do. identity.ext.* dimensions are deliberately never
    inspected: this function only ever looks at named, understood dimension strings."""
    by_dim = {d["dimension"]: d for d in differences if isinstance(d, dict) and d.get("dimension")}
    findings = []

    model_dims = [d for d in ("identity.model_sha256", "identity.model_path") if d in by_dim]
    if model_dims:
        findings.append({"classification": "model_changed", "status": "observed",
                         "summary": "The model changed between run_a and run_b.", "dimensions": model_dims})

    if "identity.template_fingerprint" in by_dim:
        findings.append({"classification": "template_changed", "status": "observed",
                         "summary": "The chat template and/or tokenizer rendering changed.",
                         "dimensions": ["identity.template_fingerprint"]})

    sampling_dims = sorted(d for d in by_dim if d.startswith("generation."))
    if sampling_dims:
        names = ", ".join(d.rsplit(".", 1)[-1] for d in sampling_dims)
        findings.append({"classification": "sampling_changed", "status": "observed",
                         "summary": f"{len(sampling_dims)} sampling parameter(s) changed: {names}.",
                         "dimensions": sampling_dims})

    ctx_count = by_dim.get("context.delivered.messages.count")
    if ctx_count and ctx_count.get("kind") == "changed":
        a, b = ctx_count.get("value_a"), ctx_count.get("value_b")
        if isinstance(a, int) and isinstance(b, int) and b < a:
            findings.append({"classification": "context_omission", "status": "observed",
                             "summary": f"{a - b} fewer message(s) were delivered in run_b than in run_a.",
                             "dimensions": ["context.delivered.messages.count"]})

    fr = by_dim.get("output.finish_reason")
    if fr:
        if fr.get("value_b") == "length" and fr.get("value_a") != "length":
            findings.append({"classification": "output_truncated", "status": "observed",
                             "summary": "run_b's output ended at the max-token limit.",
                             "dimensions": ["output.finish_reason"]})
        elif fr.get("value_a") == "length" and fr.get("value_b") != "length":
            findings.append({"classification": "output_truncated", "status": "observed",
                             "summary": "run_a's output ended at the max-token limit; run_b's did not.",
                             "dimensions": ["output.finish_reason"]})

    tc = by_dim.get("output.tool_call_status")
    if tc:
        va, vb = tc.get("value_a"), tc.get("value_b")
        if vb == "error" and va != "error":
            findings.append({"classification": "tool_parse_failed", "status": "observed",
                             "summary": "Tool-call output parsing failed in run_b.",
                             "dimensions": ["output.tool_call_status"]})
        elif va == "error" and vb != "error":
            findings.append({"classification": "tool_parse_recovered", "status": "observed",
                             "summary": "Tool-call output parsing failed in run_a but not run_b.",
                             "dimensions": ["output.tool_call_status"]})

    return findings


def _rank_for(dimension: str) -> int:
    for index, prefix in enumerate(_RANK_ORDER):
        if dimension == prefix or (prefix.endswith(".") and dimension.startswith(prefix)):
            return index
    return len(_RANK_ORDER)      # unranked: a dimension this static order does not name (should not
                                 # happen for dimensions this module itself produces; defensive fallback)


# ==================================================================================== the public surface

def compare_runs(run_a: dict, run_b: dict) -> dict:
    """Compare two recorded runs across identity/generation/context/output. Pure and never raises: a
    missing/malformed run yields {"ok": False, "missing": [...], "error": ...} (mirrors
    clozn.analysis.model_diff.diff_runs()'s own error shape -- deliberately NOT clozn.run-diff.v1-shaped,
    since it lacks `differences`/`findings`). A successful comparison always validates against
    clozn.schemas 'clozn.run-diff.v1'."""
    missing = [name for name, r in (("a", run_a), ("b", run_b)) if not isinstance(r, dict) or not r]
    if missing:
        return {"ok": False, "missing": missing,
                "error": "run " + " and ".join(missing) + " missing/unreadable -- nothing to compare"}
    try:
        return _compare(run_a, run_b)
    except Exception as exc:      # noqa: BLE001 -- the never-raise discipline, with a reason attached
        return {"ok": False, "missing": [], "error": f"comparison failed: {type(exc).__name__}: {exc}"}


def _compare(run_a: dict, run_b: dict) -> dict:
    differences = []
    differences += _identity_diff(run_a, run_b)
    differences += _generation_diff(run_a, run_b)
    differences += _context_diff(run_a, run_b)
    differences += _output_diff(run_a, run_b)

    for entry in differences:
        entry.setdefault("evidence", [])
        entry["rank"] = _rank_for(entry["dimension"])
    differences.sort(key=lambda d: (d["rank"], d["dimension"]))

    findings = _findings_from(differences)
    privacy_limited = any(d.get("kind") == "unavailable" for d in differences)

    return {
        "schema_version": SCHEMA_VERSION,
        "ok": True,
        "run_a": run_a.get("id") or "?",
        "run_b": run_b.get("id") or "?",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "privacy_limited": privacy_limited,
        "ranking": {"order": list(_RANK_ORDER), "note": _RANKING_NOTE},
        "differences": differences,
        "findings": findings,
    }


# ------------------------------------------------------------------------------------------ replay planner

_REPLAY_EXECUTION_NOTE = (
    "execution needs a live model/substrate (clozn.replay.replay.replay(), the existing re-run-with-"
    "modified-state primitive) and is NOT performed by this function -- it only decides which of the "
    "spec's three candidate swaps are available and compatible for this run pair. Display run_count/cost "
    "before actually executing any of them."
)


def plan_replay(run_a: dict, run_b: dict, diff_result: dict) -> dict:
    """Model-free replay planner: given a compare_runs() result, propose the spec's minimal swap sequence
    (context, then template, then sampling) and mark each 'available' only when this run pair's OWN
    evidence supports it -- never executes anything (no model, no GPU, no substrate). A coarse planning-
    time availability check, not a guarantee execution will succeed; the actual swap (clozn.replay.replay.
    replay()) does its own compatibility verification when it runs. Pure, never raises."""
    by_dim = {d.get("dimension"): d for d in (diff_result or {}).get("differences", [])
             if isinstance(d, dict)}

    context_available = isinstance(run_a.get("messages"), list) and bool(run_a.get("messages")) \
        and "context.delivered.messages.count" in by_dim
    candidates = [{
        "swap": "context", "order": 1,
        "description": "re-run B with A's delivered context (messages)",
        "available": context_available,
        "note": None if context_available else
                "no context.delivered.messages.count difference, or run_a's messages are not available",
    }]

    template_available = "identity.template_fingerprint" in by_dim
    candidates.append({
        "swap": "template", "order": 2,
        "description": "re-run B rendering A's chat template",
        "available": template_available,
        "note": None if template_available else
                "no identity.template_fingerprint difference detected -- nothing to swap",
    })

    sampling_dims = sorted(d for d in by_dim if d.startswith("generation."))
    candidates.append({
        "swap": "sampling", "order": 3,
        "description": ("re-run B with A's sampling configuration (" + ", ".join(
                        d.rsplit(".", 1)[-1] for d in sampling_dims) + ")") if sampling_dims else
                       "re-run B with A's sampling configuration",
        "available": bool(sampling_dims),
        "note": None if sampling_dims else
                "no generation.* difference detected -- nothing to swap",
    })

    return {
        "run_a": run_a.get("id") or "?",
        "run_b": run_b.get("id") or "?",
        "candidates": candidates,
        "runs_required": sum(1 for c in candidates if c["available"]),
        "note": _REPLAY_EXECUTION_NOTE,
    }
