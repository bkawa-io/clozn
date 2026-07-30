"""investigation_experiment.py -- the C3 controlled-experiment executor (core slice).

`run_experiment()` is the ONE place a `clozn.investigation-experiment.v1` document's `analysis` and
`causal_claim` are computed -- mirroring `clozn.analysis.transplant._derive_analysis`'s own "this field
is set HERE and only here" discipline (see that module's docstring: "a caller cannot report a site as
reference-specific by skipping the control, because the control's result is a required input to the
computation, not an optional add-on"). `causal_claim.licensed` can only ever become `True` by flowing
through `_derive_analysis()`'s `effect_specific` -- there is no second code path that sets it, no
caller-suppliable override, and no way to reach `phase: "completed"` without both `arms` (what actually
happened) and `analysis` (what that licenses) having been derived from the SAME four measured arms.

THE FOUR ARMS, AND HOW THIS GENERALIZES `clozn.analysis.transplant`'s FIVE
----------------------------------------------------------------------------
`transplant.run_site()` runs five arms at the GGUF-internal-tensor layer (reference_transplant /
candidate_self_transplant / random_equal_norm / shuffled_layer / no_write_replay) because a residual/ffn/
head WRITE is the primitive being tested. C3's five intervention types are not a tensor write -- they are
a PROMPT-CONTENT edit, a SAMPLER-PARAMETER override, or (refused in this slice) an ADAPTER rescale, driven
through `clozn.replay.replay.replay()`'s `sub.chat()` surface instead of an engine's `/score` write.
Literally invoking `transplant.run_site()` here would be dishonest -- there is no tensor site to write.
What this module composes instead is THE SAME GOVERNING RULE, generalized to the layer these five
interventions actually operate at:

  * `baseline`                     -- the run's OWN recorded reply. Never regenerated.
  * `no_op_replay`                 -- a plain, unmodified regeneration (greedy, or -- for sampler_change,
                                       the SAME derived seed with no overrides). Analogous to
                                       `no_write_replay`: proves regeneration itself reproduces the
                                       original when nothing was changed. `instrument_sane` is exactly
                                       `no_op_replay.matches_baseline`, checked FIRST and gating
                                       everything else, the same way `transplant.py`'s
                                       `candidate_self_transplant` check does.
  * `treatment`                    -- the actual requested change applied.
  * `random_equal_effect_control`  -- a matched-magnitude, NON-targeted perturbation (a same-length span
                                       elsewhere in the same message; unavailable by construction for
                                       `sampler_change`/multi-span `omit_source`, per
                                       `clozn.runs.investigation_experiment`'s own honest notes).
                                       Analogous to `random_equal_norm`: `effect_specific` is `True` only
                                       when `treatment` moved AND this arm did NOT -- transplant.py's own
                                       rule, verbatim.

NAMED CUT: transplant.py's FIVE arms include a separate `shuffled_layer` control (the same vector at a
DIFFERENT site) alongside `candidate_self_transplant` (the SAME state written back, testing the write
path specifically) and `no_write_replay` (no write at all, testing generation stability). This module
merges those last two into ONE `no_op_replay` arm: the replay/message-splice surface has no separate
"write" step to probe independently of "regenerate at all" the way an engine write does, so splitting them
would test the same thing twice. A genuinely separate "displaced" control (the SAME span's content
re-inserted at a DIFFERENT position) is not built in this slice either -- disclosed here, not hidden.

`causal_claim` is ALWAYS present when `phase: "completed"`, and is `licensed: False` with an explicit
"uncontrolled"/"no effect"/"not causally distinguishable" statement in every case except
`effect_specific is True`. A caller reading only `causal_claim.statement` still gets the honest answer
without inspecting `analysis` -- the schema-level separation the owner's brief required is not just
present in the shape, it is the DEFAULT reading.

Duck-typed against `sub` exactly like `clozn.replay.replay.replay()`/`clozn.receipts.span_receipt.
span_receipt()` -- fully unit-testable with a fake substrate, no model, no GPU. This module never itself
connects to or spawns a substrate; that remains the caller's job (a future slice's route or CLI wiring).
"""
from __future__ import annotations

import hashlib
import time

from clozn.replay import span_bridge
from clozn.replay.replay import replay as replay_run
from clozn.runs.investigation_experiment import plan_experiment

from .forced import _matched_length_filler


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def _arm(run_id, reply: str, baseline_reply: str) -> dict:
    out = {"reply_sha256": _sha256(reply), "matches_baseline": reply == baseline_reply}
    if isinstance(run_id, str) and run_id:
        out["run_id"] = run_id
    return out


def _unavailable_arm(reason: str) -> dict:
    return {"available": False, "reason": str(reason or "no control was constructed for this intervention")}


def _failed(plan_doc: dict, message: str, *, stage: str = "generation") -> dict:
    out = {key: value for key, value in plan_doc.items() if key != "phase"}
    out["phase"] = "failed"
    out["generated_at"] = _now_iso()
    out["error"] = {"stage": stage, "message": message}
    return out


def _derive_analysis(arms: dict) -> dict:
    """THE structural gate -- see module docstring. `effect_specific` is computed HERE and ONLY here,
    from `treatment`'s and `random_equal_effect_control`'s own measured `matches_baseline` -- never
    settable any other way, never defaulted when it cannot be honestly computed."""
    reasons: list = []
    instrument_sane = bool(arms["no_op_replay"].get("matches_baseline"))
    if not instrument_sane:
        reasons.append(
            "no_op_replay's reply differs from the original recorded baseline -- the replay pipeline "
            "itself does not reproduce the original run deterministically here, so no other arm's "
            "result is interpretable as caused by the requested change.")
        return {"instrument_sane": False, "reasons": reasons}

    control = arms.get("random_equal_effect_control")
    control_available = isinstance(control, dict) and "reply_sha256" in control
    treatment_moved = not bool(arms["treatment"].get("matches_baseline"))

    analysis: dict = {"instrument_sane": True}
    if not control_available:
        control_reason = control.get("reason") if isinstance(control, dict) else "unavailable"
        reasons.append(
            f"no random_equal_effect_control ran for this intervention ({control_reason}); "
            "effect_specific cannot be computed.")
        analysis["reasons"] = reasons
        return analysis

    control_moved = not bool(control.get("matches_baseline"))
    effect_specific = bool(treatment_moved and not control_moved)
    analysis["effect_specific"] = effect_specific
    if not treatment_moved:
        reasons.append(
            "the treatment arm's reply did not differ from baseline -- there is no observed difference "
            "for a causal claim to attach to.")
    elif control_moved:
        reasons.append(
            "the random equal-effect control ALSO changed the reply -- the observed difference is not "
            "specific to the requested change; this looks like perturbation sensitivity, not a targeted "
            "effect (the same failure mode docs/research/DISTRIBUTED_FUNCTION.md's prior transplant "
            "study found and corrected for with its own random-equal-norm control).")
    else:
        reasons.append(
            "the treatment arm's reply differed from baseline and the random equal-effect control's did "
            "not -- effect-specific by this harness's rule.")
    analysis["reasons"] = reasons
    return analysis


def _derive_observed(arms: dict) -> dict:
    observed: dict = {
        "treatment_reply_differs_from_baseline": not bool(arms["treatment"].get("matches_baseline")),
    }
    control = arms.get("random_equal_effect_control")
    if isinstance(control, dict) and "reply_sha256" in control:
        observed["random_control_reply_differs_from_baseline"] = not bool(control.get("matches_baseline"))
    observed["note"] = (
        "this records what changed under one greedy regeneration per arm -- a factual diff, never a "
        "causal claim. Read causal_claim for what this difference licenses a caller to say.")
    return observed


def _derive_causal_claim(analysis: dict, arms: dict) -> dict:
    if not analysis.get("instrument_sane"):
        return {
            "licensed": False,
            "statement": (
                "uncontrolled: the instrument-sanity control (a plain, unmodified replay) itself "
                "diverged from the original recorded reply, so no arm's result here can be attributed "
                "to the requested change."),
        }
    effect_specific = analysis.get("effect_specific")
    if effect_specific is None:
        return {
            "licensed": False,
            "statement": (
                "uncontrolled: no random equal-effect control could be run for this intervention in "
                "this slice, so an observed difference cannot be distinguished from any similarly-sized "
                "change producing the same shift."),
        }
    if effect_specific:
        return {
            "licensed": True,
            "statement": (
                "the requested change caused this run's answer to differ: the treatment arm's reply "
                "differs from baseline and the matched random-effect control's does not -- the "
                "difference is specific to what was changed, not merely to something having changed by "
                "a similar amount."),
        }
    if not arms["treatment"].get("matches_baseline"):
        return {
            "licensed": False,
            "statement": (
                "uncontrolled: the random equal-effect control also changed the reply, so this "
                "difference cannot be attributed specifically to the requested change."),
        }
    return {
        "licensed": False,
        "statement": "no causal claim to make: the requested change did not alter the greedy reply at all.",
    }


def _run_content_arms(run: dict, resolved: dict, sub) -> "tuple[dict | None, dict | None, str | None]":
    """(no_op_child, treatment_child, control_reply_or_None) for a span-shaped intervention
    (remove_span/replace_span_neutral/omit_source), or (None, None, None) on a generation failure."""
    no_op_child = replay_run(run, {"greedy": True}, sub)
    if no_op_child is None:
        return None, None, None

    replacement = _matched_length_filler if resolved["kind"] == "replace_span_neutral" else None
    messages = run.get("messages") or []
    ablated_messages = span_bridge.excise_spans(messages, resolved.get("spans") or [], replacement=replacement)
    treatment_child = replay_run({**run, "messages": ablated_messages}, {"greedy": True}, sub)
    if treatment_child is None:
        return no_op_child, None, None

    control_reply = None
    control_spans = resolved.get("random_control_spans")
    if control_spans:
        control_messages = span_bridge.excise_spans(messages, control_spans, replacement=replacement)
        control_child = replay_run({**run, "messages": control_messages}, {"greedy": True}, sub)
        if control_child is not None:
            control_reply = control_child.get("response") or ""
    return no_op_child, treatment_child, control_reply


def _run_sampler_arms(run: dict, resolved: dict, sub) -> "tuple[dict | None, dict | None, str | None]":
    """(no_op_child, treatment_child, None) for `sampler_change` -- both arms pinned to the SAME derived
    seed so an observed difference is attributable to the overridden parameter(s), not to seed variance.
    `no_op_replay` passes ONLY the pinned seed (the engine/model's own default temperature/top_p/top_k):
    this slice does not read a run's originally-recorded sampler settings back off the record, so this is
    a "same seed, engine defaults" control, not a "same seed, same original settings" one -- disclosed,
    not hidden. random_equal_effect_control is never available for this kind (see the planner's own note)."""
    seed = span_bridge.derive_seed(run, purpose="investigation_experiment_sampler")
    no_op_child = replay_run(run, {}, sub, sampling_override={"seed": seed})
    if no_op_child is None:
        return None, None, None
    overrides = {"seed": seed, **(resolved.get("sampler_overrides") or {})}
    treatment_child = replay_run(run, {}, sub, sampling_override=overrides)
    if treatment_child is None:
        return no_op_child, None, None
    return no_op_child, treatment_child, None


def run_experiment(run: dict, intervention: dict, sub) -> dict:
    """Plan (via `clozn.runs.investigation_experiment.plan_experiment`, which is ALWAYS consulted first --
    nothing is generated for an ineligible request) and, when eligible, execute the four-arm controlled
    experiment against the live substrate `sub`. Returns the refused/planned document UNCHANGED when
    ineligible; otherwise a `phase: "completed"` (or `"failed"`, on a generation failure) document with
    `arms`/`analysis`/`observed`/`causal_claim` all present and mutually consistent by construction.

    `sub` -- the live substrate, duck-typed exactly like `clozn.replay.replay.replay()`'s own contract
    (`.chat(messages, max_new=, sample=, ...)`). This function never obtains one itself."""
    plan_doc = plan_experiment(run, intervention)
    if plan_doc.get("phase") != "planned":
        return plan_doc

    resolved = plan_doc["plan"]["resolved"]
    baseline_reply = run.get("response") or ""

    if resolved["kind"] == "sampler_change":
        no_op_child, treatment_child, control_reply = _run_sampler_arms(run, resolved, sub)
    else:
        no_op_child, treatment_child, control_reply = _run_content_arms(run, resolved, sub)

    if no_op_child is None:
        return _failed(plan_doc, "no_op_replay generation failed")
    if treatment_child is None:
        return _failed(plan_doc, "treatment generation failed")

    no_op_reply = no_op_child.get("response") or ""
    treatment_reply = treatment_child.get("response") or ""

    arms = {
        "baseline": _arm(run.get("id"), baseline_reply, baseline_reply),
        "no_op_replay": _arm(no_op_child.get("id"), no_op_reply, baseline_reply),
        "treatment": _arm(treatment_child.get("id"), treatment_reply, baseline_reply),
    }
    if control_reply is not None:
        arms["random_equal_effect_control"] = _arm(None, control_reply, baseline_reply)
    else:
        note = resolved.get("random_control_note") or "no control was constructed for this intervention"
        arms["random_equal_effect_control"] = _unavailable_arm(note)

    analysis = _derive_analysis(arms)
    observed = _derive_observed(arms)
    causal_claim = _derive_causal_claim(analysis, arms)

    out = {key: value for key, value in plan_doc.items() if key != "phase"}
    out["phase"] = "completed"
    out["generated_at"] = _now_iso()
    out["arms"] = arms
    out["analysis"] = analysis
    out["observed"] = observed
    out["causal_claim"] = causal_claim
    return out


__all__ = ["run_experiment"]
