"""investigation_experiment.py -- the C3 eligibility planner (core slice, item 2).

ONE planner for all five intervention types the owner's audit named (remove a text span, replace a span
with a neutral marker, omit a source, change one sampler setting, detach/scale an adapter) -- not five
special cases living in five call sites. `plan_experiment()` validates a proposed
`clozn.investigation-experiment.v1` `intervention` against one recorded run and returns a document that is
EITHER `phase: "refused"` (a specific typed `Reason`, never a bare "no") or `phase: "planned"` (the exact
arm order and the bridged, replay-executable spec a later executor would run) -- and it does this WITHOUT
generating anything, replaying anything, or touching the run. Model-free, mutation-free, exactly like
`clozn.runs.investigation.build()`'s own "expensive measurements are returned as typed action descriptors;
this function never starts" discipline -- this module is that discipline applied to C3.

Two of the five kinds resolve through `clozn.replay.span_bridge` (item 3): `remove_span`/
`replace_span_neutral` via `resolve_span_address()`, `omit_source` via `resolve_source_spans()`. Their
refusal reasons are exactly the bridge's own typed reasons (span content drift, an unsupported basis, an
unresolvable message index, ...) -- this module does not invent a second vocabulary for the same failure.

`sampler_change` needs no bridge (it never touches prompt content) -- it is eligible on any run that has
at least one message, refused only when the request itself is malformed (`sampler_overrides_empty`).

`adapter_scale` is refused UNCONDITIONALLY in this slice, always with
`adapter_rescale_unavailable_in_planner`. This is a scope boundary, not a bug: rescaling a LoRA adapter
means booting a SECOND engine process with the adapter attached at a different scale (see `clozn diff-
adapter`'s own two-engine-process design, `clozn/cli/commands/diff_adapter.py`) -- a fundamentally
different, much heavier execution model than the in-process `sub.chat()` surface every other intervention
here drives through `clozn.replay.replay.replay()`. The owner's own audit called this "possible", not
"wired for on-demand use"; this planner tells the truth about that gap by name rather than pretending to
support it.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any

SCHEMA_VERSION = "clozn.investigation-experiment.v1"

ARM_ORDER = ("baseline", "no_op_replay", "treatment", "random_equal_effect_control")

_SPAN_KINDS = ("remove_span", "replace_span_neutral")
_KNOWN_KINDS = _SPAN_KINDS + ("omit_source", "sampler_change", "adapter_scale")


def _reason(code: str, message: str) -> dict:
    return {"code": code, "message": message}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _experiment_id(run_id: str, intervention: dict) -> str:
    """A stable-enough id for one (run, intervention) pair -- not persisted anywhere by this module, so
    collision-freedom only needs to hold within one process's lifetime; a content hash is simplest and
    makes two calls with an IDENTICAL request produce the SAME id, which is a useful property for a
    caller diffing repeated plans of the same question."""
    import json
    canonical = json.dumps({"run_id": run_id, "intervention": intervention},
                           sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return "invexp_" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _refused(run_id: str, intervention: dict, reason: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": _experiment_id(run_id, intervention),
        "run_id": run_id,
        "generated_at": _now_iso(),
        "phase": "refused",
        "intervention": intervention,
        "eligibility": {"state": "refused", "reason": reason},
    }


def _planned(run_id: str, intervention: dict, resolved: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": _experiment_id(run_id, intervention),
        "run_id": run_id,
        "generated_at": _now_iso(),
        "phase": "planned",
        "intervention": intervention,
        "eligibility": {"state": "eligible"},
        "plan": {"arm_order": list(ARM_ORDER), "resolved": resolved},
    }


def _plan_span_kind(run: dict, intervention: dict) -> dict:
    from clozn.replay import span_bridge

    kind = intervention["kind"]
    address_id = intervention.get("span_address_id")
    if not isinstance(address_id, str) or not address_id:
        return {"ok": False, "reason": _reason(
            "unsupported_intervention_kind", "span_address_id must be a non-empty string")}
    resolved_span = span_bridge.resolve_span_address(run, address_id)
    if not resolved_span.get("ok"):
        return {"ok": False, "reason": resolved_span["reason"]}
    span = resolved_span["span"]
    control = span_bridge.pick_random_control_span(run, span, extra=kind)
    resolved: dict = {"kind": kind, "spans": [span]}
    if control is not None:
        resolved["random_control_spans"] = [control]
    else:
        resolved["random_control_note"] = (
            "no non-overlapping same-length window exists elsewhere in this message; "
            "random_equal_effect_control will be reported unavailable, and effect_specific will not be "
            "computed for this experiment")
    return {"ok": True, "resolved": resolved}


def _plan_omit_source(run: dict, intervention: dict) -> dict:
    from clozn.replay import span_bridge

    source_id = intervention.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        return {"ok": False, "reason": _reason(
            "unsupported_intervention_kind", "source_id must be a non-empty string")}
    resolved_spans = span_bridge.resolve_source_spans(run, source_id)
    if not resolved_spans.get("ok"):
        return {"ok": False, "reason": resolved_spans["reason"]}
    spans = resolved_spans["spans"]
    resolved: dict = {"kind": "omit_source", "spans": spans}
    if len(spans) == 1:
        control = span_bridge.pick_random_control_span(run, spans[0], extra="omit_source")
        if control is not None:
            resolved["random_control_spans"] = [control]
        else:
            resolved["random_control_note"] = (
                "no non-overlapping same-length window exists elsewhere in this message; "
                "random_equal_effect_control will be reported unavailable")
    else:
        resolved["random_control_note"] = (
            "random_equal_effect_control is only constructed for a single-span intervention in this "
            f"slice; this source resolved to {len(spans)} spans, so no matched control was built -- "
            "effect_specific will not be computed for this experiment")
    return {"ok": True, "resolved": resolved}


_SAMPLER_FIELDS = ("temperature", "top_k", "top_p", "seed", "rep_penalty")


def _plan_sampler_change(run: dict, intervention: dict) -> dict:
    overrides = intervention.get("overrides")
    if not isinstance(overrides, dict) or not any(k in overrides for k in _SAMPLER_FIELDS):
        return {"ok": False, "reason": _reason(
            "sampler_overrides_empty",
            "overrides must set at least one of temperature/top_k/top_p/seed/rep_penalty")}
    resolved = {
        "kind": "sampler_change",
        "sampler_overrides": {k: overrides[k] for k in _SAMPLER_FIELDS if k in overrides},
        "random_control_note": (
            "no principled equal-magnitude-but-untargeted perturbation is defined for a sampler-"
            "parameter change in this slice; random_equal_effect_control will be reported unavailable, "
            "and effect_specific will not be computed for this experiment"),
    }
    return {"ok": True, "resolved": resolved}


def plan_experiment(run: Any, intervention: Any) -> dict:
    """Validate `intervention` (an Intervention-shaped dict, see the schema) against `run` (a run dict
    from `clozn.runs.store.get_run`) and return a `clozn.investigation-experiment.v1` document in phase
    "refused" or "planned". Never executes, never generates, never mutates `run`.

    Raises ValueError only for a caller bug (a `run` that is not a dict with a non-empty "id", or an
    `intervention` that is not a dict at all) -- mirrors `clozn.runs.investigation.build()`'s own
    convention. Every OTHER ineligibility (a bad span address, an empty sampler override, an unsupported
    kind, an adapter request) is a normal, typed `phase: "refused"` document, never an exception."""
    if not isinstance(run, dict) or not isinstance(run.get("id"), str) or not run.get("id"):
        raise ValueError("plan_experiment requires a stored run with a non-empty id")
    if not isinstance(intervention, dict):
        raise ValueError("intervention must be an object")

    run_id = str(run["id"])
    kind = intervention.get("kind")
    if kind not in _KNOWN_KINDS:
        return _refused(run_id, intervention, _reason(
            "unsupported_intervention_kind", f"intervention.kind must be one of {sorted(_KNOWN_KINDS)}"))

    messages = run.get("messages")
    if not isinstance(messages, list) or not messages:
        return _refused(run_id, intervention, _reason(
            "run_has_no_messages", "this run has no messages to point an experiment at"))

    if kind == "adapter_scale":
        return _refused(run_id, intervention, _reason(
            "adapter_rescale_unavailable_in_planner",
            "adapter rescale requires booting a second engine process with the adapter attached at a "
            "different scale (see `clozn diff-adapter`); the on-demand investigation-experiment planner "
            "in this slice only drives the live in-process replay substrate, so this intervention kind "
            "is always refused here, not silently downgraded to a no-op"))

    if kind in _SPAN_KINDS:
        result = _plan_span_kind(run, intervention)
    elif kind == "omit_source":
        result = _plan_omit_source(run, intervention)
    else:
        result = _plan_sampler_change(run, intervention)

    if not result.get("ok"):
        return _refused(run_id, intervention, result["reason"])
    return _planned(run_id, intervention, result["resolved"])


__all__ = ["ARM_ORDER", "SCHEMA_VERSION", "plan_experiment"]
