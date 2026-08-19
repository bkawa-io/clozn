"""Sequential free-generation execution for one measured influence link.

This module owns orchestration only.  Influence Map remains the authority for the measured link,
span_bridge remains the authority for fresh message-span resolution and surgery, replay() remains the
generation/child-recording path, and model_diff remains the comparison authority.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

from clozn import schemas
from clozn.analysis.model_diff import diff_runs
from clozn.receipts.forced import matched_length_neutral_filler
from clozn.replay import span_bridge
from clozn.analysis.comparison_projection import comparison_projection_from_diff
from clozn.replay.execution_fork import (
    _runtime_projection,
    parent_execution_fingerprint,
    parent_runtime_projection,
)
from clozn.replay.replay import replay as replay_run
from clozn.runs.influence_counterfactual import (
    FILLER_RECIPE,
    RESULT_SCHEMA_VERSION,
    InfluenceCounterfactualInputError,
    build_influence_counterfactual_plan,
)


_SAFE_CODES = {
    "stale_parent", "runtime_identity_unavailable", "runtime_identity_mismatch",
    "worker_identity_unavailable", "source_span_unavailable", "span_address_not_found_or_drifted",
    "recorded_steering_malformed", "steering_reproduction_unavailable", "control_generation_failed",
    "treatment_generation_failed", "specificity_generation_failed", "execution_cancelled",
    "influence_measurement_unavailable", "influence_link_not_found", "message_basis_unavailable",
}


def _reason(code: str, message: str) -> dict:
    return {"code": str(code or "counterfactual_unavailable"), "message": str(message or "counterfactual unavailable")}


def _public_reason(code: str, message: str) -> dict:
    safe_code = code if code in _SAFE_CODES else "counterfactual_unavailable"
    safe_messages = {
        "stale_parent": "the immutable parent changed after planning",
        "runtime_identity_unavailable": "the parent or selected runtime identity is unavailable",
        "runtime_identity_mismatch": "the selected runtime does not match the recorded parent runtime",
        "worker_identity_unavailable": "the selected worker identity is unavailable",
        "source_span_unavailable": "the selected context span could not be resolved",
        "span_address_not_found_or_drifted": "the selected context span is stale or drifted",
        "recorded_steering_malformed": "the recorded steering state is malformed",
        "steering_reproduction_unavailable": "the recorded steering state cannot be reproduced",
        "control_generation_failed": "the unchanged control replay did not produce a child",
        "treatment_generation_failed": "the context treatment replay did not produce a child",
        "specificity_generation_failed": "the specificity control replay did not produce a child",
        "execution_cancelled": "counterfactual execution was cancelled",
        "influence_measurement_unavailable": "the persisted influence measurement is unavailable",
        "influence_link_not_found": "the requested measured influence link is unavailable",
        "message_basis_unavailable": "the selected context span has no message-list basis",
    }
    return {"code": safe_code, "message": safe_messages.get(safe_code, "counterfactual execution was unavailable")}


def _cancelled(cancel_check) -> bool:
    if not callable(cancel_check):
        return False
    try:
        return bool(cancel_check())
    except Exception:
        return False


def _budget(run: Mapping) -> int:
    meta = run.get("meta") if isinstance(run.get("meta"), Mapping) else {}
    value = meta.get("max_tokens")
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    trace = run.get("trace") if isinstance(run.get("trace"), Mapping) else {}
    tokens = trace.get("tokens")
    if isinstance(tokens, list) and tokens:
        return len(tokens)
    return 256


def _runtime_match(parent: Mapping, runtime_identity, worker_identity) -> tuple[bool, dict | None]:
    expected = parent_runtime_projection(parent)
    if expected is None:
        return False, _public_reason("runtime_identity_unavailable", "")
    selected = _runtime_projection(runtime_identity)
    if selected is None:
        return False, _public_reason("runtime_identity_unavailable", "")
    if expected != selected:
        return False, _public_reason("runtime_identity_mismatch", "")
    if not isinstance(worker_identity, Mapping):
        return False, _public_reason("worker_identity_unavailable", "")
    required = ("worker_id", "worker_generation_id", "protocol_version")
    if any(not isinstance(worker_identity.get(key), str) or not worker_identity.get(key) for key in required):
        return False, _public_reason("worker_identity_unavailable", "")
    worker_runtime = worker_identity.get("runtime_key_sha256")
    if worker_runtime is not None and worker_runtime != selected["runtime_key_sha256"]:
        return False, _public_reason("runtime_identity_mismatch", "")
    return True, None


def _changes(plan: Mapping, arm: str, intervention: str) -> dict:
    return {
        "influence_counterfactual": {
            "test_id": plan["test_id"],
            "arm": arm,
            "source_span_id": plan["influence"]["source_span_id"],
            "answer_span_id": plan["influence"]["answer_span_id"],
            "intervention": intervention,
            "relation_to_measurement": plan["intervention"]["relation_to_measurement"],
        },
    }


def _arm_changes(parent: Mapping, plan: Mapping, arm: str, intervention: str) -> dict:
    return _changes(plan, arm, intervention)


def _arm_doc(state: str, child: Mapping | None = None, reason: dict | None = None) -> dict:
    out = {"state": state}
    if isinstance(child, Mapping) and child.get("id"):
        out["run_id"] = child["id"]
    if reason is not None:
        out["reason"] = deepcopy(reason)
    return out


def _run_replay(parent: Mapping, changes: Mapping, sub, *, messages, sampling, budget):
    try:
        return replay_run(
            parent,
            dict(changes),
            sub,
            max_new=budget,
            messages_override=deepcopy(messages),
            sampling_override=deepcopy(sampling) if isinstance(sampling, Mapping) else sampling,
        )
    except Exception:
        # A single arm failure is evidence inside the orchestration result, not permission to erase
        # already-persisted siblings or leak substrate exception text through the HTTP boundary.
        return None


def _reproduction(parent: Mapping, control: Mapping | None, diff: Mapping | None) -> dict:
    if not isinstance(control, Mapping):
        return {"state": "trace_unavailable"}
    parent_text = parent.get("response")
    control_text = control.get("response")
    if not isinstance(parent_text, str) or not isinstance(control_text, str):
        return {"state": "trace_unavailable"}
    if parent_text != control_text:
        return {"state": "diverged"}
    parent_ids = (parent.get("trace") or {}).get("token_ids") if isinstance(parent.get("trace"), Mapping) else None
    control_ids = (control.get("trace") or {}).get("token_ids") if isinstance(control.get("trace"), Mapping) else None
    if (
        isinstance(parent_ids, list) and isinstance(control_ids, list)
        and parent_ids and control_ids
        and all(isinstance(value, int) and not isinstance(value, bool) for value in parent_ids + control_ids)
    ):
        return {"state": "exact_token_and_text" if parent_ids == control_ids else "exact_text_only"}
    return {"state": "exact_text_only"}


def _relationship(comparison: Mapping, target: Mapping | None) -> str:
    if not isinstance(comparison, Mapping):
        return "unavailable"
    if comparison.get("state") == "trace_unavailable":
        return "unavailable"
    view = comparison.get("first_divergence_view")
    if not isinstance(view, Mapping):
        return "unavailable"
    if view.get("state") == "identical":
        return "no_divergence"
    location = view.get("recorded_answer_location")
    location = location.get("a") if isinstance(location, Mapping) else None
    if not isinstance(location, Mapping) or location.get("state") != "exact":
        return "unavailable"
    if not isinstance(target, Mapping):
        return "unavailable"
    start, end = location.get("start"), location.get("end")
    target_start, target_end = target.get("start"), target.get("end")
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in (start, end, target_start, target_end)):
        return "unavailable"
    if end <= target_start:
        return "before_target"
    if start >= target_end:
        return "after_target"
    return "within_target"


def _metadata_only(value, *, key: str | None = None):
    """Strip text-bearing fields from the existing diff projection without changing its facts."""
    if key in {"piece", "a_piece", "b_piece", "text", "response"}:
        return None
    if key == "window":
        return None
    if isinstance(value, Mapping):
        out = {}
        for child_key, child_value in value.items():
            safe = _metadata_only(child_value, key=str(child_key))
            if safe is not None:
                out[child_key] = safe
        return out
    if isinstance(value, list):
        return [safe for item in value
                if (safe := _metadata_only(item)) is not None]
    return deepcopy(value)


def _comparison(parent: Mapping, other: Mapping | None, target: Mapping | None) -> tuple[dict, dict | None]:
    if not isinstance(other, Mapping) or not other.get("id"):
        return {"state": "unavailable", "target_relationship": "unavailable"}, None
    diff = diff_runs(dict(parent), dict(other))
    projection = _metadata_only(comparison_projection_from_diff(parent, other, diff))
    projection["target_relationship"] = _relationship(projection, target)
    return projection, diff


def _observation(intervention: str, decode_matches: bool, reproduction: Mapping,
                 treatment_comparison: Mapping) -> dict:
    relationship = treatment_comparison.get("target_relationship")
    control_exact = reproduction.get("state") == "exact_token_and_text"
    changed = relationship in {"before_target", "within_target"}
    unchanged_through = relationship in {"after_target", "no_divergence"}
    if relationship == "unavailable":
        return {"state": "inconclusive"}
    if control_exact and decode_matches and intervention == "neutralize" and changed:
        return {"state": "recorded_answer_changed_before_or_within_target"}
    if control_exact and unchanged_through:
        return {"state": "recorded_answer_unchanged_through_target"}
    if not changed and not unchanged_through:
        return {"state": "inconclusive"}
    if changed:
        return {"state": "controlled_sensitivity_observed"}
    return {"state": "no_controlled_change_observed"}


def _base_result(parent: Mapping, plan: Mapping, runtime_match: bool, status: str) -> dict:
    measurement = {
        key: deepcopy(plan["influence"][key])
        for key in ("effect", "evidence_state", "clears_floor", "delta_nats", "abs_delta_nats")
        if key in plan["influence"]
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "test_id": plan["test_id"],
        "parent_run_id": parent["id"],
        "influence": {
            "source_span_id": plan["influence"]["source_span_id"],
            "answer_span_id": plan["influence"]["answer_span_id"],
            "measurement": measurement,
        },
        "intervention": deepcopy(plan["intervention"]),
        "execution": {
            "status": status,
            "runtime_identity_match": bool(runtime_match),
            "decode_regime": deepcopy(plan["execution"]["decode_regime"]),
            "steering": deepcopy(plan["execution"].get("steering") or {}),
            "requires_full_reprefill": True,
        },
        "arms": {
            "control": {"state": "not_attempted"},
            "treatment": {"state": "not_attempted"},
            "specificity_control": {"state": "not_attempted"},
        },
        "control_reproduction": {"state": "trace_unavailable"},
        "comparison": {
            "control_vs_treatment": {"state": "unavailable", "target_relationship": "unavailable"},
            "control_vs_specificity": {"state": "unavailable", "target_relationship": "unavailable"},
        },
        "observation": {"state": "inconclusive"},
    }


def _finalize(document: dict) -> dict:
    schemas.validate(document, RESULT_SCHEMA_VERSION)
    return document


def execute_influence_counterfactual(
    parent_run: Mapping,
    sub,
    request=None,
    *,
    runtime_identity=None,
    worker_identity=None,
    reload_parent=None,
    cancel_check=None,
    plan=None,
) -> dict:
    """Rebuild the plan, rebind the current parent, and execute at most three sibling arms."""
    current_plan = build_influence_counterfactual_plan(parent_run, request)
    if not isinstance(plan, Mapping) or plan.get("parent_fingerprint_sha256") != current_plan.get("parent_fingerprint_sha256"):
        # A supplied preview is never execution authority; current_plan is the only plan used below.
        current_plan = build_influence_counterfactual_plan(parent_run, request)

    if current_plan["execution"]["state"] != "ready":
        document = _base_result(parent_run, current_plan, False, "unavailable")
        code = current_plan["execution"].get("reason") or "counterfactual_unavailable"
        document["reasons"] = [_public_reason(code, "")]
        return _finalize(document)

    runtime_ok, runtime_reason = _runtime_match(parent_run, runtime_identity, worker_identity)
    status = "ready" if runtime_ok else "unavailable"
    document = _base_result(parent_run, current_plan, runtime_ok, status)
    if not runtime_ok:
        document["reasons"] = [runtime_reason or _public_reason("runtime_identity_unavailable", "")]
        document["observation"] = {"state": "inconclusive"}
        return _finalize(document)

    current_parent = reload_parent(parent_run["id"]) if callable(reload_parent) else parent_run
    if not isinstance(current_parent, Mapping) or parent_execution_fingerprint(current_parent) != current_plan["parent_fingerprint_sha256"]:
        document["execution"]["status"] = "stale_parent"
        document["reasons"] = [_public_reason("stale_parent", "")]
        return _finalize(document)
    parent_run = current_parent

    resolved = span_bridge.resolve_span_address(dict(parent_run), current_plan["influence"]["source_span_id"])
    source_span = resolved.get("span") if isinstance(resolved, Mapping) and resolved.get("ok") else None
    if not isinstance(source_span, Mapping):
        document["reasons"] = [_public_reason("source_span_unavailable", "")]
        return _finalize(document)
    messages = parent_run.get("messages")
    if not isinstance(messages, list):
        document["reasons"] = [_public_reason("message_basis_unavailable", "")]
        return _finalize(document)
    intervention = current_plan["intervention"]["kind"]
    replacement = matched_length_neutral_filler if intervention == "neutralize" else None
    treatment_messages = span_bridge.excise_spans(
        deepcopy(messages), [dict(source_span)], replacement=replacement)
    specificity_span = None
    if current_plan["specificity_control"].get("state") == "available":
        specificity_span = span_bridge.pick_random_control_span(
            dict(parent_run), dict(source_span),
            extra=f"{current_plan['influence']['answer_span_id']}:{intervention}",
        )
    specificity_messages = (
        span_bridge.excise_spans(deepcopy(messages), [specificity_span], replacement=replacement)
        if isinstance(specificity_span, Mapping) else None
    )
    arm_changes = _arm_changes(parent_run, current_plan, "control", intervention)
    if arm_changes is None:
        document["reasons"] = [_public_reason("steering_reproduction_unavailable", "")]
        return _finalize(document)
    sampling = current_plan["execution"]["decode_regime"].get("sampling")
    sampling_override = deepcopy(sampling) if isinstance(sampling, Mapping) else False
    budget = _budget(parent_run)

    if _cancelled(cancel_check):
        document["execution"]["status"] = "cancelled"
        document["reasons"] = [_public_reason("execution_cancelled", "")]
        return _finalize(document)
    control = _run_replay(
        parent_run, arm_changes, sub, messages=messages, sampling=sampling_override, budget=budget)
    if not isinstance(control, Mapping) or not control.get("id"):
        document["execution"]["status"] = "failed"
        document["arms"]["control"] = _arm_doc("unavailable", reason=_public_reason("control_generation_failed", ""))
        document["arms"]["treatment"] = _arm_doc("not_attempted", reason=_public_reason("control_generation_failed", ""))
        document["arms"]["specificity_control"] = _arm_doc("not_attempted", reason=_public_reason("control_generation_failed", ""))
        document["reasons"] = [_public_reason("control_generation_failed", "")]
        return _finalize(document)
    document["arms"]["control"] = _arm_doc("completed", control)
    # Keep the resource-creation count truthful on every partial early return below; routes use this
    # to distinguish a cancellation/failure after a persisted child from a no-child refusal.
    document["execution"]["children_created"] = 1
    control_diff = diff_runs(dict(parent_run), dict(control))
    document["control_reproduction"] = _reproduction(parent_run, control, control_diff)

    if _cancelled(cancel_check):
        document["execution"]["status"] = "partial_cancelled"
        document["arms"]["treatment"] = _arm_doc("not_attempted", reason=_public_reason("execution_cancelled", ""))
        document["arms"]["specificity_control"] = _arm_doc("not_attempted", reason=_public_reason("execution_cancelled", ""))
        document["reasons"] = [_public_reason("execution_cancelled", "")]
        return _finalize(document)
    treatment_changes = _arm_changes(parent_run, current_plan, "treatment", intervention)
    if treatment_changes is None:
        document["execution"]["status"] = "partial"
        document["arms"]["treatment"] = _arm_doc(
            "unavailable", reason=_public_reason("steering_reproduction_unavailable", ""))
        document["arms"]["specificity_control"] = _arm_doc(
            "not_attempted", reason=_public_reason("steering_reproduction_unavailable", ""))
        document["reasons"] = [_public_reason("steering_reproduction_unavailable", "")]
        document["execution"]["children_created"] = 1
        return _finalize(document)
    treatment = _run_replay(
        parent_run, treatment_changes, sub, messages=treatment_messages,
        sampling=sampling_override, budget=budget)
    if not isinstance(treatment, Mapping) or not treatment.get("id"):
        document["execution"]["status"] = "partial"
        document["arms"]["treatment"] = _arm_doc("unavailable", reason=_public_reason("treatment_generation_failed", ""))
        document["arms"]["specificity_control"] = _arm_doc("not_attempted", reason=_public_reason("treatment_generation_failed", ""))
        document["reasons"] = [_public_reason("treatment_generation_failed", "")]
        document["execution"]["children_created"] = 1
        return _finalize(document)
    document["arms"]["treatment"] = _arm_doc("completed", treatment)
    target = current_plan.get("answer_target")
    treatment_projection, treatment_diff = _comparison(control, treatment, target)
    document["comparison"]["control_vs_treatment"] = treatment_projection

    if _cancelled(cancel_check):
        document["execution"]["status"] = "partial_cancelled"
        document["arms"]["specificity_control"] = _arm_doc("not_attempted", reason=_public_reason("execution_cancelled", ""))
        document["reasons"] = [_public_reason("execution_cancelled", "")]
    elif not current_plan["specificity_control"].get("requested"):
        # The result keeps the fixed arm key for schema stability, but an explicitly disabled
        # specificity control is not an execution failure and must not be reported as one.
        document["arms"]["specificity_control"] = {
            "state": "not_attempted",
            "reason": {"code": "not_requested", "message": "specificity control was not requested"},
        }
    elif not isinstance(specificity_messages, list):
        document["arms"]["specificity_control"] = _arm_doc(
            "unavailable", reason=_public_reason("specificity_generation_failed", ""))
    else:
        specificity_changes = _arm_changes(parent_run, current_plan, "specificity_control", intervention)
        if specificity_changes is None:
            document["arms"]["specificity_control"] = _arm_doc(
                "unavailable", reason=_public_reason("steering_reproduction_unavailable", ""))
            document["execution"]["status"] = "partial"
            specificity_changes = None
        if specificity_changes is None:
            specificity = None
        else:
            specificity = _run_replay(
                parent_run, specificity_changes, sub, messages=specificity_messages,
                sampling=sampling_override, budget=budget)
        if isinstance(specificity, Mapping) and specificity.get("id"):
            document["arms"]["specificity_control"] = _arm_doc("completed", specificity)
            specificity_projection, _specificity_diff = _comparison(control, specificity, target)
            document["comparison"]["control_vs_specificity"] = specificity_projection
        else:
            document["arms"]["specificity_control"] = _arm_doc(
                "unavailable", reason=_public_reason("specificity_generation_failed", ""))
            document["execution"]["status"] = "partial"
    treatment_comparison = document["comparison"]["control_vs_treatment"]
    decode_matches = bool(document["execution"]["decode_regime"].get("matches_recorded_decode"))
    document["observation"] = _observation(
        intervention, decode_matches, document["control_reproduction"], treatment_comparison)
    if document["execution"]["status"] == "ready":
        document["execution"]["status"] = "completed"
    document["execution"]["children_created"] = sum(
        arm.get("state") == "completed" for arm in document["arms"].values()
    )
    return _finalize(document)


# Short conceptual seam for direct domain callers; the route and Test This use the explicit name.
influence_counterfactual = execute_influence_counterfactual


__all__ = ["execute_influence_counterfactual", "influence_counterfactual"]
