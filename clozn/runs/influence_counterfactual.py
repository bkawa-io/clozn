"""Pure planning for Influence -> Counterfactual Confirmation.

The planner binds one caller-selected projected influence link to one freshly-resolved, message-backed
source span.  It never measures influence again, calls a worker, generates, writes a run, or mutates
the parent.  Free-generation execution lives in :mod:`clozn.replay.influence_counterfactual`.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import hashlib
import json
from typing import Any

from clozn import schemas
from clozn.replay import span_bridge
from clozn.replay.controlled import recorded_sampling_config
from clozn.experiments.execution_facts import parent_execution_fingerprint, parent_runtime_projection
from clozn.runs import influence_geometry as geometry

PLAN_SCHEMA_VERSION = "clozn.influence-counterfactual-plan.v1"
RESULT_SCHEMA_VERSION = "clozn.influence-counterfactual.v1"
FILLER_RECIPE = "clozn.matched_length_neutral_filler.v1"


class InfluenceCounterfactualInputError(ValueError):
    """Malformed request data; routes expose the stable ``code`` only."""

    __test__ = False

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _reason(code: str, message: str) -> dict:
    return {"code": str(code), "message": str(message)}


def _sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _test_id(run: Mapping, source_span_id: str, answer_span_id: str, intervention: str,
             decode_source: str, specificity_control: bool) -> str:
    fingerprint = parent_execution_fingerprint(run)
    subject = {
        "parent_fingerprint_sha256": fingerprint,
        "source_span_id": source_span_id,
        "answer_span_id": answer_span_id,
        "intervention": intervention,
        "decode_regime": decode_source,
        "specificity_control": bool(specificity_control),
    }
    return "ict_" + _sha256(subject)[:24]


def _request(request: Any) -> tuple[dict, str, bool]:
    if not isinstance(request, Mapping):
        raise InfluenceCounterfactualInputError("invalid_body", "body must be an object")
    if set(request) - {"influence", "intervention", "specificity_control"}:
        raise InfluenceCounterfactualInputError(
            "invalid_body", "body contains an unsupported field")
    influence = request.get("influence")
    if not isinstance(influence, Mapping) or set(influence) != {"source_span_id", "answer_span_id"}:
        raise InfluenceCounterfactualInputError(
            "invalid_influence", "influence needs source_span_id and answer_span_id")
    source_span_id = influence.get("source_span_id")
    answer_span_id = influence.get("answer_span_id")
    if not all(isinstance(value, str) and value for value in (source_span_id, answer_span_id)):
        raise InfluenceCounterfactualInputError(
            "invalid_influence", "source_span_id and answer_span_id must be non-empty strings")
    intervention = request.get("intervention", {"kind": "neutralize"})
    if not isinstance(intervention, Mapping) or set(intervention) != {"kind"}:
        raise InfluenceCounterfactualInputError(
            "invalid_intervention", "intervention needs only kind")
    kind = intervention.get("kind")
    if kind not in {"neutralize", "remove"}:
        raise InfluenceCounterfactualInputError(
            "invalid_intervention", "intervention kind must be neutralize or remove")
    specificity = request.get("specificity_control", True)
    if not isinstance(specificity, bool):
        raise InfluenceCounterfactualInputError(
            "invalid_specificity_control", "specificity_control must be boolean")
    return {
        "source_span_id": source_span_id,
        "answer_span_id": answer_span_id,
    }, kind, specificity


def _measurement(run: Mapping, source_span_id: str, answer_span_id: str) -> dict:
    """Resolve one projected influence link without changing any measured field."""
    response, answer_reason = geometry.resolve_answer_text(dict(run))
    base = {
        "source_span_id": source_span_id,
        "answer_span_id": answer_span_id,
    }
    if response is None:
        return {
            **base,
            "measurement_state": "unavailable",
            "measurement_reason": answer_reason,
        }
    influence_map = run.get("influence_map")
    gate_state, gate_reason = geometry.gate(influence_map)
    if gate_state is not None:
        return {
            **base,
            "measurement_state": gate_state,
            "measurement_reason": gate_reason,
        }
    geo, geo_reason = geometry.resolve_geometry(str(run.get("id") or ""), influence_map, response)
    if geo is None:
        return {
            **base,
            "measurement_state": "unavailable",
            "measurement_reason": geo_reason,
        }

    answer_native_id = next(
        (native_id for native_id, public_id in geo.answer_address_by_id.items()
         if public_id == answer_span_id),
        None,
    )
    if answer_native_id is None:
        return {
            **base,
            "measurement_state": "available",
            "measurement_reason": "answer_span_not_found",
        }
    link = None
    for candidate in geo.links_by_answer_id.get(answer_native_id, ()):
        context_native_id = candidate.get("context_span_id")
        if geo.prompt_address_by_id.get(context_native_id) == source_span_id:
            link = candidate
            break
    if link is None:
        return {
            **base,
            "measurement_state": "available",
            "measurement_reason": "influence_link_not_found",
        }
    answer_interval = geo.answer_offsets.get(answer_native_id)
    measured = {
        "source_span_id": source_span_id,
        "answer_span_id": answer_span_id,
        "effect": deepcopy(link.get("effect")),
        "evidence_state": deepcopy(link.get("evidence_state")),
        "clears_floor": deepcopy(link.get("clears_floor")),
        "delta_nats": deepcopy(link.get("delta_nats")),
        "abs_delta_nats": deepcopy(link.get("abs_delta_nats")),
        "measurement_state": "available",
    }
    if answer_interval is not None:
        measured["answer_interval"] = {
            "start": answer_interval[0],
            "end": answer_interval[1],
            "unit": "unicode_code_points",
            "interval": "half_open",
        }
    return measured


def _span_projection(resolved: Mapping | None) -> dict:
    if not isinstance(resolved, Mapping):
        return {"state": "unavailable"}
    start, end = resolved.get("start"), resolved.get("end")
    message_index = resolved.get("message_index")
    if not all(isinstance(value, int) and not isinstance(value, bool)
               for value in (start, end, message_index)) or end < start:
        return {"state": "unavailable"}
    return {
        "state": "available",
        "basis": "message",
        "message_index": message_index,
        "start": start,
        "end": end,
        "length": end - start,
        "unit": "unicode_code_points",
        "interval": "half_open",
    }


def _decode_projection(run: Mapping) -> tuple[dict, Any, str]:
    config = recorded_sampling_config(dict(run))
    if config is False:
        return (
            {"source": "recorded_greedy", "matches_recorded_decode": True},
            False,
            "recorded_greedy",
        )
    if isinstance(config, Mapping) and isinstance(config.get("seed"), int) and not isinstance(config.get("seed"), bool):
        return (
            {"source": "recorded_fixed_sampling", "matches_recorded_decode": True,
             "sampling": deepcopy(dict(config))},
            deepcopy(dict(config)),
            "recorded_fixed_sampling",
        )
    return (
        {"source": "controlled_greedy_fallback", "matches_recorded_decode": False},
        False,
        "controlled_greedy_fallback",
    )


def _steering_projection(run: Mapping) -> dict:
    """Runs carry no recorded steering state since the personalization cut removed named tone dials
    (and no per-run raw steer_vec record exists to report instead), so this is always "not recorded" --
    kept as its own projection because the plan schema requires an `execution.steering` object."""
    return {"state": "not_recorded", "active": False}


def _specificity_projection(run: Mapping, source_span: Mapping, answer_span_id: str,
                            requested: bool, intervention: str) -> tuple[dict, dict | None]:
    if not requested:
        return {"requested": False, "state": "not_requested"}, None
    control = span_bridge.pick_random_control_span(
        dict(run), dict(source_span), extra=f"{answer_span_id}:{intervention}")
    if not isinstance(control, Mapping):
        return {
            "requested": True,
            "state": "unavailable",
            "reason": "no_disjoint_same_length_control_span",
        }, None
    return {
        "requested": True,
        "state": "available",
        "basis": "message",
        "message_index": control.get("message_index"),
        "start": control.get("start"),
        "end": control.get("end"),
        "length": control.get("end") - control.get("start"),
        "unit": "unicode_code_points",
        "interval": "half_open",
    }, dict(control)


def build_influence_counterfactual_plan(run: Mapping, request: Any = None, *,
                                        source_span_id: str | None = None,
                                        answer_span_id: str | None = None,
                                        intervention: str = "neutralize",
                                        specificity_control: bool = True) -> dict:
    """Build a deterministic plan from one persisted influence link and source address."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run.get("id"):
        raise InfluenceCounterfactualInputError("invalid_parent", "parent run is unavailable")
    if request is not None:
        influence, intervention, specificity_control = _request(request)
        source_span_id = influence["source_span_id"]
        answer_span_id = influence["answer_span_id"]
    if not all(isinstance(value, str) and value for value in (source_span_id, answer_span_id)):
        raise InfluenceCounterfactualInputError(
            "invalid_influence", "source_span_id and answer_span_id must be non-empty strings")
    if intervention not in {"neutralize", "remove"}:
        raise InfluenceCounterfactualInputError(
            "invalid_intervention", "intervention kind must be neutralize or remove")
    if not isinstance(specificity_control, bool):
        raise InfluenceCounterfactualInputError(
            "invalid_specificity_control", "specificity_control must be boolean")

    measured = _measurement(run, source_span_id, answer_span_id)
    resolved = span_bridge.resolve_span_address(dict(run), source_span_id)
    source_span = resolved.get("span") if isinstance(resolved, Mapping) and resolved.get("ok") else None
    span_resolution = _span_projection(source_span)
    if source_span is None:
        span_resolution = {
            "state": "unavailable",
            "reason": ((resolved.get("reason") or {}).get("code")
                       if isinstance(resolved, Mapping) else "span_address_not_found_or_drifted"),
        }

    decode, _sampling_override, decode_source = _decode_projection(run)
    steering = _steering_projection(run)
    specificity, control_span = _specificity_projection(
        run, source_span, answer_span_id, specificity_control, intervention
    ) if source_span is not None else (
        {"requested": bool(specificity_control), "state": "unavailable", "reason": "source_span_unavailable"},
        None,
    )
    test_id = _test_id(run, source_span_id, answer_span_id, intervention, decode_source, specificity_control)

    execution_state = "ready"
    execution_reason = None
    if measured.get("measurement_state") != "available":
        execution_state = "unavailable"
        execution_reason = measured.get("measurement_reason") or "influence_measurement_unavailable"
    elif measured.get("measurement_reason"):
        execution_state = "unavailable"
        execution_reason = measured["measurement_reason"]
    elif source_span is None or span_resolution.get("state") != "available":
        execution_state = "unavailable"
        execution_reason = span_resolution.get("reason") or "span_address_not_found_or_drifted"
    elif not isinstance(run.get("messages"), list):
        execution_state = "unavailable"
        execution_reason = "message_basis_unavailable"

    intervention_doc = {
        "kind": intervention,
        "relation_to_measurement": "same_intervention" if intervention == "neutralize" else "different_intervention",
    }
    if intervention == "neutralize":
        intervention_doc["recipe"] = FILLER_RECIPE

    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "run_id": run["id"],
        "test_id": test_id,
        "parent_fingerprint_sha256": parent_execution_fingerprint(run),
        "influence": measured,
        "intervention": intervention_doc,
        "span_resolution": span_resolution,
        "specificity_control": specificity,
        "execution": {
            "state": execution_state,
            "requires_generation": True,
            "requires_full_reprefill": True,
            "live_runtime_state": "not_checked",
            "decode_regime": decode,
            "steering": steering,
        },
    }
    if "answer_interval" in measured:
        plan["answer_target"] = deepcopy(measured["answer_interval"])
    if measured.get("effect") == "supports":
        plan["expected_measurement_direction"] = "neutralizing_lowered_recorded_probability"
    elif measured.get("effect") == "suppresses":
        plan["expected_measurement_direction"] = "neutralizing_raised_recorded_probability"
    else:
        plan["expected_measurement_direction"] = "no_directional_above_floor_expectation"
    if execution_reason:
        plan["execution"]["reason"] = execution_reason
    schemas.validate(plan, PLAN_SCHEMA_VERSION)
    return plan


# The builder name follows the repository's read-side projection convention; keep the shorter
# conceptual API as a compatibility seam for callers that speak in terms of planning.
plan_influence_counterfactual = build_influence_counterfactual_plan


__all__ = [
    "FILLER_RECIPE",
    "InfluenceCounterfactualInputError",
    "PLAN_SCHEMA_VERSION",
    "RESULT_SCHEMA_VERSION",
    "build_influence_counterfactual_plan",
    "plan_influence_counterfactual",
]
