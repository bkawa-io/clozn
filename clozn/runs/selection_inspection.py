"""Read-side composition for the Select -> Inspect -> Test contract.

This module is intentionally an affordance layer.  It reads the immutable run and already-recorded
derived artifacts, delegates each scientific/detail rule to its owning builder, and emits bounded
metadata-only evidence plus links to explicit operations.  It never plans a live operation and never
touches a worker, model, filesystem, clock, or run store.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

from clozn.experiments.execution_facts import parent_execution_fingerprint, parent_runtime_projection
from clozn.experiments.interventions import sampler_override_contract
from clozn.replay.controlled import recorded_sampling_config
from clozn.runs import close_calls
from clozn.runs import context_tension
from clozn.runs import context_utilization
from clozn.runs import influence_geometry as geometry
from clozn.runs import influence_query
from clozn.runs import suggested_breakpoints
from clozn.replay import branch_fan
from clozn.runs.selection_contract import (
    SelectionContractError,
    is_int as _is_int,
    is_span_id as _is_span_id,
    normalize_selection,
    public_selection,
)

SCHEMA_VERSION = "clozn.selection-inspection.v1"
PRIVACY = "metadata_only"
MAX_ALTERNATIVES = 5
MAX_ALTERNATIVE_TESTS = 3
MAX_INFLUENCE_LINKS = 12
MAX_TENSION_PAIRS = 10
MAX_RELATIONSHIPS = 12


SelectionInspectionInputError = SelectionContractError


def _number(value, *, minimum: float = 0.0, maximum: float | None = None) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    if not math.isfinite(value) or value < minimum or (maximum is not None and value > maximum):
        return None
    return value


def _trace(run: Mapping) -> dict:
    value = run.get("trace")
    return dict(value) if isinstance(value, Mapping) else {}


def _answer(run: Mapping) -> tuple[str | None, str]:
    return geometry.resolve_answer_text(dict(run))


_normalize_selection = normalize_selection


def _href(run_id: str, suffix: str) -> str:
    return f"/runs/{run_id}{suffix}"


def _body(selection: dict, test: dict) -> dict:
    return {"selection": {key: value for key, value in selection.items()
                           if key not in {"selection_id", "basis"}}, "test": test}


def _test_descriptor(run_id: str, operation_id: str, label: str, selection: dict, test: dict,
                     *, state: str = "ready_to_plan", reason: str | None = None,
                     input_contract: dict | None = None) -> dict:
    out = {"id": operation_id, "label": label, "state": state}
    if reason:
        out["reason"] = reason
    if input_contract is not None:
        out["input"] = input_contract
    if state == "ready_to_plan":
        request = _body(selection, test)
        out["plan"] = {"method": "POST", "href": _href(run_id, "/test-this/plan"), "body": request}
        out["execute"] = {"method": "POST", "href": _href(run_id, "/test-this"), "body": request}
    return out


def _navigation(run_id: str, operation_id: str, label: str, suffix: str) -> dict:
    return {
        "id": operation_id,
        "label": label,
        "state": "ready",
        "open": {"method": "GET", "href": _href(run_id, suffix)},
    }


def _measurement(run_id: str, state: str) -> dict:
    if state == "available":
        return {"id": "measure_influence", "label": "Measure what mattered", "state": "already_measured"}
    return {
        "id": "measure_influence",
        "label": "Measure what mattered",
        "state": "conditionally_available",
        "execute": {"method": "POST", "href": _href(run_id, "/influence-map/jobs"), "body": {}},
    }


def _artifact(schema: str, run_id: str, suffix: str | None = None) -> dict:
    out = {"schema": schema}
    if suffix:
        out["href"] = _href(run_id, suffix)
    return out


def _evidence(identifier: str, state: str, data: dict | None = None, *, artifact: dict | None = None,
              reason: str | None = None) -> dict:
    out = {"id": identifier, "state": state}
    if artifact is not None:
        out["artifact"] = artifact
    if reason:
        out["reason"] = reason
    if data is not None:
        out["data"] = data
    return out


def _token_intervals(run: Mapping) -> tuple[dict[int, tuple[int, int]], str | None]:
    response, reason = _answer(run)
    trace = _trace(run)
    tokens = trace.get("tokens")
    if response is None:
        return {}, reason
    if not isinstance(tokens, list) or not all(isinstance(piece, str) for piece in tokens):
        return {}, "trace_tokens_unavailable"
    if "".join(tokens) != response:
        return {}, "trace_response_mismatch"
    intervals = {}
    offset = 0
    for index, piece in enumerate(tokens):
        end = offset + len(piece)
        if end <= offset:
            return {}, "zero_width_token_piece"
        intervals[index] = (offset, end)
        offset = end
    return intervals, None


def _token_value(trace: dict, field: str, position: int):
    value = trace.get(field)
    if isinstance(value, list) and position < len(value):
        return value[position]
    return None


def _token_distribution(run: Mapping, position: int) -> dict:
    trace = _trace(run)
    chosen = _number(_token_value(trace, "confidence", position), minimum=0, maximum=1)
    raw = _token_value(trace, "alternatives", position)
    alternatives = []
    if isinstance(raw, list):
        for rank, item in enumerate(raw):
            if not isinstance(item, Mapping):
                continue
            probability = _number(item.get("prob", item.get("probability", item.get("confidence"))),
                                 minimum=0, maximum=1)
            token_id = item.get("token_id", item.get("id"))
            if token_id is not None and (not _is_int(token_id) or token_id < 0):
                token_id = None
            if probability is None and token_id is None:
                continue
            entry = {"rank": rank}
            if token_id is not None:
                entry["token_id"] = token_id
            if probability is not None:
                entry["probability"] = probability
            alternatives.append(entry)
            if len(alternatives) >= MAX_ALTERNATIVES:
                break
    if chosen is None and not alternatives:
        return _evidence("token_distribution", "unavailable", reason="distribution_unavailable")
    data = {}
    if chosen is not None:
        data["chosen_probability"] = chosen
    data["alternatives"] = alternatives
    return _evidence("token_distribution", "available", data=data,
                     artifact=_artifact("recorded_trace", str(run.get("id")), None))


def _trace_position_complete(run: Mapping, position: int) -> bool:
    trace = _trace(run)
    tokens, confidence, alternatives = trace.get("tokens"), trace.get("confidence"), trace.get("alternatives")
    return (
        isinstance(tokens, list) and isinstance(confidence, list) and isinstance(alternatives, list)
        and position < len(tokens) and position < len(confidence) and position < len(alternatives)
        and isinstance(tokens[position], str)
        and _number(confidence[position], minimum=0, maximum=1) is not None
        and isinstance(alternatives[position], list)
    )


def _close_call_evidence(run: Mapping, position: int) -> dict:
    calls = close_calls.close_calls(dict(run))
    found = next((item for item in calls if item.get("index") == position), None)
    if found is None and not _trace_position_complete(run, position):
        return _evidence("close_call", "unavailable", reason="close_call_evidence_incomplete",
                         artifact=_artifact("recorded_trace", str(run.get("id")), None))
    if found is None:
        return _evidence("close_call", "available", data={"close_call": False},
                         artifact=_artifact("recorded_trace", str(run.get("id")), None))
    return _evidence("close_call", "available", data={
        "close_call": True,
        "margin": found.get("margin"),
        "chosen_probability": found.get("top_prob") if found.get("emitted") == found.get("top") else found.get("alt_prob"),
        "rival_probability": found.get("alt_prob") if found.get("emitted") == found.get("top") else found.get("top_prob"),
        "meaningful_heuristic": found.get("meaningful") is True,
    }, artifact=_artifact("recorded_trace", str(run.get("id")), None))


def _breakpoint_evidence(run: Mapping, position: int) -> dict:
    try:
        document = suggested_breakpoints.build_suggested_breakpoints(dict(run), limit=suggested_breakpoints.MAX_LIMIT)
    except Exception:
        return _evidence("suggested_breakpoint", "unavailable", reason="suggested_breakpoint_unavailable",
                         artifact=_artifact("clozn.suggested-breakpoints.v1", str(run.get("id")), "/suggested-breakpoints"))
    found = next((item for item in document.get("breakpoints", []) if item.get("position") == position), None)
    state = document.get("analysis", {}).get("state")
    if found is not None:
        return _evidence("suggested_breakpoint", "available", data={"suggested": True, **found},
                         artifact=_artifact("clozn.suggested-breakpoints.v1", str(run.get("id")), "/suggested-breakpoints"))
    if state == "unavailable":
        return _evidence("suggested_breakpoint", "unavailable", reason="breakpoint_analysis_unavailable",
                         artifact=_artifact("clozn.suggested-breakpoints.v1", str(run.get("id")), "/suggested-breakpoints"))
    return _evidence("suggested_breakpoint", "available", data={"suggested": False},
                     artifact=_artifact("clozn.suggested-breakpoints.v1", str(run.get("id")), "/suggested-breakpoints"))


def _rewind_evidence(run: Mapping, position: int) -> dict:
    from clozn.replay.rewind_fidelity import build_rewind_fidelity
    try:
        document = build_rewind_fidelity(dict(run))
    except Exception:
        return _evidence("rewind", "unavailable", reason="rewind_fidelity_unavailable",
                         artifact=_artifact("clozn.rewind-fidelity.v1", str(run.get("id")), "/rewind-fidelity"))
    exact = ((document.get("recorded_capability") or {}).get("exact_rewind") or {})
    state = exact.get("state")
    if not isinstance(state, str):
        return _evidence("rewind", "unavailable", reason="rewind_fidelity_unavailable",
                         artifact=_artifact("clozn.rewind-fidelity.v1", str(run.get("id")), "/rewind-fidelity"))
    return _evidence("rewind", "available", data={"position": position, "static_state": state},
                     artifact=_artifact("clozn.rewind-fidelity.v1", str(run.get("id")), "/rewind-fidelity"))


def _response_token(run: Mapping, selection: dict) -> dict:
    run_id, position, trace = str(run["id"]), selection["position"], _trace(run)
    token_id = _token_value(trace, "token_ids", position)
    if not _is_int(token_id) or token_id < 0:
        token_id = None
    primary = {"kind": "response_token", "position": position}
    if token_id is not None:
        primary["token_id"] = token_id
    intervals, geometry_reason = _token_intervals(run)
    if position in intervals:
        start, end = intervals[position]
        primary["response_interval"] = {
            "start": start, "end": end, "unit": "unicode_code_points", "interval": "half_open",
        }
    evidence = [_token_distribution(run, position), _close_call_evidence(run, position),
                _breakpoint_evidence(run, position), _rewind_evidence(run, position)]
    if geometry_reason:
        evidence.append(_evidence("response_geometry", "unavailable", reason=geometry_reason))

    try:
        alternatives = branch_fan.recorded_alternative_candidates(run, position)
    except branch_fan.BranchFanInputError:
        # A token trace can be sufficient for inspection while carrying no trustworthy alternative
        # array.  That is an unavailable test affordance, not a composition failure.
        alternatives = []
    tests = []
    for candidate in alternatives[:MAX_ALTERNATIVE_TESTS]:
        rank = candidate["recorded_rank"]
        tests.append(_test_descriptor(
            run_id, f"try_alternative:{rank}", f"Try recorded alternative #{rank + 1}", selection,
            {"kind": "try_alternative", "alternative_rank": rank},
        ))
    if alternatives:
        tests.append(_test_descriptor(
            run_id, "fan_alternatives", "Explore roads not taken", selection,
            {"kind": "fan_alternatives", "limit": 3},
        ))
    sampling = recorded_sampling_config(dict(run))
    if isinstance(sampling, Mapping):
        tests.append(_test_descriptor(
            run_id, "probe_sampler_sensitivity", "Probe sampler sensitivity",
            {**selection, "kind": "sampling"},
            {"kind": "probe_sensitivity", "recipe": "nearby_v1"},
        ))
    else:
        tests.append(_test_descriptor(
            run_id, "probe_sampler_sensitivity", "Probe sampler sensitivity",
            {**selection, "kind": "sampling"}, {"kind": "probe_sensitivity", "recipe": "nearby_v1"},
            state="unavailable",
            reason="greedy_baseline_no_sampling_neighborhood" if sampling is False else "sampler_provenance_unavailable",
        ))
    navigation = [
        _navigation(run_id, "open_run", "Open full run", ""),
        _navigation(run_id, "open_rewind_fidelity", "Open Rewind Fidelity", "/rewind-fidelity"),
        _navigation(run_id, "open_suggested_breakpoints", "Open Suggested Breakpoints", "/suggested-breakpoints"),
    ]
    return {"primary": primary, "evidence": evidence, "measurements": [], "tests": tests, "navigation": navigation}


def _influence_projection(run: Mapping, query: dict, *, cap: int) -> dict:
    links = []
    for raw in query.get("links", [])[:cap]:
        if not isinstance(raw, Mapping):
            continue
        link = dict(raw)
        source_id, answer_id = link.get("source_span_id"), link.get("answer_span_id")
        if isinstance(source_id, str) and isinstance(answer_id, str):
            link["select"] = {"kind": "context_span", "source_span_id": source_id, "answer_span_id": answer_id}
        links.append(link)
    data = {
        "measurement": dict(query.get("measurement") or {}),
        "links": links,
        "summary": dict(query.get("summary") or {}),
    }
    return _evidence("measured_influence", query.get("measurement", {}).get("state", "unavailable"), data=data,
                     artifact=_artifact("clozn.influence-query.v1", str(run.get("id")), "/influence-query"))


def _tension_projection(run: Mapping, tension: dict, *, cap: int = MAX_TENSION_PAIRS) -> dict:
    state = tension.get("state")
    data = {
        "answer_spans": list(tension.get("answer_spans") or []),
        "tensions": list(tension.get("tensions") or [])[:cap],
        "summary": dict(tension.get("summary") or {}),
    }
    return _evidence("context_tension", state if state in {"available", "not_measured", "unavailable", "error"} else "error",
                     data=data if state == "available" else None,
                     reason=tension.get("reason") if state != "available" else None,
                     artifact=_artifact("clozn.context-tension.v1", str(run.get("id")), "/context-tension"))


def _answer_span(run: Mapping, selection: dict) -> dict:
    run_id, start, end = str(run["id"]), selection["start"], selection["end"]
    response_sha256 = selection["basis"]["sha256"]
    primary = {
        "kind": "answer_span",
        "interval": {"start": start, "end": end, "unit": "unicode_code_points", "interval": "half_open"},
        "basis_sha256": response_sha256,
    }
    query = influence_query.build_influence_query(run, output_start=start, output_end=end, limit=MAX_INFLUENCE_LINKS)
    tension = context_tension.build_context_tension(run, output_start=start, output_end=end, limit=MAX_TENSION_PAIRS)
    evidence = [_influence_projection(run, query, cap=MAX_INFLUENCE_LINKS), _tension_projection(run, tension)]
    measurements = [_measurement(run_id, (query.get("measurement") or {}).get("state"))]
    navigation = [
        _navigation(run_id, "open_run", "Open full run", ""),
        _navigation(run_id, "open_why_this", "Open Why This", "/influence-query?start=%d&end=%d" % (start, end)),
        _navigation(run_id, "open_context_tension", "Open Context Tension", "/context-tension"),
    ]
    return {"primary": primary, "evidence": evidence, "measurements": measurements, "tests": [], "navigation": navigation}


def _fresh_address(run: Mapping, source_id: str) -> tuple[dict | None, str | None]:
    from clozn.replay.span_bridge import resolve_span_address
    result = resolve_span_address(dict(run), source_id)
    if not result.get("ok"):
        reason = (result.get("reason") or {}).get("code") if isinstance(result.get("reason"), Mapping) else None
        return None, reason or "span_address_not_found_or_drifted"
    return result.get("span"), None


def _address_metadata(run: Mapping, source_id: str) -> dict:
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses
    try:
        document = build_persisted_text_span_addresses(dict(run), privacy=PRIVACY)
    except Exception:
        return {}
    for address in document.get("addresses", []):
        if isinstance(address, Mapping) and address.get("address_id") == source_id:
            resolution = address.get("resolution") if isinstance(address.get("resolution"), Mapping) else {}
            canonical = resolution.get("canonical") if isinstance(resolution.get("canonical"), Mapping) else {}
            out = {"basis": address.get("kind"), "state": resolution.get("state")}
            if isinstance(canonical.get("span_code_points"), int):
                out["code_points"] = canonical["span_code_points"]
            if isinstance(canonical.get("span_sha256"), str):
                out["sha256"] = canonical["span_sha256"]
            return out
    return {}


def _all_source_links(run: Mapping, source_id: str) -> tuple[dict, list[dict]]:
    response, reason = _answer(run)
    if response is None:
        return {"state": "unavailable", "reason": reason}, []
    if not response:
        return {"state": "unavailable", "reason": "empty_recorded_answer"}, []
    query = influence_query.build_influence_query(run, output_start=0, output_end=len(response), limit=50)
    links = [dict(link) for link in query.get("links", []) if link.get("source_span_id") == source_id]
    return dict(query.get("measurement") or {}), links


def _relationship_link(run: Mapping, source_id: str, answer_id: str) -> tuple[dict, dict | None]:
    measurement, links = _all_source_links(run, source_id)
    return measurement, next((link for link in links if link.get("answer_span_id") == answer_id), None)


def _relationship_tests(run: Mapping, run_id: str, selection: dict) -> list[dict]:
    tests = [
        _test_descriptor(run_id, "neutralize_context", "Test without this signal", selection, {"kind": "neutralize"}),
        _test_descriptor(run_id, "remove_context", "Test without this context", selection, {"kind": "remove"}),
    ]
    static_reason = None
    if not isinstance(run.get("messages"), list):
        static_reason = "message_basis_unavailable"
    elif recorded_sampling_config(dict(run)) is None:
        static_reason = "sampler_provenance_unavailable"
    elif parent_runtime_projection(dict(run)) is None:
        static_reason = "parent_runtime_identity_unavailable"
    tests.append(_test_descriptor(
        run_id, "bisect_context", "Narrow this context", selection,
        {"kind": "bisect", "max_depth": 3},
        state="ready_to_plan" if static_reason is None else "unavailable",
        reason=static_reason,
    ))
    return tests


def _context_span(run: Mapping, selection: dict) -> dict:
    run_id, source_id = str(run["id"]), selection["source_span_id"]
    resolved, resolution_reason = _fresh_address(run, source_id)
    metadata = _address_metadata(run, source_id)
    primary = {"kind": "context_span", "source_span_id": source_id}
    if "answer_span_id" in selection:
        primary["scope"] = "measured_relationship"
        primary["answer_span_id"] = selection["answer_span_id"]
    if resolved is not None:
        primary["resolution"] = {key: value for key, value in metadata.items() if key in {"basis", "code_points", "sha256"}}
        primary["resolution"]["state"] = "exact"
    else:
        primary["resolution"] = {"state": "unavailable", "reason": resolution_reason}

    evidence = []
    try:
        utilization = context_utilization.build_context_utilization(run)
        source = next((item for item in utilization.get("sources", [])
                       if item.get("source_span_id") == source_id), None)
        if source is None:
            evidence.append(_evidence("context_utilization", utilization.get("measurement", {}).get("state", "unavailable"),
                                      data={"source_span_id": source_id},
                                      artifact=_artifact("clozn.context-utilization.v1", run_id, "/context-utilization")))
        else:
            evidence.append(_evidence("context_utilization", utilization.get("measurement", {}).get("state", "available"),
                                      data={"source": dict(source)},
                                      artifact=_artifact("clozn.context-utilization.v1", run_id, "/context-utilization")))
    except Exception:
        evidence.append(_evidence("context_utilization", "error", reason="context_utilization_contract_invalid",
                                  artifact=_artifact("clozn.context-utilization.v1", run_id, "/context-utilization")))

    measurement, links = _all_source_links(run, source_id)
    relationship_data = []
    for link in links[:MAX_RELATIONSHIPS]:
        item = dict(link)
        answer_id = item.get("answer_span_id")
        if isinstance(answer_id, str):
            item["select"] = {"kind": "context_span", "source_span_id": source_id, "answer_span_id": answer_id}
        relationship_data.append(item)
    evidence.append(_evidence("answer_relationships", measurement.get("state", "unavailable"),
                              data={"links": relationship_data},
                              artifact=_artifact("clozn.influence-query.v1", run_id, "/influence-query"),
                              reason=measurement.get("reason") if measurement.get("state") != "available" else None))

    try:
        tension = context_tension.build_context_tension(run)
        relevant = [pair for pair in tension.get("tensions", [])
                    if source_id in {
                        ((pair.get("supporting") or {}).get("source_span_id")),
                        ((pair.get("suppressing") or {}).get("source_span_id")),
                    }]
        projected = dict(tension)
        projected["tensions"] = relevant[:MAX_TENSION_PAIRS]
        evidence.append(_tension_projection(run, projected, cap=MAX_TENSION_PAIRS))
    except Exception:
        evidence.append(_evidence("context_tension", "error", reason="context_tension_contract_invalid",
                                  artifact=_artifact("clozn.context-tension.v1", run_id, "/context-tension")))

    selected_link = None
    if "answer_span_id" in selection:
        _selected_measurement, selected_link = _relationship_link(run, source_id, selection["answer_span_id"])
        if selected_link is not None:
            primary["measured_influence"] = {
                key: selected_link.get(key)
                for key in ("effect", "evidence_state", "clears_floor", "delta_nats", "abs_delta_nats")
                if key in selected_link
            }
    tests = _relationship_tests(run, run_id, selection) if resolved is not None and selected_link is not None else []
    navigation = [
        _navigation(run_id, "open_run", "Open full run", ""),
        _navigation(run_id, "open_context_utilization", "Open Context Utilization", "/context-utilization"),
        _navigation(run_id, "open_why_this", "Open Why This", "/influence-query"),
    ]
    if selected_link is None and "answer_span_id" in selection:
        evidence.append(_evidence("selected_relationship", "unavailable", reason="influence_link_not_found"))
    measurements = [_measurement(run_id, measurement.get("state"))]
    return {"primary": primary, "evidence": evidence, "measurements": measurements,
            "tests": tests, "navigation": navigation}


def _sampling(run: Mapping, selection: dict) -> dict:
    run_id, config = str(run["id"]), recorded_sampling_config(dict(run))
    if config is False:
        recorded = {"mode": "greedy"}
    elif isinstance(config, Mapping):
        recorded = {"mode": "sample", "temperature": config.get("temperature"), "top_p": config.get("top_p"),
                    "top_k": config.get("top_k"), "rep_penalty": config.get("repeat_penalty"),
                    "seed": config.get("seed")}
    else:
        recorded = {"state": "unavailable"}
    primary = {"kind": "sampling", "position": selection["position"], "recorded": recorded}
    evidence_state = "available" if config is not None else "unavailable"
    evidence = [_evidence("recorded_sampling", evidence_state, data=recorded,
                          artifact=_artifact("recorded_run", run_id, None),
                          reason="sampler_provenance_unavailable" if config is None else None)]
    tests = [_test_descriptor(run_id, "change_sampling", "Change sampling", selection,
                              {"kind": "change_sampling", "changes": {}}, state="requires_input",
                              input_contract=sampler_override_contract())]
    if isinstance(config, Mapping):
        tests.append(_test_descriptor(run_id, "probe_sampler_sensitivity", "Probe sampler sensitivity", selection,
                                      {"kind": "probe_sensitivity", "recipe": "nearby_v1"}))
    else:
        tests.append(_test_descriptor(run_id, "probe_sampler_sensitivity", "Probe sampler sensitivity", selection,
                                      {"kind": "probe_sensitivity", "recipe": "nearby_v1"}, state="unavailable",
                                      reason="greedy_baseline_no_sampling_neighborhood" if config is False else "sampler_provenance_unavailable"))
    navigation = [_navigation(run_id, "open_run", "Open full run", ""),
                  _navigation(run_id, "open_rewind_fidelity", "Open Rewind Fidelity", "/rewind-fidelity")]
    return {"primary": primary, "evidence": evidence, "measurements": [], "tests": tests, "navigation": navigation}


def _inspection_state(evidence: list[dict], primary: Mapping) -> str:
    states = [entry.get("state") for entry in evidence]
    if primary.get("resolution", {}).get("state") == "unavailable":
        return "partially_available" if any(state == "available" for state in states) else "unavailable"
    if any(state in {"error", "unavailable"} for state in states) and any(state == "available" for state in states):
        return "partially_available"
    if any(state == "available" for state in states):
        return "available"
    return "partially_available" if states else "available"


def _summary(evidence: list[dict], measurements: list[dict], tests: list[dict], navigation: list[dict]) -> dict:
    return {
        "evidence_available": sum(1 for item in evidence if item.get("state") == "available"),
        "evidence_not_measured": sum(1 for item in evidence if item.get("state") == "not_measured"),
        "evidence_unavailable": sum(1 for item in evidence if item.get("state") in {"unavailable", "error"}),
        "tests_ready_to_plan": sum(1 for item in tests if item.get("state") == "ready_to_plan"),
        "tests_requiring_input": sum(1 for item in tests if item.get("state") == "requires_input"),
        "measurements_available": sum(1 for item in measurements if item.get("state") in {"already_measured", "conditionally_available", "available_from_recorded_evidence"}),
        "navigation_ready": sum(1 for item in navigation if item.get("state") == "ready"),
    }


def build_selection_inspection(run: dict, *, selection: dict) -> dict:
    """Build a deterministic metadata-only inspection document for one selected run object."""
    if not isinstance(run, Mapping):
        raise SelectionInspectionInputError("invalid_run", "recorded run must be an object")
    normalized = _normalize_selection(run, selection)
    kind = normalized["kind"]
    if kind == "response_token":
        parts = _response_token(run, normalized)
    elif kind == "answer_span":
        parts = _answer_span(run, normalized)
    elif kind == "context_span":
        parts = _context_span(run, normalized)
    else:
        parts = _sampling(run, normalized)
    from clozn.runs.selection_reference import encode_selection_reference
    reference = encode_selection_reference(run, public_selection(normalized))
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run["id"],
        "privacy": PRIVACY,
        "selection": normalized,
        "reference": {
            "state": reference.get("state"),
            "selection_ref": reference.get("reference"),
            "api_href": reference.get("api_href"),
            "deep_link": reference.get("deep_link"),
            **({"reason": reference.get("reason")} if reference.get("reason") else {}),
        },
        "inspection": {
            "state": _inspection_state(parts["evidence"], parts["primary"]),
            "primary": parts["primary"],
            "evidence": parts["evidence"],
        },
        "measurements": parts["measurements"],
        "tests": parts["tests"],
        "navigation": parts["navigation"],
    }
    document["summary"] = _summary(document["inspection"]["evidence"], document["measurements"],
                                    document["tests"], document["navigation"])
    from clozn import schemas
    schemas.validate(document, SCHEMA_VERSION)
    return document


__all__ = ["PRIVACY", "SCHEMA_VERSION", "SelectionInspectionInputError", "build_selection_inspection"]
