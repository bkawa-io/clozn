"""Canonical read-only diagnostics and V1 debugger capabilities for one recorded Run.

This module is deliberately a projection, not another executor or evidence store.  It composes the
existing recorded-evidence authorities (runtime projection, token prerequisites, receipt reader,
diagnosis/performance reports, and model-free recipe planners) into one versioned document.  Missing
evidence is represented explicitly; no current worker is consulted and no artifact is persisted.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from typing import Any

from clozn import schemas
from clozn.replay.execution_fork import parent_runtime_projection, recorded_fork_prerequisites
from clozn.replay.rewind_fidelity import build_rewind_fidelity
from clozn.runs.answer_preservation import generation_contract_from_run
from clozn.runs.context_receipt import read_receipt


SCHEMA_VERSION = "clozn.run-diagnostics.v1"


def _state(state: str, *, reason_code: str | None = None, reason: str | None = None,
           value: Any = None, include_value: bool = False, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"state": state}
    if reason_code:
        result["reason_code"] = reason_code
    if reason:
        result["reason"] = reason
    if include_value:
        result["value"] = deepcopy(value)
    result.update(deepcopy(extra))
    return result


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) and value else None


def _integer(value: Any, *, minimum: int = 0) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= minimum else None


def _number(value: Any, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if result == result and result not in (float("inf"), float("-inf")) and result >= minimum else None


def _mode_hint(run: Mapping[str, Any]) -> str | None:
    """Read only the explicit decode-mode marker for diagnostics of an incomplete contract."""
    explicit = run.get("generation_contract")
    if not isinstance(explicit, Mapping) or not explicit:
        explicit = run.get("output_contract")
    if not isinstance(explicit, Mapping) or not explicit:
        meta = _mapping(run.get("meta"))
        explicit = meta.get("generation_contract")
        if not isinstance(explicit, Mapping) or not explicit:
            explicit = meta.get("decode")
        if not isinstance(explicit, Mapping) or not explicit:
            mode = meta.get("sampler_mode")
            return mode if mode in {"greedy", "sample"} else None
    mode = explicit.get("decode_mode", explicit.get("mode"))
    if mode is None and isinstance(explicit.get("decode"), Mapping):
        mode = explicit["decode"].get("mode")
    return mode if mode in {"greedy", "sample"} else None


def _runtime_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    identity = _mapping(run.get("identity"))
    meta = _mapping(run.get("meta"))
    recorded_fields: dict[str, Any] = {}
    for key in (
        "model_sha256", "gguf_artifact_sha256", "template_fingerprint", "engine_build",
        "clozn_version", "tokenizer_sha256", "tokenizer_fingerprint", "adapter",
        "white_box_flags", "captured_at",
    ):
        if key in identity:
            recorded_fields[key] = deepcopy(identity[key])
    for key in ("n_ctx", "device", "model_id", "sampler_mode"):
        if key in meta and key not in recorded_fields:
            recorded_fields[key] = deepcopy(meta[key])

    normalized = None
    try:
        normalized = parent_runtime_projection(run)
    except Exception:
        normalized = None
    routing_present = "model_routing" in meta
    if normalized is not None:
        state = "available"
    elif routing_present:
        # parent_runtime_projection intentionally collapses malformed routing and overlapping
        # identity disagreement to None. Preserve that fail-closed meaning here.
        state = "contradictory"
    elif recorded_fields:
        state = "incomplete"
    else:
        state = "unavailable"

    routing = _state("unavailable", reason_code="model_routing_not_recorded")
    if routing_present:
        routing = (
            _state("available", value=meta["model_routing"], include_value=True)
            if normalized is not None else
            _state("contradictory", reason_code="runtime_identity_contradictory",
                   reason="recorded model-routing/runtime identity evidence disagrees or is malformed")
        )
    result: dict[str, Any] = {
        "state": state,
        "model_id": _state("available", value=run.get("model"), include_value=True)
        if _string(run.get("model")) else _state("unavailable", reason_code="model_id_unrecorded"),
        "substrate": _state("available", value=run.get("substrate"), include_value=True)
        if _string(run.get("substrate")) else _state("unavailable", reason_code="substrate_unrecorded"),
        "recorded_identity": _state(
            "available" if recorded_fields else "unavailable",
            value=recorded_fields, include_value=bool(recorded_fields),
            reason_code=None if recorded_fields else "runtime_identity_unrecorded",
        ),
        "normalized": deepcopy(normalized) if normalized is not None else _state(
            "unavailable", reason_code="runtime_identity_unavailable",
        ),
        "routing": routing,
    }
    if normalized is not None:
        result["normalized"] = _state("available", value=normalized, include_value=True)
    return result


def _input_projection(run: Mapping[str, Any], receipt_view: Mapping[str, Any]) -> dict[str, Any]:
    receipt = _mapping(receipt_view.get("receipt"))
    shape = receipt_view.get("shape")
    messages = run.get("messages")
    assembled = run.get("assembled_messages")
    final_prompt = run.get("final_prompt")
    memory = _mapping(run.get("memory"))
    limits = _mapping(receipt.get("limits"))

    receipt_state = (
        _state("available", shape=shape)
        if shape == "new" else
        _state("partial", shape=shape, reason_code="legacy_context_receipt")
        if shape == "legacy" else
        _state("malformed", shape=shape, reason_code="context_receipt_unrecognized")
        if shape == "unrecognized" else
        _state("unavailable", shape="absent", reason_code="context_receipt_absent")
    )
    input_truncated = receipt.get("input_truncated") if isinstance(receipt.get("input_truncated"), bool) else None
    output_cut_off = receipt.get("output_cut_off") if isinstance(receipt.get("output_cut_off"), bool) else None
    truncation = {
        "state": "available" if input_truncated is not None or output_cut_off is not None else "unavailable",
        "input_truncated": input_truncated,
        "output_cut_off": output_cut_off,
        "warnings": deepcopy(receipt.get("warnings")) if isinstance(receipt.get("warnings"), list) else [],
    }
    if truncation["state"] == "unavailable":
        truncation["reason_code"] = "truncation_evidence_unrecorded"

    prompt_tokens = _integer(limits.get("prompt_tokens"))
    context_window = _integer(limits.get("context_window_tokens"), minimum=1)
    return {
        "state": "available" if receipt_state["state"] in {"available", "partial"} else "partial",
        "messages": _state("available", value=messages, include_value=True)
        if isinstance(messages, list) else _state("unavailable", reason_code="messages_unrecorded"),
        "assembled_messages": _state("available", value=assembled, include_value=True)
        if isinstance(assembled, list) else _state("unavailable", reason_code="assembled_messages_unrecorded"),
        "final_prompt": _state("available", value=final_prompt, include_value=True)
        if isinstance(final_prompt, str) else _state("unavailable", reason_code="final_prompt_unrecorded"),
        "memory_prompt_block": _state("available", value=memory.get("prompt_block"), include_value=True)
        if isinstance(memory.get("prompt_block"), str) else _state(
            "unavailable", reason_code="memory_prompt_block_unrecorded"),
        "context_receipt": receipt_state,
        "prompt_tokens": _state("available", value=prompt_tokens, include_value=True)
        if prompt_tokens is not None else _state("unavailable", reason_code="prompt_token_count_unrecorded"),
        "context_window_tokens": _state("available", value=context_window, include_value=True)
        if context_window is not None else _state("unavailable", reason_code="context_window_unrecorded"),
        "truncation": truncation,
    }


def _generation_projection(run: Mapping[str, Any]) -> dict[str, Any]:
    contract, reason = generation_contract_from_run(run)
    if isinstance(contract, Mapping) and not reason:
        return {
            "state": "available",
            "contract": deepcopy(dict(contract)),
            "decode_mode": contract.get("decode_mode"),
            "expected_termination": deepcopy(contract.get("expected_termination")),
            "recorded_termination": deepcopy(_mapping(run.get("termination"))) or None,
        }
    mode = _mode_hint(run)
    reason_code = reason or "generation_contract_unrecorded"
    return {
        "state": "incomplete" if mode is not None or reason else "unavailable",
        "contract": None,
        "decode_mode": mode,
        "reason_code": reason_code,
        "reason": "the recorded generation contract is incomplete; sampler fields are not inferred"
        if mode == "sample" else "the recorded generation contract is unavailable",
    }


def _sequence_evidence(trace: Mapping[str, Any], key: str, count: int | None,
                       *, item_kind: str) -> dict[str, Any]:
    if key not in trace:
        return _state("unavailable", reason_code=f"{item_kind}_unrecorded")
    value = trace.get(key)
    if not isinstance(value, list):
        return _state("malformed", reason_code=f"{item_kind}_malformed")
    if count is not None and len(value) != count:
        return _state("malformed", reason_code=f"{item_kind}_length_mismatch", value=value,
                      include_value=True, expected_count=count, observed_count=len(value))
    return _state("available", value=value, include_value=True, count=len(value))


def _output_projection(run: Mapping[str, Any], receipt_view: Mapping[str, Any]) -> dict[str, Any]:
    trace = _mapping(run.get("trace"))
    prerequisites = recorded_fork_prerequisites(run)
    pieces = trace.get("tokens")
    ids = trace.get("token_ids")
    pieces_valid = prerequisites["token_pieces_available"]
    ids_valid = prerequisites["token_ids_available"]
    response = run.get("response")
    reconstructs = None
    if isinstance(response, str) and pieces_valid:
        reconstructs = "matched" if "".join(pieces) == response else "mismatch"
    ids_align = None
    if pieces_valid and ids_valid:
        ids_align = "matched" if len(pieces) == len(ids) else "mismatch"
    count = prerequisites["recorded_token_count"]
    steps = trace.get("steps")
    timed_count = 0
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, Mapping) and (
                _number(step.get("dt_ms")) is not None or _number(step.get("wall_ms")) is not None
            ):
                timed_count += 1
    if not isinstance(steps, list):
        timing = _state("unavailable", reason_code="token_step_timing_unrecorded")
    elif timed_count == 0:
        timing = _state("unavailable", reason_code="token_step_timing_unrecorded", step_count=len(steps))
    elif timed_count == len(steps):
        timing = _state("available", step_count=len(steps), timed_step_count=timed_count)
    else:
        timing = _state("partial", reason_code="token_step_timing_incomplete",
                        step_count=len(steps), timed_step_count=timed_count)

    if prerequisites["token_alignment_available"] and isinstance(response, str) and reconstructs == "matched":
        trace_state = "available"
    elif pieces_valid or ids_valid or isinstance(steps, list):
        trace_state = "partial"
    else:
        trace_state = "unavailable"

    finish = run.get("finish_reason")
    if not isinstance(finish, str) or not finish:
        finish = _mapping(run.get("meta")).get("finish_reason")
    termination = _mapping(receipt_view.get("receipt")).get("termination")
    return {
        "state": trace_state,
        "response": _state("available", value=response, include_value=True)
        if isinstance(response, str) else _state("unavailable", reason_code="response_unrecorded"),
        "token_ids": _state("available", value=ids, include_value=True, count=len(ids))
        if ids_valid else _state("unavailable", reason_code="recorded_token_ids_unavailable"),
        "token_pieces": _state("available", value=pieces, include_value=True, count=len(pieces))
        if pieces_valid else _state("unavailable", reason_code="recorded_token_pieces_unavailable"),
        "token_count": _state("available", value=count, include_value=True)
        if count is not None else _state("unavailable", reason_code="recorded_token_count_unavailable"),
        "pieces_reconstruct_response": _state("available", value=reconstructs, include_value=True)
        if reconstructs is not None else _state("unavailable", reason_code="response_or_token_pieces_unavailable"),
        "ids_and_pieces_alignment": _state("available", value=ids_align, include_value=True)
        if ids_align is not None else _state("unavailable", reason_code="token_ids_or_pieces_unavailable"),
        "token_logprobs": _sequence_evidence(trace, "logprobs", count, item_kind="token_logprobs"),
        "token_alternatives": _sequence_evidence(trace, "alternatives", count, item_kind="token_alternatives"),
        "steps": _state("available", value=steps, include_value=True, count=len(steps))
        if isinstance(steps, list) else _state("unavailable", reason_code="token_steps_unrecorded"),
        "trace_completeness": {
            "state": trace_state,
            "recorded_fork_prerequisites": deepcopy(prerequisites),
            "reason_codes": [
                key for key, present in (
                    ("recorded_token_pieces_unavailable", prerequisites["token_pieces_available"]),
                    ("recorded_token_ids_unavailable", prerequisites["token_ids_available"]),
                    ("token_alignment_unavailable", prerequisites["token_alignment_available"]),
                    ("final_prompt_unavailable", prerequisites["final_prompt_available"]),
                ) if not present
            ],
        },
        "finish_reason": _state("available", value=finish, include_value=True)
        if isinstance(finish, str) and finish else _state("unavailable", reason_code="finish_reason_unrecorded"),
        "termination": _state("available", value=termination, include_value=True)
        if isinstance(termination, Mapping) and termination else _state(
            "unavailable", reason_code="termination_evidence_unrecorded"),
        "timing_evidence": timing,
    }


def _health_projection(run: Mapping[str, Any], related_runs: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    related = list(related_runs or [])
    try:
        from clozn.runs.diagnosis import diagnose
        diagnosis = diagnose(run, related_runs=related)
    except Exception as exc:
        diagnosis = {"state": "unavailable", "reason_code": "diagnosis_unavailable",
                     "reason": type(exc).__name__}
    try:
        from clozn.runs.perf_diagnosis import build_performance_report
        performance = build_performance_report(run, related_runs=related)
    except Exception as exc:
        performance = {"state": "unavailable", "reason_code": "performance_diagnosis_unavailable",
                       "reason": type(exc).__name__}
    slow_findings = _mapping(diagnosis.get("why_slow")).get("findings", []) if isinstance(diagnosis, Mapping) else []
    cutoff_finding = _mapping(diagnosis.get("why_cut_off")).get("finding") if isinstance(diagnosis, Mapping) else None
    findings = list(slow_findings) if isinstance(slow_findings, list) else []
    if isinstance(cutoff_finding, Mapping):
        findings.append(cutoff_finding)
    perf_findings = performance.get("diagnoses", []) if isinstance(performance, Mapping) else []
    issues: list[dict[str, Any]] = []
    if isinstance(run.get("error"), str) and run["error"]:
        issues.append({"code": "recorded_error", "severity": "error"})
    if any(isinstance(item, Mapping) and item.get("id") == "output_cutoff"
           and item.get("status") == "observed" for item in findings):
        issues.append({"code": "output_cutoff_recorded", "severity": "warning"})
    if any(isinstance(item, Mapping) and item.get("status") == "unavailable" for item in findings + perf_findings):
        issues.append({"code": "health_measurement_unavailable", "severity": "unknown"})
    receipt_shape = read_receipt(dict(run)).get("shape")
    if receipt_shape in {"absent", "unrecognized"}:
        issues.append({"code": "context_receipt_unavailable", "severity": "unknown"})
    contract, contract_reason = generation_contract_from_run(run)
    if contract_reason or not isinstance(contract, Mapping):
        issues.append({"code": "generation_contract_incomplete", "severity": "unknown"})
    prerequisites = recorded_fork_prerequisites(run)
    if not prerequisites["token_alignment_available"]:
        issues.append({"code": "token_trace_incomplete", "severity": "unknown"})
    state = "available" if not issues else "degraded"
    return {
        "state": state,
        "recorded_error": _state("available", value=run.get("error"), include_value=True)
        if isinstance(run.get("error"), str) and run["error"] else _state(
            "unavailable", reason_code="error_unrecorded"),
        "diagnosis": deepcopy(diagnosis),
        "performance": deepcopy(performance),
        "issues": issues,
    }


def _source_candidates(run: Mapping[str, Any]) -> list[str]:
    manifest = run.get("context_units")
    if isinstance(manifest, Mapping) and isinstance(manifest.get("default_source_ids"), list):
        values = manifest["default_source_ids"]
    else:
        receipt = _mapping(run.get("context_receipt"))
        values = []
        for segment in receipt.get("delivered", []) if isinstance(receipt.get("delivered"), list) else []:
            if not isinstance(segment, Mapping):
                continue
            sources = segment.get("sources")
            values.extend(
                item.get("source_id") for item in sources
                if isinstance(sources, list) and isinstance(item, Mapping)
            )
            if not sources:
                values.append(segment.get("segment_id"))
    return list(dict.fromkeys(value for value in values if isinstance(value, str) and value))


def _checkpoint_pin_projection(value: Mapping[str, Any] | None) -> dict[str, Any]:
    """Project pin metadata without ever returning the hydrated checkpoint envelope/KV bytes."""
    if not isinstance(value, Mapping):
        return _state("unavailable", reason_code="checkpoint_pin_not_loaded")
    if value.get("ok") is not True:
        return _state("stale", reason_code="checkpoint_pin_unavailable")
    manifest = value.get("manifest")
    if not isinstance(manifest, Mapping):
        return _state("malformed", reason_code="checkpoint_pin_manifest_unavailable")
    result: dict[str, Any] = {"state": "available"}
    for key in ("pin_id", "run_id", "pinned_at", "pinned_ts", "note"):
        if key in manifest:
            result[key] = deepcopy(manifest[key])
    for key in ("source", "identity", "state", "blob"):
        if isinstance(manifest.get(key), Mapping):
            result[key] = deepcopy(dict(manifest[key]))
    return result


def _removable_candidates(run: Mapping[str, Any]) -> tuple[list[str], str | None]:
    from clozn.recipes.removability import plan_removability
    candidates = _source_candidates(run)
    valid: list[str] = []
    first_reason = None
    for source_id in candidates:
        try:
            plan_removability(run, [source_id])
        except Exception as exc:
            if first_reason is None:
                first_reason = str(getattr(exc, "reason", None) or "source_unavailable")
            continue
        valid.append(source_id)
    return valid, first_reason


def _capability_projection(run: Mapping[str, Any], *, historical_receipts: list,
                           checkpoint_pin: Mapping[str, Any] | None) -> dict[str, Any]:
    from clozn.recipes.context_effects import plan_context_effects
    from clozn.recipes.context_counterfactual import plan_context_counterfactual
    from clozn.runs.context_search_universe import plan_context_search_universe

    removable, removable_reason = _removable_candidates(run)
    capabilities: dict[str, Any] = {}
    try:
        effects = plan_context_effects(run)
        capabilities["context_effects"] = {
            "state": "available",
            "display_source_count": len(effects.display_source_ids),
            "measurable_source_count": len(effects.measurement_source_ids),
            "source_universe_basis": effects.source_universe_basis,
        }
    except Exception as exc:
        capabilities["context_effects"] = {
            "state": "unavailable", "reason_code": "context_effects_unavailable", "reason": str(exc),
            "display_source_count": 0, "measurable_source_count": 0,
        }

    if removable:
        capabilities["remove_and_test"] = {
            "state": "requires_input", "required_inputs": ["source_id"],
            "removable_source_count": len(removable), "candidate_source_ids": removable,
        }
    else:
        capabilities["remove_and_test"] = {
            "state": "unavailable", "reason_code": removable_reason or "no_removable_sources",
            "reason": "no canonical Context Receipt source can be validated for removal",
            "removable_source_count": 0,
        }

    contract, contract_reason = generation_contract_from_run(run)
    if not isinstance(contract, Mapping) or contract_reason:
        capabilities["context_counterfactual_generation"] = {
            "state": "unavailable", "reason_code": "generation_contract_unavailable",
            "reason": contract_reason or "the recorded generation contract is incomplete",
            "removable_source_count": len(removable),
        }
    elif not removable:
        capabilities["context_counterfactual_generation"] = {
            "state": "unavailable", "reason_code": removable_reason or "no_removable_sources",
            "reason": "the recorded generation contract is available but no canonical source is removable",
        }
    else:
        planned = []
        for source_id in removable:
            try:
                plan_context_counterfactual(run, source_id)
                planned.append(source_id)
            except Exception:
                continue
        capabilities["context_counterfactual_generation"] = {
            "state": "requires_input" if planned else "unavailable",
            "required_inputs": ["source_id"] if planned else [],
            "removable_source_count": len(planned),
            "candidate_source_ids": planned,
            **({} if planned else {"reason_code": "source_unavailable",
                                  "reason": "no source passed the counterfactual generation planner"}),
        }

    prerequisites = recorded_fork_prerequisites(run)
    try:
        universe = plan_context_search_universe(run, run.get("context_units"), max_units=50)
        universe_state = universe.get("status")
        if universe_state == "planned" and prerequisites["token_alignment_available"]:
            capabilities["minimal_context"] = {
                "state": "requires_verification", "universe_id": universe.get("universe_id"),
                "source_count": universe.get("source_count"),
                "exact_prerequisites": deepcopy(prerequisites),
            }
        else:
            condition = _mapping(universe.get("condition"))
            reason_code = condition.get("code") or (
                "recorded_token_trace_unavailable" if not prerequisites["token_alignment_available"]
                else "universe_unavailable"
            )
            capabilities["minimal_context"] = {
                "state": "unavailable", "reason_code": reason_code,
                "reason": condition.get("message") or "exact recorded-answer prerequisites are unavailable",
                "exact_prerequisites": deepcopy(prerequisites),
                "universe": deepcopy(universe),
            }
    except Exception as exc:
        capabilities["minimal_context"] = {
            "state": "unavailable", "reason_code": "universe_unavailable", "reason": str(exc),
            "exact_prerequisites": deepcopy(prerequisites),
        }

    try:
        from clozn.recipes.time_travel import time_travel_capabilities
        time_travel = time_travel_capabilities(run)
    except Exception as exc:
        time_travel = {"state": "unavailable", "reason_code": "time_travel_capabilities_unavailable",
                       "reason": type(exc).__name__}
    try:
        rewind = build_rewind_fidelity(run, historical_receipts=historical_receipts)
    except Exception as exc:
        rewind = {"state": "unavailable", "reason_code": "rewind_fidelity_unavailable",
                  "reason": type(exc).__name__}
    capabilities["time_travel"] = {
        "state": "available" if isinstance(time_travel, Mapping) and "available_operations" in time_travel
        else "unavailable",
        "projection": time_travel,
        "rewind_fidelity": rewind,
        "checkpoint_pin": _checkpoint_pin_projection(checkpoint_pin),
    }
    return capabilities


def _evidence_inventory(run: Mapping[str, Any], input_projection: Mapping[str, Any],
                        generation: Mapping[str, Any], output: Mapping[str, Any],
                        model_runtime: Mapping[str, Any], health: Mapping[str, Any],
                        checkpoint_pin: Mapping[str, Any] | None,
                        historical_receipts: list) -> dict[str, Any]:
    timing_state = output.get("timing_evidence", {}).get("state")
    pin_state = "unavailable"
    pin_reason = "checkpoint_pin_not_loaded"
    if checkpoint_pin is not None:
        if checkpoint_pin.get("ok") is True:
            pin_state, pin_reason = "available", None
        else:
            pin_state, pin_reason = "stale", "checkpoint_pin_unavailable"
    return {
        "context_receipt": input_projection["context_receipt"],
        "final_prompt": input_projection["final_prompt"],
        "assembled_messages": input_projection["assembled_messages"],
        "generation_contract": _state(
            "available" if generation.get("state") == "available" else "partial",
            reason_code=generation.get("reason_code"),
        ) if generation.get("state") != "available" else _state("available"),
        "response": output["response"],
        "token_ids": output["token_ids"],
        "token_pieces": output["token_pieces"],
        "token_logprobs": output["token_logprobs"],
        "token_alternatives": output["token_alternatives"],
        "runtime_identity": _state(
            "available" if model_runtime.get("state") == "available" else model_runtime.get("state", "unavailable"),
            reason_code="runtime_identity_contradictory" if model_runtime.get("state") == "contradictory" else None,
        ),
        "timing": _state(timing_state or "unavailable", reason_code="timing_unavailable"
                          if timing_state in {None, "unavailable"} else None),
        "checkpoint_pin": _state(pin_state, reason_code=pin_reason),
        "historical_exact_proofs": _state(
            "available" if historical_receipts else "unavailable",
            reason_code=None if historical_receipts else "historical_exact_proofs_not_loaded",
            count=len(historical_receipts),
        ),
        "execution_health": _state(
            "available" if health.get("state") == "available" else "partial",
            reason_code="health_degraded" if health.get("state") != "available" else None,
        ),
    }


def build_run_diagnostics(
    run: Mapping[str, Any], *, related_runs: Iterable[Mapping[str, Any]] = (),
    historical_receipts: Sequence[Mapping[str, Any]] = (),
    checkpoint_pin: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one deterministic, read-only ``clozn.run-diagnostics.v1`` document."""
    if not isinstance(run, Mapping) or not _string(run.get("id")):
        raise ValueError("build_run_diagnostics requires a recorded run with a non-empty id")
    receipt_view = read_receipt(dict(run))
    model_runtime = _runtime_projection(run)
    input_projection = _input_projection(run, receipt_view)
    generation = _generation_projection(run)
    output = _output_projection(run, receipt_view)
    health = _health_projection(run, related_runs)
    receipts = [deepcopy(dict(item)) for item in historical_receipts if isinstance(item, Mapping)]
    capabilities = _capability_projection(
        run, historical_receipts=receipts, checkpoint_pin=checkpoint_pin,
    )
    evidence = _evidence_inventory(
        run, input_projection, generation, output, model_runtime, health,
        checkpoint_pin, receipts,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run["id"],
        "model_runtime": model_runtime,
        "input": input_projection,
        "generation_contract": generation,
        "output": output,
        "execution_health": health,
        "evidence": evidence,
        "capabilities": capabilities,
    }
    schemas.validate(document, SCHEMA_VERSION)
    return document


__all__ = ["SCHEMA_VERSION", "build_run_diagnostics"]
