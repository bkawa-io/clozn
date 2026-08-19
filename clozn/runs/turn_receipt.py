"""Turn Receipt v1 -- Clozn's compact, read-side projection of one recorded run.

The Turn Receipt is deliberately a composition layer.  It reads the already-persisted Context Receipt,
Context Utilization, Context Tension, first-divergence, timing, and Rewind Fidelity evidence; it never
creates any of those artifacts and never asks a worker or model to produce more evidence.  The public
document contains metadata and bounded comparison pieces only -- never the full prompt or response.

All functions in this module are deterministic and side-effect free.  In particular, a missing influence
map is represented as ``not_measured`` rather than becoming an invitation to start a measurement.
"""
from __future__ import annotations

import math
import os
from collections.abc import Mapping


SCHEMA_VERSION = "clozn.turn-receipt.v1"
MAX_NOTABLE_SOURCES = 3


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _string(value) -> str | None:
    return value if isinstance(value, str) and value else None


def _int(value, *, minimum: int = 0) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        return None
    return int(value)


def _number(value, *, minimum: float = 0.0) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    if not math.isfinite(result) or result < minimum:
        return None
    return result


def _rounded(value: float, places: int = 3):
    result = round(float(value), places)
    return int(result) if result.is_integer() else result


def _run_id(run: Mapping) -> str:
    value = run.get("id") or run.get("run_id")
    if not isinstance(value, str) or not value:
        raise ValueError("run.id must be a non-empty string")
    return value


def _context_receipt(run: dict) -> tuple[str, dict]:
    from clozn.runs.context_receipt import read_receipt

    result = read_receipt(run)
    shape = result.get("shape") if isinstance(result, dict) else "absent"
    receipt = result.get("receipt") if isinstance(result, dict) else {}
    return (shape if isinstance(shape, str) else "absent", _dict(receipt))


def _receipt_privacy(shape: str, receipt: dict) -> str:
    privacy = receipt.get("privacy")
    if privacy in {"full", "metadata_only", "hashes_only", "off"}:
        return privacy
    # The pre-schema receipt had no privacy field.  Its labels were already part of the persisted
    # metadata, so treating it as metadata-only preserves what that artifact actually exposed.
    return "metadata_only" if shape == "legacy" else "off" if shape == "absent" else "metadata_only"


def _project_segment(segment: dict, *, allow_labels: bool) -> dict | None:
    segment_id = _string(segment.get("segment_id"))
    if segment_id is None:
        return None
    out = {"segment_id": segment_id}
    if allow_labels:
        label = _string(segment.get("source_label"))
        if label is not None:
            out["label"] = label
        client_id = _string(segment.get("client_source_id"))
        if client_id is not None:
            out["client_source_id"] = client_id
    if isinstance(segment.get("included"), bool):
        out["included"] = segment["included"]
    return out


def _context_projection(run: dict) -> dict:
    shape, receipt = _context_receipt(run)
    privacy = _receipt_privacy(shape, receipt)
    context: dict = {"privacy": privacy}

    # Privacy=off is intentionally a hard boundary.  Do not reconstruct provenance from the run's
    # message list or restore labels from any other artifact when the authoritative receipt opted out.
    if shape == "absent" or privacy == "off":
        context["provenance_state"] = "unavailable"
        return context

    limits = _dict(receipt.get("limits"))
    rendered = _dict(receipt.get("rendered"))
    prompt_tokens = _int(limits.get("prompt_tokens"))
    if prompt_tokens is None:
        prompt_tokens = _int(rendered.get("tokens"))
    if prompt_tokens is None:
        prompt_tokens = _int(rendered.get("token_count"))
    window_tokens = _int(receipt.get("context_window_tokens"))
    if window_tokens is None:
        window_tokens = _int(limits.get("context_window_tokens"))
    if prompt_tokens is not None:
        context["prompt_tokens"] = prompt_tokens
    if window_tokens is not None:
        context["context_window_tokens"] = window_tokens
    if prompt_tokens is not None and window_tokens is not None and window_tokens > 0:
        context["window_occupancy"] = _rounded(prompt_tokens / window_tokens, 4)

    delivered_raw = receipt.get("delivered")
    assembled_raw = receipt.get("assembled")
    delivered = [item for item in delivered_raw if isinstance(item, dict)] if isinstance(delivered_raw, list) else []
    assembled = [item for item in assembled_raw if isinstance(item, dict)] if isinstance(assembled_raw, list) else []

    legacy_delivered_count = None
    if isinstance(delivered_raw, dict) and isinstance(delivered_raw.get("messages"), list):
        legacy_delivered_count = sum(1 for item in delivered_raw["messages"] if isinstance(item, dict))

    # The old receipt carried assembled_messages under survived.  Count them when possible, but never
    # copy their contents into the Turn Receipt.
    if not assembled:
        survived = _dict(receipt.get("survived"))
        old_assembled = survived.get("assembled_messages")
        if isinstance(old_assembled, list):
            assembled_count = sum(1 for item in old_assembled if isinstance(item, dict))
        else:
            assembled_count = None
    else:
        assembled_count = len(assembled)

    omitted_ids = {
        item.get("segment_id") for item in delivered
        if item.get("included") is False and isinstance(item.get("segment_id"), str)
    }
    for omission in receipt.get("omissions") if isinstance(receipt.get("omissions"), list) else []:
        if isinstance(omission, dict) and isinstance(omission.get("segment_id"), str):
            omitted_ids.add(omission["segment_id"])
    sources = {"delivered": len(delivered) if delivered else (legacy_delivered_count or 0)}
    if assembled_count is not None:
        sources["assembled"] = assembled_count
    sources["omitted"] = len(omitted_ids)
    context["sources"] = sources

    provenance = []
    allow_labels = privacy not in {"hashes_only", "off"}
    for segment in delivered:
        projected = _project_segment(segment, allow_labels=allow_labels)
        if projected is not None:
            provenance.append(projected)
    context["provenance_state"] = "unavailable" if shape == "legacy" else "available"
    if provenance:
        context["provenance"] = provenance

    transformations = []
    raw_transformations = receipt.get("transformations")
    if isinstance(raw_transformations, list):
        for item in raw_transformations:
            reason = item.get("reason") if isinstance(item, dict) else item
            if isinstance(reason, str) and reason and reason not in transformations:
                transformations.append(reason)
    if transformations:
        context["transformations"] = transformations
    return context


def _outcome(run: dict, context: dict) -> dict:
    _shape, receipt = _context_receipt(run)
    termination = _dict(receipt.get("termination"))
    raw_reason = termination.get("reason")
    if not isinstance(raw_reason, str):
        raw_reason = None

    error = bool(run.get("error"))
    if error or raw_reason == "worker_error":
        state = "errored"
    elif raw_reason == "client_cancelled" or _dict(run.get("meta")).get("stream_failure") == "client_disconnected":
        state = "cancelled"
    elif raw_reason in {"max_tokens", "context_limit"} or run.get("finish_reason") == "length":
        state = "truncated"
    elif raw_reason in {"eos", "stop_sequence", "tool_call"} or run.get("finish_reason") in {"stop", "eos", "tool_calls"}:
        state = "completed"
    else:
        state = "unknown"

    result = {"state": state}
    if raw_reason is not None:
        result["finish_reason"] = raw_reason
    elif isinstance(run.get("finish_reason"), str) and run["finish_reason"]:
        result["finish_reason"] = run["finish_reason"]

    generated = _int(termination.get("generated_tokens"))
    if generated is None:
        generated = _int(_dict(receipt.get("limits")).get("generated_tokens"))
    if generated is None:
        generated = _int(_dict(run.get("meta")).get("generation_tokens"))
    if generated is None:
        trace = _dict(run.get("trace"))
        tokens = trace.get("tokens")
        if isinstance(tokens, list):
            generated = len(tokens)
    if generated is not None:
        result["generated_tokens"] = generated
    return result


def _model(run: dict) -> dict:
    meta = _dict(run.get("meta"))
    identity = _dict(run.get("identity"))
    result: dict = {}
    name = _string(run.get("model")) or _string(meta.get("model"))
    if name:
        # Model paths are runtime identity, not the everyday model name.  Keep a human-oriented basename
        # when a legacy run recorded the path in its model field.
        name = os.path.basename(name).removesuffix(".gguf") or name
        result["name"] = name
    quant = _string(meta.get("quant")) or _string(meta.get("quantization")) or _string(identity.get("quant"))
    if quant:
        result["quant"] = quant
    substrate = _string(run.get("substrate")) or _string(meta.get("substrate"))
    if substrate:
        result["substrate"] = substrate
    return result


def _context_utilization(run: dict):
    stored = run.get("context_utilization")
    if isinstance(stored, dict) and stored.get("schema_version") == "clozn.context-utilization.v1":
        return stored
    # The authoritative persisted source for current runs is the influence map.  An absent map is not
    # an instruction to run the scorer; it is an ordinary not_measured state.
    if not isinstance(run.get("influence_map"), dict):
        return None
    try:
        from clozn.runs.context_utilization import build_context_utilization
        return build_context_utilization(run)
    except Exception:
        return {"measurement": {"state": "unavailable", "reason": "invalid_persisted_measurement"}, "sources": [], "summary": {}}


def _source_metadata(run: dict, context: dict) -> dict:
    shape, receipt = _context_receipt(run)
    privacy = _receipt_privacy(shape, receipt)
    allow_labels = privacy not in {"hashes_only", "off"}
    by_segment = {
        item.get("segment_id"): item for item in context.get("provenance", [])
        if isinstance(item, dict) and isinstance(item.get("segment_id"), str)
    }
    by_client = {
        item.get("client_source_id"): item for item in context.get("provenance", [])
        if isinstance(item, dict) and isinstance(item.get("client_source_id"), str)
    }
    influence = _dict(run.get("influence_map"))
    prompt_sources = influence.get("prompt_sources") if isinstance(influence.get("prompt_sources"), list) else []
    by_native = {
        item.get("id"): item for item in prompt_sources
        if isinstance(item, dict) and isinstance(item.get("id"), str)
    }
    result = {}
    for raw in prompt_sources:
        if not isinstance(raw, dict) or not isinstance(raw.get("id"), str):
            continue
        native_id = raw["id"]
        candidates = [raw.get("segment_id"), raw.get("client_source_id"), raw.get("source_id")]
        matched = next(
            (
                candidate
                for value in candidates
                if value
                for candidate in (by_segment.get(value) or by_client.get(value),)
                if candidate is not None
            ),
            None,
        )
        entry = {}
        if allow_labels:
            label = _string(raw.get("source_label")) or _string(raw.get("label"))
            if label is None and isinstance(matched, dict):
                label = _string(matched.get("label"))
            if label:
                entry["label"] = label
        result[native_id] = entry
    return result


def _effect_for_source(source: dict) -> str | None:
    supporting = _int(source.get("supporting_clear_links"), minimum=0) or 0
    suppressing = _int(source.get("suppressing_clear_links"), minimum=0) or 0
    if supporting and not suppressing:
        return "supporting"
    if suppressing and not supporting:
        return "suppressing"
    if supporting and suppressing:
        return "mixed"
    return None


def _what_mattered(run: dict, context: dict) -> dict:
    utilization = _context_utilization(run)
    if utilization is None:
        return {"measurement_state": "not_measured"}
    measurement = _dict(utilization.get("measurement"))
    state = measurement.get("state") or utilization.get("measurement_state")
    if state not in {"available", "not_measured", "unavailable"}:
        state = "unavailable"
    result = {"measurement_state": state}
    if state != "available":
        return result

    summary = _dict(utilization.get("summary"))
    sources = utilization.get("sources") if isinstance(utilization.get("sources"), list) else []
    prompt_sources = _int(summary.get("prompt_sources"), minimum=0)
    measured_sources = _int(summary.get("measured_sources"), minimum=0)
    not_measured_sources = _int(summary.get("sources_not_measured"), minimum=0)
    if prompt_sources is None:
        prompt_sources = len(sources)
    if measured_sources is None:
        measured_sources = sum(1 for source in sources if isinstance(source, dict)
                               and source.get("measurement_state") == "measured")
    if not_measured_sources is None:
        not_measured_sources = max(0, prompt_sources - measured_sources)
    result["coverage"] = {
        "prompt_sources": prompt_sources,
        "measured_sources": measured_sources,
        "not_measured_sources": not_measured_sources,
    }
    clear = _int(summary.get("sources_with_clear_measured_effect"), minimum=0)
    below = _int(summary.get("sources_below_measured_floor"), minimum=0)
    if clear is None:
        clear = sum(1 for source in sources if isinstance(source, dict)
                    and source.get("effect_state") == "clear_measured_effect")
    if below is None:
        below = sum(1 for source in sources if isinstance(source, dict)
                    and source.get("effect_state") == "below_measured_floor")
    result["effect_summary"] = {
        "sources_with_clear_measured_effect": clear,
        "sources_below_measured_floor": below,
    }

    source_metadata = _source_metadata(run, context)
    notable = []
    for source in sources:                       # context-utilization supplies the deterministic order
        if not isinstance(source, dict) or source.get("measurement_state") != "measured":
            continue
        if source.get("effect_state") != "clear_measured_effect":
            continue
        effect = _effect_for_source(source)
        if effect is None:
            continue
        source_span_id = _string(source.get("source_span_id"))
        if source_span_id is None:
            continue
        item = {"source_span_id": source_span_id, "effect": effect}
        native = _dict(source.get("native"))
        metadata = source_metadata.get(native.get("source_id"), {})
        if isinstance(metadata, dict) and _string(metadata.get("label")):
            item["label"] = metadata["label"]
        if len(notable) >= MAX_NOTABLE_SOURCES:
            break
        notable.append(item)
    if notable:
        result["notable_sources"] = notable
    return result


def _context_tension_artifact(run: dict) -> dict | None:
    stored = run.get("context_tension")
    if isinstance(stored, dict) and stored.get("schema_version") == "clozn.context-tension.v1":
        return stored
    elif isinstance(run.get("influence_map"), dict):
        try:
            from clozn.runs.context_tension import build_context_tension
            return build_context_tension(run)
        except Exception:
            return {"measurement": {"state": "unavailable"}, "summary": {}}
    return None


def _context_tension(run: dict, *, artifact: dict | None = None) -> dict:
    tension = _context_tension_artifact(run) if artifact is None else artifact
    if tension is None:
        return {"measurement_state": "not_measured"}
    measurement = _dict(tension.get("measurement"))
    state = measurement.get("state") or tension.get("measurement_state")
    if state == "available":
        summary = _dict(tension.get("summary"))
        return {
            "measurement_state": "available",
            "answer_spans_with_tension": _int(summary.get("answer_spans_with_tension"), minimum=0) or 0,
            "tension_pairs": _int(summary.get("tension_pairs"), minimum=0) or 0,
        }
    if state == "not_measured":
        return {"measurement_state": "not_measured"}
    return {"measurement_state": "unavailable"}


def _divergence_candidate(run: dict) -> tuple[dict, str | None] | None:
    parent_id = _string(run.get("parent_run_id"))
    candidates = []
    for key in ("first_divergence_view", "comparison", "diff"):
        value = run.get(key)
        if isinstance(value, dict):
            candidates.append(value)
            nested = value.get("first_divergence_view")
            if isinstance(nested, dict):
                candidates.insert(0, nested)
    direct = run.get("first_divergence")
    if isinstance(direct, dict):
        candidates.append({"state": "available", "divergence": direct})
    for candidate in candidates:
        if candidate.get("state") not in {None, "available"}:
            continue
        divergence = candidate.get("divergence")
        if not isinstance(divergence, dict):
            divergence = candidate.get("first_divergence")
        if not isinstance(divergence, dict):
            continue
        index = _int(divergence.get("index"), minimum=0)
        kind = _string(divergence.get("kind"))
        if index is None or kind is None:
            continue
        a_side = _dict(divergence.get("a"))
        b_side = _dict(divergence.get("b"))
        a_piece = divergence.get("a_piece")
        b_piece = divergence.get("b_piece")
        if a_piece is None:
            a_piece = a_side.get("piece")
        if b_piece is None:
            b_piece = b_side.get("piece")
        compact = {"index": index, "kind": kind}
        if isinstance(a_piece, str):
            compact["a_piece"] = a_piece[:128]
        if isinstance(b_piece, str):
            compact["b_piece"] = b_piece[:128]
        return compact, parent_id or _string(candidate.get("a_run_id"))
    return None


def _comparison(run: dict, parent_run: dict | None) -> dict | None:
    found = _divergence_candidate(run)
    if found is None:
        return None
    divergence, candidate_parent = found
    parent_id = _string(run.get("parent_run_id"))
    if parent_id is None and isinstance(parent_run, dict):
        parent_id = _string(parent_run.get("id"))
    parent_id = parent_id or candidate_parent
    if parent_id is None:
        return None
    return {"state": "available", "parent_run_id": parent_id, "first_divergence": divergence}


# A run recorded before the evidence convergence may carry a stored v1 rewind-fidelity document.
# Its shape is read-compatible here, so an older receipt keeps rendering rather than silently
# rebuilding a projection over evidence that no longer exists for it.
_STORED_REWIND_VERSIONS = {"clozn.rewind-fidelity.v1", "clozn.rewind-fidelity.v2"}


def _rewind(run: dict, historical_observations=()) -> dict:
    stored = run.get("rewind_fidelity")
    if not (isinstance(stored, dict) and stored.get("schema_version") in _STORED_REWIND_VERSIONS):
        try:
            from clozn.replay.rewind_fidelity import build_rewind_fidelity
            stored = build_rewind_fidelity(run, historical_observations=list(historical_observations or ()))
        except Exception:
            stored = {}
    capability = _dict(stored.get("recorded_capability"))
    reconstructed = _dict(capability.get("reconstructed_replay"))
    exact = _dict(capability.get("exact_rewind"))
    proof = _dict(stored.get("historical_proof"))
    boundaries = proof.get("verified_boundaries")
    return {
        "reconstructed_replay": (
            reconstructed.get("state") if reconstructed.get("state") in {"available", "unavailable"}
            else "unavailable"
        ),
        "exact_rewind": (
            exact.get("state") if exact.get("state") in {"requires_live_plan", "static_prerequisites_unavailable"}
            else "static_prerequisites_unavailable"
        ),
        "historically_verified_boundaries": len(boundaries) if isinstance(boundaries, list) else 0,
    }


def _phase_ms(meta: dict, names: tuple[str, ...]) -> float | None:
    for key in names:
        value = _number(meta.get(key))
        if value is not None:
            return value
    candidates = []
    for container_name in ("worker_timing", "gateway_timing", "generation_timing"):
        container = _dict(meta.get(container_name))
        for key in names:
            direct = _number(container.get(key))
            if direct is not None:
                candidates.append(direct)
        phases = container.get("phases")
        if not isinstance(phases, list):
            continue
        for phase in phases:
            if not isinstance(phase, dict) or phase.get("name") not in names:
                continue
            duration_ns = _int(phase.get("duration_ns"), minimum=0)
            if duration_ns is not None:
                candidates.append(duration_ns / 1_000_000)
            else:
                duration_ms = _number(phase.get("duration_ms"))
                if duration_ms is not None:
                    candidates.append(duration_ms)
    return candidates[0] if candidates else None


def _performance(run: dict, context: dict, outcome: dict) -> dict:
    meta = _dict(run.get("meta"))
    timing = _dict(run.get("timing"))
    phase_meta = dict(timing)
    phase_meta.update(meta)
    for key in ("worker_timing", "gateway_timing", "generation_timing"):
        if key not in phase_meta and isinstance(run.get(key), dict):
            phase_meta[key] = run[key]
    result: dict = {}
    prompt_tokens = _int(context.get("prompt_tokens"))
    if prompt_tokens is not None:
        result["prompt_tokens"] = prompt_tokens
    output_tokens = _int(outcome.get("generated_tokens"))
    if output_tokens is not None:
        result["output_tokens"] = output_tokens

    prefill = _phase_ms(phase_meta, (
        "prefill", "prefill_ms", "prefill_duration_ms", "prompt_eval", "prompt_eval_ms",
        "prompt_eval_duration_ms",
    ))
    decode = _phase_ms(phase_meta, (
        "decode", "decode_ms", "generation_ms", "generation_duration_ms", "eval", "eval_ms",
        "eval_duration_ms",
    ))
    if prefill is not None:
        result["prefill_ms"] = _rounded(prefill)
    if decode is not None:
        result["decode_ms"] = _rounded(decode)

    prompt_rate = _number(phase_meta.get("prompt_tokens_per_second"))
    generation_rate = _number(phase_meta.get("generation_tokens_per_second"))
    if generation_rate is None:
        generation_rate = _number(phase_meta.get("decode_tokens_per_second"))
    if prompt_rate is None and prompt_tokens is not None and prefill and prefill > 0:
        prompt_rate = prompt_tokens / (prefill / 1000)
    if generation_rate is None and output_tokens is not None and decode and decode > 0:
        generation_rate = output_tokens / (decode / 1000)
    if prompt_rate is not None:
        result["prompt_tokens_per_second"] = _rounded(prompt_rate)
    if generation_rate is not None:
        result["generation_tokens_per_second"] = _rounded(generation_rate)

    duration = _number(_dict(run.get("timing")).get("duration_ms"))
    if duration is not None:
        result["duration_ms"] = _rounded(duration)
    return result


def _technical(run: dict) -> dict:
    meta = _dict(run.get("meta"))
    identity = _dict(run.get("identity"))
    result = {}
    for key in ("model_sha256", "template_fingerprint", "engine_build", "clozn_version"):
        value = _string(identity.get(key)) or _string(meta.get(key))
        if value:
            result[key] = value
    for key in ("backend", "device", "runtime_class", "engine"):
        value = _string(identity.get(key)) or _string(meta.get(key))
        if value:
            result[key] = value
    return result


def build_turn_receipt(run, *, parent_run=None, historical_observations=()) -> dict:
    """Build one deterministic, metadata-only ``clozn.turn-receipt.v1`` document.

    The function accepts the recorded run and optional already-loaded parent/rewind evidence.  It does
    not load them itself, write anything, call a model, start a worker, score tokens, create a checkpoint,
    or execute a rewind.  Invalid optional derived evidence becomes an explicit unavailable state.
    """
    record = dict(run) if isinstance(run, dict) else {}
    run_id = _run_id(record)
    context = _context_projection(record)
    outcome = _outcome(record, context)
    what_mattered = _what_mattered(record, context)
    tension_artifact = _context_tension_artifact(record)
    tension = _context_tension(record, artifact=tension_artifact)
    comparison = _comparison(record, parent_run if isinstance(parent_run, dict) else None)
    rewind = _rewind(record, historical_observations=historical_observations)
    performance = _performance(record, context, outcome)

    # signals.py owns the registry and detector.  The footer consumes this exact list rather than
    # running another independent set of checks.
    from clozn.runs import signals
    signal_list = signals.build_structured_signals(
        record, outcome=outcome, context=context, what_mattered=what_mattered,
        context_tension=tension, comparison=comparison,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "outcome": outcome,
        "model": _model(record),
        "context": context,
        "what_mattered": what_mattered,
        "context_tension": tension,
        "signals": signal_list,
        "performance": performance,
        "comparison": comparison,
        "rewind": rewind,
        "technical": _technical(record),
    }
    # Selection targets are a read-only decoration over the already-visible Receipt findings.  The
    # target helper owns canonical selection validation and reference encoding; it never broadens the
    # Receipt's evidence or starts an expensive operation.
    from clozn.runs.receipt_inspection import attach_inspection_targets
    document = attach_inspection_targets(
        record, document, context_tension_artifact=tension_artifact,
    )
    from clozn import schemas
    schemas.validate(document)
    return document


def _md_text(value) -> str:
    if not isinstance(value, str):
        return ""
    return value.replace("`", "'\u0060").replace("\r", " ").replace("\n", " ").strip()


def _md_number(value, *, decimals: int = 1) -> str:
    number = _number(value)
    if number is None:
        return ""
    if decimals == 0:
        return f"{number:,.0f}"
    return f"{number:,.{decimals}f}".rstrip("0").rstrip(".")


def to_markdown(turn_receipt: dict) -> str:
    """Render a compact, pasteable Markdown view of a Turn Receipt.

    Only fields already present in the metadata projection are rendered.  No prompt, full response, or
    arbitrary footer text is reconstructed here.
    """
    receipt = turn_receipt if isinstance(turn_receipt, dict) else {}
    model = _dict(receipt.get("model"))
    model_parts = [_md_text(model.get(key)) for key in ("name", "quant", "substrate")]
    model_parts = [part for part in model_parts if part]
    lines = ["# Clozn Receipt", ""]
    if model_parts:
        lines += [" · ".join(model_parts), ""]

    outcome = _dict(receipt.get("outcome"))
    state_labels = {
        "completed": "Completed normally",
        "truncated": "Generation truncated",
        "errored": "Run errored",
        "cancelled": "Cancelled",
        "unknown": "Outcome unknown",
    }
    outcome_line = state_labels.get(outcome.get("state"), "Outcome unknown")
    if _int(outcome.get("generated_tokens")) is not None:
        outcome_line += f" · {_md_number(outcome['generated_tokens'], decimals=0)} output tokens"
    lines += [outcome_line, "", "## Context", ""]

    context = _dict(receipt.get("context"))
    prompt = _int(context.get("prompt_tokens"))
    window = _int(context.get("context_window_tokens"))
    if prompt is not None and window is not None:
        lines.append(f"{_md_number(prompt, decimals=0)} / {_md_number(window, decimals=0)} context tokens")
        occupancy = _number(context.get("window_occupancy"))
        if occupancy is not None:
            lines.append(f"Context-window occupancy: {_md_number(occupancy * 100, decimals=1)}%")
    elif prompt is not None:
        lines.append(f"{_md_number(prompt, decimals=0)} context tokens")
    sources = _dict(context.get("sources"))
    if _int(sources.get("delivered")) is not None:
        lines.append(f"{_md_number(sources['delivered'], decimals=0)} sources reached the model")
    omitted = _int(sources.get("omitted"))
    if omitted is not None and omitted > 0:
        lines.append(f"{_md_number(omitted, decimals=0)} context sources were omitted from assembly")
    elif omitted == 0:
        lines.append("No context omissions detected")
    if context.get("provenance_state") == "unavailable":
        lines.append("Context provenance unavailable")
    lines += ["", "## What mattered", ""]

    mattered = _dict(receipt.get("what_mattered"))
    measurement_state = mattered.get("measurement_state")
    if measurement_state == "not_measured":
        lines.append("Influence measurement: not measured")
    elif measurement_state == "unavailable":
        lines.append("Influence measurement unavailable")
    elif measurement_state == "available":
        coverage = _dict(mattered.get("coverage"))
        measured = _int(coverage.get("measured_sources"))
        total = _int(coverage.get("prompt_sources"))
        if measured is not None and total is not None:
            lines.append(f"Influence coverage: {_md_number(measured, decimals=0)} of {_md_number(total, decimals=0)} context sources measured")
        effects = _dict(mattered.get("effect_summary"))
        clear = _int(effects.get("sources_with_clear_measured_effect"))
        below = _int(effects.get("sources_below_measured_floor"))
        not_measured = _int(coverage.get("not_measured_sources"))
        if clear is not None and measured is not None:
            lines.append(f"{_md_number(clear, decimals=0)} measured sources showed a clear effect")
        if below is not None and below > 0:
            lines.append(f"{_md_number(below, decimals=0)} measured sources showed no clear effect above the configured measurement floor.")
        if not_measured is not None and not_measured > 0:
            lines.append(f"{_md_number(not_measured, decimals=0)} sources were not measured")
        notable = mattered.get("notable_sources")
        if isinstance(notable, list) and notable:
            lines += ["", "Notable measured effects:"]
            for item in notable[:MAX_NOTABLE_SOURCES]:
                if not isinstance(item, dict):
                    continue
                label = _md_text(item.get("label")) or "Context source"
                effect = _md_text(item.get("effect"))
                lines.append(f"- {label} — {effect}" if effect else f"- {label}")

    lines += ["", "## Signals", ""]
    signal_list = receipt.get("signals")
    if isinstance(signal_list, list) and signal_list:
        for signal in signal_list:
            if isinstance(signal, dict):
                summary = _md_text(signal.get("summary"))
                if summary:
                    lines.append(summary)
    else:
        lines.append("No attention signals recorded.")

    performance = _dict(receipt.get("performance"))
    performance_lines = []
    if _number(performance.get("prompt_tokens")) is not None:
        performance_lines.append(f"Prompt {_md_number(performance['prompt_tokens'], decimals=0)} tokens")
    if _number(performance.get("output_tokens")) is not None:
        performance_lines.append(f"Output {_md_number(performance['output_tokens'], decimals=0)} tokens")
    if _number(performance.get("prefill_ms")) is not None:
        performance_lines.append(f"Prefill {_md_number(performance['prefill_ms'])} ms")
    if _number(performance.get("decode_ms")) is not None:
        performance_lines.append(f"Decode {_md_number(performance['decode_ms'])} ms")
    if _number(performance.get("prompt_tokens_per_second")) is not None:
        performance_lines.append(f"Prefill {_md_number(performance['prompt_tokens_per_second'])} tok/s")
    if _number(performance.get("generation_tokens_per_second")) is not None:
        performance_lines.append(f"Generation {_md_number(performance['generation_tokens_per_second'])} tok/s")
    if performance_lines:
        lines += ["", "## Performance", ""] + performance_lines

    comparison = receipt.get("comparison")
    if isinstance(comparison, dict) and comparison.get("state") == "available":
        divergence = _dict(comparison.get("first_divergence"))
        lines += ["", "## Comparison", ""]
        if _int(divergence.get("index")) is not None:
            lines.append(f"This branch first diverged at token {divergence['index']}.")

    rewind = _dict(receipt.get("rewind"))
    lines += ["", "## Rewind", ""]
    if rewind.get("reconstructed_replay") == "available":
        lines.append("Structural replay available")
    else:
        lines.append("Structural replay unavailable")
    if rewind.get("exact_rewind") == "requires_live_plan":
        lines.append("Exact rewind requires a live check")
    else:
        lines.append("Exact rewind prerequisites unavailable")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["MAX_NOTABLE_SOURCES", "SCHEMA_VERSION", "build_turn_receipt", "to_markdown"]
