"""Read-only synthesis for one run's answer-investigation surface.

The underlying evidence producers intentionally keep their own closed vocabularies.  This module does
not rewrite them into one pseudo-score.  It adds a small outer state describing what a caller may
honestly conclude *about availability* while retaining each artifact's native schema, status, method,
thresholds, and reasons.

Building an investigation is model-free and mutation-free.  Expensive measurements are returned as
typed action descriptors; this function never starts an influence job, NLI check, replay, or retry.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "clozn.run-investigation.v1"

EVIDENCE_STATES = frozenset({
    "supported",
    "measured_effect",
    "below_measurement_floor",
    "delivered_not_measured",
    "omitted",
    "unavailable",
    "failed",
    "inconclusive",
})

_SEGMENT_KEYS = (
    "segment_id",
    "source_type",
    "source_label",
    "client_source_id",
    "original_order",
    "delivered_bytes",
    "content_hash",
    "included",
    "reason",
    "redaction_state",
)
_KNOWN_INFLUENCE_STATES = frozenset({"causally_supported", "observed"})


def _artifact(schema: str, *, method: str, native_status: str | None = None) -> dict:
    out = {"schema": schema, "method": method}
    if isinstance(native_status, str) and native_status:
        out["native_status"] = native_status
    return out


def _segments(value: object) -> list[dict]:
    if not isinstance(value, list):
        return []
    return [
        {key: item[key] for key in _SEGMENT_KEYS if key in item}
        for item in value
        if isinstance(item, Mapping)
    ]


def _action(
    action_id: str,
    label: str,
    kind: str,
    method: str,
    href: str,
    availability: str,
    **extra: Any,
) -> dict:
    return {
        "id": action_id,
        "label": label,
        "kind": kind,
        "method": method,
        "href": href,
        "availability": availability,
        **{key: value for key, value in extra.items() if value is not None},
    }


def _context_section(run: Mapping[str, Any]) -> dict:
    from clozn.runs.context_receipt import read_receipt

    view = read_receipt(dict(run))
    shape = str(view.get("shape") or "absent")
    if shape == "absent":
        return {
            "state": "unavailable",
            "reason": "no context receipt was recorded for this run",
        }

    receipt = view.get("receipt")
    if not isinstance(receipt, Mapping):
        return {
            "state": "failed",
            "reason": "the stored context receipt is not an object",
        }

    if shape == "legacy":
        delivered = receipt.get("delivered")
        messages = delivered.get("messages") if isinstance(delivered, Mapping) else None
        return {
            "state": "delivered_not_measured",
            "artifact": _artifact(
                str(receipt.get("schema") or "clozn.context_receipt.v1"),
                method="legacy_gateway_capture",
                native_status="legacy",
            ),
            "delivered_count": len(messages) if isinstance(messages, list) else 0,
            "privacy": "legacy_unspecified",
            "reason": (
                "legacy context evidence proves delivery but has no normalized segment projection; "
                "raw message content is not duplicated into this investigation response"
            ),
        }

    delivered = _segments(receipt.get("delivered"))
    assembled = _segments(receipt.get("assembled"))
    omitted = [
        {**segment, "state": "omitted"}
        for segment in delivered
        if segment.get("included") is False
    ]
    explicit_omissions = receipt.get("omissions")
    if isinstance(explicit_omissions, list):
        omitted.extend(
            {**deepcopy(item), "state": "omitted"}
            for item in explicit_omissions
            if isinstance(item, Mapping)
        )

    # The exact rendered prompt can be obtained from the dedicated context-receipt route under its
    # privacy rules.  The investigation endpoint carries only metadata, never a second raw-text copy.
    return {
        "state": "delivered_not_measured",
        "artifact": _artifact(
            str(receipt.get("schema_version") or "clozn.context-receipt.v1"),
            method="gateway_context_receipt",
            native_status=shape,
        ),
        "privacy": receipt.get("privacy", "unknown"),
        "delivered": delivered,
        "assembled": assembled,
        "omitted": omitted,
        "rendered": deepcopy(receipt.get("rendered") or {}),
        "termination": deepcopy(receipt.get("termination") or {}),
        "limits": deepcopy(receipt.get("limits") or {}),
        "transformations": deepcopy(receipt.get("transformations") or []),
    }


def _influence_section(
    run: Mapping[str, Any],
    *,
    scoring_available: bool,
    context_available: bool,
) -> tuple[dict, dict | None]:
    run_id = str(run.get("id") or "")
    measurement_action = _action(
        "measure_prompt_source_influence",
        "Measure what mattered",
        "measurement",
        "POST",
        f"/runs/{run_id}/influence-map/jobs",
        "ready" if scoring_available else "unavailable",
        reason=None if scoring_available else "the active worker does not expose token scoring",
        request_body={},
    )
    stored = run.get("influence_map")
    if not isinstance(stored, Mapping):
        state = "delivered_not_measured" if context_available else "unavailable"
        return ({
            "state": state,
            "reason": (
                "context delivery is recorded, but no prompt/source influence experiment has run"
                if context_available else
                "neither context delivery nor prompt/source influence evidence is available"
            ),
            "action_id": measurement_action["id"],
        }, measurement_action)

    persisted_unavailable = stored.get("unavailable")
    if isinstance(persisted_unavailable, str) and persisted_unavailable:
        return ({
            "state": "unavailable",
            "artifact": _artifact(
                "clozn.context_answer_influence.v1",
                method="persisted_influence_artifact",
                native_status="unavailable",
            ),
            "reason": persisted_unavailable,
            "action_id": measurement_action["id"],
        }, measurement_action)

    native_status = str(stored.get("status") or "unknown")
    native_schema = stored.get("schema")
    if native_schema != "clozn.context_answer_influence.v1":
        return ({
            "state": "failed",
            "artifact": _artifact(
                str(native_schema or "clozn.context_answer_influence.v1"),
                method="persisted_influence_artifact",
                native_status=native_status,
            ),
            "reason": "persisted influence evidence has an unsupported or missing schema",
            "action_id": measurement_action["id"],
        }, measurement_action)
    native_method = (
        stored.get("method", {}).get("name")
        if isinstance(stored.get("method"), Mapping)
        else None
    )
    ref = _artifact(
        str(native_schema),
        method=(
            str(native_method)
            if isinstance(native_method, str) and native_method
            else "teacher_forced_counterfactual"
        ),
        native_status=native_status,
    )
    if stored.get("available") is not True:
        error = stored.get("error") if isinstance(stored.get("error"), Mapping) else {}
        state = "failed" if native_status == "error" else (
            "inconclusive" if native_status == "inconclusive" else "unavailable"
        )
        return ({
            "state": state,
            "artifact": ref,
            "reason": str(error.get("message") or stored.get("note")
                          or "the stored influence measurement is unavailable"),
            "action_id": measurement_action["id"],
        }, measurement_action)

    # The investigation response is a synthesis/navigation surface, not a second raw-text export.
    # Reuse the evidence producer's own metadata-only projection so source/answer text becomes hashes
    # while span IDs, offsets, methods, thresholds, and numeric measurements remain available.
    from clozn.receipts.context_answer_influence import portable_export
    projected = portable_export(dict(stored), privacy="metadata_only")
    links = [deepcopy(item) for item in projected.get("links", []) if isinstance(item, Mapping)]
    unknown = sorted({
        str(item.get("evidence_state"))
        for item in links
        if item.get("evidence_state") not in _KNOWN_INFLUENCE_STATES
    })
    if unknown:
        return ({
            "state": "inconclusive",
            "artifact": ref,
            "reason": "unknown native influence evidence state(s): " + ", ".join(unknown),
            "artifact_sha256": projected.get("source_artifact_sha256"),
        }, None)

    clears = any(item.get("evidence_state") == "causally_supported" for item in links)
    return ({
        "state": "measured_effect" if clears else "below_measurement_floor",
        "artifact": ref,
        "artifact_sha256": projected.get("source_artifact_sha256"),
        "privacy": "metadata_only",
        "prompt_sources": deepcopy(projected.get("prompt_sources") or []),
        "prompt_spans": deepcopy(projected.get("prompt_spans") or []),
        "answer_spans": deepcopy(projected.get("answer_spans") or []),
        "links": links,
        "thresholds": deepcopy(projected.get("thresholds") or {}),
        "summary": deepcopy(projected.get("summary") or {}),
    }, None)


def _span_address_section(run: Mapping[str, Any]) -> tuple[dict, dict]:
    """Summarize and link the pure metadata-only span projection."""
    from clozn import schemas
    from clozn.runs.text_span_addresses import build_persisted_text_span_addresses

    run_id = str(run.get("id") or "")
    action = _action(
        "open_text_span_addresses",
        "Open stable text spans",
        "navigation",
        "GET",
        f"/runs/{run_id}/span-addresses",
        "ready",
    )
    try:
        document = build_persisted_text_span_addresses(dict(run), privacy="metadata_only")
    except (TypeError, ValueError, UnicodeError, schemas.ValidationError) as exc:
        # This is a derived read surface. Fail it closed without hiding the
        # remaining investigation sections or echoing potentially private
        # values from malformed old artifacts.
        return ({
            "state": "failed",
            "artifact": _artifact(
                "clozn.text-span-addresses.v1",
                method="stable_text_span_projection",
                native_status="failed",
            ),
            "privacy": "metadata_only",
            "reason": f"stable span projection failed: {type(exc).__name__}",
            "action_id": action["id"],
        }, action)

    resolution_counts: dict[str, int] = {}
    for address in document.get("addresses") or []:
        resolution = address.get("resolution") if isinstance(address, Mapping) else None
        state = resolution.get("state") if isinstance(resolution, Mapping) else "unavailable"
        state = str(state or "unavailable")
        resolution_counts[state] = resolution_counts.get(state, 0) + 1
    influence_source = next((
        source for source in document.get("source_artifacts") or []
        if isinstance(source, Mapping)
        and str(source.get("schema") or "").startswith("clozn.context")
        and "influence" in str(source.get("schema") or "")
    ), None)
    influence_status = (
        str(influence_source.get("native_status") or "unknown")
        if isinstance(influence_source, Mapping) else "unknown"
    )
    address_count = len(document.get("addresses") or [])
    section = {
        "state": "supported" if address_count else "unavailable",
        "artifact": _artifact(
            "clozn.text-span-addresses.v1",
            method="stable_text_span_projection",
            native_status=(
                "partial"
                if address_count and influence_status in {"not_recorded", "unavailable", "failed"}
                else "available" if address_count else "unavailable"
            ),
        ),
        "privacy": "metadata_only",
        "href": action["href"],
        "address_count": address_count,
        "resolution_counts": resolution_counts,
        "influence_native_status": influence_status,
        "source_artifacts": deepcopy(document.get("source_artifacts") or []),
        "action_id": action["id"],
    }
    if influence_status in {"not_recorded", "unavailable", "failed"}:
        source_reason = (
            influence_source.get("reason") if isinstance(influence_source, Mapping) else None
        )
        section["reason"] = str(
            source_reason
            or "context addresses are available, but persisted influence spans are not"
        )
    elif not address_count:
        section["reason"] = "no canonical text-span references could be derived from this run"
    return section, action


def _comparison_section(
    run: Mapping[str, Any],
    related_runs: Sequence[Mapping[str, Any]],
) -> tuple[dict, dict]:
    from clozn.analysis import run_diff

    run_id = str(run.get("id") or "")
    records = [dict(item) for item in related_runs if isinstance(item, Mapping)]
    by_id = {str(item.get("id")): item for item in records if item.get("id")}
    reference = None
    selection: dict[str, Any] = {}
    parent_id = run.get("parent_run_id")
    if isinstance(parent_id, str) and parent_id in by_id:
        reference = by_id[parent_id]
        selection = {"mode": "parent", "reference_run_id": parent_id}
    else:
        selected = run_diff.select_reference_run(dict(run), records, mode="previous_compatible")
        if selected.get("ok") is True and isinstance(selected.get("run"), Mapping):
            reference = dict(selected["run"])
            selection = deepcopy(selected.get("selection") or {"mode": "previous_compatible"})

    if reference is None:
        action = _action(
            "compare_with_run",
            "Compare with another run",
            "navigation",
            "GET",
            f"/runs/compare?a={{reference_run_id}}&b={run_id}",
            "conditional",
            requires=["reference_run_id"],
            reason="no unambiguous earlier compatible run was found automatically",
        )
        return ({
            "state": "unavailable",
            "reason": action["reason"],
            "action_id": action["id"],
        }, action)

    reference_id = str(reference.get("id"))
    action = _action(
        "compare_with_reference",
        "Open run comparison",
        "navigation",
        "GET",
        f"/runs/compare?a={reference_id}&b={run_id}",
        "ready",
    )
    try:
        comparison = run_diff.compare_runs(reference, dict(run))
    except Exception as exc:  # structural synthesis must fail closed, never lose the rest of the endpoint
        return ({
            "state": "failed",
            "reason": f"structural run comparison failed: {type(exc).__name__}: {exc}",
            "action_id": action["id"],
        }, action)
    return ({
        "state": "supported",
        "artifact": _artifact(
            str(comparison.get("schema_version") or "clozn.run-diff.v1"),
            method="structural_run_diff",
            native_status="completed",
        ),
        "reference_run_id": reference_id,
        "selection": selection,
        "comparison": comparison,
        "action_id": action["id"],
    }, action)


def _corrective_section(run: Mapping[str, Any], registry: object) -> tuple[dict, list[dict]]:
    actions = registry.get("actions") if isinstance(registry, Mapping) else None
    if not isinstance(actions, list):
        return ({
            "state": "unavailable",
            "reason": "corrective action registry is unavailable",
        }, [])
    projected = []
    descriptors = []
    run_id = str(run.get("id") or "")
    for item in actions:
        if not isinstance(item, Mapping) or not isinstance(item.get("id"), str):
            continue
        action_id = str(item["id"])
        projected.append({
            key: deepcopy(item[key])
            for key in (
                "id", "label", "description", "backends", "scope_eligibility", "limitations"
            )
            if key in item
        })
        descriptors.append(_action(
            f"corrective:{action_id}",
            str(item.get("label") or action_id),
            "corrective",
            "POST",
            f"/runs/{run_id}/corrective-actions/preview",
            "ready",
            request_body={"action_id": action_id, "requested_backend": "prompt_policy"},
        ))
    return ({
        "state": "supported" if projected else "unavailable",
        "artifact": _artifact(
            str(registry.get("schema") or "clozn.action-registry.v1")
            if isinstance(registry, Mapping) else "clozn.action-registry.v1",
            method="deterministic_action_registry",
            native_status="available" if projected else "empty",
        ),
        "actions": projected,
        **({} if projected else {"reason": "no corrective actions apply to this run"}),
    }, descriptors)


def build(
    run: Mapping[str, Any],
    *,
    related_runs: Sequence[Mapping[str, Any]] = (),
    corrective_registry: Mapping[str, Any] | None = None,
    scoring_available: bool = False,
) -> dict:
    """Compose existing evidence for one run without executing or persisting anything."""
    if not isinstance(run, Mapping) or not isinstance(run.get("id"), str) or not run.get("id"):
        raise ValueError("investigation requires a stored run with a non-empty id")

    records = [dict(item) for item in related_runs if isinstance(item, Mapping)]
    context = _context_section(run)
    influence, influence_action = _influence_section(
        run,
        scoring_available=scoring_available,
        context_available=context.get("state") in {
            "supported", "delivered_not_measured", "omitted"
        },
    )

    from clozn.runs.diagnosis import diagnose
    diagnosis_doc = diagnose(dict(run), related_runs=records)
    diagnosis = {
        "state": "supported",
        "artifact": _artifact(
            str(diagnosis_doc.get("schema") or "clozn.run_diagnosis.v1"),
            method="deterministic_recorded_evidence_rules",
            native_status="completed",
        ),
        "diagnosis": diagnosis_doc,
    }

    comparisons, comparison_action = _comparison_section(run, records)
    corrective, corrective_actions = _corrective_section(run, corrective_registry)
    span_addresses, span_action = _span_address_section(run)
    run_id = str(run["id"])
    support_action = _action(
        "check_recorded_influence_entailment",
        "Check recorded-influence support",
        "measurement",
        "POST",
        f"/runs/{run_id}/trust_spans",
        "conditional",
        request_body={"support": True},
        reason=(
            "the optional NLI matcher may be unavailable; lack of supplied-source support is not a "
            "factual verdict"
        ),
    )
    answer_support = {
        "state": "unavailable",
        "reason": "no claim-level supplied-source verification artifact is stored for this run",
        "action_id": support_action["id"],
    }

    actions = [
        *([influence_action] if influence_action is not None else []),
        span_action,
        support_action,
        comparison_action,
        *corrective_actions,
    ]
    unavailable = []
    for measurement_id, section in (
        ("prompt_source_influence", influence),
        ("answer_support", answer_support),
        ("run_comparison", comparisons),
    ):
        state = section.get("state")
        if state in {"unavailable", "failed", "inconclusive", "delivered_not_measured"}:
            unavailable.append({
                "id": measurement_id,
                "state": state,
                "reason": str(section.get("reason") or "measurement is not present"),
                **({"action_id": section["action_id"]} if section.get("action_id") else {}),
            })
    if context.get("state") in {"unavailable", "failed"}:
        unavailable.insert(0, {
            "id": "received_context",
            "state": context["state"],
            "reason": str(context.get("reason") or "context receipt is not present"),
        })

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "sections": {
            "received_context": context,
            "prompt_source_influence": influence,
            "answer_support": answer_support,
            "text_span_addresses": span_addresses,
            "diagnosis": diagnosis,
            "comparisons": comparisons,
            "corrective_actions": corrective,
        },
        "actions": actions,
        "unavailable_measurements": unavailable,
    }
    return document
