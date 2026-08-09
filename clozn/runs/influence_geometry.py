"""Shared, private geometry/gating primitives for Clozn's read-only, influence-derived debugger
queries: `clozn.runs.influence_query` ("Why this?", E7), `clozn.runs.context_tension` ("context
tension", E8), and `clozn.runs.context_utilization` ("what mattered?", E9). NOT a public product surface
by itself -- no schema, no route, no `schema_version` of its own. It exists so every feature built on top
of `run["influence_map"]` validates the artifact, detects answer drift, resolves redaction, and projects
stable span addresses through exactly ONE code path. Two subtly different implementations of
influence-map validation is exactly the failure mode this module exists to prevent.

Reuses `clozn.runs.text_span_addresses.project_influence_addresses` -- the SAME projection
`clozn.runs.claim_support` uses -- so answer-span/context-span addressing and drift/redaction resolution
state are never recomputed by a second addressing scheme.

Model-free, network-free, worker-free: this module and everything it imports touches only already-
recorded run/artifact data. No consumer of this module needs, or should ever add, an import of an engine
client, worker, model routing module, or generation/scoring entry point.
"""
from __future__ import annotations

import hashlib

from clozn.runs.text_span_addresses import INFLUENCE_SCHEMA, project_influence_addresses

PROMPT_ADDRESS_KINDS = frozenset({
    "attached_source_span", "delivered_message", "rendered_prompt_segment",
})

# The complete set of `measurement.state` values every consumer of this module emits. Kept as a
# frozenset (rather than inline enums scattered through each feature's own function body) so a future
# state addition is a one-line diff each feature's own schema enum can be grepped against.
STATES = frozenset({"available", "not_measured", "unavailable", "error"})


def run_redaction_status(run: dict) -> str | None:
    # Mirrors clozn.runs.claims._run_redaction_status and clozn.runs.text_span_addresses._run_is_redacted
    # exactly -- deliberately re-implemented locally (three lines) rather than imported, matching this
    # codebase's own convention of one small redaction check per module, now shared by every influence-
    # derived query through THIS module instead of being re-copied a third and fourth time.
    redaction = run.get("redaction")
    status = redaction.get("status") if isinstance(redaction, dict) else None
    if status == "redacted" or "redacted" in (run.get("flags") or []):
        return "redacted"
    return status if isinstance(status, str) else None


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_answer_text(run: dict) -> tuple[str | None, str]:
    """`(response, reason)` -- `response` is `run["response"]` when it is a usable string, else `None`
    with `reason` naming why (`"answer_text_redacted"` when the persisted redaction lifecycle says so --
    authoritative even if a legacy/buggy record still carries leftover literal text in `response` itself
    -- else `"no_recorded_answer_text"`). `reason` is meaningless and should be ignored when `response`
    is not `None`."""
    response = run.get("response")
    response = response if isinstance(response, str) else None
    if run_redaction_status(run) == "redacted":
        return None, "answer_text_redacted"
    if response is None:
        return None, "no_recorded_answer_text"
    return response, ""


def _not_measured(reason: str) -> tuple[str, str]:
    return "not_measured", reason


def gate(influence_map) -> tuple[str | None, str | None]:
    """`(None, None)` when `influence_map` is a trustworthy, usable `clozn.context_answer_influence.v1`
    artifact; otherwise `(state, reason)` for one of the three non-`available` states. Mirrors
    `clozn.runs.claim_support._influence_gate`'s exact branch order (no influence map -> wrong schema
    name -> status == "unavailable" -> status != "ok"/not available -> fails its own schema), but
    surfaces the PERSISTED ARTIFACT's own `error.code` as `reason` rather than a fixed placeholder
    string, per Influence Query's own contract: "a stable machine-readable reason derived from the
    persisted artifact"."""
    if not isinstance(influence_map, dict) or not influence_map:
        return _not_measured("no_influence_map")
    if isinstance(influence_map.get("unavailable"), str):
        # clozn.runs.store.get_run's own shape for an unresolved blob-backed influence_map_ref.
        return _not_measured("no_influence_map")
    if influence_map.get("schema") != INFLUENCE_SCHEMA:
        return _not_measured("no_influence_map")

    error = influence_map.get("error")
    error_code = (
        error["code"] if isinstance(error, dict) and isinstance(error.get("code"), str) and error["code"]
        else None
    )
    status = influence_map.get("status")
    if status == "unavailable":
        return "unavailable", error_code or "influence_measurement_unavailable"
    if status != "ok" or influence_map.get("available") is not True:
        return "error", error_code or "influence_measurement_error"

    from clozn import schemas
    try:
        schemas.validate(influence_map, INFLUENCE_SCHEMA)
    except schemas.ValidationError:
        return _not_measured("invalid_influence_artifact")
    return None, None


class Geometry:
    """Everything link selection needs, once the influence map has passed `gate` and the recorded
    answer text is known. Built once per call by `resolve_geometry` below."""

    __slots__ = (
        "answer_offsets", "answer_address_by_id", "prompt_address_by_id",
        "links_by_answer_id", "links_by_context_id",
    )

    def __init__(self, answer_offsets, answer_address_by_id, prompt_address_by_id, links_by_answer_id,
                links_by_context_id):
        self.answer_offsets = answer_offsets
        self.answer_address_by_id = answer_address_by_id
        self.prompt_address_by_id = prompt_address_by_id
        self.links_by_answer_id = links_by_answer_id
        self.links_by_context_id = links_by_context_id


def resolve_geometry(run_id: str, influence_map: dict, response: str) -> tuple[Geometry | None, str | None]:
    """`(Geometry, None)` on success, else `(None, reason)` -- always an "unavailable" reason. Reuses
    `clozn.runs.text_span_addresses.project_influence_addresses` -- the SAME projection
    `clozn.runs.claim_support` uses -- so answer-span addresses and their drift/redaction resolution
    state are never recomputed by a second addressing scheme."""
    try:
        addresses = project_influence_addresses(run_id, influence_map, privacy="metadata_only")
    except (ValueError, TypeError):
        return None, "no_resolvable_answer_spans"

    response_sha256 = text_sha256(response)
    answer_offsets: dict[str, tuple[int, int]] = {}
    answer_address_by_id: dict[str, str] = {}
    prompt_address_by_id: dict[str, str] = {}
    saw_answer_span = False
    saw_hash_mismatch = False
    for address in addresses:
        native_id = address.get("native_ref", {}).get("id")
        if not isinstance(native_id, str):
            continue
        kind = address.get("kind")
        if kind == "answer_span":
            saw_answer_span = True
            answer_address_by_id[native_id] = address["address_id"]
            canonical = (address.get("resolution") or {}).get("canonical")
            if not isinstance(canonical, dict):
                continue
            if canonical.get("basis_sha256") != response_sha256:
                # The influence map's own scored answer text no longer matches this run's CURRENT
                # recorded response -- a stale map attached to a since-changed/regenerated answer.
                # Never trusted for overlap, regardless of what its offsets say.
                saw_hash_mismatch = True
                continue
            start, end = canonical.get("start"), canonical.get("end")
            if isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool):
                answer_offsets[native_id] = (start, end)
        elif kind in PROMPT_ADDRESS_KINDS:
            prompt_address_by_id[native_id] = address["address_id"]

    if not answer_offsets:
        reason = "answer_text_mismatch" if (saw_answer_span and saw_hash_mismatch) else (
            "no_resolvable_answer_spans"
        )
        return None, reason

    links_by_answer_id: dict[str, list[dict]] = {}
    links_by_context_id: dict[str, list[dict]] = {}
    for link in influence_map.get("links") or []:
        if not isinstance(link, dict):
            continue
        if isinstance(link.get("answer_span_id"), str):
            links_by_answer_id.setdefault(link["answer_span_id"], []).append(link)
        if isinstance(link.get("context_span_id"), str):
            links_by_context_id.setdefault(link["context_span_id"], []).append(link)

    return (
        Geometry(
            answer_offsets, answer_address_by_id, prompt_address_by_id,
            links_by_answer_id, links_by_context_id,
        ),
        None,
    )


def overlapping_answer_ids(geometry: Geometry, output_start: int | None, output_end: int | None) -> list[str]:
    """Native answer-span IDs overlapping half-open `[output_start, output_end)`, sorted by each span's
    own start offset (then native ID as a stable tie-break). `output_start`/`output_end` both `None`
    means "every resolvable answer span" (whole-answer mode) -- callers never pass exactly one of the
    two as `None`; that pairing is a caller-side input-validation error, not this function's concern."""
    if output_start is None and output_end is None:
        ids = list(geometry.answer_offsets.keys())
    else:
        ids = [
            native_id for native_id, (start, end) in geometry.answer_offsets.items()
            if start < output_end and end > output_start
        ]
    return sorted(ids, key=lambda native_id: (geometry.answer_offsets[native_id][0], native_id))


def available_measurement(influence_map: dict) -> dict:
    """The `measurement` object's `state == "available"` shape -- `method`/`thresholds`/
    `artifact_sha256` copied UNCHANGED from the persisted artifact, omitted (never null-padded) when the
    artifact does not carry them. Shared verbatim by every feature built on this module so the same
    persisted evidence is described identically regardless of which derived query is reading it."""
    from copy import deepcopy

    measurement = {"state": "available", "influence_schema": INFLUENCE_SCHEMA}
    method = influence_map.get("method")
    thresholds = influence_map.get("thresholds")
    artifact_sha256 = influence_map.get("artifact_sha256")
    if isinstance(method, dict):
        measurement["method"] = deepcopy(method)
    if isinstance(thresholds, dict):
        measurement["thresholds"] = deepcopy(thresholds)
    if isinstance(artifact_sha256, str) and artifact_sha256:
        measurement["artifact_sha256"] = artifact_sha256
    return measurement


__all__ = [
    "Geometry",
    "INFLUENCE_SCHEMA",
    "PROMPT_ADDRESS_KINDS",
    "STATES",
    "available_measurement",
    "gate",
    "overlapping_answer_ids",
    "resolve_answer_text",
    "resolve_geometry",
    "run_redaction_status",
    "text_sha256",
]
