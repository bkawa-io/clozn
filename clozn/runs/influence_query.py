""""Why this?" -- a derived, read-only influence query over one run's persisted context<->answer
influence measurement (E7, the debugger's read-side counterpart to E2's claim support).

Given a caller-selected half-open Unicode-code-point range of the RECORDED answer, this module answers
exactly one question: which measured context spans (`run["influence_map"]["links"]`) affected that
range's scoring? It composes evidence Clozn already has -- it introduces no new attribution algorithm,
starts no measurement, and never touches a model, worker, or the network (see the module's own import
list: only `clozn.runs.influence_geometry`, transitively `clozn.runs.text_span_addresses`, and
`clozn.schemas`).

THIS IS NOT `clozn.runs.claim_support`
---------------------------------------
`claim_support` asks "is a factual claim supported by supplied material?" and answers with a closed
verification vocabulary (`supported`/`contradicted`/...). This module asks a narrower, more literal
question: "what did the ALREADY-RECORDED measurement say about this exact answer range?" It never
re-derives `effect`, `evidence_state`, or `clears_floor` -- those are read verbatim off the persisted
`Link` objects (`clozn.context_answer_influence.v1`) and returned unchanged. It never performs textual
overlap or contradiction heuristics, and it is not a correctness or relevance judgment. A link with
`evidence_state == "observed"` and `clears_floor == False` was measured, not irrelevant -- returning it
(when it fits within `limit`) is deliberate, not an oversight; see clozn.context_answer_influence.v1's
own caveat, which this projection never weakens or omits.

THE FOUR MEASUREMENT STATES
----------------------------
`measurement.state` is the single discriminant a caller must branch on, and the four values are
deliberately never collapsed into each other:

    not_measured   no influence map was ever attached to this run (or it is not a value this module
                   can trust at all -- wrong schema name, or an artifact that fails its own schema).
                   NEVER "unsupported" and never an empty links list standing in for "nothing mattered".
    unavailable    a persisted `clozn.context_answer_influence.v1` artifact exists, but it (or the
                   recorded answer text needed to resolve this exact range) cannot honestly be used --
                   `status == "unavailable"`, redacted answer text, or a stale/drifted answer hash.
    error          the persisted artifact recorded `status == "error"` -- an intervention that should
                   have completed did not.
    available      the measurement was consulted and reconciled with this run's current answer text.
                   `links` may still be `[]` here -- that means the selected range measurably has no
                   link clearing the configured floor, which is NOT the same claim as `not_measured`.

Every non-`available` state always carries `reason`, taken from the persisted artifact's own
machine-readable error code when one exists (never a generic placeholder invented by this module).

ANSWER IDENTITY, NEVER TRUSTED BY OFFSET ALONE
------------------------------------------------
An integer offset is not proof the influence map still describes the CURRENT recorded answer. Every
answer-span address this module reads is required to reconcile its own `resolution.canonical.
basis_sha256` against `sha256(run["response"])` (via `clozn.runs.influence_geometry.resolve_geometry`,
built on `clozn.runs.text_span_addresses.project_influence_addresses`, the SAME projection
`claim_support` uses) before its offsets are trusted for overlap. A stale influence map computed for a
different (e.g. regenerated) answer never contributes a single link here -- see
`test_stale_influence_map_on_drifted_answer_is_unavailable` in this module's test suite. Redacted answer
text is treated identically: this module never reconstructs deleted text from a persisted artifact
merely because it happens to still carry a duplicated literal.

`clozn.runs.context_tension` (E8, "did competing context spans pull the same answer region in opposite
measured directions?") is built on the SAME `influence_geometry` gate/geometry primitives this module
uses -- neither module re-implements influence-map validation or answer-drift detection a second time.

DETERMINISM
-----------
Pure function of `(run, output_start, output_end, limit, privacy)`: no randomness, no wall-clock, no
model or engine access. `run` is never mutated. Links are sorted causally_supported-before-observed,
then by descending `abs_delta_nats`, then by stable native indices, then by stable address IDs, so the
same input always produces byte-identical output (proven in tests/test_influence_query.py).
"""
from __future__ import annotations

from clozn.runs import influence_geometry as geometry

SCHEMA_VERSION = "clozn.influence-query.v1"

DEFAULT_LIMIT = 12
MIN_LIMIT = 1
MAX_LIMIT = 50

# Re-exported for backward compatibility with callers that imported these names from this module before
# the shared clozn.runs.influence_geometry extraction -- the values themselves live there now.
STATES = geometry.STATES


def _sort_key(entry: dict) -> tuple:
    native = entry["native"]
    causally_first = 0 if entry["evidence_state"] == "causally_supported" else 1
    context_index = native.get("context_index") if isinstance(native.get("context_index"), int) else 0
    answer_index = native.get("answer_index") if isinstance(native.get("answer_index"), int) else 0
    return (
        causally_first,
        -float(entry["abs_delta_nats"]),
        context_index,
        answer_index,
        entry["source_span_id"],
        entry["answer_span_id"],
    )


def _summary(overlapping_count: int, links: list[dict], returned: list[dict]) -> dict:
    return {
        "selected_answer_spans": overlapping_count,
        "measured_links": len(links),
        "returned_links": len(returned),
        "causally_supported_links": sum(1 for link in links if link["evidence_state"] == "causally_supported"),
        "observed_links": sum(1 for link in links if link["evidence_state"] == "observed"),
        "supporting_links": sum(1 for link in links if link["effect"] == "supports"),
        "suppressing_links": sum(1 for link in links if link["effect"] == "suppresses"),
        "neutral_links": sum(1 for link in links if link["effect"] == "neutral"),
    }


def _empty_result(run_id: str, *, output_start: int, output_end: int, state: str, reason: str,
                  basis_sha256: str | None) -> dict:
    target = {
        "basis": "recorded_answer", "unit": "unicode_code_points", "interval": "half_open",
        "start": output_start, "end": output_end,
    }
    if basis_sha256 is not None:
        target["basis_sha256"] = basis_sha256
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "target": target,
        "measurement": {"state": state, "reason": reason},
        "links": [],
        "summary": _summary(0, [], []),
    }


def build_influence_query(run: dict, *, output_start: int, output_end: int,
                          limit: int = DEFAULT_LIMIT, privacy: str = "metadata_only") -> dict:
    """Build and validate one derived `clozn.influence-query.v1` document.

    Pure function of its arguments: reads only `run` (never mutated, never written back) and the
    supplied range/limit/privacy. Never imports an engine client, worker, or model routing module --
    `build_influence_query` works identically whether or not Clozn has an active worker attached, and a
    test that monkeypatches any engine/generation seam to raise must still pass (see
    tests/test_influence_query.py's `test_no_engine_or_worker_access`).

    Raises ValueError for structurally invalid arguments -- a non-empty run id, `output_start`/
    `output_end` as non-boolean integers with `0 <= output_start < output_end`, `output_end` within the
    recorded answer's length (when that length is knowable), and `limit` as an integer in
    [MIN_LIMIT, MAX_LIMIT]. Callers (the HTTP route) should turn a ValueError into an HTTP 400. `privacy`
    other than "metadata_only" also raises: this first version never embeds prompt/answer text, so
    "full" is refused rather than silently downgraded.
    """
    run = run if isinstance(run, dict) else {}
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")
    if privacy != "metadata_only":
        raise ValueError("privacy must be metadata_only (full is not supported by this query yet)")
    if (not isinstance(output_start, int) or isinstance(output_start, bool)
            or not isinstance(output_end, int) or isinstance(output_end, bool)):
        raise ValueError("output_start and output_end must be integers")
    if output_start < 0:
        raise ValueError("output_start must be >= 0")
    if output_end <= output_start:
        raise ValueError("output_end must be greater than output_start")
    if not isinstance(limit, int) or isinstance(limit, bool):
        raise ValueError("limit must be an integer")
    if not (MIN_LIMIT <= limit <= MAX_LIMIT):
        raise ValueError(f"limit must be between {MIN_LIMIT} and {MAX_LIMIT}")

    response, unavailable_reason = geometry.resolve_answer_text(run)
    if response is not None and output_end > len(response):
        raise ValueError("output_end is beyond the recorded answer's length")

    if response is None:
        return _empty_result(
            run_id, output_start=output_start, output_end=output_end,
            state="unavailable", reason=unavailable_reason, basis_sha256=None,
        )

    basis_sha256 = geometry.text_sha256(response)
    influence_map = run.get("influence_map")
    gate_state, gate_reason = geometry.gate(influence_map)
    if gate_state is not None:
        return _empty_result(
            run_id, output_start=output_start, output_end=output_end,
            state=gate_state, reason=gate_reason, basis_sha256=basis_sha256,
        )

    geo, geo_reason = geometry.resolve_geometry(run_id, influence_map, response)
    if geo is None:
        return _empty_result(
            run_id, output_start=output_start, output_end=output_end,
            state="unavailable", reason=geo_reason, basis_sha256=basis_sha256,
        )

    overlapping_ids = geometry.overlapping_answer_ids(geo, output_start, output_end)

    all_links: list[dict] = []
    for native_id in overlapping_ids:
        answer_span_id = geo.answer_address_by_id.get(native_id)
        if answer_span_id is None:
            continue
        for link in geo.links_by_answer_id.get(native_id, ()):
            context_id = link.get("context_span_id")
            source_span_id = (
                geo.prompt_address_by_id.get(context_id) if isinstance(context_id, str) else None
            )
            if source_span_id is None:
                # Defensive only: every context_span_id a real influence map's links carry is expected
                # to also appear in its own prompt_spans list, which project_influence_addresses always
                # projects to a real address_id. A link that fails to resolve here points at a malformed
                # or inconsistent artifact -- it is dropped rather than cited with a fabricated address.
                continue
            all_links.append({
                "source_span_id": source_span_id,
                "answer_span_id": answer_span_id,
                "effect": link.get("effect"),
                "delta_nats": link.get("delta_nats"),
                "abs_delta_nats": link.get("abs_delta_nats"),
                "clears_floor": link.get("clears_floor"),
                "evidence_state": link.get("evidence_state"),
                "native": {
                    "context_span_id": context_id,
                    "answer_span_id": native_id,
                    "context_index": link.get("context_index"),
                    "answer_index": link.get("answer_index"),
                },
            })

    all_links.sort(key=_sort_key)
    returned_links = all_links[:limit]

    target = {
        "basis": "recorded_answer", "unit": "unicode_code_points", "interval": "half_open",
        "start": output_start, "end": output_end, "basis_sha256": basis_sha256,
        "answer_span_ids": [geo.answer_address_by_id[native_id] for native_id in overlapping_ids],
    }
    measurement = geometry.available_measurement(influence_map)

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "target": target,
        "measurement": measurement,
        "links": returned_links,
        "summary": _summary(len(overlapping_ids), all_links, returned_links),
    }
    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "DEFAULT_LIMIT",
    "MAX_LIMIT",
    "MIN_LIMIT",
    "SCHEMA_VERSION",
    "STATES",
    "build_influence_query",
]
