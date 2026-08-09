"""Context tension -- a derived, read-only detector over one run's persisted context<->answer influence
measurement (E8, built directly on Feature 1/E7's shared `clozn.runs.influence_geometry` primitives).

THE QUESTION THIS ANSWERS, AND THE ONE IT DELIBERATELY DOES NOT
-------------------------------------------------------------------
This module answers: "did different pieces of context exert opposing MEASURED effects on the same part
of this answer?" It does NOT answer, and must never be read as answering: "do these two sources
semantically contradict each other?" That second question needs textual/semantic comparison (see
`clozn.runs.claim_support`'s own conservative numeric/negation heuristics, deliberately NOT reused or
copied here); this module never reads source or answer TEXT at all, only already-recorded intervention
measurements. Two context spans can be in "tension" here while being perfectly compatible prose (e.g. one
states a fact plainly and the other hedges it), and two spans can textually disagree without ever
producing measured tension (if the weaker one never clears the measurement floor). The vocabulary is
deliberately narrow:

    context_tension     -- NOT "conflict" or "contradiction": those imply the SOURCE TEXTS disagree.
    opposing_effects     -- NOT "wrong_source"/"correct_source": this module never judges correctness.
    supports/suppresses  -- read verbatim off the source Link objects, never re-derived.

A future feature could combine this measured signal with claim_support's explicit textual-contradiction
evidence for a stronger diagnosis. That composition is out of scope here on purpose -- see the module's
own test suite for the boundary this keeps.

THE CORE RULE (and nothing more)
-----------------------------------
A tension record exists for one answer span exactly when TWO DISTINCT source spans both have a link to
that SAME answer span, BOTH links have `evidence_state == "causally_supported"`, and their `effect`
values are opposite (`"supports"` vs `"suppresses"`). Every other combination -- same-direction links,
any `"neutral"` link, any link whose `evidence_state == "observed"` (measured, but never cleared the
configured floor), or two links that both resolve to the same stable source span -- never produces a
tension record. This keeps the signal intentionally high-precision (spec's own words): a caller reading
`tensions: []` learns only that no HIGH-CONFIDENCE opposing pair was found, never that the context was
"consistent" -- absence of detected tension is not proof of consistency.

BUILT ON THE SAME GATE/GEOMETRY AS "WHY THIS?" (`clozn.runs.influence_query`)
--------------------------------------------------------------------------------
Both features share `clozn.runs.influence_geometry` for influence-map validation, answer-drift/
redaction detection, and stable span-address resolution -- there is exactly one implementation of each,
never two subtly different ones. This module adds only what is genuinely new: partitioning links by
`effect`, generating opposing pairs, and a deterministic `tension_id`.

WHOLE-ANSWER VS RANGED MODE
-------------------------------
`output_start`/`output_end` are optional but must be supplied together (`None`/`None` means "scan every
resolvable answer span"; any other combination raises `ValueError`, never silently interpreted as
"from/to the start"). Range semantics when both are supplied are byte-identical to Influence Query's own
half-open `[output_start, output_end)` Unicode-code-point overlap rule.

NO AGGREGATE SCORE
----------------------
There is no `conflict_score`, `tension_percentage`, or `severity`. The raw `delta_nats`/`abs_delta_nats`
values are returned unchanged; `_rank_strength` below is sorting logic only, never surfaced in the public
artifact.

DETERMINISM
-----------
Pure function of `(run, output_start, output_end, limit, privacy)`: no randomness, no wall-clock, no
model or engine access. `run` is never mutated. `tension_id` is a stable digest of
`(run_id, answer_span_id, supporting_source_span_id, suppressing_source_span_id)` -- the same evidence
always yields the same ID (proven in tests/test_context_tension.py).
"""
from __future__ import annotations

import hashlib
import itertools
import json

from clozn.runs import influence_geometry as geometry

SCHEMA_VERSION = "clozn.context-tension.v1"

DEFAULT_LIMIT = 25
MIN_LIMIT = 1
MAX_LIMIT = 100

STATES = geometry.STATES


def _tension_id(run_id: str, answer_span_id: str, supporting_source_span_id: str,
                suppressing_source_span_id: str) -> str:
    """A stable `tension_<24 hex>` identity, the same shape/derivation technique
    `clozn.runs.text_span_addresses._address_id` uses for `span_...` IDs: canonical (sorted-key, compact,
    ensure_ascii=False) JSON of the identity fields, SHA-256, truncated to 24 hex characters. The exact
    same four inputs always produce the exact same ID -- never a random UUID."""
    identity = {
        "run_id": run_id,
        "answer_span_id": answer_span_id,
        "supporting_source_span_id": supporting_source_span_id,
        "suppressing_source_span_id": suppressing_source_span_id,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "tension_" + hashlib.sha256(encoded).hexdigest()[:24]


def _side(link: dict, source_span_id: str) -> dict:
    return {
        "source_span_id": source_span_id,
        "delta_nats": link.get("delta_nats"),
        "abs_delta_nats": link.get("abs_delta_nats"),
        "effect": link.get("effect"),
        "evidence_state": link.get("evidence_state"),
    }


def _tensions_for_answer_span(run_id: str, geo: "geometry.Geometry", native_answer_id: str,
                              answer_span_id: str) -> list[dict]:
    """Every opposing (supporting, suppressing) pair for ONE answer span -- the full Cartesian product
    of its above-floor supporting links against its above-floor suppressing links, excluding any pair
    whose two sides resolve to the SAME stable source span. Only `evidence_state == "causally_supported"`
    links participate; `"neutral"` and `"observed"` links are excluded before partitioning, not after."""
    supporting: list[tuple[dict, str]] = []
    suppressing: list[tuple[dict, str]] = []
    for link in geo.links_by_answer_id.get(native_answer_id, ()):
        if link.get("evidence_state") != "causally_supported":
            continue
        context_id = link.get("context_span_id")
        source_span_id = geo.prompt_address_by_id.get(context_id) if isinstance(context_id, str) else None
        if source_span_id is None:
            # Defensive only, mirrors influence_query's identical guard: every context_span_id a real
            # artifact's links carry is expected to also appear in prompt_spans, which
            # project_influence_addresses always projects to a real address. A link that fails to
            # resolve here points at a malformed/inconsistent artifact and is dropped, never cited with
            # a fabricated address.
            continue
        effect = link.get("effect")
        if effect == "supports":
            supporting.append((link, source_span_id))
        elif effect == "suppresses":
            suppressing.append((link, source_span_id))
        # "neutral" (or any other value) never participates in tension construction.

    tensions = []
    for (support_link, support_source), (suppress_link, suppress_source) in itertools.product(
        supporting, suppressing,
    ):
        if support_source == suppress_source:
            # Excluded by the spec: a pair where both links resolve to the same source span is not a
            # tension between two DISTINCT context spans.
            continue
        tension_id = _tension_id(run_id, answer_span_id, support_source, suppress_source)
        tensions.append({
            "tension_id": tension_id,
            "answer_span_id": answer_span_id,
            "supporting": _side(support_link, support_source),
            "suppressing": _side(suppress_link, suppress_source),
            "native": {
                "answer_span_id": native_answer_id,
                "supporting_context_span_id": support_link.get("context_span_id"),
                "suppressing_context_span_id": suppress_link.get("context_span_id"),
                "supporting_context_index": support_link.get("context_index"),
                "suppressing_context_index": suppress_link.get("context_index"),
                "answer_index": support_link.get("answer_index"),
            },
        })
    return tensions


def _rank_strength(tension: dict) -> tuple:
    """Sorting convenience ONLY -- never a "tension score" exposed in the public artifact. Ranks the
    pair whose WEAKER side is strongest first (a 4.0/3.5 genuine push-pull ranks above a 10.0/0.2 where
    one side barely registers), then by combined magnitude, then by the pair's own stable `tension_id`
    as a final deterministic tie-break."""
    supporting_abs = tension["supporting"]["abs_delta_nats"] or 0.0
    suppressing_abs = tension["suppressing"]["abs_delta_nats"] or 0.0
    weaker = min(supporting_abs, suppressing_abs)
    combined = supporting_abs + suppressing_abs
    return (-weaker, -combined, tension["tension_id"])


def _summary(answer_spans_examined: int, all_tensions: list[dict], returned: list[dict]) -> dict:
    with_tension = {t["answer_span_id"] for t in all_tensions}
    distinct_sources = set()
    for t in all_tensions:
        distinct_sources.add(t["supporting"]["source_span_id"])
        distinct_sources.add(t["suppressing"]["source_span_id"])
    return {
        "answer_spans_examined": answer_spans_examined,
        "answer_spans_with_tension": len(with_tension),
        "tension_pairs": len(all_tensions),
        "returned_tension_pairs": len(returned),
        "distinct_source_spans": len(distinct_sources),
    }


def _empty_result(run_id: str, *, scope: str, output_start: int | None, output_end: int | None,
                  state: str, reason: str, basis_sha256: str | None) -> dict:
    target = {"scope": scope, "basis": "recorded_answer", "unit": "unicode_code_points",
             "interval": "half_open"}
    if output_start is not None:
        target["start"] = output_start
        target["end"] = output_end
    if basis_sha256 is not None:
        target["basis_sha256"] = basis_sha256
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "target": target,
        "measurement": {"state": state, "reason": reason},
        "tensions": [],
        "summary": _summary(0, [], []),
    }


def build_context_tension(run: dict, *, output_start: int | None = None, output_end: int | None = None,
                          limit: int = DEFAULT_LIMIT, privacy: str = "metadata_only") -> dict:
    """Build and validate one derived `clozn.context-tension.v1` document.

    Pure function of its arguments: reads only `run` (never mutated, never written back). Never imports
    an engine client, worker, or model routing module -- works identically with no active worker
    attached (see tests/test_context_tension.py's `test_no_engine_or_worker_access`).

    `output_start`/`output_end` must be supplied together or both omitted (`None` means "scan every
    resolvable answer span" -- whole-answer mode); supplying exactly one raises `ValueError`. When both
    are supplied, the same half-open Unicode-code-point range rules as `clozn.runs.influence_query.
    build_influence_query` apply, including the same `output_end`-within-recorded-answer-length check.
    `limit` must be an integer in [MIN_LIMIT, MAX_LIMIT]. `privacy` other than "metadata_only" raises:
    this query never embeds prompt/answer text, so "full" is refused rather than silently downgraded.
    """
    run = run if isinstance(run, dict) else {}
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")
    if privacy != "metadata_only":
        raise ValueError("privacy must be metadata_only (full is not supported by this query yet)")
    if (output_start is None) != (output_end is None):
        raise ValueError("output_start and output_end must both be provided or both omitted")
    ranged = output_start is not None
    if ranged:
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

    scope = "answer_range" if ranged else "whole_answer"
    response, unavailable_reason = geometry.resolve_answer_text(run)
    if response is not None and ranged and output_end > len(response):
        raise ValueError("output_end is beyond the recorded answer's length")

    if response is None:
        return _empty_result(
            run_id, scope=scope, output_start=output_start, output_end=output_end,
            state="unavailable", reason=unavailable_reason, basis_sha256=None,
        )

    basis_sha256 = geometry.text_sha256(response)
    influence_map = run.get("influence_map")
    gate_state, gate_reason = geometry.gate(influence_map)
    if gate_state is not None:
        return _empty_result(
            run_id, scope=scope, output_start=output_start, output_end=output_end,
            state=gate_state, reason=gate_reason, basis_sha256=basis_sha256,
        )

    geo, geo_reason = geometry.resolve_geometry(run_id, influence_map, response)
    if geo is None:
        return _empty_result(
            run_id, scope=scope, output_start=output_start, output_end=output_end,
            state="unavailable", reason=geo_reason, basis_sha256=basis_sha256,
        )

    target_ids = geometry.overlapping_answer_ids(geo, output_start, output_end)

    all_tensions: list[dict] = []
    for native_id in target_ids:
        answer_span_id = geo.answer_address_by_id.get(native_id)
        if answer_span_id is None:
            continue
        all_tensions.extend(_tensions_for_answer_span(run_id, geo, native_id, answer_span_id))

    all_tensions.sort(key=_rank_strength)
    returned = all_tensions[:limit]

    target = {"scope": scope, "basis": "recorded_answer", "unit": "unicode_code_points",
             "interval": "half_open", "basis_sha256": basis_sha256}
    if ranged:
        target["start"] = output_start
        target["end"] = output_end

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "target": target,
        "measurement": geometry.available_measurement(influence_map),
        "tensions": returned,
        "summary": _summary(len(target_ids), all_tensions, returned),
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
    "build_context_tension",
]
