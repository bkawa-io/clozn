"""Context utilization -- a derived, read-only coverage view over one run's persisted context<->answer
influence measurement (E9, built on the same shared `clozn.runs.influence_geometry` primitives as
`clozn.runs.influence_query` "Why this?", E7, and `clozn.runs.context_tension`, E8).

THE QUESTION THIS ANSWERS
--------------------------
Which parts of the recorded context showed a clear measured effect on this answer, which were measured
but stayed below the measurement floor, and which were never measured at all? This is named "context
utilization", NOT "dead context": the bounded influence measurement selects only a subset of prompt
sources for scoring (`selection.max_context_spans`, "earliest policy/system source, then most recent
context"), so a source this module reports as `not_measured` may have been the single most important
piece of context in the prompt -- Clozn simply never scored it. Calling that source "dead", "unused",
"irrelevant", "ignored", or "unnecessary" would turn an attribution BUDGET into a claim about model
BEHAVIOR, which is exactly the mistake this module exists to avoid making.

THREE STATES, NEVER COLLAPSED
--------------------------------
    clear_measured_effect   the source was selected for measurement, and at least one of its measured
                             COARSE spans produced an above-floor (`evidence_state ==
                             "causally_supported"`) link to some recorded answer token.
    below_measured_floor    the source was selected and its measurement is COMPLETE, but every one of
                             its measured coarse-span links stayed at `evidence_state == "observed"` --
                             the intervention ran, nothing hidden, it just never cleared the configured
                             floor. This means ONLY "no strong effect was detected under this
                             intervention and measurement threshold for this recorded answer" -- it is
                             NEVER a claim that the source was irrelevant, and this module never says so.
    not_measured             the source was present in the assembled prompt but the bounded selection
                             never scored it at all. No influence claim -- positive or negative -- is
                             available for it. This state must NEVER be described as low-effect.

COARSE SPANS ONLY -- WHY
--------------------------
An influence artifact's `prompt_spans` mixes `level == "coarse"` (every selected source gets these) with
`level == "fine"` (generated only for the handful of coarse spans whose OWN row already cleared the
floor, via coarse-to-fine refinement -- see `clozn.receipts.context_answer_influence`'s own docstring).
Classifying a source from its fine spans would structurally bias the comparison: a strongly-effective
source would accumulate MORE measured rows than a weak one, inflating its apparent coverage for reasons
that have nothing to do with the underlying evidence. This module reads only `level == "coarse"` spans
(matched to their source by the persisted `parent_id`, never by text/label/role/index proximity) when
deciding `clear_measured_effect` vs `below_measured_floor`. A refined source's fine-span count is
surfaced separately, as optional `refinement` metadata, and never feeds the classification.

COMPLETENESS IS A PRECONDITION FOR "BELOW FLOOR", NOT AN AFTERTHOUGHT
--------------------------------------------------------------------------
A `below_measured_floor` classification asserts "every measured cell for this source stayed below the
floor" -- that assertion is only honest when the measurement that produced those cells actually
finished. Before classifying ANY source, this module requires `influence_map["matrix_complete"] is True`
and `influence_map["selection"]["complete_for_selected_spans"] is True`; if either is not exactly `True`,
the WHOLE response degrades to `measurement.state == "unavailable"` with `reason ==
"incomplete_influence_matrix"` -- no source is ever classified from partial evidence, and a source that
happens to look weak in an incomplete matrix is never reported as below-floor.

A persisted artifact whose `selection`/`prompt_sources.selected` disagree with each other, or whose
selected source is missing its expected coarse spans or links, is not "close enough" -- this module fails
CLOSED (raises, which the HTTP route turns into a generic contract-failure 500) rather than inventing a
plausible-looking coverage answer from broken bookkeeping.

RELATIONSHIP TO THE OTHER TWO INFLUENCE FEATURES
----------------------------------------------------
Influence Query answers "which measured context spans affected THIS SELECTED OUTPUT REGION?" (per-range,
per-link detail). Context Tension answers "did two sources pull the SAME answer span in opposite
directions?" (per-answer-span opposing pairs). This module answers a different, source-level question
across the WHOLE answer, and never requires an output-range selection. It deliberately does not expose
Feature 1's detailed per-link records or Feature 2's opposing-pair records -- see each module's own
`__all__`/docstring for the boundary.

NO SCORE, NO PERCENTAGE
------------------------
There is no `utilization_percent`, `importance_percent`, or `importance_score`. `max_abs_delta_nats` is
returned as a descriptive, per-source measurement and a deterministic sort key -- never normalized or
compared across differently-sized spans as a universal importance scale (the persisted artifact's own
`method.caveat` already warns against exactly that).

DETERMINISM
-----------
Pure function of `(run, privacy)`: no randomness, no wall-clock, no model or engine access. `run` is
never mutated.
"""
from __future__ import annotations

from clozn.runs import influence_geometry as geometry

SCHEMA_VERSION = "clozn.context-utilization.v1"

# Deliberately narrower than clozn.runs.influence_geometry.STATES: this feature has no per-request
# range to fail on, and folds the underlying artifact's own status=="error" into "unavailable" (see
# _translate_gate_state below) -- the spec for this feature lists exactly these three states.
STATES = frozenset({"available", "not_measured", "unavailable"})


def _translate_gate_state(gate_state: str) -> str:
    # geometry.gate() can return "error" (the persisted artifact recorded status == "error"). Context
    # Utilization does not distinguish "the job failed" from "the job says unavailable" as top-level
    # states -- both mean "no usable measurement", surfaced only through `reason`.
    return "unavailable" if gate_state == "error" else gate_state


def _summary(sources: list[dict]) -> dict:
    measured = [s for s in sources if s["measurement_state"] == "measured"]
    return {
        "prompt_sources": len(sources),
        "measured_sources": len(measured),
        "sources_with_clear_measured_effect": sum(
            1 for s in measured if s["effect_state"] == "clear_measured_effect"
        ),
        "sources_below_measured_floor": sum(
            1 for s in measured if s["effect_state"] == "below_measured_floor"
        ),
        "sources_not_measured": sum(1 for s in sources if s["measurement_state"] == "not_measured"),
    }


def _empty_result(run_id: str, *, state: str, reason: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "measurement": {"state": state, "reason": reason},
        "sources": [],
        "summary": _summary([]),
    }


def _code_points(source: dict) -> int | None:
    """A plain integer code-point COUNT (`end - start`), never text -- `prompt_sources[].start/end` are
    already Unicode-code-point offsets (this whole subsystem's `OFFSET_CONTRACT`), present on the raw
    persisted artifact regardless of privacy since they are integers, not literals. `None` when either
    offset is missing or malformed -- never estimated, never derived by retokenizing."""
    start, end = source.get("start"), source.get("end")
    if (isinstance(start, int) and not isinstance(start, bool)
            and isinstance(end, int) and not isinstance(end, bool) and end >= start):
        return end - start
    return None


def _sort_key(item: tuple[int, dict]) -> tuple:
    index, entry = item
    if entry["measurement_state"] == "not_measured":
        return (2, index, entry["source_span_id"])
    magnitude = -(entry.get("max_abs_delta_nats") or 0.0)
    bucket = 0 if entry["effect_state"] == "clear_measured_effect" else 1
    return (bucket, magnitude, entry["source_span_id"])


def build_context_utilization(run: dict, *, privacy: str = "metadata_only") -> dict:
    """Build and validate one derived `clozn.context-utilization.v1` document.

    Pure function of its arguments: reads only `run` (never mutated, never written back). Never imports
    an engine client, worker, or model routing module -- works identically with no active worker
    attached (see tests/test_context_utilization.py's `test_no_engine_or_worker_access`).

    Raises `ValueError` for structurally invalid arguments (bad run id, `privacy` other than
    "metadata_only") AND for internally inconsistent persisted evidence that cannot be honestly
    classified -- a `prompt_sources[].selected` flag that disagrees with `selection.selected_source_ids`/
    `omitted_source_ids`, a selected source missing its expected coarse spans, or a selected source with
    coarse spans but no measured links. All of these are contract failures the HTTP route turns into a
    generic, text-free 500 -- never a best-effort coverage guess built on broken bookkeeping.
    """
    run = run if isinstance(run, dict) else {}
    run_id = str(run.get("id") or "")
    if not run_id:
        raise ValueError("run.id must be a non-empty string")
    if privacy != "metadata_only":
        raise ValueError("privacy must be metadata_only (full is not supported by this query yet)")

    response, unavailable_reason = geometry.resolve_answer_text(run)
    if response is None:
        return _empty_result(run_id, state="unavailable", reason=unavailable_reason)

    influence_map = run.get("influence_map")
    gate_state, gate_reason = geometry.gate(influence_map)
    if gate_state is not None:
        return _empty_result(run_id, state=_translate_gate_state(gate_state), reason=gate_reason)

    selection = influence_map.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    if (influence_map.get("matrix_complete") is not True
            or selection.get("complete_for_selected_spans") is not True):
        return _empty_result(run_id, state="unavailable", reason="incomplete_influence_matrix")

    geo, geo_reason = geometry.resolve_geometry(run_id, influence_map, response)
    if geo is None:
        return _empty_result(run_id, state="unavailable", reason=geo_reason)

    prompt_sources_raw = influence_map.get("prompt_sources")
    prompt_sources = (
        [item for item in prompt_sources_raw if isinstance(item, dict)]
        if isinstance(prompt_sources_raw, list) else []
    )
    prompt_spans_raw = influence_map.get("prompt_spans")
    prompt_spans = (
        [item for item in prompt_spans_raw if isinstance(item, dict)]
        if isinstance(prompt_spans_raw, list) else []
    )

    selected_ids = set(selection.get("selected_source_ids") or [])
    omitted_ids = set(selection.get("omitted_source_ids") or [])
    if selected_ids & omitted_ids:
        raise ValueError(
            "influence_map.selection lists a source as both selected and omitted"
        )

    coarse_by_parent: dict[str, list[dict]] = {}
    fine_count_by_coarse_id: dict[str, int] = {}
    for span in prompt_spans:
        level = span.get("level")
        parent_id = span.get("parent_id")
        if level == "coarse" and isinstance(parent_id, str):
            coarse_by_parent.setdefault(parent_id, []).append(span)
        elif level == "fine" and isinstance(parent_id, str):
            fine_count_by_coarse_id[parent_id] = fine_count_by_coarse_id.get(parent_id, 0) + 1

    sources_out: list[tuple[int, dict]] = []
    seen_source_ids: set[str] = set()
    for index, source in enumerate(prompt_sources):
        source_id = source.get("id")
        if not isinstance(source_id, str):
            continue
        seen_source_ids.add(source_id)

        flag = source.get("selected")
        in_selected = source_id in selected_ids
        in_omitted = source_id in omitted_ids
        if in_selected == in_omitted:
            # Neither, or both -- either way the persisted selection does not say exactly one thing
            # about this source.
            raise ValueError(
                f"influence_map.selection does not classify prompt source {source_id!r} as exactly "
                f"one of selected/omitted"
            )
        if flag is not in_selected:
            raise ValueError(
                f"prompt_sources[].selected disagrees with influence_map.selection for "
                f"source {source_id!r}"
            )

        source_span_id = geo.prompt_address_by_id.get(source_id)
        if source_span_id is None:
            raise ValueError(f"prompt source {source_id!r} has no resolvable stable address")

        code_points = _code_points(source)
        native = {"source_id": source_id}

        if not in_selected:
            entry = {
                "source_span_id": source_span_id,
                "measurement_state": "not_measured",
                "reason": "omitted_by_measurement_selection",
                "native": native,
            }
            if code_points is not None:
                entry["code_points"] = code_points
            sources_out.append((index, entry))
            continue

        coarse_spans = coarse_by_parent.get(source_id) or []
        if not coarse_spans:
            raise ValueError(f"selected prompt source {source_id!r} has no measured coarse spans")

        links_for_source: list[dict] = []
        for span in coarse_spans:
            span_id = span.get("id")
            if isinstance(span_id, str):
                links_for_source.extend(geo.links_by_context_id.get(span_id, ()))
        if not links_for_source:
            raise ValueError(
                f"selected prompt source {source_id!r} has coarse spans but no measured links"
            )

        clear_links = [link for link in links_for_source if link.get("evidence_state") == "causally_supported"]
        observed_links = [link for link in links_for_source if link.get("evidence_state") == "observed"]
        effect_state = "clear_measured_effect" if clear_links else "below_measured_floor"
        coarse_ids_with_clear = {link.get("context_span_id") for link in clear_links}
        max_abs_delta_nats = max(
            (link.get("abs_delta_nats") or 0.0) for link in links_for_source
        )

        entry = {
            "source_span_id": source_span_id,
            "measurement_state": "measured",
            "effect_state": effect_state,
            "coarse_span_count": len(coarse_spans),
            "coarse_spans_with_clear_effect": len(coarse_ids_with_clear),
            "clear_link_count": len(clear_links),
            "observed_link_count": len(observed_links),
            "supporting_clear_links": sum(1 for link in clear_links if link.get("effect") == "supports"),
            "suppressing_clear_links": sum(1 for link in clear_links if link.get("effect") == "suppresses"),
            "neutral_clear_links": sum(1 for link in clear_links if link.get("effect") == "neutral"),
            "max_abs_delta_nats": max_abs_delta_nats,
            "native": native,
        }
        if code_points is not None:
            entry["code_points"] = code_points
        fine_span_count = sum(
            fine_count_by_coarse_id.get(span.get("id"), 0) for span in coarse_spans
            if isinstance(span.get("id"), str)
        )
        if fine_span_count:
            entry["refinement"] = {"available": True, "fine_span_count": fine_span_count}
        sources_out.append((index, entry))

    if not (selected_ids <= seen_source_ids) or not (omitted_ids <= seen_source_ids):
        raise ValueError("influence_map.selection references a source absent from prompt_sources")

    sources_out.sort(key=_sort_key)
    final_sources = [entry for _index, entry in sources_out]

    measurement = geometry.available_measurement(influence_map)
    measurement["matrix_complete"] = True
    measurement["selection"] = {
        "strategy": selection.get("strategy"),
        "max_context_spans": selection.get("max_context_spans"),
        "selected_sources": len(selected_ids),
        "omitted_sources": len(omitted_ids),
    }

    document = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "privacy": "metadata_only",
        "measurement": measurement,
        "sources": final_sources,
        "summary": _summary(final_sources),
    }
    from clozn import schemas
    schemas.validate(document)
    return document


__all__ = [
    "SCHEMA_VERSION",
    "STATES",
    "build_context_utilization",
]
