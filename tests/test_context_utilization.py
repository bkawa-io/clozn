"""Tests for clozn.runs.context_utilization -- the "context utilization" coverage view (E9).

No model, no network, no filesystem outside `tmp_path`. `build_context_utilization` is a pure function
of `(run, privacy)`; it is proven here to work with zero access to any engine/model/worker seam,
matching Influence Query's and Context Tension's own test suite discipline for the identical reason.
"""
from __future__ import annotations

import copy

import pytest

from clozn import schemas
from clozn.runs import context_utilization


def _method() -> dict:
    return {
        "name": "teacher_forced_matched_context_replacement", "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim", "caveat": "measured effect only, not correctness",
    }


_ANSWER = "Paris is the capital of France."


def _source(source_id: str, *, start: int, end: int, selected: bool) -> dict:
    return {"id": source_id, "start": start, "end": end, "text": "x" * (end - start), "selected": selected}


def _coarse(span_id: str, parent_id: str, *, start: int = 0, end: int = 10) -> dict:
    return {"id": span_id, "parent_id": parent_id, "level": "coarse", "start": start, "end": end,
           "text": "x" * (end - start)}


def _fine(span_id: str, parent_id: str, *, start: int = 0, end: int = 5) -> dict:
    return {"id": span_id, "parent_id": parent_id, "level": "fine", "start": start, "end": end,
           "text": "x" * (end - start)}


def _link(context_span_id, answer_span_id="as-0", *, delta_nats, effect, clears_floor, evidence_state,
         context_index=0, answer_index=0) -> dict:
    return {
        "context_span_id": context_span_id, "answer_span_id": answer_span_id,
        "context_index": context_index, "answer_index": answer_index,
        "delta_nats": delta_nats, "abs_delta_nats": abs(delta_nats),
        "effect": effect, "clears_floor": clears_floor, "evidence_state": evidence_state,
    }


def _answer_spans() -> list[dict]:
    return [{"id": "as-0", "start": 0, "end": len(_ANSWER), "text": _ANSWER}]


def _influence(*, status="ok", available=True, error=None, prompt_sources=(), prompt_spans=(),
              links=(), matrix_complete=True, complete_for_selected_spans=True,
              strategy="earliest_policy_then_recent_sources_proportional_chunks_v1",
              max_context_spans=8, artifact_sha256="c" * 64) -> dict:
    doc = {
        "schema": "clozn.context_answer_influence.v1",
        "status": status,
        "available": available,
        "method": _method(),
        "identity": {"model_sha256": "a" * 64},
    }
    if status != "ok":
        doc["error"] = error or {"code": "no_text_context", "message": "no context available"}
        return doc
    selected_ids = [s["id"] for s in prompt_sources if s.get("selected")]
    omitted_ids = [s["id"] for s in prompt_sources if not s.get("selected")]
    doc.update({
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "artifact_sha256": artifact_sha256,
        "prompt_sources": list(prompt_sources),
        "prompt_spans": list(prompt_spans),
        "answer": {"scored_text": _ANSWER},
        "answer_spans": _answer_spans(),
        "links": list(links),
        "matrix_complete": matrix_complete,
        "selection": {
            "strategy": strategy, "max_context_spans": max_context_spans,
            "selected_source_ids": selected_ids, "omitted_source_ids": omitted_ids,
            "measured_span_count": len(prompt_spans),
            "complete_for_selected_spans": complete_for_selected_spans,
        },
    })
    return doc


def _run(run_id="run-x", response=_ANSWER, **over) -> dict:
    out = {"id": run_id, "response": response}
    out.update(over)
    return out


def _by_native_id(document: dict) -> dict[str, dict]:
    return {entry["native"]["source_id"]: entry for entry in document["sources"]}


# ======================================================================================================
# 1. Measured source with clear effect
# ======================================================================================================

def test_measured_source_with_clear_effect():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=20, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=20)],
        links=[_link("p.m000.c000", delta_nats=-1.8, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)

    entry = _by_native_id(doc)["p.m000"]
    assert entry["measurement_state"] == "measured"
    assert entry["effect_state"] == "clear_measured_effect"


# ======================================================================================================
# 2. Measured source entirely below floor
# ======================================================================================================

def test_measured_source_entirely_below_floor():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=15, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=15)],
        links=[_link("p.m000.c000", delta_nats=0.05, effect="supports", clears_floor=False,
                     evidence_state="observed")],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)

    entry = _by_native_id(doc)["p.m000"]
    assert entry["measurement_state"] == "measured"
    assert entry["effect_state"] == "below_measured_floor"


# ======================================================================================================
# 3. Omitted source
# ======================================================================================================

def test_omitted_source_is_not_measured_with_no_effect_state():
    influence = _influence(
        prompt_sources=[
            _source("p.m000", start=0, end=20, selected=True),
            _source("p.m001", start=0, end=30, selected=False),
        ],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=20)],
        links=[_link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)

    entry = _by_native_id(doc)["p.m001"]
    assert entry["measurement_state"] == "not_measured"
    assert entry["reason"] == "omitted_by_measurement_selection"
    assert "effect_state" not in entry


# ======================================================================================================
# 4. Mixed coverage
# ======================================================================================================

def test_mixed_coverage_three_distinct_classifications():
    influence = _influence(
        prompt_sources=[
            _source("p.m000", start=0, end=20, selected=True),   # clear
            _source("p.m001", start=0, end=15, selected=True),   # below floor
            _source("p.m002", start=0, end=30, selected=False),  # omitted
        ],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000", start=0, end=20),
            _coarse("p.m001.c000", "p.m001", start=0, end=15),
        ],
        links=[
            _link("p.m000.c000", delta_nats=-1.8, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m001.c000", delta_nats=0.05, effect="supports", clears_floor=False,
                 evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)

    by_id = _by_native_id(doc)
    assert by_id["p.m000"]["effect_state"] == "clear_measured_effect"
    assert by_id["p.m001"]["effect_state"] == "below_measured_floor"
    assert by_id["p.m002"]["measurement_state"] == "not_measured"
    assert doc["summary"] == {
        "prompt_sources": 3, "measured_sources": 2, "sources_with_clear_measured_effect": 1,
        "sources_below_measured_floor": 1, "sources_not_measured": 1,
    }


# ======================================================================================================
# 5. Multiple coarse spans for one source -- only the second has a clear link
# ======================================================================================================

def test_multiple_coarse_spans_one_clear_makes_the_source_clear():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=30, selected=True)],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000", start=0, end=10),
            _coarse("p.m000.c001", "p.m000", start=10, end=20),
            _coarse("p.m000.c002", "p.m000", start=20, end=30),
        ],
        links=[
            _link("p.m000.c000", delta_nats=0.02, effect="supports", clears_floor=False,
                 evidence_state="observed"),
            _link("p.m000.c001", delta_nats=-2.5, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m000.c002", delta_nats=0.01, effect="neutral", clears_floor=False,
                 evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)

    entry = _by_native_id(doc)["p.m000"]
    assert entry["effect_state"] == "clear_measured_effect"
    assert entry["coarse_span_count"] == 3
    assert entry["coarse_spans_with_clear_effect"] == 1
    assert entry["clear_link_count"] == 1
    assert entry["observed_link_count"] == 2


# ======================================================================================================
# 6. Fine spans do not affect source classification
# ======================================================================================================

def test_fine_span_clear_link_never_upgrades_a_below_floor_source():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=20, selected=True)],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000", start=0, end=20),
            _fine("p.m000.c000.f000", "p.m000.c000", start=0, end=10),
        ],
        links=[
            # The coarse row itself never clears the floor...
            _link("p.m000.c000", delta_nats=0.02, effect="supports", clears_floor=False,
                 evidence_state="observed"),
            # ...but its fine descendant does. V1 classification must ignore this entirely.
            _link("p.m000.c000.f000", delta_nats=-3.0, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)

    entry = _by_native_id(doc)["p.m000"]
    assert entry["effect_state"] == "below_measured_floor"
    assert entry["clear_link_count"] == 0
    assert entry["coarse_span_count"] == 1


# ======================================================================================================
# 7. Refined strong source does not double-count
# ======================================================================================================

def test_refined_source_counts_only_the_coarse_row():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=20, selected=True)],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000", start=0, end=20),
            _fine("p.m000.c000.f000", "p.m000.c000", start=0, end=7),
            _fine("p.m000.c000.f001", "p.m000.c000", start=7, end=14),
            _fine("p.m000.c000.f002", "p.m000.c000", start=14, end=20),
        ],
        links=[
            _link("p.m000.c000", delta_nats=-2.0, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m000.c000.f000", delta_nats=-2.5, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m000.c000.f001", delta_nats=-0.3, effect="supports", clears_floor=False,
                 evidence_state="observed"),
            _link("p.m000.c000.f002", delta_nats=-1.9, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)

    entry = _by_native_id(doc)["p.m000"]
    assert entry["coarse_span_count"] == 1
    assert entry["clear_link_count"] == 1
    assert entry["observed_link_count"] == 0
    assert entry["refinement"] == {"available": True, "fine_span_count": 3}


# ======================================================================================================
# 8. Supporting and suppressing clear links
# ======================================================================================================

def test_supporting_and_suppressing_clear_links_counted_correctly():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=30, selected=True)],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000", start=0, end=15),
            _coarse("p.m000.c001", "p.m000", start=15, end=30),
        ],
        links=[
            _link("p.m000.c000", delta_nats=-2.0, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m000.c001", delta_nats=1.5, effect="suppresses", clears_floor=True,
                 evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)

    entry = _by_native_id(doc)["p.m000"]
    assert entry["effect_state"] == "clear_measured_effect"
    assert entry["supporting_clear_links"] == 1
    assert entry["suppressing_clear_links"] == 1
    assert entry["clear_link_count"] == 2


# ======================================================================================================
# 9. Complete measurement with zero clear sources
# ======================================================================================================

def test_complete_measurement_zero_clear_sources_is_available_not_not_measured():
    influence = _influence(
        prompt_sources=[
            _source("p.m000", start=0, end=15, selected=True),
            _source("p.m001", start=0, end=15, selected=True),
        ],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000", start=0, end=15),
            _coarse("p.m001.c000", "p.m001", start=0, end=15),
        ],
        links=[
            _link("p.m000.c000", delta_nats=0.01, effect="neutral", clears_floor=False,
                 evidence_state="observed"),
            _link("p.m001.c000", delta_nats=-0.02, effect="supports", clears_floor=False,
                 evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)

    assert doc["measurement"]["state"] == "available"
    assert doc["summary"]["sources_with_clear_measured_effect"] == 0
    assert doc["summary"]["sources_below_measured_floor"] == 2


# ======================================================================================================
# 10. No influence map
# ======================================================================================================

def test_no_influence_map_yields_not_measured_state():
    run = _run()
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "not_measured"
    assert doc["measurement"]["reason"] == "no_influence_map"
    assert doc["sources"] == []


# ======================================================================================================
# 11. Influence artifact unavailable
# ======================================================================================================

def test_influence_artifact_unavailable_is_typed():
    influence = _influence(status="unavailable", available=False,
                           error={"code": "no_text_context", "message": "x"})
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "no_text_context"
    assert doc["sources"] == []


# ======================================================================================================
# 12. Influence artifact error -- typed unavailable, no partial classification
# ======================================================================================================

def test_influence_artifact_error_is_typed_unavailable_not_partial():
    influence = _influence(status="error", available=False,
                           error={"code": "intervention_score_failed", "message": "boom"})
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "intervention_score_failed"
    assert doc["sources"] == []


# ======================================================================================================
# 13. Incomplete matrix
# ======================================================================================================

@pytest.mark.parametrize("matrix_complete,complete_for_selected_spans", [
    (False, True),
    (True, False),
])
def test_incomplete_matrix_never_produces_a_below_floor_claim(matrix_complete, complete_for_selected_spans):
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=15, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=15)],
        links=[_link("p.m000.c000", delta_nats=0.01, effect="supports", clears_floor=False,
                     evidence_state="observed")],
        matrix_complete=matrix_complete, complete_for_selected_spans=complete_for_selected_spans,
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "incomplete_influence_matrix"
    assert doc["sources"] == []


# ======================================================================================================
# 14. Selection inconsistency -- fail closed
# ======================================================================================================

def test_selection_inconsistency_fails_closed():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=15, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=15)],
        links=[_link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    # Corrupt the persisted selection so it disagrees with prompt_sources[].selected.
    influence["selection"]["selected_source_ids"] = []
    influence["selection"]["omitted_source_ids"] = ["p.m000"]
    run = _run(influence_map=influence)
    with pytest.raises(ValueError):
        context_utilization.build_context_utilization(run)


# ======================================================================================================
# 15. Selected source missing coarse spans -- fail closed
# ======================================================================================================

def test_selected_source_missing_coarse_spans_fails_closed():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=15, selected=True)],
        prompt_spans=[],  # no coarse spans at all for the selected source
        links=[],
    )
    run = _run(influence_map=influence)
    with pytest.raises(ValueError):
        context_utilization.build_context_utilization(run)


def test_selected_source_with_coarse_spans_but_no_links_fails_closed():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=15, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=15)],
        links=[],  # matrix claims complete, but no link exists for the coarse span
    )
    run = _run(influence_map=influence)
    with pytest.raises(ValueError):
        context_utilization.build_context_utilization(run)


# ======================================================================================================
# 16. Stable source addresses
# ======================================================================================================

def test_source_span_ids_are_real_stable_addresses():
    influence = _influence(
        prompt_sources=[
            _source("p.m000", start=0, end=20, selected=True),
            _source("p.m001", start=0, end=15, selected=False),
        ],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=20)],
        links=[_link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    for entry in doc["sources"]:
        assert entry["source_span_id"].startswith("span_")
        assert len(entry["source_span_id"]) == len("span_") + 24


# ======================================================================================================
# 17. Deterministic ordering
# ======================================================================================================

def test_ordering_clear_before_below_floor_before_not_measured_by_descending_magnitude():
    influence = _influence(
        prompt_sources=[
            _source("p.m000", start=0, end=10, selected=True),   # clear, weak (1.0)
            _source("p.m001", start=0, end=10, selected=True),   # clear, strong (5.0)
            _source("p.m002", start=0, end=10, selected=True),   # below floor, weak (0.05)
            _source("p.m003", start=0, end=10, selected=True),   # below floor, stronger (0.2)
            _source("p.m004", start=0, end=10, selected=False),  # not measured, appears first in list
            _source("p.m005", start=0, end=10, selected=False),  # not measured, appears second
        ],
        prompt_spans=[
            _coarse("p.m000.c000", "p.m000"), _coarse("p.m001.c000", "p.m001"),
            _coarse("p.m002.c000", "p.m002"), _coarse("p.m003.c000", "p.m003"),
        ],
        links=[
            _link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m001.c000", delta_nats=-5.0, effect="supports", clears_floor=True,
                 evidence_state="causally_supported"),
            _link("p.m002.c000", delta_nats=0.05, effect="supports", clears_floor=False,
                 evidence_state="observed"),
            _link("p.m003.c000", delta_nats=0.2, effect="supports", clears_floor=False,
                 evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)

    order = [entry["native"]["source_id"] for entry in doc["sources"]]
    assert order == ["p.m001", "p.m000", "p.m003", "p.m002", "p.m004", "p.m005"]


def test_ordering_is_reproducible_across_calls():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=10, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000")],
        links=[_link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    first = context_utilization.build_context_utilization(copy.deepcopy(run))
    second = context_utilization.build_context_utilization(copy.deepcopy(run))
    assert first == second


# ======================================================================================================
# 18. Run immutability
# ======================================================================================================

def test_run_is_never_mutated():
    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=20, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=20)],
        links=[_link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    before = copy.deepcopy(run)
    context_utilization.build_context_utilization(run)
    assert run == before


# ======================================================================================================
# 19. No engine/model/worker access
# ======================================================================================================

def test_no_engine_or_worker_access(monkeypatch):
    def _explode(*_a, **_kw):
        raise AssertionError("context_utilization touched an engine/model/worker seam")

    from clozn.server import app as ctx
    monkeypatch.setattr(ctx, "active_engine", _explode, raising=False)
    monkeypatch.setattr(ctx, "ENGINE", None, raising=False)
    import clozn.server.model_routing as model_routing
    monkeypatch.setattr(model_routing, "select_control_model_for_run", _explode, raising=False)

    influence = _influence(
        prompt_sources=[_source("p.m000", start=0, end=20, selected=True)],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=20)],
        links=[_link("p.m000.c000", delta_nats=-1.0, effect="supports", clears_floor=True,
                     evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)
    assert doc["measurement"]["state"] == "available"


# ======================================================================================================
# Never call "not measured" low-effect -- explicit regression, per the spec's own emphasis
# ======================================================================================================

def test_omitted_source_never_labeled_below_measured_floor_even_if_it_would_have_mattered():
    """Source B is omitted purely because max_context_spans bounded the selection -- it may have been
    hugely influential. Clozn does not know, and must never call it below_measured_floor (which would
    silently claim a measurement that never happened)."""
    influence = _influence(
        prompt_sources=[
            _source("p.m000", start=0, end=10, selected=True),
            _source("p.m001", start=0, end=500, selected=False),
        ],
        prompt_spans=[_coarse("p.m000.c000", "p.m000", start=0, end=10)],
        links=[_link("p.m000.c000", delta_nats=0.01, effect="supports", clears_floor=False,
                     evidence_state="observed")],
        max_context_spans=1,
    )
    run = _run(influence_map=influence)
    doc = context_utilization.build_context_utilization(run)

    entry = _by_native_id(doc)["p.m001"]
    assert entry["measurement_state"] == "not_measured"
    assert entry["reason"] == "omitted_by_measurement_selection"
    assert entry.get("effect_state") != "below_measured_floor"
    assert "effect_state" not in entry


# ======================================================================================================
# Input validation (defense in depth)
# ======================================================================================================

def test_requires_a_run_id():
    with pytest.raises(ValueError):
        context_utilization.build_context_utilization({"response": "x"})


def test_rejects_full_privacy():
    with pytest.raises(ValueError):
        context_utilization.build_context_utilization(_run(), privacy="full")
