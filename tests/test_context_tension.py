"""Tests for clozn.runs.context_tension -- the "context tension" detector (E8).

No model, no network, no filesystem outside `tmp_path`. `build_context_tension` is a pure function of
`(run, output_start, output_end, limit, privacy)`; it is proven here to work with zero access to any
engine/model/worker seam, matching Influence Query's own test suite discipline for the identical reason.
"""
from __future__ import annotations

import copy

import pytest

from clozn import schemas
from clozn.runs import context_tension


def _method() -> dict:
    return {
        "name": "teacher_forced_matched_context_replacement", "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim", "caveat": "measured effect only, not correctness",
    }


def _influence(*, status="ok", available=True, error=None, prompt_spans=(), answer_text="",
              answer_spans=(), links=(), artifact_sha256="c" * 64) -> dict:
    doc = {
        "schema": "clozn.context_answer_influence.v1",
        "status": status,
        "available": available,
        "method": _method(),
        "identity": {"model_sha256": "a" * 64},
    }
    if status == "ok":
        doc.update({
            "thresholds": {"cell_abs_delta_nats": 0.5},
            "artifact_sha256": artifact_sha256,
            "prompt_spans": list(prompt_spans),
            "answer": {"scored_text": answer_text},
            "answer_spans": list(answer_spans),
            "links": list(links),
        })
    else:
        doc["error"] = error or {"code": "no_text_context", "message": "no context available"}
    return doc


# A 6-token answer, tokenized on word boundaries -- same convention as test_influence_query.py.
#   "Paris " [0,6)  "is " [6,9)  "the " [9,13)  "capital " [13,21)  "of " [21,24)  "France." [24,31)
_ANSWER = "Paris is the capital of France."
_TOKENS = [
    ("as-0", 0, 6, "Paris "),
    ("as-1", 6, 9, "is "),
    ("as-2", 9, 13, "the "),
    ("as-3", 13, 21, "capital "),
    ("as-4", 21, 24, "of "),
    ("as-5", 24, 31, "France."),
]


def _answer_spans() -> list[dict]:
    return [{"id": aid, "start": start, "end": end, "text": text} for aid, start, end, text in _TOKENS]


def _link(context_span_id, answer_span_id, *, delta_nats, effect, clears_floor, evidence_state,
         context_index=0, answer_index=0) -> dict:
    return {
        "context_span_id": context_span_id, "answer_span_id": answer_span_id,
        "context_index": context_index, "answer_index": answer_index,
        "delta_nats": delta_nats, "abs_delta_nats": abs(delta_nats),
        "effect": effect, "clears_floor": clears_floor, "evidence_state": evidence_state,
    }


def _run(run_id="run-x", response=_ANSWER, **over) -> dict:
    out = {"id": run_id, "response": response}
    out.update(over)
    return out


def _prompt_spans(*ids: str) -> list[dict]:
    return [{"id": pid, "start": i * 10, "end": i * 10 + 5, "text": "x"} for i, pid in enumerate(ids)]


# ======================================================================================================
# 1. Basic opposing pair
# ======================================================================================================

def test_basic_opposing_pair_produces_one_tension():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    schemas.validate(doc)

    assert doc["measurement"]["state"] == "available"
    assert len(doc["tensions"]) == 1
    tension = doc["tensions"][0]
    assert tension["supporting"]["effect"] == "supports"
    assert tension["suppressing"]["effect"] == "suppresses"
    assert tension["supporting"]["source_span_id"] != tension["suppressing"]["source_span_id"]


# ======================================================================================================
# 2. Same-direction links -> no tension
# ======================================================================================================

def test_same_direction_links_produce_no_tension():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=-2.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["tensions"] == []


# ======================================================================================================
# 3. Neutral link never participates
# ======================================================================================================

def test_neutral_link_never_creates_tension():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=0.0, effect="neutral", clears_floor=False,
                  evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["tensions"] == []


# ======================================================================================================
# 4. Observed suppressing link -> no tension (one side below floor)
# ======================================================================================================

def test_one_side_observed_produces_no_tension():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=False,
                  evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["tensions"] == []


# ======================================================================================================
# 5. Both observed -> no tension
# ======================================================================================================

def test_both_sides_observed_produces_no_tension():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=False,
                  evidence_state="observed"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=False,
                  evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["tensions"] == []


# ======================================================================================================
# 6. Multiple sources: Cartesian product of supporting x suppressing, never same-direction pairs
# ======================================================================================================

def test_multiple_sources_produce_the_full_cartesian_opposing_pairs():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2", "ps-3", "ps-4"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=-2.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-3", "as-0", delta_nats=1.5, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-4", "as-0", delta_nats=1.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)

    pairs = {
        (t["native"]["supporting_context_span_id"], t["native"]["suppressing_context_span_id"])
        for t in doc["tensions"]
    }
    assert pairs == {("ps-1", "ps-3"), ("ps-1", "ps-4"), ("ps-2", "ps-3"), ("ps-2", "ps-4")}
    assert len(doc["tensions"]) == 4
    # Never a same-direction pair.
    assert ("ps-1", "ps-2") not in pairs
    assert ("ps-3", "ps-4") not in pairs


# ======================================================================================================
# 7. Different answer spans -> no pair merely because both appear somewhere in the answer
# ======================================================================================================

def test_opposing_links_on_different_answer_spans_produce_no_pair():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-1", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["tensions"] == []
    assert doc["summary"]["answer_spans_examined"] == 6


# ======================================================================================================
# 8. Selected range: only the overlapping answer span's tension is returned
# ======================================================================================================

def test_selected_range_returns_only_the_overlapping_answer_spans_tension():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2", "ps-3", "ps-4"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-3", "as-4", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-4", "as-4", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    # as-4 is "of " [21, 24).
    doc = context_tension.build_context_tension(run, output_start=21, output_end=24)
    assert len(doc["tensions"]) == 1
    assert doc["tensions"][0]["native"]["answer_span_id"] == "as-4"
    assert doc["summary"]["answer_spans_examined"] == 1


# ======================================================================================================
# 9. Partial span overlap: selection starting inside a measured answer span still participates
# ======================================================================================================

def test_partial_overlap_selection_still_participates():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    # [4, 8) starts inside as-0 [0,6) and ends inside as-1 [6,9) -- neither boundary token-aligned.
    doc = context_tension.build_context_tension(run, output_start=4, output_end=8)
    assert len(doc["tensions"]) == 1
    assert doc["tensions"][0]["native"]["answer_span_id"] == "as-0"


# ======================================================================================================
# 10. Whole-answer mode: no range -> all measured answer spans considered
# ======================================================================================================

def test_whole_answer_mode_considers_every_measured_span():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2", "ps-3", "ps-4"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-3", "as-4", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-4", "as-4", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["target"]["scope"] == "whole_answer"
    assert "start" not in doc["target"] and "end" not in doc["target"]
    assert doc["summary"]["answer_spans_examined"] == 6
    assert len(doc["tensions"]) == 2


# ======================================================================================================
# 11. Deterministic ranking
# ======================================================================================================

def test_ranking_prefers_the_strongest_weaker_side_then_combined_magnitude():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2", "ps-3", "ps-4"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            # Pair A: 10.0 supports / 0.2 suppresses -- weak weaker-side.
            _link("ps-1", "as-0", delta_nats=-10.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=0.2, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    influence["prompt_spans"].extend(_prompt_spans("ps-5", "ps-6"))
    influence["links"].extend([
        # Pair B (different answer span, so it doesn't cross with pair A's sources): 4.0 / 3.5 -- a
        # genuine push-pull, stronger weaker-side than pair A's 0.2.
        _link("ps-5", "as-4", delta_nats=-4.0, effect="supports", clears_floor=True,
              evidence_state="causally_supported"),
        _link("ps-6", "as-4", delta_nats=3.5, effect="suppresses", clears_floor=True,
              evidence_state="causally_supported"),
    ])
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)

    assert len(doc["tensions"]) == 2
    first, second = doc["tensions"]
    assert first["native"]["answer_span_id"] == "as-4"  # pair B: weaker side 3.5 > pair A's 0.2
    assert second["native"]["answer_span_id"] == "as-0"


def test_ranking_is_stable_and_reproducible_across_calls():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2", "ps-3"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-3", "as-0", delta_nats=2.5, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    first = context_tension.build_context_tension(copy.deepcopy(run))
    second = context_tension.build_context_tension(copy.deepcopy(run))
    assert first == second
    assert len(first["tensions"]) == 2


# ======================================================================================================
# 12. Limit applies after ranking
# ======================================================================================================

def test_limit_applies_after_ranking():
    influence = _influence(
        prompt_spans=(
            [{"id": f"ps-sup-{i}", "start": i, "end": i + 1, "text": "x"} for i in range(3)]
            + [{"id": f"ps-sup-{letter}", "start": 10 + i, "end": 11 + i, "text": "x"}
               for i, letter in enumerate(("a", "b", "c"))]
        ),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link(f"ps-sup-{i}", "as-0", delta_nats=-float(i + 1), effect="supports", clears_floor=True,
                  evidence_state="causally_supported")
            for i in range(3)
        ] + [
            _link(f"ps-sup-{letter}", "as-0", delta_nats=float(i + 1), effect="suppresses",
                  clears_floor=True, evidence_state="causally_supported")
            for i, letter in enumerate(("a", "b", "c"))
        ],
    )
    run = _run(influence_map=influence)
    full = context_tension.build_context_tension(run, limit=100)
    limited = context_tension.build_context_tension(run, limit=2)

    assert full["summary"]["tension_pairs"] == 9  # 3 supporting x 3 suppressing
    assert len(limited["tensions"]) == 2
    assert limited["summary"]["tension_pairs"] == 9
    assert limited["summary"]["returned_tension_pairs"] == 2
    assert limited["tensions"] == full["tensions"][:2]


# ======================================================================================================
# 13. No influence map -> not_measured
# ======================================================================================================

def test_no_influence_map_yields_not_measured():
    run = _run()
    doc = context_tension.build_context_tension(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "not_measured"
    assert doc["measurement"]["reason"] == "no_influence_map"
    assert doc["tensions"] == []


# ======================================================================================================
# 14. Valid map with zero tensions -> available, distinct from not_measured
# ======================================================================================================

def test_valid_measurement_with_zero_tensions_is_available_not_not_measured():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "available"
    assert doc["tensions"] == []
    assert doc["summary"]["tension_pairs"] == 0


# ======================================================================================================
# 15. Stale answer -> fail closed
# ======================================================================================================

def test_stale_influence_map_on_drifted_answer_fails_closed():
    stale_answer = "A completely different recorded answer."
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=stale_answer,
        answer_spans=[{"id": "as-0", "start": 0, "end": len(stale_answer), "text": stale_answer}],
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(response=_ANSWER, influence_map=influence)
    doc = context_tension.build_context_tension(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "answer_text_mismatch"
    assert doc["tensions"] == []


# ======================================================================================================
# 16. Redacted run -> no text reconstructed
# ======================================================================================================

def test_redacted_run_returns_unavailable_and_never_leaks_text():
    run = _run(
        run_id="run-redacted", response=None,
        redaction={"status": "redacted"}, flags=["redacted"],
        influence_map=_influence(
            prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "PRIVATE SOURCE"},
                         {"id": "ps-2", "start": 10, "end": 20, "text": "PRIVATE SOURCE B"}],
            answer_text="PRIVATE ANSWER TEXT",
            answer_spans=[{"id": "as-0", "start": 0, "end": 19, "text": "PRIVATE ANSWER TEXT"}],
            links=[
                _link("ps-1", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                      evidence_state="causally_supported"),
                _link("ps-2", "as-0", delta_nats=1.0, effect="suppresses", clears_floor=True,
                      evidence_state="causally_supported"),
            ],
        ),
    )
    doc = context_tension.build_context_tension(run)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "answer_text_redacted"
    assert doc["tensions"] == []
    assert "basis_sha256" not in doc["target"]
    assert "PRIVATE" not in repr(doc)


# ======================================================================================================
# 17. Stable tension_id across calls
# ======================================================================================================

def test_tension_id_is_stable_across_calls():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    first = context_tension.build_context_tension(copy.deepcopy(run))
    second = context_tension.build_context_tension(copy.deepcopy(run))
    assert first["tensions"][0]["tension_id"] == second["tensions"][0]["tension_id"]
    assert first["tensions"][0]["tension_id"].startswith("tension_")


# ======================================================================================================
# 18. Run immutability
# ======================================================================================================

def test_run_is_never_mutated():
    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    before = copy.deepcopy(run)
    context_tension.build_context_tension(run)
    assert run == before


# ======================================================================================================
# 19. No engine/model/worker access
# ======================================================================================================

def test_no_engine_or_worker_access(monkeypatch):
    def _explode(*_a, **_kw):
        raise AssertionError("context_tension touched an engine/model/worker seam")

    from clozn.server import app as ctx
    monkeypatch.setattr(ctx, "active_engine", _explode, raising=False)
    monkeypatch.setattr(ctx, "ENGINE", None, raising=False)
    import clozn.server.model_routing as model_routing
    monkeypatch.setattr(model_routing, "select_control_model_for_run", _explode, raising=False)

    influence = _influence(
        prompt_spans=_prompt_spans("ps-1", "ps-2"),
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = context_tension.build_context_tension(run)
    assert doc["measurement"]["state"] == "available"
    assert len(doc["tensions"]) == 1


# ======================================================================================================
# Input validation (defense in depth -- the HTTP route also validates before calling this function)
# ======================================================================================================

@pytest.mark.parametrize("kwargs", [
    {"output_start": 0},                      # end omitted
    {"output_end": 5},                        # start omitted
    {"output_start": -1, "output_end": 5},
    {"output_start": 5, "output_end": 5},
    {"output_start": 5, "output_end": 2},
    {"output_start": 0.5, "output_end": 5},
    {"output_start": 0, "output_end": 5, "limit": 0},
    {"output_start": 0, "output_end": 5, "limit": 101},
    {"output_start": 0, "output_end": 999},
    {"output_start": 0, "output_end": 5, "privacy": "full"},
])
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        context_tension.build_context_tension(_run(), **kwargs)


def test_requires_a_run_id():
    with pytest.raises(ValueError):
        context_tension.build_context_tension({"response": "x"})
