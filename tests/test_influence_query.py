"""Tests for clozn.runs.influence_query -- the "Why this?" read-only influence query (E7).

No model, no network, no filesystem outside `tmp_path`. `build_influence_query` is a pure function of
`(run, output_start, output_end, limit, privacy)`; it is proven here to work with zero access to any
engine/model/worker seam, matching claim_support's own test suite discipline for the identical reason:
"a useful test should be able to monkeypatch the active model to explode and prove the query still
works" (this feature's own spec, verbatim).
"""
from __future__ import annotations

import copy

import pytest

from clozn import schemas
from clozn.runs import influence_query


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


# A 6-token answer, tokenized on word boundaries, used across the overlap/ranking tests below.
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


def _by_native_answer(document: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for link in document["links"]:
        grouped.setdefault(link["native"]["answer_span_id"], []).append(link)
    return grouped


# ======================================================================================================
# 1. Single answer span
# ======================================================================================================

def test_single_answer_span_selection_returns_only_that_spans_links():
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Source about Paris."}],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-2.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-1", "as-1", delta_nats=-1.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=5)  # inside "Paris "
    schemas.validate(doc)

    assert doc["measurement"]["state"] == "available"
    assert len(doc["links"]) == 1
    assert doc["links"][0]["native"]["answer_span_id"] == "as-0"
    assert doc["summary"]["selected_answer_spans"] == 1


# ======================================================================================================
# 2. Selection crossing multiple answer spans
# ======================================================================================================

def test_selection_crossing_multiple_answer_spans_includes_links_from_all_of_them():
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Source A."}],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link("ps-1", aid, delta_nats=-1.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported")
            for aid in ("as-0", "as-1", "as-2", "as-3", "as-4", "as-5")
        ],
    )
    run = _run(influence_map=influence)
    # [3, 15) crosses as-0 [0,6), as-1 [6,9), as-2 [9,13), and as-3 [13,21) (13 < 15).
    doc = influence_query.build_influence_query(run, output_start=3, output_end=15)

    seen = {link["native"]["answer_span_id"] for link in doc["links"]}
    assert seen == {"as-0", "as-1", "as-2", "as-3"}
    assert doc["summary"]["selected_answer_spans"] == 4
    assert doc["summary"]["measured_links"] == 4


# ======================================================================================================
# 3. Partial-token selection: overlap works without alignment to token boundaries
# ======================================================================================================

def test_partial_token_selection_does_not_require_boundary_alignment():
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Source A."}],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-1", "as-1", delta_nats=-1.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
        ],
    )
    run = _run(influence_map=influence)
    # [4, 8) starts inside as-0 [0,6) and ends inside as-1 [6,9) -- neither boundary is token-aligned.
    doc = influence_query.build_influence_query(run, output_start=4, output_end=8)

    seen = {link["native"]["answer_span_id"] for link in doc["links"]}
    assert seen == {"as-0", "as-1"}


# ======================================================================================================
# 4. Supporting + suppressing + neutral effects are preserved exactly, never reinterpreted
# ======================================================================================================

def test_supports_suppresses_neutral_are_preserved_exactly():
    influence = _influence(
        prompt_spans=[
            {"id": "ps-1", "start": 0, "end": 20, "text": "Supporting source."},
            {"id": "ps-2", "start": 20, "end": 40, "text": "Suppressing source."},
            {"id": "ps-3", "start": 40, "end": 60, "text": "Neutral source."},
        ],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-2", "as-0", delta_nats=2.5, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported"),
            _link("ps-3", "as-0", delta_nats=0.0, effect="neutral", clears_floor=False,
                  evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=6)

    by_effect = {link["effect"]: link for link in doc["links"]}
    assert set(by_effect) == {"supports", "suppresses", "neutral"}
    assert by_effect["supports"]["delta_nats"] == -3.0
    assert by_effect["suppresses"]["delta_nats"] == 2.5
    assert by_effect["neutral"]["delta_nats"] == 0.0
    assert by_effect["neutral"]["evidence_state"] == "observed"
    assert doc["summary"]["supporting_links"] == 1
    assert doc["summary"]["suppressing_links"] == 1
    assert doc["summary"]["neutral_links"] == 1


# ======================================================================================================
# 5. Ranking: causally_supported before observed, then descending abs_delta_nats, then stable indices
# ======================================================================================================

def test_ranking_is_deterministic_and_matches_the_documented_order():
    influence = _influence(
        prompt_spans=[
            {"id": "ps-1", "start": 0, "end": 10, "text": "A"},
            {"id": "ps-2", "start": 10, "end": 20, "text": "B"},
            {"id": "ps-3", "start": 20, "end": 30, "text": "C"},
            {"id": "ps-4", "start": 30, "end": 40, "text": "D"},
        ],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=-1.0, effect="supports", clears_floor=False,
                  evidence_state="observed", context_index=0, answer_index=0),
            _link("ps-2", "as-0", delta_nats=-5.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=1, answer_index=0),
            _link("ps-3", "as-0", delta_nats=3.0, effect="suppresses", clears_floor=True,
                  evidence_state="causally_supported", context_index=2, answer_index=0),
            _link("ps-4", "as-0", delta_nats=-9.0, effect="supports", clears_floor=False,
                  evidence_state="observed", context_index=3, answer_index=0),
        ],
    )
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=6, limit=50)

    order = [link["native"]["context_span_id"] for link in doc["links"]]
    # ps-2 (causally_supported, |5.0|) before ps-3 (causally_supported, |3.0|) before the two observed
    # links, ps-4 (|9.0|) before ps-1 (|1.0|) since observed ties break on descending abs_delta_nats too.
    assert order == ["ps-2", "ps-3", "ps-4", "ps-1"]


def test_ranking_is_stable_across_repeated_calls():
    influence = _influence(
        prompt_spans=[{"id": f"ps-{i}", "start": i * 10, "end": i * 10 + 5, "text": "x"} for i in range(5)],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link(f"ps-{i}", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=i, answer_index=0)
            for i in range(5)
        ],
    )
    run = _run(influence_map=influence)
    first = influence_query.build_influence_query(copy.deepcopy(run), output_start=0, output_end=6)
    second = influence_query.build_influence_query(copy.deepcopy(run), output_start=0, output_end=6)
    assert first == second


# ======================================================================================================
# 6. Limit is applied AFTER ranking, and summary distinguishes measured vs returned
# ======================================================================================================

def test_limit_applies_after_ranking_not_before():
    influence = _influence(
        prompt_spans=[{"id": f"ps-{i}", "start": i * 10, "end": i * 10 + 5, "text": "x"} for i in range(5)],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            # Deliberately out of magnitude order in the source data.
            _link("ps-0", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=0),
            _link("ps-1", "as-0", delta_nats=-5.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=1),
            _link("ps-2", "as-0", delta_nats=-3.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=2),
            _link("ps-3", "as-0", delta_nats=-4.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=3),
            _link("ps-4", "as-0", delta_nats=-2.0, effect="supports", clears_floor=True,
                  evidence_state="causally_supported", context_index=4),
        ],
    )
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=6, limit=2)

    assert len(doc["links"]) == 2
    order = [link["native"]["context_span_id"] for link in doc["links"]]
    assert order == ["ps-1", "ps-3"]  # the two strongest by abs_delta_nats: 5.0, then 4.0
    assert doc["summary"]["measured_links"] == 5
    assert doc["summary"]["returned_links"] == 2


# ======================================================================================================
# 7. No influence map -> not_measured, never an empty "nothing mattered" result
# ======================================================================================================

def test_no_influence_map_yields_not_measured():
    run = _run()
    doc = influence_query.build_influence_query(run, output_start=0, output_end=5)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == "not_measured"
    assert doc["measurement"]["reason"] == "no_influence_map"
    assert doc["links"] == []


def test_blob_unavailable_marker_yields_not_measured():
    run = _run(influence_map={"unavailable": "blob missing", "sha256": "a" * 64})
    doc = influence_query.build_influence_query(run, output_start=0, output_end=5)
    assert doc["measurement"]["state"] == "not_measured"
    assert doc["measurement"]["reason"] == "no_influence_map"


# ======================================================================================================
# 8. Influence measurement unavailable/error -> typed result, reason from the persisted artifact
# ======================================================================================================

@pytest.mark.parametrize("status,error_code,expected_state", [
    ("unavailable", "no_text_context", "unavailable"),
    ("error", "intervention_score_failed", "error"),
])
def test_influence_map_present_but_not_ok_is_typed_not_recomputed(status, error_code, expected_state):
    influence = _influence(status=status, available=False, error={"code": error_code, "message": "x"})
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=5)
    schemas.validate(doc)
    assert doc["measurement"]["state"] == expected_state
    assert doc["measurement"]["reason"] == error_code
    assert doc["links"] == []


# ======================================================================================================
# 9. Valid measurement but below-floor only: available, zero causally_supported_links, distinct from
#    not_measured, and observed links are still returned (never suppressed as "irrelevant")
# ======================================================================================================

def test_below_floor_only_is_available_with_zero_causally_supported_but_observed_links_returned():
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Weak source."}],
        answer_text=_ANSWER,
        answer_spans=_answer_spans(),
        links=[
            _link("ps-1", "as-0", delta_nats=0.05, effect="supports", clears_floor=False,
                  evidence_state="observed"),
        ],
    )
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=6)
    schemas.validate(doc)

    assert doc["measurement"]["state"] == "available"
    assert doc["summary"]["causally_supported_links"] == 0
    assert doc["summary"]["observed_links"] == 1
    assert len(doc["links"]) == 1
    assert doc["links"][0]["evidence_state"] == "observed"
    assert doc["links"][0]["clears_floor"] is False


def test_available_state_never_conflated_with_not_measured():
    """The two 'nothing to show' shapes must never collapse: not_measured differs from available+0 by
    measurement.state alone, not by whether links happens to be empty in either case."""
    no_map = influence_query.build_influence_query(_run(), output_start=0, output_end=5)
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "x"}],
        answer_text=_ANSWER, answer_spans=_answer_spans(), links=[],
    )
    measured_empty = influence_query.build_influence_query(
        _run(influence_map=influence), output_start=0, output_end=6,
    )
    assert no_map["measurement"]["state"] == "not_measured"
    assert measured_empty["measurement"]["state"] == "available"
    assert no_map["links"] == measured_empty["links"] == []


# ======================================================================================================
# 10. Answer drift: a valid influence artifact scored for a DIFFERENT answer must never be trusted
# ======================================================================================================

def test_stale_influence_map_on_drifted_answer_is_unavailable():
    stale_answer = "A completely different recorded answer."
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "Some source."}],
        answer_text=stale_answer,
        answer_spans=[{"id": "as-0", "start": 0, "end": len(stale_answer), "text": stale_answer}],
        links=[_link("ps-1", "as-0", delta_nats=-2.0, effect="supports", clears_floor=True,
                      evidence_state="causally_supported")],
    )
    # run.response has since changed (e.g. the run was regenerated) -- the influence map still describes
    # the OLD answer text.
    run = _run(response=_ANSWER, influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=6)
    schemas.validate(doc)

    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "answer_text_mismatch"
    assert doc["links"] == []


# ======================================================================================================
# 11. Redaction: redacted answer text cannot be queried or reconstructed
# ======================================================================================================

def test_redacted_run_returns_unavailable_and_never_leaks_text():
    run = _run(
        run_id="run-redacted", response=None,
        redaction={"status": "redacted"}, flags=["redacted"],
        influence_map=_influence(
            prompt_spans=[{"id": "ps-1", "start": 0, "end": 10, "text": "PRIVATE SOURCE TEXT"}],
            answer_text="PRIVATE ANSWER TEXT", answer_spans=[
                {"id": "as-0", "start": 0, "end": 19, "text": "PRIVATE ANSWER TEXT"},
            ],
            links=[_link("ps-1", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                          evidence_state="causally_supported")],
        ),
    )
    doc = influence_query.build_influence_query(run, output_start=0, output_end=5)
    schemas.validate(doc)

    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "answer_text_redacted"
    assert doc["links"] == []
    assert "basis_sha256" not in doc["target"]
    assert "PRIVATE" not in repr(doc)


def test_redaction_authoritative_even_if_response_field_still_carries_leftover_text():
    """The redaction flag wins even if a legacy/buggy record still has literal text sitting in
    run['response'] -- the persisted redaction lifecycle is authoritative, never a field to second-guess."""
    run = _run(
        response="LEFTOVER PRIVATE TEXT should never be read",
        redaction={"status": "redacted"}, flags=["redacted"],
    )
    doc = influence_query.build_influence_query(run, output_start=0, output_end=5)
    assert doc["measurement"]["state"] == "unavailable"
    assert doc["measurement"]["reason"] == "answer_text_redacted"
    assert "LEFTOVER" not in repr(doc)


# ======================================================================================================
# 12. Run immutability
# ======================================================================================================

def test_run_is_never_mutated():
    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Source."}],
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[_link("ps-1", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                      evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    before = copy.deepcopy(run)
    influence_query.build_influence_query(run, output_start=0, output_end=6)
    assert run == before


# ======================================================================================================
# 13. No engine/model/worker access
# ======================================================================================================

def test_no_engine_or_worker_access(monkeypatch):
    """Patches every plausible engine/model-routing entry point to explode if touched, then proves the
    query still succeeds from a valid stored artifact alone -- this feature must work with no active
    worker attached (this module's own docstring guarantee)."""
    def _explode(*_a, **_kw):
        raise AssertionError("influence_query touched an engine/model/worker seam")

    from clozn.server import app as ctx
    monkeypatch.setattr(ctx, "active_engine", _explode, raising=False)
    monkeypatch.setattr(ctx, "ENGINE", None, raising=False)
    import clozn.server.model_routing as model_routing
    monkeypatch.setattr(model_routing, "select_control_model_for_run", _explode, raising=False)

    influence = _influence(
        prompt_spans=[{"id": "ps-1", "start": 0, "end": 20, "text": "Source."}],
        answer_text=_ANSWER, answer_spans=_answer_spans(),
        links=[_link("ps-1", "as-0", delta_nats=-1.0, effect="supports", clears_floor=True,
                      evidence_state="causally_supported")],
    )
    run = _run(influence_map=influence)
    doc = influence_query.build_influence_query(run, output_start=0, output_end=6)
    assert doc["measurement"]["state"] == "available"
    assert len(doc["links"]) == 1


# ======================================================================================================
# Input validation (defense in depth -- the HTTP route also validates before calling this function)
# ======================================================================================================

@pytest.mark.parametrize("kwargs", [
    {"output_start": -1, "output_end": 5},
    {"output_start": 5, "output_end": 5},
    {"output_start": 5, "output_end": 2},
    {"output_start": 0.5, "output_end": 5},
    {"output_start": 0, "output_end": True},
    {"output_start": 0, "output_end": 5, "limit": 0},
    {"output_start": 0, "output_end": 5, "limit": 51},
    {"output_start": 0, "output_end": 5, "limit": 1.5},
    {"output_start": 0, "output_end": 999},
    {"output_start": 0, "output_end": 5, "privacy": "full"},
])
def test_invalid_arguments_raise_value_error(kwargs):
    with pytest.raises(ValueError):
        influence_query.build_influence_query(_run(), **kwargs)


def test_requires_a_run_id():
    with pytest.raises(ValueError):
        influence_query.build_influence_query({"response": "x"}, output_start=0, output_end=1)
