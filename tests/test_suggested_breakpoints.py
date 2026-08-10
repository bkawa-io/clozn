"""Tests for the metadata-only Suggested Breakpoints composition layer."""
from __future__ import annotations

import copy
import json

import pytest

from clozn import schemas
from clozn.runs import close_calls, suggested_breakpoints


ANSWER = "Paris is the capital of France."
TOKENS = ["Paris ", "is ", "the ", "capital ", "of ", "France."]
TOKEN_INTERVALS = [(0, 6), (6, 9), (9, 13), (13, 21), (21, 24), (24, 31)]


def _trace(tokens=TOKENS, confidence=None, alternatives=None, token_ids=None):
    tokens = list(tokens)
    if confidence is None:
        confidence = [0.9] * len(tokens)
    if alternatives is None:
        alternatives = [[] for _ in tokens]
    trace = {
        "tokens": list(tokens),
        "confidence": list(confidence),
        "alternatives": copy.deepcopy(alternatives),
    }
    if token_ids is not None:
        trace["token_ids"] = list(token_ids)
    return trace


def _run(response=ANSWER, *, trace=None, **extra):
    out = {"id": "run-sb", "response": response}
    if trace is not None:
        out["trace"] = trace
    out.update(extra)
    return out


def _close_trace(token="answer", rival="other", emitted_probability=0.43, rival_probability=0.39,
                *, token_id=None, rival_id=None):
    alternative = {"piece": rival, "prob": rival_probability}
    if rival_id is not None:
        alternative["token_id"] = rival_id
    trace = _trace(
        [token], [emitted_probability], [[alternative]],
        [token_id] if token_id is not None else None,
    )
    return trace


def _method():
    return {
        "name": "teacher_forced_matched_context_replacement",
        "mode": "forced_score_intervention",
        "claim_limit": "no percentage claim",
        "caveat": "measured effect only, not correctness",
    }


def _prompt_spans(ids):
    return [
        {"id": source_id, "start": index * 10, "end": index * 10 + 5, "text": "PRIVATE SOURCE"}
        for index, source_id in enumerate(ids)
    ]


def _answer_spans(response=ANSWER, intervals=None):
    if intervals is None:
        intervals = TOKEN_INTERVALS
    return [
        {"id": f"as-{index}", "start": start, "end": end, "text": response[start:end]}
        for index, (start, end) in enumerate(intervals)
    ]


def _link(context_id, answer_id, delta, effect, *, evidence_state="causally_supported", index=0):
    return {
        "context_span_id": context_id,
        "answer_span_id": answer_id,
        "context_index": index,
        "answer_index": index,
        "delta_nats": delta,
        "abs_delta_nats": abs(delta),
        "effect": effect,
        "clears_floor": evidence_state == "causally_supported",
        "evidence_state": evidence_state,
    }


def _influence(response=ANSWER, *, source_ids=("ps-1", "ps-2"), intervals=None, links=()):
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": _method(),
        "identity": {"model_sha256": "a" * 64},
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "artifact_sha256": "c" * 64,
        "prompt_spans": _prompt_spans(source_ids),
        "answer": {"scored_text": response},
        "answer_spans": _answer_spans(response, intervals),
        "links": list(links),
    }


def _tension_run(*, response=ANSWER, tokens=TOKENS, intervals=None, links=(), source_ids=("ps-1", "ps-2")):
    return _run(
        response,
        trace=_trace(tokens),
        influence_map=_influence(
            response, source_ids=source_ids, intervals=intervals, links=links,
        ),
    )


def _build(run, **kwargs):
    document = suggested_breakpoints.build_suggested_breakpoints(run, **kwargs)
    schemas.validate(document)
    return document


def _first_reason(document, reason_type):
    for breakpoint in document["breakpoints"]:
        for reason in breakpoint["reasons"]:
            if reason["type"] == reason_type:
                return breakpoint, reason
    raise AssertionError(f"no {reason_type} reason in {document['breakpoints']!r}")


# Close-call evidence -------------------------------------------------------------------------------


def test_ordinary_close_call_creates_an_exact_token_breakpoint():
    document = _build(_run(response="answer", trace=_close_trace()))
    assert document["breakpoints"][0]["position"] == 0
    assert document["breakpoints"][0]["placement"] == "exact_token_decision"
    assert document["breakpoints"][0]["rank_class"] == "close_call"
    assert document["breakpoints"][0]["reasons"][0]["meaningful_heuristic"] is False


def test_meaningful_close_call_uses_the_existing_heuristic_without_relabeling_it():
    document = _build(_run(response="5", trace=_close_trace("5", "0", 0.43, 0.39)))
    assert document["breakpoints"][0]["rank_class"] == "meaningful_close_call"
    assert document["breakpoints"][0]["reasons"][0]["meaningful_heuristic"] is True
    assert "semantic" not in json.dumps(document).lower()


def test_no_close_call_produces_no_close_call_breakpoint():
    document = _build(_run(response="answer", trace=_trace(["answer"], [0.9], [[{"piece": "other", "prob": 0.1}]])))
    assert document["breakpoints"] == []
    assert document["summary"]["candidate_state"] == "none_detected"


def test_raw_low_confidence_does_not_create_a_breakpoint():
    document = _build(_run(response="answer", trace=_close_trace(emitted_probability=0.20, rival_probability=0.19)))
    assert document["breakpoints"] == []


def test_top_k_entropy_alone_does_not_create_a_breakpoint():
    trace = _trace(
        ["answer"], [0.40],
        [[{"piece": "other", "prob": 0.29}, {"piece": "third", "prob": 0.20}]],
    )
    document = _build(_run(response="answer", trace=trace))
    assert document["breakpoints"] == []


def test_sampled_runner_up_preserves_chosen_and_rival_metadata():
    document = _build(_run(
        response="blue",
        trace=_close_trace("blue", "red", 0.39, 0.43, token_id=1234, rival_id=5678),
    ))
    reason = document["breakpoints"][0]["reasons"][0]
    assert reason["emitted_probability"] == 0.39
    assert reason["rival_probability"] == 0.43
    assert reason["emitted_token_id"] == 1234
    assert reason["rival_token_id"] == 5678


def test_missing_token_ids_do_not_destroy_detection():
    document = _build(_run(response="blue", trace=_close_trace("blue", "red", 0.39, 0.43)))
    reason = document["breakpoints"][0]["reasons"][0]
    assert "emitted_token_id" not in reason
    assert "rival_token_id" not in reason


# Context Tension evidence ---------------------------------------------------------------------------


def _one_tension(*, answer_id="as-0", support=-3.0, suppress=2.0, response=ANSWER, tokens=TOKENS,
                 intervals=None):
    links = [
        _link("ps-1", answer_id, support, "supports", index=0),
        _link("ps-2", answer_id, suppress, "suppresses", index=1),
    ]
    return _tension_run(response=response, tokens=tokens, intervals=intervals, links=links)


def test_tension_only_span_projects_to_the_first_overlapping_token():
    document = _build(_one_tension())
    breakpoint = document["breakpoints"][0]
    assert breakpoint["position"] == 0
    assert breakpoint["placement"] == "answer_span_entry_proxy"
    assert breakpoint["rank_class"] == "context_tension"
    assert breakpoint["token_interval"] == {
        "start": 0, "end": 6, "unit": "unicode_code_points", "interval": "half_open",
    }
    _, reason = _first_reason(document, "context_tension")
    assert reason["tension_pair_count"] == 1


def test_same_direction_influence_does_not_create_tension():
    links = [
        _link("ps-1", "as-0", -3, "supports"),
        _link("ps-2", "as-0", -2, "supports"),
    ]
    document = _build(_tension_run(links=links))
    assert document["evidence"]["context_tension"]["state"] == "available"
    assert document["evidence"]["context_tension"]["tension_pairs"] == 0
    assert document["breakpoints"] == []


def test_observed_below_floor_link_does_not_create_tension():
    links = [
        _link("ps-1", "as-0", -3, "supports"),
        _link("ps-2", "as-0", 2, "suppresses", evidence_state="observed"),
    ]
    document = _build(_tension_run(links=links))
    assert document["evidence"]["context_tension"]["tension_pairs"] == 0
    assert document["breakpoints"] == []


def test_tension_inside_close_call_merges_at_the_same_position():
    trace = _trace(
        ["Paris ", "is ", "the ", "capital ", "of ", "France."],
        [0.43, 0.9, 0.9, 0.9, 0.9, 0.9],
        [[{"piece": "London ", "prob": 0.39}], [], [], [], [], []],
    )
    document = _build(_run(
        response=ANSWER,
        trace=trace,
        influence_map=_influence(
            links=[_link("ps-1", "as-0", -3, "supports"), _link("ps-2", "as-0", 2, "suppresses")],
        ),
    ))
    assert len(document["breakpoints"]) == 1
    assert document["breakpoints"][0]["position"] == 0
    assert document["breakpoints"][0]["placement"] == "exact_token_decision"
    assert document["breakpoints"][0]["rank_class"] == "combined"
    assert {reason["type"] for reason in document["breakpoints"][0]["reasons"]} == {
        "close_call", "context_tension",
    }


def test_multiple_tension_pairs_aggregate_without_duplicate_breakpoints():
    links = [
        _link("ps-1", "as-0", -3, "supports", index=0),
        _link("ps-2", "as-0", -4, "supports", index=1),
        _link("ps-3", "as-0", 2, "suppresses", index=2),
    ]
    run = _tension_run(links=links, source_ids=("ps-1", "ps-2", "ps-3"))
    document = _build(run)
    _, reason = _first_reason(document, "context_tension")
    assert reason["tension_pair_count"] == 2
    assert reason["distinct_source_span_count"] == 3
    assert len(document["breakpoints"]) == 1


def test_multiple_tension_spans_mapping_to_one_token_are_aggregated():
    response = "Paris is"
    intervals = [(0, 5), (3, 8)]
    links = [
        _link("ps-1", "as-0", -3, "supports"), _link("ps-2", "as-0", 2, "suppresses"),
        _link("ps-1", "as-1", -4, "supports"), _link("ps-2", "as-1", 3, "suppresses"),
    ]
    document = _build(_tension_run(
        response=response, tokens=[response], intervals=intervals, links=links,
    ))
    _, reason = _first_reason(document, "context_tension")
    assert reason["answer_span_ids"] == sorted(reason["answer_span_ids"]) or len(reason["answer_span_ids"]) == 2
    assert len(reason["answer_span_ids"]) == 2
    assert reason["tension_pair_count"] == 2
    assert len(document["breakpoints"]) == 1


def test_strongest_pair_follows_context_tension_weaker_side_ordering():
    links = [
        _link("ps-1", "as-0", -3, "supports", index=0),
        _link("ps-2", "as-0", 2, "suppresses", index=1),
        _link("ps-3", "as-0", -5, "supports", index=2),
        _link("ps-4", "as-0", 0.8, "suppresses", index=3),
    ]
    document = _build(_tension_run(links=links, source_ids=("ps-1", "ps-2", "ps-3", "ps-4")))
    _, reason = _first_reason(document, "context_tension")
    # The two (-5, 2) and (-3, 2) pairs tie on the weaker side; Context Tension then prefers the
    # larger combined magnitude, so the (-5, 2) pair leads.
    strongest = reason["strongest_pair"]
    assert strongest["supporting_source_span_id"] != strongest["suppressing_source_span_id"]
    assert strongest["supporting_abs_delta_nats"] == 5
    assert strongest["suppressing_abs_delta_nats"] == 2


# Geometry, coverage, privacy ------------------------------------------------------------------------


def test_exact_token_reconstruction_exposes_unicode_code_point_intervals():
    document = _build(_run(response="Café 😊", trace=_trace(["Café", " 😊"])))
    assert document["evidence"]["answer_alignment"]["state"] == "available"
    assert document["coordinates"]["recorded_token_count"] == 2
    # There are no breakpoint candidates, but a close call at the second token proves the interval.
    trace = _trace(["Café", " 😊"], [0.43, 0.9], [[{"piece": "Cafe", "prob": 0.39}], []])
    document = _build(_run(response="Café 😊", trace=trace))
    assert document["breakpoints"][0]["token_interval"]["start"] == 0
    assert document["breakpoints"][0]["token_interval"]["end"] == 4


def test_emoji_code_point_interval_is_not_utf16_length():
    response = "😊 now"
    document = _build(_tension_run(
        response=response,
        tokens=["😊", " now"],
        intervals=[(0, 1)],
        links=[_link("ps-1", "as-0", -3, "supports"), _link("ps-2", "as-0", 2, "suppresses")],
    ))
    assert document["breakpoints"][0]["token_interval"] == {
        "start": 0, "end": 1, "unit": "unicode_code_points", "interval": "half_open",
    }


def test_trace_response_mismatch_disables_tension_projection_but_keeps_close_calls():
    trace = _close_trace("hullo", "hello", 0.43, 0.39)
    influence = _influence(
        response="hello", intervals=[(0, 5)],
        links=[_link("ps-1", "as-0", -3, "supports"), _link("ps-2", "as-0", 2, "suppresses")],
    )
    document = _build(_run(response="hello", trace=trace, influence_map=influence))
    assert document["evidence"]["answer_alignment"] == {
        "state": "unavailable", "reason": "trace_response_mismatch",
    }
    assert document["evidence"]["context_tension"]["tension_pairs"] == 1
    assert len(document["breakpoints"]) == 1
    assert document["breakpoints"][0]["reasons"][0]["type"] == "close_call"
    assert document["analysis"]["state"] == "partially_available"


def test_redacted_answer_disables_answer_span_projection_honestly():
    run = _one_tension()
    run["redaction"] = {"status": "redacted"}
    document = _build(run)
    assert document["evidence"]["answer_alignment"] == {
        "state": "unavailable", "reason": "answer_text_redacted",
    }
    assert document["breakpoints"] == []
    assert document["analysis"]["state"] == "partially_available" or document["analysis"]["state"] == "unavailable"


def test_missing_response_and_empty_trace_are_explicitly_unavailable():
    document = _build(_run(response=None, trace={"tokens": [], "confidence": [], "alternatives": []}))
    assert document["evidence"]["answer_alignment"]["reason"] == "no_recorded_answer_text"
    assert document["evidence"]["close_calls"]["state"] == "unavailable"
    assert document["breakpoints"] == []


def test_zero_width_or_malformed_token_piece_cannot_fabricate_geometry():
    response = "a"
    influence = _influence(
        response=response, intervals=[(0, 1)],
        links=[_link("ps-1", "as-0", -3, "supports"), _link("ps-2", "as-0", 2, "suppresses")],
    )
    document = _build(_run(
        response=response,
        trace=_trace(["", "a"], [0.9, 0.9], [[], []]),
        influence_map=influence,
    ))
    assert document["breakpoints"][0]["position"] == 1
    assert document["breakpoints"][0]["token_interval"]["start"] == 0
    malformed = _build(_run(response=response, trace={"tokens": [None], "confidence": [0.9], "alternatives": [[]]}))
    assert malformed["evidence"]["answer_alignment"]["state"] == "unavailable"
    assert "token_interval" not in json.dumps(malformed)


def test_missing_influence_is_not_measured_not_zero_tension():
    document = _build(_run(response=ANSWER, trace=_trace()))
    assert document["evidence"]["context_tension"] == {
        "state": "not_measured", "reason": "no_influence_map",
    }
    assert document["analysis"]["state"] == "available"


@pytest.mark.parametrize(
    ("influence_map", "expected_state"),
    [
        ({"schema": "clozn.context_answer_influence.v1", "status": "unavailable", "available": False}, "unavailable"),
        ({"schema": "clozn.context_answer_influence.v1", "status": "error", "available": False}, "error"),
        ({"schema": "clozn.context_answer_influence.v1", "status": "ok", "available": True}, "not_measured"),
    ],
)
def test_unavailable_error_and_malformed_influence_states_remain_distinct(influence_map, expected_state):
    document = _build(_run(response=ANSWER, trace=_trace(), influence_map=influence_map))
    assert document["evidence"]["context_tension"]["state"] == expected_state


def test_partial_parallel_trace_arrays_are_reported_as_partial_coverage():
    trace = _trace(["one", "two", "three"], [0.9, 0.9], [[], []])
    document = _build(_run(response="onetwothree", trace=trace))
    assert document["evidence"]["close_calls"] == {
        "state": "partial",
        "recorded_tokens": 3,
        "analyzed_tokens": 2,
        "reason": "parallel_trace_arrays_incomplete",
        "thresholds": {"margin": close_calls.MARGIN, "min_runnerup": close_calls.MIN_RUNNERUP},
    }
    assert document["analysis"]["state"] == "partially_available"


def test_no_trace_is_unavailable_and_not_a_zero_candidate_result():
    document = _build(_run(response=ANSWER))
    assert document["evidence"]["close_calls"]["state"] == "unavailable"
    assert document["summary"]["candidate_state"] == "unavailable"


# Ordering, identity, limit, and safety ----------------------------------------------------------------


def test_rank_class_order_is_combined_then_meaningful_then_tension_then_ordinary():
    # Position 0 is combined, position 1 is meaningful-only, position 2 is a tension proxy, position 3
    # is an ordinary close call. Separate answer spans keep the evidence in distinct coordinates.
    response = "zero 1 two three"
    tokens = ["zero ", "1 ", "two ", "three"]
    alternatives = [
        [{"piece": "hero ", "prob": 0.39}],
        [{"piece": "2 ", "prob": 0.39}],
        [],
        [{"piece": "four", "prob": 0.39}],
    ]
    confidence = [0.43, 0.43, 0.9, 0.43]
    intervals = [(0, 5), (5, 7), (7, 11), (11, 16)]
    links = [_link("ps-1", "as-2", -3, "supports"), _link("ps-2", "as-2", 2, "suppresses")]
    # Add tension to position 0 as well as position 2, so position 0 is combined and position 2 is proxy.
    links += [_link("ps-1", "as-0", -3, "supports"), _link("ps-2", "as-0", 2, "suppresses")]
    influence = _influence(response, source_ids=("ps-1", "ps-2"), intervals=intervals, links=links)
    document = _build(_run(response=response, trace=_trace(tokens, confidence, alternatives), influence_map=influence))
    assert [item["rank_class"] for item in document["breakpoints"]] == [
        "combined", "meaningful_close_call", "context_tension", "close_call",
    ]


def test_close_call_order_uses_margin_then_rival_probability_then_position():
    response = "alpha beta gamma"
    tokens = ["alpha ", "beta ", "gamma"]
    trace = _trace(
        tokens,
        [0.46, 0.43, 0.43],
        [
            [{"piece": "bravo ", "prob": 0.40}],
            [{"piece": "delta ", "prob": 0.40}],
            [{"piece": "epsilon", "prob": 0.40}],
        ],
    )
    document = _build(_run(response=response, trace=trace))
    assert [item["position"] for item in document["breakpoints"]] == [1, 2, 0]


def test_breakpoint_identity_is_stable_and_depends_only_on_run_id_and_position():
    close = _build(_run(response="answer", trace=_close_trace()))
    tension = _build(_one_tension())
    assert close["breakpoints"][0]["breakpoint_id"] != ""
    assert suggested_breakpoints._breakpoint_id("run-sb", 0) == close["breakpoints"][0]["breakpoint_id"]
    # The tension-only reason can be added later without changing the coordinate identity.
    assert suggested_breakpoints._breakpoint_id("run-sb", 0) == tension["breakpoints"][0]["breakpoint_id"]


def test_limit_is_applied_after_merge_and_ranking():
    response = "zero 1 two"
    tokens = ["zero ", "1 ", "two"]
    trace = _trace(
        tokens,
        [0.43, 0.43, 0.9],
        [[{"piece": "hero ", "prob": 0.39}], [{"piece": "2 ", "prob": 0.39}], []],
    )
    influence = _influence(
        response, intervals=[(0, 5), (5, 7), (7, 10)],
        links=[_link("ps-1", "as-2", -3, "supports"), _link("ps-2", "as-2", 2, "suppresses")],
    )
    document = _build(_run(response=response, trace=trace, influence_map=influence), limit=1)
    assert document["summary"]["suggested_breakpoints"] == 3
    assert document["summary"]["returned_breakpoints"] == 1
    assert document["summary"]["combined_breakpoints"] == 0
    assert document["summary"]["meaningful_close_call_breakpoints"] == 1
    assert document["summary"]["context_tension_breakpoints"] == 1
    assert document["summary"]["ordinary_close_call_breakpoints"] == 1


def test_repeated_build_is_deterministic_and_does_not_mutate_the_run():
    run = _one_tension()
    before = copy.deepcopy(run)
    first = _build(run)
    second = _build(run)
    assert first == second
    assert run == before


def test_artifact_is_metadata_only_and_contains_no_recorded_text():
    run = _one_tension()
    document = _build(run)
    encoded = json.dumps(document, ensure_ascii=False)
    assert "Paris is the capital of France." not in encoded
    assert "PRIVATE SOURCE" not in encoded
    assert "Paris " not in encoded
    assert all("text" not in reason for item in document["breakpoints"] for reason in item["reasons"])


def test_builder_does_not_touch_model_worker_scoring_influence_or_fork_seams(monkeypatch):
    run = _run(response="answer", trace=_close_trace())

    def explode(*_args, **_kwargs):
        raise AssertionError("Suggested Breakpoints reached an execution seam")

    import clozn.receipts.context_answer_influence as influence_module
    import clozn.server.model_routing as routing
    from clozn.server import app as server_app

    monkeypatch.setattr(influence_module, "context_answer_influence", explode)
    monkeypatch.setattr(routing, "select_control_model_for_run", explode)
    for name in ("score_tokens", "generate", "execution_fork", "execution_fork_checkpoint"):
        monkeypatch.setattr(server_app.EngineSubstrate, name, explode, raising=False)

    document = _build(run)
    assert document["breakpoints"]
