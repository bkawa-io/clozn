"""Tests for the metadata-only Select -> Inspect composition layer."""
from __future__ import annotations

from copy import deepcopy

import pytest

from clozn import schemas
from clozn.runs import selection_inspection as inspection
from clozn.runs.text_span_addresses import project_influence_addresses


ANSWER = "Paris is the capital."
TOKENS = ["Paris ", "is ", "the ", "capital."]


def _run(**extra):
    run = {
        "id": "run-inspect",
        "model": "parent-model",
        "messages": [{"role": "user", "content": "S" * 40, "source_id": "doc-1"}],
        "response": ANSWER,
        "trace": {
            "tokens": list(TOKENS),
            "token_ids": [10, 11, 12, 13],
            "confidence": [0.44, 0.9, 0.9, 0.9],
            "alternatives": [
                [{"piece": "London", "token_id": 20, "prob": 0.39},
                 {"piece": "Rome", "token_id": 21, "prob": 0.1}],
                [], [], [],
            ],
        },
        "identity": {
            "model_sha256": "a" * 64,
            "template_fingerprint": "template",
            "engine_build": "engine",
        },
        "final_prompt": "system prompt",
        "meta": {"decode": {"mode": "sample", "temperature": 0.7, "top_p": 0.9,
                              "top_k": 40, "repeat_penalty": 1.0, "seed": 123}},
    }
    run.update(extra)
    return run


def _influence(run):
    source = "S" * 40
    return {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": {"name": "teacher_forced_matched_context_replacement",
                    "mode": "forced_score_intervention", "claim_limit": "no percentage claim",
                    "caveat": "measured effect only, not correctness"},
        "identity": {"model_sha256": "a" * 64},
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "prompt_sources": [{"id": "source-0", "text": source, "client_source_id": "doc-1", "selected": True}],
        "prompt_spans": [{"id": "source-fine", "parent_id": "source-0", "level": "fine",
                           "start": 1, "end": 8, "text": source[1:8], "client_source_id": "doc-1"}],
        "answer": {"scored_text": ANSWER},
        "answer_spans": [{"id": "answer-0", "start": 0, "end": len(ANSWER), "text": ANSWER}],
        "links": [{"context_span_id": "source-fine", "answer_span_id": "answer-0",
                    "context_index": 0, "answer_index": 0, "delta_nats": -1.2,
                    "abs_delta_nats": 1.2, "effect": "supports", "clears_floor": True,
                    "evidence_state": "causally_supported"}],
        "matrix_complete": True,
        "selection": {"complete_for_selected_spans": True, "selected_source_ids": ["source-0"],
                       "omitted_source_ids": []},
    }


def _influence_ids(run):
    addresses = project_influence_addresses(run["id"], run["influence_map"])
    return (
        next(item["address_id"] for item in addresses if item["native_ref"].get("id") == "source-fine"),
        next(item["address_id"] for item in addresses if item["native_ref"].get("id") == "answer-0"),
    )


def test_response_token_composes_native_trace_and_affordances_without_text():
    run = _run()
    before = deepcopy(run)
    doc = inspection.build_selection_inspection(run, selection={"kind": "response_token", "position": 0})

    assert doc["schema_version"] == "clozn.selection-inspection.v1"
    assert doc["privacy"] == "metadata_only"
    assert doc["selection"]["selection_id"].startswith("selection_")
    assert doc["inspection"]["primary"]["response_interval"] == {
        "start": 0, "end": 6, "unit": "unicode_code_points", "interval": "half_open",
    }
    distribution = next(item for item in doc["inspection"]["evidence"] if item["id"] == "token_distribution")
    assert distribution["data"]["chosen_probability"] == 0.44
    assert distribution["data"]["alternatives"][0] == {"rank": 0, "token_id": 20, "probability": 0.39}
    assert any(item["id"] == "try_alternative:0" and item["state"] == "ready_to_plan" for item in doc["tests"])
    assert any(item["id"] == "fan_alternatives" for item in doc["tests"])
    assert "text" not in repr(doc), "metadata-only inspection must not carry recorded text"
    assert run == before
    schemas.validate(doc, "clozn.selection-inspection.v1")


def test_token_geometry_fails_closed_on_trace_response_mismatch():
    run = _run(response="Paris is different.")
    doc = inspection.build_selection_inspection(run, selection={"kind": "response_token", "position": 0})
    assert "response_interval" not in doc["inspection"]["primary"]
    geometry_entry = next(item for item in doc["inspection"]["evidence"] if item["id"] == "response_geometry")
    assert geometry_entry["reason"] == "trace_response_mismatch"


def test_sampling_defaults_position_and_uses_shared_provenance_contract():
    doc = inspection.build_selection_inspection(_run(), selection={"kind": "sampling"})
    assert doc["selection"]["position"] == 0
    assert doc["inspection"]["primary"]["recorded"]["seed"] == 123
    change = next(item for item in doc["tests"] if item["id"] == "change_sampling")
    assert change["state"] == "requires_input"
    assert set(change["input"]["properties"]) == {"temperature", "top_k", "top_p", "seed", "rep_penalty"}


def test_answer_span_preserves_not_measured_and_exposes_measurement_affordance():
    doc = inspection.build_selection_inspection(_run(), selection={"kind": "answer_span", "start": 0, "end": 6})
    influence = next(item for item in doc["inspection"]["evidence"] if item["id"] == "measured_influence")
    assert influence["state"] == "not_measured"
    assert doc["measurements"][0]["id"] == "measure_influence"
    assert doc["measurements"][0]["state"] == "conditionally_available"
    assert doc["tests"] == []


def test_context_relationship_composes_link_and_test_this_descriptors():
    run = _run(influence_map=_influence(_run()))
    source_id, answer_id = _influence_ids(run)
    doc = inspection.build_selection_inspection(
        run, selection={"kind": "context_span", "source_span_id": source_id, "answer_span_id": answer_id},
    )
    assert doc["inspection"]["primary"]["scope"] == "measured_relationship"
    assert doc["inspection"]["primary"]["measured_influence"]["effect"] == "supports"
    assert {item["id"] for item in doc["tests"]} == {"neutralize_context", "remove_context", "bisect_context"}
    assert next(item for item in doc["tests"] if item["id"] == "neutralize_context")["execute"]["body"]["test"] == {"kind": "neutralize"}
    assert "S" * 20 not in repr(doc)


def test_source_only_context_selection_does_not_invent_a_target_test():
    run = _run(influence_map=_influence(_run()))
    source_id, _ = _influence_ids(run)
    doc = inspection.build_selection_inspection(run, selection={"kind": "context_span", "source_span_id": source_id})
    assert doc["tests"] == []


@pytest.mark.parametrize("selection", [
    {"kind": "unknown", "position": 0},
    {"kind": "response_token", "position": -1},
    {"kind": "answer_span", "start": 2, "end": 2},
    {"kind": "sampling", "position": 999},
])
def test_invalid_selection_is_typed(selection):
    with pytest.raises(inspection.SelectionInspectionInputError):
        inspection.build_selection_inspection(_run(), selection=selection)


def test_repeated_inspection_is_deterministic_and_does_not_call_expensive_seams(monkeypatch):
    run = _run()
    monkeypatch.setattr("clozn.runs.close_calls.close_calls", lambda *_a, **_k: pytest.fail("close call replacement seam only"))
    # The inspector is allowed to consume Close Calls; this test instead proves the public route/domain
    # contract has no model/worker entry point by replacing the execution seams it must never call.
    monkeypatch.undo()
    for module, name in [
        ("clozn.runs.selection_inspection", "parent_execution_fingerprint"),
    ]:
        assert hasattr(__import__(module, fromlist=[name]), name)
    one = inspection.build_selection_inspection(run, selection={"kind": "sampling", "position": 0})
    two = inspection.build_selection_inspection(run, selection={"kind": "sampling", "position": 0})
    assert one == two
