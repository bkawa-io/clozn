"""Model-free coverage for Influence -> Counterfactual Confirmation."""
from __future__ import annotations

from copy import deepcopy
import json

from clozn import schemas
from clozn.replay import influence_counterfactual as execution
from clozn.runs import influence_counterfactual as planning
from clozn.runs.text_span_addresses import project_influence_addresses


ANSWER = "Paris is the capital of France."
ANSWER_TOKENS = ["Paris ", "is ", "the ", "capital ", "of ", "France."]
SOURCE = "x" * 80


def _link(*, evidence_state="causally_supported", effect="supports", delta=-1.2,
          clears_floor=True):
    return {
        "context_span_id": "source-fine",
        "answer_span_id": "answer-0",
        "context_index": 0,
        "answer_index": 0,
        "delta_nats": delta,
        "abs_delta_nats": abs(delta),
        "effect": effect,
        "clears_floor": clears_floor,
        "evidence_state": evidence_state,
    }


def _influence(*, link=None, status="ok", available=True):
    out = {
        "schema": "clozn.context_answer_influence.v1",
        "status": status,
        "available": available,
        "method": {
            "name": "teacher_forced_matched_context_replacement",
            "mode": "forced_score_intervention",
            "claim_limit": "no percentage claim",
            "caveat": "measured effect only, not correctness",
        },
        "identity": {"model_sha256": "a" * 64},
    }
    if status != "ok":
        out["error"] = {"code": "no_text_context", "message": "no text context"}
        return out
    out.update({
        "prompt_sources": [{
            "id": "source-0", "text": SOURCE, "client_source_id": "doc-1", "selected": True,
        }],
        "prompt_spans": [{
            "id": "source-fine", "parent_id": "source-0", "level": "fine",
            "start": 10, "end": 20, "text": SOURCE[10:20], "client_source_id": "doc-1",
        }],
        "answer": {"scored_text": ANSWER},
        "answer_spans": [{"id": "answer-0", "start": 0, "end": len(ANSWER), "text": ANSWER}],
        "links": [_link() if link is None else link],
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "matrix_complete": True,
        "selection": {"complete_for_selected_spans": True, "selected_source_ids": ["source-0"]},
    })
    return out


def _run(*, influence=None):
    return {
        "id": "run-ict",
        "model": "parent-model",
        "messages": [{"role": "user", "content": SOURCE, "source_id": "doc-1"}],
        "response": ANSWER,
        "trace": {"tokens": list(ANSWER_TOKENS), "token_ids": [1, 2, 3, 4, 5, 6]},
        "meta": {"decode": {"mode": "greedy"}},
        **({"influence_map": influence} if influence is not None else {}),
    }


def _ids(run):
    addresses = project_influence_addresses(run["id"], run["influence_map"])
    return (
        next(a["address_id"] for a in addresses if a["native_ref"].get("id") == "source-fine"),
        next(a["address_id"] for a in addresses if a["native_ref"].get("id") == "answer-0"),
    )


def _request(run, *, kind="neutralize", specificity=False):
    source_id, answer_id = _ids(run)
    return {
        "influence": {"source_span_id": source_id, "answer_span_id": answer_id},
        "intervention": {"kind": kind},
        "specificity_control": specificity,
    }


def test_plan_copies_measured_link_and_is_metadata_only():
    run = _run(influence=_influence())
    before = deepcopy(run)
    plan = planning.build_influence_counterfactual_plan(run, _request(run))
    schemas.validate(plan, "clozn.influence-counterfactual-plan.v1")
    assert plan["influence"]["effect"] == "supports"
    assert plan["influence"]["delta_nats"] == -1.2
    assert plan["influence"]["evidence_state"] == "causally_supported"
    assert plan["intervention"] == {
        "kind": "neutralize",
        "relation_to_measurement": "same_intervention",
        "recipe": "clozn.matched_length_neutral_filler.v1",
    }
    assert SOURCE not in json.dumps(plan)
    assert ANSWER not in json.dumps(plan)
    assert run == before


def test_remove_is_explicitly_a_different_intervention():
    run = _run(influence=_influence())
    plan = planning.build_influence_counterfactual_plan(run, _request(run, kind="remove"))
    assert plan["intervention"] == {"kind": "remove", "relation_to_measurement": "different_intervention"}


def test_observed_below_floor_link_remains_testable():
    run = _run(influence=_influence(link=_link(evidence_state="observed", clears_floor=False, delta=-0.1)))
    plan = planning.build_influence_counterfactual_plan(run, _request(run))
    assert plan["influence"]["measurement_state"] == "available"
    assert plan["influence"]["evidence_state"] == "observed"
    assert plan["influence"]["clears_floor"] is False
    assert plan["execution"]["state"] == "ready"


def test_missing_measurement_is_distinct_from_unavailable_span():
    run = _run()
    plan = planning.build_influence_counterfactual_plan(run, {
        "influence": {"source_span_id": "span_missing", "answer_span_id": "span_answer"},
    })
    assert plan["influence"]["measurement_state"] == "not_measured"
    assert plan["execution"]["state"] == "unavailable"


def test_stale_source_address_fails_closed_but_measurement_remains_available():
    run = _run(influence=_influence())
    source_id, answer_id = _ids(run)
    stale = deepcopy(run)
    stale["messages"][0]["content"] = "changed" + SOURCE
    plan = planning.build_influence_counterfactual_plan(stale, {
        "influence": {"source_span_id": source_id, "answer_span_id": answer_id},
    })
    assert plan["influence"]["measurement_state"] == "available"
    assert plan["span_resolution"]["state"] == "unavailable"
    assert plan["execution"]["state"] == "unavailable"


def test_plan_is_deterministic_and_id_does_not_use_measurement_values():
    run = _run(influence=_influence())
    request = _request(run)
    first = planning.build_influence_counterfactual_plan(run, request)
    second = planning.build_influence_counterfactual_plan(run, request)
    assert first == second
    assert first["test_id"].startswith("ict_") and len(first["test_id"]) == 28


def test_execution_generates_control_and_treatment_as_sibling_children(monkeypatch):
    run = _run(influence=_influence())
    request = _request(run)
    calls = []

    def fake_replay(parent, changes, sub, **kwargs):
        calls.append((parent, changes, kwargs["messages_override"]))
        if len(calls) == 1:
            response, tokens = ANSWER, ANSWER_TOKENS
        else:
            response = "Paris is not the capital of France."
            tokens = ["Paris ", "is ", "not ", "the ", "capital ", "of ", "France."]
        return {
            "id": f"child-{len(calls)}",
            "parent_run_id": parent["id"],
            "messages": deepcopy(kwargs["messages_override"]),
            "response": response,
            "trace": {"tokens": tokens, "token_ids": list(range(1, len(tokens) + 1))},
        }

    monkeypatch.setattr(execution, "replay_run", fake_replay)
    monkeypatch.setattr(execution, "_runtime_match", lambda *_args: (True, None))
    result = execution.execute_influence_counterfactual(
        run, object(), request, runtime_identity={}, worker_identity={},
        plan=planning.build_influence_counterfactual_plan(run, request),
    )
    schemas.validate(result, "clozn.influence-counterfactual.v1")
    assert result["execution"]["status"] == "completed"
    assert result["control_reproduction"]["state"] == "exact_token_and_text"
    assert result["arms"]["control"]["run_id"] == "child-1"
    assert result["arms"]["treatment"]["run_id"] == "child-2"
    assert all(call[0] is run for call in calls)
    assert all(call[1]["influence_counterfactual"]["test_id"] == result["test_id"] for call in calls)
    assert calls[0][2][0]["content"] == SOURCE
    assert calls[1][2][0]["content"] != SOURCE
    assert result["observation"]["state"] in {
        "recorded_answer_changed_before_or_within_target", "controlled_sensitivity_observed"
    }
    assert ANSWER not in json.dumps(result)
    assert "Paris is not the capital of France." not in json.dumps(result)


def test_execution_never_runs_specificity_control_when_not_requested(monkeypatch):
    run = _run(influence=_influence())
    request = _request(run, specificity=False)
    monkeypatch.setattr(execution, "_runtime_match", lambda *_args: (True, None))
    monkeypatch.setattr(execution, "replay_run", lambda parent, changes, sub, **kwargs: {
        "id": "child", "parent_run_id": parent["id"], "response": ANSWER,
        "trace": {"tokens": ANSWER_TOKENS, "token_ids": [1, 2, 3, 4, 5, 6]},
    })
    result = execution.execute_influence_counterfactual(run, object(), request,
                                                         runtime_identity={}, worker_identity={})
    assert result["arms"]["specificity_control"]["state"] == "not_attempted"
    assert result["execution"]["children_created"] == 2
