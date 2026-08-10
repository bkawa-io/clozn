"""Model-free planning and bounded orchestration coverage for Context Bisect."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest

from clozn import schemas
from clozn.replay import context_bisect as execution
from clozn.runs import context_bisect as planning
from clozn.runs.text_span_addresses import project_influence_addresses


RUNTIME = {
    "model_sha256": "a" * 64,
    "template_fingerprint": "b" * 16,
    "engine_build": "test-build",
    "context_size": 4096,
    "backend": "cpu",
    "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
    "white_box_flags": {},
}
WORKER = {"worker_id": "worker-a", "worker_generation_id": "generation-a", "protocol_version": "1.1"}


def _run():
    source = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen"
    answer = "Paris is the capital of France."
    influence = {
        "schema": "clozn.context_answer_influence.v1",
        "status": "ok",
        "available": True,
        "method": {"name": "teacher_forced_matched_context_replacement", "mode": "forced_score_intervention", "claim_limit": "no percentage claim", "caveat": "measured effect only"},
        "identity": {"model_sha256": "a" * 64},
        "prompt_sources": [{"id": "source-0", "text": source, "client_source_id": "doc-1", "selected": True}],
        "prompt_spans": [{"id": "source-fine", "parent_id": "source-0", "level": "fine", "start": 0, "end": len(source), "text": source, "client_source_id": "doc-1"}],
        "answer": {"scored_text": answer},
        "answer_spans": [{"id": "answer-0", "start": 0, "end": len(answer), "text": answer}],
        "links": [{"context_span_id": "source-fine", "answer_span_id": "answer-0", "context_index": 0, "answer_index": 0, "delta_nats": -1.2, "abs_delta_nats": 1.2, "effect": "supports", "clears_floor": True, "evidence_state": "causally_supported"}],
        "thresholds": {"cell_abs_delta_nats": 0.5},
        "matrix_complete": True,
        "selection": {"complete_for_selected_spans": True, "selected_source_ids": ["source-0"]},
    }
    tokens = ["Paris ", "is ", "the ", "capital ", "of ", "France."]
    return {
        "id": "run-bisect",
        "model": "parent-model",
        "messages": [{"role": "user", "content": source, "source_id": "doc-1"}],
        "response": answer,
        "trace": {"tokens": tokens, "token_ids": [1, 2, 3, 4, 5, 6]},
        "meta": {"decode": {"mode": "greedy", "temperature": 0}},
        "identity": deepcopy(RUNTIME),
        "influence_map": influence,
    }


def _request(run, **overrides):
    addresses = project_influence_addresses(run["id"], run["influence_map"])
    source = next(a["address_id"] for a in addresses if a["native_ref"].get("id") == "source-fine")
    answer = next(a["address_id"] for a in addresses if a["native_ref"].get("id") == "answer-0")
    request = {"influence": {"source_span_id": source, "answer_span_id": answer}}
    request.update(overrides)
    return request


def test_split_region_is_nearest_whitespace_and_exact_partition():
    text = "aaaaa bbbbb ccccc ddddd"
    left, right = planning.split_region(text, 0, len(text), 5)
    assert left["end"] == right["start"]
    assert left["start"] == 0 and right["end"] == len(text)
    assert text[left["start"]:left["end"]] + text[right["start"]:right["end"]] == text


def test_split_region_tie_chooses_lower_boundary_and_unsplittable_does_not_cut_words():
    assert planning.split_region("aaaa bbbb cccc", 0, 14, 4)[0]["end"] == 5
    assert planning.split_region("abcdefgh", 0, 8, 2) is None


def test_plan_copies_link_freshly_resolves_span_and_is_metadata_only():
    run = _run()
    before = deepcopy(run)
    plan = planning.plan_context_bisect(run, request=_request(run))
    schemas.validate(plan, planning.PLAN_SCHEMA_VERSION)
    assert plan["execution"]["state"] == "ready"
    assert plan["intervention"] == {"kind": "neutralize", "recipe": planning.FILLER_RECIPE}
    assert plan["influence"]["delta_nats"] == -1.2
    assert plan["span_resolution"]["state"] == "available"
    assert plan["root_region"]["code_points"] == len(run["messages"][0]["content"])
    assert run == before
    encoded = json.dumps(plan)
    assert run["messages"][0]["content"] not in encoded
    assert run["response"] not in encoded


def test_plan_identity_excludes_execution_budgets():
    run = _run()
    first = planning.plan_context_bisect(run, request=_request(run, max_runs=8, max_seconds=120))
    second = planning.plan_context_bisect(run, request=_request(run, max_runs=12, max_seconds=300))
    assert first["search_id"] == second["search_id"]


def test_missing_measurement_is_not_root_not_reproduced():
    run = _run()
    run.pop("influence_map")
    plan = planning.plan_context_bisect(run, request={"influence": {"source_span_id": "span_x", "answer_span_id": "span_y"}})
    assert plan["execution"]["state"] == "unavailable"
    assert plan["execution"]["reason"] == "no_influence_map"


def test_observed_link_remains_eligible():
    run = _run()
    run["influence_map"]["links"][0]["evidence_state"] = "observed"
    run["influence_map"]["links"][0]["clears_floor"] = False
    plan = planning.plan_context_bisect(run, request=_request(run))
    assert plan["execution"]["state"] == "ready"
    assert plan["influence"]["evidence_state"] == "observed"


def test_control_and_root_are_sibling_arms_and_budget_is_shared(monkeypatch):
    run = _run()
    request = _request(run, max_depth=0, max_runs=2)
    plan = planning.plan_context_bisect(run, request=request)
    calls = []

    def fake_replay(parent, changes, sub, **kwargs):
        calls.append((parent, deepcopy(changes), deepcopy(kwargs["messages_override"])))
        is_control = changes["context_bisect"]["arm"] == "control"
        response = parent["response"] if is_control else "Different trajectory."
        tokens = ["Paris ", "is ", "the ", "capital ", "of ", "France."] if is_control else ["Different", " trajectory."]
        return {"id": f"child-{len(calls)}", "parent_run_id": parent["id"], "response": response,
                "messages": deepcopy(kwargs["messages_override"]),
                "trace": {"tokens": tokens, "token_ids": list(range(1, len(tokens) + 1))}}

    monkeypatch.setattr(execution, "replay_run", fake_replay)
    monkeypatch.setattr(execution.counterfactual, "_runtime_match", lambda *_args: (True, None))
    result = execution.execute_context_bisect(
        run, object(), request, runtime_identity=RUNTIME, worker_identity=WORKER, plan=plan)
    schemas.validate(result, planning.RESULT_SCHEMA_VERSION)
    assert result["execution"]["children_created"] == 2
    assert result["control"]["run_id"] == "child-1"
    assert result["regions"][0]["child_run_id"] == "child-2"
    assert all(call[0]["id"] == run["id"] for call in calls)
    assert all(call[1]["context_bisect"]["search_id"] == plan["search_id"] for call in calls)
    assert calls[1][2][0]["content"] != calls[0][2][0]["content"]
    assert result["coverage"]["state"] == "complete_within_limits"


def test_cancellation_preserves_control_child(monkeypatch):
    run = _run()
    request = _request(run, max_depth=0, max_runs=2)
    calls = []

    def fake_replay(parent, changes, sub, **kwargs):
        calls.append(changes["context_bisect"]["arm"])
        return {"id": "child-control", "parent_run_id": parent["id"], "response": parent["response"],
                "trace": deepcopy(parent["trace"])}

    monkeypatch.setattr(execution, "replay_run", fake_replay)
    monkeypatch.setattr(execution.counterfactual, "_runtime_match", lambda *_args: (True, None))
    result = execution.execute_context_bisect(
        run, object(), request, runtime_identity=RUNTIME, worker_identity=WORKER,
        cancel_check=lambda: True)
    assert result["execution"]["children_created"] == 0
    assert result["coverage"]["state"] == "cancelled"
    assert calls == []
