"""HTTP contract tests for Sampler Sensitivity planning and execution."""
from __future__ import annotations

import pytest

import clozn.runs.store as runlog
from clozn.server.routes import sampler_sensitivity as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def parent():
    return {
        "id": "run_sampler_route", "model": "parent-model",
        "trace": {"tokens": ["a", "b"], "token_ids": [1, 2]},
        "meta": {"decode": {"mode": "sample", "temperature": 0.7, "top_p": 0.9,
                              "top_k": 40, "repeat_penalty": 1.0, "seed": 12}},
    }


def test_plan_is_read_only_and_defaults_are_explicit(monkeypatch):
    run = parent()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_a, **_k: pytest.fail("planning selected a worker"))
    h = Handler()
    assert route.try_post(h, "/runs/run_sampler_route/sampler-sensitivity/plan", {}) is True
    assert h.status == 200
    assert h.body["position"] == 0
    assert h.body["recipe"]["id"] == "nearby_v1"
    assert h.body["execution"]["live_state"] == "not_checked"


@pytest.mark.parametrize("body,code", [
    ({"position": -1}, "invalid_position"),
    ({"position": True}, "invalid_position"),
    ({"recipe": "nearby_v2"}, "invalid_recipe"),
    ({"seed_probes": 3}, "invalid_seed_probes"),
    ({"unexpected": 1}, "invalid_body"),
])
def test_invalid_plan_input_is_400_and_does_not_route(monkeypatch, body, code):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: parent())
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_a, **_k: pytest.fail("invalid input selected a worker"))
    h = Handler()
    route.try_post(h, "/runs/run_sampler_route/sampler-sensitivity/plan", body)
    assert h.status == 400
    assert h.body["code"] == code


def test_missing_parent_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    route.try_post(h, "/runs/missing/sampler-sensitivity/plan", {})
    assert h.status == 404


def test_greedy_execute_is_typed_422_without_worker(monkeypatch):
    run = parent()
    run["meta"]["decode"] = {"mode": "greedy", "temperature": 0.0}
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_a, **_k: pytest.fail("unavailable plan selected a worker"))
    h = Handler()
    route.try_post(h, "/runs/run_sampler_route/sampler-sensitivity", {})
    assert h.status == 422
    assert h.body["execution"]["reason"] == "greedy_baseline_no_sampling_neighborhood"


def test_route_is_autoloaded():
    from clozn.server import app
    assert route in app._POST_ROUTES
