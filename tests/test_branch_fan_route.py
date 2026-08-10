"""HTTP contract coverage for POST /runs/<id>/branch-fan."""
from __future__ import annotations

from copy import deepcopy

import pytest

import clozn.runs.store as runlog
from clozn import schemas
from clozn.replay import branch_fan as fan
from clozn.server.routes import branch_fan as route


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


class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


class Sub:
    def __init__(self):
        self.engine = object()


class Selection:
    def __init__(self):
        self.sub = Sub()
        self.runtime_key = deepcopy(RUNTIME)
        self.worker_identity = deepcopy(WORKER)
        self.engine = self.sub.engine


def _run():
    return {
        "id": "run_branch_route",
        "model": "parent-model",
        "trace": {
            "tokens": ["one", " committed", " tail"],
            "token_ids": [1, 2, 3],
            "alternatives": [[], [{"piece": " alt", "token_id": 9, "prob": 0.3}], []],
        },
    }


def _document(*, status="completed", children=1):
    branches = []
    for index in range(children):
        branches.append({
            "recorded_alternative": {"rank": index, "token_id": 9 + index, "probability": 0.3},
            "state": "completed",
            "outcome": "exact_execution_fork",
            "child_run_id": f"child-{index}",
            "execution_fork_execution_id": "fork_exec_" + ("a" * 20),
            "exactness": {"proof_status": "confirmed"},
            "unchanged_control": {"status": "matched"},
            "reasons": [],
            "comparison": {"state": "trace_unavailable", "first_divergence_view": {"state": "trace_unavailable"}},
        })
    return {
        "schema_version": "clozn.branch-fan.v1",
        "parent_run_id": "run_branch_route",
        "position": 1,
        "selection": {"source": "recorded_alternatives", "state": "available",
                       "recorded_alternatives": 1, "selected_alternatives": children,
                       "requested_limit": 3},
        "execution": {"policy": "exact_first", "order": "sequential",
                       "checkpoint_capture": {"state": "available", "reused_for_exact_candidates": True},
                       "fidelity": "all_exact" if children else "none_completed"},
        "branches": branches,
        "summary": {"status": status, "requested_branches": children, "attempted_branches": children,
                    "children_created": children, "exact_children": children, "reconstructed_children": 0,
                    "unavailable_branches": 0, "not_attempted_branches": 0},
    }


@pytest.fixture
def routed(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    seen = {}

    def select(handler, model, *, route):
        seen.update({"model": model, "route": route})
        return Selection()

    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run", select)
    return run, seen


def test_existing_parent_returns_201_and_uses_parent_model(monkeypatch, routed):
    run, seen = routed
    called = {}

    def fake(parent, sub, position, **kwargs):
        called.update({"parent": parent, "sub": sub, "position": position, **kwargs})
        return _document()

    monkeypatch.setattr(fan, "branch_fan", fake)
    h = Handler(f"/runs/{run['id']}/branch-fan")
    assert route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1}) is True
    assert h.status == 201
    schemas.validate(h.body, "clozn.branch-fan.v1")
    assert seen == {"model": "parent-model", "route": "/runs/<id>/branch-fan"}
    assert called["position"] == 1
    assert called["limit"] == 3
    assert called["parent"] is run


def test_parent_worker_unavailable_is_typed_503(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)

    class UnreadySelection:
        sub = None
        runtime_key = None
        worker_identity = None
        engine = None

    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_args, **_kwargs: UnreadySelection())
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1})
    assert h.status == 503
    assert h.body["code"] == "branch_fan_worker_unavailable"


def test_custom_limit_is_forwarded(routed, monkeypatch):
    run, _ = routed
    seen = {}

    def fake(parent, sub, position, **kwargs):
        seen.update(kwargs)
        return _document()

    monkeypatch.setattr(fan, "branch_fan", fake)
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1, "limit": 4})
    assert h.status == 201
    assert seen["limit"] == 4


def test_missing_parent_is_404(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda _rid: None)
    h = Handler()
    assert route.try_post(h, "/runs/missing/branch-fan", {"position": 1}) is True
    assert h.status == 404


@pytest.mark.parametrize("body,code", [
    ({}, "invalid_position"),
    ({"position": -1}, "invalid_position"),
    ({"position": True}, "invalid_position"),
    ({"position": 1.5}, "invalid_position"),
    ({"position": 1, "limit": 0}, "invalid_limit"),
    ({"position": 1, "limit": 5}, "invalid_limit"),
    ({"position": 1, "limit": True}, "invalid_limit"),
    ({"position": 1, "limit": "3"}, "invalid_limit"),
    ({"position": 1, "unexpected": 1}, "invalid_body"),
])
def test_malformed_input_is_typed_400(monkeypatch, body, code):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_args, **_kwargs: pytest.fail("invalid input selected a worker"))
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", body)
    assert h.status == 400
    assert h.body["code"] == code


def test_structurally_valid_position_failure_is_400(monkeypatch, routed):
    run, _ = routed
    def invalid(*_args, **_kwargs):
        raise fan.BranchFanInputError("invalid_position", "position is outside the recorded response token range")
    monkeypatch.setattr(fan, "branch_fan", invalid)
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 99})
    assert h.status == 400
    assert h.body["code"] == "invalid_position"


def test_no_children_is_422_and_cancellation_without_children_is_409(monkeypatch, routed):
    run, _ = routed
    monkeypatch.setattr(fan, "branch_fan", lambda *_args, **_kwargs: _document(status="unavailable", children=0))
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1})
    assert h.status == 422

    cancelled = _document(status="cancelled", children=0)
    monkeypatch.setattr(fan, "branch_fan", lambda *_args, **_kwargs: cancelled)
    h2 = Handler()
    route.try_post(h2, f"/runs/{run['id']}/branch-fan", {"position": 1})
    assert h2.status == 409


def test_no_recorded_alternatives_returns_422_without_worker_selection(monkeypatch):
    run = _run()
    run["trace"]["alternatives"] = [[], [], []]
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr("clozn.server.model_routing.select_control_model_for_run",
                        lambda *_args, **_kwargs: pytest.fail("no-candidate fan selected a worker"))
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1})
    assert h.status == 422
    assert h.body["selection"]["reason"] == "no_recorded_alternatives"


def test_contract_failure_is_sanitized(monkeypatch, routed):
    run, _ = routed
    monkeypatch.setattr(fan, "branch_fan", lambda *_args, **_kwargs: (_ for _ in ()).throw(
        RuntimeError("PRIVATE CHECKPOINT PATH")))
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1})
    assert h.status == 500
    assert h.body == {
        "error": "branch fan response could not be composed",
        "code": "branch_fan_contract_invalid",
    }


def test_route_does_not_start_analysis_or_model_generation(monkeypatch, routed):
    run, _ = routed
    def explode(*_args, **_kwargs):
        raise AssertionError("Branch Fan triggered an unrelated analysis seam")

    import clozn.receipts.context_answer_influence as influence
    monkeypatch.setattr(influence, "context_answer_influence", explode)
    monkeypatch.setattr(fan, "branch_fan", lambda *_args, **_kwargs: _document())
    h = Handler()
    route.try_post(h, f"/runs/{run['id']}/branch-fan", {"position": 1})
    assert h.status == 201


def test_route_is_autoloaded():
    from clozn.server import app
    assert route in app._POST_ROUTES
