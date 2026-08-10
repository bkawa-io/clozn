"""HTTP behavior for creation and GET-by-reference selection routes."""
from __future__ import annotations

import copy

import clozn.runs.store as runlog
from clozn.runs.selection_reference import encode_selection_reference
from clozn.server.routes import selection_inspection, selection_reference
from tests.test_selection_inspection import _run


class Handler:
    def __init__(self, path=""):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_post_reference_is_read_only_and_get_round_trips(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    post = Handler()
    assert selection_reference.try_post(
        post, "/runs/run-inspect/selection/reference",
        {"selection": {"kind": "response_token", "position": 0}},
    ) is True
    assert post.status == 200
    ref = post.body["reference"]

    get = Handler(f"/runs/run-inspect/selection/inspect?ref={ref}")
    assert selection_inspection.try_get(get, "/runs/run-inspect/selection/inspect") is True
    assert get.status == 200
    assert get.body["selection"]["kind"] == "response_token"
    assert get.body["reference"]["selection_ref"] == ref


def test_post_reference_malformed_and_unbindable_statuses(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    malformed = Handler()
    selection_reference.try_post(
        malformed, "/runs/run-inspect/selection/reference", {"selection": {"kind": "unknown"}},
    )
    assert malformed.status == 400
    assert malformed.body["code"] == "invalid_selection_kind"

    unavailable = Handler()
    selection_reference.try_post(
        unavailable, "/runs/run-inspect/selection/reference",
        {"selection": {"kind": "context_span", "source_span_id": "span_" + "f" * 24}},
    )
    assert unavailable.status == 422


def test_get_malformed_stale_and_missing_reference(monkeypatch):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: None if rid == "missing" else run)

    missing = Handler()
    selection_inspection.try_get(missing, "/runs/missing/selection/inspect")
    assert missing.status == 404

    malformed = Handler("/runs/run-inspect/selection/inspect?ref=bad")
    selection_inspection.try_get(malformed, "/runs/run-inspect/selection/inspect")
    assert malformed.status == 400

    reference = encode_selection_reference(run, {"kind": "response_token", "position": 0})["reference"]
    changed = copy.deepcopy(run)
    changed["trace"]["token_ids"][0] = 500
    monkeypatch.setattr(runlog, "get_run", lambda _rid: changed)
    stale = Handler(f"/runs/run-inspect/selection/inspect?ref={reference}")
    selection_inspection.try_get(stale, "/runs/run-inspect/selection/inspect")
    assert stale.status == 409
    assert stale.body["code"] == "parent_execution_changed"


def test_route_modules_are_autoloaded():
    from clozn.server import app
    assert selection_reference in app._POST_ROUTES
    assert selection_inspection in app._GET_ROUTES
