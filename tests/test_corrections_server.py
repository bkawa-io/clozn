"""HTTP contract tests for the F5/F6 correction-store adapter."""
from __future__ import annotations

import clozn.runs.store as store
from clozn.server.routes import corrections as route


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_correction_lifecycle_and_resolution_routes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    store._schema_verified.clear()

    h = Handler()
    assert route.try_post(h, "/corrections", {
        "scope_kind": "session", "scope_value": "route-session",
        "type": "style", "content": "Answer in bullets.",
    })
    assert h.status == 201
    correction_id = h.body["id"]
    assert h.body["enabled"] is False

    h = Handler()
    assert route.try_post(h, f"/corrections/{correction_id}/confirm", {})
    assert h.status == 200
    assert h.body["correction"]["enabled"] is True

    h = Handler()
    assert route.try_post(h, "/corrections/resolve", {"session_id": "route-session"})
    assert h.status == 200
    assert h.body["applied"][0]["correction_id"] == correction_id

    h = Handler()
    assert route.try_get(h, f"/corrections/{correction_id}/export")
    assert h.status == 200
    assert h.body["correction_id"] == correction_id
    assert h.body["events"][-1]["event_type"] == "confirmed"

    h = Handler()
    assert route.try_post(h, f"/corrections/{correction_id}/disable", {})
    assert h.status == 200
    assert h.body["enabled"] is False

    h = Handler()
    assert route.try_post(h, f"/corrections/{correction_id}/undo", {})
    assert h.status == 200
    assert h.body["enabled"] is True


def test_correction_route_keeps_draft_inert_and_typed_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    store._schema_verified.clear()

    h = Handler()
    assert route.try_post(h, "/corrections", {
        "scope_kind": "global_local", "type": "style", "content": "Draft only",
    })
    correction_id = h.body["id"]
    h = Handler()
    assert route.try_post(h, "/corrections/resolve", {})
    assert h.status == 200
    assert h.body["applied"] == []

    h = Handler()
    assert route.try_post(h, f"/corrections/{correction_id}/disable", {})
    assert h.status == 409
    assert h.body["type"] == "CorrectionStateError"

    h = Handler()
    assert route.try_post(h, "/corrections/corr_not-real/confirm", {})
    assert h.status == 400


def test_teaching_loop_verify_route_records_the_exact_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "RUNS_DIR", str(tmp_path / "runs"))
    store._schema_verified.clear()
    from clozn.runs import corrections

    draft = corrections.draft_correction(
        scope_kind="session", scope_value="route-verify", correction_type="style",
        content="Use short paragraphs.",
    )
    target = store.record(source="test", messages=[{"role": "user", "content": "hi"}], response="bad")
    child = store.record(source="test", messages=[{"role": "user", "content": "hi"}], response="fixed")
    h = Handler()
    assert route.try_post(h, f"/corrections/{draft['id']}/verify", {
        "target_run_id": target,
        "child_run_id": child,
        "match_criterion": "exact_output",
    })
    assert h.status == 200
    assert h.body["verification"] == "passed"
    assert h.body["promoted"] is True
    assert h.body["target_run_id"] == target
    assert h.body["child_run_id"] == child
