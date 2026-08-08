from __future__ import annotations

from clozn.replay import corrective
from clozn.server.routes import corrective_retries as route
import clozn.runs.store as runlog


class Handler:
    def __init__(self):
        self._inj_sub = type("Sub", (), {"chat": lambda *_args, **_kwargs: "ok"})()
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def comparison():
    return {
        "preset": "less-verbose", "baseline_reply": "long",
        "corrected_reply": "short", "changed": True, "coherence": {"degenerate": False},
        "intervention_observed": True,
    }


def test_retry_returns_compare_with_automatic_undo(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    monkeypatch.setattr(corrective, "retry_compare",
                        lambda run, preset, sub, backend=None: comparison())
    handler = Handler()
    assert route.try_post(handler, "/runs/run_x/retry", {"preset": "less-verbose"})
    assert handler.status == 200
    assert handler.body["undo"]["status"] == "automatic_restored"
    assert handler.body["policy"] == {
        "status": "request_local", "scope": "once", "target": None, "presets": ["less-verbose"],
    }


def test_retry_leaves_no_scope_option(monkeypatch):
    """A request-local retry never has a scope to choose -- there is no --scope/session/profile
    concept left in the wire request or the retry_compare call it drives."""
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [], "identity": {}, "session_key": "session_exact",
    })
    seen = {}

    def fake_retry_compare(run, preset, sub, backend=None):
        seen["called_with"] = {"preset": preset, "backend": backend}
        return comparison()
    monkeypatch.setattr(corrective, "retry_compare", fake_retry_compare)
    handler = Handler()
    # A caller-supplied "scope" is simply ignored -- the route has no scope parameter to read it into.
    route.try_post(handler, "/runs/run_x/retry", {"preset": "less-verbose", "scope": "session"})
    assert handler.status == 200
    assert seen["called_with"] == {"preset": "less-verbose", "backend": None}
    assert handler.body["policy"]["scope"] == "once"


def test_bad_backend_value_is_a_clean_400(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry",
                   {"preset": "less-verbose", "backend": "made_up"})
    assert handler.status == 400
    assert "backend" in handler.body["error"]


def test_omitted_backend_is_not_forwarded_as_a_string(monkeypatch):
    """An absent `backend` key must reach retry_compare as None (the "unchanged default" sentinel),
    never as the literal string "None" or an empty string."""
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    seen = {}

    def fake_retry_compare(run, preset, sub, backend=None):
        seen["backend"] = backend
        return comparison()
    monkeypatch.setattr(corrective, "retry_compare", fake_retry_compare)
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry", {"preset": "less-verbose"})
    assert handler.status == 200
    assert seen["backend"] is None


def test_explicit_control_vector_backend_is_forwarded(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    seen = {}

    def fake_retry_compare(run, preset, sub, backend=None):
        seen["backend"] = backend
        return comparison()
    monkeypatch.setattr(corrective, "retry_compare", fake_retry_compare)
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry",
                   {"preset": "less-verbose", "backend": "control_vector"})
    assert handler.status == 200
    assert seen["backend"] == "control_vector"


def test_no_undo_route_survives_for_persistent_activation():
    """`/corrective-retries/<id>/undo` only ever existed to reverse a persistent session/profile
    activation; with that gone, the route has nothing left to dispatch to."""
    handler = Handler()
    assert route.try_post(handler, "/corrective-retries/repair_x/undo", {}) is False
