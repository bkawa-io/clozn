from __future__ import annotations

from clozn.behavior import corrective_retries as policy
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


def comparison(scope="once"):
    return {
        "preset": "less-verbose", "scope": scope, "baseline_reply": "long",
        "corrected_reply": "short", "changed": True, "coherence": {"degenerate": False},
        "intervention_observed": True,
    }


def test_once_retry_returns_compare_with_automatic_undo(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    monkeypatch.setattr(policy, "effective_presets", lambda **_kwargs: [])
    monkeypatch.setattr(corrective, "retry_compare",
                        lambda run, preset, sub, scope, active_presets, backend=None: comparison(scope))
    handler = Handler()
    assert route.try_post(handler, "/runs/run_x/retry",
                          {"preset": "less-verbose", "scope": "once"})
    assert handler.status == 200
    assert handler.body["undo"]["status"] == "automatic_restored"
    assert handler.body["policy"]["status"] == "request_local"


def test_session_retry_activates_only_exact_run_session(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [], "identity": {}, "session_key": "session_exact",
    })
    monkeypatch.setattr(policy, "effective_presets", lambda **_kwargs: [])
    monkeypatch.setattr(corrective, "retry_compare",
                        lambda run, preset, sub, scope, active_presets, backend=None: comparison(scope))
    calls = []
    monkeypatch.setattr(policy, "activate", lambda scope, target, preset: (
        calls.append((scope, target, preset)) or {
            "status": "activated", "scope": scope, "target": target,
            "presets": [preset], "undo_id": "repair_x",
        }
    ))
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry",
                   {"preset": "less-verbose", "scope": "session"})
    assert handler.status == 200
    assert calls == [("session", "session_exact", "less-verbose")]
    assert handler.body["undo"] == {"status": "available", "available": True, "id": "repair_x"}


def test_bad_backend_value_is_a_clean_400(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry",
                   {"preset": "less-verbose", "scope": "once", "backend": "made_up"})
    assert handler.status == 400
    assert "backend" in handler.body["error"]


def test_omitted_backend_is_not_forwarded_as_a_string(monkeypatch):
    """An absent `backend` key must reach retry_compare as None (the "unchanged default" sentinel),
    never as the literal string "None" or an empty string."""
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    monkeypatch.setattr(policy, "effective_presets", lambda **_kwargs: [])
    seen = {}

    def fake_retry_compare(run, preset, sub, scope, active_presets, backend=None):
        seen["backend"] = backend
        return comparison(scope)
    monkeypatch.setattr(corrective, "retry_compare", fake_retry_compare)
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry", {"preset": "less-verbose", "scope": "once"})
    assert handler.status == 200
    assert seen["backend"] is None


def test_explicit_control_vector_backend_is_forwarded(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: {
        "id": rid, "messages": [{"role": "user", "content": "x"}], "identity": {},
    })
    monkeypatch.setattr(policy, "effective_presets", lambda **_kwargs: [])
    seen = {}

    def fake_retry_compare(run, preset, sub, scope, active_presets, backend=None):
        seen["backend"] = backend
        return comparison(scope)
    monkeypatch.setattr(corrective, "retry_compare", fake_retry_compare)
    handler = Handler()
    route.try_post(handler, "/runs/run_x/retry",
                   {"preset": "less-verbose", "scope": "once", "backend": "control_vector"})
    assert handler.status == 200
    assert seen["backend"] == "control_vector"
