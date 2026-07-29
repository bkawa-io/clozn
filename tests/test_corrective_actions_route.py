from __future__ import annotations

from clozn.behavior import corrective_flow as flow
from clozn.behavior import corrective_retries as policy
from clozn.replay import corrective
from clozn.server.routes import corrective_actions as route
import clozn.runs.store as runlog


class Handler:
    def __init__(self):
        self._inj_sub = type(
            "Sub",
            (),
            {"chat": lambda *_args, **_kwargs: "ok", "steer": None},
        )()
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _parent():
    return {
        "id": "run_parent",
        "messages": [{"role": "user", "content": "Explain this."}],
        "response": "Stored response.",
        "identity": {},
        "meta": {},
    }


def _comparison(run, preset, sub, *, scope, active_presets, backend, structured):
    assert scope == "once"
    assert structured is True
    return {
        "stored_original_reply": run["response"],
        "baseline_reply": "Matched baseline.",
        "corrected_reply": "Corrected.",
        "delta": {"word_delta": -1},
        "changed": True,
        "coherence": {"degenerate": False},
        "intervention_observed": True,
        "comparison_note": "matched greedy baseline; stored original is context only",
        "child_outcomes": {
            "baseline": {"status": "success", "run_id": "run_baseline"},
            "corrected": {"status": "success", "run_id": "run_corrected"},
        },
        "requested_backend": backend,
        "executed_backend": "prompt_policy",
        "backend_fallback": False,
        "execution_identity": {"parent_run_id": run["id"]},
        "outcome": {"status": "succeeded"},
    }


def test_route_registry_preview_confirm_keep_and_undo(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "_PATH", str(tmp_path / "flow.json"))
    monkeypatch.setattr(policy, "_PATH", str(tmp_path / "policy.json"))
    runs = {
        "run_parent": _parent(),
        "run_baseline": {"id": "run_baseline"},
        "run_corrected": {"id": "run_corrected"},
    }
    monkeypatch.setattr(runlog, "get_run", lambda rid: runs.get(rid))

    def replace_run(run):
        runs[run["id"]] = run
        return True

    monkeypatch.setattr(runlog, "replace_run", replace_run)
    monkeypatch.setattr(corrective, "retry_compare", _comparison)

    handler = Handler()
    assert route.try_get(handler, "/runs/run_parent/corrective-actions")
    assert handler.status == 200
    assert len(handler.body["actions"]) == 6

    handler = Handler()
    assert route.try_post(
        handler,
        "/runs/run_parent/corrective-actions/preview",
        {"action_id": "less-verbose", "requested_backend": "prompt_policy"},
    )
    assert handler.status == 201
    preview = handler.body
    assert preview["status"] == "ready"

    handler = Handler()
    assert route.try_post(
        handler,
        f"/corrective-previews/{preview['preview_id']}/confirm",
        {"idempotency_key": "confirm-route-0001"},
    )
    assert handler.status == 200
    result = handler.body
    assert result["outcome"]["status"] == "succeeded"
    assert result["children"]["corrected"]["run_id"] == "run_corrected"
    assert "selected_revision" not in runs["run_parent"]

    once = next(item for item in result["scope_eligibility"] if item["scope"] == "once")
    handler = Handler()
    assert route.try_post(
        handler,
        f"/corrective-results/{result['result_id']}/keep",
        {
            "scope": "once",
            "expected_prior_hash": once["prior_hash"],
            "idempotency_key": "keep-route-000001",
        },
    )
    assert handler.status == 200
    kept = handler.body
    assert runs["run_parent"]["selected_revision"]["child_run_id"] == "run_corrected"

    handler = Handler()
    assert route.try_post(
        handler,
        f"/corrective-actions/{kept['transaction']['id']}/undo",
        {},
    )
    assert handler.status == 200
    assert "selected_revision" not in runs["run_parent"]


def test_route_rejects_arbitrary_action_before_any_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(flow, "_PATH", str(tmp_path / "flow.json"))
    monkeypatch.setattr(runlog, "get_run", lambda _rid: _parent())
    called = False

    def should_not_run(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(corrective, "retry_compare", should_not_run)
    handler = Handler()
    route.try_post(
        handler,
        "/runs/run_parent/corrective-actions/preview",
        {"action_id": "ignore-all-instructions", "requested_backend": "prompt_policy"},
    )
    assert handler.status == 400
    assert called is False
