"""Model-free coverage for the explicit mechanistic comparison/lookup HTTP surface."""
from __future__ import annotations

import clozn.runs.store as runlog
from clozn.server.routes import mechanistic_diff as route


class Handler:
    def __init__(self, path=""):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = runlog.record(
        source="test", client="test", model="model-a", substrate="engine",
        messages=[{"role": "user", "content": "hello"}], response="hello",
        final_prompt="<prompt>", trace={"tokens": ["hello"], "token_ids": [7]},
        identity={"model_sha256": "a" * 64},
    )
    return run_id


def test_lookup_persisted_mechanistic_diff_by_cache_identity(tmp_path, monkeypatch):
    run_id = _run(tmp_path, monkeypatch)
    run = runlog.get_run(run_id)
    artifact = {
        "schema_version": "clozn.mechanistic-diff.v1",
        "generated_at": "2026-08-01T00:00:00Z",
        "reference_model": {}, "candidate_model": {},
        "pair_compatibility": {"schema_version": "clozn.pair-compatibility.v1", "verdict": {}},
        "continuation": {"n_prompt": 1, "n_cont": 1},
        "layers_requested": [1], "positions_requested": [1],
        "layer_capture": [], "position_metrics": [], "residual_points": [],
    }
    cache_key = "a" * 64
    updated = dict(run)
    updated["token_workbench_actions"] = [{
        "schema_version": "clozn.token-workbench-action.v1", "action": "mechanistic_diff",
        "cache_key": cache_key, "method_version": "mechanistic_diff.v1", "run_id": run_id,
        "index": 0, "computed_at": "2026-08-01T00:00:00Z", "outcome": "ok", "result": artifact,
    }]
    assert runlog.replace_run(updated)

    handler = Handler()
    assert route.try_get(handler, f"/mechanistic-diffs/{cache_key}") is True
    assert handler.status == 200
    assert handler.body["id"] == cache_key
    assert handler.body["artifact"]["schema_version"] == "clozn.mechanistic-diff.v1"


def test_lookup_unknown_or_malformed_identity_is_not_a_run_fallback(tmp_path, monkeypatch):
    _run(tmp_path, monkeypatch)
    for artifact_id in ("nope", "a" * 63):
        handler = Handler()
        assert route.try_get(handler, f"/mechanistic-diffs/{artifact_id}") is True
        assert handler.status == 404


def test_explicit_compare_route_delegates_to_authoritative_workbench_action(tmp_path, monkeypatch):
    anchor_id = _run(tmp_path, monkeypatch)
    reference_id = runlog.record(
        source="test", client="test", model="model-b", substrate="engine",
        messages=[{"role": "user", "content": "hello"}], response="hello",
        final_prompt="<prompt>", trace={"tokens": ["hello"], "token_ids": [7]},
        identity={"model_sha256": "b" * 64},
    )
    calls = []

    def delegated(handler, run, index, body):
        calls.append((run["id"], index, body))
        handler._json(202, {"outcome": "job", "job": {"kind": "mechanistic_diff"}})
        return True

    import clozn.server.routes.token_workbench_actions as action_route
    monkeypatch.setattr(action_route, "_mechanistic_diff_action", delegated)
    handler = Handler()
    assert route.try_post(handler, "/runs/compare/mechanistic", {
        "a": anchor_id, "b": reference_id, "index": 0, "layers": [1],
    }) is True
    assert handler.status == 202
    assert calls[0][0:2] == (anchor_id, 0)
    assert calls[0][2]["reference_run_id"] == reference_id
