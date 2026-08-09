"""Route coverage for GET /runs/<id>/rewind-fidelity (clozn/server/routes/rewind_fidelity.py) --
read-only, offline-safe rewind fidelity, served fresh on every request from the immutable run plus its
already-persisted terminal execution-fork receipts. Mirrors tests/test_context_utilization_route.py's
Handler stub, autoload-registration, and contract-failure patterns.

Model-free: nothing here touches clozn/engine, a worker, or a model file. `build_rewind_fidelity` is a
pure function of already-recorded data (see its own docstring), so this whole suite runs offline.
"""
from __future__ import annotations

import copy
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import clozn.runs.store as runlog  # noqa: E402
from clozn import schemas  # noqa: E402
from clozn.replay import execution_fork, execution_fork_results  # noqa: E402
from clozn.server.routes import rewind_fidelity as route  # noqa: E402


class Handler:
    def __init__(self, path="/"):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _identity(*, model="a"):
    return {
        "model_sha256": model * 64,
        "template_fingerprint": "b" * 16,
        "engine_build": "clozn-engine-test",
        "white_box_flags": {},
    }


def _run(*, run_id="run_x", tokens=("one", " two", " three"), token_ids=(11, 22, 33),
        final_prompt="<prompt>", identity=True, **over):
    out = {
        "id": run_id,
        "messages": [{"role": "user", "content": "hi"}],
        "meta": {"n_ctx": 4096, "device": "cpu"},
        "response": "".join(tokens) if tokens else "",
        "finish_reason": "stop",
    }
    if identity:
        out["identity"] = _identity()
    if final_prompt is not None:
        out["final_prompt"] = final_prompt
    if tokens is not None:
        out["trace"] = {"tokens": list(tokens), "token_ids": list(token_ids)}
    out.update(over)
    return out


def _receipt(run, *, execution_id, position=1):
    fingerprint = execution_fork.parent_execution_fingerprint(run)
    return {
        "schema_version": "clozn.execution-fork.v1",
        "plan_id": "fork_plan_" + ("0" * 20),
        "execution_id": execution_id,
        "phase": "completed",
        "classification": "exact_execution_fork",
        "parent_run_id": run["id"],
        "parent_fingerprint_sha256": fingerprint,
        "request": {"position": position, "change": {"type": "none"},
                   "execution_change": {"type": "none"}, "change_sha256": "b" * 64},
        "identity": {"parent_runtime": {
            "runtime_key_sha256": "c" * 64, "model_sha256": "a" * 64,
            "template_fingerprint": "b" * 16, "engine_build": "x", "context_size": 4096,
            "backend": "cpu",
            "adapter": {"present": False, "identity_sha256": None, "artifact_sha256": None, "scale": None},
            "white_box_flags": {},
        }},
        "exactness": {"regime": "generated_token_live_kv", "source": "live_kv",
                     "proof_status": "confirmed", "truncate_to": 11, "boundary_shape_true": True},
        "unavoidable_differences": [],
        "unchanged_control": {"required": True, "status": "matched",
                              "result": {"status": "matched", "exact_match": True}},
        "child_lineage": {"parent_run_id": run["id"], "source": "fork",
                          "change_sha256": "b" * 64, "receipt_status": "created"},
        "execution": {"status": "succeeded", "started_ts": 1.0, "ended_ts": 2.0},
        "reasons": [{"code": "execution_succeeded", "message": "ok"}],
    }


def test_route_200_full_capability(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    assert route.try_get(h, f"/runs/{run['id']}/rewind-fidelity") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.rewind-fidelity.v1")
    assert h.body["recorded_capability"]["state"] == "available"
    assert h.body["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"


def test_route_200_historically_verified_proof(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    execution_fork_results.save(_receipt(run, execution_id="fork_exec_" + "a" * 20))

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    schemas.validate(h.body, "clozn.rewind-fidelity.v1")
    boundaries = h.body["historical_proof"]["verified_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["position"] == 1
    # historical proof never upgrades the live view
    assert h.body["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"
    assert h.body["live_execution"]["state"] == "not_checked"


def test_route_200_reconstructed_only(monkeypatch, tmp_path):
    run = _run(identity=False)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert h.body["recorded_capability"]["state"] == "limited"
    assert h.body["recorded_capability"]["reconstructed_replay"]["state"] == "available"
    assert h.body["recorded_capability"]["exact_rewind"]["state"] == "static_prerequisites_unavailable"


def test_route_404_when_run_not_found():
    original = runlog.get_run
    try:
        runlog.get_run = lambda _rid: None
        h = Handler("/runs/missing/rewind-fidelity")
        assert route.try_get(h, "/runs/missing/rewind-fidelity") is True
        assert h.status == 404
    finally:
        runlog.get_run = original


def test_route_registered_before_the_runs_fallback():
    from clozn.server import app as server
    assert route in server._GET_ROUTES
    assert server._GET_ROUTES.index(route) < server._GET_ROUTES.index(server._runs_fallback_routes)


def test_route_does_not_match_unrelated_paths():
    h = Handler("/runs/x/context-utilization")
    assert route.try_get(h, "/runs/x/context-utilization") is False
    assert route.try_get(h, "/runs/x/execution-fork") is False
    assert route.try_get(h, "/runs/x") is False
    assert route.try_get(h, "/other") is False


def test_route_creates_no_execution_fork_db_when_history_is_absent(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    results_dir = tmp_path / "execution-forks"
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(results_dir))

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert not results_dir.exists()


def test_route_never_imports_server_app_worker_state(monkeypatch, tmp_path):
    """The route module itself must not depend on clozn.server.app (no SUB/ENGINE access) -- see its
    own docstring. Patch app.ENGINE/app.SUB to explode if ever touched and prove the route still works."""
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))

    from clozn.server import app as ctx

    def _explode(*_a, **_kw):
        raise AssertionError("rewind-fidelity route touched server worker state")

    monkeypatch.setattr(ctx, "active_engine", _explode, raising=False)
    monkeypatch.setattr(ctx, "ENGINE", None, raising=False)

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200


def test_route_response_is_metadata_only(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.body["privacy"] == "metadata_only"


def test_route_contract_failure_does_not_echo_private_exception_text(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    monkeypatch.setattr(
        "clozn.replay.rewind_fidelity.build_rewind_fidelity",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(TypeError("PRIVATE MALFORMED SOURCE")),
    )

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    assert route.try_get(h, f"/runs/{run['id']}/rewind-fidelity") is True
    assert h.status == 500
    assert h.body == {
        "error": "run rewind fidelity could not be composed",
        "code": "rewind_fidelity_contract_invalid",
    }
    assert "PRIVATE MALFORMED SOURCE" not in repr(h.body)


def test_route_malformed_historical_receipt_does_not_crash(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))
    monkeypatch.setattr(
        execution_fork_results, "list_for_parent", lambda _parent_run_id: [{"not": "a valid receipt"}],
    )

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert h.body["historical_proof"]["state"] == "partially_unavailable"
    assert h.body["historical_proof"]["verified_boundaries"] == []


def test_route_run_not_mutated(monkeypatch, tmp_path):
    run = _run()
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    monkeypatch.setattr(execution_fork_results, "RESULTS_DIR", str(tmp_path / "execution-forks"))

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert run == before
