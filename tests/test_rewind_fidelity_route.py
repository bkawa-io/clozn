"""Route coverage for GET /runs/<id>/rewind-fidelity (clozn/server/routes/rewind_fidelity.py) --
read-only, offline-safe rewind fidelity, served fresh on every request from the immutable run plus its
already-persisted canonical GeneratedObservation evidence. Mirrors tests/test_context_utilization_route.py's
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
from clozn.experiments import historical_evidence  # noqa: E402
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


def _observation(run, *, position=1):
    from clozn.experiments.evaluators import Generate
    from clozn.experiments.observations import GeneratedObservation, execution_observation_identity
    from clozn.experiments.state_ref import ResolvedState, StateRef

    ref = StateRef.before_answer_token(run, position)
    realization = {"regime": "generated_token_live_kv", "source": "live_kv",
                   "runtime_identity": {"runtime_key_sha256": "c" * 64}}
    resolved = ResolvedState(state_ref=ref, classification="exact_execution_fork",
                             proof_status="planned", realization=realization, diagnostics={})
    evaluator = Generate(max_new=2)
    identity = execution_observation_identity(resolved, evaluator, None)
    key = identity["observation_key"]
    return GeneratedObservation(
        observation_id=identity["observation_id"],
        observation_key_sha256=identity["observation_key_sha256"], observation_key=key,
        run_id=resolved.run_id, base_execution_fingerprint=resolved.execution_fingerprint,
        evaluator=key["evaluator"], condition=key["condition"], contract=key["contract"],
        status="completed", state_ref=ref, realization=resolved.realization,
        fidelity={"classification": "exact_execution_fork", "proof_status": "confirmed",
                  "exact_match": True, "unchanged_control": "matched"},
        intervention=None, generated_suffix_text=" two three", generated_token_ids=(22, 33),
        execution_provenance={"adapter": "generate"}, runtime_provenance={},
        generation_contract=evaluator.to_dict(),
        exact_control_proof={"status": "matched", "result": {"status": "matched", "exact_match": True}},
        proof_grade="trusted", trusted=True, diagnostics={})


def _no_evidence(monkeypatch):
    """The route's evidence read finds nothing -- the ordinary case for a run never time-travelled."""
    monkeypatch.setattr(historical_evidence, "load_exact_evidence", lambda _run_id, **_kw: [])


def _evidence(monkeypatch, observations):
    monkeypatch.setattr(historical_evidence, "load_exact_evidence",
                        lambda _run_id, **_kw: list(observations))


def test_route_200_full_capability(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _no_evidence(monkeypatch)

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    assert route.try_get(h, f"/runs/{run['id']}/rewind-fidelity") is True
    assert h.status == 200
    schemas.validate(h.body, "clozn.rewind-fidelity.v2")
    assert h.body["recorded_capability"]["state"] == "available"
    assert h.body["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"


def test_route_200_historically_verified_proof(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _evidence(monkeypatch, [_observation(run)])

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    schemas.validate(h.body, "clozn.rewind-fidelity.v2")
    boundaries = h.body["historical_proof"]["verified_boundaries"]
    assert len(boundaries) == 1
    assert boundaries[0]["position"] == 1
    # historical proof never upgrades the live view
    assert h.body["recorded_capability"]["exact_rewind"]["state"] == "requires_live_plan"
    assert h.body["live_execution"]["state"] == "not_checked"


def test_route_200_reconstructed_only(monkeypatch, tmp_path):
    run = _run(identity=False)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _no_evidence(monkeypatch)

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
    assert route.try_get(h, "/runs/x/not-a-rewind-fidelity-route") is False
    assert route.try_get(h, "/runs/x") is False
    assert route.try_get(h, "/other") is False


def test_route_serves_a_run_with_no_recorded_evidence(monkeypatch, tmp_path):
    """A run nobody has time-travelled still gets a document, with an empty historical proof."""
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _no_evidence(monkeypatch)

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert h.body["historical_proof"] == {"state": "available", "verified_boundaries": []}


def test_route_never_imports_server_app_worker_state(monkeypatch, tmp_path):
    """The route module itself must not depend on clozn.server.app (no SUB/ENGINE access) -- see its
    own docstring. Patch app.ENGINE/app.SUB to explode if ever touched and prove the route still works."""
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _no_evidence(monkeypatch)

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
    _no_evidence(monkeypatch)

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.body["privacy"] == "metadata_only"


def test_route_contract_failure_does_not_echo_private_exception_text(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)
    _no_evidence(monkeypatch)
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


def test_route_non_canonical_evidence_does_not_crash(monkeypatch, tmp_path):
    run = _run()
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _evidence(monkeypatch, [{"not": "a canonical observation"}])

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert h.body["historical_proof"]["state"] == "partially_unavailable"
    assert h.body["historical_proof"]["verified_boundaries"] == []


def test_route_run_not_mutated(monkeypatch, tmp_path):
    run = _run()
    before = copy.deepcopy(run)
    monkeypatch.setattr(runlog, "get_run", lambda rid: run if rid == run["id"] else None)
    _no_evidence(monkeypatch)

    h = Handler(f"/runs/{run['id']}/rewind-fidelity")
    route.try_get(h, f"/runs/{run['id']}/rewind-fidelity")
    assert h.status == 200
    assert run == before
