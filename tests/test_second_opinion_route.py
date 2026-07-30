"""Route coverage for `clozn/server/routes/second_opinion.py` (E4):

    GET  /runs/<id>/second-opinion/candidates
    POST /runs/<id>/second-opinion

Mirrors tests/test_run_scoped_model_routing.py's two-model `PreloadedModelRouter` harness (the closest,
most authoritative analog for a run-scoped route that must resolve a worker through the managed router
rather than `ctx.active_sub`) -- FakeEngine/FakeSub/`_two_model_router` are re-declared locally rather
than imported, matching every other route test file's own self-contained-fixture convention in this
codebase (clozn/schemas/__init__.py's "pure addition, no shared file to edit" discipline, one level up).

No model, no GPU: every engine here is a small deterministic fake.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import clozn.runs.store as runlog
from clozn.cli.worker_registry import AdapterRuntimeIdentity, RuntimeKey
from clozn.server import app as server
from clozn.server.model_routing import PreloadedModelBinding, PreloadedModelRouter
from clozn.server.routes import second_opinion as route

WHITE_BOX = {"sae": False, "jlens": False, "attn_knockout": False}
_ABSENT = AdapterRuntimeIdentity.absent()


class FakeEngine:
    def __init__(self, *, digest, template, build, generation, base):
        self.digest = digest
        self.template = template
        self.build = build
        self.generation = generation
        self.base = base

    def health(self):
        return {
            "status": "ok", "worker_generation_id": self.generation, "protocol_version": "1.1",
            "model_sha256": self.digest, "n_ctx": 4096, "device": "cpu", "engine_build": self.build,
            "template_fingerprint": self.template, "capabilities": dict(WHITE_BOX),
        }


class FakeSub:
    def __init__(self, engine, *, reply="a fresh answer", raises=None):
        self.engine = engine
        self._reply = reply
        self._raises = raises
        self.last_chat_messages = None

    def chat(self, messages, max_new=256, sample=True, trace_out=None, **_kw):
        self.last_chat_messages = messages
        if self._raises is not None:
            raise self._raises
        if trace_out is not None:
            trace_out.extend([{"id": 1}, {"id": 2}])
        return self._reply

    def last_finish_reason(self):
        return "stop"


def _binding(model_id, *, digest, template, build, generation, base, reply="a fresh answer", raises=None):
    key = RuntimeKey(
        gguf_artifact_sha256=digest, context_size=4096, backend="cpu", adapter=_ABSENT,
        template_fingerprint=template, engine_build=build, white_box_flags=WHITE_BOX,
    ).as_dict()
    engine = FakeEngine(digest=digest, template=template, build=build, generation=generation, base=base)
    sub = FakeSub(engine, reply=reply, raises=raises)
    binding = PreloadedModelBinding(
        model_id=model_id,
        resolved_artifact={"model_id": model_id, "format": "gguf", "artifact_sha256": digest},
        runtime_key=key, adapter=_ABSENT.as_dict(), state="ready",
        worker_identity={
            "worker_id": generation, "worker_generation_id": generation, "worker_generation": 1,
            "runtime_key_sha256": key["key_sha256"], "protocol_version": "1.1",
            "engine_build": build, "backend": "cpu",
        },
        sub=sub, engine=engine,
    )
    return binding, sub, engine


def _two_model_router(*, beta_reply="a fresh answer", beta_raises=None):
    alpha_binding, alpha_sub, alpha_engine = _binding(
        "alpha", digest="a" * 64, template="1" * 16, build="build-alpha", generation="gen-alpha",
        base="http://127.0.0.1:44101")
    beta_binding, beta_sub, beta_engine = _binding(
        "beta", digest="b" * 64, template="2" * 16, build="build-beta", generation="gen-beta",
        base="http://127.0.0.1:44102", reply=beta_reply, raises=beta_raises)
    router = PreloadedModelRouter(
        [alpha_binding, beta_binding], default_model_id="alpha",
        preload_model_ids=["alpha", "beta"], max_loaded_workers=2,
    )
    return router, SimpleNamespace(sub=alpha_sub, engine=alpha_engine), SimpleNamespace(sub=beta_sub, engine=beta_engine)


class Handler:
    def __init__(self, path=""):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    return tmp_path


@pytest.fixture
def managed(monkeypatch, request):
    kwargs = getattr(request, "param", {})
    router, alpha, beta = _two_model_router(**kwargs)
    previous = server.MODEL_ROUTER, server.SUB, server.ENGINE

    class Poison:
        def __getattr__(self, name):
            raise AssertionError(
                f"second-opinion route touched the legacy default worker (.{name}) instead of "
                "resolving the requested second model through MODEL_ROUTER")

    server.MODEL_ROUTER = router
    server.SUB = Poison()
    server.ENGINE = Poison()
    try:
        yield router, alpha, beta
    finally:
        server.MODEL_ROUTER, server.SUB, server.ENGINE = previous


def _run(**overrides):
    values = {
        "source": "engine_chat", "client": "studio", "model": "alpha", "substrate": "engine",
        "messages": [{"role": "user", "content": "What year was the bridge built?"}],
        "response": "The bridge was built in 1920.", "finish_reason": "stop",
        "identity": {"model_sha256": "a" * 64, "template_fingerprint": "1" * 16},
    }
    values.update(overrides)
    run_id = runlog.record(**values)
    assert run_id
    return runlog.get_run(run_id)


# ==================================================================================== request validation

def test_run_not_found_is_404(iso, managed):
    h = Handler()
    assert route.try_post(h, "/runs/does_not_exist/second-opinion", {"model": "beta"})
    assert h.status == 404


def test_missing_body_model_is_400(iso, managed):
    run = _run()
    h = Handler()
    assert route.try_post(h, f"/runs/{run['id']}/second-opinion", {})
    assert h.status == 400


def test_same_model_as_the_run_is_400(iso, managed):
    run = _run(model="alpha")
    h = Handler()
    assert route.try_post(h, f"/runs/{run['id']}/second-opinion", {"model": "alpha"})
    assert h.status == 400


# ============================================================================= no managed router at all

def test_no_managed_router_refuses_503(iso, monkeypatch):
    previous = server.MODEL_ROUTER, server.SUB, server.ENGINE
    server.MODEL_ROUTER = None
    server.SUB = None
    server.ENGINE = None
    try:
        run = _run()
        h = Handler()
        assert route.try_post(h, f"/runs/{run['id']}/second-opinion", {"model": "beta"})
        assert h.status == 503
        assert h.body["code"] == "second_opinion_requires_managed_router"
    finally:
        server.MODEL_ROUTER, server.SUB, server.ENGINE = previous


# ==================================================================== unresolvable second model (typed)

def test_unknown_second_model_refuses_typed_never_falls_back(iso, managed):
    run = _run(model="alpha")
    h = Handler()
    assert route.try_post(h, f"/runs/{run['id']}/second-opinion", {"model": "not-configured"})
    assert h.status >= 400
    artifact = h.body.get("clozn_model_routing")
    assert isinstance(artifact, dict)
    assert artifact["schema_version"] == "clozn.model-routing.v1"
    assert h.body["error"]["code"] == "unknown_model"


# ============================================================================================ success

def test_second_opinion_success_uses_betas_worker_never_alphas(iso, managed):
    router, alpha, beta = managed
    run = _run(model="alpha")
    h = Handler()
    assert route.try_post(h, f"/runs/{run['id']}/second-opinion", {"model": "beta"})
    assert h.status == 200
    document = h.body
    assert document["schema_version"] == "clozn.model-second-opinion.v1"
    assert document["arm_a"]["status"] == "ok"
    assert document["arm_a"]["response_text"] == run["response"]
    assert document["arm_b"]["status"] == "ok"
    assert document["arm_b"]["model_id"] == "beta"
    assert beta.sub.last_chat_messages is not None
    assert alpha.sub.last_chat_messages is None   # arm_a's own worker was never touched


@pytest.mark.parametrize("managed", [{"beta_raises": RuntimeError("worker died mid-generation")}],
                        indirect=True)
def test_second_opinion_arm_b_failure_still_returns_200_with_arm_a_intact(iso, managed):
    """The route-level proof of the honesty requirement: a live generation failure on the SECOND
    model is not an HTTP error -- the anchor's evidence is returned regardless."""
    run = _run(model="alpha")
    h = Handler()
    assert route.try_post(h, f"/runs/{run['id']}/second-opinion", {"model": "beta"})
    assert h.status == 200
    document = h.body
    assert document["arm_a"]["status"] == "ok"
    assert document["arm_a"]["response_text"] == run["response"]
    assert document["arm_b"]["status"] == "generation_error"
    assert "comparison" not in document


# ========================================================================================== candidates

def test_candidates_lists_the_other_ready_models(iso, managed):
    run = _run(model="alpha")
    h = Handler()
    assert route.try_get(h, f"/runs/{run['id']}/second-opinion/candidates")
    assert h.status == 200
    assert h.body["managed"] is True
    assert h.body["own_model_id"] == "alpha"
    ids = {c["model_id"] for c in h.body["candidates"]}
    assert ids == {"beta"}
    assert all(c["ready"] for c in h.body["candidates"])


def test_candidates_reports_a_not_ready_model_as_unready_not_a_crash(iso, monkeypatch):
    """The exact scenario that requires `/runs/<id>/second-opinion/candidates` to be a registered
    `clozn.model-routing.v1` route template (see this file's enum widening): a not-ready binding makes
    `select_control_model` build and validate a refusal artifact internally even though
    `peek_control_model_for_run` swallows it. An unregistered route string would raise a schema
    ValidationError from INSIDE that construction, crashing this read instead of degrading it."""
    router, alpha, beta = _two_model_router()
    gamma_key = RuntimeKey(
        gguf_artifact_sha256="c" * 64, context_size=4096, backend="cpu", adapter=_ABSENT,
        template_fingerprint="3" * 16, engine_build="build-gamma", white_box_flags=WHITE_BOX,
    ).as_dict()
    gamma_binding = PreloadedModelBinding(
        model_id="gamma", resolved_artifact={"model_id": "gamma", "format": "gguf", "artifact_sha256": "c" * 64},
        runtime_key=gamma_key, adapter=_ABSENT.as_dict(), state="loading", worker_identity=None,
        sub=None, engine=None,
    )
    three_model_router = PreloadedModelRouter(
        list(router._by_id.values()) + [gamma_binding], default_model_id="alpha",
        preload_model_ids=["alpha", "beta", "gamma"], max_loaded_workers=3,
    )
    previous = server.MODEL_ROUTER, server.SUB, server.ENGINE
    server.MODEL_ROUTER = three_model_router
    server.SUB = server.ENGINE = None
    try:
        run = _run(model="alpha")
        h = Handler()
        assert route.try_get(h, f"/runs/{run['id']}/second-opinion/candidates")
        assert h.status == 200
        by_id = {c["model_id"]: c["ready"] for c in h.body["candidates"]}
        assert by_id == {"beta": True, "gamma": False}
    finally:
        server.MODEL_ROUTER, server.SUB, server.ENGINE = previous


def test_candidates_without_a_managed_router(iso, monkeypatch):
    previous = server.MODEL_ROUTER
    server.MODEL_ROUTER = None
    try:
        run = _run()
        h = Handler()
        assert route.try_get(h, f"/runs/{run['id']}/second-opinion/candidates")
        assert h.status == 200
        assert h.body["managed"] is False
        assert h.body["candidates"] == []
    finally:
        server.MODEL_ROUTER = previous


def test_candidates_run_not_found_is_404(iso, managed):
    h = Handler()
    assert route.try_get(h, "/runs/does_not_exist/second-opinion/candidates")
    assert h.status == 404


# ===================================================================================== path parsing

def test_try_post_ignores_unrelated_paths(iso):
    h = Handler()
    assert route.try_post(h, "/runs/x/fork", {}) is False


def test_try_get_ignores_unrelated_paths(iso):
    h = Handler()
    assert route.try_get(h, "/runs/x/investigation") is False
