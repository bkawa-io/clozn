"""Model-free contracts for RT-BOOT-01 managed preloaded serving."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from clozn.cli import runtime_process
from clozn.cli.commands import serve as serve_command
from clozn.cli.engine_process import EngineDiscovery
from clozn.cli.managed_models import (
    ManagedModelsConfigError,
    load_managed_models,
)
from clozn.cli.worker_registry import (
    AdapterRuntimeIdentity,
    RuntimeKey,
    WorkerDefinition,
)
from clozn.server import app
from clozn.server.model_routing import (
    PreloadedModelBinding,
    PreloadedModelRouter,
)
from clozn.server.routes import health


WHITE_BOX = {
    "sae": False,
    "jlens": False,
    "attn_knockout": False,
}
TEMPLATE = "1" * 16


class FakeProcess:
    _next_pid = 7100

    def __init__(self, code=None):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.returncode = code
        self.terminated = False
        self.killed = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def kill(self):
        self.killed = True
        self.returncode = -9

    def wait(self, timeout=None):
        return self.returncode


class CaptureHandler:
    def __init__(self):
        self.code = None
        self.json = None

    def _json(self, code, value, extra_headers=None):
        self.code = code
        self.json = value


def _sha(path: os.PathLike[str] | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _definition(
    model_id: str,
    model: Path,
    *,
    executable_sha: str,
) -> WorkerDefinition:
    return WorkerDefinition(
        model_id=model_id,
        model=str(model),
        runtime_key=RuntimeKey(
            gguf_artifact_sha256=_sha(model),
            context_size=2048,
            backend="cpu",
            adapter=AdapterRuntimeIdentity.absent(),
            template_fingerprint=TEMPLATE,
            engine_build=f"sha256:{executable_sha}",
            white_box_flags=WHITE_BOX,
        ),
        flags={"ctx": 2048},
        prefer_gpu=False,
    )


@dataclass
class ManagedHarness:
    definitions: tuple[WorkerDefinition, ...]
    stack: object
    worker_processes: dict[str, list[FakeProcess]]
    health_by_port: dict[int, dict]
    spawn_calls: list[tuple]


def _spawn_harness(
    monkeypatch,
    tmp_path: Path,
    *,
    fail_once: set[str] | None = None,
) -> ManagedHarness:
    engine = tmp_path / "clozn-server"
    engine.write_bytes(b"qualified-engine-bytes")
    model_a = tmp_path / "alpha.gguf"
    model_b = tmp_path / "beta.gguf"
    model_a.write_bytes(b"alpha-model")
    model_b.write_bytes(b"beta-model")
    definitions = (
        _definition("alpha", model_a, executable_sha=_sha(engine)),
        _definition("beta", model_b, executable_sha=_sha(engine)),
    )
    discovery = EngineDiscovery(
        exe=str(engine),
        dll_dirs=[],
        gpu=False,
        discovery_source="repo_dev_build",
        backend="cpu",
    )
    monkeypatch.setattr(
        runtime_process, "find_engine_ex", lambda prefer_gpu=True: discovery
    )
    ports = iter((43101, 43102, 43103, 43104))
    monkeypatch.setattr(runtime_process, "_free_port", lambda: next(ports))
    monkeypatch.setattr(
        runtime_process, "_worker_template_fingerprint", lambda _port: TEMPLATE
    )
    monkeypatch.setattr(runtime_process, "port_is_open", lambda _port: False)
    monkeypatch.setattr(
        runtime_process, "gateway_health", lambda _port: {"status": "ok"}
    )

    failures = set(fail_once or ())
    worker_processes: dict[str, list[FakeProcess]] = {
        "alpha": [],
        "beta": [],
    }
    health_by_port: dict[int, dict] = {}
    spawn_calls = []

    def fake_spawn(model, port, flags, **kwargs):
        model_id = Path(model).stem
        spawn_calls.append((model_id, port, dict(flags), dict(kwargs)))
        assert kwargs["engine_discovery"] is discovery
        assert flags["_disable_auto_jlens"] is True
        if model_id in failures:
            failures.remove(model_id)
            raise RuntimeError(f"{model_id} failed once")
        process = FakeProcess()
        worker_processes[model_id].append(process)
        definition = next(
            item for item in definitions if item.model_id == model_id
        )
        worker_health = {
            "status": "ok",
            "protocol_version": "1.1",
            "worker_generation_id": f"{model_id}-{process.pid}",
            "model_sha256": definition.runtime_key.gguf_artifact_sha256,
            "n_ctx": 2048,
            "device": "cpu",
            "mode": "autoregressive",
            "capabilities": dict(WHITE_BOX),
        }
        health_by_port[port] = worker_health
        return process, worker_health, False

    monkeypatch.setattr(runtime_process, "spawn_engine", fake_spawn)
    class FakeServer:
        def __init__(self):
            import threading
            self.stop_event = threading.Event()

        def serve_forever(self):
            self.stop_event.wait()

        def shutdown(self):
            self.stop_event.set()

        def server_close(self):
            pass

    fake_server = FakeServer()
    monkeypatch.setattr(
        app,
        "EngineClient",
        lambda *, port: type(
            "FakeEngine", (), {"health": lambda _self: dict(health_by_port[port])}
        )(),
    )
    monkeypatch.setattr(
        app,
        "EngineSubstrate",
        lambda *, engine: type("FakeSubstrate", (), {"engine": engine})(),
    )
    def fake_build_server(**kwargs):
        monkeypatch.setattr(app, "MODEL_ROUTER", kwargs["model_router"])
        monkeypatch.setattr(app, "ENGINE", kwargs["engine"])
        monkeypatch.setattr(app, "SUB", app.MODEL_ROUTER.control_pair()[0])
        return fake_server

    monkeypatch.setattr(app, "build_server", fake_build_server)
    stack = runtime_process.spawn_runtime(
        runtime_process.RuntimeConfig(
            model=str(model_a),
            public_port=8181,
            worker_definitions=definitions,
            default_model_id="alpha",
            preload_model_ids=("alpha", "beta"),
            max_loaded_models=2,
        )
    )
    return ManagedHarness(
        definitions=definitions,
        stack=stack,
        worker_processes=worker_processes,
        health_by_port=health_by_port,
        spawn_calls=spawn_calls,
    )


def test_managed_runtime_uses_live_in_memory_projection(monkeypatch, tmp_path):
    harness = _spawn_harness(monkeypatch, tmp_path)
    stack = harness.stack
    try:
        assert stack.gateway.pid == os.getpid()
        selected = app.MODEL_ROUTER.select_control_model(
            "beta", route="/runs/<id>/execution-fork/plan"
        )
        assert selected.model_id == "beta"
        assert selected.worker_identity["runtime_key_sha256"] == (
            harness.definitions[1].runtime_key.key_sha256
        )
    finally:
        stack.stop()


def _manifest(definitions: tuple[WorkerDefinition, ...]) -> dict:
    return {
        "schema_version": "clozn.managed-models.v1",
        "default_model_id": "alpha",
        "preload_model_ids": ["alpha", "beta"],
        "max_loaded_models": 2,
        "models": [
            {
                "model_id": definition.model_id,
                "model": definition.model,
                "runtime_key": definition.runtime_key.as_dict(),
                "flags": dict(definition.flags),
                "prefer_gpu": definition.prefer_gpu,
            }
            for definition in definitions
        ],
    }


def test_qualified_manifest_verifies_files_and_forbids_configured_ports(
    tmp_path
):
    engine = tmp_path / "clozn-server"
    engine.write_bytes(b"engine")
    models = []
    for model_id in ("alpha", "beta"):
        model = tmp_path / f"{model_id}.gguf"
        model.write_bytes(model_id.encode())
        models.append(_definition(model_id, model, executable_sha=_sha(engine)))
    manifest = _manifest(tuple(models))
    path = tmp_path / "models.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_managed_models(str(path))
    assert loaded.default_model_id == "alpha"
    assert loaded.preload_model_ids == ("alpha", "beta")
    assert loaded.max_loaded_models == 2

    manifest["models"][0]["port"] = 9000
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ManagedModelsConfigError, match="property 'port'"):
        load_managed_models(str(path))


def test_managed_active_substrate_never_falls_back_to_control_worker():
    alpha = SimpleNamespace(chat=lambda *_args, **_kwargs: "wrong worker")
    previous = app.MODEL_ROUTER, app.SUB, app.ENGINE
    app.MODEL_ROUTER = SimpleNamespace(
        control_pair=lambda: (alpha, SimpleNamespace())
    )
    app.SUB = alpha
    app.ENGINE = SimpleNamespace()
    try:
        handler = SimpleNamespace()
        assert app.active_sub(handler) is None
        assert app.active_engine(handler) is None
    finally:
        app.MODEL_ROUTER, app.SUB, app.ENGINE = previous


def test_legacy_engine_substrate_identity_is_exact_fork_eligible(
    monkeypatch, tmp_path
):
    from clozn.replay.execution_fork import (
        _runtime_projection,
        parent_runtime_projection,
    )
    from clozn.server.routes.execution_fork import _sub_facts
    from clozn.server.substrates import EngineSubstrate

    model_sha = hashlib.sha256(b"legacy-model").hexdigest()
    executable_sha = hashlib.sha256(b"legacy-engine").hexdigest()
    rendered = "<system>canonical</system>"
    template = hashlib.sha256(rendered.encode()).hexdigest()[:16]
    monkeypatch.setenv("CLOZN_ENGINE_ARTIFACT_SHA256", executable_sha)
    monkeypatch.setenv("CLOZN_ENGINE_DISCOVERY_SOURCE", "repo_dev_build")
    monkeypatch.setenv("CLOZN_ENGINE_BACKEND", "cpu")

    class Engine:
        base = "http://127.0.0.1:43112"

        def health(self):
            return {
                "status": "ok",
                "protocol_version": "1.1",
                "worker_generation_id": "legacy-process-generation",
                "model": str(tmp_path / "legacy.gguf"),
                "model_sha256": model_sha,
                "n_ctx": 2048,
                "device": "cpu",
                "mode": "autoregressive",
                "capabilities": dict(WHITE_BOX),
            }

        def apply_template(self, _messages):
            return rendered

    class Steer:
        def load_state(self, _path):
            return None

        def load_calibration(self, _value):
            return None

    sub = EngineSubstrate(engine=Engine(), steer=Steer())
    meta = sub.run_meta()
    identity = sub.identity_meta()
    assert identity["template_fingerprint"] == template
    assert "engine_build" not in identity
    assert (
        identity["ext"]["engine_artifact"]["artifact_sha256"]
        == executable_sha
    )
    assert meta["white_box_flags"] == WHITE_BOX

    parent = {
        "id": "run_legacy",
        "model": "legacy",
        "identity": identity,
        "meta": meta,
    }
    parent_runtime = parent_runtime_projection(parent)
    selected_runtime, worker, selected_engine = _sub_facts(sub)
    assert selected_engine is sub.engine
    assert worker["worker_generation_id"] == "legacy-process-generation"
    assert parent_runtime is not None
    assert _runtime_projection(selected_runtime) == parent_runtime
    assert parent_runtime["engine_build"] == f"sha256:{executable_sha}"


def test_degraded_managed_gateway_remains_visible_to_ps(
    monkeypatch, capsys
):
    registry = {
        "8181": {
            "model": "alpha.gguf",
            "gpu": False,
            "mode": "autoregressive",
            "models": [
                {
                    "model_id": "alpha",
                    "state": "failed",
                    "worker_pid": None,
                }
            ],
        }
    }
    writes = []
    monkeypatch.setattr(serve_command, "_reg_read", lambda: dict(registry))
    monkeypatch.setattr(
        serve_command,
        "gateway_liveness",
        lambda _port, timeout=1.0: {"status": "ok"},
    )
    monkeypatch.setattr(
        serve_command, "gateway_health", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(serve_command, "_reg_write", writes.append)
    monkeypatch.setattr(serve_command, "_worker_url", lambda _port: "-")

    serve_command.cmd_ps(SimpleNamespace())
    assert "8181" in capsys.readouterr().out
    assert writes == []


def test_stop_falls_back_to_every_nested_managed_worker_pid(
    monkeypatch
):
    entry = {
        "model": "alpha.gguf",
        "pid": 8001,
        "gateway_pid": 8002,
        "worker_pid": 8003,
        "models": [
            {"model_id": "alpha", "worker_pid": 8003},
            {"model_id": "beta", "worker_pid": "8004"},
            {"model_id": "bad", "worker_pid": "not-a-pid"},
        ],
    }
    reads = iter(({"8181": entry}, {"8181": entry}))
    killed = []
    awaited = []
    writes = []
    monkeypatch.setattr(serve_command, "_reg_read", lambda: next(reads))
    monkeypatch.setattr(serve_command, "_kill", killed.append)
    monkeypatch.setattr(
        serve_command, "_await_dead", lambda pids, timeout: awaited.append(set(pids))
    )
    monkeypatch.setattr(serve_command, "_reg_write", writes.append)
    monkeypatch.setattr(serve_command.time, "sleep", lambda _seconds: None)

    serve_command.cmd_stop(SimpleNamespace(which="8181"))
    assert killed == [8001, 8002, 8003, 8004]
    assert awaited == [{8001, 8002, 8003, 8004}]
    assert writes[-1] == {}


def test_unselected_run_engine_routes_never_use_default_worker(
    monkeypatch
):
    """RT-BOOT-01's original point: `alpha` is the router's default/control-pair worker, but the
    run under test belongs to "beta" -- a DIFFERENT, unconfigured model. These routes must resolve
    "beta" through the router (clozn.server.model_routing.select_control_model_for_run), not
    silently fall through to alpha/SUB/ENGINE.

    Before RT-MRSR-01 (this fix) that resolution didn't exist: every one of these routes called the
    bare `ctx.active_sub(h)`, which fails closed to None whenever a router is configured at all (see
    ctx.active_sub's own docstring) -- so they 503'd for EVERY run under a managed gateway, not just
    ones genuinely pointed at an unconfigured model. Now they ask the router for "beta" specifically,
    "beta" is genuinely unknown to it, and the typed `clozn.model-routing.v1` refusal is what a
    caller actually sees -- never a bare 503, and never alpha/SUB/ENGINE touched either way.
    """
    from clozn.server.routes import fork, influence_map, receipts, replay
    import clozn.runs.store as runlog

    calls = []

    class Alpha:
        engine = SimpleNamespace()

        def chat(self, *_args, **_kwargs):
            calls.append("chat")

        def score_tokens(self, *_args, **_kwargs):
            calls.append("score")

    alpha = Alpha()
    run = {"id": "run_beta", "model": "beta"}
    monkeypatch.setattr(runlog, "get_run", lambda _rid: run)

    # A REAL router, deliberately configured with ONLY "alpha" -- "beta" is genuinely unknown to
    # it, exactly like a run whose model was never preloaded on this gateway.
    key = RuntimeKey(
        gguf_artifact_sha256="a" * 64, context_size=4096, backend="cpu",
        adapter=AdapterRuntimeIdentity.absent(), template_fingerprint=TEMPLATE,
        engine_build="build-alpha", white_box_flags=WHITE_BOX,
    ).as_dict()
    alpha_binding = PreloadedModelBinding(
        model_id="alpha",
        resolved_artifact={"model_id": "alpha", "format": "gguf", "artifact_sha256": "a" * 64},
        runtime_key=key,
        adapter=AdapterRuntimeIdentity.absent().as_dict(),
        state="ready",
        worker_identity={
            "worker_id": "gen-alpha", "worker_generation_id": "gen-alpha",
            "worker_generation": 1, "runtime_key_sha256": key["key_sha256"],
            "protocol_version": "1.1", "engine_build": "build-alpha", "backend": "cpu",
        },
        sub=alpha, engine=alpha.engine,
    )
    router = PreloadedModelRouter(
        [alpha_binding], default_model_id="alpha",
        preload_model_ids=["alpha"], max_loaded_workers=1,
    )

    previous = app.MODEL_ROUTER, app.SUB, app.ENGINE
    app.MODEL_ROUTER = router
    app.SUB = alpha
    app.ENGINE = alpha.engine
    try:
        cases = (
            (replay.try_post, "/runs/run_beta/replay", {}),
            (
                receipts.try_post,
                "/runs/run_beta/receipts",
                {"mode": "regen"},
            ),
            (
                influence_map.try_post,
                "/runs/run_beta/influence-map",
                {},
            ),
            (
                fork.try_post,
                "/runs/run_beta/fork",
                {"position": 0, "token": "x"},
            ),
        )
        for function, path, body in cases:
            handler = CaptureHandler()
            assert function(handler, path, body)
            # A typed clozn.model-routing.v1 refusal (unknown_model -> 404), never a bare 503 and
            # never a silently-succeeded 200 against alpha's worker.
            assert handler.code == 404, (function, handler.code, handler.json)
            artifact = handler.json.get("clozn_model_routing")
            assert isinstance(artifact, dict)
            assert artifact["schema_version"] == "clozn.model-routing.v1"
            assert handler.json["error"]["code"] == "unknown_model"
        assert calls == []

        # Metadata-only reads remain usable without any selected worker.
        stored = {
            "schema": "clozn.context_answer_influence.v1",
            "available": False,
        }
        run["influence_map"] = stored
        handler = CaptureHandler()
        assert influence_map.try_get(
            handler, "/runs/run_beta/influence-map"
        )
        assert handler.code == 200
        assert handler.json == stored
    finally:
        app.MODEL_ROUTER, app.SUB, app.ENGINE = previous
