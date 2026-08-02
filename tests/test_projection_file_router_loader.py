"""Model-free concurrent acceptance tests for the merged in-process router."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import threading
import time

import clozn.runs.store as runlog
import clozn.settings as clozn_settings
from clozn import schemas
from clozn.cli.worker_registry import (
    AdapterRuntimeIdentity,
    RuntimeKey,
    WorkerDefinition,
    WorkerRegistry,
)
from clozn.server import app as cs
from clozn.server.model_routing import (
    ColdLoadOutcome,
    InMemoryProjectionRouter,
    ModelRoutingError,
)
from clozn.server.request_gate import WorkerGateRegistry


WHITE_BOX = {"sae": False, "jlens": False, "attn_knockout": False}


# --- shared fakes (mirrors tests/test_worker_concurrency_gate.py's fakes) ------------------------


class FakeSteer:
    def active(self):
        return {}


class FakeEngine:
    def __init__(self, model_id: str, digest: str, template: str, build: str):
        self.model_id = model_id
        self.digest = digest
        self.template = template
        self.build = build
        self.worker_generation_id = f"{model_id}-worker-1"

    def health(self):
        return {
            "status": "ok",
            "protocol_version": "1.1",
            "worker_generation_id": self.worker_generation_id,
            "model": f"{self.model_id}.gguf",
            "model_sha256": self.digest,
            "n_ctx": 2048,
            "device": "cpu",
            "engine_build": self.build,
            "template_fingerprint": self.template,
            "capabilities": dict(WHITE_BOX),
        }


class FakeSub:
    name = "engine"
    brain = None

    def __init__(self, engine: FakeEngine):
        self.engine = engine
        self.steer = FakeSteer()
        self.calls: list[dict] = []
        self._request = None

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        call_id = f"{self.engine.model_id}-{len(self.calls)}"
        self.calls.append({"call_id": call_id})
        self._request = call_id
        if mem_out is not None:
            mem_out.update(
                assembled_messages=[dict(m) for m in messages],
                final_prompt=f"<{call_id}>rendered</{call_id}>",
                actual_prompt_tokens=7,
            )
        if trace_out is not None:
            trace_out.append({"pos": 0, "token_id": 1, "piece": call_id, "prob": 1.0, "alts": []})
        return f"{call_id} reply"

    def last_finish_reason(self):
        return f"stop:{self._request}"

    def last_stream_trace(self):
        return [{"pos": 0, "token_id": 1, "piece": self.engine.model_id, "prob": 1.0, "alts": []}]

    def run_meta(self):
        return {
            "model_id": self.engine.model_id,
            "model_sha256": self.engine.digest,
            "last_request": self._request,
        }

    def identity_meta(self):
        return {
            "model_sha256": self.engine.digest,
            "template_fingerprint": self.engine.template,
            "engine_build": self.engine.build,
        }

    def last_prompt_tokens(self):
        return 7


class _FakeSupervisorProcess:
    """Minimal Popen-shaped stand-in for WorkerHandle's process field."""

    _next_pid = 9500

    def __init__(self):
        type(self)._next_pid += 1
        self.pid = type(self)._next_pid
        self.code = None

    def poll(self):
        return self.code

    def terminate(self):
        self.code = -15

    def kill(self):
        self.code = -9

    def wait(self, timeout=None):
        return self.code


def _definitions() -> tuple[WorkerDefinition, WorkerDefinition]:
    def definition(model_id: str, digest: str, template: str, *, port: int) -> WorkerDefinition:
        return WorkerDefinition(
            model_id=model_id,
            model=f"{model_id}.gguf",
            runtime_key=RuntimeKey(
                gguf_artifact_sha256=digest,
                context_size=2048,
                backend="cpu",
                adapter=AdapterRuntimeIdentity.absent(),
                template_fingerprint=template,
                engine_build=f"engine-{model_id}",
                white_box_flags=WHITE_BOX,
            ),
            flags={"ctx": 2048},
            port=port,
        )
    return (
        definition("alpha", "a" * 64, "1" * 32, port=19801),
        definition("cold", "c" * 64, "3" * 32, port=19802),
    )


def _registry_handshake(definition: WorkerDefinition, generation: int) -> dict:
    key = definition.runtime_key
    return {
        "status": "ok",
        "protocol_version": "1.1",
        "worker_generation_id": f"{definition.model_id}-worker-{generation}",
        "model": definition.model,
        "model_sha256": key.gguf_artifact_sha256,
        "n_ctx": key.context_size,
        "device": "cpu",
        "engine_build": key.engine_build,
        "template_fingerprint": key.template_fingerprint,
        "capabilities": dict(key.white_box_flags),
    }


def _loader_from_registry(registry: WorkerRegistry):
    """Adapt a real WorkerRegistry.ensure_loaded to the router's ColdLoader contract.

    Test-only glue -- mirrors tests/test_model_routing_gateway.py's and
    tests/test_worker_concurrency_gate.py's identical helper. See ColdLoadOutcome's docstring for
    why clozn/server itself must never build this adapter (it would have to import clozn.cli).
    """
    def loader(model_id: str, timeout: float) -> ColdLoadOutcome:
        result = registry.ensure_loaded(model_id, timeout=timeout)
        handle = registry.worker_handle(model_id)
        status = {w["model_id"]: w for w in registry.status()["workers"]}
        return ColdLoadOutcome(
            state=result.state_after.value,
            kind=result.kind,
            outcome=result.outcome,
            coalesced=result.coalesced,
            wait_ms=result.wait_ms,
            worker_port=handle.port if handle is not None else None,
            worker_identity=status[model_id]["worker_identity"],
            failure_code=result.failure_code,
            message=result.error,
            event_id=result.event_id,
        )
    return loader


def _dispatch(method: str, path: str, body=None) -> bytes:
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest-loader"}
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    return handler.wfile.getvalue()


def _response(raw: bytes):
    headers, _, body = raw.partition(b"\r\n\r\n")
    first = headers.splitlines()[0].decode("ascii")
    return int(first.split()[1]), json.loads(body.decode("utf-8"))


def _wire(monkeypatch, directory: Path, router) -> None:
    monkeypatch.setattr(runlog, "RUNS_DIR", str(directory / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(directory / "settings.json"))
    monkeypatch.setattr(cs, "MODEL_ROUTER", router)
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "ENGINE", None)
    # Warm the run store's SQLite schema single-threaded before any concurrent dispatch below --
    # exactly what clozn.server.app.main() itself now does at gateway boot, and for the identical
    # reason tests/test_worker_concurrency_gate.py's _wire_router documents.
    runlog.list_runs(limit=1)


# --- 1. the new seam itself: parameter exists, defaults safely, survives refresh() -----------------


def test_in_memory_router_reaches_a_real_registry_cold_loader(tmp_path, monkeypatch):
    """ADR 008 Stage 2's direct supervisor-to-router seam loads a cold model."""
    alpha_def, cold_def = _definitions()

    def spawn(model, port, flags, **kwargs):
        definition = alpha_def if model == alpha_def.model else cold_def
        return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=2, spawn=spawn, busy_tracking_wired=True,
    )
    registry.start_preloaded()
    router = InMemoryProjectionRouter(
        registry.routing_projection,
        engine_factory=lambda port: FakeEngine(
            "cold", "c" * 64, "3" * 32, "engine-cold"
        ),
        substrate_factory=lambda engine: FakeSub(engine),
        loader=_loader_from_registry(registry),
        worker_call_tracker=registry.track_call,
        gate=WorkerGateRegistry(["alpha", "cold"]),
    )
    _wire(monkeypatch, tmp_path, router)
    try:
        raw = _dispatch("POST", "/v1/chat/completions", {
            "model": "cold",
            "messages": [{"role": "user", "content": "wake up"}],
        })
        status, payload = _response(raw)
        assert status == 200
        logged = runlog.get_run(payload["clozn_run_id"])
        event = logged["meta"]["model_routing"]["result"]["receipt"]["load_event"]
        assert event["kind"] == "cold_load"
        assert event["outcome"] == "loaded"
        assert event["coalesced"] is False
        assert registry.status()["workers"][1]["state"] == "ready"
        assert registry.worker_handle("cold").busy is False
    finally:
        registry.stop_all()


def test_ten_concurrent_http_posts_through_in_memory_router_coalesce_one_load(
    tmp_path, monkeypatch
):
    """The same single-flight proof against ADR 008's in-process construction seam."""
    alpha_def, cold_def = _definitions()
    call_count = 0
    call_lock = threading.Lock()
    spawn_entered = threading.Event()
    release_spawn = threading.Event()

    def spawn(model, port, flags, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        definition = alpha_def if model == alpha_def.model else cold_def
        if definition is cold_def:
            spawn_entered.set()
            assert release_spawn.wait(timeout=5)
        return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=2, spawn=spawn, busy_tracking_wired=True,
    )
    registry.start_preloaded()
    call_count = 0
    router = InMemoryProjectionRouter(
        registry.routing_projection,
        engine_factory=lambda port: FakeEngine(
            "cold", "c" * 64, "3" * 32, "engine-cold"
        ),
        substrate_factory=lambda engine: FakeSub(engine),
        loader=_loader_from_registry(registry),
        worker_call_tracker=registry.track_call,
        gate=WorkerGateRegistry(["alpha", "cold"]),
    )
    _wire(monkeypatch, tmp_path, router)
    try:
        barrier = threading.Barrier(10)

        def send_one(_index):
            barrier.wait(timeout=5)
            return _response(_dispatch("POST", "/v1/chat/completions", {
                "model": "cold",
                "messages": [{"role": "user", "content": "wake up"}],
            }))

        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [pool.submit(send_one, index) for index in range(10)]
            assert spawn_entered.wait(timeout=5)
            time.sleep(0.05)
            release_spawn.set()
            responses = [future.result(timeout=10) for future in futures]

        assert call_count == 1
        assert all(status == 200 for status, _payload in responses)
        events = []
        for _status, payload in responses:
            logged = runlog.get_run(payload["clozn_run_id"])
            events.append(
                logged["meta"]["model_routing"]["result"]["receipt"]["load_event"]
            )
        assert sum(event["coalesced"] is False for event in events) == 1
        assert sum(event["coalesced"] is True for event in events) == 9
        assert registry.worker_handle("cold").busy is False
    finally:
        registry.stop_all()


def test_in_memory_runtime_lifecycle_soak_alternates_eviction_and_restart(
    tmp_path, monkeypatch
):
    """Bounded Stage 3 gate: repeated cold loads never exceed residency or lose busy safety."""
    alpha_def, cold_def = _definitions()
    spawn_calls = []

    def spawn(model, port, flags, **kwargs):
        spawn_calls.append(model)
        definition = alpha_def if model == alpha_def.model else cold_def
        return _FakeSupervisorProcess(), _registry_handshake(definition, len(spawn_calls)), False

    registry = WorkerRegistry(
        [alpha_def, cold_def],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        spawn=spawn,
        busy_tracking_wired=True,
    )
    registry.start_preloaded()

    def engine_for_port(port):
        if port == alpha_def.port:
            return FakeEngine("alpha", "a" * 64, "1" * 32, "engine-alpha")
        return FakeEngine("cold", "c" * 64, "3" * 32, "engine-cold")

    router = InMemoryProjectionRouter(
        registry.routing_projection,
        engine_factory=engine_for_port,
        substrate_factory=lambda engine: FakeSub(engine),
        loader=_loader_from_registry(registry),
        worker_call_tracker=registry.track_call,
        gate=WorkerGateRegistry(["alpha", "cold"]),
    )

    def select_and_release(model_id):
        selection = router.select(
            model_id,
            field_present=True,
            surface="native",
            route="/api/clozn/generate",
        )
        identity = selection.worker_identity["worker_generation_id"]
        if selection.gate_release is not None:
            selection.gate_release()
        return identity

    try:
        for cycle in range(40):
            model_id = "alpha" if cycle % 2 == 0 else "cold"
            selected_identity = select_and_release(model_id)
            status = {item["model_id"]: item for item in registry.status()["workers"]}
            assert status[model_id]["state"] == "ready"
            assert sum(item["state"] == "ready" for item in status.values()) <= 1
            assert registry.worker_handle(model_id).busy is False
            assert selected_identity.startswith(model_id + "-worker-")

            if cycle in {13, 29}:
                handle = registry.worker_handle(model_id)
                handle.process.code = 1
                registry.maintain()
                restarted = {
                    item["model_id"]: item
                    for item in registry.status()["workers"]
                }
                assert restarted[model_id]["state"] == "ready"

        active = "alpha" if registry.worker_handle("alpha") is not None else "cold"
        other = "cold" if active == "alpha" else "alpha"
        with registry.track_call(active):
            try:
                router.select(
                    other,
                    field_present=True,
                    surface="native",
                    route="/api/clozn/generate",
                )
            except ModelRoutingError as error:
                assert error.code == "no_evictable_worker"
            else:
                raise AssertionError("busy resident worker was evicted during soak")
        select_and_release(other)
        assert sum(
            item["state"] == "ready" for item in registry.status()["workers"]
        ) == 1
        assert len(spawn_calls) >= 20
    finally:
        registry.stop_all()
