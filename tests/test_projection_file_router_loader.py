"""RT-04's loader closes onto ProjectionFileRouter -- the class clozn/server/app.py's main()
actually constructs for `clozn serve --models-config`.

Before this change, ``ProjectionFileRouter.__init__`` accepted ``gate=`` but had no ``loader=``
parameter at all: RT-04's cold-load coalescing/eviction (``WorkerRegistry.ensure_loaded``, proven
directly in tests/test_worker_registry.py) and its ``PreloadedModelRouter``-level wiring (proven in
tests/test_model_routing_gateway.py, and -- through real concurrent HTTP dispatch -- in
tests/test_worker_concurrency_gate.py) were reachable everywhere EXCEPT the one router class the
real gateway process actually builds. This file proves the file-backed router closes that specific
gap: it accepts a loader, threads it through every ``refresh()`` rebuild, and drives a real
``WorkerRegistry``'s single-flight coalescing correctly under real concurrent HTTP dispatch -- with
real threads and real synchronization, never a sleep standing in for a lock.

What this file deliberately does NOT claim: production wiring of a live supervisor
``WorkerRegistry`` into clozn/server/app.py's own ``ProjectionFileRouter`` construction remains out
of scope. ``clozn serve``'s gateway process (``python -m clozn.server.app``) is a separate OS
process from the ``clozn serve`` supervisor that owns the real ``WorkerRegistry`` and the model file
paths/flags needed to spawn a worker (see app.py's comment at ``MODEL_ROUTER``'s construction, and
substrates.py's ``_ENGINE_DISCOVERY_ENV_KEYS`` comment for the same process-boundary fact), and
``clozn/server`` must never import ``clozn/cli`` (``ColdLoadOutcome``'s docstring). That cross-process
integration is separately owned -- exactly as tests/test_model_routing_gateway.py's own
``_loader_from_registry`` docstring already says for ``PreloadedModelRouter``. This file only proves
the ``ProjectionFileRouter`` seam itself is real, threaded correctly, and safe under concurrency.
"""
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
from clozn.server.model_routing import ColdLoadOutcome, ProjectionFileRouter
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


def _write_projection(directory: Path, registry: WorkerRegistry) -> Path:
    path = directory / "projection.json"
    path.write_text(json.dumps(registry.routing_projection()), encoding="utf-8")
    return path


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


def test_projection_file_router_accepts_and_exposes_a_loader(tmp_path, monkeypatch):
    alpha_def, cold_def = _definitions()

    def spawn(model, port, flags, **kwargs):
        definition = alpha_def if model == alpha_def.model else cold_def
        return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=2, spawn=spawn,
    )
    registry.start_preloaded()
    projection_path = _write_projection(tmp_path, registry)
    try:
        no_loader_router = ProjectionFileRouter(
            str(projection_path),
            engine_factory=lambda port: FakeEngine("alpha", "a" * 64, "1" * 32, "engine-alpha"),
            substrate_factory=lambda engine: FakeSub(engine),
        )
        assert no_loader_router.loader is None

        loader = _loader_from_registry(registry)
        loaded_router = ProjectionFileRouter(
            str(projection_path),
            engine_factory=lambda port: FakeEngine("cold", "c" * 64, "3" * 32, "engine-cold"),
            substrate_factory=lambda engine: FakeSub(engine),
            loader=loader,
        )
        assert loaded_router.loader is loader
    finally:
        registry.stop_all()


def test_projection_file_router_reuses_the_configured_loader_across_refresh_rebuilds(
    tmp_path, monkeypatch
):
    """refresh() rebuilds the underlying PreloadedModelRouter on every fingerprint change (see
    gate's identical `_built_gate` reuse). The loader must be threaded into EVERY such rebuild, not
    only the constructor's first one -- otherwise a live gateway would silently lose cold-loading
    the moment a sibling worker's routine restart republished the projection file."""
    alpha_def, cold_def = _definitions()
    loader_calls = []

    def spawn(model, port, flags, **kwargs):
        definition = alpha_def if model == alpha_def.model else cold_def
        return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=2, spawn=spawn,
    )
    registry.start_preloaded()
    projection_path = _write_projection(tmp_path, registry)

    def loader(model_id, timeout):
        loader_calls.append(model_id)
        real = _loader_from_registry(registry)
        return real(model_id, timeout)

    router = ProjectionFileRouter(
        str(projection_path),
        engine_factory=lambda port: FakeEngine("cold", "c" * 64, "3" * 32, "engine-cold"),
        substrate_factory=lambda engine: FakeSub(engine),
        loader=loader,
    )
    try:
        # Force a rebuild via the public API -- the same thing a routine projection republish
        # (e.g. a sibling worker's restart) triggers in production.
        assert router.refresh(force=True) is True
        assert router.loader is loader

        selection = router.select(
            "cold", field_present=True, surface="native", route="/api/clozn/generate"
        )
        assert selection.model_id == "cold"
        assert loader_calls == ["cold"]
    finally:
        registry.stop_all()


def test_projection_file_router_without_a_loader_keeps_rt03_fail_fast_byte_identical(
    tmp_path, monkeypatch
):
    """No loader configured (the default) -- ProjectionFileRouter must behave exactly like RT-03:
    a not-ready configured model fails immediately with model_not_ready, never attempting a load."""
    alpha_def, cold_def = _definitions()

    def spawn(model, port, flags, **kwargs):
        if model == cold_def.model:
            raise AssertionError("no loader is configured; cold must never be spawned")
        return _FakeSupervisorProcess(), _registry_handshake(alpha_def, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=2, spawn=spawn,
    )
    registry.start_preloaded()
    projection_path = _write_projection(tmp_path, registry)
    router = ProjectionFileRouter(
        str(projection_path),
        engine_factory=lambda port: FakeEngine("alpha", "a" * 64, "1" * 32, "engine-alpha"),
        substrate_factory=lambda engine: FakeSub(engine),
    )
    assert router.loader is None
    _wire(monkeypatch, tmp_path, router)

    try:
        raw = _dispatch("POST", "/api/generate", {
            "model": "cold", "prompt": "must not load", "stream": False,
        })
        status, payload = _response(raw)
        assert status == 409
        assert payload["code"] == "model_not_ready"
        assert payload["retryable"] is True
        schemas.validate(payload["clozn_model_routing"])
        assert payload["clozn_model_routing"]["result"]["receipt"]["load_event"] == {
            "event_id": None,
            "kind": "not_started",
            "outcome": "not_started",
            "state_before": "unloaded",
            "state_after": "unloaded",
            "coalesced": False,
            "wait_ms": 0,
        }
        assert runlog.list_runs() == []
    finally:
        registry.stop_all()


# --- 2. typed failure state, never a generic exception ---------------------------------------------


def test_projection_file_router_load_failure_is_a_typed_state_not_a_generic_exception(
    tmp_path, monkeypatch
):
    alpha_def, cold_def = _definitions()

    def spawn(model, port, flags, **kwargs):
        if model == cold_def.model:
            raise RuntimeError("boom: engine refused to start")
        return _FakeSupervisorProcess(), _registry_handshake(alpha_def, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=2, spawn=spawn,
    )
    registry.start_preloaded()
    projection_path = _write_projection(tmp_path, registry)
    router = ProjectionFileRouter(
        str(projection_path),
        engine_factory=lambda port: FakeEngine("cold", "c" * 64, "3" * 32, "engine-cold"),
        substrate_factory=lambda engine: FakeSub(engine),
        loader=_loader_from_registry(registry),
    )
    _wire(monkeypatch, tmp_path, router)

    try:
        raw = _dispatch("POST", "/api/chat", {
            "model": "cold",
            "messages": [{"role": "user", "content": "wake up"}],
            "stream": False,
        })
        status, payload = _response(raw)
        assert status == 503
        assert payload["code"] == "model_load_failed"
        assert payload["retryable"] is True
        schemas.validate(payload["clozn_model_routing"])
        load_event = payload["clozn_model_routing"]["result"]["receipt"]["load_event"]
        assert load_event["outcome"] == "failed"
        assert load_event["state_after"] == "failed"
        assert load_event["coalesced"] is False
        assert runlog.list_runs() == []

        status_after = {w["model_id"]: w for w in registry.status()["workers"]}
        assert status_after["cold"]["state"] == "failed"
        assert status_after["cold"]["failure_code"] == "model_load_failed"
    finally:
        registry.stop_all()


# --- 3. eviction never touches in-flight work; resident limit is never exceeded --------------------


def test_projection_file_router_eviction_respects_in_flight_work_and_resident_limit(
    tmp_path, monkeypatch
):
    """max_loaded_workers=1: alpha is resident and has REAL in-flight work (tracked via
    WorkerRegistry.track_call). Cold-loading a second model must fail closed with
    no_evictable_worker through the exact ProjectionFileRouter+HTTP path -- alpha must never be
    evicted out from under its in-flight call, and the resident count must never exceed the
    configured limit."""
    alpha_def, cold_def = _definitions()
    spawn_calls = []

    def spawn(model, port, flags, **kwargs):
        spawn_calls.append(model)
        definition = alpha_def if model == alpha_def.model else cold_def
        return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

    registry = WorkerRegistry(
        [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
        max_loaded_workers=1, spawn=spawn,
        # alpha's busy state below is injected for real via track_call(); tell the
        # registry to trust it -- see UnverifiableWorkerStateError / ADR 006. Without
        # this, eviction would fail closed with no_verifiable_idle_worker instead of
        # no_evictable_worker, because nothing else in this test's construction
        # declares a trustworthy busy signal.
        busy_tracking_wired=True,
    )
    registry.start_preloaded()
    projection_path = _write_projection(tmp_path, registry)
    router = ProjectionFileRouter(
        str(projection_path),
        engine_factory=lambda port: FakeEngine("cold", "c" * 64, "3" * 32, "engine-cold"),
        substrate_factory=lambda engine: FakeSub(engine),
        loader=_loader_from_registry(registry),
    )
    _wire(monkeypatch, tmp_path, router)

    try:
        with registry.track_call("alpha"):  # alpha has real, tracked in-flight work
            raw = _dispatch("POST", "/api/chat", {
                "model": "cold",
                "messages": [{"role": "user", "content": "no room"}],
                "stream": False,
            })
        status, payload = _response(raw)
        assert status == 503
        assert payload["code"] == "no_evictable_worker"
        assert payload["retryable"] is True
        schemas.validate(payload["clozn_model_routing"])

        status_after = {w["model_id"]: w for w in registry.status()["workers"]}
        assert status_after["alpha"]["state"] == "ready"  # never evicted
        assert status_after["cold"]["state"] == "failed"  # cold load failed closed instead
        resident = sum(1 for w in status_after.values() if w["state"] == "ready")
        assert resident == 1 == registry.max_loaded_workers
        assert spawn_calls == [alpha_def.model]  # cold's spawn never even ran
    finally:
        registry.stop_all()


# --- 4. the acceptance test: N concurrent HTTP POSTs for one cold model cause exactly ONE load -----


def test_ten_concurrent_http_posts_through_projection_file_router_cause_exactly_one_load(
    tmp_path, monkeypatch
):
    """The specific gap this ticket closes.

    Not PreloadedModelRouter.select() called directly (tests/test_model_routing_gateway.py already
    proves that invariant at the router level). Not a hand-built PreloadedModelRouter monkeypatched
    onto MODEL_ROUTER (tests/test_worker_concurrency_gate.py already proves THAT through real HTTP
    dispatch). Specifically ProjectionFileRouter -- the class clozn/server/app.py's main() actually
    constructs for `clozn serve --models-config` -- now threading a real loader through its
    file-backed refresh() cycle, driven by ten real concurrent HTTP POSTs through the real do_POST
    dispatch path, for a model that starts genuinely cold (never preloaded).

    Flake-hunted: builds a completely fresh registry/router/projection/run-store each of ITERATIONS
    times below, so every iteration independently races the same single-flight guarantee from
    scratch. See the printed summary (and the returned counts) for the real executed/failed totals.
    """
    ITERATIONS = 25
    THREADS = 10
    total_requests = 0
    iteration_failures = []

    for iteration in range(ITERATIONS):
        iter_dir = tmp_path / f"iter{iteration}"
        iter_dir.mkdir()
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
                assert release_spawn.wait(timeout=5), "test setup: release_spawn was never set"
            return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

        registry = WorkerRegistry(
            [alpha_def, cold_def], default_model_id="alpha", preload_model_ids=["alpha"],
            max_loaded_workers=2, spawn=spawn,
        )
        registry.start_preloaded()
        call_count = 0  # reset: only "cold"'s spawns below are under test
        projection_path = _write_projection(iter_dir, registry)

        router = ProjectionFileRouter(
            str(projection_path),
            engine_factory=lambda port: FakeEngine("cold", "c" * 64, "3" * 32, "engine-cold"),
            substrate_factory=lambda engine: FakeSub(engine),
            loader=_loader_from_registry(registry),
            gate=WorkerGateRegistry(["alpha", "cold"]),
        )
        _wire(monkeypatch, iter_dir, router)

        start_barrier = threading.Barrier(THREADS)

        def send_one(_index):
            start_barrier.wait(timeout=5)
            raw = _dispatch("POST", "/v1/chat/completions", {
                "model": "cold",
                "messages": [{"role": "user", "content": "wake up"}],
            })
            return _response(raw)

        try:
            with ThreadPoolExecutor(max_workers=THREADS) as pool:
                futures = [pool.submit(send_one, i) for i in range(THREADS)]
                assert spawn_entered.wait(timeout=5), "loader never reached spawn"
                # Let the other nine requests genuinely reach the registry's coalescing wait before
                # releasing the spawn -- this is scheduling generosity, not the correctness
                # mechanism itself (that's call_count and the coalesced-flag counts asserted below,
                # both computed from real post-hoc state, never from timing).
                time.sleep(0.05)
                release_spawn.set()
                responses = [f.result(timeout=10) for f in futures]

            total_requests += THREADS
            assert call_count == 1, (
                f"iteration {iteration}: {call_count} spawns for {THREADS} concurrent requests"
            )
            assert all(status == 200 for status, _ in responses), (
                f"iteration {iteration}: statuses {[s for s, _ in responses]}"
            )
            coalesced = []
            worker_ids = set()
            for _status, payload in responses:
                logged = runlog.get_run(payload["clozn_run_id"])
                receipt = logged["meta"]["model_routing"]["result"]["receipt"]
                schemas.validate(logged["meta"]["model_routing"])
                coalesced.append(receipt["load_event"]["coalesced"])
                worker_ids.add(receipt["worker_identity"]["worker_id"])
            assert coalesced.count(False) == 1 and coalesced.count(True) == THREADS - 1, (
                f"iteration {iteration}: coalesced flags {coalesced}"
            )
            assert worker_ids == {"cold-worker-1"}, (
                f"iteration {iteration}: worker_ids {worker_ids}"
            )
            status_after = {w["model_id"]: w for w in registry.status()["workers"]}
            assert status_after["cold"]["state"] == "ready"
            resident = sum(1 for w in status_after.values() if w["state"] == "ready")
            assert resident <= registry.max_loaded_workers
        except AssertionError as error:
            iteration_failures.append((iteration, str(error)))
        finally:
            registry.stop_all()

    print(
        f"\nflake-hunt: {ITERATIONS} iterations x {THREADS} threads = "
        f"{total_requests} HTTP requests executed through the real dispatch path, "
        f"{len(iteration_failures)} iteration failures"
    )
    assert not iteration_failures, iteration_failures
