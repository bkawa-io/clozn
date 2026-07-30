"""RT-05: real per-worker concurrency through the actual HTTP dispatch path.

RT-04's own finding was that the ten-concurrent-cold-requests coalescing test could not be written at the
HTTP layer: clozn/server/app.py's POST_GATE was a single global lock serializing every POST, so RT-04's
single-flight guarantee (WorkerRegistry.ensure_loaded, proven directly in tests/test_worker_registry.py)
was correct but LATENT -- concurrent HTTP requests never actually overlapped. RT-05 replaces that one
global turn with a per-worker generation gate (clozn.server.request_gate.WorkerGateRegistry), acquired
INSIDE clozn.server.model_routing.select_for_handler AFTER model selection/cold-load succeeds (never
before -- gating any earlier would itself re-serialize the requests RT-04's coalescing needs to see arrive
concurrently).

This file proves, with real threads and real synchronization (never a sleep standing in for a lock):
  1. the HTTP-layer coalescing test that was impossible before RT-05 now passes;
  2. a GET never queues behind a slow in-flight generation;
  3. two different workers generate concurrently; two requests to the SAME worker serialize;
  4. run isolation holds -- no state bleed between overlapping runs, including the specific hazard
     request_context.py's module docstring documents (one shared self._request-shaped attribute torn by
     an interleaved second call).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
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
    PreloadedModelBinding,
    PreloadedModelRouter,
)
from clozn.server.request_gate import WorkerGateRegistry


SHA_ALPHA = "a" * 64
SHA_BETA = "b" * 64
SHA_COLD = "c" * 64
WHITE_BOX = {"sae": False, "jlens": False, "attn_knockout": False}


# --- shared fakes -------------------------------------------------------------------------------------


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


class ProbeFakeSub:
    """A per-worker fake substrate whose chat() blocks until released and
    records real enter/exit timestamps on a shared ConcurrencyProbe -- so a
    test can assert genuine overlap (or its absence) rather than inferring
    it from ordering alone.

    Also reproduces the SPECIFIC hazard clozn/server/request_context.py's
    module docstring documents: chat() publishes onto ONE shared
    self._request-shaped attribute, and last_finish_reason() (called
    separately, by _log_run, AFTER chat() returns) reads it back. If two
    chat() calls on the SAME sub ever truly overlapped, a second call's
    write could land between the first call's return and the first
    caller's read of last_finish_reason() -- silently handing the first
    caller the SECOND call's answer. This is real, not decorative: it is
    the actual reason EngineSubstrate needs same-worker generation
    serialized (see substrates.py's RequestContext usage), reproduced here
    with a minimal stand-in so this test exercises the real hazard shape
    instead of a fake that happens not to have it.
    """

    name = "engine"
    brain = None

    def __init__(self, engine: FakeEngine, probe: "ConcurrencyProbe", *, release: threading.Event | None = None):
        self.engine = engine
        self.steer = FakeSteer()
        self.calls = []
        self._probe = probe
        self._release = release
        self._request = None  # the shared, torn-read-prone attribute

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        call_id = f"{self.engine.model_id}-{len(self.calls)}"
        self.calls.append({"messages": [dict(m) for m in messages], "call_id": call_id})
        self._probe.enter(self.engine.model_id, call_id)
        try:
            self._request = call_id
            if self._release is not None:
                assert self._release.wait(timeout=10), f"{call_id} was never released"
            if mem_out is not None:
                mem_out.update(
                    assembled_messages=[dict(m) for m in messages],
                    final_prompt=f"<{call_id}>rendered</{call_id}>",
                    actual_prompt_tokens=7,
                )
            if trace_out is not None:
                trace_out.append({"pos": 0, "token_id": 1, "piece": call_id, "prob": 1.0, "alts": []})
            return f"{call_id} reply"
        finally:
            self._probe.exit(self.engine.model_id, call_id)

    def last_finish_reason(self):
        # Reads the SAME shared attribute chat() wrote -- a torn interleave
        # would surface here as the wrong call_id for the caller reading it.
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


class ConcurrencyProbe:
    """Real-time peak-concurrency tracker: enter()/exit() around the section
    under test, from any number of threads. max_active is the true observed
    peak; events is the ordered enter/exit log for pairwise overlap checks."""

    def __init__(self):
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.events = []  # (kind, worker_id, call_id, monotonic_ts)

    def enter(self, worker_id, call_id):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.events.append(("enter", worker_id, call_id, time.monotonic()))

    def exit(self, worker_id, call_id):
        with self._lock:
            self.active -= 1
            self.events.append(("exit", worker_id, call_id, time.monotonic()))

    def intervals(self, call_id):
        starts = [e[3] for e in self.events if e[0] == "enter" and e[2] == call_id]
        ends = [e[3] for e in self.events if e[0] == "exit" and e[2] == call_id]
        assert starts and ends, f"{call_id} never completed"
        return starts[0], ends[0]


def _runtime_key(digest: str, template: str, build: str) -> dict:
    return RuntimeKey(
        gguf_artifact_sha256=digest,
        context_size=2048,
        backend="cpu",
        adapter=AdapterRuntimeIdentity.absent(),
        template_fingerprint=template,
        engine_build=build,
        white_box_flags=WHITE_BOX,
    ).as_dict()


def _ready_binding(model_id: str, digest: str, template: str, probe: ConcurrencyProbe,
                    *, release: threading.Event | None = None, sub_class=ProbeFakeSub):
    build = f"engine-{model_id}"
    engine = FakeEngine(model_id, digest, template, build)
    sub = sub_class(engine, probe, release=release)
    key = _runtime_key(digest, template, build)
    binding = PreloadedModelBinding(
        model_id=model_id,
        resolved_artifact={"model_id": model_id, "format": "gguf", "artifact_sha256": digest},
        runtime_key=key,
        adapter=AdapterRuntimeIdentity.absent().as_dict(),
        state="ready",
        worker_identity={
            "worker_id": engine.worker_generation_id,
            "worker_generation_id": engine.worker_generation_id,
            "worker_generation": 1,
            "runtime_key_sha256": key["key_sha256"],
            "protocol_version": "1.1",
            "engine_build": build,
            "backend": "cpu",
        },
        sub=sub,
        engine=engine,
    )
    return binding, sub


def _dispatch(method: str, path: str, body=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest-concurrency"}
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


def _wire_router(monkeypatch, tmp_path, router):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "MODEL_ROUTER", router)
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "ENGINE", None)
    # Warm the run store's SQLite schema single-threaded, exactly as clozn.server.app.main() now does
    # at gateway boot (see that function). Without this, the FIRST concurrent requests in a test below
    # can race clozn/runs/store.py's own from-scratch schema creation -- a real, pre-existing gap in a
    # file this ticket does not own (discovered via this exact test file; see the RT-05 commit message),
    # now reachable because these tests genuinely run requests concurrently for the first time. This
    # mirrors the production fix, not a workaround specific to tests.
    runlog.list_runs(limit=1)


# --- 1. the HTTP-layer coalescing test that was impossible before RT-05 -------------------------------


class _FakeSupervisorProcess:
    _next_pid = 9000

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


def _registry_worker_definition(model_id: str, digest: str, template: str, *, port: int):
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


def _registry_handshake(definition, generation: int) -> dict:
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
    """Adapt a real WorkerRegistry.ensure_loaded to the router's loader contract (test-only glue -- see
    ColdLoadOutcome's docstring for why clozn/server must never import clozn/cli directly)."""
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


def test_ten_concurrent_http_posts_for_one_cold_model_cause_exactly_one_load(tmp_path, monkeypatch):
    """The test RT-04's own writeup said could not be written at the HTTP layer.

    Ten real threads dispatch full HTTP /v1/chat/completions requests, through the real do_POST, for the
    same never-loaded model at once. The router's loader is a real WorkerRegistry with a counting fake
    spawn: exactly one process may be spawned, all ten requests must still succeed with 200, and this
    only works at all because RT-05 gates AFTER selection -- gating before it (the pre-RT-05 global
    POST_GATE) would have serialized these ten requests before any of them reached the loader.
    """
    probe = ConcurrencyProbe()
    alpha_binding, alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe)
    cold_binding = PreloadedModelBinding(
        model_id="cold",
        resolved_artifact={"model_id": "cold", "format": "gguf", "artifact_sha256": SHA_COLD},
        runtime_key=_runtime_key(SHA_COLD, "3" * 32, "engine-cold"),
        adapter=AdapterRuntimeIdentity.absent().as_dict(),
        state="unloaded",
        worker_identity=None,
        sub=None,
        engine=None,
        preloaded=False,
    )
    cold_key_sha = cold_binding.runtime_key["key_sha256"]

    registry_alpha_def = _registry_worker_definition("alpha", SHA_ALPHA, "1" * 32, port=9901)
    registry_cold_def = _registry_worker_definition("cold", SHA_COLD, "3" * 32, port=9902)
    assert registry_cold_def.runtime_key.key_sha256 == cold_key_sha

    call_count = 0
    call_lock = threading.Lock()
    spawn_entered = threading.Event()
    release_spawn = threading.Event()
    blocking = False

    def spawn(model, port, flags, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        definition = registry_cold_def if model == registry_cold_def.model else registry_alpha_def
        if blocking:
            spawn_entered.set()
            assert release_spawn.wait(timeout=5), "test setup: release_spawn was never set"
        return _FakeSupervisorProcess(), _registry_handshake(definition, 1), False

    registry = WorkerRegistry(
        [registry_alpha_def, registry_cold_def],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()
    call_count = 0
    blocking = True

    cold_reply_engine_by_port = {}

    def engine_factory(port):
        engine = FakeEngine("cold", SHA_COLD, "3" * 32, "engine-cold")
        cold_reply_engine_by_port[port] = engine
        return engine

    def substrate_factory(engine):
        return ProbeFakeSub(engine, probe)

    router = PreloadedModelRouter(
        [alpha_binding, cold_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        loader=_loader_from_registry(registry),
        engine_factory=engine_factory,
        substrate_factory=substrate_factory,
        gate=WorkerGateRegistry(["alpha", "cold"]),
    )
    _wire_router(monkeypatch, tmp_path, router)

    start_barrier = threading.Barrier(10)

    def send_one(_index):
        start_barrier.wait(timeout=5)
        raw = _dispatch("POST", "/v1/chat/completions", {
            "model": "cold",
            "messages": [{"role": "user", "content": "wake up"}],
        })
        return _response(raw)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(send_one, i) for i in range(10)]
        assert spawn_entered.wait(timeout=5), "loader never reached spawn"
        time.sleep(0.1)  # let the other nine requests reach the registry's coalescing wait
        release_spawn.set()
        responses = [f.result(timeout=10) for f in futures]

    assert call_count == 1, "exactly one process must be spawned for ten concurrent HTTP requests"
    assert all(status == 200 for status, _payload in responses)
    assert all(
        payload["choices"][0]["message"]["content"].startswith("cold-")
        for _status, payload in responses
    )
    assert alpha_sub.calls == []

    worker_ids, coalesced_flags = set(), []
    for _status, payload in responses:
        logged = runlog.get_run(payload["clozn_run_id"])
        receipt = logged["meta"]["model_routing"]["result"]["receipt"]
        schemas.validate(logged["meta"]["model_routing"])
        worker_ids.add(receipt["worker_identity"]["worker_id"])
        coalesced_flags.append(receipt["load_event"]["coalesced"])
    assert worker_ids == {"cold-worker-1"}
    assert coalesced_flags.count(False) == 1
    assert coalesced_flags.count(True) == 9

    status = {w["model_id"]: w for w in registry.status()["workers"]}
    assert status["cold"]["state"] == "ready"


# --- 2. reads never queue behind a slow generation ------------------------------------------------------


def test_get_never_queues_behind_a_slow_in_flight_generation(tmp_path, monkeypatch):
    """A GET must return promptly even while a generation is genuinely blocked in flight.

    do_GET has no gate at all (see app.py) -- this proves it under real overlap, not by code inspection:
    a generation is held open on a real ProbeFakeSub.chat() call, and a concurrent GET must complete well
    before that generation is released.
    """
    probe = ConcurrencyProbe()
    release = threading.Event()
    alpha_binding, alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe, release=release)
    router = PreloadedModelRouter(
        [alpha_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        gate=WorkerGateRegistry(["alpha"]),
    )
    _wire_router(monkeypatch, tmp_path, router)

    def send_generation():
        return _response(_dispatch("POST", "/v1/chat/completions", {
            "model": "alpha",
            "messages": [{"role": "user", "content": "take a while"}],
        }))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(send_generation)
        deadline = time.monotonic() + 5
        while probe.active == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert probe.active == 1, "generation never actually started"

        get_started = time.monotonic()
        raw = _dispatch("GET", "/healthz")
        get_elapsed = time.monotonic() - get_started
        status, payload = _response(raw)
        assert status == 200
        assert payload["status"] == "ok"
        # Generous relative to the 5s+ the still-blocked generation would take if the GET were queued
        # behind it; this only needs to prove "did not wait for release", not race a tight bound.
        assert get_elapsed < 1.0, f"GET took {get_elapsed:.3f}s -- it queued behind the generation"

        release.set()
        gen_status, gen_payload = future.result(timeout=5)

    assert gen_status == 200
    assert gen_payload["choices"][0]["message"]["content"].startswith("alpha-")


def test_investigation_and_workbench_gets_never_queue_behind_a_slow_generation(tmp_path, monkeypatch):
    """The two specific routes called out as pure projections that touch no worker.

    A made-up run id 404s immediately either way; what this proves is that the 404 arrives promptly while
    a real generation is held open, not that the run exists.
    """
    probe = ConcurrencyProbe()
    release = threading.Event()
    alpha_binding, _alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe, release=release)
    router = PreloadedModelRouter(
        [alpha_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        gate=WorkerGateRegistry(["alpha"]),
    )
    _wire_router(monkeypatch, tmp_path, router)

    def send_generation():
        return _response(_dispatch("POST", "/v1/chat/completions", {
            "model": "alpha",
            "messages": [{"role": "user", "content": "take a while"}],
        }))

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(send_generation)
        deadline = time.monotonic() + 5
        while probe.active == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert probe.active == 1, "generation never actually started"

        for path in ("/runs/does-not-exist/investigation", "/runs/does-not-exist/tokens/0/workbench"):
            started = time.monotonic()
            raw = _dispatch("GET", path)
            elapsed = time.monotonic() - started
            status, _payload = _response(raw)
            assert status == 404
            assert elapsed < 1.0, f"GET {path} took {elapsed:.3f}s -- it queued behind the generation"

        release.set()
        gen_status, _gen_payload = future.result(timeout=5)
    assert gen_status == 200


# --- 3. cross-worker parallelism, same-worker serialization --------------------------------------------


def test_two_different_workers_generate_concurrently(tmp_path, monkeypatch):
    """The core RT-05 proof: worker A and worker B must overlap in time."""
    probe = ConcurrencyProbe()
    barrier = threading.Barrier(2, timeout=5)

    class _RendezvousSub(ProbeFakeSub):
        def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
            call_id = f"{self.engine.model_id}-{len(self.calls)}"
            self.calls.append({"call_id": call_id})
            self._probe.enter(self.engine.model_id, call_id)
            try:
                self._request = call_id
                barrier.wait()  # both workers must be "inside" simultaneously, or this raises/deadlocks
                if mem_out is not None:
                    mem_out.update(assembled_messages=list(messages), final_prompt=f"<{call_id}>",
                                    actual_prompt_tokens=7)
                if trace_out is not None:
                    trace_out.append({"pos": 0, "token_id": 1, "piece": call_id, "prob": 1.0, "alts": []})
                return f"{call_id} reply"
            finally:
                self._probe.exit(self.engine.model_id, call_id)

    alpha_binding, _ = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe, sub_class=_RendezvousSub)
    beta_binding, _ = _ready_binding("beta", SHA_BETA, "2" * 32, probe, sub_class=_RendezvousSub)

    router = PreloadedModelRouter(
        [alpha_binding, beta_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        gate=WorkerGateRegistry(["alpha", "beta"]),
    )
    _wire_router(monkeypatch, tmp_path, router)

    def send(model_id):
        return _response(_dispatch("POST", "/v1/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": "go"}],
        }))

    with ThreadPoolExecutor(max_workers=2) as pool:
        alpha_future = pool.submit(send, "alpha")
        beta_future = pool.submit(send, "beta")
        # If these were still serialized (RT-04-era global gate), the SECOND request could never even
        # reach chat() until the first fully finished -- the barrier() above would then time out and
        # raise BrokenBarrierError inside whichever thread's chat() got there first, which surfaces as a
        # 500 below rather than a hang, so this test fails loudly instead of stalling the suite.
        alpha_status, alpha_payload = alpha_future.result(timeout=10)
        beta_status, beta_payload = beta_future.result(timeout=10)

    assert alpha_status == 200 and beta_status == 200
    assert alpha_payload["choices"][0]["message"]["content"].startswith("alpha-")
    assert beta_payload["choices"][0]["message"]["content"].startswith("beta-")
    assert probe.max_active == 2, "alpha and beta generations never actually overlapped"


def test_two_requests_to_the_same_worker_serialize(tmp_path, monkeypatch):
    """Two requests naming the SAME worker must never overlap -- proven by real timestamps, not ordering."""
    probe = ConcurrencyProbe()
    release_first = threading.Event()
    alpha_binding, alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe, release=release_first)
    gate = WorkerGateRegistry(["alpha"])
    router = PreloadedModelRouter(
        [alpha_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        gate=gate,
    )
    _wire_router(monkeypatch, tmp_path, router)

    def send():
        return _response(_dispatch("POST", "/v1/chat/completions", {
            "model": "alpha",
            "messages": [{"role": "user", "content": "go"}],
        }))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(send)
        deadline = time.monotonic() + 5
        while probe.active == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert probe.active == 1, "first request never actually started"

        second = pool.submit(send)
        # Confirm the second request is genuinely queued on alpha's gate (not merely "hasn't run yet") --
        # a real predicate, not a sleep, and NOT reached via chat() (which would mean it slipped past
        # the gate: alpha_sub's release_first is still unset, so a second chat() call would hang on it,
        # which the timeout below turns into a loud failure rather than a false pass).
        deadline = time.monotonic() + 5
        while gate.snapshot()["alpha"]["waiting"] == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        assert gate.snapshot()["alpha"]["waiting"] == 1, "second request never reached the gate"
        assert probe.active == 1, "second request entered chat() before the first was released"

        release_first.set()
        first_status, first_payload = first.result(timeout=5)
        # Second call also blocks on the SAME release_first event (ProbeFakeSub shares it) -- it is
        # already set now, so this returns promptly once it gets its turn.
        second_status, second_payload = second.result(timeout=5)

    assert first_status == 200 and second_status == 200
    assert len(alpha_sub.calls) == 2
    first_call_id = alpha_sub.calls[0]["call_id"]
    second_call_id = alpha_sub.calls[1]["call_id"]
    first_start, first_end = probe.intervals(first_call_id)
    second_start, second_end = probe.intervals(second_call_id)
    assert first_end <= second_start, (
        f"same-worker calls overlapped: first ran [{first_start}, {first_end}], "
        f"second ran [{second_start}, {second_end}]"
    )
    assert probe.max_active == 1, "same-worker requests must never be simultaneously active"


# --- 4. run isolation under concurrency -----------------------------------------------------------------


def test_run_isolation_holds_under_concurrent_cross_worker_generation(tmp_path, monkeypatch):
    """No state bleed between overlapping runs: each journaled run must carry exactly its own worker's
    identity, response, and receipt -- never another concurrently-running request's."""
    probe = ConcurrencyProbe()
    barrier = threading.Barrier(2, timeout=5)

    class _RendezvousSub(ProbeFakeSub):
        def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
            call_id = f"{self.engine.model_id}-{len(self.calls)}"
            self.calls.append({"call_id": call_id})
            self._probe.enter(self.engine.model_id, call_id)
            try:
                self._request = call_id
                barrier.wait()
                if mem_out is not None:
                    mem_out.update(assembled_messages=list(messages), final_prompt=f"<{call_id}>",
                                    actual_prompt_tokens=7)
                if trace_out is not None:
                    trace_out.append({"pos": 0, "token_id": 1, "piece": call_id, "prob": 1.0, "alts": []})
                return f"{call_id} reply"
            finally:
                self._probe.exit(self.engine.model_id, call_id)

    alpha_binding, _ = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe, sub_class=_RendezvousSub)
    beta_binding, _ = _ready_binding("beta", SHA_BETA, "2" * 32, probe, sub_class=_RendezvousSub)

    router = PreloadedModelRouter(
        [alpha_binding, beta_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        gate=WorkerGateRegistry(["alpha", "beta"]),
    )
    _wire_router(monkeypatch, tmp_path, router)

    def send(model_id, question):
        return _response(_dispatch("POST", "/v1/chat/completions", {
            "model": model_id,
            "messages": [{"role": "user", "content": question}],
        }))

    with ThreadPoolExecutor(max_workers=2) as pool:
        alpha_future = pool.submit(send, "alpha", "alpha question")
        beta_future = pool.submit(send, "beta", "beta question")
        alpha_status, alpha_payload = alpha_future.result(timeout=10)
        beta_status, beta_payload = beta_future.result(timeout=10)

    assert alpha_status == 200 and beta_status == 200
    assert probe.max_active == 2  # confirms this actually raced, not just two sequential calls

    alpha_run = runlog.get_run(alpha_payload["clozn_run_id"])
    beta_run = runlog.get_run(beta_payload["clozn_run_id"])

    # Each run's own model/response/identity must match ONLY its own worker -- never the concurrently
    # racing sibling's.
    assert alpha_run["model"] == "alpha"
    assert beta_run["model"] == "beta"
    assert alpha_run["response"].startswith("alpha-")
    assert beta_run["response"].startswith("beta-")
    assert alpha_run["identity"]["model_sha256"] == SHA_ALPHA
    assert beta_run["identity"]["model_sha256"] == SHA_BETA
    assert "beta" not in alpha_run["response"]
    assert "alpha" not in beta_run["response"]

    alpha_receipt = alpha_run["meta"]["model_routing"]["result"]["receipt"]
    beta_receipt = beta_run["meta"]["model_routing"]["result"]["receipt"]
    schemas.validate(alpha_run["meta"]["model_routing"])
    schemas.validate(beta_run["meta"]["model_routing"])
    assert alpha_receipt["resolved_model_id"] == "alpha"
    assert beta_receipt["resolved_model_id"] == "beta"
    assert alpha_receipt["worker_identity"]["worker_id"] == "alpha-worker-1"
    assert beta_receipt["worker_identity"]["worker_id"] == "beta-worker-1"
    assert alpha_receipt["runtime_key"]["gguf_artifact_sha256"] == SHA_ALPHA
    assert beta_receipt["runtime_key"]["gguf_artifact_sha256"] == SHA_BETA
    # The two run ids themselves must be distinct -- a real collision would mean one run silently
    # overwrote the other.
    assert alpha_payload["clozn_run_id"] != beta_payload["clozn_run_id"]


def test_run_isolation_holds_when_two_concurrent_requests_share_one_worker(tmp_path, monkeypatch):
    """The same-worker case: two overlapping-in-time requests to alpha must still each get their own,
    uncorrupted receipt/response even though they serialize through the same gate."""
    probe = ConcurrencyProbe()
    release_first = threading.Event()
    alpha_binding, alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32, probe, release=release_first)
    router = PreloadedModelRouter(
        [alpha_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        gate=WorkerGateRegistry(["alpha"]),
    )
    _wire_router(monkeypatch, tmp_path, router)

    def send(question):
        return _response(_dispatch("POST", "/v1/chat/completions", {
            "model": "alpha",
            "messages": [{"role": "user", "content": question}],
        }))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(send, "first question")
        deadline = time.monotonic() + 5
        while probe.active == 0 and time.monotonic() < deadline:
            time.sleep(0.005)
        second = pool.submit(send, "second question")
        time.sleep(0.05)  # let the second request genuinely reach and queue on the gate
        release_first.set()
        first_status, first_payload = first.result(timeout=5)
        second_status, second_payload = second.result(timeout=5)

    assert first_status == 200 and second_status == 200
    assert first_payload["clozn_run_id"] != second_payload["clozn_run_id"]

    first_run = runlog.get_run(first_payload["clozn_run_id"])
    second_run = runlog.get_run(second_payload["clozn_run_id"])
    # Each run's response must match its OWN call, never the other's -- the exact torn-read hazard
    # ProbeFakeSub.last_finish_reason() is built to expose (see its docstring).
    assert first_run["response"] == first_payload["choices"][0]["message"]["content"]
    assert second_run["response"] == second_payload["choices"][0]["message"]["content"]
    assert first_run["response"] != second_run["response"]
