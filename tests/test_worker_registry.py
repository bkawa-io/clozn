"""Model-free tests for the ADR 004 preloaded worker registry."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import threading
import time

import pytest

from clozn.cli.worker_registry import (
    AdapterRuntimeIdentity,
    EvictionTimeoutError,
    RuntimeKey,
    UnknownWorkerModelError,
    UnverifiableWorkerStateError,
    WorkerBusyError,
    WorkerDefinition,
    WorkerLifecycleState,
    WorkerRegistry,
    WorkerRegistryConfigError,
)


SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
TEMPLATE_A = "1" * 32
TEMPLATE_B = "2" * 32
BASE_WHITE_BOX_FLAGS = {
    "sae": False,
    "jlens": False,
    "attn_knockout": False,
}


class FakeProcess:
    next_pid = 5000

    def __init__(self, code=None):
        type(self).next_pid += 1
        self.pid = type(self).next_pid
        self.code = code
        self.terminated = False

    def poll(self):
        return self.code

    def terminate(self):
        self.terminated = True
        self.code = -15

    def kill(self):
        self.terminated = True
        self.code = -9

    def wait(self, timeout=None):
        return self.code


def runtime_key(
    *,
    model_sha=SHA_A,
    context_size=2048,
    backend="cpu",
    adapter=None,
    template=TEMPLATE_A,
    engine_build="engine-1",
    flags=None,
):
    return RuntimeKey(
        gguf_artifact_sha256=model_sha,
        context_size=context_size,
        backend=backend,
        adapter=adapter or AdapterRuntimeIdentity.absent(),
        template_fingerprint=template,
        engine_build=engine_build,
        white_box_flags=flags or {},
    )


def definition(
    model_id,
    *,
    model_sha=SHA_A,
    port=None,
    backend="cpu",
    adapter=None,
    template=TEMPLATE_A,
    flags=None,
):
    key = runtime_key(
        model_sha=model_sha,
        backend=backend,
        adapter=adapter,
        template=template,
        flags=flags if flags is not None else BASE_WHITE_BOX_FLAGS,
    )
    launch_flags = {"ctx": 2048}
    if key.adapter.present:
        launch_flags.update({
            "adapter": f"{model_id}.lora.gguf",
            "adapter_scale": key.adapter.scale,
        })
    return WorkerDefinition(
        model_id=model_id,
        model=f"{model_id}.gguf",
        runtime_key=key,
        flags=launch_flags,
        prefer_gpu=backend != "cpu",
        port=port,
    )


def handshake(definition, generation, *, overrides=None):
    key = definition.runtime_key
    health = {
        "status": "ok",
        "protocol_version": "1.1",
        "worker_generation_id": f"worker-{generation}",
        "model": definition.model,
        "model_sha256": key.gguf_artifact_sha256,
        "n_ctx": key.context_size,
        "device": key.backend if key.backend in {"cpu", "cuda"} else "cuda",
        "mode": "autoregressive",
        "capabilities": dict(key.white_box_flags),
    }
    if key.adapter.present:
        health["lora"] = {
            "path": definition.flags["adapter"],
            "scale": key.adapter.scale,
        }
    health.update(overrides or {})
    return health


def test_runtime_key_is_canonical_and_every_declared_facet_is_load_bearing():
    left = runtime_key(flags={"z_flag": False, "a_flag": True})
    right = runtime_key(flags={"a_flag": True, "z_flag": False})
    assert left.key_sha256 == right.key_sha256
    assert left.key_sha256 == (
        "737a45efcfe3db9f88b0fa2f06e6b75b815257961df42f788b152a73231fb814"
    )
    assert left.as_dict()["white_box_flags"] == {
        "a_flag": True, "z_flag": False,
    }

    variants = (
        runtime_key(model_sha=SHA_B),
        runtime_key(context_size=4096),
        runtime_key(backend="cuda"),
        runtime_key(
            adapter=AdapterRuntimeIdentity(
                present=True,
                identity_sha256=SHA_C,
                artifact_sha256=SHA_D,
                scale=0.5,
            ),
        ),
        runtime_key(template=TEMPLATE_B),
        runtime_key(engine_build="engine-2"),
        runtime_key(flags={"jlens": True}),
    )
    assert all(variant.key_sha256 != left.key_sha256 for variant in variants)


def test_runtime_key_and_launch_facets_fail_closed():
    with pytest.raises(WorkerRegistryConfigError, match="lowercase 64-character"):
        runtime_key(model_sha="A" * 64)
    with pytest.raises(WorkerRegistryConfigError, match="white_box_flags"):
        runtime_key(flags={"jlens": 1})
    with pytest.raises(WorkerRegistryConfigError, match="context"):
        WorkerDefinition(
            model_id="a",
            model="a.gguf",
            runtime_key=runtime_key(context_size=2048),
            flags={"ctx": 4096},
        )
    with pytest.raises(WorkerRegistryConfigError, match="adapter presence"):
        WorkerDefinition(
            model_id="a",
            model="a.gguf",
            runtime_key=runtime_key(),
            flags={"ctx": 2048, "adapter": "surprise.gguf"},
        )
    with pytest.raises(WorkerRegistryConfigError, match="value-bearing"):
        WorkerDefinition(
            model_id="a",
            model="a.gguf",
            runtime_key=runtime_key(flags=BASE_WHITE_BOX_FLAGS),
            flags={"ctx": 2048, "mask": 126336},
        )
    with pytest.raises(WorkerRegistryConfigError, match="white_box_flags"):
        WorkerDefinition(
            model_id="a",
            model="a.gguf",
            runtime_key=runtime_key(flags={
                **BASE_WHITE_BOX_FLAGS,
                "jlens": True,
            }),
            flags={"ctx": 2048},
        )

    caller_extra_args = ["--no-flash-attn"]
    no_flash = WorkerDefinition(
        model_id="a",
        model="a.gguf",
        runtime_key=runtime_key(flags={
            **BASE_WHITE_BOX_FLAGS,
            "attn_knockout": True,
        }),
        flags={"ctx": 2048, "extra_args": caller_extra_args},
    )
    caller_extra_args.clear()
    assert no_flash.runtime_key.white_box_flags["attn_knockout"] is True
    assert no_flash.flags["extra_args"] == ("--no-flash-attn",)


@pytest.mark.parametrize(
    ("definitions", "default", "preloads", "limit", "message"),
    [
        (
            [definition("a"), definition("a", model_sha=SHA_B)],
            "a", ["a"], 1, "duplicate canonical model ID",
        ),
        (
            [definition("a"), replace(definition("a"), model_id="b")],
            "a", ["a"], 1, "duplicate runtime key",
        ),
        (
            [definition("a")],
            "missing", ["a"], 1, "default model",
        ),
        (
            [definition("a")],
            "a", ["a", "a"], 2, "duplicate",
        ),
        (
            [definition("a")],
            "a", ["missing", "a"], 2, "not configured",
        ),
        (
            [definition("a"), definition("b", model_sha=SHA_B)],
            "a", ["b"], 1, "default model must be preloaded",
        ),
        (
            [definition("a"), definition("b", model_sha=SHA_B)],
            "a", ["a", "b"], 1, "exceeds",
        ),
    ],
)
def test_registry_configuration_rejects_ambiguous_or_unserviceable_preloads(
    definitions, default, preloads, limit, message
):
    with pytest.raises(WorkerRegistryConfigError, match=message):
        WorkerRegistry(
            definitions,
            default_model_id=default,
            preload_model_ids=preloads,
            max_loaded_workers=limit,
        )


def test_two_preloads_start_and_project_independent_exact_identities():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9101),
        definition("beta", model_sha=SHA_B, port=9102),
    ]
    processes = {}

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        process = FakeProcess()
        processes[selected.model_id] = process
        return process, handshake(selected, 1), selected.runtime_key.backend != "cpu"

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    status = registry.start_preloaded()

    assert [worker["model_id"] for worker in status["workers"]] == [
        "alpha", "beta",
    ]
    assert all(worker["state"] == "ready" for worker in status["workers"])
    assert status["workers"][0]["default"] is True
    assert status["workers"][1]["default"] is False
    assert {
        worker["worker_identity"]["runtime_key_sha256"]
        for worker in status["workers"]
    } == {item.runtime_key.key_sha256 for item in definitions}
    assert status["workers"][0]["worker_identity"]["worker_generation_id"] == "worker-1"
    assert registry.definition("beta") is definitions[1]
    assert (
        registry.by_runtime_key(definitions[0].runtime_key.key_sha256)
        is definitions[0]
    )
    with pytest.raises(UnknownWorkerModelError):
        registry.definition("alpha-alias")
    assert processes["alpha"].poll() is None
    assert processes["beta"].poll() is None


def test_routing_projection_carries_exact_keys_without_paths_or_failure_text():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9151),
        definition("beta", model_sha=SHA_B, port=9152),
    ]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        process = FakeProcess()
        return process, handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()
    projection = registry.routing_projection()

    assert projection["default_model_id"] == "alpha"
    assert projection["preload_model_ids"] == ["alpha", "beta"]
    assert [item["model_id"] for item in projection["models"]] == [
        "alpha", "beta",
    ]
    beta = projection["models"][1]
    assert beta["resolved_artifact"] == {
        "model_id": "beta",
        "format": "gguf",
        "artifact_sha256": SHA_B,
    }
    assert beta["runtime_key"] == definitions[1].runtime_key.as_dict()
    assert beta["worker_identity"]["worker_generation_id"] == "worker-1"
    assert "model" not in beta
    assert "path" not in beta["resolved_artifact"]
    assert "last_error" not in beta

    # The gateway projection owns its JSON values; callers cannot mutate the
    # registry's runtime key or later status receipts through a returned dict.
    beta["runtime_key"]["white_box_flags"]["sae"] = True
    fresh = registry.routing_projection()["models"][1]
    assert fresh["runtime_key"]["white_box_flags"]["sae"] is False


def test_failed_preload_is_isolated_and_can_recover_without_touching_sibling():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9201),
        definition("beta", model_sha=SHA_B, port=9202),
    ]
    attempts = {"alpha": 0, "beta": 0}
    processes = {"alpha": [], "beta": []}

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        attempts[selected.model_id] += 1
        if selected.model_id == "alpha" and attempts["alpha"] == 1:
            raise RuntimeError("synthetic boot failure")
        process = FakeProcess()
        processes[selected.model_id].append(process)
        return (
            process,
            handshake(selected, attempts[selected.model_id]),
            False,
        )

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    first = {worker["model_id"]: worker for worker in registry.start_preloaded()["workers"]}
    assert first["alpha"]["state"] == "failed"
    assert first["alpha"]["failure_code"] == "model_load_failed"
    assert first["beta"]["state"] == "ready"
    beta_pid = first["beta"]["worker_pid"]

    assert registry.recover_failed("alpha") is True
    recovered = {
        worker["model_id"]: worker for worker in registry.status()["workers"]
    }
    assert recovered["alpha"]["state"] == "ready"
    assert recovered["beta"]["worker_pid"] == beta_pid
    assert attempts == {"alpha": 2, "beta": 1}


def test_identity_mismatch_stops_only_the_unqualified_worker():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9301),
        definition("beta", model_sha=SHA_B, port=9302),
    ]
    processes = {}

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        process = FakeProcess()
        processes[selected.model_id] = process
        overrides = {"model_sha256": SHA_C} if selected.model_id == "alpha" else {}
        return process, handshake(selected, 1, overrides=overrides), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    status = {worker["model_id"]: worker for worker in registry.start_preloaded()["workers"]}
    assert status["alpha"]["state"] == "failed"
    assert status["alpha"]["failure_code"] == "worker_identity_mismatch"
    assert processes["alpha"].terminated is True
    assert status["beta"]["state"] == "ready"
    assert processes["beta"].terminated is False


def test_dead_worker_restarts_with_new_generation_without_restarting_sibling():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9401),
        definition("beta", model_sha=SHA_B, port=9402),
    ]
    attempts = {"alpha": 0, "beta": 0}
    processes = {"alpha": [], "beta": []}

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        attempts[selected.model_id] += 1
        process = FakeProcess()
        processes[selected.model_id].append(process)
        return (
            process,
            handshake(selected, attempts[selected.model_id]),
            False,
        )

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()
    beta_pid = processes["beta"][0].pid
    processes["alpha"][0].code = 1

    status = {worker["model_id"]: worker for worker in registry.maintain()["workers"]}
    assert status["alpha"]["state"] == "ready"
    assert status["alpha"]["worker_identity"]["worker_generation"] == 2
    assert status["alpha"]["worker_identity"]["worker_id"] == "worker-2"
    assert status["beta"]["worker_pid"] == beta_pid
    assert attempts == {"alpha": 2, "beta": 1}


def test_health_requalification_and_stop_are_worker_scoped():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9501),
        definition("beta", model_sha=SHA_B, port=9502),
    ]
    processes = {}
    probes = {
        9501: handshake(definitions[0], 1, overrides={"n_ctx": 4096}),
        9502: handshake(definitions[1], 1),
    }

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        process = FakeProcess()
        processes[selected.model_id] = process
        return process, handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
        health_probe=lambda port: probes[port],
    )
    registry.start_preloaded()

    assert registry.refresh_health("alpha") is False
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "failed"
    assert status["alpha"]["failure_code"] == "worker_identity_mismatch"
    assert processes["alpha"].terminated is True
    assert status["beta"]["state"] == "ready"

    registry.stop("beta")
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["beta"]["state"] == "unloaded"
    assert status["beta"]["worker_pid"] is None
    assert processes["beta"].terminated is True


# --- RT-04: cold load, coalescing, idle-LRU eviction, typed failure states ---


def test_ten_concurrent_cold_requests_cause_exactly_one_load():
    """The load-bearing RT-04 guarantee: a burst on one cold model spawns once.

    Ten real threads call ensure_loaded() for the same never-loaded model at
    once.  A counting fake spawn proves -- by actually counting invocations,
    not by inspecting a mock -- that only one process is ever spawned; the
    other nine coalesce behind it instead of each starting their own load.
    """
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9601),
        definition("cold", model_sha=SHA_B, port=9602),
    ]
    call_count = 0
    call_lock = threading.Lock()
    spawn_entered = threading.Event()
    release = threading.Event()
    blocking = False

    def spawn(model, port, flags, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        selected = next(item for item in definitions if item.model == model)
        if blocking:
            spawn_entered.set()
            # Block until every concurrent caller has had a real chance to
            # observe the in-progress load, so coalescing is proven against
            # genuine overlap rather than a lucky race that happened not to
            # trigger it.
            assert release.wait(timeout=5), "test setup: release was never set"
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()  # spawns alpha; irrelevant to the "cold" count below
    call_count = 0
    blocking = True

    start_barrier = threading.Barrier(10)

    def call_ensure_loaded():
        start_barrier.wait(timeout=5)
        return registry.ensure_loaded("cold", timeout=10)

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(call_ensure_loaded) for _ in range(10)]
        assert spawn_entered.wait(timeout=5), "loader never reached spawn"
        time.sleep(0.05)  # let the other nine threads reach the coalescing wait
        release.set()
        results = [future.result(timeout=10) for future in futures]

    assert call_count == 1, (
        "exactly one process must be spawned for ten concurrent cold requests"
    )
    assert all(result.outcome == "loaded" for result in results)
    assert all(result.state_after == WorkerLifecycleState.READY for result in results)
    assert all(isinstance(result.state_after, WorkerLifecycleState) for result in results)
    coalesced_flags = [result.coalesced for result in results]
    assert coalesced_flags.count(False) == 1
    assert coalesced_flags.count(True) == 9
    assert all(result.wait_ms >= 0 for result in results)
    # Every coalesced waiter reports the same runtime load event as the
    # loader -- including `kind`, which a waiter cannot derive correctly from
    # its own state_before alone (it may observe "loading", not the original
    # "unloaded"/"failed" that decided cold_load vs. reload).
    assert len({result.event_id for result in results}) == 1
    assert len({result.kind for result in results}) == 1
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["cold"]["state"] == "ready"


def test_ensure_loaded_already_ready_short_circuits_without_spawning():
    definitions = [definition("alpha", model_sha=SHA_A, port=9603)]
    calls = {"count": 0}

    def spawn(model, port, flags, **kwargs):
        calls["count"] += 1
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        spawn=spawn,
    )
    registry.start_preloaded()
    assert calls["count"] == 1

    result = registry.ensure_loaded("alpha")
    assert result.outcome == "already_ready"
    assert result.kind == "not_required"
    assert result.coalesced is False
    assert result.wait_ms == 0
    assert result.event_id is None
    assert calls["count"] == 1


def test_ensure_loaded_failure_is_typed_and_a_retry_is_a_reload():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9611),
        definition("cold", model_sha=SHA_B, port=9612),
    ]
    attempts = {"count": 0}

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        if selected.model_id == "cold":
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise RuntimeError("synthetic boot failure")
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()

    first = registry.ensure_loaded("cold")
    assert first.outcome == "failed"
    assert first.kind == "cold_load"
    assert first.coalesced is False
    assert first.state_before == WorkerLifecycleState.UNLOADED
    assert first.state_after == WorkerLifecycleState.FAILED
    assert first.failure_code == "model_load_failed"
    assert first.error is not None
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["cold"]["state"] == "failed"
    assert status["cold"]["failure_code"] == "model_load_failed"

    second = registry.ensure_loaded("cold")
    assert second.outcome == "loaded"
    assert second.kind == "reload"
    assert second.coalesced is False
    assert second.state_before == WorkerLifecycleState.FAILED
    assert second.state_after == WorkerLifecycleState.READY
    assert attempts["count"] == 2


def test_coalesced_waiter_can_time_out_while_the_loader_keeps_going():
    """A slow load lets one waiter give up without starting a second spawn."""
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9621),
        definition("cold", model_sha=SHA_B, port=9622),
    ]
    call_count = 0
    call_lock = threading.Lock()
    spawn_entered = threading.Event()
    release = threading.Event()
    blocking = False

    def spawn(model, port, flags, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        selected = next(item for item in definitions if item.model == model)
        if blocking:
            spawn_entered.set()
            assert release.wait(timeout=5), "test setup: release was never set"
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()  # spawns alpha; irrelevant to the "cold" count below
    call_count = 0
    blocking = True

    with ThreadPoolExecutor(max_workers=1) as pool:
        loader_future = pool.submit(registry.ensure_loaded, "cold", timeout=10)
        assert spawn_entered.wait(timeout=5), "loader never reached spawn"

        waiter_result = registry.ensure_loaded("cold", timeout=0.1)
        assert waiter_result.outcome == "timed_out"
        assert waiter_result.coalesced is True
        assert waiter_result.failure_code == "model_load_timeout"
        # The waiter legitimately observes the loader's already-flipped state:
        # by the time it calls in, the loader has moved "cold" to loading.
        assert waiter_result.state_before == WorkerLifecycleState.LOADING

        release.set()
        loader_result = loader_future.result(timeout=5)

    assert loader_result.outcome == "loaded"
    assert loader_result.coalesced is False
    assert call_count == 1, "a timed-out waiter must never trigger a second spawn"


def test_idle_lru_eviction_picks_the_least_recently_used_idle_worker():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9631),
        definition("beta", model_sha=SHA_B, port=9632),
        definition("gamma", model_sha=SHA_C, port=9633),
    ]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
        # This test deliberately exercises the real idle-LRU ordering logic (not
        # just the fail-closed refusal), so it must declare a trustworthy busy
        # signal itself -- exactly the affirmative step production never takes.
        # See UnverifiableWorkerStateError / ADR 006.
        busy_tracking_wired=True,
    )
    registry.start_preloaded()
    time.sleep(0.01)
    registry.touch("beta")  # beta is now more recently used than alpha

    result = registry.ensure_loaded("gamma")

    assert result.outcome == "loaded"
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "unloaded"
    assert status["beta"]["state"] == "ready"
    assert status["gamma"]["state"] == "ready"
    assert sum(worker["state"] == "ready" for worker in status.values()) == 2


def test_eviction_never_picks_a_worker_with_active_generation_in_flight():
    """Eviction must consult real in-flight state, not a timestamp alone."""
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9641),
        definition("beta", model_sha=SHA_B, port=9642),
        definition("gamma", model_sha=SHA_C, port=9643),
    ]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
        # Real busy state is injected below via track_call(); tell the registry
        # to trust it, matching how a real ADR-006 caller would.
        busy_tracking_wired=True,
    )
    registry.start_preloaded()
    time.sleep(0.01)
    registry.touch("beta")  # alpha is the idlest worker by the clock alone

    with registry.track_call("alpha"):  # ...but alpha has active work in flight
        result = registry.ensure_loaded("gamma")

    assert result.outcome == "loaded"
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "ready"  # skipped: busy, not idle
    assert status["beta"]["state"] == "unloaded"  # evicted instead
    assert status["gamma"]["state"] == "ready"


def test_capacity_fails_closed_with_no_evictable_worker_when_everything_is_busy():
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9651),
        definition("beta", model_sha=SHA_B, port=9652),
        definition("gamma", model_sha=SHA_C, port=9653),
    ]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
        busy_tracking_wired=True,
    )
    registry.start_preloaded()

    with registry.track_call("alpha"), registry.track_call("beta"):
        result = registry.ensure_loaded("gamma")

    assert result.outcome == "failed"
    assert result.failure_code == "no_evictable_worker"
    assert result.state_after == WorkerLifecycleState.FAILED
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "ready"
    assert status["beta"]["state"] == "ready"
    assert status["gamma"]["state"] == "failed"
    # Never exceed the configured resident limit, even on a failed cold load.
    assert sum(worker["state"] == "ready" for worker in status.values()) == 2


def test_evict_refuses_a_busy_worker_and_can_wait_for_it_honestly():
    definitions = [definition("alpha", model_sha=SHA_A, port=9661)]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        spawn=spawn,
        busy_tracking_wired=True,
    )
    registry.start_preloaded()

    release = threading.Event()
    entered = threading.Event()

    def hold_call():
        with registry.track_call("alpha"):
            entered.set()
            release.wait(timeout=5)

    holder = threading.Thread(target=hold_call)
    holder.start()
    assert entered.wait(timeout=5)

    with pytest.raises(WorkerBusyError):
        registry.evict("alpha")

    with pytest.raises(EvictionTimeoutError):
        registry.evict("alpha", wait_for_inflight=True, timeout=0.05)

    release.set()
    holder.join(timeout=5)

    registry.evict("alpha", wait_for_inflight=True, timeout=5)
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "unloaded"


def test_eviction_refuses_when_busy_state_cannot_be_verified():
    """Reproduces the RT-04 vacuity directly, using production's actual default.

    RT-04's idle-LRU eviction claims it never evicts a worker with in-flight work.
    That guarantee was vacuous: eviction consulted ``handle.busy``, but nothing in
    production ever entered :meth:`WorkerHandle.track_call` -- real generation
    traffic flows gateway<->worker directly and never touches the supervisor
    process at all (docs/design/006-cross-process-cold-load-protocol.md's Context
    section). So ``handle.busy`` was always False in production, and every worker
    always looked idle, whether or not it actually was.

    This test's registry construction is deliberately the exact shape every real
    caller uses -- ``clozn/cli/runtime_process.py`` and
    ``clozn/cli/managed_models.py`` never pass ``busy_tracking_wired=True`` and
    never call ``track_call()`` for real traffic, so neither does this test. On the
    vacuous pre-fix code, that made ``_select_eviction_candidate`` treat alpha and
    beta's permanently-False ``busy`` as "verified idle": loading a third model
    over capacity silently evicted alpha (the least-recently-touched resident) and
    reported ``outcome == "loaded"``, exactly as if idleness had been confirmed --
    with zero evidence it actually was. Run against that code, every assertion
    below fails: ``result.outcome`` is ``"loaded"`` (not ``"failed"``), alpha's
    state is ``"unloaded"`` (not ``"ready"``), and gamma is ``"ready"`` (not
    ``"failed"``).

    After the fix, the same default, unmodified construction instead refuses with
    the typed ``no_verifiable_idle_worker`` code and touches neither resident --
    proving the fail-closed gate is live on the exact path production takes, not
    just in tests that opt into ``busy_tracking_wired=True``.
    """
    definitions = [
        definition("alpha", model_sha=SHA_A, port=9681),
        definition("beta", model_sha=SHA_B, port=9682),
        definition("gamma", model_sha=SHA_C, port=9683),
    ]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    # No busy_tracking_wired kwarg, and track_call() is never entered below for
    # either resident -- this is production's real, unmodified construction shape.
    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()
    time.sleep(0.01)
    registry.touch("beta")  # alpha is the least-recently-used resident

    result = registry.ensure_loaded("gamma")

    assert result.outcome == "failed", (
        "with no verified busy signal, eviction must refuse -- not silently pick "
        "alpha as a victim just because handle.busy happens to read False"
    )
    assert result.failure_code == "no_verifiable_idle_worker"
    assert result.state_after == WorkerLifecycleState.FAILED
    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "ready", "never evicted on an unverified assumption"
    assert status["beta"]["state"] == "ready"
    assert status["gamma"]["state"] == "failed"
    assert sum(worker["state"] == "ready" for worker in status.values()) == 2


def test_explicit_evict_refuses_when_busy_state_cannot_be_verified():
    """The explicit evict() path fails closed the same way, not just the LRU path.

    ``evict()`` used to consult ``handle.busy`` directly (worker_registry.py:887 in
    the vacuous version) and proceed unconditionally whenever it read False --
    exactly as unverifiable in production as the idle-LRU path above, since
    nothing ever calls track_call() for real traffic. With no
    ``busy_tracking_wired=True`` opt-in, evict() must refuse outright rather than
    ever reaching (or silently skipping) the busy/wait_for_inflight logic.
    """
    definitions = [definition("alpha", model_sha=SHA_A, port=9691)]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        spawn=spawn,
    )
    registry.start_preloaded()

    with pytest.raises(UnverifiableWorkerStateError):
        registry.evict("alpha")

    # Even the "honestly wait for it" opt-in cannot rescue an unverifiable signal:
    # wait_until_idle() would return True immediately (nothing was ever tracked),
    # which is exactly the false confidence this refusal exists to prevent.
    with pytest.raises(UnverifiableWorkerStateError):
        registry.evict("alpha", wait_for_inflight=True, timeout=5)

    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["alpha"]["state"] == "ready", "never evicted on an unverified assumption"


def test_lifecycle_state_is_a_typed_enum_that_still_compares_as_a_string():
    definitions = [definition("alpha", model_sha=SHA_A, port=9671)]

    def spawn(model, port, flags, **kwargs):
        selected = next(item for item in definitions if item.model == model)
        return FakeProcess(), handshake(selected, 1), False

    registry = WorkerRegistry(
        definitions,
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        spawn=spawn,
    )
    registry.start_preloaded()

    result = registry.ensure_loaded("alpha")
    assert isinstance(result.state_after, WorkerLifecycleState)
    assert result.state_after == "ready"
    assert result.state_after == WorkerLifecycleState.READY

    # status() crosses a JSON boundary; the enum type itself must never leak.
    status = registry.status()["workers"][0]
    assert type(status["state"]) is str
