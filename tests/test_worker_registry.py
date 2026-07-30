"""Model-free tests for the ADR 004 preloaded worker registry."""
from __future__ import annotations

from dataclasses import replace

import pytest

from clozn.cli.worker_registry import (
    AdapterRuntimeIdentity,
    RuntimeKey,
    UnknownWorkerModelError,
    WorkerDefinition,
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
