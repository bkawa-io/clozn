"""Model-free end-to-end coverage for RT-03/RT-04 preloaded multi-model routing."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import io
import json
import threading
import time

import pytest

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
from clozn.server import generation_gateway
from clozn.server.model_routing import (
    ColdLoadOutcome,
    PreloadedModelBinding,
    PreloadedModelRouter,
)


SHA_ALPHA = "a" * 64
SHA_BETA = "b" * 64
SHA_COLD = "c" * 64
SHA_BROKEN = "d" * 64
WHITE_BOX = {
    "sae": False,
    "jlens": False,
    "attn_knockout": False,
}


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
        self.base = f"http://127.0.0.1/{model_id}"
        self.timeout = 10
        self.health_calls = 0

    def health(self):
        self.health_calls += 1
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
        self.calls = []

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "max_new": max_new,
            "sample": sample,
        })
        if mem_out is not None:
            mem_out.update(
                assembled_messages=[dict(message) for message in messages],
                final_prompt=f"<{self.engine.model_id}>rendered</{self.engine.model_id}>",
                actual_prompt_tokens=7,
            )
        if trace_out is not None:
            trace_out.append({
                "pos": 0,
                "token_id": 1,
                "piece": self.engine.model_id,
                "prob": 1.0,
                "alts": [],
            })
        return f"{self.engine.model_id} reply"

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        self.calls.append({
            "messages": [dict(message) for message in messages],
            "max_new": max_new,
            "sample": sample,
            "stream": True,
        })
        if mem_out is not None:
            mem_out.update(
                assembled_messages=[dict(message) for message in messages],
                final_prompt=f"<{self.engine.model_id}>stream</{self.engine.model_id}>",
                actual_prompt_tokens=7,
            )
        yield self.engine.model_id
        yield " streamed"

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return [{
            "pos": 0,
            "token_id": 1,
            "piece": self.engine.model_id,
            "prob": 1.0,
            "alts": [],
        }]

    def run_meta(self):
        return {
            "model_id": self.engine.model_id,
            "model_sha256": self.engine.digest,
            "worker_marker": self.engine.worker_generation_id,
        }

    def identity_meta(self):
        return {
            "model_sha256": self.engine.digest,
            "template_fingerprint": self.engine.template,
            "engine_build": self.engine.build,
        }

    def last_prompt_tokens(self):
        return 7


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


def _ready_binding(model_id: str, digest: str, template: str):
    build = f"engine-{model_id}"
    engine = FakeEngine(model_id, digest, template, build)
    sub = FakeSub(engine)
    key = _runtime_key(digest, template, build)
    binding = PreloadedModelBinding(
        model_id=model_id,
        resolved_artifact={
            "model_id": model_id,
            "format": "gguf",
            "artifact_sha256": digest,
        },
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
    return binding, engine, sub


def _inactive_binding(model_id: str, digest: str, state: str):
    template = "3" * 32 if state != "failed" else "4" * 32
    build = f"engine-{model_id}"
    key = _runtime_key(digest, template, build)
    return PreloadedModelBinding(
        model_id=model_id,
        resolved_artifact={
            "model_id": model_id,
            "format": "gguf",
            "artifact_sha256": digest,
        },
        runtime_key=key,
        adapter=AdapterRuntimeIdentity.absent().as_dict(),
        state=state,
        worker_identity=None,
        sub=None,
        engine=None,
        preloaded=False,
        failure_code="model_load_failed" if state == "failed" else None,
    )


@pytest.fixture
def routed(tmp_path, monkeypatch):
    alpha, alpha_engine, alpha_sub = _ready_binding(
        "alpha", SHA_ALPHA, "1" * 32
    )
    beta, beta_engine, beta_sub = _ready_binding(
        "beta", SHA_BETA, "2" * 32
    )
    router = PreloadedModelRouter(
        [
            alpha,
            beta,
            _inactive_binding("cold", SHA_COLD, "unloaded"),
            _inactive_binding("broken", SHA_BROKEN, "failed"),
        ],
        default_model_id="alpha",
        preload_model_ids=["alpha", "beta"],
        max_loaded_workers=2,
    )
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json")
    )
    monkeypatch.setattr(cs, "MODEL_ROUTER", router)
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "ENGINE", None)
    return {
        "router": router,
        "alpha_engine": alpha_engine,
        "beta_engine": beta_engine,
        "alpha_sub": alpha_sub,
        "beta_sub": beta_sub,
    }


def _dispatch(method: str, path: str, body=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {
        "Content-Length": str(len(raw)),
        "User-Agent": "pytest-routing",
    }
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    return handler.wfile.getvalue(), handler


def _response(raw: bytes):
    headers, _, body = raw.partition(b"\r\n\r\n")
    first = headers.splitlines()[0].decode("ascii")
    return int(first.split()[1]), json.loads(body.decode("utf-8")), headers


def _assert_clean(handler):
    for field in (
        "_route_sub",
        "_route_engine",
        "_route_subname",
        "_selected_model_id",
        "_model_routing_artifact",
    ):
        assert not hasattr(handler, field)


def _wire_router(monkeypatch, tmp_path, router):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(
        clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json")
    )
    monkeypatch.setattr(cs, "MODEL_ROUTER", router)
    monkeypatch.setattr(cs, "SUB", None)
    monkeypatch.setattr(cs, "ENGINE", None)


def test_openai_explicit_model_dispatches_only_that_worker_and_journals_identity(
    routed,
):
    raw, handler = _dispatch("POST", "/v1/chat/completions", {
        "model": "beta",
        "messages": [{"role": "user", "content": "route this"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["model"] == "beta"
    assert payload["choices"][0]["message"]["content"] == "beta reply"
    assert routed["alpha_sub"].calls == []
    assert len(routed["beta_sub"].calls) == 1

    logged = runlog.get_run(payload["clozn_run_id"])
    receipt = logged["meta"]["model_routing"]["result"]["receipt"]
    schemas.validate(logged["meta"]["model_routing"])
    assert logged["model"] == "beta"
    assert logged["meta"]["model_id"] == "beta"
    assert logged["identity"]["model_sha256"] == SHA_BETA
    assert receipt["requested_model"] == "beta"
    assert receipt["resolved_model_id"] == "beta"
    assert receipt["resolved_artifact"]["artifact_sha256"] == SHA_BETA
    assert receipt["runtime_key"]["gguf_artifact_sha256"] == SHA_BETA
    assert receipt["worker_identity"]["worker_id"] == "beta-worker-1"
    assert receipt["adapter"]["present"] is False
    assert receipt["load_event"]["outcome"] == "already_ready"
    _assert_clean(handler)


def test_omitted_model_selects_configured_default_without_fabricating_request(
    routed,
):
    raw, _handler = _dispatch("POST", "/v1/chat/completions", {
        "messages": [{"role": "user", "content": "default"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["model"] == "alpha"
    logged = runlog.get_run(payload["clozn_run_id"])
    receipt = logged["meta"]["model_routing"]["result"]["receipt"]
    assert receipt["requested_model"] is None
    assert receipt["selection_source"] == "default"
    assert receipt["resolved_model_id"] == "alpha"
    assert len(routed["alpha_sub"].calls) == 1
    assert routed["beta_sub"].calls == []


def test_legacy_openai_completion_is_not_a_default_worker_escape_hatch(routed):
    raw, handler = _dispatch("POST", "/v1/completions", {
        "model": "beta",
        "prompt": "legacy route",
        "max_tokens": 4,
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["model"] == "beta"
    assert payload["choices"][0]["text"] == "beta reply"
    assert routed["alpha_sub"].calls == []
    assert len(routed["beta_sub"].calls) == 1
    logged = runlog.get_run(payload["clozn_run_id"])
    routing = logged["meta"]["model_routing"]
    schemas.validate(routing)
    assert routing["protocol"] == {
        "surface": "openai",
        "route": "/v1/completions",
    }
    assert routing["result"]["receipt"]["resolved_model_id"] == "beta"
    assert logged["identity"]["model_sha256"] == SHA_BETA
    _assert_clean(handler)


def test_ollama_chat_uses_same_exact_worker_and_receipt(routed):
    raw, handler = _dispatch("POST", "/api/chat", {
        "model": "beta",
        "messages": [{"role": "user", "content": "ollama route"}],
        "stream": False,
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["model"] == "beta"
    assert payload["message"]["content"] == "beta reply"
    logged = runlog.get_run(payload["clozn_run_id"])
    assert logged["source"] == "ollama_api"
    assert logged["identity"]["model_sha256"] == SHA_BETA
    routing = logged["meta"]["model_routing"]
    schemas.validate(routing)
    assert routing["protocol"] == {"surface": "ollama", "route": "/api/chat"}
    assert routing["result"]["receipt"]["worker_identity"]["worker_id"] == (
        "beta-worker-1"
    )
    assert routed["alpha_sub"].calls == []
    _assert_clean(handler)


def test_openai_stream_keeps_selected_worker_through_terminal_journal(routed):
    raw, handler = _dispatch("POST", "/v1/chat/completions", {
        "model": "beta",
        "messages": [{"role": "user", "content": "stream route"}],
        "stream": True,
    })
    headers, _, body = raw.partition(b"\r\n\r\n")
    assert int(headers.splitlines()[0].split()[1]) == 200
    chunks = []
    for line in body.decode("utf-8").splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            chunks.append(json.loads(line.removeprefix("data: ")))
    assert all(chunk["model"] == "beta" for chunk in chunks)
    assert "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
    ) == "beta streamed"
    terminal = next(
        chunk for chunk in chunks if chunk.get("clozn_run_id")
    )
    logged = runlog.get_run(terminal["clozn_run_id"])
    assert logged["model"] == "beta"
    assert logged["identity"]["model_sha256"] == SHA_BETA
    assert (
        logged["meta"]["model_routing"]["result"]["receipt"]["worker_identity"][
            "worker_id"
        ]
        == "beta-worker-1"
    )
    assert routed["alpha_sub"].calls == []
    assert routed["beta_sub"].calls[0]["stream"] is True
    _assert_clean(handler)


def test_unknown_and_not_ready_models_never_fall_back(routed):
    raw, unknown_handler = _dispatch("POST", "/v1/chat/completions", {
        "model": "not-configured",
        "messages": [{"role": "user", "content": "must fail"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 404
    assert payload["error"]["code"] == "unknown_model"
    assert payload["error"]["type"] == "model_routing_error"
    assert payload["error"]["retryable"] is False
    assert payload["error"]["phase"] == "resolution"
    schemas.validate(payload["clozn_model_routing"])
    assert payload["clozn_model_routing"]["request"]["requested_model"] == (
        "not-configured"
    )

    raw, cold_handler = _dispatch("POST", "/api/generate", {
        "model": "cold",
        "prompt": "must not load",
        "stream": False,
    })
    status, payload, _headers = _response(raw)
    assert status == 409
    assert payload["code"] == "model_not_ready"
    assert payload["retryable"] is True
    schemas.validate(payload["clozn_model_routing"])
    assert routed["alpha_sub"].calls == routed["beta_sub"].calls == []
    assert runlog.list_runs() == []
    _assert_clean(unknown_handler)
    _assert_clean(cold_handler)


def test_failed_model_has_stable_typed_ollama_error(routed):
    raw, handler = _dispatch("POST", "/api/chat", {
        "model": "broken",
        "messages": [{"role": "user", "content": "must fail"}],
        "stream": False,
    })
    status, payload, _headers = _response(raw)
    assert status == 503
    assert payload["code"] == "model_load_failed"
    assert payload["phase"] == "load"
    assert payload["retryable"] is True
    artifact = payload["clozn_model_routing"]
    schemas.validate(artifact)
    assert artifact["result"]["lifecycle_state"] == "failed"
    _assert_clean(handler)


def test_literal_whitespace_is_invalid_instead_of_alias_like_normalization(routed):
    raw, _handler = _dispatch("POST", "/v1/chat/completions", {
        "model": " beta ",
        "messages": [{"role": "user", "content": "must fail"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 400
    assert payload["error"]["code"] == "invalid_model_selection"
    artifact = payload["clozn_model_routing"]
    assert artifact["request"]["requested_model"] == " beta "
    assert artifact["result"]["receipt"]["requested_model"] == " beta "


def test_native_route_uses_selected_private_engine_and_returns_receipt(
    routed, monkeypatch
):
    seen = {}

    class Response:
        status = 200

        def read(self):
            return json.dumps({
                "choices": [{
                    "text": "native beta",
                    "index": 0,
                    "finish_reason": "stop",
                }]
            }).encode("utf-8")

        def close(self):
            seen["closed"] = True

    def fake_urlopen(request, timeout=None):
        seen["url"] = request.full_url
        seen["timeout"] = timeout
        seen["body"] = json.loads(request.data)
        return Response()

    monkeypatch.setattr(generation_gateway.urllib.request, "urlopen", fake_urlopen)
    raw, handler = _dispatch("POST", "/api/clozn/generate", {
        "model": "beta",
        "prompt": "raw prompt",
        "max_tokens": 3,
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["choices"][0]["text"] == "native beta"
    assert seen["url"] == "http://127.0.0.1/beta/v1/completions"
    assert seen["body"] == {"prompt": "raw prompt", "max_tokens": 3}
    assert seen["closed"] is True
    artifact = payload["clozn_model_routing"]
    schemas.validate(artifact)
    assert artifact["protocol"] == {
        "surface": "native",
        "route": "/api/clozn/generate",
    }
    assert artifact["result"]["receipt"]["resolved_model_id"] == "beta"
    _assert_clean(handler)


def test_model_catalogs_expose_canonical_configured_ids(routed):
    raw, _handler = _dispatch("GET", "/v1/models")
    status, payload, _headers = _response(raw)
    assert status == 200
    assert [item["id"] for item in payload["data"]] == [
        "alpha", "beta", "broken", "cold",
    ]

    raw, _handler = _dispatch("GET", "/api/tags")
    status, payload, _headers = _response(raw)
    assert status == 200
    by_name = {item["name"]: item for item in payload["models"]}
    assert set(by_name) == {"alpha", "beta", "broken", "cold"}
    assert by_name["beta"]["digest"] == f"sha256:{SHA_BETA}"
    assert by_name["broken"]["clozn_state"] == "failed"


def test_worker_restart_advances_generation_in_next_immutable_receipt(routed):
    first = routed["router"].select(
        "beta",
        field_present=True,
        surface="openai",
        route="/v1/chat/completions",
    ).artifact
    routed["beta_engine"].worker_generation_id = "beta-worker-2"
    second = routed["router"].select(
        "beta",
        field_present=True,
        surface="openai",
        route="/v1/chat/completions",
    ).artifact
    first_identity = first["result"]["receipt"]["worker_identity"]
    second_identity = second["result"]["receipt"]["worker_identity"]
    assert first_identity["worker_generation"] == 1
    assert first_identity["worker_id"] == "beta-worker-1"
    assert second_identity["worker_generation"] == 2
    assert second_identity["worker_id"] == "beta-worker-2"
    schemas.validate(first)
    schemas.validate(second)


def test_live_identity_drift_fails_only_selected_binding(routed):
    routed["beta_engine"].digest = SHA_ALPHA
    raw, _handler = _dispatch("POST", "/v1/chat/completions", {
        "model": "beta",
        "messages": [{"role": "user", "content": "detect drift"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 502
    assert payload["error"]["code"] == "worker_identity_mismatch"
    assert routed["alpha_sub"].calls == routed["beta_sub"].calls == []

    raw, _handler = _dispatch("POST", "/v1/chat/completions", {
        "model": "alpha",
        "messages": [{"role": "user", "content": "sibling survives"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["model"] == "alpha"


# --- RT-04: routing through a configured cold-load loader --------------------
#
# Without a loader, a not-ready model still fails immediately (unchanged --
# see test_unknown_and_not_ready_models_never_fall_back above).  These tests
# cover the opt-in path: a router constructed with `loader=` performs a real
# cold load through it and returns a routed receipt carrying the real
# LoadEvent, or the matching typed error when the load fails/times out.


def test_select_cold_loads_through_a_configured_loader_and_journals_real_load_event(
    tmp_path, monkeypatch
):
    alpha, _alpha_engine, _alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32)
    cold_binding = _inactive_binding("cold", SHA_COLD, "unloaded")
    cold_key_sha = cold_binding.runtime_key["key_sha256"]
    cold_engine = FakeEngine("cold", SHA_COLD, "3" * 32, "engine-cold")
    cold_sub = FakeSub(cold_engine)
    loader_calls = []

    def loader(model_id, timeout):
        loader_calls.append((model_id, timeout))
        return ColdLoadOutcome(
            state="ready",
            kind="cold_load",
            outcome="loaded",
            coalesced=False,
            wait_ms=42,
            worker_port=9999,
            worker_identity={
                "worker_id": "cold-worker-1",
                "worker_generation": 1,
                "runtime_key_sha256": cold_key_sha,
                "protocol_version": "1.1",
                "engine_build": "engine-cold",
                "backend": "cpu",
            },
            event_id="load_deadbeef_1",
        )

    router = PreloadedModelRouter(
        [alpha, cold_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        loader=loader,
        engine_factory=lambda port: cold_engine,
        substrate_factory=lambda engine: cold_sub,
    )
    _wire_router(monkeypatch, tmp_path, router)

    raw, handler = _dispatch("POST", "/v1/chat/completions", {
        "model": "cold",
        "messages": [{"role": "user", "content": "wake up"}],
    })
    status, payload, _headers = _response(raw)
    assert status == 200
    assert payload["model"] == "cold"
    assert payload["choices"][0]["message"]["content"] == "cold reply"
    assert loader_calls == [("cold", 180.0)]

    logged = runlog.get_run(payload["clozn_run_id"])
    receipt = logged["meta"]["model_routing"]["result"]["receipt"]
    schemas.validate(logged["meta"]["model_routing"])
    load_event = receipt["load_event"]
    assert load_event == {
        "event_id": "load_deadbeef_1",
        "kind": "cold_load",
        "outcome": "loaded",
        "state_before": "unloaded",
        "state_after": "ready",
        "coalesced": False,
        "wait_ms": 42,
    }
    assert receipt["worker_identity"]["worker_id"] == "cold-worker-1"
    _assert_clean(handler)

    # The binding is now materialized ready in place; a second request must
    # reuse it -- not call the loader again.
    second = router.select(
        "cold", field_present=True, surface="openai", route="/v1/chat/completions"
    )
    assert loader_calls == [("cold", 180.0)]
    assert second.artifact["result"]["receipt"]["load_event"]["outcome"] == (
        "already_ready"
    )


def test_select_cold_load_timeout_produces_typed_error_with_real_load_event(
    tmp_path, monkeypatch
):
    alpha, _alpha_engine, _alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32)
    cold_binding = _inactive_binding("cold", SHA_COLD, "unloaded")

    def loader(model_id, timeout):
        return ColdLoadOutcome(
            state="loading",
            kind="cold_load",
            outcome="timed_out",
            coalesced=True,
            wait_ms=5000,
            failure_code="model_load_timeout",
            message="timed out waiting for a coalesced load",
            event_id="load_deadbeef_2",
        )

    router = PreloadedModelRouter(
        [alpha, cold_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        loader=loader,
        engine_factory=lambda port: None,
        substrate_factory=lambda engine: None,
    )
    _wire_router(monkeypatch, tmp_path, router)

    raw, handler = _dispatch("POST", "/api/generate", {
        "model": "cold",
        "prompt": "must time out",
        "stream": False,
    })
    status, payload, _headers = _response(raw)
    assert status == 504
    assert payload["code"] == "model_load_timeout"
    assert payload["retryable"] is True
    artifact = payload["clozn_model_routing"]
    schemas.validate(artifact)
    load_event = artifact["result"]["receipt"]["load_event"]
    assert load_event["outcome"] == "timed_out"
    assert load_event["coalesced"] is True
    assert load_event["wait_ms"] == 5000
    assert artifact["result"]["lifecycle_state"] == "loading"
    assert runlog.list_runs() == []
    _assert_clean(handler)


def test_select_no_evictable_worker_is_a_typed_retryable_error(tmp_path, monkeypatch):
    alpha, _alpha_engine, _alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32)
    cold_binding = _inactive_binding("cold", SHA_COLD, "unloaded")

    def loader(model_id, timeout):
        return ColdLoadOutcome(
            state="failed",
            kind="cold_load",
            outcome="failed",
            coalesced=False,
            wait_ms=3,
            failure_code="no_evictable_worker",
            message="no idle worker available to evict",
        )

    router = PreloadedModelRouter(
        [alpha, cold_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=1,
        loader=loader,
        engine_factory=lambda port: None,
        substrate_factory=lambda engine: None,
    )
    _wire_router(monkeypatch, tmp_path, router)

    raw, handler = _dispatch("POST", "/api/chat", {
        "model": "cold",
        "messages": [{"role": "user", "content": "no room"}],
        "stream": False,
    })
    status, payload, _headers = _response(raw)
    assert status == 503
    assert payload["code"] == "no_evictable_worker"
    assert payload["retryable"] is True
    schemas.validate(payload["clozn_model_routing"])
    _assert_clean(handler)


def test_router_without_a_loader_keeps_the_original_rt03_fail_fast_behavior(routed):
    """No loader configured -- must behave exactly as before RT-04."""
    assert routed["router"]._loader is None
    raw, handler = _dispatch("POST", "/api/generate", {
        "model": "cold",
        "prompt": "must not load",
        "stream": False,
    })
    status, payload, _headers = _response(raw)
    assert status == 409
    assert payload["code"] == "model_not_ready"
    assert routed["alpha_sub"].calls == routed["beta_sub"].calls == []
    _assert_clean(handler)


# --- Full-stack coalescing proof: a real WorkerRegistry behind the router ----


class _FakeSupervisorProcess:
    """Minimal Popen-shaped stand-in for WorkerHandle's process field."""

    _next_pid = 8000

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
    """Adapt a real WorkerRegistry.ensure_loaded to the router's loader contract.

    This adapter is test-only glue.  Production wiring of a live supervisor's
    WorkerRegistry to the gateway's PreloadedModelRouter belongs to the
    (separately owned) process/IPC integration -- clozn/server must never
    import clozn/cli directly (see clozn/server/routes/models.py and
    ColdLoadOutcome's docstring).  This test proves the two owned contracts
    (WorkerRegistry.ensure_loaded and PreloadedModelRouter's loader hook)
    compose correctly end to end.
    """
    def loader(model_id: str, timeout: float) -> ColdLoadOutcome:
        result = registry.ensure_loaded(model_id, timeout=timeout)
        handle = registry.worker_handle(model_id)
        status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
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


def test_ten_concurrent_router_selects_for_one_cold_model_cause_exactly_one_load():
    """The RT-04 guarantee proven through the router's real loader wiring, too.

    tests/test_worker_registry.py proves the single-flight guarantee against
    WorkerRegistry.ensure_loaded directly.  This test proves the *router's*
    loader hook doesn't undermine it: ten real threads calling
    PreloadedModelRouter.select() for the same never-loaded model, backed by
    a real WorkerRegistry with a counting fake spawn, must still cause
    exactly one process spawn -- not once per router.select() call.

    (Calling through the live HTTP surface instead would only prove the
    pre-existing global POST_GATE in clozn/server/app.py serializes requests
    -- a real, separately-owned RT-05 concern, not the coalescing invariant
    this ticket owns.  Driving the router directly isolates the guarantee
    this ticket is actually responsible for.)
    """
    alpha, _alpha_engine, _alpha_sub = _ready_binding("alpha", SHA_ALPHA, "1" * 32)
    cold_binding = _inactive_binding("cold", SHA_COLD, "unloaded")
    cold_key_sha = cold_binding.runtime_key["key_sha256"]
    assert cold_binding.resolved_artifact["artifact_sha256"] == SHA_COLD

    registry_alpha_def = _registry_worker_definition(
        "alpha", SHA_ALPHA, "1" * 32, port=9801
    )
    registry_cold_def = _registry_worker_definition(
        "cold", SHA_COLD, "3" * 32, port=9802
    )
    assert registry_cold_def.runtime_key.key_sha256 == cold_key_sha, (
        "registry and router must agree on the cold model's exact runtime key"
    )

    call_count = 0
    call_lock = threading.Lock()
    spawn_entered = threading.Event()
    release = threading.Event()
    blocking = False

    def spawn(model, port, flags, **kwargs):
        nonlocal call_count
        with call_lock:
            call_count += 1
        definition = (
            registry_cold_def if model == registry_cold_def.model
            else registry_alpha_def
        )
        if blocking:
            spawn_entered.set()
            assert release.wait(timeout=5), "test setup: release was never set"
        return (
            _FakeSupervisorProcess(),
            _registry_handshake(definition, 1),
            False,
        )

    registry = WorkerRegistry(
        [registry_alpha_def, registry_cold_def],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        spawn=spawn,
    )
    registry.start_preloaded()  # spawns alpha; irrelevant to the "cold" count below
    call_count = 0
    blocking = True

    def engine_factory(port):
        return FakeEngine("cold", SHA_COLD, "3" * 32, "engine-cold")

    router = PreloadedModelRouter(
        [alpha, cold_binding],
        default_model_id="alpha",
        preload_model_ids=["alpha"],
        max_loaded_workers=2,
        loader=_loader_from_registry(registry),
        engine_factory=engine_factory,
        substrate_factory=lambda engine: FakeSub(engine),
    )

    start_barrier = threading.Barrier(10)

    def select_one(_index):
        start_barrier.wait(timeout=5)
        selection = router.select(
            "cold", field_present=True, surface="openai",
            route="/v1/chat/completions",
        )
        return selection.artifact

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(select_one, index) for index in range(10)]
        assert spawn_entered.wait(timeout=5), "loader never reached spawn"
        # Give the other nine select() calls time to reach the registry's
        # coalescing wait, so this proves real overlap rather than a race
        # that happened not to bite.
        time.sleep(0.1)
        release.set()
        artifacts = [future.result(timeout=10) for future in futures]

    assert call_count == 1, (
        "exactly one process must be spawned for ten concurrent router selects"
    )
    for artifact in artifacts:
        schemas.validate(artifact)
    worker_ids = {
        artifact["result"]["receipt"]["worker_identity"]["worker_id"]
        for artifact in artifacts
    }
    assert worker_ids == {"cold-worker-1"}
    coalesced_flags = [
        artifact["result"]["receipt"]["load_event"]["coalesced"]
        for artifact in artifacts
    ]
    assert coalesced_flags.count(False) == 1
    assert coalesced_flags.count(True) == 9

    status = {worker["model_id"]: worker for worker in registry.status()["workers"]}
    assert status["cold"]["state"] == "ready"
