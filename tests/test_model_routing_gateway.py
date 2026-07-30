"""Model-free end-to-end coverage for RT-03 preloaded multi-model routing."""
from __future__ import annotations

import io
import json

import pytest

import clozn.runs.store as runlog
import clozn.settings as clozn_settings
from clozn import schemas
from clozn.cli.worker_registry import AdapterRuntimeIdentity, RuntimeKey
from clozn.server import app as cs
from clozn.server import generation_gateway
from clozn.server.model_routing import (
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
