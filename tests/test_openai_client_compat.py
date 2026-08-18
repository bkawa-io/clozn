"""Integration with the real ``openai`` Python package against a model-free local gateway.

CI installs the package explicitly. A developer environment without it skips this one file; the pure
field-policy tests remain mandatory and dependency-free.
"""
from __future__ import annotations

import threading
from http.server import ThreadingHTTPServer

import pytest


openai = pytest.importorskip("openai")

from clozn.server import app as cs  # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402


import clozn.runs.store as runlog  # noqa: E402


class _Substrate:
    name = "engine"

    def __init__(self):
        self._finish = "stop"

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None, assembled_messages=list(messages))
        return "SDK round trip."

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None, assembled_messages=list(messages))
        yield "SDK "
        yield "stream."

    def last_finish_reason(self):
        return self._finish

    def last_stream_trace(self):
        return []

    def run_meta(self):
        return {"model_id": "clozn-local", "sampler_mode": "greedy", "temperature": 0.0}


@pytest.fixture
def openai_gateway(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", _Substrate())
    server = ThreadingHTTPServer(("127.0.0.1", 0), cs.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _client(base_url):
    return openai.OpenAI(api_key="local-test-key", base_url=base_url, max_retries=0, timeout=5.0)


def test_real_openai_client_lists_models_and_parses_chat(openai_gateway):
    client = _client(openai_gateway)
    models = client.models.list()
    assert models.data and models.data[0].id == "clozn-local"

    reply = client.chat.completions.create(
        model="clozn-local",
        messages=[{"role": "developer", "content": "Be concise."},
                  {"role": "user", "content": "Hello"}],
        max_completion_tokens=12,
        temperature=0,
    )
    assert reply.choices[0].message.content == "SDK round trip."
    assert reply.choices[0].finish_reason == "stop"
    assert reply.usage is None                 # Clozn omits unknown counts instead of returning fake zeros
    rid = (reply.model_extra or {}).get("clozn_run_id")
    assert rid and runlog.get_run(rid)["response"] == "SDK round trip."


def test_real_openai_client_parses_stream(openai_gateway):
    client = _client(openai_gateway)
    stream = client.chat.completions.create(
        model="clozn-local",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
        temperature=0,
    )
    pieces = []
    terminal_run_id = None
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            pieces.append(chunk.choices[0].delta.content)
        if chunk.choices and chunk.choices[0].finish_reason:
            terminal_run_id = (chunk.model_extra or {}).get("clozn_run_id")
    assert "".join(pieces) == "SDK stream."
    assert terminal_run_id and runlog.get_run(terminal_run_id)["response"] == "SDK stream."


def test_real_openai_client_receives_typed_400_for_unqualified_tools(openai_gateway):
    client = _client(openai_gateway)
    with pytest.raises(openai.BadRequestError) as caught:
        client.chat.completions.create(
            model="clozn-local",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[{"type": "function", "function": {
                "name": "weather",
                "parameters": {"type": "object", "properties": {},
                               "additionalProperties": False},
                "strict": True,
            }}],
        )
    assert caught.value.status_code == 400
    error = caught.value.body.get("error", caught.value.body)
    assert error["param"] == "tools"
    assert error["type"] == "invalid_request_error"
    assert error["code"] == "model_not_qualified"
