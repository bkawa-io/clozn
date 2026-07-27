"""Official Ollama JavaScript SDK conformance against Clozn's real local gateway.

The test is model-free: a deterministic substrate is installed behind the actual HTTP
handler, while the pinned upstream SDK drives tags, chat, generate, and both NDJSON
streaming paths. Developer environments without Node or ``npm ci`` dependencies skip
cleanly; CI installs ``tests/clients/package-lock.json``.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from clozn.server import app as cs
import clozn.settings as clozn_settings


import clozn.runs.store as runlog


CLIENT_DIR = Path(__file__).with_name("clients")
PROBE = CLIENT_DIR / "ollama-js-probe.mjs"


class _Memory:
    memory_strength = 1.0
    rules = []
    prefix = None


class _Steer:
    strength = {}

    def active(self):
        return {}


class _Substrate:
    name = "engine"

    def __init__(self):
        self.memory = self._mem = _Memory()
        self.steer = _Steer()
        self._finish = "stop"
        self._stream_trace = []

    @staticmethod
    def _reply(messages, *, streaming=False):
        prompt = str(messages[-1].get("content", "")) if messages else ""
        operation = "chat" if "chat" in prompt else "generate"
        mode = "stream" if streaming else "nonstream"
        return f"SDK {mode} {operation}."

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None, assembled_messages=list(messages))
        return self._reply(messages)

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None, assembled_messages=list(messages))
        reply = self._reply(messages, streaming=True)
        midpoint = len(reply) // 2
        pieces = (reply[:midpoint], reply[midpoint:])
        self._stream_trace = []
        for pos, piece in enumerate(pieces):
            self._stream_trace.append({"pos": pos, "piece": piece})
            yield piece

    def last_finish_reason(self):
        return self._finish

    def last_stream_trace(self):
        return list(self._stream_trace)

    def last_prompt_tokens(self):
        return 2

    def run_meta(self):
        return {"model_id": "sdk-fixture", "sampler_mode": "greedy", "temperature": 0.0}


class _Engine:
    def health(self):
        return {"model": "sdk-fixture.gguf", "model_sha256": "0" * 64}


@pytest.fixture
def node_with_ollama_sdk():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is unavailable")
    check = subprocess.run(
        [node, "--input-type=module", "-e", "import('ollama')"],
        cwd=CLIENT_DIR,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    if check.returncode:
        pytest.skip("Ollama JavaScript SDK is unavailable; run `npm ci` in tests/clients")
    return node


@pytest.fixture
def ollama_gateway(tmp_path, monkeypatch, node_with_ollama_sdk):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", _Substrate())
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    monkeypatch.setattr(cs, "ENGINE", _Engine())
    server = ThreadingHTTPServer(("127.0.0.1", 0), cs.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_official_ollama_javascript_sdk_round_trips(node_with_ollama_sdk, ollama_gateway):
    env = os.environ.copy()
    env["CLOZN_OLLAMA_HOST"] = ollama_gateway
    completed = subprocess.run(
        [node_with_ollama_sdk, "--enable-source-maps", str(PROBE)],
        cwd=CLIENT_DIR,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["modelListed"] is True
    run_ids = {value for key, value in result.items() if key.endswith("RunId")}
    assert len(run_ids) == 4
    assert all(runlog.get_run(run_id) is not None for run_id in run_ids)
