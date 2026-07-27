"""Run the released Aider CLI against Clozn's real OpenAI-compatible gateway.

CI pins ``aider-chat==0.86.2``.  A developer checkout without the executable skips this file; request
policy remains covered by dependency-free tests.  This is an actual subprocess invocation, not a
hand-authored approximation of Aider's request body.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from http.server import ThreadingHTTPServer

import pytest

from clozn.server import app as cs
import clozn.settings as clozn_settings


import clozn.runs.store as runlog


AIDER = shutil.which("aider")
pytestmark = pytest.mark.skipif(AIDER is None, reason="aider-chat is not installed")


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

    def __init__(self, answer):
        self.answer = answer
        self.memory = self._mem = _Memory()
        self.steer = _Steer()

    def _fill(self, messages, mem_out):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None, assembled_messages=list(messages))

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._fill(messages, mem_out)
        return self.answer

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        self._fill(messages, mem_out)
        midpoint = max(1, len(self.answer) // 2)
        yield self.answer[:midpoint]
        yield self.answer[midpoint:]

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return []

    def run_meta(self):
        return {"model_id": "clozn-local", "sampler_mode": "greedy", "temperature": 0.0}


@pytest.fixture
def aider_gateway(tmp_path, monkeypatch, request):
    answer = request.param
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", _Substrate(answer))
    server = ThreadingHTTPServer(("127.0.0.1", 0), cs.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", answer
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize(
    "aider_gateway,stream",
    [("Aider non-stream round trip.", False), ("Aider streaming round trip.", True)],
    indirect=["aider_gateway"],
)
def test_released_aider_cli_round_trips_and_journals_exactly_once(aider_gateway, stream, tmp_path):
    base_url, answer = aider_gateway
    env = os.environ.copy()
    env.update({
        "OPENAI_API_BASE": base_url,
        "OPENAI_API_KEY": "local-test-key",
        "AIDER_ANALYTICS": "false",
        "AIDER_CHECK_UPDATE": "false",
        "AIDER_SHOW_RELEASE_NOTES": "false",
    })
    command = [
        AIDER,
        "--model", "openai/clozn-local",
        "--message", f"Reply exactly: {answer}",
        "--no-git",
        "--no-pretty",
        "--stream" if stream else "--no-stream",
        "--yes-always",
        "--no-check-update",
        "--no-show-model-warnings",
        "--map-tokens", "0",
    ]
    completed = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert answer in completed.stdout

    rows = runlog.list_runs(10)
    assert len(rows) == 1
    recorded = runlog.get_run(rows[0]["id"])
    assert recorded["source"] == "openai_api"
    assert recorded["response"] == answer
    assert recorded["finish_reason"] == "stop"
    assert recorded["prompt_summary"].startswith("Reply exactly:")
