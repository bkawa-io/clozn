"""Real-handler coverage for the instrumented legacy OpenAI Completions route.

No model or network is used.  The fake substrate exercises the same chat/chat_stream
contract as EngineSubstrate while the real HTTP handler, serializers, and run journal
remain in the loop.
"""
from __future__ import annotations

import io
import json

import pytest

import clozn.settings as clozn_settings

import clozn.runs.store as runlog
from clozn.server import app as cs


class _Memory:
    memory_strength = 0.6
    prefix = None
    rules = []


class _Steer:
    strength = {"concise": 0.25}

    def active(self):
        return dict(self.strength)


class _InstrumentedSubstrate:
    name = "engine"
    brain = None

    def __init__(self):
        self.memory = self._mem = _Memory()
        self.steer = _Steer()
        self.calls = []
        self.fail = False
        self._stream = False
        self.stream_closed = False
        self.reply = "legacy reply"
        self.stream_pieces = ["legacy", " reply"]
        self.finish = "length"

    def _memory(self, messages, mem_out):
        if mem_out is not None:
            mem_out.update(
                mode="prompt",
                applied=[{"id": None, "text": "Be concise.", "relevance": 0.91}],
                gate=0.81,
                strength=0.6,
                prompt_block="Memory:\n- Be concise.",
                assembled_messages=list(messages),
                final_prompt="<rendered>legacy prompt + memory</rendered>",
            )

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._stream = False
        self.calls.append(("chat", list(messages), max_new, sample))
        self._memory(messages, mem_out)
        if self.fail:
            raise RuntimeError("synthetic completion failure")
        if trace_out is not None:
            trace_out.extend([
                {"pos": 0, "token_id": 10, "piece": "legacy", "prob": 0.9},
                {"pos": 1, "token_id": 11, "piece": " reply", "prob": 0.6},
            ])
        return self.reply

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        self._stream = True
        self.calls.append(("stream", list(messages), max_new, sample))
        self._memory(messages, mem_out)
        if self.fail:
            raise RuntimeError("synthetic stream failure")
        class _Request:
            cancelled = False

            def cancel(inner_self):
                inner_self.cancelled = True

        self._request = _Request()
        try:
            yield from self.stream_pieces
        finally:
            self.stream_closed = True

    def last_finish_reason(self):
        return self.finish

    def last_stream_trace(self):
        return [{"pos": pos, "token_id": 10 + pos, "piece": piece, "prob": 0.9 - pos / 10}
                for pos, piece in enumerate(self.stream_pieces)]

    def run_meta(self):
        return {"model_id": "fake-model", "sampler_mode": "sample", "temperature": 0.4,
                "stream": self._stream}


def _dispatch(body, *, writer=None):
    raw_body = json.dumps(body).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = "/v1/completions"
    handler.rfile = io.BytesIO(raw_body)
    handler.wfile = writer or io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw_body)), "User-Agent": "openai-python/test"}
    handler.requestline = "POST /v1/completions HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = "POST"
    handler.close_connection = False
    handler.do_POST()
    return handler.wfile.getvalue()


def _body(raw):
    return json.loads(raw.partition(b"\r\n\r\n")[2])


def _sse(raw):
    frames = []
    for line in raw.partition(b"\r\n\r\n")[2].decode().splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            frames.append(json.loads(line[6:]))
    return frames


@pytest.fixture
def iso(tmp_path, monkeypatch):
    sub = _InstrumentedSubstrate()
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return sub




def test_stream_completion_is_strict_openai_shape_and_journaled(iso):
    raw = _dispatch({"model": "fake-model", "prompt": "stream this", "stream": True,
                     "max_tokens": 9, "temperature": 0})
    frames = _sse(raw)

    assert [frame["choices"][0]["text"] for frame in frames] == ["legacy", " reply", ""]
    assert frames[-1]["choices"][0]["finish_reason"] == "length"
    base_keys = {"id", "object", "created", "model", "choices"}
    assert all(set(frame) == base_keys for frame in frames[:-1])
    assert set(frames[-1]) == base_keys | {"clozn_run_id", "clozn_warnings"}
    assert frames[-1]["clozn_warnings"][0]["code"] == "output_truncated"
    assert all(frame["object"] == "text_completion" for frame in frames)
    assert raw.endswith(b"data: [DONE]\n\n")
    assert b"X-Clozn-Run-Id" not in raw.partition(b"\r\n\r\n")[0]
    assert iso.calls == [("stream", [{"role": "user", "content": "stream this"}], 9,
                          {"temperature": 0.0})]

    rows = runlog.list_runs(5)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert frames[-1]["clozn_run_id"] == logged["id"]
    assert logged["response"] == "legacy reply"
    assert logged["trace"]["token_ids"] == [10, 11]
    assert logged["final_prompt"] == "<rendered>legacy prompt + memory</rendered>"
    assert logged["meta"]["stream"] is True
    assert logged["meta"]["openai_operation"] == "completion"


def test_completion_think_tags_never_reach_nonstream_or_stream_clients(iso):
    iso.reply = "<think>secret</think>answer"
    iso.stream_pieces = ["<thi", "nk>secret</thi", "nk>an", "swer"]
    iso.finish = "stop"

    nonstream = _body(_dispatch({"model": "fake-model", "prompt": "reason"}))
    assert nonstream["choices"][0]["text"] == "answer"
    assert runlog.get_run(nonstream["clozn_run_id"])["reasoning"]["blocks"][0]["text"] == "secret"

    frames = _sse(_dispatch({"model": "fake-model", "prompt": "reason", "stream": True}))
    wire_text = "".join(frame["choices"][0]["text"] for frame in frames)
    assert wire_text == "answer"
    assert "secret" not in wire_text and "think" not in wire_text
    assert runlog.get_run(frames[-1]["clozn_run_id"])["reasoning"]["blocks"][0]["text"] == "secret"


def test_nonstream_generation_failure_is_journaled_and_returned_as_error(iso):
    iso.fail = True
    raw = _dispatch({"model": "fake-model", "prompt": "fail"})
    out = _body(raw)
    assert b" 502 " in raw.partition(b"\r\n\r\n")[0]
    assert out["error"]["type"] == "upstream_error"
    assert "synthetic completion failure" in out["error"]["message"]
    rows = runlog.list_runs(5)
    assert len(rows) == 1
    assert runlog.get_run(rows[0]["id"])["error"] == "synthetic completion failure"


def test_stream_worker_failure_is_in_band_and_journaled(iso):
    iso.fail = True
    raw = _dispatch({"model": "fake-model", "prompt": "fail", "stream": True})
    frames = _sse(raw)
    assert frames == [{"error": {"message": "synthetic stream failure", "type": "upstream_error"}}]
    assert raw.endswith(b"data: [DONE]\n\n")
    rows = runlog.list_runs(5)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert logged["error"] == "synthetic stream failure"
    assert logged["meta"]["stream_failure"] == "worker_disconnected"


def test_stream_client_disconnect_cancels_generation_and_logs_partial_run(iso):
    class _DisconnectOnData(io.BytesIO):
        def write(self, data):
            if data.startswith(b"data: "):
                raise BrokenPipeError("client went away")
            return super().write(data)

    _dispatch({"model": "fake-model", "prompt": "disconnect", "stream": True},
              writer=_DisconnectOnData())

    assert iso._request.cancelled is True
    assert iso.stream_closed is True
    rows = runlog.list_runs(5)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert logged["response"] == "legacy"
    assert "client disconnected mid-stream" in logged["error"]
    assert logged["finish_reason"] is None
    assert logged["meta"]["stream_failure"] == "client_disconnected"
