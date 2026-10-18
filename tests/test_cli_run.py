"""The retained CLI uses the recorded Chat Completions surface."""
from __future__ import annotations

import json

import pytest

from clozn.cli.commands import run as cli_run


def _sse(value):
    payload = value if isinstance(value, str) else json.dumps(value)
    return f"data: {payload}\n\n".encode("utf-8")


class _Response:
    def __init__(self, frames):
        self.frames = frames

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def __iter__(self):
        return iter(self.frames)


def test_messages_preserve_the_conversation_without_rendering_a_template():
    assert cli_run._messages([("first", "answer")], "second") == [
        {"role": "system", "content": cli_run.SYS},
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "answer"},
        {"role": "user", "content": "second"},
    ]


def test_stream_chat_uses_public_text_api_and_requires_durable_run_id(monkeypatch, capsys):
    seen = {}
    frames = [
        _sse({"choices": [{"delta": {"role": "assistant"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "hel"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {"content": "lo"}, "finish_reason": None}]}),
        _sse({"choices": [{"delta": {}, "finish_reason": "stop"}],
              "clozn_run_id": "run_123"}),
        _sse({"choices": [], "usage": {"completion_tokens": 2}}),
        _sse("[DONE]"),
    ]

    def request(port, body):
        seen.update(port=port, body=body)
        return _Response(frames)

    monkeypatch.setattr(cli_run, "_chat_request", request)
    result = cli_run.stream_chat(
        8080, [{"role": "user", "content": "hi"}], 12
    )

    assert capsys.readouterr().out == "hello"
    assert seen["port"] == 8080
    assert seen["body"]["stream"] is True
    assert seen["body"]["stream_options"] == {"include_usage": True}
    assert result == {
        "text": "hello",
        "finish_reason": "stop",
        "run_id": "run_123",
        "usage": {"completion_tokens": 2},
    }


def test_stream_chat_fails_when_recording_did_not_return_a_run_id(monkeypatch):
    monkeypatch.setattr(
        cli_run,
        "_chat_request",
        lambda *_args: _Response([
            _sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
            _sse("[DONE]"),
        ]),
    )
    with pytest.raises(RuntimeError, match="durable Clozn run id"):
        cli_run.stream_chat(8080, [{"role": "user", "content": "hi"}], 12)
