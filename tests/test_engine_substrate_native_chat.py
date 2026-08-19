"""Model-free tests for EngineSubstrate's private atomic native chat-I/O seam."""
from __future__ import annotations

import json

import pytest

from clozn.server import app as cs


class _AtomicEngine:
    def __init__(self, response=None, error=None):
        self.response = response or _native_response()
        self.error = error
        self.calls = []

    def complete_chat(self, messages, **options):
        self.calls.append({"messages": [dict(message) for message in messages], "options": dict(options)})
        if self.error is not None:
            raise self.error
        return self.response


class _Steer:
    def __init__(self):
        self.strength = {"warm": 0.65}
        self.layer = 14

    def steer_vector(self, strengths):
        assert strengths == self.strength
        return [0.25, -0.5]


def _native_response(*, trace=True, parse_error=None):
    message = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{
            "id": "call_native",
            "type": "function",
            "function": {"name": "weather", "arguments": '{"city":"Kyoto"}'},
        }],
    }
    raw = '  <tool_call>{"name":"weather","arguments":{"city":"Kyoto"}}</tool_call>  '
    chat_io = {
        "raw_model_output": raw,
        "rendered_prompt": "<s>[INST] Weather? [/INST]",
        "model_sha256": "a" * 64,
        "message": message,
        "openai_json": json.dumps(message, separators=(",", ":")),
        "format": "ministral-v3",
        "pipeline": {
            "executor_id": "clozn.chat_io.atomic_executor.v1",
            "renderer_id": "clozn.chat_io.llama_common.renderer.v1",
            "grammar_id": "clozn.chat_io.ar_grammar.v1",
            "parser_id": "clozn.chat_io.llama_common.parser.v1",
        },
        "trace": [],
    }
    if trace:
        chat_io["trace"] = [
            {"type": "tokens_committed", "items": [
                {"pos": 18, "id": 71, "piece": "<tool_call>", "conf": 0.91234},
                {"pos": 19, "id": 72, "piece": "{", "conf": 0.80126},
            ]},
            {"type": "step_lens", "positions": [18], "k": 2,
             "ids": [71, 99], "pieces": ["<tool_call>", "<think>"], "probs": [0.91, 0.04]},
        ]
    if parse_error is not None:
        chat_io.pop("message")
        chat_io.pop("openai_json")
        chat_io["parse_error"] = dict(parse_error)
    return {
        "id": "cmpl-native",
        "object": "text_completion",
        "choices": [{"text": raw, "index": 0, "finish_reason": "stop"}],
        "board": [],
        "layout": [],
        "usage": {"prompt_tokens": 18, "completion_tokens": 2, "steps_total": 2},
        "chat_io": chat_io,
    }


def _substrate(engine, steer=None):
    sub = object.__new__(cs.EngineSubstrate)
    sub.engine = engine
    sub.steer = steer
    sub.memory = object()
    sub._mem = sub.memory
    return sub





def test_private_native_chat_retains_raw_trace_and_usage_when_native_parse_fails(monkeypatch):
    parse_error = {"code": "native_parse_failed", "message": "expected a tool close tag"}
    engine = _AtomicEngine(_native_response(parse_error=parse_error))
    sub = _substrate(engine)
    trace_out = []

    result = sub._complete_chat_native(
        [{"role": "user", "content": "Weather?"}],
        json_schema={"type": "object"},
        sample=False,
        trace_out=trace_out,
    )

    assert result["raw_model_output"] == engine.response["chat_io"]["raw_model_output"]
    assert result["parse_error"] == parse_error
    assert result["message"] is None
    assert result["openai_json"] is None
    assert result["usage"] == engine.response["usage"]
    assert result["trace"] == trace_out == sub._request.trace
    assert [step["piece"] for step in trace_out] == ["<tool_call>", "{"]
    assert sub._request.finish_reason == "stop"
    assert sub._request.prompt_tokens == 18


