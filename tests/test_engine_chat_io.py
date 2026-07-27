"""Model-free wire-contract tests for the private native chat-I/O engine seam."""

import json
import os
import sys

import pytest


HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(REPO_ROOT, "engine", "client"))

from clozn_engine import EngineClient, EngineError  # noqa: E402


def _prepared(**updates):
    value = {
        "prompt": "<user>hello</user><assistant>",
        "grammar": 'root ::= "{" "}"',
        "grammar_lazy": True,
        "grammar_triggers": [{"type": "word", "value": "<tool>", "token": -1}],
        "preserved_tokens": ["<tool>"],
        "additional_stops": ["</tool>"],
        "generation_prompt": "<assistant>",
        "parser": "serialized-parser",
        "format": "generic",
        "capabilities": {"tools": True, "tool_choice": True},
        "supports_thinking": True,
        "thinking_start_tag": "<think>",
        "thinking_end_tag": "</think>",
        "reasoning_format": "none",
        "parse_tool_calls": True,
        "renderer": "llama-common",
        "template_source": "model",
    }
    value.update(updates)
    return value


def _parsed(**updates):
    message = {"role": "assistant", "content": "hello"}
    value = {
        "role": "assistant",
        "content": "hello",
        "reasoning_content": "",
        "tool_name": "",
        "tool_call_id": "",
        "tool_calls": [],
        "openai_json": json.dumps(message, separators=(",", ":")),
        "message": message,
    }
    value.update(updates)
    return value


def _atomic(**updates):
    message = {"role": "assistant", "content": "hello"}
    value = {
        "id": "cmpl-1",
        "object": "text_completion",
        "choices": [{"text": "hello", "index": 0, "finish_reason": "stop"}],
        "board": [1, 2],
        "layout": [
            {"pos": 0, "id": 1, "masked": False, "piece": "hello"},
            {"pos": 1, "id": 2, "masked": False, "piece": ""},
        ],
        "usage": {"prompt_tokens": 4, "completion_tokens": 1, "steps_total": 1},
        "chat_io": {
            "raw_model_output": "hello",
            "rendered_prompt": "<user>hello</user><assistant>",
            "model_sha256": "a" * 64,
            "openai_json": json.dumps(message, separators=(",", ":")),
            "message": message,
            "format": "generic",
            "trace": [],
            "pipeline": {
                "executor_id": "native-executor-v1",
                "renderer_id": "native-renderer-v1",
                "grammar_id": "native-grammar-v1",
                "parser_id": "native-parser-v1",
            },
        },
    }
    value.update(updates)
    return value


def test_prepare_chat_sends_explicit_defaults_and_retains_additive_response_fields(monkeypatch):
    seen = {}
    response = _prepared(worker_contract_version="1.1")
    client = EngineClient(port=1)
    monkeypatch.setattr(
        client,
        "_post",
        lambda path, body: seen.update(path=path, body=body) or response,
    )

    messages = [{"role": "user", "content": "hello"}]
    result = client.prepare_chat(messages)

    assert seen == {
        "path": "/prepare_chat",
        "body": {
            "messages": messages,
            "tools": [],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "add_generation_prompt": True,
            "enable_thinking": True,
            "reasoning_format": "none",
        },
    }
    assert result == response
    assert result["worker_contract_version"] == "1.1"


def test_prepare_chat_forwards_structured_options_without_mutating_inputs(monkeypatch):
    seen = {}
    client = EngineClient(port=1)
    monkeypatch.setattr(
        client,
        "_post",
        lambda path, body: seen.update(path=path, body=body) or _prepared(),
    )
    messages = ({"role": "user", "content": "weather"},)
    tools = ({"type": "function", "function": {"name": "weather"}},)
    tool_choice = {"type": "function", "function": {"name": "weather"}}

    client.prepare_chat(
        messages,
        tools=tools,
        tool_choice=tool_choice,
        parallel_tool_calls=True,
        add_generation_prompt=False,
        enable_thinking=False,
        reasoning_format="deepseek",
    )

    assert seen["path"] == "/prepare_chat"
    assert seen["body"] == {
        "messages": list(messages),
        "tools": list(tools),
        "tool_choice": tool_choice,
        "parallel_tool_calls": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
        "reasoning_format": "deepseek",
    }
    assert seen["body"]["messages"] is not messages
    assert seen["body"]["tools"] is not tools
    assert seen["body"]["tool_choice"] is not tool_choice


def test_prepare_chat_forwards_json_schema_without_active_tools(monkeypatch):
    seen = {}
    client = EngineClient(port=1)
    monkeypatch.setattr(
        client, "_post", lambda path, body: seen.update(path=path, body=body) or _prepared())
    schema = {"type": "object", "properties": {"temperature": {"type": "number"}}}
    client.prepare_chat(
        [{"role": "user", "content": "weather"}],
        tool_choice="none",
        json_schema=schema,
    )
    assert seen["body"]["json_schema"] == schema
    assert seen["body"]["json_schema"] is not schema


@pytest.mark.parametrize(
    ("args", "kwargs", "error"),
    [
        (([],), {}, "messages must be a non-empty array"),
        (("hello",), {}, "messages must be an array"),
        ((["hello"],), {}, r"messages\[0\] must be an object"),
        (([{"role": "user", "content": "hi"}],), {"tools": "bad"}, "tools must be an array"),
        (([{"role": "user", "content": "hi"}],), {"tool_choice": 3}, "tool_choice"),
        (([{"role": "user", "content": "hi"}],), {"json_schema": []}, "json_schema"),
        (([{"role": "user", "content": "hi"}],), {"parallel_tool_calls": 1}, "parallel_tool_calls"),
        (([{"role": "user", "content": "hi"}],), {"reasoning_format": None}, "reasoning_format"),
    ],
)
def test_prepare_chat_rejects_mistyped_requests_before_http(monkeypatch, args, kwargs, error):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match=error):
        client.prepare_chat(*args, **kwargs)


@pytest.mark.parametrize(
    "response",
    [
        [],
        _prepared(prompt=None),
        _prepared(grammar_lazy=1),
        _prepared(grammar_triggers=[{"type": "word", "value": "x", "token": True}]),
        _prepared(preserved_tokens=[1]),
        _prepared(capabilities={"tools": 1}),
        _prepared(renderer=None),
    ],
)
def test_prepare_chat_fails_closed_on_malformed_worker_response(monkeypatch, response):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: response)
    with pytest.raises(EngineError, match="/prepare_chat"):
        client.prepare_chat([{"role": "user", "content": "hello"}])


def test_parse_chat_round_trips_complete_descriptor_and_retains_new_fields(monkeypatch):
    seen = {}
    prepared = _prepared(future_parser_option={"mode": "strict"})
    response = _parsed(future_message_metadata={"qualified": True})
    client = EngineClient(port=1)
    monkeypatch.setattr(
        client,
        "_post",
        lambda path, body: seen.update(path=path, body=body) or response,
    )

    result = client.parse_chat(prepared, "hello", is_partial=True)

    assert seen["path"] == "/parse_chat"
    assert seen["body"] == {
        "prepared": prepared,
        "model_output": "hello",
        "is_partial": True,
    }
    assert seen["body"]["prepared"] is not prepared
    assert seen["body"]["prepared"]["future_parser_option"] == {"mode": "strict"}
    assert result == response


def test_complete_rejects_client_held_prepared_chat_before_http(monkeypatch):
    prepared = _prepared(future_generation_option={"version": 2})
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match="use complete_chat"):
        client.complete(prepared["prompt"], prepared_chat=prepared, max_tokens=12)


def test_complete_chat_sends_one_atomic_nonstream_request_and_preserves_extensions(monkeypatch):
    seen = {}
    response = _atomic(native_contract_version="1.1")
    response["chat_io"]["future_parse_metadata"] = {"strict": True}
    client = EngineClient(port=1)
    monkeypatch.setattr(
        client,
        "_post",
        lambda path, body: seen.update(path=path, body=body) or response,
    )
    messages = [{"role": "user", "content": "hello"}]
    tools = [{"type": "function", "function": {"name": "echo"}}]

    result = client.complete_chat(
        messages,
        tools=tools,
        tool_choice="required",
        parallel_tool_calls=False,
        add_generation_prompt=True,
        enable_thinking=False,
        reasoning_format="none",
        max_tokens=17,
        temperature=0.2,
        top_k=8,
    )

    assert seen == {
        "path": "/v1/completions",
        "body": {
            "chat_request": {
                "messages": messages,
                "tools": tools,
                "tool_choice": "required",
                "parallel_tool_calls": False,
                "add_generation_prompt": True,
                "enable_thinking": False,
                "reasoning_format": "none",
            },
            "stream": False,
            "max_tokens": 17,
            "temperature": 0.2,
            "top_k": 8,
        },
    }
    assert "prompt" not in seen["body"]
    assert "prepared_chat" not in seen["body"]
    assert result == response
    assert result["native_contract_version"] == "1.1"
    assert result["chat_io"]["future_parse_metadata"] == {"strict": True}


def test_complete_chat_preserves_native_parse_failure_evidence(monkeypatch):
    response = _atomic()
    response["chat_io"].pop("message")
    response["chat_io"].pop("openai_json")
    response["chat_io"]["parse_error"] = {
        "code": "native_parse_failed", "message": "invalid tool envelope",
    }
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: response)
    result = client.complete_chat(
        [{"role": "user", "content": "hello"}],
        json_schema={"type": "object"},
    )
    assert result["chat_io"]["raw_model_output"] == "hello"
    assert result["chat_io"]["parse_error"]["code"] == "native_parse_failed"


@pytest.mark.parametrize("max_tokens", [0, -1, 1.5, True])
def test_complete_chat_rejects_invalid_max_tokens_before_http(monkeypatch, max_tokens):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match="max_tokens"):
        client.complete_chat([{"role": "user", "content": "hi"}], max_tokens=max_tokens)


@pytest.mark.parametrize("reserved", ["prompt", "prepared_chat", "chat_request", "stream"])
def test_complete_chat_refuses_atomicity_overrides_before_http(monkeypatch, reserved):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match="reserved field"):
        client.complete_chat(
            [{"role": "user", "content": "hi"}],
            **{reserved: True},
        )


def test_complete_chat_reuses_chat_request_validation_before_http(monkeypatch):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match="messages must be a non-empty array"):
        client.complete_chat([])
    with pytest.raises(ValueError, match="tool_choice"):
        client.complete_chat([{"role": "user", "content": "hi"}], tool_choice=3)
    with pytest.raises(ValueError, match="requires active tools or json_schema"):
        client.complete_chat([{"role": "user", "content": "hi"}])
    with pytest.raises(ValueError, match="mutually exclusive"):
        client.complete_chat(
            [{"role": "user", "content": "hi"}],
            tools=[{"type": "function", "function": {"name": "echo"}}],
            json_schema={"type": "object"},
        )


@pytest.mark.parametrize(
    "response",
    [
        [],
        _atomic(id=None),
        _atomic(choices=[]),
        _atomic(choices=[{"text": "hello", "index": True, "finish_reason": "stop"}]),
        _atomic(chat_io=None),
        _atomic(chat_io={
            "raw_model_output": "different",
            "openai_json": '{"role":"assistant","content":"hello"}',
            "message": {"role": "assistant", "content": "hello"},
            "format": "generic",
        }),
        _atomic(chat_io={
            "raw_model_output": "hello",
            "openai_json": "not-json",
            "message": {"role": "assistant", "content": "hello"},
            "format": "generic",
        }),
        _atomic(chat_io={
            "raw_model_output": "hello",
            "openai_json": '{"role":"assistant","content":"different"}',
            "message": {"role": "assistant", "content": "hello"},
            "format": "generic",
        }),
        _atomic(chat_io={
            "raw_model_output": "hello",
            "openai_json": '{"role":"assistant","content":"hello"}',
            "message": {"role": "assistant", "content": "hello"},
            "format": None,
        }),
    ],
)
def test_complete_chat_fails_closed_on_malformed_or_incoherent_response(monkeypatch, response):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: response)
    with pytest.raises(EngineError, match="/v1/completions"):
        client.complete_chat(
            [{"role": "user", "content": "hello"}],
            json_schema={"type": "object"},
        )


def test_parse_chat_accepts_wire_prepared_object_without_endpoint_metadata(monkeypatch):
    """renderer/template_source describe /prepare_chat, not the descriptor /parse_chat requires."""
    prepared = _prepared()
    prepared.pop("renderer")
    prepared.pop("template_source")
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: _parsed())
    assert client.parse_chat(prepared, "hello")["content"] == "hello"


def test_parse_chat_rejects_malformed_request_before_http(monkeypatch):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: pytest.fail("HTTP must not be called"))
    with pytest.raises(ValueError, match="complete native chat descriptor"):
        client.parse_chat({}, "hello")
    with pytest.raises(ValueError, match="model_output"):
        client.parse_chat(_prepared(), None)
    with pytest.raises(ValueError, match="is_partial"):
        client.parse_chat(_prepared(), "hello", is_partial=1)


@pytest.mark.parametrize(
    "response",
    [
        [],
        _parsed(content=None),
        _parsed(tool_calls=[{"id": "call_1", "name": "weather"}]),
        _parsed(message=[]),
        _parsed(openai_json="not json"),
        _parsed(openai_json='{"role":"assistant","content":"different"}'),
    ],
)
def test_parse_chat_fails_closed_on_malformed_or_incoherent_response(monkeypatch, response):
    client = EngineClient(port=1)
    monkeypatch.setattr(client, "_post", lambda *_: response)
    with pytest.raises(EngineError, match="/parse_chat"):
        client.parse_chat(_prepared(), "hello")
