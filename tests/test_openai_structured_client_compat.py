"""Released OpenAI SDK contract for Clozn's qualified native structured-I/O slice.

The fake substrate exposes the same private atomic result as the C++ worker: raw
model text, a llama-common parsed assistant message (or typed parse error), and the
exact execution-pipeline identity.  The real HTTP gateway must qualify the active
substrate by exact model/template/pipeline identity, independently validate the
native message, buffer structured streams until validation succeeds, and then
serialize ordinary OpenAI Chat Completions objects and deltas.

CI pins ``openai==2.46.0``.  Developer environments without that optional client
skip this file; the dependency-free structured-I/O unit tests remain mandatory.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from http.server import ThreadingHTTPServer

import pytest


openai = pytest.importorskip("openai")

import clozn.memory.cards as memory_cards  # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.memory.mode as memory_mode  # noqa: E402
import clozn.runs.store as runlog  # noqa: E402
from clozn.server import app as cs  # noqa: E402
from clozn.server import structured_io as sio  # noqa: E402


MODEL_SHA256 = "a" * 64
TEMPLATE_FINGERPRINT = "b" * 64
TOOL_RAW_OUTPUT = json.dumps(
    {"name": "get_weather", "arguments": {"city": "Paris"}},
    separators=(",", ":"),
)
TOOL_RAW_OUTPUT = f"<tool_call>{TOOL_RAW_OUTPUT}</tool_call>"
FINAL_RAW_OUTPUT = "Paris is 18 C."
JSON_OBJECT_RAW_OUTPUT = json.dumps(
    {"city": "Paris", "temperature_c": 18},
    separators=(",", ":"),
)
THINK_RAW_OUTPUT = (
    '<think>{"name":"get_weather","arguments":{"city":"Paris"}}</think>'
    "I can answer without a tool."
)
NATIVE_PIPELINE = dict(sio.NATIVE_WORKER_PIPELINE)


class _Memory:
    memory_strength = 1.0
    rules = []
    prefix = None


class _Steer:
    strength = {}

    def active(self):
        return {}


class _StructuredSubstrate:
    """Model-free stand-in for EngineSubstrate's atomic native chat seam.

    ``chat_stream`` intentionally raises for an active contract: structured SSE
    must be fully generated, parsed, validated, and journaled before the gateway
    commits any model-derived bytes.
    """

    name = "engine"
    model_sha256 = MODEL_SHA256
    template_fingerprint = TEMPLATE_FINGERPRINT

    def __init__(self):
        self.memory = self._mem = _Memory()
        self.steer = _Steer()
        self._finish = "stop"
        self.native_calls = []
        self.engine = self._Engine(self)

    class _Engine:
        def __init__(self, substrate):
            self.substrate = substrate

        def health(self):
            return {
                "model": "qualified-local.gguf",
                "model_sha256": self.substrate.model_sha256,
                "native_chat_io": {
                    "available": True,
                    **NATIVE_PIPELINE,
                },
            }

    @staticmethod
    def _raw_reply(messages) -> str:
        texts = [str(message.get("content") or "") for message in messages or []]
        joined = "\n".join(texts)
        if "MALFORMED_PROBE" in joined:
            return "not one JSON object"
        if "REWRITE_PROBE" in joined:
            return ('{"type":"tool_call","name":"<think>junk</think>get_weather",'
                    '"arguments":{"city":"Paris"}}')
        if "THINK_PROBE" in joined:
            return THINK_RAW_OUTPUT
        if "The object itself is the requested answer" in joined:
            return JSON_OBJECT_RAW_OUTPUT
        if '"type":"tool_result"' in joined:
            return FINAL_RAW_OUTPUT
        if any(message.get("role") == "tool" for message in messages or []):
            return FINAL_RAW_OUTPUT
        return TOOL_RAW_OUTPUT

    @staticmethod
    def _native_message(messages, *, tools=None, json_schema=None):
        joined = "\n".join(str(message.get("content") or "") for message in messages or [])
        if "MALFORMED_PROBE" in joined:
            return None
        if "REWRITE_PROBE" in joined:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "native-untrusted",
                    "type": "function",
                    "function": {
                        "name": "<think>junk</think>get_weather",
                        "arguments": '{"city":"Paris"}',
                    },
                }],
            }
        if "THINK_PROBE" in joined:
            return {
                "role": "assistant",
                "content": "I can answer without a tool.",
                "reasoning_content": (
                    '{"name":"get_weather","arguments":{"city":"Paris"}}'
                ),
            }
        if json_schema is not None:
            return {"role": "assistant", "content": JSON_OBJECT_RAW_OUTPUT}
        if any(message.get("role") == "tool" for message in messages or []):
            return {"role": "assistant", "content": FINAL_RAW_OUTPUT}
        assert tools
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                # Native association IDs are deliberately ignored by the public gateway.
                "id": "native-untrusted",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
            }],
        }

    def _complete_chat_native(
        self, messages, *, tools=None, tool_choice="auto", json_schema=None,
        parallel_tool_calls=False, max_new=256, sample=True, trace_out=None,
        mem_out=None, **options,
    ):
        self.native_calls.append({
            "messages": [dict(message) for message in messages],
            "tools": tools,
            "tool_choice": tool_choice,
            "json_schema": json_schema,
            "parallel_tool_calls": parallel_tool_calls,
            "max_new": max_new,
            "sample": sample,
            **options,
        })
        joined = "\n".join(str(message.get("content") or "") for message in messages or [])
        raw = self._raw_reply(messages)
        if json_schema is not None and not any(
            marker in joined for marker in ("MALFORMED_PROBE", "REWRITE_PROBE", "THINK_PROBE")
        ):
            raw = JSON_OBJECT_RAW_OUTPUT
        if mem_out is not None:
            mem_out.update(
                mode="prompt",
                applied=[],
                gate=None,
                assembled_messages=[dict(message) for message in messages],
                final_prompt="<native-rendered-prompt>",
            )
        if trace_out is not None:
            trace_out.append({"pos": 0, "token_id": 91, "piece": raw, "prob": 0.9})
        message = self._native_message(messages, tools=tools, json_schema=json_schema)
        parse_error = None
        if "MALFORMED_PROBE" in joined:
            parse_error = {
                "code": "native_parse_failed",
                "message": "llama-common rejected the generated assistant message",
            }
        result = {
            "raw_model_output": raw,
            "rendered_prompt": "<native-rendered-prompt>",
            "model_sha256": (
                "c" * 64 if "POSTFLIGHT_PROBE" in joined else self.model_sha256
            ),
            "message": message,
            "openai_json": (
                json.dumps(message, separators=(",", ":")) if message is not None else None
            ),
            "format": "test-native-chat-template",
            "pipeline": dict(NATIVE_PIPELINE),
            "parse_error": parse_error,
            "trace": list(trace_out or []),
            "usage": {"prompt_tokens": 8, "completion_tokens": 3, "steps_total": 3},
        }
        if "POSTFLIGHT_PIPELINE_PROBE" in joined:
            result["pipeline"] = {
                **NATIVE_PIPELINE,
                "parser_id": "clozn.chat_io.drifted_parser.v1",
            }
        return result

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        reply = self._raw_reply(messages)
        if mem_out is not None:
            mem_out.update(
                mode="prompt",
                applied=[],
                gate=None,
                assembled_messages=[dict(message) for message in messages],
                final_prompt="\n".join(str(message.get("content") or "") for message in messages),
            )
        if trace_out is not None:
            trace_out.append({"pos": 0, "token_id": 91, "piece": reply, "prob": 0.9})
        return reply

    def chat_stream(self, messages, max_new=256, mem_out=None, **kwargs):
        joined = "\n".join(str(message.get("content") or "") for message in messages or [])
        if "Clozn structured I/O protocol" in joined:
            raise AssertionError("structured SSE must buffer and parse via chat(), not stream raw model text")
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None,
                           assembled_messages=[dict(message) for message in messages],
                           final_prompt=joined)
        yield "Paris is 18 C."

    def last_finish_reason(self):
        return self._finish

    def last_stream_trace(self):
        return []

    def run_meta(self):
        return {
            "model_id": "qualified-local",
            "model_sha256": self.model_sha256,
            "template_fingerprint": self.template_fingerprint,
            "sampler_mode": "greedy",
            "temperature": 0.0,
        }

    def identity_meta(self):
        return {
            "model_sha256": self.model_sha256,
            "template_fingerprint": self.template_fingerprint,
        }


def _registry() -> dict:
    return {
        "schema_version": sio.QUALIFICATION_SCHEMA,
        "entries": [
            {
                "model_sha256": MODEL_SHA256,
                "template_fingerprint": TEMPLATE_FINGERPRINT,
                "features": ["tools", "json_object", "json_schema"],
                "schema_subsets": {
                    "tool_parameters": sio.JSON_SCHEMA_SUBSET_ID,
                    "json_schema": sio.JSON_SCHEMA_SUBSET_ID,
                },
                "pipeline": dict(sio.NATIVE_QUALIFICATION_PIPELINE),
                "evidence": {
                    "schema_version": sio.QUALIFICATION_EVIDENCE_SCHEMA,
                    "suite_id": sio.QUALIFICATION_SUITE_ID,
                    "artifact_version": 2,
                    "payload_sha256": "d" * 64,
                },
            }
        ],
    }


@dataclass
class _Gateway:
    base_url: str
    substrate: _StructuredSubstrate


@pytest.fixture
def structured_gateway(tmp_path, monkeypatch):
    substrate = _StructuredSubstrate()
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "missing.pt")])
    monkeypatch.setattr(cs, "SUB", substrate)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    registry_path = tmp_path / "structured-io-qualifications.json"
    registry_path.write_text(json.dumps(_registry()), encoding="utf-8")
    monkeypatch.setenv(sio.QUALIFICATIONS_ENV, str(registry_path))

    server = ThreadingHTTPServer(("127.0.0.1", 0), cs.make_handler())
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield _Gateway(
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
            substrate=substrate,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _client(gateway: _Gateway):
    return openai.OpenAI(
        api_key="local-test-key",
        base_url=gateway.base_url,
        max_retries=0,
        timeout=5.0,
    )


def _weather_tool() -> dict:
    return {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the current weather for one city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    }


def _error_body(exc) -> dict:
    body = exc.body if isinstance(exc.body, dict) else {}
    return body.get("error", body)


def test_released_sdk_nonstream_tool_call_and_tool_result_continuation(structured_gateway):
    client = _client(structured_gateway)
    tools = [_weather_tool()]
    first = client.chat.completions.create(
        model="qualified-local",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=tools,
        parallel_tool_calls=False,
        temperature=0,
    )

    choice = first.choices[0]
    assert choice.finish_reason == "tool_calls"
    assert choice.message.content is None
    assert choice.message.tool_calls and len(choice.message.tool_calls) == 1
    call = choice.message.tool_calls[0]
    assert call.id.startswith("call_")
    assert call.type == "function"
    assert call.function.name == "get_weather"
    assert json.loads(call.function.arguments) == {"city": "Paris"}

    second = client.chat.completions.create(
        model="qualified-local",
        messages=[
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": call.id, "content": "18 C"},
        ],
        tools=tools,
        parallel_tool_calls=False,
        temperature=0,
    )

    assert second.choices[0].finish_reason == "stop"
    assert second.choices[0].message.content == "Paris is 18 C."
    assert second.choices[0].message.tool_calls is None
    runs = runlog.iter_runs()
    assert len(runs) == 2
    assert all(run.get("error") is None for run in runs)
    # The journal is intentionally newest-first.
    assert [run["response"] for run in runs] == [FINAL_RAW_OUTPUT, TOOL_RAW_OUTPUT]
    outcome = runs[1]["output_contract"]["outcome"]
    assert {key: outcome[key] for key in ("status", "kind", "tool_name")} == {
        "status": "parsed", "kind": "tool_call", "tool_name": "get_weather",
    }
    assert outcome["call_id"] == call.id
    assert runs[1]["output_contract"]["schema"] == "clozn.output_contract.v2"
    assert runs[1]["output_contract"]["native"]["parse_status"] == "parsed"
    assert runs[1]["output_contract"]["qualification"]["pipeline"] == (
        sio.NATIVE_QUALIFICATION_PIPELINE
    )
    assert "tool-call" in runs[1]["flags"]
    assert runs[0]["messages"][1]["tool_calls"][0]["id"] == call.id
    assert runs[0]["messages"][2]["role"] == "tool"


def test_released_sdk_aggregates_buffered_tool_call_sse(structured_gateway):
    stream = _client(structured_gateway).chat.completions.create(
        model="qualified-local",
        messages=[{"role": "user", "content": "What is the weather in Paris?"}],
        tools=[_weather_tool()],
        parallel_tool_calls=False,
        stream=True,
        temperature=0,
    )

    role_seen = False
    call_id = None
    call_name = None
    argument_parts = []
    terminal_finish = None
    terminal_run_id = None
    for chunk in stream:
        assert chunk.choices and chunk.choices[0].index == 0
        choice = chunk.choices[0]
        role_seen = role_seen or choice.delta.role == "assistant"
        for delta_call in choice.delta.tool_calls or []:
            assert delta_call.index == 0
            call_id = delta_call.id or call_id
            if delta_call.function is not None:
                call_name = delta_call.function.name or call_name
                argument_parts.append(delta_call.function.arguments or "")
        if choice.finish_reason is not None:
            terminal_finish = choice.finish_reason
            terminal_run_id = (chunk.model_extra or {}).get("clozn_run_id")

    assert role_seen is True
    assert call_id and call_id.startswith("call_")
    assert call_name == "get_weather"
    assert json.loads("".join(argument_parts)) == {"city": "Paris"}
    assert terminal_finish == "tool_calls"
    runs = runlog.iter_runs()
    assert len(runs) == 1
    assert terminal_run_id == runs[0]["id"]
    assert runs[0]["response"] == TOOL_RAW_OUTPUT
    assert runs[0]["output_contract"]["outcome"]["call_id"] == call_id


def test_tool_result_continuation_can_omit_tools_and_stream_journals_original_history(
        structured_gateway):
    client = _client(structured_gateway)
    first = client.chat.completions.create(
        model="qualified-local",
        messages=[{"role": "user", "content": "Weather?"}],
        tools=[_weather_tool()],
        parallel_tool_calls=False,
    )
    call = first.choices[0].message.tool_calls[0]
    history = [
        {"role": "user", "content": "Weather?"},
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": call.id, "type": "function",
            "function": {"name": call.function.name, "arguments": call.function.arguments},
        }]},
        {"role": "tool", "tool_call_id": call.id, "content": "18 C"},
    ]

    final = client.chat.completions.create(
        model="qualified-local", messages=history, temperature=0,
    )
    assert final.choices[0].message.content == "Paris is 18 C."

    streamed = client.chat.completions.create(
        model="qualified-local", messages=history, tools=[_weather_tool()],
        tool_choice="none", stream=True, temperature=0,
    )
    assert "".join(
        chunk.choices[0].delta.content or "" for chunk in streamed if chunk.choices
    ) == "Paris is 18 C."
    latest = runlog.iter_runs()[0]
    assert latest["messages"][1]["tool_calls"][0]["id"] == call.id
    assert latest["messages"][2]["role"] == "tool"
    assert "clozn-structured-history" not in json.dumps(latest["messages"])


def test_released_sdk_json_object_is_validated_before_return(structured_gateway):
    response = _client(structured_gateway).chat.completions.create(
        model="qualified-local",
        messages=[{"role": "user", "content": "Return the requested answer as JSON."}],
        response_format={"type": "json_object"},
        temperature=0,
    )

    assert response.choices[0].finish_reason == "stop"
    assert json.loads(response.choices[0].message.content) == {
        "city": "Paris",
        "temperature_c": 18,
    }
    assert len(runlog.iter_runs()) == 1


def test_released_sdk_restricted_json_schema_is_validated_before_return(structured_gateway):
    response = _client(structured_gateway).chat.completions.create(
        model="qualified-local",
        messages=[{"role": "user", "content": "Return the requested answer as JSON."}],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "temperature_c": {"type": "integer"},
                    },
                    "required": ["city", "temperature_c"],
                    "additionalProperties": False,
                },
            },
        },
        temperature=0,
    )

    assert json.loads(response.choices[0].message.content) == {
        "city": "Paris", "temperature_c": 18,
    }
    run = runlog.iter_runs()[0]
    assert run["output_contract"]["request"]["mode"] == "json_schema"
    assert run["output_contract"]["outcome"]["status"] == "parsed"


def test_unqualified_active_identity_is_a_named_400_without_generation(structured_gateway):
    structured_gateway.substrate.model_sha256 = "c" * 64

    with pytest.raises(openai.BadRequestError) as caught:
        _client(structured_gateway).chat.completions.create(
            model="qualified-local",
            messages=[{"role": "user", "content": "What is the weather in Paris?"}],
            tools=[_weather_tool()],
            parallel_tool_calls=False,
        )

    error = _error_body(caught.value)
    assert caught.value.status_code == 400
    assert error["type"] == "invalid_request_error"
    assert error["param"] == "tools"
    assert error["code"] == "model_not_qualified"
    assert structured_gateway.substrate.native_calls == []
    assert runlog.iter_runs() == []


def test_malformed_model_output_is_502_and_one_errored_run(structured_gateway):
    with pytest.raises(openai.APIStatusError) as caught:
        _client(structured_gateway).chat.completions.create(
            model="qualified-local",
            messages=[{"role": "user", "content": "MALFORMED_PROBE"}],
            response_format={"type": "json_object"},
            temperature=0,
        )

    error = _error_body(caught.value)
    assert caught.value.status_code == 502
    assert error["type"] == "model_output_error"
    assert error["code"] == "native_parse_failed"
    runs = runlog.iter_runs()
    assert len(runs) == 1
    assert runs[0].get("error")
    assert runs[0]["response"] == "not one JSON object"
    assert runs[0]["output_contract"]["outcome"]["status"] == "error"
    assert runs[0]["output_contract"]["outcome"]["code"] == "native_parse_failed"
    assert runs[0]["output_contract"]["native"]["parse_status"] == "error"
    assert runs[0]["output_contract"]["native"]["parse_error"]["code"] == (
        "native_parse_failed"
    )
    assert "output-parse-error" in runs[0]["flags"]


def test_think_sanitation_cannot_repair_invalid_tool_json(structured_gateway):
    with pytest.raises(openai.APIStatusError) as caught:
        _client(structured_gateway).chat.completions.create(
            model="qualified-local",
            messages=[{"role": "user", "content": "REWRITE_PROBE"}],
            tools=[_weather_tool()],
            parallel_tool_calls=False,
        )
    error = _error_body(caught.value)
    assert caught.value.status_code == 502
    assert error["code"] == "unknown_tool"
    run = runlog.iter_runs()[0]
    assert run["output_contract"]["raw_model_output"].startswith('{"type":"tool_call"')
    assert "<think>junk</think>" in run["output_contract"]["raw_model_output"]
    assert run["output_contract"]["native"]["parse_status"] == "parsed"
    assert run["output_contract"]["validator"]["result"]["allowed_tools"] == [
        "get_weather"
    ]
    assert run["output_contract"]["recovery"]["python_repair"] == "none"


def test_structured_success_fails_if_atomic_journal_persistence_fails(
        structured_gateway, monkeypatch):
    monkeypatch.setattr(runlog, "record", lambda **_kwargs: None)
    with pytest.raises(openai.APIStatusError) as caught:
        _client(structured_gateway).chat.completions.create(
            model="qualified-local",
            messages=[{"role": "user", "content": "Weather?"}],
            tools=[_weather_tool()],
            parallel_tool_calls=False,
        )
    error = _error_body(caught.value)
    assert caught.value.status_code == 502
    assert error["code"] == "journal_persistence_failed"
    assert runlog.iter_runs() == []


def test_tool_call_inside_think_tags_never_becomes_public(structured_gateway):
    response = _client(structured_gateway).chat.completions.create(
        model="qualified-local",
        messages=[{"role": "user", "content": "THINK_PROBE"}],
        tools=[_weather_tool()],
        parallel_tool_calls=False,
        temperature=0,
    )

    assert response.choices[0].finish_reason == "stop"
    assert response.choices[0].message.content == "I can answer without a tool."
    assert response.choices[0].message.tool_calls is None
    run = runlog.iter_runs()[0]
    assert run["response"] == "I can answer without a tool."
    assert run["output_contract"]["raw_model_output"] == THINK_RAW_OUTPUT
    assert "get_weather" in run["reasoning"]["blocks"][0]["text"]
    assert run["output_contract"]["outcome"]["kind"] == "message"
    assert run["output_contract"]["native"]["reasoning_content"].startswith('{"name"')
    assert "tool-call" not in run["flags"]


def test_postflight_model_identity_drift_is_502_and_durably_journaled(structured_gateway):
    with pytest.raises(openai.APIStatusError) as caught:
        _client(structured_gateway).chat.completions.create(
            model="qualified-local",
            messages=[{"role": "user", "content": "POSTFLIGHT_PROBE"}],
            tools=[_weather_tool()],
            parallel_tool_calls=False,
        )

    error = _error_body(caught.value)
    assert caught.value.status_code == 502
    assert error["code"] == "qualification_postflight_mismatch"
    runs = runlog.iter_runs()
    assert len(runs) == 1
    outcome = runs[0]["output_contract"]["outcome"]
    assert outcome["status"] == "error"
    assert outcome["code"] == "qualification_postflight_mismatch"
    assert runs[0]["output_contract"]["native"]["model_sha256"] == "c" * 64


def test_postflight_pipeline_drift_is_502_and_durably_journaled(structured_gateway):
    with pytest.raises(openai.APIStatusError) as caught:
        _client(structured_gateway).chat.completions.create(
            model="qualified-local",
            messages=[{"role": "user", "content": "POSTFLIGHT_PIPELINE_PROBE"}],
            tools=[_weather_tool()],
            parallel_tool_calls=False,
        )

    error = _error_body(caught.value)
    assert caught.value.status_code == 502
    assert error["code"] == "qualification_postflight_mismatch"
    runs = runlog.iter_runs()
    assert len(runs) == 1
    contract = runs[0]["output_contract"]
    assert contract["outcome"]["status"] == "error"
    assert contract["outcome"]["code"] == "qualification_postflight_mismatch"
    assert contract["native"]["pipeline"]["parser_id"] == (
        "clozn.chat_io.drifted_parser.v1"
    )
