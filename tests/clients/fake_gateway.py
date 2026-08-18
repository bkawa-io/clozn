"""Deterministic external-process gateway for released-client conformance jobs."""
from __future__ import annotations

import argparse
import json
import os
import time
from http.server import ThreadingHTTPServer

TOOL_NAME = "lookup_weather"
TOOL_CALL_ID = "call_clozn_weather"
TOOL_ARGUMENTS = {"city": "Paris"}
TOOL_RESULT = {"temperature_f": 72, "condition": "clear"}
TOOL_FINAL_ANSWER = "Paris is 72 F and clear."
TOOL_SPEC = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Return deterministic weather for a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


class _Substrate:
    name = "engine"

    @staticmethod
    def _fill(messages, mem_out):
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], gate=None, assembled_messages=list(messages))

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._fill(messages, mem_out)
        return "External client round trip."

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        self._fill(messages, mem_out)
        yield "External client "
        yield "round trip."

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return []

    def run_meta(self):
        return {"model_id": "clozn-local", "sampler_mode": "greedy", "temperature": 0.0}


def _tool_error(message: str) -> tuple[int, dict]:
    return 400, {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "conformance_probe_mismatch",
        }
    }


def tool_completion(body: dict) -> tuple[int, dict]:
    """Validate and answer the two-turn Open WebUI tool transport probe.

    This belongs only to the deterministic external-client harness. It is not a
    model implementation and intentionally does not alter Clozn's production
    OpenAI route or qualify a real model for function calling.
    """
    if body.get("stream"):
        return _tool_error("the deterministic tool probe is non-streaming")
    if body.get("tools") != [TOOL_SPEC]:
        return _tool_error("Open WebUI did not forward the tool definition unchanged")

    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        return _tool_error("Open WebUI did not forward a message list")
    tool_results = [message for message in messages if message.get("role") == "tool"]

    common = {
        "id": "chatcmpl-openwebui-tool-probe",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": str(body.get("model") or "clozn-local"),
    }
    if not tool_results:
        if messages[-1] != {"role": "user", "content": "What is the weather in Paris?"}:
            return _tool_error("Open WebUI reshaped the initial user message")
        tool_call = {
            "id": TOOL_CALL_ID,
            "type": "function",
            "function": {
                "name": TOOL_NAME,
                "arguments": json.dumps(TOOL_ARGUMENTS, separators=(",", ":")),
            },
        }
        return 200, {
            **common,
            "choices": [{
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [tool_call]},
            }],
        }

    if len(messages) < 3:
        return _tool_error("tool continuation history is incomplete")
    assistant, result = messages[-2:]
    calls = assistant.get("tool_calls") if assistant.get("role") == "assistant" else None
    if not isinstance(calls, list) or len(calls) != 1:
        return _tool_error("assistant tool_calls history was not forwarded")
    call = calls[0]
    function = call.get("function") or {}
    if (call.get("id") != TOOL_CALL_ID or call.get("type") != "function"
            or function.get("name") != TOOL_NAME):
        return _tool_error("assistant tool call was reshaped incompatibly")
    try:
        forwarded_arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError):
        return _tool_error("assistant tool arguments were not forwarded as JSON text")
    if forwarded_arguments != TOOL_ARGUMENTS:
        return _tool_error("assistant tool arguments were reshaped incompatibly")
    if result.get("role") != "tool" or result.get("tool_call_id") != TOOL_CALL_ID:
        return _tool_error("matching tool result history was not forwarded")
    if result.get("name") != TOOL_NAME:
        return _tool_error("tool result name was not forwarded")
    try:
        forwarded_result = json.loads(result.get("content", ""))
    except (TypeError, json.JSONDecodeError):
        return _tool_error("tool result content was not valid JSON text")
    if forwarded_result != TOOL_RESULT:
        return _tool_error("tool result content was reshaped incompatibly")

    return 200, {
        **common,
        "choices": [{
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": TOOL_FINAL_ANSWER},
        }],
    }


def make_handler():
    from clozn.server import app as cs

    base_handler = cs.make_handler()

    class ConformanceHandler(base_handler):
        def _dispatch_post(self, path, body):
            if path == "/v1/chat/completions" and "tools" in body:
                status, payload = tool_completion(body)
                self._json(status, payload)
                return
            super()._dispatch_post(path, body)

    return ConformanceHandler


def main() -> int:
    from clozn.server import app as cs
    import clozn.runs.store as runlog

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18181)
    args = parser.parse_args()
    runlog.RUNS_DIR = os.environ.get("CLOZN_TEST_RUNS_DIR", runlog.RUNS_DIR)
    cs.SUB = _Substrate()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler())
    print(f"ready http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

