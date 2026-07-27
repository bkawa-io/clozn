"""clozn_engine.py — the Python SDK for the clozn-server white-box HTTP API.

The C++ engine (engine/core/serve/clozn_server.cpp) exposes a model's interior over HTTP:
READ activations (`/harvest`), WRITE them back and observe the effect (`/state`), and
STEER a generation (`/intervene`). Those endpoints close the read -> edit -> write ->
observe loop on a live ggml/llama.cpp model. This module is the thin Python seam over
them, so the research stack (SAE discovery, feature circuits, concept probes — all numpy
already) can drive the production engine instead of a separate HF model:

    from clozn_engine import EngineClient
    eng = EngineClient(port=8080)
    h = eng.harvest("The capital of France is")      # h.activations: [n_tokens, n_embd] f32
    # ... run a discovery harness on h.activations (SAE encode, PCA, a learned edit) ...
    obs = eng.write_state("The capital of France is", h.layer,
                          positions=[h.n_tokens - 1], values=edited_last_row)
    print(obs.moved_l2, obs.baseline_top, obs.edited_top)   # how the next token moved

Dependencies are deliberately minimal: the standard library for HTTP/JSON/base64 plus
numpy for the activation matrices. No `requests`, no client framework.

The wire format for tensors is SPEC.md's {dtype, shape, data}, where `data` is the
base64 of the raw little-endian float32 bytes. x86 and CUDA are little-endian, so the
in-memory floats ARE those bytes; decoding is a straight np.frombuffer(..., '<f4').

Run `python clozn_engine.py --selftest` to validate the codec offline (no server), or
`python clozn_engine.py --demo` to run a live read -> edit -> write -> observe round-trip
against a running clozn-server.
"""

from __future__ import annotations

import argparse
import base64
import json
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float], Sequence[Sequence[float]]]


# --------------------------------------------------------------------------- wire codec

def decode_tensor(obj: dict) -> np.ndarray:
    """Decode a wire tensor {dtype:"float32", shape:[...], data:base64-LE} to a numpy array.

    Mirrors tensor_json_f32 in clozn_server.cpp: the bytes are little-endian float32, row
    major, so np.frombuffer('<f4').reshape(shape) reconstructs the matrix exactly (no copy
    beyond the base64 decode). Raises on a non-float32 dtype or a shape/byte-count mismatch.
    """
    dtype = obj.get("dtype")
    if dtype != "float32":
        raise ValueError(f"unsupported wire dtype {dtype!r} (only float32)")
    shape = tuple(int(d) for d in obj["shape"])
    raw = base64.b64decode(obj["data"])
    arr = np.frombuffer(raw, dtype="<f4")
    expected = int(np.prod(shape)) if shape else 0
    if arr.size != expected:
        raise ValueError(f"tensor byte count {arr.size} != shape product {expected} {shape}")
    return arr.reshape(shape)


def flatten_values(values: ArrayLike) -> list:
    """Flatten an edit (a [P, n_embd] matrix or already-flat vector) to the row-major list of
    Python floats /state expects. The server reads it as a std::vector<float> and checks
    values.size() == positions.size() * n_embd, so the order must be position-major."""
    arr = np.ascontiguousarray(np.asarray(values, dtype="<f4")).reshape(-1)
    return arr.tolist()


# --------------------------------------------------------------------------- result types

@dataclass
class Harvest:
    """The result of POST /harvest: every input token's residual at the tap `layer`."""
    tokens: list[str]                 # decoded piece per input token
    layer: int                        # the layer actually read (server may clamp the request)
    activations: np.ndarray           # [n_tokens, n_embd], float32

    @property
    def n_tokens(self) -> int:
        return int(self.activations.shape[0])

    @property
    def n_embd(self) -> int:
        return int(self.activations.shape[1])


@dataclass
class Observation:
    """The result of POST /state: how the model's next-token prediction moved under the write."""
    applied: bool                     # False if the write was rejected (bad layer / size)
    layer: int
    moved_l2: float                   # L2 distance between baseline and edited logit vectors
    baseline_top: list = field(default_factory=list)   # [{token, prob}, ...] top-3 before
    edited_top: list = field(default_factory=list)      # [{token, prob}, ...] top-3 after
    error: Optional[str] = None

    def shifted(self) -> bool:
        """True iff the argmax next token changed under the write (a visible behavioral effect)."""
        return bool(self.applied and self.baseline_top and self.edited_top
                    and self.baseline_top[0]["token"] != self.edited_top[0]["token"])

    def summary(self) -> str:
        if not self.applied:
            return f"rejected: {self.error}"
        b = ", ".join(f"{t['token']!r} {t['prob']:.3f}" for t in self.baseline_top)
        e = ", ".join(f"{t['token']!r} {t['prob']:.3f}" for t in self.edited_top)
        flag = "  [TOP-1 SHIFTED]" if self.shifted() else ""
        return f"moved_l2={self.moved_l2:.3f}{flag}\n  baseline: {b}\n  edited:   {e}"


# --------------------------------------------------------------------------- the client

class EngineError(RuntimeError):
    """An error returned by the engine (non-2xx with a JSON {error: ...} body)."""


def _chat_io_object(value: object, label: str) -> dict[str, Any]:
    """Copy a JSON object or fail before a malformed private chat-I/O round trip.

    The native chat descriptor is intentionally returned as an ordinary dict: callers must be
    able to preserve fields added by a newer worker when they send it back to /parse_chat.  These
    small validators therefore check only the fields this client must understand and retain every
    unknown field verbatim.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _chat_io_sequence(value: object, label: str, *, nonempty: bool = False) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    result = list(value)
    if nonempty and not result:
        raise ValueError(f"{label} must be a non-empty array")
    return result


def _require_chat_io_response_object(value: object, endpoint: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise EngineError(f"POST {endpoint} returned a non-object JSON response")
    return dict(value)


def _require_response_type(response: Mapping[str, Any], field_name: str, expected: type,
                           endpoint: str) -> Any:
    value = response.get(field_name)
    # bool is an int subclass, so an integer field must explicitly exclude it.
    valid = isinstance(value, expected) and not (expected is int and isinstance(value, bool))
    if not valid:
        raise EngineError(
            f"POST {endpoint} returned invalid {field_name!r} (expected {expected.__name__})")
    return value


def _validate_prepared_chat_response(value: object, *, require_metadata: bool = True) -> dict[str, Any]:
    endpoint = "/prepare_chat"
    response = _require_chat_io_response_object(value, endpoint)
    for field_name in (
        "prompt", "grammar", "generation_prompt", "parser", "format",
        "thinking_start_tag", "thinking_end_tag", "reasoning_format",
    ):
        _require_response_type(response, field_name, str, endpoint)
    if require_metadata:
        for field_name in ("renderer", "template_source"):
            _require_response_type(response, field_name, str, endpoint)
    for field_name in ("grammar_lazy", "supports_thinking", "parse_tool_calls"):
        _require_response_type(response, field_name, bool, endpoint)

    for field_name in ("preserved_tokens", "additional_stops"):
        items = _require_response_type(response, field_name, list, endpoint)
        if any(not isinstance(item, str) for item in items):
            raise EngineError(f"POST {endpoint} returned invalid {field_name!r} (expected strings)")

    capabilities = _require_response_type(response, "capabilities", dict, endpoint)
    if any(not isinstance(name, str) or not isinstance(enabled, bool)
           for name, enabled in capabilities.items()):
        raise EngineError(f"POST {endpoint} returned invalid 'capabilities'")

    triggers = _require_response_type(response, "grammar_triggers", list, endpoint)
    for index, trigger in enumerate(triggers):
        if not isinstance(trigger, Mapping):
            raise EngineError(f"POST {endpoint} returned invalid grammar_triggers[{index}]")
        if not isinstance(trigger.get("type"), str) or not isinstance(trigger.get("value"), str):
            raise EngineError(f"POST {endpoint} returned invalid grammar_triggers[{index}]")
        token = trigger.get("token")
        if not isinstance(token, int) or isinstance(token, bool):
            raise EngineError(f"POST {endpoint} returned invalid grammar_triggers[{index}]")
    return response


def _validate_prepared_chat_request(value: object, label: str = "prepared") -> dict[str, Any]:
    prepared = _chat_io_object(value, label)
    try:
        return _validate_prepared_chat_response(prepared, require_metadata=False)
    except EngineError as exc:
        raise ValueError(f"{label} is not a complete native chat descriptor: {exc}") from None


def _normalize_chat_request(
    messages: Sequence[Mapping[str, Any]],
    *,
    tools: Optional[Sequence[Mapping[str, Any]]] = None,
    tool_choice: Union[str, Mapping[str, Any]] = "auto",
    json_schema: Optional[Mapping[str, Any]] = None,
    parallel_tool_calls: bool = False,
    add_generation_prompt: bool = True,
    enable_thinking: bool = True,
    reasoning_format: str = "none",
) -> dict[str, Any]:
    message_items = _chat_io_sequence(messages, "messages", nonempty=True)
    normalized_messages = [
        _chat_io_object(message, f"messages[{index}]")
        for index, message in enumerate(message_items)
    ]
    tool_items = _chat_io_sequence([] if tools is None else tools, "tools")
    normalized_tools = [
        _chat_io_object(tool, f"tools[{index}]")
        for index, tool in enumerate(tool_items)
    ]
    if not isinstance(tool_choice, (str, Mapping)):
        raise ValueError("tool_choice must be a string or object")
    if not isinstance(parallel_tool_calls, bool):
        raise ValueError("parallel_tool_calls must be a bool")
    if not isinstance(add_generation_prompt, bool):
        raise ValueError("add_generation_prompt must be a bool")
    if not isinstance(enable_thinking, bool):
        raise ValueError("enable_thinking must be a bool")
    if not isinstance(reasoning_format, str):
        raise ValueError("reasoning_format must be a string")
    active_tools = bool(normalized_tools) and tool_choice != "none"
    if active_tools and json_schema is not None:
        raise ValueError("json_schema and active tools are mutually exclusive")

    body: dict[str, Any] = {
        "messages": normalized_messages,
        "tools": normalized_tools,
        "tool_choice": dict(tool_choice) if isinstance(tool_choice, Mapping) else tool_choice,
        "parallel_tool_calls": parallel_tool_calls,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
        "reasoning_format": reasoning_format,
    }
    if json_schema is not None:
        body["json_schema"] = _chat_io_object(json_schema, "json_schema")
    return body


def _validate_openai_message_pair(container: Mapping[str, Any], endpoint: str) -> None:
    openai_json = _require_response_type(container, "openai_json", str, endpoint)
    message = container.get("message")
    if not isinstance(message, Mapping):
        raise EngineError(f"POST {endpoint} returned invalid 'message' (expected object)")
    try:
        decoded_message = json.loads(openai_json)
    except json.JSONDecodeError:
        raise EngineError(f"POST {endpoint} returned invalid 'openai_json'") from None
    if not isinstance(decoded_message, dict) or decoded_message != dict(message):
        raise EngineError(f"POST {endpoint} returned inconsistent message/openai_json")


def _validate_parsed_chat_response(value: object) -> dict[str, Any]:
    endpoint = "/parse_chat"
    response = _require_chat_io_response_object(value, endpoint)
    for field_name in (
        "role", "content", "reasoning_content", "tool_name", "tool_call_id",
    ):
        _require_response_type(response, field_name, str, endpoint)

    calls = _require_response_type(response, "tool_calls", list, endpoint)
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping) or any(
            not isinstance(call.get(field_name), str)
            for field_name in ("id", "name", "arguments")
        ):
            raise EngineError(f"POST {endpoint} returned invalid tool_calls[{index}]")

    _validate_openai_message_pair(response, endpoint)
    return response


def _validate_atomic_chat_response(value: object) -> dict[str, Any]:
    endpoint = "/v1/completions"
    response = _require_chat_io_response_object(value, endpoint)
    for field_name in ("id", "object"):
        _require_response_type(response, field_name, str, endpoint)
    _require_response_type(response, "board", list, endpoint)
    _require_response_type(response, "layout", list, endpoint)
    _require_response_type(response, "usage", dict, endpoint)

    choices = _require_response_type(response, "choices", list, endpoint)
    if not choices or not isinstance(choices[0], Mapping):
        raise EngineError(f"POST {endpoint} returned invalid 'choices'")
    first_choice = choices[0]
    text = first_choice.get("text")
    index = first_choice.get("index")
    finish_reason = first_choice.get("finish_reason")
    if (not isinstance(text, str) or not isinstance(index, int) or isinstance(index, bool)
            or not isinstance(finish_reason, str)):
        raise EngineError(f"POST {endpoint} returned invalid 'choices[0]'")

    chat_io = response.get("chat_io")
    if not isinstance(chat_io, Mapping):
        raise EngineError(f"POST {endpoint} returned invalid 'chat_io' (expected object)")
    raw_output = _require_response_type(chat_io, "raw_model_output", str, endpoint)
    _require_response_type(chat_io, "rendered_prompt", str, endpoint)
    model_sha256 = _require_response_type(chat_io, "model_sha256", str, endpoint)
    if (len(model_sha256) != 64
            or any(ch not in "0123456789abcdef" for ch in model_sha256)):
        raise EngineError(f"POST {endpoint} returned invalid 'chat_io.model_sha256'")
    _require_response_type(chat_io, "format", str, endpoint)
    trace = _require_response_type(chat_io, "trace", list, endpoint)
    if any(not isinstance(frame, Mapping) for frame in trace):
        raise EngineError(f"POST {endpoint} returned invalid 'chat_io.trace'")
    pipeline = chat_io.get("pipeline")
    if not isinstance(pipeline, Mapping) or any(
        not isinstance(pipeline.get(field_name), str)
        for field_name in ("executor_id", "renderer_id", "grammar_id", "parser_id")
    ):
        raise EngineError(f"POST {endpoint} returned invalid 'chat_io.pipeline'")
    parse_error = chat_io.get("parse_error")
    if parse_error is not None:
        if (not isinstance(parse_error, Mapping)
                or not isinstance(parse_error.get("code"), str)
                or not isinstance(parse_error.get("message"), str)):
            raise EngineError(f"POST {endpoint} returned invalid 'chat_io.parse_error'")
        if "message" in chat_io or "openai_json" in chat_io:
            raise EngineError(f"POST {endpoint} returned incoherent parsed fields and parse_error")
    else:
        _validate_openai_message_pair(chat_io, endpoint)
    if raw_output != text:
        raise EngineError(f"POST {endpoint} returned inconsistent choices[0].text/chat_io.raw_model_output")
    return response


class EngineClient:
    """A thin HTTP client for one running clozn-server.

    All calls are synchronous. The server serializes generation on its context pool, so
    concurrent calls from multiple clients are fine (they queue on a free worker); within
    one client the calls are sequential by construction.
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8080, timeout: float = 120.0):
        self.base = f"http://{host}:{port}"
        self.timeout = timeout

    # -- transport -----------------------------------------------------------

    def _request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json"} if data else {})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            # The engine reports client errors as 400 + {"error": "..."}; surface that message.
            payload = e.read().decode("utf-8", "replace")
            try:
                msg = json.loads(payload).get("error", payload)
            except json.JSONDecodeError:
                msg = payload
            raise EngineError(f"{method} {path} -> {e.code}: {msg}") from None

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    # -- endpoints -----------------------------------------------------------

    def health(self) -> dict:
        """GET /health -> runtime/model capabilities including n_layer/n_embd/vocab_size."""
        return self._get("/health")

    def harvest(self, text: str, layer: Optional[int] = None) -> Harvest:
        """POST /harvest: read every token's residual at the tap layer in ONE causal forward.

        `layer` overrides the server's default tap (the calibrated early read layer). An
        out-of-range layer falls back to the final layer server-side; the Harvest carries the
        layer actually used, so thread Harvest.layer into write_state to read and write at the
        same depth.
        """
        body: dict = {"text": text}
        if layer is not None:
            body["layer"] = int(layer)
        r = self._post("/harvest", body)
        return Harvest(tokens=r["tokens"], layer=int(r["layer"]),
                       activations=decode_tensor(r["activations"]))

    def harvest_layers(self, text: str) -> dict:
        """POST /harvest/layers: per-layer activation SUMMARY in ONE causal forward -- the L2 norm of every
        token's residual at EVERY layer (the depth x position "MRI" map) + a per-layer mean. Unlike
        harvest() (one layer's full tensor), this is the cheap cross-depth view: one forward, all layers.
        Returns {tokens, n_tokens, n_layer, norms:[n_layer][n_tokens], layer_mean:[n_layer]} -- plain
        floats (no tensor codec), so it's handed back as-is for the UI to render."""
        r = self._post("/harvest/layers", {"text": text})
        return {"tokens": r.get("tokens", []),
                "n_tokens": int(r.get("n_tokens", 0)),
                "n_layer": int(r.get("n_layer", 0)),
                "norms": r.get("norms", []),
                "layer_mean": r.get("layer_mean", [])}

    def write_state(self, text: str, layer: int, positions: Sequence[int],
                    values: ArrayLike) -> Observation:
        """POST /state: overwrite `positions`' residual at `layer` with `values`, then observe.

        `values` is a [len(positions), n_embd] matrix (or the equivalent flat vector); it is
        flattened position-major to match the server's contract. The server runs a baseline
        forward, applies the write via the eval-callback activation patch, runs again, clears
        the write, and reports how the next-token logits moved.
        """
        positions = [int(p) for p in positions]
        body = {"text": text, "layer": int(layer), "positions": positions,
                "values": flatten_values(values)}
        r = self._post("/state", body)
        return Observation(applied=bool(r.get("applied", False)),
                           layer=int(r.get("layer", layer)),
                           moved_l2=float(r.get("moved_l2", 0.0)),
                           baseline_top=r.get("baseline_top", []),
                           edited_top=r.get("edited_top", []),
                           error=r.get("error"))

    def edit_and_observe(self, text: str, transform: Optional[Callable[[np.ndarray], np.ndarray]] = None,
                         layer: Optional[int] = None,
                         positions: Optional[Sequence[int]] = None) -> tuple[Harvest, Observation]:
        """The full loop in one call: harvest `text`, apply `transform`, write the edit back.

        `transform(acts) -> acts` receives a copy of the [n_tokens, n_embd] matrix and returns
        the edited matrix of the same shape (default: identity, a no-op write that should move
        nothing — a useful sanity check). The write happens at the SAME layer the harvest read
        from (Harvest.layer), which is what makes editing-then-writing a row meaningful. By
        default only the rows the transform actually changed are written back; pass `positions`
        to force a specific set. Returns (harvest, observation).
        """
        h = self.harvest(text, layer)
        edited = h.activations.copy() if transform is None else np.asarray(transform(h.activations.copy()))
        edited = np.ascontiguousarray(edited, dtype="<f4")
        if edited.shape != h.activations.shape:
            raise ValueError(f"transform changed the shape {h.activations.shape} -> {edited.shape}")
        if positions is None:
            changed = np.nonzero(np.abs(edited - h.activations).sum(axis=1) > 0.0)[0]
            positions = changed.tolist() if changed.size else list(range(h.n_tokens))
        rows = edited[list(positions)]
        obs = self.write_state(text, h.layer, positions, rows)
        return h, obs

    def intervene(self, prompt: str, concept: Optional[str] = None, coef: float = 1.0,
                  vector: Optional[ArrayLike] = None, layer: int = 0, **gen) -> dict:
        """POST /intervene (kind:"steer"): push a direction into the residual during generation.

        Either a NAMED concept (one of the server's calibrated probes — see the 'available'
        list it returns on an unknown name) or a RAW `vector` of length n_embd. `coef` scales
        it; `layer` pins the steer layer (0 = the calibrated mid-depth band). Extra generation
        params (max_tokens, steps, topk, ...) pass through as `target`.
        """
        if concept is None and vector is None:
            raise ValueError("intervene needs a concept name or a raw vector")
        target = dict(gen)
        target["prompt"] = prompt
        body: dict = {"kind": "steer", "coef": float(coef), "layer": int(layer), "target": target}
        if concept is not None:
            body["concept"] = concept
        if vector is not None:
            body["vector"] = flatten_values(vector)
        return self._post("/intervene", body)

    def complete(self, prompt: str, **params) -> dict:
        """POST /v1/completions: a plain generation (no white-box). params: max_tokens, steps,
        topk, temperature, ... Returns the OpenAI-ish body {choices, board, layout, usage}."""
        body = dict(params)
        if "prepared_chat" in body:
            raise ValueError(
                "client-held prepared_chat generation is unsupported; use complete_chat() atomically")
        body["prompt"] = prompt
        return self._post("/v1/completions", body)

    def complete_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Union[str, Mapping[str, Any]] = "auto",
        json_schema: Optional[Mapping[str, Any]] = None,
        parallel_tool_calls: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_format: str = "none",
        max_tokens: int = 32,
        **options: Any,
    ) -> dict[str, Any]:
        """Atomically prepare, constrain, generate, and parse one private native chat request.

        Unlike the low-level prepare_chat/complete/parse_chat sequence, the native worker retains
        the prepared grammar and parser for the whole operation. Structured generation is buffered
        and validated, so this helper always sends ``stream:false`` and cannot target diffusion.
        Unknown response fields are retained after the completion and chat-I/O invariants pass.
        """
        if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer")
        reserved = {"prompt", "prepared_chat", "chat_request", "stream"}.intersection(options)
        if reserved:
            names = ", ".join(sorted(reserved))
            raise ValueError(f"complete_chat options cannot override reserved field(s): {names}")
        chat_request = _normalize_chat_request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_schema=json_schema,
            parallel_tool_calls=parallel_tool_calls,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_format=reasoning_format,
        )
        active_tools = bool(chat_request["tools"]) and chat_request["tool_choice"] != "none"
        if not active_tools and "json_schema" not in chat_request:
            raise ValueError("complete_chat requires active tools or json_schema")
        body = dict(options)
        body.update({
            "chat_request": chat_request,
            "stream": False,
            "max_tokens": max_tokens,
        })
        return _validate_atomic_chat_response(self._post("/v1/completions", body))

    def apply_template(self, messages: Sequence[dict], add_assistant: bool = True) -> str:
        """POST /apply_template: render chat `messages` into a prompt string using THE MODEL'S OWN
        embedded chat template (the GGUF's tokenizer.chat_template), applied server-side. This is what
        makes clozn model-agnostic -- Qwen gets ChatML, Llama-3 gets its header format, Gemma gets
        <start_of_turn>, etc. -- instead of a hardcoded Qwen string. `messages` is [{role, content}];
        `add_assistant` ends the prompt with the assistant-turn opener (the generation cue). Raises
        EngineError if the model has no embedded template (never silently mis-formats). Returns the
        rendered prompt string."""
        return self.apply_template_info(messages, add_assistant=add_assistant)["prompt"]

    def apply_template_info(self, messages: Sequence[dict], add_assistant: bool = True) -> dict[str, Any]:
        """Render with :meth:`apply_template` plus exact worker token-count evidence when available.

        New workers return ``prompt_tokens`` from the same tokenizer seam used for generation.  Older
        workers returned only ``prompt``; that response remains valid and simply omits the optional
        count.  :meth:`apply_template` continues to return only the prompt string.
        """
        r = self._post("/apply_template", {
            "messages": list(messages),
            "add_assistant": bool(add_assistant),
        })
        info = {"prompt": r["prompt"]}
        if "prompt_tokens" in r:
            prompt_tokens = r["prompt_tokens"]
            if (not isinstance(prompt_tokens, int) or isinstance(prompt_tokens, bool)
                    or prompt_tokens < 0):
                raise EngineError(
                    "POST /apply_template returned invalid 'prompt_tokens' "
                    "(expected non-negative int)")
            info["prompt_tokens"] = prompt_tokens
        return info

    def prepare_chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
        tool_choice: Union[str, Mapping[str, Any]] = "auto",
        json_schema: Optional[Mapping[str, Any]] = None,
        parallel_tool_calls: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = True,
        reasoning_format: str = "none",
    ) -> dict[str, Any]:
        """POST /prepare_chat: build the native model-specific chat generation descriptor.

        This is a private worker seam, not a public compatibility promise.  The returned object
        contains the rendered prompt plus llama-common grammar, trigger, stop, and parser state.  It
        is intentionally kept as a dict so fields from a newer worker survive an unchanged round trip
        into :meth:`parse_chat`; known fields are validated so generation never proceeds from a
        partial or mistyped descriptor.
        """
        body = _normalize_chat_request(
            messages,
            tools=tools,
            tool_choice=tool_choice,
            json_schema=json_schema,
            parallel_tool_calls=parallel_tool_calls,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
            reasoning_format=reasoning_format,
        )
        return _validate_prepared_chat_response(self._post("/prepare_chat", body))

    def parse_chat(self, prepared: Mapping[str, Any], model_output: str, *,
                   is_partial: bool = False) -> dict[str, Any]:
        """POST /parse_chat: parse native output with its matching prepared descriptor.

        Unknown descriptor fields are forwarded unchanged.  This lets the client bridge compatible
        additive worker upgrades while the worker remains the sole owner of parser semantics.
        """
        # Validate locally before returning a descriptor to the worker. Metadata added by the endpoint
        # is retained; only the known generation/parse fields are required.
        normalized_prepared = _validate_prepared_chat_request(prepared)
        if not isinstance(model_output, str):
            raise ValueError("model_output must be a string")
        if not isinstance(is_partial, bool):
            raise ValueError("is_partial must be a bool")
        body = {
            "prepared": normalized_prepared,
            "model_output": model_output,
            "is_partial": is_partial,
        }
        return _validate_parsed_chat_response(self._post("/parse_chat", body))

    def score(self, prompt: Optional[str] = None, prompt_ids: Optional[Sequence[int]] = None,
              continuation_ids: Optional[Sequence[int]] = None, continuation: Optional[str] = None,
              topk: int = 0, steer: Optional[dict] = None, steer_vec: Optional[ArrayLike] = None,
              attn_knockout: Optional[Sequence[Mapping[str, Any]]] = None) -> dict:
        """POST /score: teacher-forced per-token logprob of a continuation under given conditions --
        NEVER sampling (the reproduce-and-prove foundation).
        One causal decode of prompt++continuation on the engine reads back, for each continuation
        token, the log-softmax probability the model assigned to the token it was actually forced to
        see next -- usable both to verify a generated reply (self-consistency) and to measure how
        much an influence (memory block / tone dial) shaped an answer (score WITH vs WITHOUT it).

        `prompt_ids` (exact token ids, e.g. from a stored trace) take precedence over `prompt` text;
        likewise `continuation_ids` is the PRIMARY continuation form -- `continuation` text is a
        fallback that retokenizes independently and can drift at the prompt/continuation BPE boundary
        (the server flags this `boundary_approximate` in the response; treat it as approximate).
        `steer`/`steer_vec` mirror /v1/completions' dial path (a raw n_embd direction + {coef, layer}),
        so a scored call can reproduce a steered run's conditions.
        `attn_knockout` (Phase 4.2, roadmap §7 item 2): zero or more {layer, queries, keys,
        renormalize?} attention-edge cuts applied during THIS forward -- see
        clozn.receipts.hook_vocabulary's kq_soft_max-<il> entry for exact semantics and the
        --no-flash-attn requirement (GET /health.capabilities.attn_knockout). Passed through verbatim;
        this client does not validate the specs (the engine does, and refuses cleanly).

        Returns {n_prompt, n_cont, tokens:[{id, piece, logprob, topk?}], sum_logprob}.
        """
        body: dict = {"topk": int(topk)}
        if prompt_ids is not None:
            body["prompt_ids"] = [int(x) for x in prompt_ids]
        elif prompt is not None:
            body["prompt"] = prompt
        if continuation_ids is not None:
            body["continuation_ids"] = [int(x) for x in continuation_ids]
        elif continuation is not None:
            body["continuation"] = continuation
        if steer is not None:
            body["steer"] = steer
        if steer_vec is not None:
            body["steer_vec"] = flatten_values(steer_vec)
        if attn_knockout is not None:
            body["attn_knockout"] = [dict(spec) for spec in attn_knockout]
        return self._post("/score", body)

    def cancel(self, req_id: str) -> dict:
        """POST /cancel: cooperative cancel of a running generation. Idempotent — an unknown or
        already-finished id returns {cancelled: false}. Returns {cancelled: bool, req: str}."""
        return self._post("/cancel", {"req": req_id})

    def jlens(self, text: str, layer: Optional[int] = None, topk: int = 5) -> dict:
        """POST /jlens: the J-lens (Jacobian-lens) readout -- per position, the top-k tokens that
        position is 'disposed to say later' (Anthropic 2026, transferred to this GGUF).
        Deterministic linear read, NO sampling. `layer` selects a fitted
        sidecar (omit -> the engine's default tap); an unloaded layer 400s with the available layers.
        Returns {layer, n_tokens, tokens:[piece...], readouts:[[{id,piece,score}...topk]...n_tokens]}."""
        body: dict = {"text": text, "topk": int(topk)}
        if layer is not None:
            body["layer"] = int(layer)
        return self._post("/jlens", body)

    def unembed_row(self, token_id: int) -> dict:
        """POST /jlens/unembed_row: ONE row of the model's own (quantized) unembed/lm_head
        matrix, W_U[token_id] -- the ingredient clozn/behavior/steering/concept_dir.py's dir(c) =
        normalize(J_l^T @ W_U[c]) needs but has no other in-product source for (J_l ships in the
        product J-lens sidecar; W_U doesn't -- see concept_dir.py's BLOCKER_NOTE). Extracted
        server-side via ggml_get_rows (dequantizes whatever GGUF quant type the head is), so only
        d_model floats cross the wire, never the full [vocab, d_model] matrix. Requires the
        engine to have a J-lens sidecar loaded (same requirement as jlens()); 400s otherwise.
        Returns {token_id, piece, d_model, vector:[d_model floats]}."""
        return self._post("/jlens/unembed_row", {"token_id": int(token_id)})


# --------------------------------------------------------------------------- CLI / selftest

def _selftest() -> int:
    """Offline validation of the wire codec and value flattening — needs no server."""
    # A deterministic [5, 7] float32 matrix with fractional values (exercises f32, not ints).
    a = (np.arange(35, dtype="<f4").reshape(5, 7) * 0.5 - 3.25)
    wire = {"dtype": "float32", "shape": [5, 7],
            "data": base64.b64encode(a.tobytes()).decode("ascii")}
    b = decode_tensor(wire)
    assert b.shape == (5, 7), b.shape
    assert np.array_equal(a, b), "tensor codec round-trip is not exact"

    # write flattening is position-major: row i of the slice lands at offset i*n_embd.
    rows = a[[0, 2, 4]]
    flat = flatten_values(rows)
    assert len(flat) == 3 * 7, len(flat)
    assert abs(flat[7] - float(a[2, 0])) < 1e-6, "flatten is not row-major"
    assert abs(flat[0] - float(a[0, 0])) < 1e-6

    # a malformed tensor (byte count != shape product) must raise, not silently truncate.
    bad = {"dtype": "float32", "shape": [5, 8], "data": wire["data"]}
    try:
        decode_tensor(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("decode_tensor accepted a shape/byte mismatch")

    print("selftest OK: tensor codec exact, flatten row-major, shape guard fires")
    return 0


def _demo(args) -> int:
    """A live read -> edit -> write -> observe round-trip against a running clozn-server."""
    eng = EngineClient(host=args.host, port=args.port)
    info = eng.health()
    print(f"server: {info.get('model')}  mode={info.get('mode')}")

    h = eng.harvest(args.text, args.layer)
    print(f"harvested {h.n_tokens} tokens x {h.n_embd} dims at layer {h.layer}")
    print("tokens:", " ".join(repr(t) for t in h.tokens))

    pos = args.pos if args.pos >= 0 else h.n_tokens - 1   # default: the last token (drives next-token)
    pos = max(0, min(pos, h.n_tokens - 1))

    def amplify(acts: np.ndarray) -> np.ndarray:
        acts[pos] = acts[pos] * args.scale       # scale one position's residual
        return acts

    print(f"\nediting position {pos} ({h.tokens[pos]!r}) x{args.scale} at layer {h.layer}")
    _, obs = eng.edit_and_observe(args.text, transform=amplify, layer=args.layer, positions=[pos])
    print(obs.summary())

    # The identity-write control: writing the harvested rows back unchanged should barely move.
    _, ctrl = eng.edit_and_observe(args.text, layer=args.layer, positions=[pos])
    print(f"\ncontrol (identity write): moved_l2={ctrl.moved_l2:.3f} "
          f"(should be ~0 vs {obs.moved_l2:.3f} for the real edit)")
    return 0


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="clozn-server white-box Python client")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--selftest", action="store_true", help="validate the wire codec offline (no server)")
    ap.add_argument("--demo", action="store_true", help="run a live read->edit->write->observe round-trip")
    ap.add_argument("--text", default="The capital of France is", help="text to harvest for the demo")
    ap.add_argument("--layer", type=int, default=None, help="tap layer (default: server's read tap)")
    ap.add_argument("--pos", type=int, default=-1, help="position to edit (default: last token)")
    ap.add_argument("--scale", type=float, default=4.0, help="scale factor for the edited position")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if args.demo:
        return _demo(args)
    # Default with no flag: run the offline selftest (safe, needs nothing running).
    return _selftest()


if __name__ == "__main__":
    raise SystemExit(main())
