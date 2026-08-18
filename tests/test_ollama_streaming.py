"""tests/test_ollama_streaming.py -- roadmap PRODUCT_ROADMAP.md Phase 2 item 1 ("Ollama NDJSON
streaming") and item 2 ("explicit-or-rejected shim fields"), both landed together in
clozn/server/routes/ollama.py + the new clozn/server/ndjson.py.

Model-free throughout, mirroring tests/test_stream_cancellation.py's approach for the SSE stream:
StreamingSub's chat_stream() is a REAL generator (so gen.close()/GeneratorExit/finally behave exactly
like EngineSubstrate's), and requests are driven through the REAL registered handler
(clozn.server.app.make_handler(), the same `_dispatch` pattern test_ollama_instrumented.py already uses)
so this exercises the actual route dispatch, not a hand-rolled stand-in.
"""
from __future__ import annotations

import io
import json

import pytest

import clozn.settings as clozn_settings

import clozn.runs.store as runlog
from clozn.server import app as cs


# ==================================================================================== fakes


class _FakeRequestContext:
    """Just enough of request_context.RequestContext's surface for clozn.server.ndjson's disconnect
    branch: .cancel()/.is_cancelled() (mirrors test_stream_cancellation.py's own stand-in)."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


class StreamingSub:
    """Fake substrate exercising both chat() (the non-stream path, for `stream: false` coverage) and
    chat_stream() (a REAL generator). `pieces` is the full reply; `fail_at` (an index into `pieces`)
    makes chat_stream raise `fail_exc` INSTEAD OF yielding that piece, simulating the worker dying
    mid-stream. `.closed` records whether the generator's own `finally` ran."""

    name = "engine"
    brain = None

    def __init__(self, pieces=("Hel", "lo", "!"), finish="stop", prompt_tokens=5,
                fail_at=None, fail_exc=None):
        self.pieces = list(pieces)
        self.finish = finish
        self.prompt_tokens = prompt_tokens
        self.fail_at = fail_at
        self.fail_exc = fail_exc
        self._request = None
        self.closed = False
        self.calls = []
        self.emitted: list[str] = []

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "max_new": max_new, "sample": sample})
        text = "".join(self.pieces)
        if trace_out is not None:
            trace_out.extend([{"pos": i, "token_id": 100 + i, "piece": p, "prob": 0.5}
                              for i, p in enumerate(self.pieces)])
        return text

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        self._request = _FakeRequestContext()
        self.calls.append({"messages": [dict(m) for m in messages],
                           "max_new": max_new, "sample": sample})
        self.emitted = []
        try:
            for i, piece in enumerate(self.pieces):
                if self._request.is_cancelled():
                    return
                if self.fail_at is not None and i == self.fail_at:
                    raise self.fail_exc
                self.emitted.append(piece)
                yield piece
        finally:
            self.closed = True

    def last_finish_reason(self):
        return self.finish

    def last_stream_trace(self):
        return [{"pos": i, "piece": p} for i, p in enumerate(self.emitted)]

    def last_prompt_tokens(self):
        return self.prompt_tokens

    def run_meta(self):
        return {"model_id": "fake-qwen"}


class _FailingWfile(io.BytesIO):
    """A wfile stand-in that raises BrokenPipeError starting on the Nth NDJSON *line* write (1-indexed)
    -- counts only writes that look like a JSON object (`{...}`), so the header bytes the real
    BaseHTTPRequestHandler writes first (status line + headers, via send_response/send_header/
    end_headers) are never mistaken for a content write."""

    def __init__(self, fail_after):
        super().__init__()
        self.fail_after = fail_after
        self.line_writes = 0

    def write(self, b):
        if b.startswith(b"{"):
            self.line_writes += 1
            if self.line_writes > self.fail_after:
                raise BrokenPipeError("simulated client disconnect")
        return super().write(b)


# ==================================================================================== dispatch helpers


def _dispatch(method: str, path: str, body=None, wfile=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = wfile if wfile is not None else io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "ollama-python/0.test"}
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    return handler.wfile.getvalue()


def _split(raw: bytes):
    header, _, body = raw.partition(b"\r\n\r\n")
    return header, body


def _status(raw: bytes) -> int:
    return int(raw.split(b"\r\n", 1)[0].split(b" ")[1])


def _payload(raw: bytes) -> dict:
    return json.loads(raw.partition(b"\r\n\r\n")[2].decode("utf-8"))


def _ndjson_lines(raw: bytes) -> list:
    _, body = _split(raw)
    return [json.loads(chunk) for chunk in body.decode("utf-8").splitlines() if chunk.strip()]


@pytest.fixture
def isolated_runlog(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return monkeypatch


def _activate(monkeypatch, sub):
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return sub


# ==================================================================================== chunk shapes


def test_chat_ndjson_chunk_shape_and_final_line(isolated_runlog):
    sub = _activate(isolated_runlog, StreamingSub(pieces=("Hel", "lo", "!"), finish="stop",
                                                  prompt_tokens=5))
    raw = _dispatch("POST", "/api/chat", {
        "model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hi"}],
    })
    header, _ = _split(raw)
    assert b"Content-Type: application/x-ndjson" in header
    lines = _ndjson_lines(raw)

    content_lines = lines[:-1]
    assert [l["message"]["content"] for l in content_lines] == ["Hel", "lo", "!"]
    for l in content_lines:
        assert l["model"] == "qwen3.5:9b"
        assert l["message"]["role"] == "assistant"
        assert l["done"] is False
        assert isinstance(l["created_at"], str) and l["created_at"].endswith("Z")

    final = lines[-1]
    assert final["done"] is True
    assert final["done_reason"] == "stop"
    assert final["message"] == {"role": "assistant", "content": ""}
    assert isinstance(final["total_duration"], int) and final["total_duration"] >= 0
    assert final["eval_count"] == 3                 # len(last_stream_trace()) -- 3 committed pieces
    assert final["prompt_eval_count"] == 5           # sub.last_prompt_tokens()
    assert runlog.get_run(final["clozn_run_id"])["response"] == "Hello!"
    # never fabricated -- see clozn/server/ndjson.py's module docstring
    assert "load_duration" not in final
    assert "prompt_eval_duration" not in final
    assert "eval_duration" not in final

    assert sub.closed is True
    rows = runlog.list_runs(5)
    assert len(rows) == 1                            # exactly one coherent run record
    logged = runlog.get_run(rows[0]["id"])
    assert logged["response"] == "Hello!"
    assert logged["source"] == "ollama_api"
    assert logged["finish_reason"] == "stop"
    assert logged["meta"]["compatibility_api"] == "ollama"
    assert logged["meta"]["ollama_operation"] == "chat"


def test_generate_ndjson_chunk_shape(isolated_runlog):
    _activate(isolated_runlog, StreamingSub(pieces=("The ", "sky ", "is blue."), finish="length"))
    raw = _dispatch("POST", "/api/generate", {
        "model": "qwen3.5:9b", "prompt": "Why is the sky blue?",
    })
    lines = _ndjson_lines(raw)
    content_lines = lines[:-1]
    assert [l["response"] for l in content_lines] == ["The ", "sky ", "is blue."]
    for l in content_lines:
        assert "message" not in l
        assert l["done"] is False
    final = lines[-1]
    assert final["response"] == ""
    assert final["done"] is True
    assert final["done_reason"] == "length"
    assert final["clozn_warnings"][0]["code"] == "output_truncated"

    rows = runlog.list_runs(5)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert logged["response"] == "The sky is blue."
    assert logged["meta"]["ollama_operation"] == "generate"


def test_ollama_stream_separates_split_thinking_chunks_from_public_content(isolated_runlog):
    _activate(isolated_runlog, StreamingSub(
        pieces=("<thi", "nk>secret</thi", "nk>an", "swer"), finish="stop"
    ))
    lines = _ndjson_lines(_dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}],
    }))
    payload = lines[:-1]
    thinking = "".join(line["message"].get("thinking", "") for line in payload)
    content = "".join(line["message"]["content"] for line in payload)
    assert thinking == "secret"
    assert content == "answer"
    assert all("<think>" not in json.dumps(line).lower() for line in lines)
    assert lines[-1]["eval_count"] == 4  # raw generation accounting is retained
    rec = runlog.get_run(runlog.list_runs(1)[0]["id"])
    assert rec["response"] == "answer"
    assert rec["reasoning"]["blocks"] == [{"text": "secret", "closed": True}]
    assert "".join(rec["trace"]["tokens"]) == "answer"


# ==================================================================================== default-stream flip


def test_stream_false_takes_the_non_stream_json_path(isolated_runlog):
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False,
    })
    header, _ = _split(raw)
    assert b"Content-Type: application/json" in header
    assert b"application/x-ndjson" not in header
    out = _payload(raw)
    assert out["message"]["content"] == "Hello!"
    assert out["done"] is True
    assert runlog.list_runs(5)[0] is not None


def test_stream_omitted_defaults_to_streaming(isolated_runlog):
    """Upstream Ollama streams unless the caller opts out with `stream: false`
    (https://docs.ollama.com/api/streaming) -- an omitted key must take the SAME path as `stream: true`."""
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/generate", {"prompt": "hi"})
    header, _ = _split(raw)
    assert b"Content-Type: application/x-ndjson" in header


# ==================================================================================== disconnect / cancellation


def test_disconnect_mid_stream_finalizes_exactly_one_run_with_the_partial_reply(isolated_runlog):
    sub = _activate(isolated_runlog, StreamingSub(pieces=("Hel", "lo", "!")))
    # fail_after=1: the first NDJSON line ("Hel") succeeds, the second ("lo") raises. Mirrors
    # clozn.server.sse's own accumulate-then-write order: the piece whose WRITE failed is still
    # appended to `acc` before the failed attempt (so the logged response is "Hello", not "Hel") --
    # same pre-existing contract test_stream_cancellation.py exercises for the SSE stream.
    raw = _dispatch("POST", "/api/chat", {"messages": [{"role": "user", "content": "hi"}]},
                    wfile=_FailingWfile(fail_after=1))
    lines = _ndjson_lines(raw)
    assert [l["message"]["content"] for l in lines] == ["Hel"]     # "lo"/"!"/final never reached the client

    assert sub.closed is True                         # the generator's own finally ran (gen.close())
    assert sub._request.is_cancelled() is True         # durable cancellation record

    rows = runlog.list_runs(5)
    assert len(rows) == 1                              # exactly one run -- never zero, never two
    logged = runlog.get_run(rows[0]["id"])
    assert logged["response"] == "Hello"
    assert logged["finish_reason"] is None              # never "stop" -- it did not finish
    assert "client disconnected mid-stream" in logged["error"]
    assert logged["meta"]["stream_failure"] == "client_disconnected"
    assert logged["meta"]["compatibility_api"] == "ollama"


def test_worker_failure_mid_stream_emits_one_error_line_and_finalizes_one_run(isolated_runlog):
    sub = _activate(isolated_runlog, StreamingSub(
        pieces=("Hel", "lo", "!"), fail_at=1, fail_exc=ConnectionResetError("worker connection reset")))
    raw = _dispatch("POST", "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    lines = _ndjson_lines(raw)
    assert lines[0]["message"]["content"] == "Hel"
    assert lines[-1]["done"] is True
    assert "worker connection reset" in lines[-1].get("error", "")
    assert runlog.get_run(lines[-1]["clozn_run_id"])["error"]
    assert sub.closed is True
    assert sub._request.is_cancelled() is False         # NOT a client disconnect -- must not be conflated

    rows = runlog.list_runs(5)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert logged["finish_reason"] is None
    assert logged["meta"]["stream_failure"] == "worker_disconnected"


# ==================================================================================== done_reason mapping


@pytest.mark.parametrize("finish,expected", [("stop", "stop"), ("length", "length")])
def test_done_reason_maps_clozn_finish_reasons(isolated_runlog, finish, expected):
    _activate(isolated_runlog, StreamingSub(pieces=("hi",), finish=finish))
    raw = _dispatch("POST", "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert _ndjson_lines(raw)[-1]["done_reason"] == expected


def test_done_reason_is_omitted_when_finish_reason_is_unmapped_or_missing(isolated_runlog):
    _activate(isolated_runlog, StreamingSub(pieces=("hi",), finish=None))
    raw = _dispatch("POST", "/api/chat", {"messages": [{"role": "user", "content": "hi"}]})
    assert "done_reason" not in _ndjson_lines(raw)[-1]


# ==================================================================================== Gate-0 field rejection


@pytest.mark.parametrize("field,value", [
    ("raw", True),
    ("template", "custom {{ .Prompt }}"),
    ("format", "json"),
    ("suffix", "def tail(): pass"),
    ("context", [1, 2, 3]),
    ("think", True),
    ("images", ["aGVsbG8="]),
])
def test_behavior_bearing_fields_are_rejected_with_a_named_400(isolated_runlog, field, value):
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False, field: value,
    })
    assert _status(raw) == 400
    out = _payload(raw)
    assert field in out["error"]
    assert runlog.list_runs(5) == []                    # rejected before any generation is attempted


@pytest.mark.parametrize("field,neutral_value", [
    ("raw", False), ("suffix", ""), ("context", []), ("think", False), ("images", []),
])
def test_neutral_field_values_are_accepted_not_rejected(isolated_runlog, field, neutral_value):
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False, field: neutral_value,
    })
    assert _status(raw) == 200


def test_keep_alive_is_accepted_and_silently_ignored(isolated_runlog):
    """keep_alive has nothing to opt into on this runtime (one always-resident engine process, no
    unload/reload lifecycle) -- unlike raw/template/etc., it is not a Gate-0 violation to accept it."""
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False, "keep_alive": "10m",
    })
    assert _status(raw) == 200


# ==================================================================================== options{} policy


def test_unknown_option_key_is_rejected_by_name(isolated_runlog):
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False,
        "options": {"num_ctx": 4096},
    })
    assert _status(raw) == 400
    assert "options.num_ctx" in _payload(raw)["error"]


def test_stop_option_is_rejected_the_gateway_has_no_stop_sequence_support(isolated_runlog):
    _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False,
        "options": {"stop": ["\n\n"]},
    })
    assert _status(raw) == 400
    assert "options.stop" in _payload(raw)["error"]


def test_mapped_options_are_forwarded_correctly(isolated_runlog):
    sub = _activate(isolated_runlog, StreamingSub())
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hi"}], "stream": False,
        "options": {"temperature": 0.3, "top_p": 0.8, "top_k": 24, "repeat_penalty": 1.15,
                    "seed": 17, "num_predict": 9},
    })
    assert _status(raw) == 200
    assert sub.calls == [{
        "messages": [{"role": "user", "content": "hi"}],
        "max_new": 9,
        "sample": {"temperature": 0.3, "top_p": 0.8, "top_k": 24, "repeat_penalty": 1.15, "seed": 17},
    }]


def test_num_predict_maps_to_max_tokens_on_the_generate_route_too(isolated_runlog):
    sub = _activate(isolated_runlog, StreamingSub())
    _dispatch("POST", "/api/generate", {
        "prompt": "hi", "stream": False, "options": {"num_predict": 42},
    })
    assert sub.calls[0]["max_new"] == 42


# ==================================================================================== /api/tags digest honesty


class _FakeEngine:
    def __init__(self, health):
        self._health = health

    def health(self):
        return self._health


def test_tags_uses_real_model_sha256_when_the_engine_reports_one(monkeypatch):
    monkeypatch.setattr(cs, "ENGINE", _FakeEngine(
        {"model": "some/model.Q4_K_M.gguf", "model_sha256": "abc123"}))
    monkeypatch.setattr(cs, "SUB", object())            # no .model_sha256 attr -- falls back to health()
    out = _payload(_dispatch("GET", "/api/tags"))
    assert out["models"][0]["digest"] == "sha256:abc123"


def test_tags_prefers_the_substrates_own_resolved_model_sha256(monkeypatch):
    class _Sub:
        model_sha256 = "cached-digest"

    monkeypatch.setattr(cs, "ENGINE", _FakeEngine({"model": "some/model.gguf"}))   # no model_sha256 here
    monkeypatch.setattr(cs, "SUB", _Sub())
    out = _payload(_dispatch("GET", "/api/tags"))
    assert out["models"][0]["digest"] == "sha256:cached-digest"


def test_tags_omits_digest_entirely_when_unavailable(monkeypatch):
    monkeypatch.setattr(cs, "ENGINE", _FakeEngine({"model": "some/model.gguf"}))
    monkeypatch.setattr(cs, "SUB", object())
    out = _payload(_dispatch("GET", "/api/tags"))
    assert "digest" not in out["models"][0]
