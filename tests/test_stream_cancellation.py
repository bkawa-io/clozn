"""test_stream_cancellation -- backlog #2 ("request isolation + cancellation"): sse.sse_chat's two distinct
mid-stream failure modes.

CLIENT DISCONNECT: the reader on the far end of `handler.wfile` is gone (a write raises BrokenPipeError/
ConnectionAbortedError/ConnectionResetError -- all OSError). sse.py must stop pulling from the worker
immediately (via an explicit `gen.close()`, not CPython refcounting eventually noticing), mark the
substrate's RequestContext cancelled, and log the partial reply as a distinct, honest "client_disconnected"
failure -- never as a normal "stop".

WORKER-DIES-MIDSTREAM: the failure comes from ITERATING chat_stream (reading the worker), not from writing
to the client, who is presumably still there. sse.py must emit an in-band `data: {"error": ...}` frame
(HTTP status is already committed to 200) followed by `data: [DONE]`, and log the run as a distinct,
honest "worker_disconnected" failure -- never a hang, never a silent empty 200.

Model-free throughout: a FakeStreamSub stands in for EngineSubstrate (its chat_stream is a real generator,
so gen.close()/GeneratorExit/finally all behave exactly as the real substrate's does), and a RecordingHandler
captures every _log_run call verbatim (by parameter name, mirroring app.py's real signature) plus the raw
bytes written to a BytesIO-backed (optionally failing) wfile.
"""
from __future__ import annotations

import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from clozn.server import app as cs   # noqa: E402
from clozn.server import sse          # noqa: E402


class _FakeRequestContext:
    """Just enough of request_context.RequestContext's surface for sse.py's disconnect branch:
    .cancel()/.is_cancelled()."""

    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    def is_cancelled(self):
        return self._cancelled


class FakeStreamSub:
    """A chat_stream-only double. `pieces` is the full reply; `fail_at` (an index into `pieces`) makes the
    generator raise `fail_exc` INSTEAD OF yielding that piece -- simulating the worker dying mid-stream.
    `.closed` records whether the generator's own `finally` ran (mirrors EngineSubstrate.chat_stream's
    `finally: resp.close()` -- proof the upstream connection was released on every exit path, not just the
    happy one)."""

    def __init__(self, pieces, fail_at=None, fail_exc=None):
        self.pieces = list(pieces)
        self.fail_at = fail_at
        self.fail_exc = fail_exc
        self._request = None
        self.closed = False

    def chat_stream(self, messages, max_new, mem_out=None):
        self._request = _FakeRequestContext()
        try:
            for i, piece in enumerate(self.pieces):
                if self._request.is_cancelled():
                    return
                if self.fail_at is not None and i == self.fail_at:
                    raise self.fail_exc
                yield piece
        finally:
            self.closed = True

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return []


class _FailingWfile(io.BytesIO):
    """A wfile stand-in that raises BrokenPipeError on write call number `fail_after + 1` (1-indexed) --
    simulating the client's TCP connection dying partway through delivery. Writes up to that point are
    kept (readable via .getvalue()), so a test can assert exactly what reached the client before the
    disconnect."""

    def __init__(self, fail_after):
        super().__init__()
        self.fail_after = fail_after
        self.calls = 0

    def write(self, b):
        self.calls += 1
        if self.calls > self.fail_after:
            raise BrokenPipeError("simulated client disconnect")
        return super().write(b)


class RecordingHandler:
    """A minimal BaseHTTPRequestHandler stand-in: enough of the surface sse.sse_chat needs
    (send_response/send_header/end_headers/wfile/headers/_log_run), with _log_run recording every call by
    PARAMETER NAME (mirrors app.py's real _log_run signature exactly) so assertions don't depend on
    positional-vs-keyword call style."""

    def __init__(self, wfile=None):
        self.code = None
        self.headers = {"Host": "localhost"}
        self.sent_headers = []
        self.wfile = wfile if wfile is not None else io.BytesIO()
        self.log_calls = []

    def send_response(self, code):
        self.code = code

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        pass

    def _log_run(self, source, messages, response, model, started, error=None, trace=None,
                mem_out=None, finish_reason=None, finish_reason_fallback=None, extra_meta=None,
                sections=None):
        self.log_calls.append(dict(source=source, messages=messages, response=response, model=model,
                                   started=started, error=error, trace=trace, mem_out=mem_out,
                                   finish_reason=finish_reason,
                                   finish_reason_fallback=finish_reason_fallback, extra_meta=extra_meta,
                                   sections=sections))
        return "run_test"


def _sse_frames(raw: bytes):
    """Parse the `data: {...}` JSON frames out of raw SSE bytes, in order (skips the literal [DONE])."""
    out = []
    for line in raw.decode("utf-8").splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            out.append(json.loads(line[len("data: "):]))
    return out


# ==================================================================================== CLIENT DISCONNECT

def test_client_disconnect_stops_the_generator_and_logs_it_distinctly(monkeypatch):
    sub = FakeStreamSub(["Hel", "lo", "!"])
    monkeypatch.setattr(cs, "SUB", sub)
    # write #1 = the opening {"role": "assistant"} chunk, write #2 = the "Hel" content chunk -- both
    # succeed; write #3 (the "lo" content chunk) raises, simulating the disconnect right after "Hel".
    handler = RecordingHandler(wfile=_FailingWfile(fail_after=2))

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")

    assert sub.closed is True                        # the upstream connection was released via gen.close()
    assert sub._request.is_cancelled() is True        # the context records the cancellation durably
    assert len(handler.log_calls) == 1
    call = handler.log_calls[0]
    assert call["error"].startswith("client disconnected mid-stream:")
    assert call["extra_meta"] == {"stream_failure": "client_disconnected"}
    assert call["finish_reason"] is None              # never "stop" -- the honesty invariant
    # only the two successful writes ever reached the "client" -- no [DONE], no error frame, no "!" piece
    wire = handler.wfile.getvalue().decode("utf-8")
    assert '"content": "Hel"' in wire
    assert '"content": "lo"' not in wire
    assert "[DONE]" not in wire
    assert '"error"' not in wire


def test_client_disconnect_on_the_very_first_write_never_starts_the_generator(monkeypatch):
    """The opening {"role": "assistant"} chunk itself can fail if the client was ALREADY gone before
    generation started -- chat_stream's body (which publishes self._request) never even runs in that case
    (generator functions defer their body to the first iteration). Must not crash, must log the disconnect."""
    sub = FakeStreamSub(["Hel", "lo"])
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler(wfile=_FailingWfile(fail_after=0))   # every write fails, including the first

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")

    assert sub._request is None                       # chat_stream's body never ran
    assert len(handler.log_calls) == 1
    assert handler.log_calls[0]["extra_meta"] == {"stream_failure": "client_disconnected"}
    assert handler.log_calls[0]["response"] == ""


# ==================================================================================== WORKER-DIES-MIDSTREAM

def test_worker_dying_midstream_emits_an_honest_error_frame_then_done(monkeypatch):
    sub = FakeStreamSub(["Hel", "lo", "!"], fail_at=1, fail_exc=ConnectionResetError("worker connection reset"))
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler()                       # a normal, non-failing wfile -- the CLIENT is fine

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")

    assert sub.closed is True                          # released even though the failure was mid-read
    assert sub._request.is_cancelled() is False         # NOT a client disconnect -- must not be conflated
    assert len(handler.log_calls) == 1
    call = handler.log_calls[0]
    assert "worker connection reset" in call["error"]
    assert call["extra_meta"] == {"stream_failure": "worker_disconnected"}
    assert call["finish_reason"] is None                # never "stop" -- a cut-short generation must not claim it

    wire = handler.wfile.getvalue()
    text = wire.decode("utf-8")
    assert '"content": "Hel"' in text                   # the piece generated before the failure still reached the client
    frames = _sse_frames(wire)
    assert any("error" in f and "worker connection reset" in f["error"] for f in frames)
    assert text.rstrip().endswith("data: [DONE]")        # a well-behaved SSE consumer knows to stop reading


def test_worker_dying_before_any_piece_is_still_reported_honestly(monkeypatch):
    # fail_at=0: the generator raises before yielding its first (and only) placeholder piece -- the
    # placeholder must never actually be reached, which the response=="" assertion below confirms.
    sub = FakeStreamSub(["never yielded"], fail_at=0, fail_exc=ConnectionRefusedError("connection refused"))
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler()

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")

    assert len(handler.log_calls) == 1
    assert handler.log_calls[0]["response"] == ""
    assert handler.log_calls[0]["extra_meta"] == {"stream_failure": "worker_disconnected"}
    frames = _sse_frames(handler.wfile.getvalue())
    assert any("connection refused" in f.get("error", "") for f in frames)


# ==================================================================================== the happy path is unaffected

def test_normal_completion_still_closes_the_generator_and_logs_once(monkeypatch):
    """gen.close() now runs UNCONDITIONALLY in `finally`, including after a clean [DONE] finish -- must be
    a no-op there (Python generators document close() on an exhausted generator as a no-op) and must not
    cause a second, spurious _log_run call."""
    sub = FakeStreamSub(["Hel", "lo", "!"])
    monkeypatch.setattr(cs, "SUB", sub)
    handler = RecordingHandler()

    sse.sse_chat(handler, [{"role": "user", "content": "hi"}], 8, "m")

    assert sub.closed is True
    assert sub._request.is_cancelled() is False
    assert len(handler.log_calls) == 1
    call = handler.log_calls[0]
    assert call["response"] == "Hello!"
    assert call["error"] is None
    assert call["extra_meta"] is None                   # the happy path never sets a stream_failure tag
    wire = handler.wfile.getvalue().decode("utf-8")
    assert wire.rstrip().endswith("data: [DONE]")
