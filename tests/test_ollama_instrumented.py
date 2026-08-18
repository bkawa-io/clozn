"""End-to-end contracts for the Ollama-shaped entrance to Clozn's runtime.

The important compatibility invariant is not merely that the response looks like
Ollama.  /api/chat and /api/generate must cross the active substrate's chat seam so
the prompt manifest, per-token trace capture, finish reasons, and the run journal all
observe the same execution.  These tests drive the real handler without a model or
socket and resolve the returned X-Clozn-Run-Id against an isolated journal.
"""
from __future__ import annotations

import io
import json

import pytest

import clozn.settings as clozn_settings

import clozn.runs.store as runlog
from clozn.server import app as cs


class InstrumentedSub:
    name = "engine"
    brain = None

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = []
        self._meta = {"model_id": "fake-qwen", "sampler_mode": "sample",
                      "sampling": "sample", "temperature": 0.8}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls.append({"messages": [dict(m) for m in messages],
                           "max_new": max_new, "sample": sample})
        self._meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            block = "You are a helpful assistant.\n- Prefer grounded answers."
            mem_out.update(
                mode="prompt",
                applied=[{"id": None, "text": "Prefer grounded answers.", "relevance": 0.88}],
                gate=0.73,
                strength=0.75,
                prompt_block=block,
                assembled_messages=[{"role": "system", "content": block}]
                + [dict(m) for m in messages],
                final_prompt="<rendered>grounded prompt</rendered>",
            )
        if self.fail:
            raise RuntimeError("synthetic decode failure")
        if trace_out is not None:
            trace_out.extend([
                {"pos": 0, "token_id": 41, "piece": "Observed", "prob": 0.93,
                 "alts": [{"token_id": 42, "piece": "Maybe", "prob": 0.04}]},
                {"pos": 1, "token_id": 43, "piece": " reply.", "prob": 0.47,
                 "alts": [{"token_id": 44, "piece": " answer.", "prob": 0.39}]},
            ])
        return "Observed reply."

    def chat_stream(self, messages, max_new=256, mem_out=None, sample=True):
        """Streaming twin of chat() above -- a real generator, so gen.close()/GeneratorExit semantics
        match the product substrate (clozn.server.ndjson relies on that, same as clozn.server.sse
        already does for /v1/chat/completions). Kept minimal here: this file's job is the instrumented-
        substrate contract for both the streaming and non-streaming shapes; the NDJSON wire details,
        field-rejection policy, and disconnect/cancellation coverage live in test_ollama_streaming.py."""
        self.calls.append({"messages": [dict(m) for m in messages],
                           "max_new": max_new, "sample": sample})
        self._meta.update(max_tokens=int(max_new), stream=True)
        if self.fail:
            raise RuntimeError("synthetic decode failure")
        for piece in ("Observed", " reply."):
            yield piece

    def last_finish_reason(self):
        return "stop"

    def last_stream_trace(self):
        return [{"pos": 0, "token_id": 41, "piece": "Observed", "prob": 0.93},
                {"pos": 1, "token_id": 43, "piece": " reply.", "prob": 0.47}]

    def run_meta(self):
        return dict(self._meta)


def _dispatch(method: str, path: str, body=None):
    raw = json.dumps(body if body is not None else {}).encode("utf-8")
    handler_type = cs.make_handler()
    handler = object.__new__(handler_type)
    handler.path = path
    handler.rfile = io.BytesIO(raw)
    handler.wfile = io.BytesIO()
    handler.headers = {"Content-Length": str(len(raw)), "User-Agent": "ollama-python/0.test"}
    handler.requestline = f"{method} {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.command = method
    handler.close_connection = False
    getattr(handler, f"do_{method}")()
    return handler.wfile.getvalue()


def _payload(raw: bytes) -> dict:
    return json.loads(raw.partition(b"\r\n\r\n")[2].decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    sub = InstrumentedSub()
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return sub


def test_ollama_routes_are_registered_on_the_real_handler(iso):
    out = _payload(_dispatch("GET", "/api/version"))
    assert out == {"version": "0.0.0-clozn"}




def test_ollama_generate_enters_the_same_path_as_a_user_turn(iso):
    out = _payload(_dispatch("POST", "/api/generate", {
        "model": "qwen3.5:9b",
        "system": "Answer tersely.",
        "prompt": "Why is the sky blue?",
        "stream": False,
        "options": {"num_predict": 9, "temperature": 0},
    }))
    rid = out["clozn_run_id"]
    assert out["response"] == "Observed reply."
    assert iso.calls[0] == {
        "messages": [{"role": "system", "content": "Answer tersely."},
                     {"role": "user", "content": "Why is the sky blue?"}],
        "max_new": 9,
        "sample": {"temperature": 0},
    }
    logged = runlog.get_run(rid)
    assert logged["prompt_summary"] == "Why is the sky blue?"
    assert logged["meta"]["ollama_operation"] == "generate"
    assert logged["trace"]["token_ids"] == [41, 43]


def test_ollama_separates_thinking_from_content_and_never_echoes_it_into_history(iso):
    def thinking_chat(messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        iso.calls.append({"messages": [dict(m) for m in messages], "max_new": max_new, "sample": sample})
        if mem_out is not None:
            mem_out.update(mode="prompt", applied=[], final_prompt="assistant\n<think>\n")
        if trace_out is not None:
            trace_out.extend([
                {"pos": 0, "piece": "check", "prob": .6},
                {"pos": 1, "piece": "</think>", "prob": .7},
                {"pos": 2, "piece": "answer", "prob": .9},
            ])
        return "check</think>answer"

    iso.chat = thinking_chat
    out = _payload(_dispatch("POST", "/api/chat", {
        "messages": [
            {"role": "assistant", "content": "prior", "thinking": "do not echo"},
            {"role": "user", "content": "next"},
        ],
        "stream": False,
    }))
    assert out["message"] == {"role": "assistant", "content": "answer", "thinking": "check"}
    assert iso.calls[0]["messages"][0] == {"role": "assistant", "content": "prior"}
    rec = runlog.get_run(out["clozn_run_id"])
    assert rec["response"] == "answer"
    assert rec["reasoning"]["blocks"] == [{"text": "check", "closed": True}]
    assert "".join(rec["trace"]["tokens"]) == "answer"


def test_ollama_nonstream_cutoff_has_warning_body_and_header(iso):
    iso.last_finish_reason = lambda: "length"
    raw = _dispatch("POST", "/api/chat", {
        "messages": [{"role": "user", "content": "hello"}],
        "stream": False, "options": {"num_predict": 2},
    })
    out = _payload(raw)
    assert out["done_reason"] == "length"
    assert out["clozn_warnings"][0]["code"] == "output_truncated"
    assert b"X-Clozn-Warning: output-truncated" in raw.partition(b"\r\n\r\n")[0]
    logged = runlog.get_run(out["clozn_run_id"])
    assert logged["context_receipt"]["output_cut_off"] is True


def test_ollama_generation_failure_is_still_an_inspectable_run(iso):
    iso.fail = True
    out = _payload(_dispatch("POST", "/api/chat", {
        "model": "qwen3.5:9b", "messages": [{"role": "user", "content": "break"}],
        "stream": False,
    }))
    assert out == {"error": "engine: synthetic decode failure"}
    rows = runlog.list_runs(1)
    assert len(rows) == 1
    logged = runlog.get_run(rows[0]["id"])
    assert logged["source"] == "ollama_api"
    assert logged["error"] == "synthetic decode failure"
    assert logged["meta"]["compatibility_api"] == "ollama"


def test_ollama_chat_streams_by_default_when_stream_is_omitted(iso):
    """DEFAULT-STREAM SEMANTICS (roadmap PRODUCT_ROADMAP.md Phase 2 item 1): upstream Ollama streams
    unless the caller explicitly opts out with `stream: false` -- https://docs.ollama.com/api/streaming.
    Before this shipped, an omitted `stream` took clozn's own non-stream path (and an explicit `stream:
    true` was a clean 501); omitting it now must match upstream and stream instead. Wire-shape depth
    (chunk fields, done_reason, disconnect handling, ...) lives in test_ollama_streaming.py -- this is
    just the smoke check that the DEFAULT actually flipped."""
    raw = _dispatch("POST", "/api/chat", {
        "model": "qwen3.5:9b", "messages": [{"role": "user", "content": "hello"}],
    })
    header, _, body = raw.partition(b"\r\n\r\n")
    assert b"application/x-ndjson" in header
    lines = [json.loads(chunk) for chunk in body.decode("utf-8").splitlines() if chunk.strip()]
    assert lines[-1]["done"] is True
    assert "".join(c["message"]["content"] for c in lines if not c["done"]) == "Observed reply."
    assert iso.calls == [{"messages": [{"role": "user", "content": "hello"}],
                         "max_new": 256, "sample": True}]
    logged = runlog.get_run(runlog.list_runs(1)[0]["id"])
    assert logged["response"] == "Observed reply."
    assert logged["meta"]["ollama_operation"] == "chat"
