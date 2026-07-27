"""test_bridge_server -- the M5 any-client run_id bridge (EXPLAIN_THIS_ANSWER_SPEC.md): a user chatting
through ANY OpenAI-compatible client gets the clozn run_id back with their reply, so a companion
`clozn inspect <run_id>` can inspect that exact reply -- without that client needing to know clozn exists.

Unlike M2-M4 (receipts.py / counterfactual.py / narrate.py, each unit-tested standalone with the server test
only proving thin wiring on top), M5 has no separate module: the whole feature IS the wiring -- `_log_run`
returning the run id it already computes, and `_json`/`_send` gaining an optional `extra_headers` param --
so this file is where that logic actually gets exercised end to end.

No model, no GPU: drives the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick used
an otherwise-untouched OpenAI chat.completion body that ALSO carries "clozn_run_id"; that id resolves via
runlog.get_run() to the exact run just logged; the raw HTTP response carries an X-Clozn-Run-Id header with
the same value; a logging failure (runlog.record -> None) omits both cleanly rather than emitting a literal
"null"/"None"; the pre-existing 503-no-substrate path and _json's other (no-extra-header) call sites are
untouched; and streaming carries the finalized run id on its ordinary terminal chunk before `[DONE]`.
"""
from __future__ import annotations

import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs   # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402


import clozn.runs.store as runlog                # noqa: E402


# --- a fake, qwen-shaped substrate: /v1/chat/completions' non-stream path calls chat(messages, max_new,
# sample, trace_out=, mem_out=); its stream path (exercised only by the one streaming-sanity test below)
# calls chat_stream(messages, max_new, mem_out=). ---------------------------------------------------------

class FakeSteer:
    def __init__(self, strength=None):
        self.strength = dict(strength or {})

    def active(self):
        return {k: v for k, v in self.strength.items() if v}


class FakeMem:
    def __init__(self, strength=1.0):
        self.memory_strength = float(strength)
        self.rules = []
        self.prefix = None


class FakeSub:
    name = "qwen"

    def __init__(self, finish_reason="stop"):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self.calls = 0
        self.finish_reason = finish_reason
        self._run_meta = {"model_id": "fake-qwen", "sampler_mode": "greedy",
                          "sampling": "greedy", "temperature": 0.0}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self.calls += 1
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        return "A plain reply."

    def chat_stream(self, messages, max_new=256, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=True)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        for piece in ("Hel", "lo"):
            yield piece

    def last_finish_reason(self):
        return self.finish_reason

    def run_meta(self):
        return dict(self._run_meta)


class NoFinishSub:
    name = "qwen"

    def __init__(self):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self._run_meta = {"model_id": "legacy-fake-qwen", "sampler_mode": "sample",
                          "sampling": "sample", "temperature": 0.7}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        return "A plain reply."

    def chat_stream(self, messages, max_new=256, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=True)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        for piece in ("Hel", "lo"):
            yield piece

    def run_meta(self):
        return dict(self._run_meta)


class PromptCaptureSub(FakeSub):
    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False, prompt_tokens=13)
        block = "You are a helpful assistant talking with a returning user.\n- Keep it brief."
        assembled = [{"role": "system", "content": block}] + [dict(m) for m in messages]
        if mem_out is not None:
            mem_out.update(mode="prompt",
                           applied=[{"id": None, "text": "Keep it brief.", "relevance": 0.82}],
                           candidate_cards=[{"id": None, "text": "Keep it brief."}],
                           omitted_cards=[], selection_stage="active_prompt_cards_considered_by_turn_gate",
                           baseline_prompt_tokens=7, gate=0.91, prompt_block=block,
                           assembled_messages=assembled)
        return "Brief reply."


class InternalizedSub(FakeSub):
    def __init__(self):
        super().__init__()
        self.memory.prefix = "PREFIX"
        self.memory.rules = ["Keep it brief."]

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        return "Prefix-shaped reply."


# --- driving the real handler without a socket (mirrors test_counterfactual_server / test_narrate_server) ---

def _dispatch(method, path, body_obj=None, headers=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest", **(headers or {})}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    return h.wfile.getvalue()


def _post_raw(path, body_obj=None):
    """The full raw bytes off the (fake) wire -- header block AND body -- for header assertions."""
    return _dispatch("POST", path, body_obj)


def _post(path, body_obj=None):
    """Just the parsed JSON body (matches the other server tests' convention)."""
    _, _, payload = _post_raw(path, body_obj).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _get(path, headers=None):
    _, _, payload = _dispatch("GET", path, headers=headers).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run/card/settings stores; SUB starts as a FakeSub (tests that want the 503 path
    override it to None explicitly)."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(cs, "SUB", FakeSub())
    return tmp_path


# ==================================================================================== /v1/chat/completions

def test_chat_completions_needs_the_substrate_503_unchanged(iso, monkeypatch):
    """The pre-existing 503 path (a plain 2-arg self._json call) must be byte-for-byte unchanged: no bridge
    field in the body, no X-Clozn-Run-Id header -- proving _json's extra_headers param is opt-in only."""
    monkeypatch.setattr(cs, "SUB", None)
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    header_block, _, payload = raw.partition(b"\r\n\r\n")
    assert b"X-Clozn-Run-Id" not in header_block
    assert json.loads(payload.decode("utf-8")) == {"error": {"message": "model worker unavailable", "type": "service_unavailable"}}


def test_chat_completions_carries_clozn_run_id_that_resolves_to_the_logged_run(iso):
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi there"}]})
    # the OpenAI shape is untouched except for the one additive field
    assert set(out.keys()) == {"id", "object", "created", "model", "choices", "clozn_run_id"}
    assert "usage" not in out                     # unknown counts are omitted, never fabricated as zeros
    assert out["object"] == "chat.completion"
    assert out["choices"][0]["finish_reason"] == "stop"
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "A plain reply."}
    # the bridge: a real id that resolves, via runlog, to the exact run this reply just produced
    rid = out["clozn_run_id"]
    assert isinstance(rid, str) and rid
    logged = runlog.get_run(rid)
    assert logged is not None
    assert logged["source"] == "openai_api"
    assert logged["finish_reason"] == "stop"
    assert logged["meta"]["finish_reason_source"] == "substrate"
    assert logged["meta"]["sampler_mode"] == "greedy"
    assert logged["meta"]["temperature"] == 0.0
    assert logged["meta"]["max_tokens"] == 256
    assert logged["response"] == "A plain reply."
    assert logged["messages"] == [{"role": "user", "content": "hi there"}]






def test_chat_completions_uses_real_length_finish_reason_end_to_end(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(finish_reason="length"))
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi"}],
                                         "max_tokens": 3})
    assert out["choices"][0]["finish_reason"] == "length"
    assert out["clozn_warnings"][0]["code"] == "output_truncated"
    logged = runlog.get_run(out["clozn_run_id"])
    assert logged["finish_reason"] == "length"
    assert "truncated" in logged["flags"]
    assert logged["meta"]["finish_reason_source"] == "substrate"
    assert logged["meta"]["max_tokens"] == 3


def test_chat_completions_length_sets_warning_header(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(finish_reason="length"))
    raw = _post_raw("/v1/chat/completions", {
        "messages": [{"role": "user", "content": "hi"}], "max_tokens": 3,
    })
    assert b"X-Clozn-Warning: output-truncated" in raw.partition(b"\r\n\r\n")[0]


def test_chat_completions_fallback_finish_reason_is_explicit_not_persisted_as_real(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", NoFinishSub())
    out = _post("/v1/chat/completions", {"model": "clozn-qwen",
                                         "messages": [{"role": "user", "content": "hi"}],
                                         "max_tokens": 4})
    assert out["choices"][0]["finish_reason"] == "stop"
    logged = runlog.get_run(out["clozn_run_id"])
    assert logged["finish_reason"] is None
    assert logged["meta"]["finish_reason_source"] == "fallback"
    assert logged["meta"]["finish_reason_fallback"] == "stop"
    assert logged["meta"]["max_tokens"] == 4


def test_chat_completions_raw_http_response_carries_the_x_clozn_run_id_header(iso):
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi there"}]})
    header_block, _, payload = raw.partition(b"\r\n\r\n")
    rid = json.loads(payload.decode("utf-8"))["clozn_run_id"]
    assert f"X-Clozn-Run-Id: {rid}".encode("utf-8") in header_block


def test_opt_in_association_headers_resolve_exact_latest_without_storing_raw_ids(iso):
    raw_client = "studio-install-123"
    raw_session = "studio-tab-456"
    raw_project = "workspace-clozn"
    raw = _dispatch("POST", "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "associated"}]},
                    headers={"X-Clozn-Client-Id": raw_client,
                             "X-Clozn-Session-Id": raw_session,
                             "X-Clozn-Project-Id": raw_project})
    _, _, payload = raw.partition(b"\r\n\r\n")
    rid = json.loads(payload)["clozn_run_id"]
    rec = runlog.get_run(rid)
    assert rec["client_key"].startswith("client_")
    assert rec["client_key_source"] == "header"
    assert rec["session_key"].startswith("session_")
    assert rec["project_key"].startswith("project_")
    assert raw_client not in json.dumps(rec)
    assert raw_session not in json.dumps(rec)
    assert raw_project not in json.dumps(rec)

    latest = _get("/runs/latest", headers={"X-Clozn-Client-Id": raw_client,
                                            "X-Clozn-Session-Id": raw_session})
    assert latest["available"] is True
    assert latest["association"] == {"exact": True, "ambiguous": False, "selector": "session"}
    assert latest["run"]["id"] == rid


def test_latest_requires_an_explicit_association_selector(iso):
    out = _get("/runs/latest")
    assert out["error"]["code"] == "association_selector_required"


def test_runs_watch_endpoint_pages_new_matches_from_an_opaque_cursor(iso):
    headers = {"X-Clozn-Client-Id": "watch-client", "X-Clozn-Session-Id": "watch-session"}
    first = json.loads(_dispatch(
        "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "one"}]},
        headers=headers,
    ).partition(b"\r\n\r\n")[2])["clozn_run_id"]
    cursor = runlog.cursor_for_run(first)
    second = json.loads(_dispatch(
        "POST", "/v1/chat/completions", {"messages": [{"role": "user", "content": "two"}]},
        headers=headers,
    ).partition(b"\r\n\r\n")[2])["clozn_run_id"]
    page = _get("/runs/watch?after=" + cursor, headers=headers)
    assert [run["id"] for run in page["runs"]] == [second]
    assert page["next_cursor"] != cursor


def test_invalid_association_header_is_typed_400_and_records_nothing(iso):
    raw = _dispatch("POST", "/v1/chat/completions",
                    {"messages": [{"role": "user", "content": "no"}]},
                    headers={"X-Clozn-Session-Id": "contains a space"})
    header, _, payload = raw.partition(b"\r\n\r\n")
    assert b" 400 " in header
    assert json.loads(payload)["error"]["code"] == "invalid_association_id"
    assert runlog.list_runs() == []


def test_chat_completions_omits_the_bridge_cleanly_when_logging_fails(iso, monkeypatch):
    """runlog.record failing (returning None, its own documented contract) must not surface a literal
    "null"/"None" anywhere, and must not break the reply itself -- logging must never break the request."""
    monkeypatch.setattr(runlog, "record", lambda **kw: None)
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]})
    header_block, _, payload = raw.partition(b"\r\n\r\n")
    body = json.loads(payload.decode("utf-8"))
    assert "clozn_run_id" not in body
    assert b"X-Clozn-Run-Id" not in header_block
    assert "null" not in payload.decode("utf-8").lower()
    assert body["choices"][0]["message"]["content"] == "A plain reply."   # the reply itself is unaffected


def test_streaming_terminal_chunk_carries_exact_resolvable_run_id(iso):
    """Headers are already committed, so the ordinary terminal finish chunk carries the stable id."""
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}], "stream": True})
    text = raw.decode("utf-8")
    assert "data: [DONE]" in text
    assert "X-Clozn-Run-Id" not in text
    frames = [json.loads(line[6:]) for line in text.splitlines()
              if line.startswith("data: {")]
    terminal = frames[-1]
    rid = terminal["clozn_run_id"]
    assert terminal["choices"][0]["finish_reason"] == "stop"
    assert runlog.get_run(rid)["response"] == "Hello"
    # the reply still streamed through untouched
    assert '"content": "Hel"' in text and '"content": "lo"' in text


def test_think_blocks_are_clean_on_openai_wire_history_trace_and_journal(iso, monkeypatch):
    class ThinkSub(FakeSub):
        def __init__(self):
            super().__init__()
            self.seen = []

        @staticmethod
        def _fill(mem_out):
            if mem_out is not None:
                mem_out.update(applied=[], gate=None, final_prompt="assistant\n<think>\n")

        def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
            self.seen = [dict(m) for m in messages]
            self._fill(mem_out)
            if trace_out is not None:
                trace_out.extend([
                    {"pos": 0, "piece": "private plan", "prob": .7},
                    {"pos": 1, "piece": "</think>", "prob": .8},
                    {"pos": 2, "piece": "answer", "prob": .9},
                ])
            return "private plan</think>answer"

        def chat_stream(self, messages, max_new=256, mem_out=None):
            self.seen = [dict(m) for m in messages]
            self._fill(mem_out)
            yield "private "
            yield "plan</thi"
            yield "nk>an"
            yield "swer"

        def last_stream_trace(self):
            return [
                {"pos": 0, "piece": "private "}, {"pos": 1, "piece": "plan</thi"},
                {"pos": 2, "piece": "nk>an"}, {"pos": 3, "piece": "swer"},
            ]

    sub = ThinkSub()
    monkeypatch.setattr(cs, "SUB", sub)
    old = {"role": "assistant", "content": "old scratch</think>old answer"}
    body = {"messages": [old, {"role": "user", "content": "next"}]}
    out = _post("/v1/chat/completions", body)
    assert out["choices"][0]["message"]["content"] == "answer"
    assert sub.seen[0]["content"] == "old answer"
    rec = runlog.get_run(out["clozn_run_id"])
    assert rec["response"] == "answer"
    assert rec["reasoning"]["blocks"][0]["text"] == "private plan"
    assert "".join(rec["trace"]["tokens"]) == "answer"

    raw = _post_raw("/v1/chat/completions", {**body, "stream": True})
    wire = raw.decode("utf-8")
    assert "private" not in wire and "think" not in wire
    assert '"content": "an"' in wire and '"content": "swer"' in wire
    streamed = runlog.get_run(runlog.list_runs(1)[0]["id"])
    assert streamed["response"] == "answer"
    assert streamed["reasoning"]["blocks"][0]["text"] == "private plan"


def test_streaming_path_uses_real_length_finish_reason_and_logs_it(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", FakeSub(finish_reason="length"))
    raw = _post_raw("/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}],
                                             "stream": True, "max_tokens": 2})
    text = raw.decode("utf-8")
    assert '"finish_reason": "length"' in text
    assert '"clozn_warnings"' in text and '"output_truncated"' in text
    logged = runlog.get_run(runlog.list_runs(1)[0]["id"])
    assert logged["finish_reason"] == "length"
    assert logged["meta"]["finish_reason_source"] == "substrate"
    assert logged["meta"]["max_tokens"] == 2
    assert logged["meta"]["stream"] is True
