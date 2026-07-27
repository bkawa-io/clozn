"""test_trust_field_server -- "trust as an API field" (FRONTIER §1.1): POST /v1/chat/completions can
OPT IN (clozn_trust:true) to get claim-level confidence spans over the reply attached to the OpenAI
response, so an agent can branch on per-claim confidence inline. Default OFF -> a standard OpenAI body is
byte-unchanged (compat). The spans come from the SAME producer as GET /runs/<id>/spans
(confidence_spans.spans over the normalized token trace).

HONESTY (FRONTIER §6): the spans are RAW, UNCALIBRATED model probabilities -- the response carries an
explicit clozn_spans_note saying so; NO calibration, nothing implies confidence == correctness.

Model-free: drives the REAL clozn_server do_POST handler with no socket (object.__new__(H)), isolated
runlog/cards/settings stores, a FAKE substrate whose chat() fills a per-token trace. Mirrors
test_bridge_server.py's conventions.
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
from clozn.runs import confidence_spans     # noqa: E402
import clozn.memory.cards as memory_cards         # noqa: E402
import clozn.memory.mode as memory_mode          # noqa: E402
import clozn.runs.store as runlog                # noqa: E402


class FakeSteer:
    def __init__(self):
        self.strength = {}

    def active(self):
        return {}


class FakeMem:
    def __init__(self):
        self.memory_strength = 1.0
        self.rules = []
        self.prefix = None


# A reply whose per-token confidence crosses bands + a sentence boundary -> at least two spans.
TRACE_STEPS = [
    {"piece": "The", "conf": 0.95},
    {"piece": " sky", "conf": 0.9},
    {"piece": " is", "conf": 0.86},
    {"piece": " maybe", "conf": 0.3},        # shaky band
    {"piece": " green.", "conf": 0.4},       # shaky, ends the sentence
]
REPLY_TEXT = "The sky is maybe green."


class TraceSub:
    """A qwen-shaped substrate whose chat() fills trace_out with a real per-token trace (so
    confidence_spans has something to segment)."""
    name = "qwen"

    def __init__(self, steps=TRACE_STEPS, reply=REPLY_TEXT):
        self.memory = FakeMem()
        self._mem = self.memory
        self.steer = FakeSteer()
        self._steps = steps
        self._reply = reply
        self._run_meta = {"model_id": "fake-qwen", "sampler_mode": "greedy",
                          "sampling": "greedy", "temperature": 0.0}

    def chat(self, messages, max_new=256, sample=True, trace_out=None, mem_out=None):
        self._run_meta.update(max_tokens=int(max_new), stream=False)
        if mem_out is not None:
            mem_out.update(applied=[], gate=None)
        if trace_out is not None:
            trace_out.extend([dict(s) for s in self._steps])
        return self._reply

    def last_finish_reason(self):
        return "stop"

    def run_meta(self):
        return dict(self._run_meta)


def _dispatch(path, body_obj):
    raw = json.dumps(body_obj).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    return h.wfile.getvalue()


def _post(path, body_obj):
    _, _, payload = _dispatch(path, body_obj).partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _get(path):
    raw = json.dumps({}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(memory_mode, "LEGACY_PREFIX_PATHS", [str(tmp_path / "no_such.pt")])
    monkeypatch.setattr(cs, "SUB", TraceSub())
    return tmp_path


UNCAL_NOTE = "uncalibrated raw token confidence -- self-confidence != correctness"


# ================================================================= opt-in ON

def test_opt_in_attaches_clozn_spans_and_uncalibrated_note(iso):
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "clozn_trust": True,
                 "messages": [{"role": "user", "content": "what color is the sky?"}]})
    assert "clozn_spans" in out
    assert isinstance(out["clozn_spans"], list) and out["clozn_spans"]      # real spans, not empty
    # the OpenAI reply itself is untouched
    assert out["choices"][0]["message"] == {"role": "assistant", "content": REPLY_TEXT}


def test_uncalibrated_note_is_present_and_honest(iso):
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "clozn_trust": True,
                 "messages": [{"role": "user", "content": "hi"}]})
    assert out["clozn_spans_note"] == UNCAL_NOTE
    # never implies confidence == correctness; says raw/uncalibrated + the self-confidence caveat
    assert "uncalibrated" in out["clozn_spans_note"]
    assert "self-confidence != correctness" in out["clozn_spans_note"]


def test_span_shape_matches_confidence_spans_producer(iso):
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "clozn_trust": True,
                 "messages": [{"role": "user", "content": "hi"}]})
    span = out["clozn_spans"][0]
    assert set(span.keys()) == {"start", "end", "text", "band", "mean_conf",
                                "min_conf", "n_tokens", "hesitations"}
    # crosses a band -> >= 2 spans (strong opening, shaky close)
    bands = [s["band"] for s in out["clozn_spans"]]
    assert "strong" in bands and "shaky" in bands


def test_clozn_spans_equals_the_runs_spans_endpoint(iso):
    """The MVP reuses the SAME producer + normalized trace as GET /runs/<id>/spans, so the inline spans
    are IDENTICAL to what a second call to /runs/<id>/spans would return for this run."""
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "clozn_trust": True,
                 "messages": [{"role": "user", "content": "hi"}]})
    rid = out["clozn_run_id"]
    endpoint = _get(f"/runs/{rid}/spans")
    assert out["clozn_spans"] == endpoint["spans"]


def test_opt_in_with_no_trace_gives_empty_spans_but_keeps_the_note(iso, monkeypatch):
    """A reply that recorded no trace -> honest empty spans (never fabricated), note still present."""
    monkeypatch.setattr(cs, "SUB", TraceSub(steps=[]))
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "clozn_trust": True,
                 "messages": [{"role": "user", "content": "hi"}]})
    assert out["clozn_spans"] == []
    assert out["clozn_spans_note"] == UNCAL_NOTE


# ================================================================= opt-in OFF (compat)

def test_default_off_is_byte_compatible_no_trust_fields(iso):
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "messages": [{"role": "user", "content": "hi"}]})
    assert "clozn_spans" not in out
    assert "clozn_spans_note" not in out
    # unchanged OpenAI shape (+ the pre-existing clozn_run_id bridge field)
    assert set(out.keys()) == {"id", "object", "created", "model", "choices", "clozn_run_id"}


def test_explicit_false_does_not_attach_spans(iso):
    out = _post("/v1/chat/completions",
                {"model": "clozn-qwen", "clozn_trust": False,
                 "messages": [{"role": "user", "content": "hi"}]})
    assert "clozn_spans" not in out
    assert "clozn_spans_note" not in out
