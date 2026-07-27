"""test_steer_engine_down.py -- the /steer/* write API while the engine is unreachable (engine-down
pressure test finding #1, 2026-07-19).

Two things this file proves, model-free (no C++ engine process, no GPU, no real socket):

  1a. Substrate._ensure_steer() (clozn/server/substrates.py) used to let a raw urllib.error.URLError from
      EngineSteer.compute()'s first harvest() round-trip escape all the way to app.py's generic
      `except Exception as e: self._json(500, {"error": f"{type(e).__name__}: {e}"})` fallback -- a wrong
      status code (500, implying an internal bug) and an unhelpful bare-URLError message with no route
      context, unlike every other engine-touching route's clean 502 (`"engine-chat: ..."`,
      `"engine: ..."`, etc.). It's now caught at the _ensure_steer() boundary and re-raised as
      ctx.EngineUnavailable, which _dispatch_post catches specially -> a clean 502 with
      ctx._engine_unreachable_message().
  1b. /steer/custom_delete (a pure dict-pop) and /steer/concept/set + /steer/concept/check (a DIFFERENT
      mechanism, ConceptSteer, unrelated to _ensure_steer()'s diff-of-means calibration) used to sit
      behind that SAME unconditional _ensure_steer() call regardless of path -- so a user couldn't delete
      a stale custom dial, or use the any-concept dial, while the engine was down, even though neither
      operation touches what _ensure_steer() calibrates. /steer/concept/* is exercised (with its own
      mechanism un-gated) in test_steer_concept_routes.py; this file's custom_delete test proves the same
      un-gating here, directly against a FakeEngine whose harvest() would blow up if ever called.

FakeEngine.harvest() always raises urllib.error.URLError -- the exact exception clozn_engine.EngineClient
lets propagate on a connection refusal (EngineClient._request only translates an HTTPError, i.e. the
engine responding with a JSON 4xx, into EngineError; a plain connection failure is a raw URLError). Drives
the real do_POST handler via the object.__new__(H) no-socket trick (mirrors test_rewrite_route.py's
_post/_post_raw, which also returns the HTTP status alongside the body -- needed here since the whole
point under test is the STATUS CODE, not just the message).
"""
from __future__ import annotations

import io
import json
import os
import sys
import urllib.error

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.server import app as cs          # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
import clozn.memory.cards as memory_cards                # noqa: E402
import clozn.memory.mode as memory_mode                 # noqa: E402


class FakeEngine:
    """.base -- what ctx._engine_unreachable_message() reports. .harvest -- always raises URLError,
    mirroring a connection refusal (EngineSteer.compute()'s very first engine round-trip)."""

    def __init__(self):
        self.base = "http://127.0.0.1:1"
        self.timeout = 0.2
        self.harvest_calls = []

    def harvest(self, text, layer=None):
        self.harvest_calls.append(text)
        raise urllib.error.URLError("refused")


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate every path this suite might touch, mirroring test_engine_add_custom.py's own iso fixture."""
    monkeypatch.setattr(cs, "CLOZN_DIR", str(tmp_path))
    monkeypatch.setattr(memory_cards, "CARDS_PATH", str(tmp_path / "cards.json"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    return tmp_path


@pytest.fixture
def fake_engine(iso, monkeypatch):
    """clozn.server.app.ENGINE -> a fresh FakeEngine whose harvest() always refuses; ENGINE_STEER reset
    so _engine_steer() builds a real steering.EngineSteer on it (construction makes no harvest call --
    only compute(), via _ensure_steer(), would)."""
    fe = FakeEngine()
    monkeypatch.setattr(cs, "ENGINE", fe)
    monkeypatch.setattr(cs, "ENGINE_STEER", None)
    sub = cs.EngineSubstrate()
    monkeypatch.setattr(cs, "SUB", sub)
    monkeypatch.setattr(cs, "SUBNAME", "engine")
    return fe


def _post_raw(path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    return h.wfile.getvalue()


def _post(path, body_obj=None):
    raw = _post_raw(path, body_obj)
    head, _, payload = raw.partition(b"\r\n\r\n")
    status = int(head.split(b" ", 2)[1])
    return status, json.loads(payload.decode("utf-8"))


# ==================================================================================== 1a. _ensure_steer() itself

def test_ensure_steer_raises_engine_unavailable_not_a_bare_urlerror(fake_engine):
    """Unit-level: the exception TYPE conversion happens at the _ensure_steer() boundary itself, not just
    incidentally at the HTTP layer."""
    sub = cs.SUB
    with pytest.raises(cs.EngineUnavailable) as exc_info:
        sub._ensure_steer()
    assert "engine not reachable at http://127.0.0.1:1" in str(exc_info.value)
    assert "is it running" in str(exc_info.value)
    assert not sub._steer_ready                    # the failed attempt never flips readiness


@pytest.mark.parametrize("path,body", [
    ("/steer/set", {"name": "warm", "value": 0.5}),
    ("/steer/check", {"name": "warm", "value": 1.0, "prompt": "hi"}),
    ("/steer/compute", {}),
])
def test_steer_routes_needing_calibration_are_a_clean_502_when_the_engine_is_down(fake_engine, path, body):
    status, out = _post(path, body)
    assert status == 502
    assert out == {"error": "engine not reachable at http://127.0.0.1:1 -- is it running?"}
    # not the old shape: a bare, route-less "URLError: <urlopen error ...>" 500 body
    assert "URLError" not in out["error"]


def test_steer_set_failure_is_not_sticky_the_engine_returning_self_heals(fake_engine):
    """Confirms the pre-existing self-heal behavior the pressure report noted still holds under the new
    error path: _steer_ready only flips True on an actual successful compute(), so a later call (once the
    engine responds) is not permanently poisoned by the earlier failure."""
    status, out = _post("/steer/set", {"name": "warm", "value": 0.5})
    assert status == 502
    assert cs.SUB._steer_ready is False              # still un-poisoned, ready to retry


# ==================================================================================== 1b. custom_delete un-gated

def test_steer_custom_delete_bypasses_the_calibration_harvest_when_the_engine_is_down(fake_engine):
    """The exact regression this fix targets: /steer/custom_delete is a pure dict-pop and must succeed (or
    fail on its own terms) without ever touching the engine, even while it's down."""
    cs.SUB.steer.custom["mine"] = {"pos": "a", "neg": "b", "max": 0.6, "poles": ["mine", "neutral"],
                                   "source": "user"}
    cs.SUB.steer.vecs["mine"] = [0.0] * 8

    status, out = _post("/steer/custom_delete", {"name": "mine"})

    assert status == 200
    assert out == {"custom": []}
    assert "mine" not in cs.SUB.steer.custom
    assert fake_engine.harvest_calls == []           # the ~35-round-trip calibration never ran
    assert cs.SUB._steer_ready is False               # _ensure_steer() was never even attempted


# ==================================================================================== control: /steer/axes untouched

def test_steer_axes_is_unaffected_by_a_down_engine(fake_engine):
    """/steer/axes is metadata-only and never calls _ensure_steer() -- confirms it stays a clean 200 even
    while every calibration-needing route 502s (the pressure report's own control comparison)."""
    status, out = _post("/steer/axes", {})
    assert status == 200
    assert out["ready"] is False
    assert fake_engine.harvest_calls == []
