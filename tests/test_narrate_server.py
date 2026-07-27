"""test_narrate_server -- POST /runs/<id>/narrate, the M4 endpoint wiring (EXPLAIN_THIS_ANSWER_SPEC.md):
makes the accountable-self narration + confabulation-diff REACHABLE from the product.

No model, no GPU: drives the REAL clozn_server do_POST handler (the object.__new__(H) no-socket trick used
by test_counterfactual_server / test_receipts_server / test_explain_server) against isolated stores, with a
FAKE substrate for the two generations narrate() makes (constrained + unconstrained). narrate.py's four
contracts + the trap guard are exhaustively unit-tested in test_narrate.py, and the REAL cross-encoder judge
in the gated test_semantic_matcher_gated.py; this file proves only the THIN endpoint wiring: route matches,
missing run -> 404, no substrate -> 503 (both arms generate), and one real request comes back with narrate's
4-key shape over HTTP.

The matcher SELECTION is forced to the lexical fallback here (monkeypatch semantic_matcher.available ->
False) so this stays model-free -- loading the ~440MB cross-encoder is the gated test's job, not a unit
test's. The endpoint's real behavior (use the NLI judge when its checkpoint is present, else the labeled
lexical proxy) is exactly that branch; this pins that the fallback path routes and that the response's own
note names the matcher that ran.
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

from clozn.server import app as cs      # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402


import clozn.runs.store as runlog                  # noqa: E402
import clozn.receipts.semantic_matcher as semantic_matcher        # noqa: E402


_CONFAB = "I answered because of my deep love of medieval chess."


class NarrateFakeSub:
    """Two-armed: the unconstrained "why" call (asked "Why did you answer that way?") returns a
    confabulation; the constrained-narration call returns a plain cited line. Keys off the prompt text,
    exactly like narrate.py routes the two calls."""
    name = "qwen"

    def __init__(self):
        self.calls = 0

    def chat(self, messages, max_new=256, sample=True):
        self.calls += 1
        joined = " ".join(m.get("content", "") for m in messages if isinstance(m, dict))
        if "Why did you answer that way" in joined:
            return _CONFAB
        return "I kept it brief and clear."


def _dispatch(method, path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"{method} {path} HTTP/1.1", "HTTP/1.1", method
    getattr(h, f"do_{method}")()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


def _post(path, body_obj=None):
    return _dispatch("POST", path, body_obj)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the stores; SUB is the fake two-armed substrate; force the LEXICAL fallback so no checkpoint
    loads (the NLI path is the gated test). Tests that want the 503 path override SUB to None."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    monkeypatch.setattr(semantic_matcher, "available", lambda: False)   # model-free: no cross-encoder load
    monkeypatch.setattr(cs, "SUB", NarrateFakeSub())
    return tmp_path


def _seed_run():
    return runlog.record(source="studio_chat", client="studio", model="clozn-qwen", substrate="QwenSubstrate",
                         messages=[{"role": "user", "content": "Explain TCP vs UDP."}],
                         response="TCP is reliable; UDP is fast.",
                         behavior={"active_dials": {"warm": 0.2}})


def test_narrate_missing_run_is_a_clean_404(iso):
    out = _post("/runs/run_does_not_exist/narrate", {})
    assert out == {"error": "run not found"}


def test_narrate_needs_the_substrate_503(iso, monkeypatch):
    monkeypatch.setattr(cs, "SUB", None)
    rid = _seed_run()
    out = _post(f"/runs/{rid}/narrate", {})
    assert out == {"error": "narration requires a ready product model worker"}


def test_narrate_happy_path_over_http(iso):
    rid = _seed_run()
    out = _post(f"/runs/{rid}/narrate", {})
    assert "error" not in out
    # narrate()'s exact 4-key shape round-tripped over HTTP -- no answer/why/response field exists.
    assert set(out.keys()) == {"constrained_narration", "flags", "unsupported_claims", "note"}
    # THE TRAP GUARD: the confabulation appears ONLY wrapped in a flag, NEVER as the narration/answer surface
    # and never as unflagged prose. (The claim text inside a "WARNING: credits ..." flag is the point --
    # showing exactly what was fabricated -- so it SHOULD be in flags, just never trusted.)
    assert "chess" not in str(out["constrained_narration"]), out["constrained_narration"]
    assert any("chess" in f for f in out["flags"]), out            # flagged, wrapped in a WARNING
    assert all(f.startswith("WARNING:") for f in out["flags"]), out
    # the fallback matcher actually ran, and the response says so (self-describing honesty level).
    assert "lexical_default" in out["note"]
    # with an empty lexicon (no cards; only a dial named 'warm'), lexical flags the chess confab as unsupported
    assert any("chess" in e.get("claim", "") for e in out["unsupported_claims"]), out
