"""test_explain_cli -- model-free tests for clozn_cli.py's `clozn explain` (EXPLAIN_THIS_ANSWER_SPEC.md
Milestone 5, the TUI half). M1's /runs/<id>/explain (research/explain.py, wired into clozn_server.py) already
assembles the explanation object server-side; this command is DISPLAY ONLY, per the milestone's scope -- it
generates nothing, it just renders what the endpoint returns. This file tests format_explain(), the pure
"explanation object -> terminal text" function clozn_cli.py factors out specifically so that's possible with
a canned dict: no running Studio, no model, no GPU.

Layout:
  * canned-dict tests drive format_explain() directly against hand-built /explain-shaped fixtures -- exactly
    the "no server, no model" contract the milestone asks for -- and assert the honesty invariants (no
    aggregate %, "was active" not "caused", every not-available note renders verbatim).
  * a wire-format check drives the REAL clozn_server /explain endpoint in-process (the same no-socket
    object.__new__(H) trick test_explain_server.py already uses) so the CLI renderer is proven against a
    genuine server response, not just a hand-guessed shape.
  * _last_run_id() is checked against a real (tmp-path-isolated) runlog directory.
  * _fetch_explain()'s honest failure path is checked against a guaranteed-closed local port -- no network
    flakiness, no live Studio required.

What this file can NOT exercise: a live HTTP round trip against a *running* `clozn studio` process. Per the
task at hand the Studio is down right now; a full `clozn studio` + `clozn explain --last` smoke test remains
a manual follow-up once it's back up.
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))          # tests/
REPO = os.path.dirname(HERE)                                 # repo root (clozn/cli.py lives here)
sys.path.insert(0, REPO)

import clozn.cli.main as clozn_cli                                            # noqa: E402
from clozn.server import app as cs                                    # noqa: E402

import clozn.runs.store as runlog                                                # noqa: E402

_PCT_RE = re.compile(r"\d+(\.\d+)?\s*%")   # any aggregate-percentage-shaped substring -- must NEVER appear


# ---------------------------------------------------------------------------------- canned-dict fixtures

HAPPY_PATH = {
    "run_id": "run_deadbeef",
    "confidence": {
        "available": True,
        "threshold": 0.5,
        "n_tokens": 5,
        "summary": "2 hesitations",
        "uncertain_moments": [
            {"index": 1, "token": " sky", "confidence": 0.30, "alternatives": [{"piece": " sea", "prob": 0.22}]},
            {"index": 3, "token": " blue", "confidence": 0.41,
             "alternatives": [{"piece": " grey", "prob": 0.31}, {"piece": " green", "prob": 0.10}]},
        ],
    },
    "influences_active": {
        "gate": 0.77,
        "mode": "prompt",
        "cards": [
            {"id": "mem_1", "text": "Keep it brief.", "causal_verified": None, "has_provenance": True,
             "source_run_id": "run_src", "source_turn": 1, "quoted_span": "please keep it brief"},
        ],
        "dials": [{"name": "concise", "value": 0.5, "causal_verified": None}],
    },
    "concepts": {"available": False, "note": "concept readout needs the qwen/PyTorch substrate (SAE) — not available on this run."},
}


def test_happy_path_renders_hesitations_alternatives_and_gate():
    out = clozn_cli.format_explain(HAPPY_PATH)
    assert "run_deadbeef" in out
    assert "2 hesitations" in out
    assert "sky" in out and "sea" in out and "0.22" in out          # the alternative + its probability
    assert "0.30" in out                                             # the recorded per-token confidence
    assert "0.77" in out and "prompt" in out                         # gate value + mode
    assert "Keep it brief." in out
    assert "please keep it brief" in out                             # the provenance quote, verbatim
    assert "concise" in out and "0.50" in out                        # the dial + its value


def test_sparkline_renders_and_uses_only_block_glyphs():
    out = clozn_cli.format_explain(HAPPY_PATH)
    assert any(ch in out for ch in clozn_cli._SPARK)


def test_causal_verified_null_labels_was_active_not_caused():
    out = clozn_cli.format_explain(HAPPY_PATH)
    assert "was active" in out
    assert "caused" not in out.lower()


def test_causal_verified_true_or_false_labels_appropriately():
    """M1 never emits true/false (it's always null), but the tag function must not overclaim if a future
    M2 receipt ever attaches one -- proven only says so when TOLD so, never inferred from "active"."""
    assert clozn_cli._verified_tag(True) == "proven"
    assert clozn_cli._verified_tag(False) == "ruled out"
    assert clozn_cli._verified_tag(None) == "was active"


def test_never_renders_an_aggregate_percentage():
    """The whole honesty point: no synthesized overall confidence number, anywhere, in any form -- not even
    accidentally as a percentage sign near the word confidence."""
    out = clozn_cli.format_explain(HAPPY_PATH)
    assert not _PCT_RE.search(out), out
    assert "confidence:" not in out.lower()   # no single colon-scalar summary line either


def test_not_available_notes_render_verbatim_not_hidden():
    expl = {
        "run_id": "run_no_trace",
        "confidence": {"available": False, "note": "token trace captured on the engine path"},
        "influences_active": {"cards": [], "dials": [], "gate": None, "mode": None, "note": "no memory applied"},
        "concepts": {"available": False, "note": "concept readout needs the qwen/PyTorch substrate (SAE) — not available on this run."},
    }
    out = clozn_cli.format_explain(expl)
    assert "token trace captured on the engine path" in out
    assert "no memory applied" in out
    assert "concept readout needs the qwen/PyTorch substrate (SAE)" in out
    assert "not available" in out.lower()
    assert not _PCT_RE.search(out)


def test_zero_hesitations_and_no_influences_still_renders_honestly_not_hidden():
    expl = {
        "run_id": "run_confident",
        "confidence": {"available": True, "threshold": 0.5, "n_tokens": 3, "summary": "0 hesitations",
                       "uncertain_moments": []},
        "influences_active": {"cards": [], "dials": [], "gate": None, "mode": None, "note": "no memory applied"},
        "concepts": {"available": False, "note": "concept readout needs the qwen/PyTorch substrate (SAE) — not available on this run."},
    }
    out = clozn_cli.format_explain(expl)
    assert "0 hesitations" in out
    assert "no memory applied" in out
    assert "no dials active" in out
    assert not _PCT_RE.search(out)


def test_concepts_available_renders_features_and_scores():
    expl = dict(HAPPY_PATH)
    expl["concepts"] = {"available": True, "spans": [
        {"position": 0, "piece": "Dragons", "features": [
            {"id": "sae:42", "label": "mythical-creature", "score": 0.91},
            {"id": "sae:9001", "label": "dragon", "score": 0.83},
        ]},
    ]}
    out = clozn_cli.format_explain(expl)
    assert "mythical-creature" in out and "0.91" in out
    assert "dragon" in out and "0.83" in out


@pytest.mark.parametrize("garbage", [
    None, "not a dict", 42, [], ["also", "not", "a", "dict"],
    {"confidence": "nope", "influences_active": 3, "concepts": ["nope"]},
    {"confidence": {"uncertain_moments": "not-a-list"}, "influences_active": {"cards": "nope", "dials": {}},
     "concepts": {"available": True, "spans": "nope"}},
])
def test_never_raises_on_malformed_input(garbage):
    out = clozn_cli.format_explain(garbage)      # must not raise
    assert isinstance(out, str) and out
    assert not _PCT_RE.search(out)


def test_empty_dict_degrades_fully_and_honestly():
    out = clozn_cli.format_explain({})
    assert "not available" in out.lower()
    assert not _PCT_RE.search(out)


# --------------------------------------------------------------------------------- wire-format compatibility
# Drives the REAL clozn_server /explain handler in-process (mirrors test_explain_server.py's own no-socket
# object.__new__(H) trick exactly) so format_explain() is proven against a genuine server response, not just
# a fixture I hand-built above.

def _post(path, body_obj=None):
    raw = json.dumps(body_obj if body_obj is not None else {}).encode("utf-8")
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(raw)
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": str(len(raw)), "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"POST {path} HTTP/1.1", "HTTP/1.1", "POST"
    h.do_POST()
    _, _, payload = h.wfile.getvalue().partition(b"\r\n\r\n")
    return json.loads(payload.decode("utf-8"))


@pytest.fixture
def iso(tmp_path, monkeypatch):
    """Isolate the run log + card store (mirrors test_explain_server.py's `iso`); SUB stays None -- explain
    needs no substrate, so this exercises the exact same "free" path a real Studio would take."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path




def test_format_explain_renders_a_genuine_404_shape(iso):
    """/explain's 404 is {"error": "run not found"} -- format_explain must not choke on a payload with none
    of the three panel keys it normally expects."""
    out_json = _post("/runs/run_does_not_exist/explain")
    assert out_json == {"error": "run not found"}
    out = clozn_cli.format_explain(out_json)
    assert isinstance(out, str) and out
    assert not _PCT_RE.search(out)


# ------------------------------------------------------------------------------------------- _last_run_id

def test_last_run_id_finds_the_newest_run(iso):
    """Explicit, clearly-ordered `started=` timestamps (record()'s id embeds a millisecond-resolution
    timestamp) -- two back-to-back calls can otherwise land in the same millisecond, making sort order
    between them depend on a random uuid suffix instead of creation order. Deterministic > timing-lucky."""
    runlog.record(source="cli", messages=[{"role": "user", "content": "first"}], response="a", started=1000.0)
    newest = runlog.record(source="cli", messages=[{"role": "user", "content": "second"}], response="b",
                           started=2000.0)
    assert clozn_cli._last_run_id() == newest


def test_last_run_id_is_none_when_no_runs_exist(iso):
    assert clozn_cli._last_run_id() is None


def test_last_run_id_skips_internal_replay_runs(iso):
    """Regression (receipt-journal spam): a `/runs/<id>/receipts` prove-all persists its leave-one-out
    re-generations as their own runs (source="replay") -- `clozn explain --last` must resolve to the
    user's own last turn, not a newer internal regeneration."""
    real = runlog.record(source="cli", messages=[{"role": "user", "content": "first"}], response="a",
                         started=1000.0)
    runlog.record(source="replay", messages=[{"role": "user", "content": "first"}], response="b",
                 started=2000.0, parent_run_id=real)
    assert clozn_cli._last_run_id() == real


# ------------------------------------------------------------------------------------------- _fetch_explain

def test_fetch_explain_is_an_honest_cloznerror_when_studio_is_down():
    """No live Studio required (and per the task at hand, none is running): a guaranteed-closed local port
    must fail as one clean, actionable CloznError -- never a raw urllib traceback."""
    port = clozn_cli._free_port()   # bound then released -- nothing is listening on it
    with pytest.raises(clozn_cli.CloznError):
        clozn_cli._fetch_explain(port, "run_x")
