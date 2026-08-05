"""GET /runs/<id>/family -- the WHOLE branch family as GET /runs-shaped summaries, past the /runs
80-window, so the studio lineage tree-builder (buildLineageFromRuns) can render a run's COMPLETE family
even when older ancestors / sibling branches fell outside the recent-runs slice.

Model-free: drives the REAL clozn_server do_GET handler with no socket (object.__new__(H)), against an
isolated runlog store, SUB=None (the endpoint needs no substrate). Mirrors test_run_lineage_server.py.
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

from clozn.server import app as cs  # noqa: E402
import clozn.runs.store as runlog
from clozn.runs import summaries  # noqa: E402


def _get(path):
    H = cs.make_handler()
    h = object.__new__(H)
    h.path = path
    h.rfile = io.BytesIO(b"")
    h.wfile = io.BytesIO()
    h.headers = {"Content-Length": "0", "User-Agent": "pytest"}
    h.requestline, h.request_version, h.command = f"GET {path} HTTP/1.1", "HTTP/1.1", "GET"
    h.do_GET()
    head, _, body = h.wfile.getvalue().partition(b"\r\n\r\n")
    return head.decode("latin-1"), json.loads(body)


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


def _seed_family():
    """root -> child -> grandchild, plus a sibling of child; and one UNRELATED run. Returns the ids."""
    root = runlog.record(source="studio_chat", client="studio",
                         messages=[{"role": "user", "content": "root"}], response="root",
                         started=1000.0, ended=1000.1)
    child = runlog.record(source="replay", client="studio",
                          messages=[{"role": "user", "content": "child"}], response="child",
                          parent_run_id=root, changes_applied={"memory_off": True},
                          started=1001.0, ended=1001.2)
    grandchild = runlog.record(source="replay", client="studio",
                               messages=[{"role": "user", "content": "gc"}], response="gc",
                               parent_run_id=child, started=1002.0, ended=1002.1)
    sibling = runlog.record(source="branch", client="studio",
                            messages=[{"role": "user", "content": "sib"}], response="sib",
                            parent_run_id=root, started=1003.0, ended=1003.1)
    unrelated = runlog.record(source="studio_chat", client="studio",
                              messages=[{"role": "user", "content": "other"}], response="other",
                              started=1004.0, ended=1004.1)
    return dict(root=root, child=child, grandchild=grandchild, sibling=sibling, unrelated=unrelated)


# ============================================================================ runlog.lineage_family unit

def test_lineage_family_unknown_id_is_none(iso):
    assert runlog.lineage_family("run_nope") is None


def test_lineage_family_returns_whole_connected_component(iso):
    ids = _seed_family()
    fam = runlog.lineage_family(ids["grandchild"])          # ask from a LEAF -- must still reach the whole tree
    got = {r["id"] for r in fam}
    assert got == {ids["root"], ids["child"], ids["grandchild"], ids["sibling"]}   # the family...
    assert ids["unrelated"] not in got                                             # ...but NOT the unrelated run


def test_lineage_family_summary_shape_matches_runs_listing(iso):
    ids = _seed_family()
    fam = runlog.lineage_family(ids["child"])
    listing = runlog.list_runs(80)
    # Every family entry has the same key set as a GET /runs list entry, so buildLineageFromRuns is
    # identical on both. Compared with the confidence group excluded on BOTH sides: those five keys
    # are sent only for a run that recorded a trace (see summaries._summary), so two runs can honestly
    # differ there while still being the same shape for the tree-builder's purposes.
    variable = set(summaries._CONFIDENCE_FIELDS)
    listing_keys = set(listing[0].keys())
    assert listing_keys <= set(runlog.SUMMARY_FIELDS)
    assert set(runlog.SUMMARY_FIELDS) - listing_keys <= variable
    for entry in fam:
        assert set(entry.keys()) - variable == listing_keys - variable
    # spot-check the fields the client's tree-builder actually reads.
    for entry in fam:
        for k in ("id", "parent_run_id", "created_at", "source", "prompt_summary",
                  "response_summary", "finish_reason", "timing"):
            assert k in entry


def test_lineage_family_is_newest_first(iso):
    ids = _seed_family()
    fam = runlog.lineage_family(ids["root"])
    ts = [e["timing"]["started_at"] for e in fam]
    assert ts == sorted(ts, reverse=True)                  # newest-first, like GET /runs


# ============================================================================ endpoint wiring

def test_family_endpoint_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/family")
    assert "404" in head
    assert body == {"error": "run not found"}


def test_family_endpoint_returns_the_full_family(iso):
    ids = _seed_family()
    head, body = _get(f"/runs/{ids['child']}/family")
    assert "200" in head
    assert set(body.keys()) == {"runs"}
    got = {r["id"] for r in body["runs"]}
    assert got == {ids["root"], ids["child"], ids["grandchild"], ids["sibling"]}
    assert ids["unrelated"] not in got


def test_family_endpoint_needs_no_substrate(iso):
    ids = _seed_family()
    assert cs.SUB is None
    head, body = _get(f"/runs/{ids['grandchild']}/family")
    assert "200" in head
    assert ids["root"] in {r["id"] for r in body["runs"]}
