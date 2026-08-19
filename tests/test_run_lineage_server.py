"""GET /runs/<id>/lineage endpoint wiring for branch/replay tree data."""
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
import clozn.runs.store as runlog  # noqa: E402


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
    return head.decode("latin-1"), body


@pytest.fixture
def iso(tmp_path, monkeypatch):
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(cs, "SUB", None)
    return tmp_path


def test_lineage_missing_run_is_a_clean_404(iso):
    head, body = _get("/runs/run_nope/lineage")
    assert "404" in head
    assert json.loads(body) == {"error": "run not found"}


def test_lineage_endpoint_needs_no_substrate(iso):
    root = runlog.record(source="studio_chat", client="studio",
                         messages=[{"role": "user", "content": "root"}], response="root",
                         started=1000.0, ended=1000.1)
    child = runlog.record(source="replay", client="studio",
                          messages=[{"role": "user", "content": "child"}], response="child",
                          parent_run_id=root, changes_applied={"plain": True},
                          started=1001.0, ended=1001.2)

    assert cs.SUB is None
    head, body = _get(f"/runs/{child}/lineage")

    assert "200" in head
    data = json.loads(body)
    assert data["run_id"] == child
    assert data["original"]["id"] == root
    assert data["current"]["id"] == child
    assert data["ancestors"][0]["id"] == root
    assert data["tree"]["children"][0]["id"] == child
    assert data["tree"]["children"][0]["change_label"] == "re-roll"
