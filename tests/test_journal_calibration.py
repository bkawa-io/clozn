"""GET /journal/calibration -- the TRUTH-tier calibration report served from disk (persisted by
`clozn eval --save`), beside the PROXY curve at /journal/actuary."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clozn.eval import store as eval_store          # noqa: E402
from clozn.server.routes import journal as journal_routes  # noqa: E402


class Handler:
    def _json(self, status, obj):
        self.status = status
        self.obj = obj


def _get(path):
    h = Handler()
    assert journal_routes.try_get(h, path) is True
    return h


def test_available_false_when_nothing_saved(tmp_path, monkeypatch):
    monkeypatch.setattr(eval_store, "_PATH", str(tmp_path / "calibration_profiles.json"))
    h = _get("/journal/calibration")
    assert h.status == 200 and h.obj["available"] is False
    assert "clozn eval --save" in h.obj["note"]                  # tells the user how to populate it


def test_serves_the_saved_report_with_age(tmp_path, monkeypatch):
    path = str(tmp_path / "calibration_profiles.json")
    monkeypatch.setattr(eval_store, "_PATH", path)
    eval_store.save_profile(
        {"model": "model-a", "set": "arith", "saved_ts": 100.0,
         "report": {"available": True, "ece": 0.15}},
        task="arith",
    )
    h = _get("/journal/calibration")
    assert h.status == 200 and h.obj["available"] is True
    assert h.obj["set"] == "arith" and h.obj["report"]["ece"] == 0.15
    assert "saved_ago_s" in h.obj                                # freshness surfaced to the consumer


def test_unrelated_path_is_not_claimed():
    h = Handler()
    assert journal_routes.try_get(h, "/journal/nope") is False
