"""Route tests for original-versus-reduced-child comparison."""
from __future__ import annotations

import clozn.runs.store as runlog
from clozn.server.routes import run_compare as route


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def _run(rid, *, parent=None):
    rec = {
        "id": rid,
        "response": "same",
        "finish_reason": "stop",
        "identity": {"model_sha256": "a" * 64},
        "context_receipt": {"rendered": {"tokens": 20 if parent is None else 10}},
        "context_units": {"default_source_ids": ["src_a", "src_b"] if parent is None else ["src_a"]},
    }
    if parent is not None:
        rec.update({
            "parent_run_id": parent,
            "changes_applied": {"experiment": {
                "intervention": {"kind": "delete_source", "source_ids": ["src_b"]},
            }},
        })
    return rec


def test_autoload_marker_is_set():
    assert route.CLOZN_ROUTE_AUTOLOAD is True


def test_unrelated_path_is_not_handled():
    h = Handler("/runs/latest")
    assert route.try_get(h, "/runs/latest") is False
    assert h.status is None


def test_both_explicit_ids_are_required():
    for path in ("/runs/compare", "/runs/compare?a=run_a", "/runs/compare?b=run_b"):
        h = Handler(path)
        assert route.try_get(h, "/runs/compare") is True
        assert h.status == 400
        assert h.body["code"] == "comparison_pair_required"


def test_missing_runs_yield_404_naming_each_missing_id(monkeypatch):
    monkeypatch.setattr(runlog, "get_run", lambda rid: None)
    h = Handler("/runs/compare?a=run_a&b=run_b")

    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 404
    assert h.body["missing"] == ["run_a", "run_b"]


def test_arbitrary_run_pair_is_rejected(monkeypatch):
    records = {"run_a": _run("run_a"), "run_b": _run("run_b")}
    monkeypatch.setattr(runlog, "get_run", records.get)
    h = Handler("/runs/compare?a=run_a&b=run_b")

    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 409
    assert h.body["code"] == "not_reduction_child"


def test_saved_reduction_returns_narrow_comparison(monkeypatch):
    records = {"run_a": _run("run_a"), "run_b": _run("run_b", parent="run_a")}
    monkeypatch.setattr(runlog, "get_run", records.get)
    h = Handler("/runs/compare?a=run_a&b=run_b")

    assert route.try_get(h, "/runs/compare") is True
    assert h.status == 200
    assert h.body["schema_version"] == "clozn.reduction-comparison.v1"
    assert h.body["preservation"]["exact"] is True
    assert h.body["rendered_prompt_tokens"]["saved"] == 10
    assert h.body["context"]["removed_source_ids"] == ["src_b"]
