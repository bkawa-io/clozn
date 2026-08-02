"""HTTP list/unpin coverage for the Studio durable snapshot ledger."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from clozn.server.routes import snapshots


class Handler:
    def __init__(self, path):
        self.path = path
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_list_route_returns_versioned_manifests(monkeypatch):
    monkeypatch.setattr(
        "clozn.replay.checkpoint_pin_store.list_pins",
        lambda: [{"schema_version": "clozn.pinned-checkpoint.v1", "run_id": "run_a"}],
    )
    h = Handler("/snapshots")
    assert snapshots.try_get(h, "/snapshots") is True
    assert h.status == 200
    assert h.body == {
        "schema_version": "clozn.pinned-checkpoint-list.v1",
        "snapshots": [{"schema_version": "clozn.pinned-checkpoint.v1", "run_id": "run_a"}],
    }


def test_unpin_route_passes_cascade_and_returns_receipt(monkeypatch):
    seen = {}

    def unpin(run_id, *, cascade):
        seen.update(run_id=run_id, cascade=cascade)
        return {"ok": True, "action": "unpin", "run_id": run_id, "cascade": cascade}

    monkeypatch.setattr("clozn.replay.checkpoint_pin_store.unpin_checkpoint", unpin)
    h = Handler("/snapshots/run_abc/unpin")
    assert snapshots.try_post(h, "/snapshots/run_abc/unpin", {"cascade": True}) is True
    assert h.status == 200
    assert seen == {"run_id": "run_abc", "cascade": True}
    assert h.body["ok"] is True


def test_unpin_route_rejects_non_boolean_cascade():
    h = Handler("/snapshots/run_abc/unpin")
    assert snapshots.try_post(h, "/snapshots/run_abc/unpin", {"cascade": "yes"}) is True
    assert h.status == 400
    assert h.body["code"] == "snapshot_unpin_invalid_cascade"


def test_unpin_route_surfaces_dependents(monkeypatch):
    class Dependents(Exception):
        children = ["run_child"]

    from clozn.replay import checkpoint_pin_store as pins
    monkeypatch.setattr(pins, "PinHasDependentsError", Dependents)
    monkeypatch.setattr(pins, "unpin_checkpoint", lambda *_args, **_kwargs: (_ for _ in ()).throw(Dependents("has children")))
    h = Handler("/snapshots/run_abc/unpin")
    snapshots.try_post(h, "/snapshots/run_abc/unpin", {})
    assert h.status == 409
    assert h.body["code"] == "snapshot_unpin_has_dependents"
    assert h.body["children"] == ["run_child"]
