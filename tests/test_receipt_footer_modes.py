"""Persisted and per-request footer-mode tests."""
from __future__ import annotations

import clozn.settings as settings
from clozn.server.routes import receipt_link


class Handler:
    def __init__(self):
        self.status = None
        self.body = None
        self.headers = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body

    def _send(self, status, body, *_args, **_kwargs):
        self.status, self.body = status, body
        self.headers = _args[1] if len(_args) > 1 else _kwargs.get("extra_headers")


def test_default_is_off_and_request_mode_is_explicit(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    assert receipt_link.receipt_mode() == "off"
    assert receipt_link.request_receipt_mode({}) == "off"
    assert receipt_link.request_receipt_mode({"clozn_receipt_mode": "exceptions"}) == "exceptions"


def test_persisted_always_mode_and_per_request_override(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    settings.set_setting("receipt_footer_mode", "always")
    assert receipt_link.receipt_mode() == "always"
    assert receipt_link.request_receipt_mode({}) == "always"
    assert receipt_link.request_receipt_mode({"clozn_receipt_mode": "exceptions"}) == "exceptions"
    assert receipt_link.request_receipt_mode({"clozn_receipt_mode": "off"}) == "off"


def test_mode_route_persists_new_setting_without_changing_r_permalink(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    h = Handler()
    assert receipt_link.try_post(h, "/receipt/mode", {"receipt_footer_mode": "always"}) is True
    assert h.status == 200 and h.body["receipt_footer_mode"] == "always"
    assert receipt_link.try_get(h, "/receipt/mode") is True
    assert h.body["receipt_footer_mode"] == "always"

    receipt_link.try_get(h, "/r/run_123")
    assert h.status == 302
    assert h.body == ""
    assert h.headers == {"Location": "/app/index.html#/inspect/run_123"}
