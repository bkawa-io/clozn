"""Persisted and per-request footer-mode compatibility tests."""
from __future__ import annotations

import clozn.settings as settings
from clozn.server.routes import receipt_link


class Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body

    def _send(self, status, body, *_args, **_kwargs):
        self.status, self.body = status, body


def test_enabled_legacy_boolean_defaults_to_exceptions(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    settings.set_setting("receipt_footer", True)
    assert receipt_link.receipt_mode() == "exceptions"
    assert receipt_link.request_receipt_mode({"clozn_receipt": True}) == "exceptions"


def test_disabled_legacy_boolean_stays_off_until_per_request_opt_in(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    assert receipt_link.receipt_mode() == "off"
    assert receipt_link.request_receipt_mode({}) == "off"
    assert receipt_link.request_receipt_mode({"clozn_receipt": True}) == "exceptions"
    assert receipt_link.request_receipt_mode({"clozn_receipt": False}) == "off"


def test_persisted_always_mode_and_per_request_override(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    settings.set_setting("receipt_footer_mode", "always")
    settings.set_setting("receipt_footer", True)
    assert receipt_link.receipt_mode() == "always"
    assert receipt_link.request_receipt_mode({}) == "always"
    assert receipt_link.request_receipt_mode({"clozn_receipt_mode": "exceptions"}) == "exceptions"
    assert receipt_link.request_receipt_mode({"clozn_receipt_mode": "off"}) == "off"


def test_mode_route_persists_new_setting_without_changing_r_permalink(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "settings.json"))
    h = Handler()
    assert receipt_link.try_post(h, "/receipt/mode", {"receipt_footer_mode": "always"}) is True
    assert h.status == 200 and h.body["receipt_footer_mode"] == "always"
    assert h.body["receipt_footer"] is True
    assert receipt_link.try_get(h, "/receipt/mode") is True
    assert h.body["receipt_footer_mode"] == "always"

    # The existing /r/<id> behavior remains the Studio redirect.
    receipt_link.try_get(h, "/r/run_123")
    assert h.status == 302
