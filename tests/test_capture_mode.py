"""test_capture_mode -- the Light/Standard/Deep/Lab capture tier: the module (persist / validate / policy)
and the GET/POST /capture/tier endpoint wiring. Model-free -- the setting is a tmp studio_settings.json,
and the endpoints are driven through the real handler with the no-socket object.__new__(H) trick."""
import io
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
RESEARCH = os.path.dirname(HERE)
sys.path.insert(0, RESEARCH)

from clozn.runs import capture_mode        # noqa: E402
import clozn.settings as clozn_settings          # noqa: E402
from clozn.server import app as cs  # noqa: E402
import clozn.memory.mode as memory_mode         # noqa: E402


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setattr(clozn_settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))
    return tmp_path


# --- the module -------------------------------------------------------------------------------------------

def test_default_tier_is_standard(settings):
    assert capture_mode.tier() == "standard"


def test_set_tier_persists_and_rejects_unknown(settings):
    assert capture_mode.set_tier("light") is True
    assert capture_mode.tier() == "light"
    assert capture_mode.set_tier("telepathy") is False      # unknown -> rejected, not persisted
    assert capture_mode.tier() == "light"


def test_set_tier_is_case_and_space_insensitive(settings):
    assert capture_mode.set_tier("  DEEP ") is True
    assert capture_mode.tier() == "deep"


def test_captures_trace_only_light_drops_it(settings):
    assert capture_mode.captures_trace("light") is False
    for t in ("standard", "deep", "lab"):
        assert capture_mode.captures_trace(t) is True


def test_garbage_setting_degrades_to_standard(settings):
    memory_mode.set_setting("capture_tier", "nonsense")
    assert capture_mode.tier() == "standard"


# --- the endpoints ----------------------------------------------------------------------------------------

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


def test_endpoint_get_reports_tier_and_the_ladder(settings):
    out = _dispatch("GET", "/capture/tier")
    assert out["tier"] == "standard"
    assert out["tiers"] == ["light", "standard", "deep", "lab"]


def test_endpoint_post_sets_the_tier(settings):
    assert _dispatch("POST", "/capture/tier", {"tier": "deep"}) == {"ok": True, "tier": "deep"}
    assert _dispatch("GET", "/capture/tier")["tier"] == "deep"


def test_endpoint_post_rejects_unknown_tier(settings):
    out = _dispatch("POST", "/capture/tier", {"tier": "telepathy"})
    assert "error" in out
    assert _dispatch("GET", "/capture/tier")["tier"] == "standard"   # unchanged
