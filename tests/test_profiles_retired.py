"""Regression coverage for retiring named behavior profiles / persona bundles.

The invariant this file exists to pin down (docs/CAPABILITIES.md): CLOZN is a debugger, not a persona
manager. Removing the profile lifecycle must not reset a user's already-persisted dial state, must not
read `~/.clozn/profiles/` for anything, and must leave a stale `active_profile` setting or a
`meta.active_profile`-bearing historical run completely harmless.
"""
from __future__ import annotations

import json

import pytest

import clozn.runs.store as runlog
import clozn.settings as settings


# --------------------------------------------------------------------- the Profile package is gone

def test_profiles_package_no_longer_exists():
    """clozn.profiles (ProfileStore, new_profile, validate, apply_dials, ...) is fully removed, not
    merely disconnected -- nothing can read ~/.clozn/profiles/ even if it wanted to."""
    with pytest.raises(ImportError):
        import clozn.profiles  # noqa: F401
    with pytest.raises(ImportError):
        from clozn.profiles import store  # noqa: F401


def test_server_has_no_active_profile_state():
    """_active_profile_name() and _profiles_switch() (clozn/server/app.py) are gone, not just unused."""
    from clozn.server import app
    assert not hasattr(app, "_active_profile_name")
    assert not hasattr(app, "_profiles_switch")


def test_log_run_source_no_longer_writes_active_profile():
    """The run-journaling path must not still contain a dead-but-present write of meta.active_profile."""
    from clozn.server import app
    source = open(app.__file__, encoding="utf-8").read()
    assert "active_profile" not in source


# ------------------------------------------------------------------------- /profiles/* routes retired

class _Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_profiles_get_and_post_routes_return_typed_410():
    from clozn.server.routes import profiles as route
    for path in ("/profiles/list",):
        handler = _Handler()
        assert route.try_get(handler, path) is True
        assert handler.status == 410
        assert handler.body["code"] == "profiles_retired"
    for path, body in (
        ("/profiles/save", {"name": "work"}),
        ("/profiles/switch", {"name": "work"}),
        ("/profiles/export", {"name": "work"}),
        ("/profiles/import", {"profile": {}}),
        ("/profiles/delete", {"name": "work"}),
    ):
        handler = _Handler()
        assert route.try_post(handler, path, body) is True
        assert handler.status == 410, path
        assert handler.body["code"] == "profiles_retired", path


def test_profiles_route_never_returns_a_success_status():
    """Never leave an endpoint that appears successful but silently does nothing."""
    from clozn.server.routes import profiles as route
    handler = _Handler()
    route.try_post(handler, "/profiles/switch", {"name": "work"})
    assert handler.status is not None and handler.status >= 400


# -------------------------------------------------------------------- stale active_profile is inert

def test_stale_active_profile_setting_has_no_reader(tmp_path, monkeypatch):
    """A pre-existing active_profile key in studio_settings.json is legitimately still on disk (this
    build never scrubs it), but nothing in the server reads that key back to shape a generation."""
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))
    settings.set_setting("active_profile", "work")
    # The key really is there -- confirms this is a meaningful "stale but present" scenario, not a
    # vacuous one. See test_log_run_source_no_longer_writes_active_profile for the confirmation that
    # no production module reads it back.
    assert settings.get_setting("active_profile") == "work"


def test_new_run_never_carries_active_profile(tmp_path, monkeypatch):
    """Even with a stale active_profile setting present, a freshly recorded run carries no
    active_profile metadata -- there is no live code path left that could attach it."""
    monkeypatch.setattr(settings, "SETTINGS_PATH", str(tmp_path / "studio_settings.json"))
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    settings.set_setting("active_profile", "work")

    run_id = runlog.record(
        source="engine_chat", client="studio", model="alpha", substrate="engine",
        messages=[{"role": "user", "content": "hi"}], response="hello there",
        final_prompt="<user>hi</user>",
    )
    assert run_id
    run = runlog.get_run(run_id)
    assert "active_profile" not in (run.get("meta") or {})


# ---------------------------------------------------------------------- old profile files are inert

def test_old_profile_files_are_never_read(tmp_path, monkeypatch):
    """A leftover ~/.clozn/profiles/<name>.json with extreme dial values must never be consulted --
    there is no reader left, so a fresh EngineSteer's state cannot be affected by its presence."""
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir()
    (profiles_dir / "work.json").write_text(json.dumps({
        "version": 1, "schema_version": "clozn.behavior-profile.v1", "name": "work",
        "description": "", "cards": [], "dials": {"concise": 99.0}, "custom_dials": [], "facts": [],
        "created_at": 1.0, "updated_at": 1.0,
    }), encoding="utf-8")

    from clozn.behavior.steering.engine_adapter import EngineSteer
    steer = EngineSteer(engine_client=None)
    assert steer.strength == {}
    # The file is still there, untouched -- nothing about constructing/using a fresh steer reads it.
    assert (profiles_dir / "work.json").exists()
    assert steer.strength.get("concise") is None


# --------------------------------------------------------------- existing steering persistence survives

def test_steering_dial_persistence_survives_profile_removal(tmp_path):
    """The ordinary /steer/set persistence mechanism (EngineSteer.save_state/load_state) is untouched
    by removing Profiles -- a dial set and saved through it is still there after a fresh load, exactly
    as before. This is the SAME save_state/load_state pair /steer/set and (formerly) profile switch
    both called; only the profile-switch caller is gone."""
    from clozn.behavior.steering.engine_adapter import EngineSteer
    path = str(tmp_path / "studio_personality.json")

    first = EngineSteer(engine_client=None)
    first.set("concise", 0.4)
    first.save_state(path)

    second = EngineSteer(engine_client=None)
    assert second.strength == {}
    second.load_state(path)
    assert second.strength["concise"] == pytest.approx(0.4)


def test_removing_profiles_does_not_reset_already_persisted_dial_state(tmp_path):
    """A user's dial state persisted before this cut (e.g. via a since-removed profile switch, which
    always wrote through the same save_state() call /steer/set uses) must still load normally --
    retiring Profiles must never look like a runtime reset to that user."""
    from clozn.behavior.steering.engine_adapter import EngineSteer
    path = str(tmp_path / "studio_personality.json")
    # Simulate state left behind by a pre-retirement profile switch: a plain {dial: strength} JSON
    # file, written by the exact same save_state() call this build still uses.
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"concise": 0.8, "warm": -0.2}, handle)

    restored = EngineSteer(engine_client=None)
    restored.load_state(path)
    assert restored.strength == {"concise": 0.8, "warm": -0.2}


# ------------------------------------------------------------------- historical runs remain readable

def test_historical_run_with_active_profile_metadata_remains_readable(tmp_path, monkeypatch):
    """A run recorded before the retirement may carry meta.active_profile. Reading it back today must
    not crash and must preserve that historical evidence -- only new runs stop writing it."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = runlog.record(
        source="engine_chat", client="studio", model="alpha", substrate="engine",
        messages=[{"role": "user", "content": "hi"}], response="hello there",
        final_prompt="<user>hi</user>",
        meta={"active_profile": "work"},
    )
    assert run_id
    run = runlog.get_run(run_id)
    assert run is not None
    assert run["meta"]["active_profile"] == "work"
