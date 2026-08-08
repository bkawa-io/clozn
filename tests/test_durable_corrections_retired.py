"""Regression coverage for the "retire durable personalization / Teach Once" cut.

The invariant this file exists to pin down (docs/CAPABILITIES.md): nothing persisted by the old
personalization system may influence a new generation. One-shot corrective retries remain fully
functional; only the DURABLE, auto-applying machinery (session/profile-scoped corrective_retries,
the F5/F6 correction registry) was removed.
"""
from __future__ import annotations

import json

import pytest

import clozn.runs.store as runlog
from clozn.profiles import store as profiles


# ------------------------------------------------------------------ the persistent modules are gone

def test_persistent_corrective_retries_module_no_longer_exists():
    """The session/profile-scoped policy store (~/.clozn/corrective_retries.json,
    effective_presets/inject/activate/undo) is fully removed, not merely disconnected -- a
    reawakened import can never resurrect it."""
    with pytest.raises(ImportError):
        import clozn.behavior.corrective_retries  # noqa: F401


def test_durable_correction_registry_modules_no_longer_exist():
    """F5 (clozn.runs.corrections) and F6 (clozn.runs.teaching_loop) -- the durable "Teach Once"
    draft/confirm/promote/enable/disable lifecycle -- are fully removed."""
    with pytest.raises(ImportError):
        import clozn.runs.corrections  # noqa: F401
    with pytest.raises(ImportError):
        import clozn.runs.teaching_loop  # noqa: F401


def test_generation_gateway_has_no_persistent_injection_functions():
    """The three functions that used to look up and splice a saved policy into a live request
    (apply_corrective_policy, apply_scoped_corrections, reapply_scoped_resolution) are gone from the
    generation path, not merely unreachable."""
    from clozn.server import generation_gateway as gw
    for name in ("apply_corrective_policy", "apply_scoped_corrections", "reapply_scoped_resolution"):
        assert not hasattr(gw, name), f"generation_gateway still exposes {name}"


def test_generation_routes_no_longer_import_the_retired_correction_store():
    """openai.py/ollama.py must not still import clozn.runs.corrections or reference a saved
    correction resolution on the request handler."""
    import clozn.server.routes.openai as openai_route
    import clozn.server.routes.ollama as ollama_route
    for module in (openai_route, ollama_route):
        source = open(module.__file__, encoding="utf-8").read()
        assert "clozn.runs.corrections" not in source
        assert "_correction_resolution" not in source


# ------------------------------------------------------------------------ legacy profile is inert

def test_legacy_profile_response_policies_survives_load_without_crashing(tmp_path):
    """A profile bundle saved before the retirement may still carry `response_policies`. Loading it
    must not raise, and the field is preserved verbatim (never destructively rewritten) -- but it is
    also never consulted anywhere in the product to shape a generation anymore."""
    store = profiles.ProfileStore(str(tmp_path / "profiles"))
    legacy = profiles.new_profile("legacy")
    legacy["response_policies"] = ["less-verbose", "use-context"]
    store.save(legacy)

    loaded = store.load("legacy")
    assert loaded["response_policies"] == ["less-verbose", "use-context"]

    # Saving it again (e.g. the user tweaks a dial and re-saves) must not destroy the legacy field.
    loaded["dials"]["warm"] = 0.3
    store.save(loaded)
    assert store.load("legacy")["response_policies"] == ["less-verbose", "use-context"]


def test_new_profiles_never_carry_the_retired_field(tmp_path):
    store = profiles.ProfileStore(str(tmp_path / "profiles"))
    fresh = profiles.new_profile("fresh")
    assert "response_policies" not in fresh
    store.save(fresh)
    assert "response_policies" not in store.load("fresh")


def test_legacy_profile_with_junk_response_policies_does_not_crash(tmp_path):
    """A hand-edited or corrupted legacy field degrades gracefully -- validate() must never raise
    over data in a field nothing reads anymore."""
    store = profiles.ProfileStore(str(tmp_path / "profiles"))
    bundle = profiles.new_profile("junk")
    bundle["response_policies"] = "not-a-list"
    store.save(bundle)
    loaded = store.load("junk")
    assert "response_policies" not in loaded


# ------------------------------------------------------------------- historical runs remain readable

def test_historical_correction_bearing_run_remains_readable(tmp_path, monkeypatch):
    """A run recorded before the retirement may carry applied_corrections/correction_conflicts
    (F5 receipt fields) and a corrective_retry.scope of "session"/"profile" inside its behavior
    metadata. Reading it back today must not crash and must preserve that historical evidence."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = runlog.record(
        source="engine_chat", client="studio", model="alpha", substrate="engine",
        messages=[{"role": "user", "content": "hi"}], response="hello there",
        final_prompt="<user>hi</user>",
        applied_corrections=[{
            "correction_id": "corr_" + "a" * 24, "type": "style",
            "scope": {"kind": "session"}, "content_hash": "deadbeef",
        }],
        correction_conflicts=[],
    )
    assert run_id
    run = runlog.get_run(run_id)
    assert run is not None
    assert run["applied_corrections"][0]["correction_id"] == "corr_" + "a" * 24
    assert run["context_receipt"]["applied_corrections"][0]["type"] == "style"


def test_historical_run_with_legacy_corrective_retry_scope_remains_readable(tmp_path, monkeypatch):
    """Old runs from the session/profile-scoped one-shot retry route recorded
    `behavior_intervention`-shaped identity/metadata with a "scope" of "session"/"profile". A new
    build must still be able to read that run back; it just never writes that shape again."""
    monkeypatch.setattr(runlog, "RUNS_DIR", str(tmp_path / "runs"))
    run_id = runlog.record(
        source="engine_chat", client="studio", model="alpha", substrate="engine",
        messages=[{"role": "user", "content": "hi"}], response="short reply",
        final_prompt="<user>hi</user>",
        meta={"corrective_retry": {"arm": "corrected", "preset": "less-verbose", "scope": "session"}},
    )
    assert run_id
    run = runlog.get_run(run_id)
    assert run["meta"]["corrective_retry"]["scope"] == "session"


# ------------------------------------------------------------------------- retired /corrections route

class _Handler:
    def __init__(self):
        self.status = None
        self.body = None

    def _json(self, status, body, **_kwargs):
        self.status, self.body = status, body


def test_corrections_get_route_returns_typed_410():
    from clozn.server.routes import corrections as route
    handler = _Handler()
    assert route.try_get(handler, "/corrections") is True
    assert handler.status == 410
    assert handler.body["code"] == "durable_corrections_retired"

    handler = _Handler()
    assert route.try_get(handler, "/corrections/corr_" + "a" * 24) is True
    assert handler.status == 410
    assert handler.body["code"] == "durable_corrections_retired"


def test_corrections_post_route_returns_typed_410_for_every_lifecycle_action():
    from clozn.server.routes import corrections as route
    for path, body in (
        ("/corrections", {"scope_kind": "session", "type": "style", "content": "x"}),
        ("/corrections/corr_" + "a" * 24 + "/confirm", {}),
        ("/corrections/corr_" + "a" * 24 + "/enable", {}),
        ("/corrections/corr_" + "a" * 24 + "/disable", {}),
        ("/corrections/corr_" + "a" * 24 + "/delete", {}),
        ("/corrections/corr_" + "a" * 24 + "/undo", {}),
        ("/corrections/corr_" + "a" * 24 + "/verify", {}),
        ("/corrections/resolve", {}),
    ):
        handler = _Handler()
        assert route.try_post(handler, path, body) is True
        assert handler.status == 410, path
        assert handler.body["code"] == "durable_corrections_retired", path


def test_corrections_route_never_returns_a_success_status():
    """Never leave an endpoint that appears successful but no longer applies the correction."""
    from clozn.server.routes import corrections as route
    handler = _Handler()
    route.try_post(handler, "/corrections", {"scope_kind": "session", "type": "style", "content": "x"})
    assert handler.status is not None and handler.status >= 400


def test_corrections_cli_command_no_longer_registered():
    """`clozn corrections ...` (F5/F6) is fully removed from the CLI, not just the HTTP surface."""
    import clozn.cli.commands._autoload as autoload
    assert "corrections" not in autoload._candidates()
