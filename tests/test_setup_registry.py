from __future__ import annotations

import os

import pytest

from clozn.setup import registry
from clozn.setup.errors import RegistryError


def _record(entrypoint="x"):
    """A syntactically valid installed_artifact record. Pure -- writes nothing to disk. Every function
    under test here (record_install/rollback/save/load) treats `entrypoint` as an opaque string; only
    prune_missing() ever stat()s it, and the tests that exercise that path build a REAL file under
    tmp_path explicitly rather than relying on this helper to have created one."""
    return {
        "version": "1.0.0", "os": "windows", "arch": "x86_64", "backend": "cpu",
        "sha256": "a" * 64, "protocol_version": "1.0",
        "entrypoint": entrypoint, "installed_at": "2026-07-27T00:00:00+00:00",
    }


def test_load_missing_registry_is_empty_but_valid(tmp_path):
    doc = registry.load(str(tmp_path))
    assert doc == {"schema_version": "clozn.engine-registry.v1"}


def test_load_corrupt_registry_self_heals_to_empty(tmp_path):
    os.makedirs(registry.engines_dir(str(tmp_path)))
    with open(registry.registry_path(str(tmp_path)), "w", encoding="utf-8") as handle:
        handle.write("{ not json")
    doc = registry.load(str(tmp_path))
    assert doc == {"schema_version": "clozn.engine-registry.v1"}


def test_save_then_load_round_trips(tmp_path):
    doc = registry.record_install(
        {}, "1.0.0/windows-x86_64-cpu", _record(str(tmp_path / "bin" / "clozn-server.exe")),
        make_active=True)
    registry.save(str(tmp_path), doc)
    reloaded = registry.load(str(tmp_path))
    assert reloaded["active"] == "1.0.0/windows-x86_64-cpu"
    assert "1.0.0/windows-x86_64-cpu" in reloaded["installed"]


def test_save_refuses_an_invalid_document(tmp_path):
    with pytest.raises(RegistryError):
        registry.save(str(tmp_path), {"active": 12345})   # active must be a string per the schema


def test_record_install_first_install_has_no_previous():
    doc = registry.record_install({}, "1.0.0/windows-x86_64-cpu", _record(), make_active=True)
    assert doc["active"] == "1.0.0/windows-x86_64-cpu"
    assert "previous" not in doc


def test_record_install_upgrade_sets_previous():
    doc = registry.record_install({}, "1.0.0/windows-x86_64-cpu", _record(), make_active=True)
    doc = registry.record_install(doc, "1.1.0/windows-x86_64-cpu", _record(), make_active=True)
    assert doc["active"] == "1.1.0/windows-x86_64-cpu"
    assert doc["previous"] == "1.0.0/windows-x86_64-cpu"
    assert set(doc["installed"]) == {"1.0.0/windows-x86_64-cpu", "1.1.0/windows-x86_64-cpu"}


def test_record_install_force_reinstall_of_active_does_not_clobber_previous():
    doc = registry.record_install({}, "1.0.0/windows-x86_64-cpu", _record(), make_active=True)
    doc = registry.record_install(doc, "1.1.0/windows-x86_64-cpu", _record(), make_active=True)
    doc = registry.record_install(doc, "1.1.0/windows-x86_64-cpu", _record(), make_active=True)
    assert doc["active"] == "1.1.0/windows-x86_64-cpu"
    assert doc["previous"] == "1.0.0/windows-x86_64-cpu"   # unchanged, not overwritten with itself


def test_rollback_swaps_active_and_previous():
    doc = registry.record_install({}, "1.0.0/windows-x86_64-cpu", _record(), make_active=True)
    doc = registry.record_install(doc, "1.1.0/windows-x86_64-cpu", _record(), make_active=True)
    rolled = registry.rollback(doc)
    assert rolled["active"] == "1.0.0/windows-x86_64-cpu"
    assert rolled["previous"] == "1.1.0/windows-x86_64-cpu"


def test_rollback_with_no_previous_raises():
    doc = registry.record_install({}, "1.0.0/windows-x86_64-cpu", _record(), make_active=True)
    with pytest.raises(RegistryError, match="nothing to roll back to"):
        registry.rollback(doc)


def test_rollback_when_previous_was_removed_from_installed_raises():
    doc = {"active": "1.0.0/x", "previous": "0.9.0/x", "installed": {"1.0.0/x": _record()}}
    with pytest.raises(RegistryError, match="no longer in the registry"):
        registry.rollback(doc)


def test_prune_missing_drops_entries_whose_entrypoint_is_gone(tmp_path):
    present = str(tmp_path / "present.exe")
    with open(present, "w", encoding="utf-8") as handle:
        handle.write("x")
    missing = str(tmp_path / "missing.exe")   # never created
    doc = {
        "active": "1.0.0/x", "previous": "0.9.0/x",
        "installed": {
            "1.0.0/x": _record(present),
            "0.9.0/x": _record(missing),
        },
    }
    pruned = registry.prune_missing(doc)
    assert set(pruned["installed"]) == {"1.0.0/x"}
    assert pruned["active"] == "1.0.0/x"
    assert "previous" not in pruned   # its target vanished along with the entry
