"""Contract tests for the namespaced run-identity extension seam (clozn/runs/identity_ext.py)."""
from __future__ import annotations

import pytest

from clozn import schemas
from clozn.runs import identity, identity_ext


@pytest.fixture(autouse=True)
def _isolate_discovery_cache():
    """identity_ext caches its provider scan per process; these tests repoint _DIR, so the cache must be
    dropped on both sides or a test either sees a stale list or poisons the ones after it."""
    identity_ext.reset_cache()
    yield
    identity_ext.reset_cache()


def _shim(tmp_path, monkeypatch, name="zz_ident_shim"):
    import sys
    import types
    module = types.ModuleType(name)
    module.__path__ = [str(tmp_path)]
    sys.modules[name] = module
    monkeypatch.setattr(identity_ext, "_DIR", str(tmp_path))
    monkeypatch.setattr(identity_ext, "_PACKAGE", name)
    return lambda: sys.modules.pop(name, None)


def test_no_shipped_provider_failed_to_load():
    """The teeth behind collect()'s broad except: a provider that raises is swallowed at runtime so it
    cannot cost a real user their run. That must never mean a broken provider ships unnoticed."""
    identity_ext.collect({})
    assert identity_ext.COLLECT_FAILURES == [], (
        "identity providers failed: "
        + "; ".join(f"{n}: {type(e).__name__}: {e}" for n, e in identity_ext.COLLECT_FAILURES))


def test_a_provider_lands_under_its_namespace(tmp_path, monkeypatch):
    (tmp_path / "engine_artifact.py").write_text(
        "NAME = 'engine_artifact'\n"
        "def identity(context):\n"
        "    return {'protocol_version': (context or {}).get('probe', 7)}\n",
        encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    try:
        assert identity_ext.collect({"probe": 9}) == {"engine_artifact": {"protocol_version": 9}}
    finally:
        cleanup()


@pytest.mark.parametrize("returns", ["None", "{}", "'not a dict'", "[]", "0"])
def test_an_unestablished_facet_is_omitted_not_null_padded(tmp_path, monkeypatch, returns):
    """identity.py's contract is that a key is OMITTED, never null-padded, when it cannot be honestly
    measured. A provider with nothing to say must therefore produce no namespace at all."""
    (tmp_path / "empty_facet.py").write_text(
        f"NAME = 'empty_facet'\ndef identity(context):\n    return {returns}\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    try:
        assert identity_ext.collect({}) == {}
    finally:
        cleanup()


def test_a_raising_provider_costs_its_facet_not_the_run(tmp_path, monkeypatch):
    """A provider runs on the path that records a real user's run. It must never be able to break it."""
    (tmp_path / "aa_explodes.py").write_text(
        "NAME = 'explodes'\ndef identity(context):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    (tmp_path / "bb_healthy.py").write_text(
        "NAME = 'healthy'\ndef identity(context):\n    return {'ok': True}\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    before = len(identity_ext.COLLECT_FAILURES)
    try:
        # The healthy provider still contributes -- one bad facet does not poison the others.
        assert identity_ext.collect({}) == {"healthy": {"ok": True}}
        assert len(identity_ext.COLLECT_FAILURES) == before + 1
    finally:
        cleanup()
        del identity_ext.COLLECT_FAILURES[before:]


def test_a_malformed_provider_module_is_recorded(tmp_path, monkeypatch):
    (tmp_path / "no_name.py").write_text("def identity(context):\n    return {'x': 1}\n",
                                         encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    before = len(identity_ext.COLLECT_FAILURES)
    try:
        assert identity_ext.collect({}) == {}
        assert len(identity_ext.COLLECT_FAILURES) == before + 1
    finally:
        cleanup()
        del identity_ext.COLLECT_FAILURES[before:]


def test_runtime_identity_still_validates_against_its_schema():
    """The seam changed runtime_identity()'s output shape; the shipped schema must still describe it."""
    block = identity.runtime_identity()
    block["schema_version"] = "clozn.run-identity.v1"
    schemas.validate(block)


def test_runtime_identity_omits_ext_when_no_provider_contributes():
    """No providers ship yet, so `ext` must be absent entirely rather than present-and-empty."""
    assert "ext" not in identity.runtime_identity()


def test_runtime_identity_never_raises_on_a_broken_provider(tmp_path, monkeypatch):
    (tmp_path / "boom.py").write_text(
        "NAME = 'boom'\ndef identity(context):\n    raise RuntimeError('boom')\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    before = len(identity_ext.COLLECT_FAILURES)
    try:
        block = identity.runtime_identity()
        assert "captured_at" in block and "ext" not in block
    finally:
        cleanup()
        del identity_ext.COLLECT_FAILURES[before:]
