"""Contract tests for the opt-in route autoloader (clozn/server/routes/_autoload.py)."""
from __future__ import annotations

import itertools
import sys
import types

import pytest

from clozn.server import app
from clozn.server.routes import _autoload

_SHIM_SEQ = itertools.count()


def _shim(tmp_path, monkeypatch):
    """Point the autoloader at `tmp_path` as a throwaway route package.

    Unique name per call, cleanup purges submodules -- see tests/test_identity_ext.py's `_shim` for why
    a fixed name produces a stale-module bug that passes alone and fails in a full run.
    """
    name = f"zz_route_shim_{next(_SHIM_SEQ)}"
    module = types.ModuleType(name)
    module.__path__ = [str(tmp_path)]
    sys.modules[name] = module
    monkeypatch.setattr(_autoload, "_DIR", str(tmp_path))
    monkeypatch.setattr(_autoload, "_PACKAGE", name)

    def _cleanup():
        for key in [k for k in sys.modules if k == name or k.startswith(f"{name}.")]:
            sys.modules.pop(key, None)
    return _cleanup


def test_no_route_module_failed_to_load():
    """A route family that opts in and then fails to import degrades to a stderr warning so it cannot
    take down the whole gateway. This is what stops that warning from being the only signal."""
    assert _autoload.LOAD_FAILURES == [], (
        "route modules opted into autoload but failed: "
        + "; ".join(f"{n}: {type(e).__name__}: {e}" for n, e in _autoload.LOAD_FAILURES))


def test_the_runs_fallback_is_still_last_in_get_routes():
    """The load-bearing ordering property. app.py's own comment: the generic GET /runs/<id> fallback
    must be registered LAST so every more-specific /runs/<id>/<suffix> family gets first refusal. An
    autoloaded route spliced after it would be shadowed -- and shadowed as a wrong-shaped 200 rather
    than a 404, which is the version of this bug that costs an afternoon."""
    assert app._GET_ROUTES[-1] is app._runs_fallback_routes


def test_existing_route_families_survived_the_splice():
    """A representative sample, to catch a splice that dropped or reordered the hand-wired families."""
    from clozn.server import static as static_routes
    from clozn.server.routes import health, openai, runs
    assert app._GET_ROUTES[0] is static_routes, "static must stay first"
    for mod in (health, runs, openai):
        assert mod in app._GET_ROUTES, f"{mod.__name__} disappeared from _GET_ROUTES"
    for mod in (health, openai):
        assert mod in app._POST_ROUTES, f"{mod.__name__} disappeared from _POST_ROUTES"


def test_a_get_only_module_is_not_added_to_post_routes(tmp_path, monkeypatch):
    (tmp_path / "getonly.py").write_text(
        "CLOZN_ROUTE_AUTOLOAD = True\n"
        "def try_get(h, p):\n"
        "    return False\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    try:
        found = _autoload.discover()
        assert len(_autoload.with_try_get(found)) == 1
        assert _autoload.with_try_post(found) == []
    finally:
        cleanup()


def test_a_post_only_module_is_not_added_to_get_routes(tmp_path, monkeypatch):
    (tmp_path / "postonly.py").write_text(
        "CLOZN_ROUTE_AUTOLOAD = True\n"
        "def try_post(h, p, body):\n"
        "    return False\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    try:
        found = _autoload.discover()
        assert _autoload.with_try_get(found) == []
        assert len(_autoload.with_try_post(found)) == 1
    finally:
        cleanup()


def test_discovery_order_is_deterministic(tmp_path, monkeypatch):
    """Dispatch order decides which family gets first refusal on an overlapping path, so it must not
    depend on filesystem iteration order."""
    for name in ("zeta", "alpha", "mu"):
        (tmp_path / f"{name}.py").write_text(
            "CLOZN_ROUTE_AUTOLOAD = True\ndef try_get(h, p):\n    return False\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    try:
        names = [m.__name__.rsplit(".", 1)[-1] for m in _autoload.discover()]
        assert names == ["alpha", "mu", "zeta"]
    finally:
        cleanup()


def test_a_module_without_the_marker_is_not_imported(tmp_path, monkeypatch):
    (tmp_path / "inert.py").write_text("raise AssertionError('must not be imported')\n",
                                       encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    before = len(_autoload.LOAD_FAILURES)
    try:
        assert _autoload.discover() == []
        assert len(_autoload.LOAD_FAILURES) == before
    finally:
        cleanup()


def test_a_module_opting_in_with_no_handler_is_recorded(tmp_path, monkeypatch, capsys):
    """Roadmap rule 3: a module that says it serves routes and then serves none is a defect, not a
    no-op to pass over quietly."""
    (tmp_path / "handlerless.py").write_text("CLOZN_ROUTE_AUTOLOAD = True\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    before = len(_autoload.LOAD_FAILURES)
    try:
        assert _autoload.discover() == []
        assert len(_autoload.LOAD_FAILURES) == before + 1
        assert "handlerless" in capsys.readouterr().err
    finally:
        cleanup()
        del _autoload.LOAD_FAILURES[before:]


def test_a_broken_module_is_recorded_not_raised(tmp_path, monkeypatch, capsys):
    (tmp_path / "broken.py").write_text(
        "CLOZN_ROUTE_AUTOLOAD = True\nimport nonexistent_module_xyz\n", encoding="utf-8")
    cleanup = _shim(tmp_path, monkeypatch)
    before = len(_autoload.LOAD_FAILURES)
    try:
        assert _autoload.discover() == []           # did not raise
        assert len(_autoload.LOAD_FAILURES) == before + 1
        assert "broken" in capsys.readouterr().err
    finally:
        cleanup()
        del _autoload.LOAD_FAILURES[before:]


def test_marker_constant_is_the_documented_name():
    assert _autoload.MARKER == "CLOZN_ROUTE_AUTOLOAD"
