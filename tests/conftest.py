"""tests conftest -- the `-m model` gate, plus the real-user-data tripwire.

The `-m model` gate: a bare `@pytest.mark.model` test would otherwise RUN in the plain suite -- and
these tests load a real checkpoint onto the GPU and TTT-train a soft prefix for minutes. This makes the
marker actually gate: model-marked tests are skipped unless the run's mark expression names "model".

The tripwire (`_never_write_the_real_user_data`): dozens of test files isolate themselves by
monkeypatching clozn.settings.SETTINGS_PATH (and the card/run stores) at a tmp dir. That isolation is
invisible when it FAILS -- a test that misses the patch just quietly rewrites the developer's own
~/.clozn/studio_settings.json (sampling, guard config) and still passes. This autouse fixture hashes
the real files before and after every test and fails the test that changed one.

It checks the FILES, not the module globals, on purpose: a path assertion would fire on the hundreds of
tests that legitimately never touch settings, and would miss a write that reached the real store by some
other route. Hashing answers the question that actually matters -- "did this test edit my real data?" --
and stays correct through refactors that move the module or rename the global.
"""
from __future__ import annotations

import hashlib
import os

import pytest

_CLOZN = os.path.expanduser("~/.clozn")
_GUARDED = [os.path.join(_CLOZN, name) for name in ("studio_settings.json",)]


def _digest(path):
    """(size, sha256) for an existing file, None when absent. Absent-vs-present is itself a change."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        return len(data), hashlib.sha256(data).hexdigest()
    except FileNotFoundError:
        return None
    except OSError:
        return "unreadable"


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "model: loads a real checkpoint on the GPU and trains (slow); deselected unless run with -m model")


@pytest.fixture(autouse=True)
def _never_write_the_real_user_data():
    """Fail any test that creates, edits, or deletes the developer's real ~/.clozn user data."""
    before = {p: _digest(p) for p in _GUARDED}
    yield
    for path, was in before.items():
        now = _digest(path)
        assert now == was, (
            f"this test modified REAL user data at {path}.\n"
            "Some store was not redirected to tmp_path -- e.g.\n"
            "  monkeypatch.setattr(clozn.settings, 'SETTINGS_PATH', str(tmp_path / 'settings.json'))"
        )


@pytest.fixture(autouse=True)
def _required_test_model_router():
    """Route tests use the same mandatory-router shape as the product gateway.

    Individual routing tests replace this double with a real router. The double resolves the substrate
    and engine lazily because many route fixtures install their fakes after pytest fixtures have started.
    """
    from clozn.server import app
    from clozn.server.model_routing import ModelSelection

    if app.MODEL_ROUTER is not None:
        yield
        return

    class TestModelRouter:
        gate = None
        worker_call_tracker = None
        default_model_id = "test-model"

        @staticmethod
        def _pair():
            sub = app.SUB
            engine = getattr(sub, "engine", None) if sub is not None else None
            return sub, engine if engine is not None else app.ENGINE

        @staticmethod
        def _model_id(requested=None):
            return str(requested or app.PUBLIC_MODEL_ID or app.SUBNAME or "test-model")

        def select(self, requested, **_kwargs):
            sub, engine = self._pair()
            return ModelSelection(self._model_id(requested), sub, engine, None)

        def select_control_model(self, requested, **_kwargs):
            sub, engine = self._pair()
            return ModelSelection(self._model_id(requested), sub, engine, None)

        def control_pair(self):
            return self._pair()

        def model_ids(self):
            return [self._model_id()]

        def catalog(self):
            sub, _engine = self._pair()
            digest = getattr(sub, "model_sha256", None) if sub is not None else None
            return [{
                "model_id": self._model_id(),
                "artifact_sha256": str(digest or ("0" * 64)),
                "state": "ready" if sub is not None else "unloaded",
            }]

        def runtime_status(self):
            sub, engine = self._pair()
            ready = sub is not None and engine is not None
            model_id = self._model_id()
            return {
                "default_model_id": model_id,
                "preload_model_ids": [model_id],
                "max_loaded_models": 1,
                "configured_count": 1,
                "resident_count": 1 if ready else 0,
                "models": [{
                    "model_id": model_id,
                    "state": "ready" if ready else "unloaded",
                    "default": True,
                }],
            }

    router = TestModelRouter()
    app.MODEL_ROUTER = router
    try:
        yield
    finally:
        if app.MODEL_ROUTER is router:
            app.MODEL_ROUTER = None


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", "") or ""
    if "model" in markexpr:            # explicit opt-in (`-m model`) or exclusion (`-m "not model"`):
        return                         # let pytest's own -m selection decide
    skip = pytest.mark.skip(reason="model-gated (loads a GPU checkpoint): run with -m model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
