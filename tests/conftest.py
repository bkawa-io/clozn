"""tests conftest -- the `-m model` gate, plus the real-user-data tripwire.

The `-m model` gate: a bare `@pytest.mark.model` test would otherwise RUN in the plain suite -- and
these tests load a real checkpoint onto the GPU and TTT-train a soft prefix for minutes. This makes the
marker actually gate: model-marked tests are skipped unless the run's mark expression names "model".

The tripwire (`_never_write_the_real_user_data`): dozens of test files isolate themselves by
monkeypatching clozn.settings.SETTINGS_PATH (and the card/run stores) at a tmp dir. That isolation is
invisible when it FAILS -- a test that misses the patch just quietly rewrites the developer's own
~/.clozn/studio_settings.json (active profile, sampling, guard config) and still passes. This autouse
fixture hashes the real files before and after every test and fails the test that changed one.

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
_GUARDED = [os.path.join(_CLOZN, name) for name in
            ("studio_settings.json", "studio_memory_cards.json", "studio_library.json",
             "studio_personality.json", "dial_calibration.json")]


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


def pytest_collection_modifyitems(config, items):
    markexpr = config.getoption("markexpr", "") or ""
    if "model" in markexpr:            # explicit opt-in (`-m model`) or exclusion (`-m "not model"`):
        return                         # let pytest's own -m selection decide
    skip = pytest.mark.skip(reason="model-gated (loads a GPU checkpoint): run with -m model")
    for item in items:
        if "model" in item.keywords:
            item.add_marker(skip)
