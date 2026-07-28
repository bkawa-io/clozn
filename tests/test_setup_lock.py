from __future__ import annotations

import os
import time

import pytest

from clozn.setup.errors import LockError
from clozn.setup.lock import SetupLock


def test_lock_acquires_and_releases(tmp_path):
    path = str(tmp_path / ".lock")
    with SetupLock(path):
        assert os.path.isfile(path)
    assert not os.path.isfile(path)


def test_lock_refuses_a_concurrent_holder(tmp_path):
    path = str(tmp_path / ".lock")
    with SetupLock(path):
        with pytest.raises(LockError, match="another `clozn setup`"):
            with SetupLock(path):
                pass


def test_lock_releases_on_exception_inside_the_with_block(tmp_path):
    path = str(tmp_path / ".lock")
    with pytest.raises(RuntimeError):
        with SetupLock(path):
            raise RuntimeError("boom")
    assert not os.path.isfile(path)


def test_lock_steals_a_stale_lock_file(tmp_path, monkeypatch):
    path = str(tmp_path / ".lock")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("old")
    stale_time = time.time() - 3600
    os.utime(path, (stale_time, stale_time))
    with SetupLock(path):
        assert os.path.isfile(path)


def test_lock_creates_its_parent_directory(tmp_path):
    path = str(tmp_path / "nested" / "dir" / ".lock")
    with SetupLock(path):
        assert os.path.isfile(path)
