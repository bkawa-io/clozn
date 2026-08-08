"""Tests for clozn.cli.commands._fileops: the generic file-integrity primitives `clozn adopt ollama`
uses to hash and copy a model blob it does not own. Formerly part of _connector.py's larger third-party
connector framework (detect/plan/apply/undo config-file mutation); that framework was retired along
with `clozn connect` -- see docs/CAPABILITIES.md. Only these two primitives survived, proven equivalent
to their pre-refactor behavior by these same round-trip assertions.
"""
from __future__ import annotations

from clozn.cli.commands import _fileops as fileops


def test_sha256_path_matches_hashlib(tmp_path):
    import hashlib
    target = tmp_path / "file.txt"
    target.write_bytes(b"hello\n")
    assert fileops.sha256_path(target) == hashlib.sha256(b"hello\n").hexdigest()


def test_atomic_copy_file_produces_byte_identical_independent_file(tmp_path):
    source = tmp_path / "source.bin"
    target = tmp_path / "nested" / "target.bin"
    source.write_bytes(b"model weights" * 1000)
    fileops.atomic_copy_file(source, target)
    assert target.read_bytes() == source.read_bytes()
    assert fileops.sha256_path(target) == fileops.sha256_path(source)


def test_atomic_copy_file_never_leaves_a_partial_target_on_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.bin"
    target = tmp_path / "target.bin"
    source.write_bytes(b"x" * 1000)

    real_replace = __import__("os").replace

    def _boom(*_a, **_k):
        raise OSError("simulated failure before replace")
    monkeypatch.setattr("os.replace", _boom)

    import pytest
    with pytest.raises(OSError):
        fileops.atomic_copy_file(source, target)
    assert not target.exists()
    monkeypatch.setattr("os.replace", real_replace)
