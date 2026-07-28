"""Model-free checks for the pinned llama.cpp bootstrap's Windows-safe filesystem/Git helpers."""
from __future__ import annotations

import importlib.util
import os
import stat
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "engine" / "core" / "third_party" / "bootstrap_llama.py"
SPEC = importlib.util.spec_from_file_location("clozn_bootstrap_llama", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(bootstrap)


def test_git_long_path_override_is_scoped_to_windows():
    command = bootstrap._git("status", "--short")
    assert command[0] == "git"
    if os.name == "nt":
        assert command[1:3] == ["-c", "core.longpaths=true"]
    else:
        assert "core.longpaths=true" not in command


def test_remove_tree_handles_read_only_files(tmp_path):
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    packed = checkout / "pack.idx"
    packed.write_bytes(b"fixture")
    packed.chmod(stat.S_IREAD)

    bootstrap._remove_tree(str(checkout))

    assert not checkout.exists()
