#!/usr/bin/env python3
"""Reproducibly reconstruct a pristine checkout of pinned upstream llama.cpp.

`third_party/llama.cpp` is a build dependency that is GITIGNORED (local-only) -- it is NOT committed. The
whole upstream tree stays out of the repo while the build remains exactly reproducible: this script
shallow-clones the pinned tag and verifies that the checkout remains pristine. Git long-path support is
enabled per command on Windows so the same pin can be reconstructed below an ordinary user workspace
without changing global Git configuration.

    python engine/core/third_party/bootstrap_llama.py            # clone/verify the pinned checkout
    python engine/core/third_party/bootstrap_llama.py --force    # wipe + redo (e.g. after re-pinning)

Re-pinning to a newer llama.cpp: update TAG/COMMIT below, run with --force, and rebuild the engine.
"""
from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "llama.cpp")

REPO = "https://github.com/ggml-org/llama.cpp.git"
# Pinned upstream base. GGML 0.15.0.
TAG = "b9606"
COMMIT = "88a39274ecf88ba11686acd357b59685b1cbf03d"


def _git(*args: str) -> list[str]:
    """Git command with checkout-local Windows long-path support.

    The pinned llama.cpp tree contains UI source paths that exceed Git for Windows' default 260-character
    checkout limit when Clozn itself lives below an ordinary user workspace. Keep the override scoped to
    these commands instead of requiring or mutating a developer's global Git configuration.
    """
    return ["git", *(["-c", "core.longpaths=true"] if os.name == "nt" else []), *args]


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _remove_tree(path: str) -> None:
    """Remove a disposable checkout, including read-only Git pack/index files on Windows."""
    def _make_writable_and_retry(func, failed_path, _exc_info):
        os.chmod(failed_path, stat.S_IWRITE)
        func(failed_path)

    shutil.rmtree(path, onerror=_make_writable_and_retry)


def _verify() -> bool:
    """Verify that DEST is a clean checkout at the pinned upstream commit."""
    git_dir = os.path.join(DEST, ".git")
    if not os.path.isdir(git_dir):
        print(
            f"ERROR: {DEST} has no Git metadata, so its pinned source revision cannot be verified. "
            "Run with --force to reconstruct it.",
            file=sys.stderr,
        )
        return False

    head_result = subprocess.run(
        _git("rev-parse", "HEAD"), cwd=DEST, check=False, capture_output=True, text=True
    )
    if head_result.returncode != 0:
        print(
            f"ERROR: {DEST} has invalid Git metadata and its pinned source revision cannot be verified. "
            "Run with --force to reconstruct it.",
            file=sys.stderr,
        )
        return False
    head = head_result.stdout.strip()
    if head != COMMIT:
        print(
            f"ERROR: llama.cpp HEAD is {head[:12]}, expected {COMMIT[:12]}. "
            "Run with --force to reconstruct it.",
            file=sys.stderr,
        )
        return False

    status = subprocess.run(
        _git("status", "--porcelain", "--untracked-files=all"),
        cwd=DEST,
        check=False,
        capture_output=True,
        text=True,
    )
    if status.returncode != 0:
        print(
            "ERROR: could not inspect the llama.cpp worktree. Run with --force to reconstruct it.",
            file=sys.stderr,
        )
        return False
    if status.stdout:
        print(
            "ERROR: llama.cpp checkout is modified, deleted, or contains untracked files. "
            "Run with --force to reconstruct the pristine dependency.",
            file=sys.stderr,
        )
        return False

    print(f"verified pristine llama.cpp @ {TAG} ({COMMIT})")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Clone or verify pristine upstream llama.cpp at the pinned commit.")
    ap.add_argument("--force", action="store_true", help="remove any existing llama.cpp and re-clone")
    args = ap.parse_args(argv)

    if os.path.isdir(DEST) and os.listdir(DEST) and not args.force:
        return 0 if _verify() else 1
    if os.path.isdir(DEST):
        _remove_tree(DEST)

    # Shallow single-tag clone: the pinned source only, no history.
    _run(_git("clone", "--depth", "1", "--branch", TAG, REPO, DEST))

    head = subprocess.run(_git("rev-parse", "HEAD"), cwd=DEST, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        print(f"ERROR: tag {TAG} resolved to {head[:12]}, expected {COMMIT[:12]}. Upstream tag moved -- "
              f"re-pin COMMIT before trusting the build.", file=sys.stderr)
        return 1

    if not _verify():
        return 1

    print(f"\nOK: llama.cpp ready at {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
