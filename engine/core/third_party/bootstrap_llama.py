#!/usr/bin/env python3
"""Reproducibly reconstruct the vendored llama.cpp from a PINNED upstream commit + the local CLOZN patches.

`third_party/llama.cpp` is a build dependency that is GITIGNORED (local-only) -- it is NOT committed. Only
this script, PATCHES.md, and patches/*.patch are tracked, so the whole ~340k-line upstream tree stays out
of the repo while the build remains exactly reproducible: this script shallow-clones the pinned tag and
applies our patches on top.

    python engine/core/third_party/bootstrap_llama.py            # clone + patch if missing
    python engine/core/third_party/bootstrap_llama.py --force    # wipe + redo (e.g. after re-pinning)

Re-pinning to a newer llama.cpp: bump COMMIT/TAG below, run with --force, and if a patch no longer applies,
regenerate it (`git -C llama.cpp diff > patches/0001-...patch`) against the new base. See PATCHES.md.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEST = os.path.join(HERE, "llama.cpp")
PATCH_DIR = os.path.join(HERE, "patches")

REPO = "https://github.com/ggml-org/llama.cpp.git"
# Pinned base. Recovered 2026-07-13 by matching the patch's pre-image blob hashes against upstream trees
# (the original checkout kept no .git, so the SHA was reconstructed, not read) and verified with
# `git apply --check`. GGML 0.15.0.
TAG = "b9606"
COMMIT = "88a39274ecf88ba11686acd357b59685b1cbf03d"


def _run(cmd: list[str], cwd: str | None = None) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def _patches() -> list[str]:
    return sorted(f for f in os.listdir(PATCH_DIR) if f.endswith(".patch"))


def _verify() -> bool:
    """Verify that DEST is the pinned checkout with every tracked patch applied."""
    git_dir = os.path.join(DEST, ".git")
    if not os.path.isdir(git_dir):
        print(
            f"ERROR: {DEST} has no Git metadata, so its pinned source revision cannot be verified. "
            "Run with --force to reconstruct it.",
            file=sys.stderr,
        )
        return False

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=DEST, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != COMMIT:
        print(
            f"ERROR: llama.cpp HEAD is {head[:12]}, expected {COMMIT[:12]}. "
            "Run with --force to reconstruct it.",
            file=sys.stderr,
        )
        return False

    for patch in _patches():
        path = os.path.join(PATCH_DIR, patch)
        checked = subprocess.run(
            ["git", "apply", "--check", "--reverse", path],
            cwd=DEST,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        if checked.returncode != 0:
            print(
                f"ERROR: tracked patch is not applied cleanly: {patch}\n{checked.stderr.strip()}",
                file=sys.stderr,
            )
            return False

    checked = subprocess.run(["git", "diff", "--check"], cwd=DEST)
    if checked.returncode != 0:
        print("ERROR: patched llama.cpp tree fails git diff --check", file=sys.stderr)
        return False

    print(f"verified llama.cpp @ {TAG} ({COMMIT[:12]}) + {len(_patches())} tracked patch(es)")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Reconstruct vendored llama.cpp @ pinned commit + CLOZN patches.")
    ap.add_argument("--force", action="store_true", help="remove any existing llama.cpp and re-clone")
    args = ap.parse_args(argv)

    if os.path.isdir(DEST) and os.listdir(DEST) and not args.force:
        return 0 if _verify() else 1
    if os.path.isdir(DEST):
        shutil.rmtree(DEST)

    # Shallow single-tag clone: the pinned source only, no history.
    _run(["git", "clone", "--depth", "1", "--branch", TAG, REPO, DEST])

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=DEST, check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != COMMIT:
        print(f"ERROR: tag {TAG} resolved to {head[:12]}, expected {COMMIT[:12]}. Upstream tag moved -- "
              f"re-pin COMMIT before trusting the build.", file=sys.stderr)
        return 1

    patches = _patches()
    for p in patches:
        path = os.path.join(PATCH_DIR, p)
        _run(["git", "apply", "--check", path], cwd=DEST)      # fail loudly if the base drifted
        _run(["git", "apply", path], cwd=DEST)
        print(f"  applied {p}")

    if not _verify():
        return 1

    # Security hardening: upstream llama.cpp ships a root CLAUDE.md that instructs coding agents to
    # "review AGENTS.md before beginning any work" -- third-party agent directives that Claude Code and
    # similar tools AUTO-READ from a dependency tree. Strip these agent-instruction files so no agent
    # working in THIS repo can be steered by upstream's directives. Neither is build-relevant, so this
    # is safe and reproducible (re-applied on every bootstrap). See docs/RUNTIME_SPLIT.md.
    for _name in ("CLAUDE.md", "AGENTS.md"):
        _p = os.path.join(DEST, _name)
        if os.path.isfile(_p):
            os.remove(_p)
            print(f"  stripped upstream {_name} (agent-instruction file)")

    print(f"\nOK: llama.cpp ready at {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
