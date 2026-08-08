"""Generic, safe file-integrity primitives shared by commands that adopt files Clozn does not own.

`clozn adopt ollama` (clozn.cli.commands.adopt) is the one caller: it needs to hash a source blob it
did not create (to verify what it adopts), and to copy that blob into Clozn's own model directory
without ever holding a multi-gigabyte file in memory or leaving a partially-written target if the
process is interrupted mid-copy.

This module used to be `_connector.py`, home to a generic third-party-app "Connector" framework
(detect/plan/apply/undo, transactional config-file mutation with backup and drift detection) that
Clozn used to safely patch files like `.aider.conf.yml`. That framework -- and the product feature it
served, `clozn connect` -- was retired: Clozn does not own another application's configuration, so it
no longer edits it. See docs/CAPABILITIES.md. Only the two primitives below survive, because
`clozn adopt ollama` still has a genuine, unrelated need for them: safely handling a *model file*
Clozn itself is about to own a copy of, not mutating a file that belongs to another application.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path


def sha256_path(path: Path) -> str:
    """Chunked read so a multi-GB file (an adopted model) is never loaded whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_copy_file(source: Path, target: Path, *, chunk_size: int = 1 << 20) -> None:
    """Stream-copy `source` to `target` via a same-directory tempfile + os.replace -- `target` is
    either untouched or fully replaced, never partially written. Used for `clozn adopt ollama --copy`,
    where `source` may be a multi-GB model blob."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".copy", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as dst, Path(source).open("rb") as src:
            while chunk := src.read(chunk_size):
                dst.write(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        shutil.copystat(source, temporary, follow_symlinks=True)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise
