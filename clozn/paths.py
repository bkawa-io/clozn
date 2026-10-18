"""Paths owned by the current Clozn product.

The v1 product deliberately gets its own data root.  Older run journals under
``~/.clozn/runs`` are historical data and are not implicitly imported or
modified by the v1 application.

``CLOZN_DATA_ROOT`` is intentionally resolved by a function rather than only
at import time so tests and embedding applications can inject a disposable
root before constructing a store.  Product modules still expose their
traditional module-level path knobs where callers already rely on them.
"""
from __future__ import annotations

import os


LEGACY_ROOT = os.path.join(os.path.expanduser("~"), ".clozn")
DEFAULT_DATA_ROOT = os.path.join(LEGACY_ROOT, "v1")


def data_root(value: str | os.PathLike[str] | None = None) -> str:
    """Return the absolute root for v1-owned state.

    An explicit argument is useful for tests and embedding applications.  When
    omitted, ``CLOZN_DATA_ROOT`` controls the location; otherwise the isolated
    default ``~/.clozn/v1`` is used.
    """
    configured = value
    if configured is None:
        configured = os.environ.get("CLOZN_DATA_ROOT")
    if configured is None or not str(configured).strip():
        configured = DEFAULT_DATA_ROOT
    return os.path.abspath(os.path.expanduser(os.fspath(configured)))


def data_path(*parts: str, root: str | os.PathLike[str] | None = None) -> str:
    """Return a path below the resolved v1 data root."""
    return os.path.join(data_root(root), *parts)


__all__ = ["DEFAULT_DATA_ROOT", "LEGACY_ROOT", "data_path", "data_root"]
