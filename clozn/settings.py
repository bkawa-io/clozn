"""settings.py -- the product's persisted key/value store (~/.clozn/studio_settings.json).

This lived at `clozn/memory/mode.py` until the 2026-07-27 memory cut, which is why so much of the
product reached into a module called "memory" to read things that have nothing to do with memory:
`sampling` / `sample_top_k`, `timetravel_budget_mb` / `_cap` / enabled, the run-capture mode, the
receipt-link setting, `memory_strength`. Memory cards are gone; the settings store they happened to
share a file with is not, so it moved here under its real name.

`generation_guard` (a persisted server-wide guard default, retired -- Clozn's concept guard is now
request-local only, via `clozn_guard`) and `selective_generation` (a persisted answer-rewriting default,
retired -- selective-generation is calibration evidence only now, see
`clozn.server.generation_gateway.policy_signal`) are RETIRED keys: no code reads or writes either, but
this module never scrubs a key just because its feature retired, so a pre-retirement install may still
carry it on disk, inertly, forever.

Contract, inherited verbatim from the old module and depended on across the server:
  * IO NEVER RAISES. A missing, unreadable, or malformed settings file degrades to the default --
    never to a crashed request. Callers rely on this and do not wrap these calls.
  * WRITES ARE ATOMIC (clozn._io.atomic_write_json): temp-file-then-rename, and a non-serializable
    value raises out of json.dumps BEFORE the real file is opened. So a bad set_setting() can never
    truncate or corrupt the store -- every already-persisted key survives untouched. This is not
    incidental: non-atomic writes to this file were a measured data-loss bug once already.
  * SETTINGS_PATH is a module global so tests can point it at a tmp dir. tests/conftest.py asserts
    it is redirected during any test that writes, so a missed monkeypatch fails loudly instead of
    quietly editing the developer's real ~/.clozn.

Stdlib only (torch-free), so every model-free consumer -- replay, receipts, the CLI -- can import it.
"""
from __future__ import annotations

import json
import os

from clozn._io import atomic_write_json

_CLOZN = os.path.expanduser("~/.clozn")
SETTINGS_PATH = os.path.join(_CLOZN, "studio_settings.json")


def _load_settings() -> dict:
    """The whole settings dict; {} if missing or unreadable (never raises)."""
    try:
        if not os.path.isfile(SETTINGS_PATH):
            return {}
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_setting(key: str, default=None):
    """Read one settings key; `default` when missing or unreadable."""
    return _load_settings().get(key, default)


def set_setting(key: str, value) -> bool:
    """Persist one settings key (merge-write); False on IO failure (never raises).

    Atomic -- see the module docstring. Other keys are always preserved: this is a read-merge-write,
    not a replace.
    """
    try:
        settings = _load_settings()
        settings[key] = value
        atomic_write_json(SETTINGS_PATH, settings)
        return True
    except Exception:
        return False
