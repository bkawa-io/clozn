"""The product's persisted settings store (``~/.clozn/v1/studio_settings.json`` by default).

Only keys in :data:`SUPPORTED_KEYS` belong to the current product contract. Reads
ignore unknown fields and writes reject unknown keys, so stale or misspelled
configuration can never silently become behavior.

Contract:
  * IO NEVER RAISES. A missing, unreadable, or malformed settings file degrades to the default --
    never to a crashed request. Callers rely on this and do not wrap these calls.
  * WRITES ARE ATOMIC (clozn._io.atomic_write_json): temp-file-then-rename, and a non-serializable
    value raises out of json.dumps BEFORE the real file is opened. So a bad set_setting() can never
    truncate or corrupt the store -- every already-persisted key survives untouched. This is not
    incidental: non-atomic writes to this file were a measured data-loss bug once already.
  * SETTINGS_PATH is a module global so tests can point it at a tmp dir. tests/conftest.py asserts
    it is redirected during any test that writes.

Stdlib only (torch-free), so every model-free consumer -- replay, receipts, the CLI -- can import it.
"""
from __future__ import annotations

import json
import os

from clozn._io import atomic_write_json
from clozn.paths import data_root

SETTINGS_PATH = os.path.join(data_root(), "studio_settings.json")
SUPPORTED_KEYS = frozenset({
    "receipt_privacy",
    "sampling",
    "sample_temperature",
    "sample_top_p",
    "sample_top_k",
    "sample_repeat_penalty",
})


def _load_settings() -> dict:
    """The whole settings dict; {} if missing or unreadable (never raises)."""
    try:
        if not os.path.isfile(SETTINGS_PATH):
            return {}
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {}
        return {key: value for key, value in data.items() if key in SUPPORTED_KEYS}
    except Exception:
        return {}


def get_setting(key: str, default=None):
    """Read one settings key; `default` when missing or unreadable."""
    if key not in SUPPORTED_KEYS:
        return default
    return _load_settings().get(key, default)


def set_setting(key: str, value) -> bool:
    """Persist one settings key (merge-write); False on IO failure (never raises).

    Atomic -- see the module docstring. Other keys are always preserved: this is a read-merge-write,
    not a replace.
    """
    if key not in SUPPORTED_KEYS:
        return False
    try:
        settings = _load_settings()
        settings[key] = value
        atomic_write_json(SETTINGS_PATH, settings)
        return True
    except Exception:
        return False
