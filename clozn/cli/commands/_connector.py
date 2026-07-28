"""Shared transactional-mutation primitives, plus the generic `Connector` interface, for every clozn
command that safely edits state outside its own control: a third-party app's config file
(clozn.cli.commands.connect) or Clozn's own model directory when adopting an external model
(clozn.cli.commands.adopt).

WHY THIS EXISTS (roadmap rule 4, "transactional mutation" -- notes/agent_roadmap/README.md)
--------------------------------------------------------------------------------------------
Any command that mutates something the user did not ask this exact command to create must support
dry-run, backup, drift detection, and safe undo. `clozn.cli.commands.connect` proved this pattern first
(atomic write, pre-write backup, sha256-before/after, drift-checked restore -- see
tests/test_connect_cli.py's six tests) for one hardcoded app, Aider. The four primitives below are
lifted out of connect.py UNCHANGED -- same bytes, same behavior, same call sites in connect.py now
delegating here -- so a second consumer (adopt.py) does not reimplement (and inevitably drift from) the
exact safety properties connect.py's tests already prove. This mirrors clozn.models.inventory's own
extraction of clozn.cli.commands.models's former private _model_dirs/_scan_models (see that module's
docstring for the identical reasoning): connect.py's public functions (configure_aider/undo_aider) and
their tests are UNTOUCHED by this refactor.

THE GENERIC `Connector` INTERFACE (notes/agent_roadmap/11-adopt-ollama.md "Generic connector framework")
-----------------------------------------------------------------------------------------------------
    class Connector:
        id: str
        def detect(self) -> Detection
        def plan(self, **kwargs) -> MutationPlan
        def apply(self, plan) -> Transaction
        def undo(self, transaction) -> UndoResult

`AiderConnector` below is the first (and, in this release, only) implementation: a thin adapter over
connect.py's already-tested `configure_aider`/`undo_aider`, not a reimplementation of Aider's YAML
patching. `clozn adopt ollama --connect aider` (clozn.cli.commands.adopt) uses this adapter so an
adoption's optional client-config step reuses the exact same tested mutation path as `clozn connect
aider` -- the CLI-facing `clozn connect` command itself is intentionally left calling configure_aider/
undo_aider directly (unchanged), since widening it to route through this adapter carries no behavior
change and no test benefit for this release.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# --------------------------------------------------------------------- primitives (moved from connect.py)

def atomic_write_text(path: Path, text: str, *, prior_mode: int | None) -> None:
    """Write `text` to `path` via a same-directory tempfile + fsync + os.replace -- `path` is either
    untouched or fully replaced, never partially written or truncated."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        if prior_mode is not None:
            os.chmod(temporary, prior_mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    """Chunked read so a multi-GB file (an adopted model, in adopt.py's case) is never loaded whole."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_restore(source: Path, target: Path) -> None:
    """Copy `source` over `target` via a same-directory tempfile + os.replace -- used by undo paths so a
    restore is itself all-or-nothing, never a partially-written target."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".restore", dir=target.parent)
    os.close(fd)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, target)
    except BaseException:
        try:
            os.remove(temporary)
        except OSError:
            pass
        raise


# ------------------------------------------------------------------------------------- Connector interface

@dataclass
class Detection:
    """What `Connector.detect()` found about whether an app is present and configurable at all.
    "Detected" never implies "safe to modify" (spec's own wording) -- apply() still runs its own
    dry-run-shaped safety checks regardless of what detect() reports."""
    installed: bool
    app: str
    executable_path: str | None = None
    config_path: str | None = None
    note: str | None = None


@dataclass
class MutationPlan:
    """What `Connector.apply()` WOULD do, without doing it -- the dry-run-shaped preview."""
    app: str
    status: str
    target: str | None
    details: dict = field(default_factory=dict)


@dataclass
class Transaction:
    """What `Connector.apply()` DID. `state_path` is where undo() will look for the recorded
    transaction; `report` is the same dict shape connect.cmd_connect already prints/JSON-dumps."""
    app: str
    state_path: str | None
    report: dict = field(default_factory=dict)


@dataclass
class UndoResult:
    app: str
    status: str
    report: dict = field(default_factory=dict)


class Connector:
    """Base class for a transactional third-party-app connector. Subclasses implement all four methods;
    this base class carries no behavior of its own (kept as a plain class, not `abc.ABC`, so a subclass
    that only implements the methods it needs -- e.g. detect()-only during development -- fails with a
    clear AttributeError/NotImplementedError at the call site rather than at import time, matching this
    codebase's general "explicit over clever" preference elsewhere)."""

    id: str = ""

    def detect(self) -> Detection:
        raise NotImplementedError

    def plan(self, **kwargs) -> MutationPlan:
        raise NotImplementedError

    def apply(self, plan: MutationPlan, **kwargs) -> Transaction:
        raise NotImplementedError

    def undo(self, **kwargs) -> UndoResult:
        raise NotImplementedError


class AiderConnector(Connector):
    """Thin `Connector` adapter over connect.py's configure_aider/undo_aider. Every actual file
    operation (backup, atomic write, hash, restore) happens inside those two functions, unchanged;
    this class only shapes their existing inputs/outputs into the generic interface."""

    id = "aider"

    def __init__(self, *, config_path: Path | None = None):
        self._config_path = config_path or (Path.home() / ".aider.conf.yml")

    def detect(self) -> Detection:
        exe = shutil.which("aider")
        return Detection(
            installed=exe is not None, app=self.id, executable_path=exe,
            config_path=str(self._config_path),
            note=None if exe else "no 'aider' executable found on PATH")

    def plan(self, *, base_url: str, model: str, api_key: str, state_path: Path, **_ignored) -> MutationPlan:
        from clozn.cli.commands.connect import configure_aider
        report = configure_aider(self._config_path, base_url=base_url, model=model, api_key=api_key,
                                 state_path=state_path, dry_run=True)
        return MutationPlan(app=self.id, status=report["status"], target=report["path"], details=report)

    def apply(self, plan: MutationPlan | None = None, *, base_url: str, model: str, api_key: str,
             state_path: Path, **_ignored) -> Transaction:
        from clozn.cli.commands.connect import configure_aider
        report = configure_aider(self._config_path, base_url=base_url, model=model, api_key=api_key,
                                 state_path=state_path, dry_run=False)
        return Transaction(app=self.id, state_path=str(state_path), report=report)

    def undo(self, *, state_path: Path, **_ignored) -> UndoResult:
        from clozn.cli.commands.connect import undo_aider
        report = undo_aider(state_path, expected_path=self._config_path)
        return UndoResult(app=self.id, status=report["status"], report=report)
