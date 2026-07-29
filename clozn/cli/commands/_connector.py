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
import json
import os
import shutil
import stat
import tempfile
from datetime import datetime, timezone
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


def atomic_copy_file(source: Path, target: Path, *, chunk_size: int = 1 << 20) -> None:
    """Stream-copy `source` to `target` via a same-directory tempfile + os.replace. Used for `clozn
    adopt ollama --copy`, where `source` may be a multi-GB model blob: this never holds the whole file
    in memory (unlike shutil.copy2 called on a Path object, which is fine size-wise -- it also streams
    -- but is kept out of this module's own vocabulary so every "did this write land atomically" answer
    for adopt.py routes through the same primitive family as its other three operations)."""
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


def _plan_expectations(
    plan: MutationPlan,
    *,
    app: str,
    target: Path,
) -> tuple[bool, str | None, str]:
    """Validate an apply plan and return its compare-and-swap expectations."""
    normalized_target = str(Path(os.path.abspath(target.expanduser())))
    if not isinstance(plan, MutationPlan):
        raise ValueError(f"{app} apply requires the MutationPlan returned by plan()")
    if plan.app != app or plan.target != normalized_target:
        raise ValueError(f"{app} mutation plan does not match target {normalized_target}")
    prior_exists = plan.details.get("expected_prior_exists")
    prior_sha256 = plan.details.get("expected_prior_sha256")
    after_sha256 = plan.details.get("after_sha256")
    if not isinstance(prior_exists, bool) or not isinstance(after_sha256, str):
        raise ValueError(f"{app} mutation plan is missing integrity expectations")
    if prior_exists != isinstance(prior_sha256, str):
        raise ValueError(f"{app} mutation plan has inconsistent prior-state expectations")
    return prior_exists, prior_sha256, after_sha256


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

    def apply(self, plan: MutationPlan, *, base_url: str, model: str, api_key: str,
             state_path: Path, **_ignored) -> Transaction:
        from clozn.cli.commands.connect import configure_aider
        prior_exists, prior_sha256, after_sha256 = _plan_expectations(
            plan, app=self.id, target=self._config_path
        )
        report = configure_aider(self._config_path, base_url=base_url, model=model, api_key=api_key,
                                 state_path=state_path, dry_run=False,
                                 expected_prior_exists=prior_exists,
                                 expected_prior_sha256=prior_sha256,
                                 expected_after_sha256=after_sha256)
        return Transaction(app=self.id, state_path=str(state_path), report=report)

    def undo(self, *, state_path: Path, **_ignored) -> UndoResult:
        from clozn.cli.commands.connect import undo_aider
        report = undo_aider(state_path, expected_path=self._config_path)
        return UndoResult(app=self.id, status=report["status"], report=report)


def _render_env(existing: str, values: dict[str, str], *, app: str) -> str:
    """Replace only the connector-owned dotenv keys and preserve all other text."""
    normalized = {}
    for key, raw in values.items():
        value = str(raw)
        if not key or not key.replace("_", "").isalnum() or not key.upper() == key:
            raise ValueError(f"invalid environment key for {app}: {key!r}")
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} must be a single-line value")
        normalized[key] = value
    newline = "\r\n" if "\r\n" in existing else "\n"
    lines = existing.splitlines()
    found = set()
    out = []
    for line in lines:
        key, sep, _value = line.partition("=")
        if sep and key in normalized:
            if key in found:
                raise ValueError(f"config contains duplicate {key!r} entries")
            found.add(key)
            out.append(f"{key}={json.dumps(normalized[key], ensure_ascii=False)}")
        else:
            out.append(line)
    missing = [key for key in normalized if key not in found]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append(f"# Added by `clozn connect {app}`.")
        out.extend(f"{key}={json.dumps(normalized[key], ensure_ascii=False)}" for key in missing)
    return newline.join(out) + newline


def _configure_env(
    path: Path,
    *,
    app: str,
    values: dict[str, str],
    state_path: Path,
    dry_run: bool,
    expected_prior_exists: bool | None = None,
    expected_prior_sha256: str | None = None,
    expected_after_sha256: str | None = None,
) -> dict:
    from clozn._io import atomic_write_json

    path = Path(os.path.abspath(path.expanduser()))
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlinked config: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"config path is not a regular file: {path}")
    existed = path.exists()
    existing_bytes = path.read_bytes() if existed else b""
    prior_sha256 = sha256_bytes(existing_bytes) if existed else None
    if expected_prior_exists is not None:
        if existed != expected_prior_exists or (
            existed and prior_sha256 != expected_prior_sha256
        ):
            raise ValueError(
                f"{app} config changed since preview; refusing to apply a stale plan"
            )
    try:
        existing = existing_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"config is not UTF-8 and was left unchanged: {path}") from None
    rendered = _render_env(existing, values, app=app)
    after_sha256 = sha256_bytes(rendered.encode("utf-8"))
    if expected_after_sha256 is not None and after_sha256 != expected_after_sha256:
        raise ValueError(
            f"{app} requested configuration differs from preview; refusing to apply"
        )
    integrity = {"expected_prior_exists": existed, "after_sha256": after_sha256}
    if prior_sha256 is not None:
        integrity["expected_prior_sha256"] = prior_sha256
    if rendered == existing:
        return {
            "app": app, "path": str(path), "status": "unchanged",
            "variables": sorted(values), **integrity,
        }
    report = {
        "app": app, "path": str(path),
        "status": "dry_run" if dry_run else "updated" if existed else "created",
        "variables": sorted(values), **integrity,
    }
    if dry_run:
        return report
    state_path = Path(os.path.abspath(state_path.expanduser()))
    if state_path.is_symlink():
        raise ValueError(f"refusing to replace symlinked transaction state: {state_path}")
    backup = None
    prior_mode = None
    if existed:
        prior_mode = stat.S_IMODE(path.stat().st_mode)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        if backup.exists():
            raise ValueError(f"backup path already exists; config was left unchanged: {backup}")
        shutil.copy2(path, backup)
        report["backup"] = str(backup)
    try:
        atomic_write_text(path, rendered, prior_mode=prior_mode)
        after = sha256_path(path)
        transaction = {
            "schema_version": "clozn.connect.transaction.v1",
            "app": app,
            "target": str(path),
            "target_existed": existed,
            "after_sha256": after,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if backup is not None:
            transaction["backup"] = str(backup)
            transaction["before_sha256"] = prior_sha256
        atomic_write_json(str(state_path), transaction, ensure_ascii=False, indent=2, sort_keys=True)
    except BaseException:
        try:
            if existed and backup is not None and backup.exists():
                atomic_restore(backup, path)
            elif not existed and path.exists():
                path.unlink()
        except Exception:
            pass
        raise
    report["after_sha256"] = after
    return report


def _undo_env(state_path: Path, *, app: str, expected_path: Path | None = None) -> dict:
    state_path = Path(os.path.abspath(state_path.expanduser()))
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError(f"no recorded {app} connect transaction to undo")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"{app} connect transaction is unreadable: {exc}") from None
    required = {
        "schema_version", "app", "target", "target_existed",
        "after_sha256", "created_at",
    }
    allowed = required | {"backup", "before_sha256"}
    if (
        not isinstance(state, dict)
        or not required.issubset(state)
        or not set(state).issubset(allowed)
        or state.get("schema_version") != "clozn.connect.transaction.v1"
        or state.get("app") != app
        or not isinstance(state.get("target"), str)
        or not isinstance(state.get("target_existed"), bool)
        or not isinstance(state.get("after_sha256"), str)
    ):
        raise ValueError(f"{app} connect transaction has an invalid shape")
    target = Path(state["target"])
    if expected_path is not None and Path(os.path.abspath(expected_path.expanduser())) != target:
        raise ValueError(f"recorded transaction targets {target}, not {expected_path}")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"target no longer exists as the recorded regular file: {target}")
    if sha256_path(target) != state["after_sha256"]:
        raise ValueError("target changed after `clozn connect`; refusing to overwrite external edits")
    if state["target_existed"]:
        backup = Path(state["backup"]) if isinstance(state.get("backup"), str) else None
        if (
            backup is None or backup.is_symlink() or not backup.is_file()
            or sha256_path(backup) != state.get("before_sha256")
        ):
            raise ValueError("recorded backup is unavailable or changed; refusing unsafe restore")
        atomic_restore(backup, target)
        status = "restored"
    else:
        if state.get("backup") is not None or state.get("before_sha256") is not None:
            raise ValueError("recorded new-file transaction is inconsistent")
        target.unlink()
        status = "removed"
    state_path.unlink()
    report = {"app": app, "path": str(target), "status": status}
    if isinstance(state.get("backup"), str):
        report["backup"] = state["backup"]
    return report


class EnvFileConnector(Connector):
    """Transactional, sourceable dotenv output for clients with URL overrides."""

    values_kind = "openai"

    def __init__(self, *, config_path: Path):
        self._config_path = config_path

    def detect(self) -> Detection:
        return Detection(
            installed=True,
            app=self.id,
            config_path=str(self._config_path),
            note="portable environment-file connector",
        )

    def _values(self, *, base_url: str, model: str, api_key: str) -> dict[str, str]:
        from clozn.cli.commands.connect import _base_url
        normalized = _base_url(base_url)
        if self.values_kind == "open-webui":
            return {
                "OPENAI_API_BASE_URLS": normalized,
                "OPENAI_API_KEYS": str(api_key),
                "DEFAULT_MODELS": str(model),
            }
        if self.values_kind == "ollama-sdk":
            return {
                "OLLAMA_HOST": normalized[:-3] if normalized.endswith("/v1") else normalized,
                "OLLAMA_MODEL": str(model),
            }
        return {
            "OPENAI_BASE_URL": normalized,
            "OPENAI_API_KEY": str(api_key),
            "OPENAI_MODEL": str(model),
        }

    def plan(self, *, base_url: str, model: str, api_key: str, state_path: Path, **_ignored):
        report = _configure_env(
            self._config_path,
            app=self.id,
            values=self._values(base_url=base_url, model=model, api_key=api_key),
            state_path=state_path,
            dry_run=True,
        )
        return MutationPlan(app=self.id, status=report["status"], target=report["path"], details=report)

    def apply(self, plan: MutationPlan, *, base_url: str, model: str, api_key: str,
              state_path: Path, **_ignored):
        prior_exists, prior_sha256, after_sha256 = _plan_expectations(
            plan, app=self.id, target=self._config_path
        )
        report = _configure_env(
            self._config_path,
            app=self.id,
            values=self._values(base_url=base_url, model=model, api_key=api_key),
            state_path=state_path,
            dry_run=False,
            expected_prior_exists=prior_exists,
            expected_prior_sha256=prior_sha256,
            expected_after_sha256=after_sha256,
        )
        return Transaction(app=self.id, state_path=str(state_path), report=report)

    def undo(self, *, state_path: Path, **_ignored):
        report = _undo_env(state_path, app=self.id, expected_path=self._config_path)
        return UndoResult(app=self.id, status=report["status"], report=report)


class GenericOpenAIConnector(EnvFileConnector):
    id = "openai-env"

    def __init__(self, *, config_path: Path | None = None):
        super().__init__(
            config_path=config_path or Path.home() / ".config" / "clozn" / "openai.env"
        )


class OpenWebUIConnector(EnvFileConnector):
    id = "open-webui"
    values_kind = "open-webui"

    def __init__(self, *, config_path: Path | None = None):
        super().__init__(
            config_path=config_path or Path.home() / ".open-webui" / "clozn.env"
        )


class OllamaSDKConnector(EnvFileConnector):
    id = "ollama-sdk"
    values_kind = "ollama-sdk"

    def __init__(self, *, config_path: Path | None = None):
        super().__init__(
            config_path=config_path or Path.home() / ".config" / "clozn" / "ollama-sdk.env"
        )


def connector_for(app: str, *, config_path: Path | None = None) -> Connector:
    choices = {
        "aider": AiderConnector,
        "openai-env": GenericOpenAIConnector,
        "open-webui": OpenWebUIConnector,
        "ollama-sdk": OllamaSDKConnector,
    }
    factory = choices.get(str(app))
    if factory is None:
        raise ValueError(f"unsupported connector {app!r}; choose one of: {', '.join(sorted(choices))}")
    return factory(config_path=config_path)
