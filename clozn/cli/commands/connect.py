"""Safe setup helpers for pointing supported third-party apps at Clozn.

The atomic-write/backup/hash/restore primitives this module uses now live in
clozn.cli.commands._connector (extracted so clozn.cli.commands.adopt's Ollama model linking can reuse
the exact same tested safety properties -- see that module's docstring). This module's own public
functions (configure_aider, undo_aider, cmd_connect, add_subparser) and their observable behavior are
unchanged by that extraction; tests/test_connect_cli.py's six tests exercise them exactly as before.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from clozn._io import atomic_write_json
from clozn.cli.commands._connector import (
    atomic_restore as _atomic_restore,
    atomic_write_text as _atomic_write_text,
    sha256_bytes as _sha256_bytes,
    sha256_path as _sha256_path,
)


_AIDER_KEYS = ("model", "openai-api-base", "openai-api-key")
_TOP_LEVEL_KEY = re.compile(r"^(model|openai-api-base|openai-api-key)\s*:")
_TRANSACTION_SCHEMA = "clozn.connect.transaction.v1"


def _base_url(value: str) -> str:
    parsed = urlsplit(str(value).strip())
    if (parsed.scheme not in {"http", "https"} or not parsed.netloc
            or parsed.query or parsed.fragment or parsed.username or parsed.password):
        raise ValueError("--url must be an http(s) gateway URL without credentials, query, or fragment")
    path = parsed.path.rstrip("/")
    if path not in {"", "/v1"}:
        raise ValueError("--url path must be empty or /v1")
    return urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))


def _aider_model(value: str) -> str:
    model = str(value).strip()
    if not model or any(ch.isspace() for ch in model):
        raise ValueError("--model must be a non-empty model name without whitespace")
    return model if "/" in model else f"openai/{model}"


def render_aider_config(existing: str, *, base_url: str, model: str,
                        api_key: str) -> str:
    """Update only Clozn's three top-level Aider keys, preserving all other YAML text."""
    desired = {
        "model": _aider_model(model),
        "openai-api-base": _base_url(base_url),
        "openai-api-key": str(api_key),
    }
    if not desired["openai-api-key"] or "\n" in desired["openai-api-key"] or "\r" in desired["openai-api-key"]:
        raise ValueError("--api-key must be a non-empty single-line value")
    newline = "\r\n" if "\r\n" in existing else "\n"
    lines = existing.splitlines()
    found: set[str] = set()
    output: list[str] = []
    for line in lines:
        match = _TOP_LEVEL_KEY.match(line)
        if match:
            key = match.group(1)
            if key in found:
                raise ValueError(f"config contains duplicate top-level {key!r} entries")
            found.add(key)
            output.append(f"{key}: {json.dumps(desired[key], ensure_ascii=False)}")
        else:
            output.append(line)
    missing = [key for key in _AIDER_KEYS if key not in found]
    if missing:
        if output and output[-1].strip():
            output.append("")
        output.append("# Added by `clozn connect aider`.")
        output.extend(f"{key}: {json.dumps(desired[key], ensure_ascii=False)}" for key in missing)
    return newline.join(output) + newline


def configure_aider(path: Path, *, base_url: str, model: str, api_key: str,
                    state_path: Path, dry_run: bool = False, now=None) -> dict:
    """Plan or apply one conservative Aider YAML update with backup-before-write."""
    path = Path(os.path.abspath(path.expanduser()))
    if path.is_symlink():
        raise ValueError(f"refusing to replace symlinked config: {path}")
    if path.exists() and not path.is_file():
        raise ValueError(f"config path is not a regular file: {path}")
    existed = path.exists()
    try:
        existing_bytes = path.read_bytes() if existed else b""
        existing = existing_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"config is not UTF-8 and was left unchanged: {path}") from None
    rendered = render_aider_config(existing, base_url=base_url, model=model, api_key=api_key)
    if rendered == existing:
        return {"app": "aider", "path": str(path), "status": "unchanged", "backup": None,
                "base_url": _base_url(base_url), "model": _aider_model(model)}
    report = {"app": "aider", "path": str(path),
              "status": "dry_run" if dry_run else "updated" if existed else "created",
              "backup": None, "base_url": _base_url(base_url), "model": _aider_model(model)}
    if dry_run:
        return report
    state_path = Path(os.path.abspath(state_path.expanduser()))
    if state_path.is_symlink():
        raise ValueError(f"refusing to replace symlinked transaction state: {state_path}")
    backup = None
    prior_mode = None
    if existed:
        prior_mode = stat.S_IMODE(path.stat().st_mode)
        clock = now or datetime.now(timezone.utc)
        stamp = clock.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        backup = path.with_name(f"{path.name}.bak-{stamp}")
        if backup.exists():
            raise ValueError(f"backup path already exists; config was left unchanged: {backup}")
        shutil.copy2(path, backup)
        report["backup"] = str(backup)
    try:
        _atomic_write_text(path, rendered, prior_mode=prior_mode)
        after_sha256 = _sha256_path(path)
        transaction = {
            "schema_version": _TRANSACTION_SCHEMA,
            "app": "aider",
            "target": str(path),
            "target_existed": existed,
            "backup": str(backup) if backup else None,
            "before_sha256": _sha256_bytes(existing_bytes) if existed else None,
            "after_sha256": after_sha256,
            "created_at": (now or datetime.now(timezone.utc)).astimezone(timezone.utc).isoformat(),
        }
        atomic_write_json(str(state_path), transaction, ensure_ascii=False, indent=2, sort_keys=True)
    except BaseException:
        # If transaction recording fails, roll back so an untracked mutation is never left active.
        try:
            if existed and backup is not None and backup.exists():
                _atomic_restore(backup, path)
            elif not existed and path.exists() and _sha256_path(path) == _sha256_bytes(rendered.encode("utf-8")):
                path.unlink()
        except Exception:
            pass
        raise
    return report


def undo_aider(state_path: Path, *, expected_path: Path | None = None) -> dict:
    """Undo the latest recorded Aider transaction only when neither side has drifted."""
    state_path = Path(os.path.abspath(state_path.expanduser()))
    if state_path.is_symlink() or not state_path.is_file():
        raise ValueError("no recorded Aider connect transaction to undo")
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Aider connect transaction is unreadable: {exc}") from None
    required = {"schema_version", "app", "target", "target_existed", "backup",
                "before_sha256", "after_sha256", "created_at"}
    if (not isinstance(state, dict) or set(state) != required
            or state.get("schema_version") != _TRANSACTION_SCHEMA or state.get("app") != "aider"
            or not isinstance(state.get("target"), str)
            or not isinstance(state.get("target_existed"), bool)
            or not isinstance(state.get("after_sha256"), str)):
        raise ValueError("Aider connect transaction has an invalid shape")
    target = Path(state["target"])
    if expected_path is not None:
        expected = Path(os.path.abspath(expected_path.expanduser()))
        if expected != target:
            raise ValueError(f"recorded transaction targets {target}, not {expected}")
    if target.is_symlink() or not target.is_file():
        raise ValueError(f"target no longer exists as the recorded regular file: {target}")
    current_sha256 = _sha256_path(target)
    if current_sha256 != state["after_sha256"]:
        raise ValueError("target changed after `clozn connect`; refusing to overwrite external edits")

    backup_value = state["backup"]
    if state["target_existed"]:
        if not isinstance(backup_value, str) or not isinstance(state["before_sha256"], str):
            raise ValueError("recorded restore transaction is incomplete")
        backup = Path(backup_value)
        if backup.is_symlink() or not backup.is_file():
            raise ValueError(f"recorded backup is unavailable: {backup}")
        if _sha256_path(backup) != state["before_sha256"]:
            raise ValueError("recorded backup changed; refusing unsafe restore")
        _atomic_restore(backup, target)
        status = "restored"
    else:
        if backup_value is not None or state["before_sha256"] is not None:
            raise ValueError("recorded new-file transaction is inconsistent")
        target.unlink()
        status = "removed"
    state_path.unlink()
    return {"app": "aider", "path": str(target), "status": status,
            "backup": backup_value, "base_url": None, "model": None}


def add_subparser(sub):
    parser = sub.add_parser("connect", help="safely point a supported third-party app at Clozn")
    parser.add_argument("app", choices=("aider",))
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1",
                        help="Clozn OpenAI base URL (default http://127.0.0.1:8080/v1)")
    parser.add_argument("--model", default="clozn-local",
                        help="model label Aider sends (default clozn-local; openai/ is added if absent)")
    parser.add_argument("--api-key", default="local-clozn",
                        help="local placeholder API key (default local-clozn)")
    parser.add_argument("--config", default=None,
                        help="explicit app config path (Aider default: ~/.aider.conf.yml)")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="show the planned action without writing")
    action.add_argument("--undo", action="store_true",
                        help="restore the latest backed-up config if it has not changed externally")
    parser.add_argument("--json", action="store_true", help="print a machine-readable result")
    parser.set_defaults(fn=cmd_connect)
    return parser


def cmd_connect(args):
    from clozn.cli import main as ctx
    try:
        path = Path(args.config).expanduser() if args.config else Path.home() / ".aider.conf.yml"
        state_path = Path(ctx.HOME) / "connect" / "aider.json"
        if args.undo:
            report = undo_aider(state_path, expected_path=path if args.config else None)
        else:
            report = configure_aider(path, base_url=args.url, model=args.model,
                                     api_key=args.api_key, state_path=state_path,
                                     dry_run=args.dry_run)
    except (OSError, ValueError) as exc:
        raise ctx.CloznError(f"could not connect {args.app}: {exc}") from None
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    status = report["status"]
    if status == "unchanged":
        print(f"aider is already connected: {report['path']}")
    elif status == "dry_run":
        print(f"would configure aider: {report['path']}")
    elif status in {"restored", "removed"}:
        verb = "restored" if status == "restored" else "removed"
        print(f"{verb} aider config: {report['path']}")
        return 0
    else:
        print(f"configured aider: {report['path']}")
        if report["backup"]:
            print(f"backup: {report['backup']}")
    print(f"endpoint: {report['base_url']}")
    print(f"model: {report['model']}")
    print("next: start Clozn on that port, then run `aider`")
    return 0
