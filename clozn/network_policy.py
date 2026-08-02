"""Process-wide outbound HTTP policy and privacy-safe attempt ledger.

The product's Python HTTP clients all converge on :func:`urllib.request.urlopen`,
including the bundled engine client.  Installing one wrapper there provides a
pre-connection local-only check and one append-only audit seam without teaching
every caller about policy.  Ledger rows intentionally contain no URL path,
query, headers, request body, response body, or exception message.
"""
from __future__ import annotations

import ipaddress
import json
import os
import secrets
import socket
import threading
import urllib.request
from datetime import datetime, timezone
from urllib.parse import urlsplit

from clozn._io import atomic_write_json


POLICY_PATH = os.path.join(os.path.expanduser("~/.clozn"), "network_policy.json")
LEDGER_PATH = os.path.join(os.path.expanduser("~/.clozn"), "outbound_attempts.jsonl")
POLICY_ENV = "CLOZN_NETWORK_POLICY_PATH"
LEDGER_ENV = "CLOZN_OUTBOUND_LEDGER_PATH"
LOCAL_ONLY_ENV = "CLOZN_LOCAL_ONLY"
SCHEMA_VERSION = "clozn.outbound_attempt.v1"
VERIFY_SCHEMA_VERSION = "clozn.offline_verification.v1"

_APPEND_LOCK = threading.Lock()
_FALSE_VALUES = frozenset({"", "0", "false", "no", "off"})
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


class LocalOnlyViolation(PermissionError):
    """An outbound destination was blocked before urllib could open it."""

    def __init__(self, host: str | None, category: str):
        self.host = host
        self.category = category
        super().__init__(
            f"local-only mode blocked outbound network access to {host or 'an unknown host'} "
            f"({category})"
        )


def _policy_path(environ=None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get(POLICY_ENV) or POLICY_PATH)


def _ledger_path(environ=None) -> str:
    env = os.environ if environ is None else environ
    return str(env.get(LEDGER_ENV) or LEDGER_PATH)


def _env_bool(value) -> bool:
    text = str(value).strip().casefold()
    if text in _FALSE_VALUES:
        return False
    if text in _TRUE_VALUES:
        return True
    # An operator who set the security flag but misspelled its value gets the safe result.
    return True


def local_only_enabled(*, environ=None) -> bool:
    """Return the effective flag; environment overrides the persisted setting.

    A missing policy defaults off for backward compatibility. A present but unreadable
    or malformed policy fails closed rather than silently widening network access.
    """
    env = os.environ if environ is None else environ
    if LOCAL_ONLY_ENV in env:
        return _env_bool(env.get(LOCAL_ONLY_ENV))
    path = _policy_path(env)
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError:
        return False
    except Exception:
        return True
    return bool(value.get("local_only")) if isinstance(value, dict) else True


def _persisted_policy(*, environ=None) -> dict:
    try:
        with open(_policy_path(environ), encoding="utf-8") as handle:
            value = json.load(handle)
    except Exception:
        return {}
    return dict(value) if isinstance(value, dict) else {}


def set_local_only(enabled: bool, *, environ=None) -> dict:
    """Persist the local-only flag without changing environment overrides."""
    if not isinstance(enabled, bool):
        raise TypeError("enabled must be a boolean")
    env = os.environ if environ is None else environ
    path = _policy_path(env)
    previous = _persisted_policy(environ=env)
    now = datetime.now(timezone.utc).isoformat()
    activated_at = previous.get("activated_at") if previous.get("local_only") else None
    document = {
        "schema_version": 1,
        "local_only": enabled,
        "activated_at": (activated_at or now) if enabled else None,
        "updated_at": now,
    }
    atomic_write_json(path, document)
    return {
        "configured": enabled,
        "effective": local_only_enabled(environ=env),
        "environment_override": LOCAL_ONLY_ENV in env,
        "activated_at": document["activated_at"],
        "path": path,
    }


def _destination(target) -> dict:
    if isinstance(target, urllib.request.Request):
        raw = target.full_url
    else:
        raw = str(target)
    try:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.casefold()
        host = parsed.hostname
        host = host.rstrip(".").casefold() if host else None
        port = parsed.port
    except Exception:
        return {"category": "invalid", "scheme": None, "host": None, "port": None}
    if scheme == "file":
        # A host-bearing file URL can open an SMB/NFS-style remote share. Only hostless and explicit
        # localhost file URLs are local; never let a network filesystem bypass the HTTP policy.
        category = "local_file" if host in {None, "localhost"} else "external"
        return {"category": category, "scheme": scheme, "host": host, "port": port}
    if scheme not in {"http", "https"} or not host:
        return {"category": "invalid", "scheme": scheme or None, "host": host, "port": port}
    if host == "localhost" or host.endswith(".localhost"):
        category = "loopback"
    else:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            category = "external"
        else:
            if address.is_loopback:
                category = "loopback"
            elif address.is_private or address.is_link_local:
                category = "private_network"
            else:
                category = "external"
    return {"category": category, "scheme": scheme, "host": host, "port": port}


def _method(target, data) -> str:
    if isinstance(target, urllib.request.Request):
        try:
            return str(target.get_method() or "GET").upper()
        except Exception:
            return "HTTP"
    return "POST" if data is not None else "GET"


def _append_attempt(event: dict, *, environ=None) -> bool:
    """Append one compact JSON line in one OS write; ledger failure never exposes request data."""
    path = _ledger_path(environ)
    payload = (json.dumps(event, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
    try:
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        flags = os.O_APPEND | os.O_CREAT | os.O_WRONLY
        if hasattr(os, "O_BINARY"):
            flags |= os.O_BINARY
        with _APPEND_LOCK:
            fd = os.open(path, flags, 0o600)
            try:
                if os.write(fd, payload) != len(payload):
                    raise OSError("short outbound-ledger append")
                os.fsync(fd)
            finally:
                os.close(fd)
        return True
    except Exception:
        return False


def _event(destination: dict, operation: str, outcome: str, *, local_only: bool,
           error_type: str | None = None, attempt_id: str | None = None) -> dict:
    row = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": attempt_id or ("net_" + secrets.token_hex(8)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "destination_category": destination.get("category"),
        "host": destination.get("host"),
        "port": destination.get("port"),
        "scheme": destination.get("scheme"),
        "operation": operation,
        "outcome": outcome,
        "local_only": bool(local_only),
    }
    if error_type:
        row["error_type"] = error_type
    return row


_installed = getattr(urllib.request.urlopen, "_clozn_network_guard", False)
_transport_urlopen = getattr(
    urllib.request.urlopen, "_clozn_original_urlopen", urllib.request.urlopen)


def guarded_urlopen(url, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, *args, **kwargs):
    """Policy-check and audit one urllib request without retaining sensitive payload data."""
    destination = _destination(url)
    # Derive the operation from the HTTP method instead of accepting caller text; an arbitrary operation
    # label could itself accidentally contain a prompt, token, path, or account identifier.
    op = f"http_{_method(url, data).lower()}"
    local_only = local_only_enabled()
    category = destination.get("category")
    if local_only and category not in {"loopback", "local_file"}:
        _append_attempt(_event(destination, op, "blocked", local_only=True))
        raise LocalOnlyViolation(destination.get("host"), str(category))
    try:
        response = _transport_urlopen(url, data, timeout, *args, **kwargs)
    except Exception as exc:
        _append_attempt(_event(
            destination, op, "failed", local_only=local_only, error_type=type(exc).__name__))
        raise
    _append_attempt(_event(destination, op, "succeeded", local_only=local_only))
    return response


guarded_urlopen._clozn_network_guard = True
guarded_urlopen._clozn_original_urlopen = _transport_urlopen


def install_urllib_guard() -> bool:
    """Install the process-wide wrapper once. Returns True only when this call installed it."""
    global _installed
    if getattr(urllib.request.urlopen, "_clozn_network_guard", False):
        _installed = True
        return False
    urllib.request.urlopen = guarded_urlopen
    _installed = True
    return True


def read_outbound_attempts(limit: int | None = None, *, environ=None) -> list[dict]:
    """Read valid ledger rows oldest-first; malformed/torn lines are ignored."""
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
        raise ValueError("limit must be a non-negative integer or None")
    try:
        with open(_ledger_path(environ), encoding="utf-8") as handle:
            rows = []
            for line in handle:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                if isinstance(row, dict) and row.get("schema_version") == SCHEMA_VERSION:
                    rows.append(row)
    except FileNotFoundError:
        return []
    if limit is None:
        return rows
    return rows[-limit:] if limit else []


def verify_offline(*, since: str | None = None, environ=None) -> dict:
    """Actively verify enforcement, then inspect post-activation non-loopback attempts."""
    effective = local_only_enabled(environ=environ)
    guard_installed = bool(getattr(urllib.request.urlopen, "_clozn_network_guard", False))
    probe_blocked = False
    probe_recorded = False
    probe_error = None
    probe_started = datetime.now(timezone.utc).isoformat()
    if effective and guard_installed:
        try:
            # .invalid is reserved and cannot name a real destination. A working guard blocks this before
            # urllib reaches DNS/socket setup; any other result is an enforcement failure.
            urllib.request.urlopen("https://clozn-offline-verification.invalid/", timeout=0.01)
        except LocalOnlyViolation:
            probe_blocked = True
        except Exception as exc:
            probe_error = type(exc).__name__

    ledger_error = None
    try:
        rows = read_outbound_attempts(environ=environ)
    except OSError as exc:
        rows = []
        ledger_error = type(exc).__name__
    configured = _persisted_policy(environ=environ)
    effective_since = str(since) if since is not None else configured.get("activated_at")
    if effective_since is not None:
        # ``activated_at`` is captured in the same clock domain as ledger timestamps. An allowed
        # request can legitimately receive the exact activation timestamp when the clock resolution
        # is coarse; treat that boundary as pre-activation rather than reporting it as a violation.
        rows = [row for row in rows if str(row.get("timestamp") or "") > effective_since]
    external = [row for row in rows
                if row.get("destination_category") not in {"loopback", "local_file"}]
    violations = [row for row in external if row.get("outcome") != "blocked"]
    blocked = [row for row in external if row.get("outcome") == "blocked"]
    probe_recorded = any(
        row.get("host") == "clozn-offline-verification.invalid"
        and row.get("outcome") == "blocked"
        and str(row.get("timestamp") or "") >= probe_started
        for row in rows
    )
    verified = bool(effective and guard_installed and probe_blocked and probe_recorded
                    and ledger_error is None and not violations)
    return {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "verified": verified,
        "local_only": effective,
        "guard_installed": guard_installed,
        "probe_blocked": probe_blocked,
        "probe_recorded": probe_recorded,
        "probe_error": probe_error,
        "since": effective_since,
        "attempt_count": len(rows),
        "external_attempt_count": len(external),
        "blocked_external_attempt_count": len(blocked),
        "ledger_error": ledger_error,
        "violations": violations,
    }


__all__ = [
    "LEDGER_ENV", "LEDGER_PATH", "LOCAL_ONLY_ENV", "LocalOnlyViolation", "POLICY_ENV",
    "POLICY_PATH", "guarded_urlopen", "install_urllib_guard", "local_only_enabled",
    "read_outbound_attempts", "set_local_only", "verify_offline",
]
