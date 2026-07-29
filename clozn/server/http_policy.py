"""Small HTTP safety policy shared by JSON and streaming responses.

The product binds loopback by default, but a browser on an unrelated website can still
attempt requests to localhost.  CORS is therefore loopback-only unless the operator opts
additional exact origins in through ``CLOZN_ORIGINS``.  Request bodies are bounded before
they are read so a malformed or hostile client cannot make the gateway allocate without
limit.
"""
from __future__ import annotations

import ipaddress
import os
import select
import socket
from urllib.parse import urlsplit


DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_ALLOW_HEADERS = (
    "Accept, Authorization, Content-Type, OpenAI-Organization, OpenAI-Project, "
    "X-Requested-With"
)


def max_request_bytes() -> int:
    """Configured request-body ceiling, falling back safely on invalid input."""
    raw = os.environ.get("CLOZN_MAX_REQUEST_BYTES", "").strip()
    if not raw:
        return DEFAULT_MAX_REQUEST_BYTES
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES
    return value if value > 0 else DEFAULT_MAX_REQUEST_BYTES


def _configured_origins() -> set[str]:
    return {
        value.strip().rstrip("/")
        for value in os.environ.get("CLOZN_ORIGINS", "").split(",")
        if value.strip()
    }


def origin_allowed(origin: str | None) -> bool:
    """Allow loopback browser origins plus explicit exact operator opt-ins."""
    if not origin:
        return True
    if any(ord(char) < 0x20 or ord(char) == 0x7f for char in origin):
        return False
    origin = origin.strip().rstrip("/")
    configured = _configured_origins()
    if "*" in configured or origin in configured:
        return True
    try:
        parsed = urlsplit(origin)
        parsed.port  # validate a present port rather than accepting malformed values
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
        ):
            return False
        host = parsed.hostname.lower()
        if host == "localhost" or host.endswith(".localhost"):
            return True
        return ipaddress.ip_address(host).is_loopback
    except (ValueError, TypeError):
        return False


def request_origin(handler) -> str | None:
    headers = getattr(handler, "headers", None)
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return None
    value = getter("Origin")
    return str(value) if value else None


def client_gone(handler) -> bool:
    """Best-effort probe: True iff the client's TCP connection has already closed.

    WHY: RequestGate (request_gate.py) bounds how long an admitted-but-not-yet-running POST waits for its
    turn, but that bound (CLOZN_QUEUE_TIMEOUT, 600s by default) was previously the ONLY way a queued
    request ever left the queue -- a client that closed its tab while queued behind other generation
    requests still occupied a bounded queue slot for up to 10 minutes, starving real requests out of that
    same bound. This probe lets `do_POST` pass RequestGate.acquire a `cancel_check` so a vanished request's
    slot is freed within one poll tick instead.

    Mechanism: select() on the raw socket with a zero timeout is non-blocking and tells us whether there is
    unread data OR an EOF pending; MSG_PEEK then reads without consuming so we can tell the two apart --
    b"" means the peer sent a FIN (closed), any bytes means the client is still there (e.g. HTTP pipelining
    or stray bytes, none of our business to consume here). Both select() and MSG_PEEK are supported on
    Windows' WinSock as well as POSIX, so this works unmodified on this project's Windows dev environment.

    Fails CLOSED toward "still connected": a bare/mock handler with no real socket (every unit test that
    drives do_POST via object.__new__(H) -- raw_gateway_request, the M5 bridge tests, etc.) has no
    `.connection` attribute at all and reads as connected (False), so this probe is silently inert unless a
    real socket is present. A live-request false-cancel (treating a connected client as gone) is a real
    outage; a dead-request false-negative (treating a gone client as connected) just falls back to the
    existing wait_timeout bound -- so ambiguity always resolves toward "connected"."""
    sock = getattr(handler, "connection", None)
    if sock is None:
        return False
    try:
        ready, _, _ = select.select([sock], [], [], 0)
        if not ready:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True     # the socket itself is unusable (closed fd, bad state) -- nothing to serve either way
    except Exception:
        return False    # an unexpected probe failure must never wrongly cancel a live request


def send_cors_headers(handler) -> bool:
    """Write CORS response headers when this request's origin is allowed."""
    origin = request_origin(handler)
    if not origin or not origin_allowed(origin):
        return not origin
    handler.send_header("Access-Control-Allow-Origin", origin)
    handler.send_header("Vary", "Origin")
    handler.send_header(
        "Access-Control-Expose-Headers",
        "X-Clozn-Run-Id, X-Clozn-Influence-Cache",
    )
    return True
