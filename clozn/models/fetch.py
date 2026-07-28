"""Secure resolver for one artifact pinned by ``clozn.model-lock.v1``.

The lockfile parser remains network-free.  This module is the separate, explicit
network seam used by ``clozn model-lock fetch``: it streams one selected role to
an immutable SHA-256-keyed path, verifies it before promotion, and re-verifies
every cache hit.

Only stdlib modules are used.  Production URLs must be HTTPS.  Plain HTTP is
available solely through the explicit ``allow_loopback_http`` function argument
so model-free tests can use an in-process loopback server without weakening the
CLI's production policy.
"""
from __future__ import annotations

import hashlib
import os
import socket
import tempfile
import urllib.error
import urllib.request
from urllib.parse import urljoin, urlsplit

from clozn.models.lockfile import LockfileError, load_lockfile, pinned_model

DEFAULT_MAX_REDIRECTS = 5
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_CHUNK_SIZE = 1 << 20
USER_AGENT = "clozn-model-lock/0.1"


class ModelFetchError(RuntimeError):
    """A pinned artifact could not be fetched and verified safely."""


def _is_loopback_host(host: str | None) -> bool:
    if not host:
        return False
    host = host.rstrip(".").casefold()
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        import ipaddress

        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_transport_url(url: str, *, allow_loopback_http: bool) -> str:
    """Return the normalized scheme or refuse the URL without echoing it.

    Userinfo is rejected even over HTTPS.  Authentication belongs in a
    credential-aware transport seam, not in a checked-in lockfile URL where it
    could leak through process listings, exceptions, or CI logs.
    """
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.casefold()
        host = parsed.hostname
        has_userinfo = parsed.username is not None or parsed.password is not None
        # Force validation of a malformed/non-numeric port here.
        _ = parsed.port
    except (TypeError, ValueError):
        raise ModelFetchError("model URL is malformed") from None
    if has_userinfo:
        raise ModelFetchError("model URL must not contain embedded credentials")
    if scheme == "https" and host:
        return scheme
    if allow_loopback_http and scheme == "http" and _is_loopback_host(host):
        return scheme
    raise ModelFetchError("model URL must use HTTPS")


class _SecureRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Validate every redirect target before urllib opens it."""

    def __init__(self, *, max_redirects: int, allow_loopback_http: bool):
        super().__init__()
        self._max_redirects = max_redirects
        self._allow_loopback_http = allow_loopback_http
        self._redirects = 0

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        self._redirects += 1
        if self._redirects > self._max_redirects:
            raise ModelFetchError(
                f"model download exceeded the redirect limit ({self._max_redirects})")
        target = urljoin(req.full_url, newurl)
        source_scheme = _validate_transport_url(
            req.full_url, allow_loopback_http=self._allow_loopback_http)
        target_scheme = _validate_transport_url(
            target, allow_loopback_http=self._allow_loopback_http)
        if source_scheme == "https" and target_scheme != "https":
            raise ModelFetchError("refusing model download redirect from HTTPS to a weaker scheme")
        return super().redirect_request(req, fp, code, msg, headers, target)


def _policy_guarded_opener(redirect_handler: _SecureRedirectHandler):
    """Build a redirect-aware opener without bypassing Clozn's outbound policy.

    A custom opener is required to enforce redirect policy before connection,
    while the process-wide ``guarded_urlopen`` owns urllib's default opener.
    Wrap this private opener with the same classifier, local-only decision, and
    privacy-safe ledger event helpers. Redirects recursively call their parent
    opener, so every hop passes through this wrapper before it is opened.
    """
    from clozn import network_policy

    opener = urllib.request.build_opener(redirect_handler)
    transport_open = opener.open

    def guarded_open(target, data=None, timeout=socket._GLOBAL_DEFAULT_TIMEOUT):
        destination = network_policy._destination(target)
        operation = f"http_{network_policy._method(target, data).lower()}"
        local_only = network_policy.local_only_enabled()
        category = destination.get("category")
        if local_only and category not in {"loopback", "local_file"}:
            network_policy._append_attempt(network_policy._event(
                destination, operation, "blocked", local_only=True))
            raise network_policy.LocalOnlyViolation(
                destination.get("host"), str(category))
        try:
            response = transport_open(target, data=data, timeout=timeout)
        except Exception as exc:
            network_policy._append_attempt(network_policy._event(
                destination, operation, "failed", local_only=local_only,
                error_type=type(exc).__name__))
            raise
        network_policy._append_attempt(network_policy._event(
            destination, operation, "succeeded", local_only=local_only))
        return response

    opener.open = guarded_open
    return opener


def _verified_file(path: str, expected_sha256: str, expected_size: int | None) -> tuple[bool, int]:
    """Hash an existing candidate completely; a size match alone is never a cache hit."""
    try:
        size = os.path.getsize(path)
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(DEFAULT_CHUNK_SIZE)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError:
        return False, 0
    if expected_size is not None and size != expected_size:
        return False, size
    return digest.hexdigest() == expected_sha256, size


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def cache_path(out_dir: str, sha256: str) -> str:
    """Return the discoverable, immutable cache path for a pinned GGUF.

    ``--out`` is always a directory.  The filename is the mandatory artifact
    SHA, so two roles with identical bytes share one cache entry and a changed
    pin can never silently reuse an older artifact.
    """
    return os.path.join(os.path.abspath(os.path.expanduser(out_dir)), f"{sha256}.gguf")


def fetch_locked_model(lockfile_path: str, role: str, out_dir: str, *,
                       allow_loopback_http: bool = False,
                       max_redirects: int = DEFAULT_MAX_REDIRECTS,
                       timeout: float = DEFAULT_TIMEOUT_SECONDS,
                       chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    """Fetch and verify one role from a model lockfile.

    The returned JSON-safe dict contains no source URL.  On success, ``path`` is
    ``OUT/<sha256>.gguf`` and ``cache`` is ``"hit"`` or ``"downloaded"``.
    Any temporary download is created beside that destination and atomically
    promoted only after both SHA-256 and optional size verification pass.
    """
    if not isinstance(role, str) or not role:
        raise ModelFetchError("a non-empty model role is required")
    if not isinstance(out_dir, str) or not out_dir:
        raise ModelFetchError("a non-empty output directory is required")
    if isinstance(max_redirects, bool) or not isinstance(max_redirects, int) or max_redirects < 0:
        raise ModelFetchError("max_redirects must be a non-negative integer")
    if timeout <= 0:
        raise ModelFetchError("timeout must be positive")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ModelFetchError("chunk_size must be a positive integer")

    document = load_lockfile(lockfile_path, allow_loopback_http=allow_loopback_http)
    pinned = pinned_model(document, role)
    expected_sha256 = str(pinned["sha256"]).casefold()
    expected_size = pinned.get("size_bytes")
    url = pinned["url"]
    _validate_transport_url(url, allow_loopback_http=allow_loopback_http)

    destination = cache_path(out_dir, expected_sha256)
    output_root = os.path.dirname(destination)
    try:
        os.makedirs(output_root, exist_ok=True)
    except OSError as exc:
        raise ModelFetchError(
            f"could not create output directory: {type(exc).__name__}") from None

    if os.path.exists(destination):
        verified, size = _verified_file(destination, expected_sha256, expected_size)
        if verified:
            return {
                "ok": True,
                "role": role,
                "path": destination,
                "sha256": expected_sha256,
                "size_bytes": size,
                "cache": "hit",
            }
        _remove_quietly(destination)
        if os.path.exists(destination):
            raise ModelFetchError("cached model failed verification and could not be removed")

    handler = _SecureRedirectHandler(
        max_redirects=max_redirects, allow_loopback_http=allow_loopback_http)
    opener = _policy_guarded_opener(handler)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    temporary = None
    digest = hashlib.sha256()
    written = 0
    try:
        fd, temporary = tempfile.mkstemp(
            prefix=f".{expected_sha256}.", suffix=".part", dir=output_root)
        with os.fdopen(fd, "wb") as handle:
            try:
                with opener.open(request, timeout=timeout) as response:
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        handle.write(chunk)
                        digest.update(chunk)
                        written += len(chunk)
                        if expected_size is not None and written > expected_size:
                            raise ModelFetchError(
                                f"model size mismatch: downloaded more than the expected "
                                f"{expected_size} bytes")
            except ModelFetchError:
                raise
            except urllib.error.HTTPError as exc:
                raise ModelFetchError(
                    f"model download failed with HTTP status {exc.code}") from None
            except Exception as exc:
                raise ModelFetchError(
                    f"model download failed ({type(exc).__name__})") from None
            handle.flush()
            os.fsync(handle.fileno())

        if expected_size is not None and written != expected_size:
            raise ModelFetchError(
                f"model size mismatch: downloaded {written} bytes, expected {expected_size}")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise ModelFetchError(
                f"model SHA-256 mismatch: downloaded {actual_sha256}, expected {expected_sha256}")
        os.replace(temporary, destination)
        temporary = None
    except LockfileError:
        raise
    except ModelFetchError:
        raise
    except OSError as exc:
        raise ModelFetchError(
            f"could not store verified model: {type(exc).__name__}") from None
    finally:
        if temporary is not None:
            _remove_quietly(temporary)

    return {
        "ok": True,
        "role": role,
        "path": destination,
        "sha256": expected_sha256,
        "size_bytes": written,
        "cache": "downloaded",
    }


__all__ = [
    "DEFAULT_MAX_REDIRECTS",
    "ModelFetchError",
    "cache_path",
    "fetch_locked_model",
]
