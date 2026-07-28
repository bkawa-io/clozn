"""fetch_bytes() / download_to_file() -- the only functions in clozn/setup that open a URL.

Both call `urllib.request.urlopen` directly (never a raw `http.client` socket), which is deliberate: the
product's global outbound guard (clozn.network_policy.install_urllib_guard) wraps that exact name, so a
`clozn setup` run under CLOZN_LOCAL_ONLY=1 is blocked and ledgered by the SAME mechanism `clozn pull`
already goes through -- nothing here needs to know local-only mode exists. This module adds the one
policy on top that network_policy does not: engine downloads must be https (or an explicit loopback/
file:// override for local development and tests -- see _check_scheme's docstring).
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from urllib.parse import urlsplit

from clozn.setup.errors import TransportError, VerificationError

USER_AGENT = "clozn-setup/0.1"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1"})


def _check_scheme(url: str) -> None:
    """https:// is the only scheme a production `clozn setup` invocation should ever see. Two
    exceptions, both inert in a real release: `file://` (an explicit developer/test manifest override --
    CLOZN_ENGINE_MANIFEST_URL is documented as exactly that) and `http://` to 127.0.0.1/localhost/::1
    (the loopback fixture servers this feature's own test suite uses in place of a real network call).
    Anything else -- plain http to a real host above all -- is refused outright (roadmap: 'Require HTTPS
    for release downloads')."""
    parsed = urlsplit(url)
    scheme = parsed.scheme.casefold()
    if scheme == "https":
        return
    if scheme == "file":
        return
    if scheme == "http" and (parsed.hostname or "").casefold() in _LOOPBACK_HOSTS:
        return
    raise TransportError(
        f"refusing to fetch {url!r}: engine manifests/artifacts must be https://. A loopback http:// "
        f"URL and file:// are only accepted as an explicit development/test override.")


def fetch_bytes(url: str, *, timeout: float = 30.0, max_bytes: int = 4 * 1024 * 1024) -> bytes:
    """Fetch a small document (an engine manifest) fully into memory. Refuses a response over
    `max_bytes` -- a manifest is a few KB of JSON; anything claiming megabytes is not one."""
    _check_scheme(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read(max_bytes + 1)
    except TransportError:
        raise
    except Exception as error:
        raise TransportError(f"could not fetch {url!r}: {type(error).__name__}: {error}") from None
    if len(data) > max_bytes:
        raise TransportError(f"{url!r} returned more than {max_bytes} bytes; refusing (not a manifest)")
    return data


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


def download_to_file(url: str, dest_path: str, *, expected_sha256: "str | None" = None,
                      expected_size: "int | None" = None, timeout: float = 120.0,
                      chunk_size: int = 1 << 20, progress=None) -> str:
    """Stream `url` into `dest_path` via a same-directory `.part` file (mirrors
    clozn/cli/commands/models.py's cmd_pull), hashing as it goes so a multi-GB artifact is never held
    fully in memory. Verifies size/sha256 BEFORE the atomic `os.replace` into `dest_path` -- on a
    mismatch the `.part` file is removed and dest_path is never created (VerificationError). Returns the
    verified sha256 hex digest. `progress(bytes_written)` is called after every chunk when given;
    optional, and its own failure is not caught (a caller-provided callback that raises is a caller bug,
    not a download failure)."""
    _check_scheme(url)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    tmp_path = dest_path + ".part"
    digest = hashlib.sha256()
    written = 0
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with open(tmp_path, "wb") as handle:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    handle.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
                    if progress is not None:
                        progress(written)
    except TransportError:
        _remove_quietly(tmp_path)
        raise
    except Exception as error:
        _remove_quietly(tmp_path)
        raise TransportError(f"could not download {url!r}: {type(error).__name__}: {error}") from None

    if expected_size is not None and written != expected_size:
        _remove_quietly(tmp_path)
        raise VerificationError(
            f"{url!r}: downloaded {written} bytes, manifest declares size_bytes={expected_size}")
    actual_sha256 = digest.hexdigest()
    if expected_sha256 is not None and actual_sha256 != str(expected_sha256).lower():
        _remove_quietly(tmp_path)
        raise VerificationError(
            f"{url!r}: sha256 mismatch (downloaded {actual_sha256}, manifest declares "
            f"{expected_sha256}) -- refusing to install a payload that does not match its manifest")
    os.replace(tmp_path, dest_path)
    return actual_sha256
