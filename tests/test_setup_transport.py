"""fetch_bytes()/download_to_file() tested against a loopback http.server started in-process (never a
real network host) and file:// URLs -- both are the sanctioned model-free/network-free test doubles for
this feature (see the roadmap plan). A plain http:// URL to a non-loopback host must be refused outright:
that assertion is what proves the HTTPS requirement is real enforcement, not a comment.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import threading

import pytest

from clozn.setup import transport
from clozn.setup.errors import TransportError, VerificationError


@pytest.fixture
def loopback_server(tmp_path):
    """A tiny stdlib HTTP server on 127.0.0.1 serving files out of tmp_path. Bound to port 0 (OS-assigned,
    loopback-only) and torn down at test end -- no real network host is ever contacted. Uses the
    `directory=` kwarg (no os.chdir) so it never mutates the test process's global cwd."""
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


# ------------------------------------------------------------------------------------------- scheme policy

def test_fetch_bytes_rejects_plain_http_to_a_real_host():
    with pytest.raises(TransportError, match="https"):
        transport.fetch_bytes("http://example.invalid/manifest.json")


def test_download_to_file_rejects_plain_http_to_a_real_host(tmp_path):
    with pytest.raises(TransportError, match="https"):
        transport.download_to_file("http://example.invalid/engine.zip", str(tmp_path / "engine.zip"))


def test_fetch_bytes_accepts_a_loopback_http_url(loopback_server, tmp_path):
    (tmp_path / "manifest.json").write_text('{"ok": true}', encoding="utf-8")
    data = transport.fetch_bytes(f"{loopback_server}/manifest.json")
    assert data == b'{"ok": true}'


def test_fetch_bytes_accepts_a_file_url(tmp_path):
    doc = tmp_path / "manifest.json"
    doc.write_text('{"ok": true}', encoding="utf-8")
    data = transport.fetch_bytes(doc.as_uri())
    assert data == b'{"ok": true}'


def test_fetch_bytes_refuses_a_response_over_max_bytes(loopback_server, tmp_path):
    (tmp_path / "big.json").write_bytes(b"x" * 2000)
    with pytest.raises(TransportError, match="more than"):
        transport.fetch_bytes(f"{loopback_server}/big.json", max_bytes=1000)


# ------------------------------------------------------------------------------------------- download flow

def test_download_to_file_verifies_sha256_and_writes_dest(loopback_server, tmp_path):
    payload = b"pretend engine archive bytes"
    (tmp_path / "engine.tar.gz").write_bytes(payload)
    dest = tmp_path / "downloaded" / "engine.tar.gz"
    digest = hashlib.sha256(payload).hexdigest()
    actual = transport.download_to_file(
        f"{loopback_server}/engine.tar.gz", str(dest), expected_sha256=digest, expected_size=len(payload))
    assert actual == digest
    assert dest.read_bytes() == payload
    assert not (dest.parent / (dest.name + ".part")).exists()


def test_download_to_file_rejects_a_sha256_mismatch_and_leaves_no_file(loopback_server, tmp_path):
    (tmp_path / "engine.tar.gz").write_bytes(b"pretend engine archive bytes")
    dest = tmp_path / "downloaded" / "engine.tar.gz"
    with pytest.raises(VerificationError, match="sha256 mismatch"):
        transport.download_to_file(f"{loopback_server}/engine.tar.gz", str(dest), expected_sha256="f" * 64)
    assert not dest.exists()
    assert not (dest.parent / (dest.name + ".part")).exists()


def test_download_to_file_rejects_a_size_mismatch(loopback_server, tmp_path):
    (tmp_path / "engine.tar.gz").write_bytes(b"pretend engine archive bytes")
    dest = tmp_path / "downloaded" / "engine.tar.gz"
    with pytest.raises(VerificationError, match="size_bytes"):
        transport.download_to_file(f"{loopback_server}/engine.tar.gz", str(dest), expected_size=999999)
    assert not dest.exists()


def test_download_to_file_reports_a_clear_error_on_404(loopback_server, tmp_path):
    dest = tmp_path / "downloaded" / "engine.tar.gz"
    with pytest.raises(TransportError, match="could not download"):
        transport.download_to_file(f"{loopback_server}/does-not-exist.tar.gz", str(dest))
    assert not dest.exists()


def test_download_to_file_calls_progress_with_running_total(loopback_server, tmp_path):
    payload = b"x" * (5 * 1024)
    (tmp_path / "engine.bin").write_bytes(payload)
    dest = tmp_path / "downloaded" / "engine.bin"
    seen = []
    transport.download_to_file(f"{loopback_server}/engine.bin", str(dest),
                               chunk_size=1024, progress=seen.append)
    assert seen and seen[-1] == len(payload)
    assert seen == sorted(seen)
