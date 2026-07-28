"""Model-free secure fetch tests using only an in-process loopback HTTP server."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

import pytest  # noqa: E402

import clozn.cli.commands.model_lock as model_lock_cli  # noqa: E402
import clozn.models.fetch as model_fetch  # noqa: E402
from clozn import network_policy  # noqa: E402
from clozn.models.fetch import ModelFetchError, fetch_locked_model  # noqa: E402
from clozn.models.lockfile import LockfileError  # noqa: E402


PAYLOAD = b"GGUF\0model-lock-fetch-test-payload"


@pytest.fixture(autouse=True)
def _isolated_network_policy(monkeypatch, tmp_path):
    monkeypatch.setenv(
        network_policy.LEDGER_ENV, str(tmp_path / "outbound_attempts.jsonl"))
    monkeypatch.setenv(
        network_policy.POLICY_ENV, str(tmp_path / "network_policy.json"))
    monkeypatch.delenv(network_policy.LOCAL_ONLY_ENV, raising=False)


class _Handler(BaseHTTPRequestHandler):
    requests = 0

    def do_GET(self):
        type(self).requests += 1
        if self.path == "/artifact.gguf":
            self.send_response(200)
            self.send_header("Content-Length", str(len(PAYLOAD)))
            self.end_headers()
            self.wfile.write(PAYLOAD)
            return
        if self.path == "/redirect":
            self.send_response(302)
            self.send_header("Location", "/artifact.gguf")
            self.end_headers()
            return
        if self.path == "/redirect-external":
            self.send_response(302)
            self.send_header("Location", "http://example.invalid/private?token=do-not-log")
            self.end_headers()
            return
        if self.path == "/redirect-loop":
            self.send_response(302)
            self.send_header("Location", "/redirect-loop")
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, _format, *_args):
        return


@pytest.fixture
def loopback_server():
    _Handler.requests = 0
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def _write_lock(tmp_path, url, *, sha256=None, size_bytes=None):
    entry = {
        "url": url,
        "sha256": sha256 or hashlib.sha256(PAYLOAD).hexdigest(),
    }
    if size_bytes is not None:
        entry["size_bytes"] = size_bytes
    path = tmp_path / "clozn.lock.json"
    path.write_text(json.dumps({
        "schema_version": "clozn.model-lock.v1",
        "models": {"candidate": entry},
    }), encoding="utf-8")
    return str(path)


def _parts(out):
    return list(out.glob("*.part")) + list(out.glob(".*.part"))


def test_fetch_success_follows_loopback_redirect_and_promotes_sha_keyed_file(
        loopback_server, tmp_path):
    lock = _write_lock(
        tmp_path, f"{loopback_server}/redirect", size_bytes=len(PAYLOAD))
    out = tmp_path / "models"

    result = fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)

    expected_sha = hashlib.sha256(PAYLOAD).hexdigest()
    assert result == {
        "ok": True,
        "role": "candidate",
        "path": str(out / f"{expected_sha}.gguf"),
        "sha256": expected_sha,
        "size_bytes": len(PAYLOAD),
        "cache": "downloaded",
    }
    assert (out / f"{expected_sha}.gguf").read_bytes() == PAYLOAD
    assert _parts(out) == []


@pytest.mark.parametrize(
    ("sha256", "size_bytes", "message"),
    [
        ("f" * 64, None, "SHA-256 mismatch"),
        (None, len(PAYLOAD) + 1, "size mismatch"),
    ],
)
def test_fetch_mismatch_cleans_temporary_and_never_promotes(
        loopback_server, tmp_path, sha256, size_bytes, message):
    lock = _write_lock(
        tmp_path, f"{loopback_server}/artifact.gguf",
        sha256=sha256, size_bytes=size_bytes)
    out = tmp_path / "models"

    with pytest.raises(ModelFetchError, match=message):
        fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)

    assert list(out.glob("*.gguf")) == []
    assert _parts(out) == []


def test_fetch_refuses_non_loopback_http_redirect_before_connection(loopback_server, tmp_path):
    lock = _write_lock(tmp_path, f"{loopback_server}/redirect-external")
    out = tmp_path / "models"

    with pytest.raises(ModelFetchError, match="must use HTTPS") as caught:
        fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)

    assert "token" not in str(caught.value)
    assert "example.invalid" not in str(caught.value)
    assert list(out.glob("*.gguf")) == []
    assert _parts(out) == []


def test_fetch_bounds_redirects(loopback_server, tmp_path):
    lock = _write_lock(tmp_path, f"{loopback_server}/redirect-loop")
    with pytest.raises(ModelFetchError, match=r"redirect limit \(1\)"):
        fetch_locked_model(
            lock, "candidate", str(tmp_path / "models"),
            allow_loopback_http=True, max_redirects=1)


def test_redirect_handler_refuses_https_downgrade_before_building_request():
    handler = model_fetch._SecureRedirectHandler(
        max_redirects=5, allow_loopback_http=True)
    request = urllib.request.Request("https://models.example/artifact.gguf")

    with pytest.raises(ModelFetchError, match="weaker scheme"):
        handler.redirect_request(
            request, None, 302, "Found",
            {"Location": "http://127.0.0.1/artifact.gguf"},
            "http://127.0.0.1/artifact.gguf")


def test_plain_loopback_http_requires_explicit_test_seam(loopback_server, tmp_path):
    lock = _write_lock(tmp_path, f"{loopback_server}/artifact.gguf")
    with pytest.raises(LockfileError, match="does not conform"):
        fetch_locked_model(lock, "candidate", str(tmp_path / "models"))
    assert _Handler.requests == 0


def test_fetch_custom_opener_honors_local_only_and_records_redacted_attempt(monkeypatch, tmp_path):
    lock = _write_lock(
        tmp_path, "https://models.example/private/model.gguf?token=do-not-record")
    monkeypatch.setenv(network_policy.LOCAL_ONLY_ENV, "1")

    with pytest.raises(ModelFetchError, match="LocalOnlyViolation") as caught:
        fetch_locked_model(lock, "candidate", str(tmp_path / "models"))

    assert "token" not in str(caught.value)
    assert "models.example" not in str(caught.value)
    rows = network_policy.read_outbound_attempts()
    assert len(rows) == 1
    assert rows[0]["outcome"] == "blocked"
    assert rows[0]["host"] == "models.example"
    assert "path" not in rows[0] and "query" not in rows[0]


def test_verified_cache_hit_is_rehashed_without_network(loopback_server, tmp_path):
    lock = _write_lock(tmp_path, f"{loopback_server}/artifact.gguf")
    out = tmp_path / "models"
    first = fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)
    request_count = _Handler.requests

    second = fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)

    assert first["cache"] == "downloaded"
    assert second["cache"] == "hit"
    assert _Handler.requests == request_count


def test_corrupt_cache_is_invalidated_and_replaced(loopback_server, tmp_path):
    lock = _write_lock(tmp_path, f"{loopback_server}/artifact.gguf")
    out = tmp_path / "models"
    first = fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)
    with open(first["path"], "wb") as handle:
        handle.write(b"corrupt-but-same-cache-key")
    request_count = _Handler.requests

    second = fetch_locked_model(lock, "candidate", str(out), allow_loopback_http=True)

    assert second["cache"] == "downloaded"
    assert _Handler.requests == request_count + 1
    assert open(second["path"], "rb").read() == PAYLOAD


def test_fetch_unknown_role_is_clear_and_does_not_open_network(loopback_server, tmp_path):
    lock = _write_lock(tmp_path, f"{loopback_server}/artifact.gguf")
    with pytest.raises(LockfileError, match="no model pinned for role 'baseline'"):
        fetch_locked_model(
            lock, "baseline", str(tmp_path / "models"), allow_loopback_http=True)
    assert _Handler.requests == 0


def _args(**overrides):
    values = {
        "lockfile": "models/clozn.lock.json",
        "role": "candidate",
        "out": ".models",
        "json": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(out=".models"):
    return {
        "ok": True,
        "role": "candidate",
        "path": os.path.join(out, "a" * 64 + ".gguf"),
        "sha256": "a" * 64,
        "size_bytes": 123,
        "cache": "downloaded",
    }


def test_fetch_cli_human_output(monkeypatch, capsys):
    monkeypatch.setattr(model_fetch, "fetch_locked_model", lambda *a, **kw: _result())
    assert model_lock_cli.cmd_model_lock_fetch(_args()) == 0
    output = capsys.readouterr().out
    assert "candidate downloaded" in output
    assert "a" * 64 in output
    assert "https://" not in output


def test_fetch_cli_json_output(monkeypatch, capsys):
    expected = _result("cache")
    monkeypatch.setattr(model_fetch, "fetch_locked_model", lambda *a, **kw: expected)
    assert model_lock_cli.cmd_model_lock_fetch(_args(out="cache", json=True)) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_fetch_cli_redacts_url_from_error(monkeypatch, capsys):
    def fail(*_args, **_kwargs):
        raise ModelFetchError(
            "could not fetch https://user:secret@example.test/private?token=hidden")

    monkeypatch.setattr(model_fetch, "fetch_locked_model", fail)
    assert model_lock_cli.cmd_model_lock_fetch(_args(json=True)) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "secret" not in payload["error"]
    assert "hidden" not in payload["error"]
    assert "<redacted-url>" in payload["error"]
