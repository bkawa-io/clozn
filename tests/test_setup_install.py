"""End-to-end tests for clozn/setup/install.py's plan_install/run_install/run_rollback/read_status
against a loopback http.server standing in for a real GitHub Releases host -- never a real network call.
Every test passes an explicit `home=str(tmp_path)`, so nothing here can ever touch the developer's real
~/.clozn (clozn/setup/* modules take `home` as an explicit parameter precisely so their tests do not need
the ctx.HOME-monkeypatch dance clozn.cli.* tests use).

The "extracted engine binary" in every fixture archive is actually a tiny Python script, invoked via
`argv_prefix=[sys.executable]` (install.py's own test-only seam) -- qualify_entrypoint() ends up running
`python <extracted-script> --version`, which is a faithful stand-in for `<real-binary> --version` without
needing a compiled, platform-specific executable this test suite could not produce. See install.py's
qualify_entrypoint docstring for why a real clozn-server build does NOT support --version today.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import sys
import tarfile
import threading
import zipfile

import pytest

from clozn.setup import install
from clozn.setup import registry as registry_mod
from clozn.setup.errors import LockError, SelectionError, SetupError, TransportError, VerificationError
from clozn.setup.lock import SetupLock

FAKE_SERVER_SCRIPT = (
    "import sys\n"
    "if '--version' in sys.argv[1:]:\n"
    "    print('fake-clozn-server 9.9.9-test')\n"
    "    sys.exit(0)\n"
    "sys.exit(2)\n"
)


@pytest.fixture
def http_server(tmp_path):
    serve_dir = tmp_path / "_served"
    serve_dir.mkdir()
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(serve_dir))
    server = http.server.HTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield serve_dir, f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join(timeout=5)


def _write_zip_archive(path, *, entrypoint="bin/clozn-server", extra_members=None) -> bytes:
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(entrypoint, FAKE_SERVER_SCRIPT)
        for name, content in (extra_members or {}).items():
            zf.writestr(name, content)
    return path.read_bytes()


def _manifest(artifact_url, sha256, size_bytes, *, entrypoint="bin/clozn-server",
              clozn_version="1.0.0", os_name="linux", arch="x86_64", backend="cpu"):
    return {
        "schema_version": "clozn.engine-manifest.v1",
        "clozn_version": clozn_version,
        "protocol_version": "1.0",
        "artifacts": [{
            "os": os_name, "arch": arch, "backend": backend,
            "url": artifact_url, "sha256": sha256, "size_bytes": size_bytes, "entrypoint": entrypoint,
        }],
    }


def _serve_manifest_and_archive(http_server, **manifest_kwargs):
    """Write a real archive + manifest into the loopback server's directory tree; returns the manifest
    URL. `manifest_kwargs` forwards to _manifest() for the non-computed fields (entrypoint/version/etc)."""
    serve_dir, base_url = http_server
    archive_path = serve_dir / "clozn-engine-1.0.0.zip"
    data = _write_zip_archive(archive_path, entrypoint=manifest_kwargs.get("entrypoint", "bin/clozn-server"))
    sha256 = hashlib.sha256(data).hexdigest()
    doc = _manifest(f"{base_url}/clozn-engine-1.0.0.zip", sha256, len(data), **manifest_kwargs)
    (serve_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    return f"{base_url}/manifest.json"


PLATFORM = {"os": "linux", "arch": "x86_64", "gpu_backend": None, "cuda_major": None}


# ------------------------------------------------------------------------------------------- plan_install

def test_plan_install_never_touches_disk(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = tmp_path / "home"
    plan = install.plan_install(manifest_url=url, home=str(home), platform=PLATFORM)
    assert plan["install_key"] == "1.0.0/linux-x86_64-cpu"
    assert plan["already_installed"] is False
    assert not home.exists()   # plan_install is pure -- no ~/.clozn/engines was ever created


def test_plan_install_rejects_a_version_mismatch(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    with pytest.raises(SelectionError, match="clozn_version"):
        install.plan_install(manifest_url=url, version="9.9.9", home=str(tmp_path / "home"),
                             platform=PLATFORM)


def test_plan_install_propagates_a_selection_error_for_an_unmatched_platform(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    macos = {"os": "macos", "arch": "arm64", "gpu_backend": "metal", "cuda_major": None}
    with pytest.raises(SetupError):
        install.plan_install(manifest_url=url, home=str(tmp_path / "home"), platform=macos)


# ------------------------------------------------------------------------------------------- run_install

def test_run_install_happy_path(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    result = install.run_install(manifest_url=url, home=home, platform=PLATFORM,
                                 argv_prefix=[sys.executable])
    assert result["action"] == "installed"
    assert result["record"]["qualification"]["ran"] is True
    assert result["record"]["qualification"]["returncode"] == 0

    reg = registry_mod.load(home)
    assert reg["active"] == "1.0.0/linux-x86_64-cpu"
    assert "previous" not in reg
    entrypoint = reg["installed"]["1.0.0/linux-x86_64-cpu"]["entrypoint"]
    import os
    assert os.path.isfile(entrypoint)
    # the download/staging scratch directories never survive a successful install
    assert not os.path.isdir(os.path.join(registry_mod.engines_dir(home), ".staging")) or \
        not os.listdir(os.path.join(registry_mod.engines_dir(home), ".staging"))
    assert not os.listdir(os.path.join(registry_mod.engines_dir(home), ".download"))


def test_run_install_is_idempotent_when_already_active(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    install.run_install(manifest_url=url, home=home, platform=PLATFORM, argv_prefix=[sys.executable])
    second = install.run_install(manifest_url=url, home=home, platform=PLATFORM,
                                 argv_prefix=[sys.executable])
    assert second["action"] == "noop_already_active"


def test_run_install_with_force_redownloads_and_reextracts(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    install.run_install(manifest_url=url, home=home, platform=PLATFORM, argv_prefix=[sys.executable])
    result = install.run_install(manifest_url=url, home=home, platform=PLATFORM,
                                 argv_prefix=[sys.executable], force=True)
    assert result["action"] == "installed"


def test_run_install_rejects_a_tampered_archive(http_server, tmp_path):
    serve_dir, base_url = http_server
    archive_path = serve_dir / "clozn-engine-1.0.0.zip"
    data = _write_zip_archive(archive_path)
    real_sha = hashlib.sha256(data).hexdigest()
    tampered_sha = "f" * 64
    doc = _manifest(f"{base_url}/clozn-engine-1.0.0.zip", tampered_sha, len(data))
    (serve_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    home = str(tmp_path / "home")

    with pytest.raises(VerificationError):
        install.run_install(manifest_url=f"{base_url}/manifest.json", home=home, platform=PLATFORM,
                            argv_prefix=[sys.executable])
    assert registry_mod.load(home).get("active") is None
    assert real_sha != tampered_sha   # sanity: the test really did tamper the declared hash


def test_run_install_rejects_a_manifest_entrypoint_that_is_not_in_the_archive(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server, entrypoint="bin/nonexistent-server")
    # rewrite the manifest's declared entrypoint to something the archive never contained
    serve_dir, _base_url = http_server
    doc = json.loads((serve_dir / "manifest.json").read_text(encoding="utf-8"))
    doc["artifacts"][0]["entrypoint"] = "bin/this-path-is-not-in-the-zip"
    (serve_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    home = str(tmp_path / "home")
    with pytest.raises(SetupError, match="not found in the extracted archive"):
        install.run_install(manifest_url=url, home=home, platform=PLATFORM, argv_prefix=[sys.executable])
    assert registry_mod.load(home).get("active") is None


def test_run_install_rejects_a_path_traversal_archive(http_server, tmp_path):
    serve_dir, base_url = http_server
    archive_path = serve_dir / "evil.zip"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr("../../evil.txt", "pwned")
    data = archive_path.read_bytes()
    doc = _manifest(f"{base_url}/evil.zip", hashlib.sha256(data).hexdigest(), len(data))
    (serve_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    home = str(tmp_path / "home")
    with pytest.raises(SetupError):
        install.run_install(manifest_url=f"{base_url}/manifest.json", home=home, platform=PLATFORM,
                            argv_prefix=[sys.executable])
    assert registry_mod.load(home).get("active") is None


def test_run_install_a_second_engine_upgrade_retains_previous_for_rollback(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    install.run_install(manifest_url=url, home=home, platform=PLATFORM, argv_prefix=[sys.executable])

    url_v2 = _serve_manifest_and_archive(http_server, clozn_version="1.1.0")
    result = install.run_install(manifest_url=url_v2, home=home, platform=PLATFORM,
                                 argv_prefix=[sys.executable])
    assert result["action"] == "installed"
    reg = registry_mod.load(home)
    assert reg["active"] == "1.1.0/linux-x86_64-cpu"
    assert reg["previous"] == "1.0.0/linux-x86_64-cpu"
    # the old version's install directory is untouched, not deleted by the upgrade
    import os
    assert os.path.isfile(reg["installed"]["1.0.0/linux-x86_64-cpu"]["entrypoint"])


def test_run_install_refuses_a_concurrent_invocation(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    with SetupLock(registry_mod.lock_path(home)):
        with pytest.raises(LockError):
            install.run_install(manifest_url=url, home=home, platform=PLATFORM,
                                argv_prefix=[sys.executable])
    assert registry_mod.load(home).get("active") is None


def test_run_install_reports_a_clear_error_when_the_manifest_host_is_unreachable(tmp_path):
    home = str(tmp_path / "home")
    with pytest.raises(TransportError):
        install.run_install(manifest_url="http://127.0.0.1:1/manifest.json", home=home, platform=PLATFORM)
    assert registry_mod.load(home).get("active") is None


# ------------------------------------------------------------------------------------------ run_rollback

def test_run_rollback_restores_the_previous_engine(http_server, tmp_path):
    url_v1 = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    install.run_install(manifest_url=url_v1, home=home, platform=PLATFORM, argv_prefix=[sys.executable])
    url_v2 = _serve_manifest_and_archive(http_server, clozn_version="1.1.0")
    install.run_install(manifest_url=url_v2, home=home, platform=PLATFORM, argv_prefix=[sys.executable])

    result = install.run_rollback(home=home)
    assert result["active"] == "1.0.0/linux-x86_64-cpu"
    reg = registry_mod.load(home)
    assert reg["active"] == "1.0.0/linux-x86_64-cpu"
    assert reg["previous"] == "1.1.0/linux-x86_64-cpu"


def test_run_rollback_with_nothing_to_roll_back_to_raises(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    install.run_install(manifest_url=url, home=home, platform=PLATFORM, argv_prefix=[sys.executable])
    with pytest.raises(SetupError, match="nothing to roll back to"):
        install.run_rollback(home=home)


# -------------------------------------------------------------------------------------------- read_status

def test_read_status_reports_active_and_installed(http_server, tmp_path):
    url = _serve_manifest_and_archive(http_server)
    home = str(tmp_path / "home")
    install.run_install(manifest_url=url, home=home, platform=PLATFORM, argv_prefix=[sys.executable])
    status = install.read_status(home=home)
    assert status["active_key"] == "1.0.0/linux-x86_64-cpu"
    assert status["previous"] is None
    assert len(status["installed"]) == 1


def test_read_status_on_a_fresh_home_reports_nothing_installed(tmp_path):
    status = install.read_status(home=str(tmp_path / "home"))
    assert status["active"] is None
    assert status["installed"] == []


# --------------------------------------------------------------------------------------- qualify_entrypoint

def test_qualify_entrypoint_reports_ran_false_for_a_missing_file(tmp_path):
    result = install.qualify_entrypoint([str(tmp_path / "does-not-exist")])
    assert result["ran"] is False


def test_qualify_entrypoint_reports_ran_true_with_exit_code(tmp_path):
    script = tmp_path / "fake.py"
    script.write_text(FAKE_SERVER_SCRIPT, encoding="utf-8")
    result = install.qualify_entrypoint([sys.executable, str(script)])
    assert result["ran"] is True
    assert result["returncode"] == 0
    assert "fake-clozn-server" in result["stdout"]


def test_qualify_entrypoint_treats_a_nonzero_exit_as_ran_not_a_failure(tmp_path):
    """The real clozn-server does not implement --version yet (see the module docstring this mirrors) --
    a nonzero exit must still count as 'ran', or every real engine build would fail qualification."""
    script = tmp_path / "no_version_flag.py"
    script.write_text("import sys\nsys.exit(2)\n", encoding="utf-8")
    result = install.qualify_entrypoint([sys.executable, str(script)])
    assert result["ran"] is True
    assert result["returncode"] == 2


# --------------------------------------------------------------------------------------- tarball artifacts

def test_run_install_accepts_a_tar_gz_artifact(http_server, tmp_path):
    serve_dir, base_url = http_server
    archive_path = serve_dir / "clozn-engine-1.0.0.tar.gz"
    with tarfile.open(archive_path, "w:gz") as tf:
        import io
        data = FAKE_SERVER_SCRIPT.encode("utf-8")
        info = tarfile.TarInfo(name="bin/clozn-server")
        info.size = len(data)
        info.mode = 0o755
        tf.addfile(info, io.BytesIO(data))
    data = archive_path.read_bytes()
    doc = _manifest(f"{base_url}/clozn-engine-1.0.0.tar.gz", hashlib.sha256(data).hexdigest(), len(data))
    (serve_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    home = str(tmp_path / "home")
    result = install.run_install(manifest_url=f"{base_url}/manifest.json", home=home, platform=PLATFORM,
                                 argv_prefix=[sys.executable])
    assert result["action"] == "installed"
