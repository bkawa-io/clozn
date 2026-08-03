"""`clozn setup`/`setup status`/`setup upgrade`/`setup rollback` end to end, calling cmd_setup() directly
with an argparse.Namespace (the same pattern tests/test_doctor.py uses for cmd_doctor) against a loopback
http.server fixture manifest -- never a real network call, never CLOZN_ENGINE_MANIFEST_URL pointed at a
real host. ctx.HOME is monkeypatched to tmp_path throughout, matching tests/test_cli_branch.py's
`monkeypatch.setattr(cli, "HOME", ...)` convention -- nothing here can touch the developer's real
~/.clozn.
"""
from __future__ import annotations

import functools
import hashlib
import http.server
import json
import sys
import threading

import pytest

from clozn.cli import main as cli
from clozn.cli.commands import setup_engine
from clozn.cli.main import CloznError


def _host_platform() -> tuple[str, str]:
    """Use the installer's canonical names so this genuinely cross-platform fixture matches host."""
    from clozn.setup.platform_detect import detect_platform

    detected = detect_platform(probe_gpu=False)
    return detected["os"], detected["arch"]


HOST_OS, HOST_ARCH = _host_platform()


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


def _publish_manifest(http_server, *, clozn_version="1.0.0"):
    """Publish a manifest whose one artifact's entrypoint is a REAL genuinely-launchable executable --
    unlike tests/test_setup_install.py (which uses argv_prefix=[sys.executable] to run a .py fixture as
    a test-only seam), this file exercises setup_engine.py's actual production code path, which has no
    such seam. The simplest thing that is honestly launchable on every platform without a compiler is
    this process's OWN interpreter binary. The isolated fixture replaces only the build-info subprocess
    probe with a contract-valid result; tests/test_setup_install.py exercises the real strict parser."""
    import zipfile
    serve_dir, base_url = http_server
    archive_path = serve_dir / f"clozn-engine-{clozn_version}.zip"
    entrypoint_name = "bin/clozn-server.exe" if sys.platform == "win32" else "bin/clozn-server"
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.write(sys.executable, entrypoint_name)
    data = archive_path.read_bytes()
    doc = {
        "schema_version": "clozn.engine-manifest.v1",
        "clozn_version": clozn_version,
        "protocol_version": "1.0",
        "artifacts": [{
            "os": HOST_OS, "arch": HOST_ARCH, "backend": "cpu",
            "url": f"{base_url}/{archive_path.name}",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data), "entrypoint": entrypoint_name,
            "build_id": f"release-{clozn_version}-cpu",
            "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
            "feature_flags": {
                "jlens": True, "lora": True, "native_chat_io": True,
                "sae": False, "whitebox": True,
            },
        }],
    }
    (serve_dir / "manifest.json").write_text(json.dumps(doc), encoding="utf-8")
    return f"{base_url}/manifest.json"


INSTALL_KEY = f"1.0.0/{HOST_OS}-{HOST_ARCH}-cpu"
INSTALL_KEY_V2 = f"1.1.0/{HOST_OS}-{HOST_ARCH}-cpu"


@pytest.fixture
def isolated(tmp_path, monkeypatch, http_server):
    monkeypatch.setattr(cli, "HOME", str(tmp_path / ".clozn"))
    url = _publish_manifest(http_server)
    monkeypatch.setenv("CLOZN_ENGINE_MANIFEST_URL", url)
    # detect_platform()'s GPU probe is host-dependent (nvidia-smi may or may not exist on the runner);
    # pin it off explicitly so backend selection is deterministic and matches the cpu-only artifact
    # _publish_manifest() offers -- os/arch are left to the real detector (matching this runner truthfully).
    import clozn.setup.install as install_mod
    real_detect = install_mod.platform_detect.detect_platform
    monkeypatch.setattr(
        install_mod.platform_detect, "detect_platform",
        lambda **kw: {**real_detect(probe_gpu=False), "gpu_backend": None, "cuda_major": None})
    def _qualified_build_info(_argv, *, timeout=5.0, expected=None):
        build_info = {
            "engine_version": "1.0.0",
            "build_id": "release-1.0.0-cpu",
            "protocol_version": "1.0",
            "backend": "cpu",
            "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
            "feature_flags": {
                "jlens": True, "lora": True, "native_chat_io": True,
                "sae": False, "whitebox": True,
            },
        }
        build_info.update({key: value for key, value in (expected or {}).items() if value is not None})
        return {
            "ran": True, "qualified": True, "returncode": 0,
            "stdout": json.dumps(build_info), "stderr": "", "build_info": build_info,
        }
    monkeypatch.setattr(install_mod, "qualify_entrypoint", _qualified_build_info)
    return tmp_path, http_server


def _ns(**kw):
    """A minimal argparse.Namespace with every attribute add_subparser() ever sets a default for,
    matching what argparse itself would produce for a given subcommand path."""
    base = dict(backend="auto", version=None, dry_run=False, force=False, json=False, setup_cmd=None)
    base.update(kw)
    import argparse
    return argparse.Namespace(**base)


# --------------------------------------------------------------------------------------------- bare install

def test_setup_dry_run_json_changes_nothing_and_is_deterministic(isolated, capsys):
    tmp_path, _http = isolated
    setup_engine.cmd_setup(_ns(dry_run=True, json=True))
    first = capsys.readouterr().out
    setup_engine.cmd_setup(_ns(dry_run=True, json=True))
    second = capsys.readouterr().out
    assert first == second
    assert not (tmp_path / ".clozn").exists()
    payload = json.loads(first)
    assert payload["action"] == "plan"
    assert payload["install_key"] == INSTALL_KEY


def test_setup_installs_and_activates(isolated, capsys):
    setup_engine.cmd_setup(_ns())
    out = capsys.readouterr().out
    assert f"installed and activated {INSTALL_KEY}" in out


def test_setup_json_reports_four_states_separately(isolated, capsys):
    setup_engine.cmd_setup(_ns(json=True))
    payload = json.loads(capsys.readouterr().out)
    states = payload["states"]
    assert set(states) == {
        "python_package_installed", "compatible_engine_installed",
        "core_inference_qualification", "white_box_qualification",
    }
    assert states["python_package_installed"]["status"] == "passed"
    assert states["compatible_engine_installed"]["status"] == "found"
    assert states["core_inference_qualification"]["status"] == "skipped"
    assert states["white_box_qualification"]["status"] == "skipped"
    # never compressed into one boolean -- each has its OWN status, and they may legitimately differ
    assert states["compatible_engine_installed"]["status"] != states["core_inference_qualification"]["status"]


def test_setup_is_idempotent_on_a_second_run(isolated, capsys):
    setup_engine.cmd_setup(_ns())
    capsys.readouterr()
    setup_engine.cmd_setup(_ns())
    out = capsys.readouterr().out
    assert "already installed and active" in out


def test_setup_wraps_setup_errors_as_clozn_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "HOME", str(tmp_path / ".clozn"))
    monkeypatch.setenv("CLOZN_ENGINE_MANIFEST_URL", "http://127.0.0.1:1/manifest.json")
    with pytest.raises(CloznError):
        setup_engine.cmd_setup(_ns())


# ------------------------------------------------------------------------------------------------- status

def test_status_before_any_install(isolated, capsys):
    setup_engine.cmd_setup(_ns(setup_cmd="status", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["active_key"] is None
    assert payload["installed"] == []


def test_status_after_install(isolated, capsys):
    setup_engine.cmd_setup(_ns())
    capsys.readouterr()
    setup_engine.cmd_setup(_ns(setup_cmd="status", json=True))
    payload = json.loads(capsys.readouterr().out)
    assert payload["active_key"] == INSTALL_KEY
    assert len(payload["installed"]) == 1


# ------------------------------------------------------------------------------------------------ upgrade

def test_upgrade_retains_previous_and_dry_run_reports_the_plan(isolated, http_server, monkeypatch, capsys):
    setup_engine.cmd_setup(_ns())
    capsys.readouterr()

    url_v2 = _publish_manifest(http_server, clozn_version="1.1.0")
    monkeypatch.setenv("CLOZN_ENGINE_MANIFEST_URL", url_v2)

    setup_engine.cmd_setup(_ns(setup_cmd="upgrade", dry_run=True, json=True))
    plan = json.loads(capsys.readouterr().out)
    assert plan["install_key"] == INSTALL_KEY_V2
    assert plan["currently_active"] == INSTALL_KEY

    setup_engine.cmd_setup(_ns(setup_cmd="upgrade", json=True))
    result = json.loads(capsys.readouterr().out)
    assert result["action"] == "installed"

    setup_engine.cmd_setup(_ns(setup_cmd="status", json=True))
    status = json.loads(capsys.readouterr().out)
    assert status["active_key"] == INSTALL_KEY_V2
    assert status["previous_key"] == INSTALL_KEY


# ----------------------------------------------------------------------------------------------- rollback

def test_rollback_restores_the_previous_engine(isolated, http_server, monkeypatch, capsys):
    setup_engine.cmd_setup(_ns())
    capsys.readouterr()
    url_v2 = _publish_manifest(http_server, clozn_version="1.1.0")
    monkeypatch.setenv("CLOZN_ENGINE_MANIFEST_URL", url_v2)
    setup_engine.cmd_setup(_ns(setup_cmd="upgrade", json=True))
    capsys.readouterr()

    setup_engine.cmd_setup(_ns(setup_cmd="rollback", json=True))
    result = json.loads(capsys.readouterr().out)
    assert result["active"] == INSTALL_KEY

    setup_engine.cmd_setup(_ns(setup_cmd="status", json=True))
    status = json.loads(capsys.readouterr().out)
    assert status["active_key"] == INSTALL_KEY
    assert status["previous_key"] == INSTALL_KEY_V2


def test_rollback_with_nothing_to_roll_back_to_raises_clozn_error(isolated, capsys):
    setup_engine.cmd_setup(_ns())
    capsys.readouterr()
    with pytest.raises(CloznError, match="nothing to roll back to"):
        setup_engine.cmd_setup(_ns(setup_cmd="rollback"))


# ---------------------------------------------------------------------------------- argparse wiring itself

def test_setup_is_registered_via_autoload():
    from clozn.cli.main import build_parser
    parser = build_parser()
    import argparse
    subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert "setup" in subparsers_action.choices


def test_setup_subcommands_are_registered():
    from clozn.cli.main import build_parser
    parser = build_parser()
    import argparse
    subparsers_action = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    setup_parser = subparsers_action.choices["setup"]
    nested = next(a for a in setup_parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(nested.choices) == {"status", "upgrade", "rollback"}


def test_setup_parses_dry_run_and_json_via_the_real_parser():
    from clozn.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["setup", "--dry-run", "--json", "--backend", "cpu"])
    assert args.dry_run is True and args.json is True and args.backend == "cpu"
    assert args.fn is setup_engine.cmd_setup


def test_setup_status_subcommand_parses_via_the_real_parser():
    from clozn.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["setup", "status", "--json"])
    assert args.setup_cmd == "status"
    assert args.json is True
    assert args.fn is setup_engine.cmd_setup
