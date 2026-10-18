"""The retained package, worker, and core-inference checks in ``clozn doctor``."""
from __future__ import annotations

import argparse
import json

import pytest

from clozn.cli import engine_process
from clozn.cli.commands import doctor


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(engine_process, "ENGINE_CORE", str(tmp_path / "engine_core"))
    from clozn.cli import main as cli
    monkeypatch.setattr(cli, "HOME", str(tmp_path / ".clozn"))
    for var in ("CLOZN_ENGINE", "CLOZN_ENGINE_GPU"):
        monkeypatch.delenv(var, raising=False)
    return tmp_path


def _touch_exe(tmp_path, rel):
    path = tmp_path / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fake")
    return str(path)


# --------------------------------------------------------------------------------------- individual checks

def test_check_package_installed_reports_ok_with_version():
    result = doctor._check_package_installed()
    assert result["status"] == "OK"
    import clozn
    assert clozn.__version__ in result["detail"]


def test_check_engine_warns_with_no_binary_and_points_to_supported_install_paths(isolated):
    result = doctor._check_engine()
    assert result["status"] == "WARN"
    assert "docs/DEVELOPMENT.md" in result["detail"]
    assert "CLOZN_ENGINE" in result["detail"]
    assert "clozn setup" not in result["detail"]


def test_check_engine_reports_discovery_source_when_found(isolated, tmp_path):
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")
    result = doctor._check_engine()
    assert result["status"] == "OK"
    assert "source=repo_dev_build" in result["detail"]
    assert result["discovery_source"] == "repo_dev_build"


def test_check_engine_deep_false_never_launches_anything(isolated, tmp_path, monkeypatch):
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")

    def _boom(*a, **k):
        raise AssertionError("qualify_entrypoint must not be called when deep=False")

    monkeypatch.setattr("clozn.setup.install.qualify_entrypoint", _boom)
    result = doctor._check_engine(deep=False)
    assert result["status"] == "OK"
    assert "--deep" not in result["detail"]


def test_check_engine_deep_true_runs_build_identity_check_and_warns_on_failure(isolated, tmp_path):
    """The fake exe above is not a real Win32/ELF binary, so a genuine subprocess launch attempt fails --
    proving --deep actually exercises build identity rather than trusting the file's existence."""
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")
    result = doctor._check_engine(deep=True)
    assert result["status"] == "WARN"
    assert "--deep" in result["detail"]


def test_check_engine_deep_true_passes_for_qualified_build_identity(
        isolated, tmp_path, monkeypatch):
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")
    build_info = {
        "engine_version": "0.1.0",
        "build_id": "development",
        "protocol_version": "1.0",
        "backend": "cpu",
        "llama_cpp_commit": "88a39274ecf88ba11686acd357b59685b1cbf03d",
        "feature_flags": {"lora": True},
    }
    monkeypatch.setattr(
        "clozn.setup.install.qualify_entrypoint",
        lambda argv, *, expected=None: {
            "ran": True, "qualified": True, "returncode": 0,
            "stdout": json.dumps(build_info), "stderr": "", "build_info": build_info,
        },
    )
    result = doctor._check_engine(deep=True)
    assert result["status"] == "OK"
    assert "identity verified" in result["detail"]


def test_check_core_inference_qualification_is_always_warn_not_ok():
    result = doctor._check_core_inference_qualification()
    assert result["status"] == "WARN"
    assert "not verified" in result["detail"]
    assert "clozn run" in result["detail"]


# ----------------------------------------------------------------------------------------------- _run_all

def test_run_all_includes_only_retained_runtime_qualification_states(isolated):
    labels = [c["label"] for c in doctor._run_all()]
    assert "python package installed" in labels
    assert "engine binary" in labels
    assert "core inference qualification" in labels
    assert "white-box qualification" not in labels


def test_run_all_deep_false_by_default(isolated, tmp_path, monkeypatch):
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")

    def _boom(*a, **k):
        raise AssertionError("deep check must not run by default")

    monkeypatch.setattr("clozn.setup.install.qualify_entrypoint", _boom)
    doctor._run_all()   # must not raise


def test_run_all_deep_true_threads_through_to_check_engine(isolated, tmp_path):
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")
    checks = doctor._run_all(deep=True)
    engine_row = next(c for c in checks if c["label"] == "engine binary")
    assert "--deep" in engine_row["detail"]


# ---------------------------------------------------------------------------------------------- cmd_doctor

def test_cmd_doctor_json_includes_retained_runtime_rows(isolated, capsys):
    args = argparse.Namespace(json=True, verify_offline=False, deep=False)
    doctor.cmd_doctor(args)
    out = json.loads(capsys.readouterr().out)
    labels = [c["label"] for c in out["checks"]]
    assert {"python package installed", "engine binary", "core inference qualification"}.issubset(set(labels))
    assert "white-box qualification" not in labels


def test_cmd_doctor_without_deep_flag_attribute_defaults_to_false(isolated, capsys):
    """A caller that constructs args without ever setting `deep` (e.g. an older test, or a future
    subcommand reusing cmd_doctor) must not crash -- getattr(..., False) is the fallback."""
    args = argparse.Namespace(json=True, verify_offline=False)
    doctor.cmd_doctor(args)   # must not raise AttributeError
    out = json.loads(capsys.readouterr().out)
    assert out["status"] in ("OK", "WARN", "FAIL")


def test_cmd_doctor_deep_flag_reaches_the_engine_check(isolated, tmp_path, capsys):
    _touch_exe(tmp_path, "engine_core/build-cpu/clozn-server.exe")
    args = argparse.Namespace(json=True, verify_offline=False, deep=True)
    doctor.cmd_doctor(args)
    out = json.loads(capsys.readouterr().out)
    engine_row = next(c for c in out["checks"] if c["label"] == "engine binary")
    assert "--deep" in engine_row["detail"]


# --------------------------------------------------------------------------------------- outbound disclosure

def test_only_the_supported_pull_command_is_disclosed_as_outbound_capable():
    result = doctor._check_offline()
    assert result["known_outbound_capable_commands"] == [
        "clozn pull <model> -- downloads a GGUF from HuggingFace",
    ]


# --------------------------------------------------------------------------------------- argparse wiring

def test_doctor_deep_flag_parses_via_the_real_parser():
    from clozn.cli.main import build_parser
    parser = build_parser()
    args = parser.parse_args(["doctor", "--deep", "--json"])
    assert args.deep is True
    assert args.json is True
