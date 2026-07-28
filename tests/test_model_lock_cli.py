"""tests/test_model_lock_cli.py -- clozn/cli/commands/model_lock.py (`clozn model-lock verify FILE`).

Registered via CLOZN_AUTOLOAD (docs/SEAMS.md Seam 1); `test_model_lock_is_registered` proves the marker
actually wires it into `build_parser()` end to end, mirroring tests/test_ci_check.py's own
`test_ci_is_registered`. Model-free and network-free throughout -- this command never does more than
`clozn.models.lockfile.load_lockfile` already does.
"""
from __future__ import annotations

import json
import os
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)
FIXTURES = os.path.join(HERE, "fixtures", "schemas", "clozn.model-lock.v1")

import pytest  # noqa: E402

import clozn.cli.commands.model_lock as model_lock  # noqa: E402
from clozn.cli.main import build_parser  # noqa: E402


def _fixture(name: str) -> str:
    return os.path.join(FIXTURES, name)


# ==================================================================================================== argparse

def _subparser_choices(p):
    for a in p._actions:
        if getattr(a, "choices", None) and "model-lock" in a.choices:
            return a.choices
    return {}


def test_model_lock_is_registered():
    assert "model-lock" in _subparser_choices(build_parser())


def test_model_lock_autoload_marker_is_set():
    assert model_lock.CLOZN_AUTOLOAD is True


def test_model_lock_verify_parses():
    ns = build_parser().parse_args(["model-lock", "verify", "lock.json"])
    assert ns.lockfile == "lock.json"
    assert ns.json is False
    assert ns.fn is model_lock.cmd_model_lock_verify


def test_model_lock_fetch_parses_with_explicit_role_and_output_directory():
    ns = build_parser().parse_args([
        "model-lock", "fetch", "models/clozn.lock.json",
        "--role", "candidate", "--out", ".models",
    ])
    assert ns.lockfile == "models/clozn.lock.json"
    assert ns.role == "candidate"
    assert ns.out == ".models"
    assert ns.json is False
    assert ns.fn is model_lock.cmd_model_lock_fetch


def test_model_lock_no_subcommand_returns_2(capsys):
    ns = build_parser().parse_args(["model-lock"])
    rc = ns.fn(ns)
    assert rc == 2
    assert "model-lock verify" in capsys.readouterr().out


# =============================================================================================== cmd_model_lock_verify

def _args(**overrides):
    base = dict(lockfile=None, json=False)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_cmd_model_lock_verify_valid_lockfile_exit_0_text(capsys):
    rc = model_lock.cmd_model_lock_verify(_args(lockfile=_fixture("valid__two_models.json")))
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out and "baseline" in out and "candidate" in out


def test_cmd_model_lock_verify_valid_lockfile_json(capsys):
    rc = model_lock.cmd_model_lock_verify(_args(lockfile=_fixture("valid__two_models.json"), json=True))
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"ok": True, "path": _fixture("valid__two_models.json"),
                       "roles": ["baseline", "candidate"]}


def test_cmd_model_lock_verify_invalid_lockfile_exit_1_text(capsys):
    rc = model_lock.cmd_model_lock_verify(_args(lockfile=_fixture("invalid__missing_sha256.json")))
    assert rc == 1
    assert "does not conform" in capsys.readouterr().out


def test_cmd_model_lock_verify_invalid_lockfile_json(capsys):
    rc = model_lock.cmd_model_lock_verify(_args(lockfile=_fixture("invalid__non_https_url.json"),
                                                 json=True))
    assert rc == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert payload["path"] == _fixture("invalid__non_https_url.json")
    assert "error" in payload


def test_cmd_model_lock_verify_missing_file_exit_1(capsys):
    rc = model_lock.cmd_model_lock_verify(_args(lockfile=os.path.join(FIXTURES, "nope.json")))
    assert rc == 1
    assert "could not read lockfile" in capsys.readouterr().out
