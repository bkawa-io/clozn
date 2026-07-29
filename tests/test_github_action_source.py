"""Model-free contract tests for the publication-ready GitHub Action source."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import types

import pytest


ROOT = Path(__file__).resolve().parents[1]
ACTION_ROOT = ROOT / "integrations" / "github-action"
SCRIPT = ACTION_ROOT / "scripts" / "clozn_action.py"
SPEC = importlib.util.spec_from_file_location("clozn_action_source", SCRIPT)
action = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(action)


def _write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _base_env(tmp_path: Path) -> dict[str, str]:
    evidence = _write_json(tmp_path / "evidence.json", {
        "schema_version": "clozn.experiment.result.v0",
        "cells": [],
    })
    return {
        "GITHUB_WORKSPACE": str(tmp_path),
        "RUNNER_TEMP": str(tmp_path / "temp"),
        "GITHUB_EVENT_NAME": "pull_request",
        "INPUT_MODE": "verify",
        "INPUT_CLOZN_VERSION": "0.1.0",
        "INPUT_EVIDENCE": str(evidence),
        "INPUT_MAX_EXECUTION_ERRORS": "0",
        "INPUT_MAX_TARGET_REGRESSIONS": "0",
        "INPUT_MAX_GUARD_REGRESSIONS": "0",
        "INPUT_MIN_TARGET_GAINS": "0",
        "INPUT_RECEIPT_BUNDLE": "false",
        "INPUT_COMMENT": "off",
    }


def _run_env(tmp_path: Path) -> dict[str, str]:
    env = _base_env(tmp_path)
    manifest = _write_json(tmp_path / "experiment.json", {
        "schema_version": "clozn.experiment.v0",
        "name": "gate",
        "seeds": [0],
        "baseline_variant": "base",
        "defaults": {},
        "variants": [
            {"name": "base", "kind": "base"},
            {"name": "candidate", "kind": "prompt", "system_prompt": "changed"},
        ],
        "suites": {
            "target": {"cases": [{"name": "target", "prompt": "prompt"}]},
            "guard": {"cases": [{"name": "guard", "prompt": "prompt"}]},
        },
    })
    lockfile = _write_json(tmp_path / "clozn.lock.json", {
        "schema_version": "clozn.model-lock.v1",
        "models": {
            "baseline": {
                "url": "https://example.com/base.gguf",
                "sha256": "a" * 64,
                "chat_template_sha256": "b" * 64,
            },
            "candidate": {
                "url": "https://example.com/candidate.gguf",
                "sha256": "c" * 64,
                "chat_template_sha256": "d" * 64,
            },
        },
    })
    adapter = tmp_path / "adapter.gguf"
    adapter.write_bytes(b"adapter-v1")
    # Evidence is an output in run mode, not a pre-existing input.
    Path(env["INPUT_EVIDENCE"]).unlink()
    env.update({
        "GITHUB_EVENT_NAME": "workflow_dispatch",
        "INPUT_MODE": "run",
        "INPUT_TRUSTED_RUN": "true",
        "INPUT_MANIFEST": str(manifest),
        "INPUT_MODEL_LOCK": str(lockfile),
        "INPUT_ENGINE_VERSION": "0.1.0",
        "INPUT_ADAPTER_ARTIFACTS": str(adapter),
        "INPUT_PRODUCER_ARGV": json.dumps([
            "python", "produce.py", "--out", "{evidence}",
            "--candidate", "{model:candidate}",
        ]),
    })
    return env


def test_verify_mode_is_engine_free_and_rejects_every_run_input(tmp_path):
    env = _base_env(tmp_path)
    state = action.prepare_environment(env)
    command = action._verify_argv(state)
    joined = " ".join(command)
    assert "ci check --experiment" in joined
    assert "model-lock" not in joined
    assert "setup" not in joined
    assert "serve" not in joined
    assert state["cache_identity"] is None

    for key, value in (
        ("INPUT_MANIFEST", "experiment.json"),
        ("INPUT_MODEL_LOCK", "lock.json"),
        ("INPUT_ENGINE_VERSION", "0.1.0"),
        ("INPUT_ADAPTER_ARTIFACTS", "adapter.gguf"),
        ("INPUT_PRODUCER_ARGV", '["python", "produce.py"]'),
    ):
        changed = dict(env)
        changed[key] = value
        with pytest.raises(action.ActionError, match="run-only inputs"):
            action.prepare_environment(changed)


def test_verify_exit_is_preserved_after_summary_and_junit(tmp_path):
    state = action.prepare_environment(_base_env(tmp_path))
    calls = []

    def fake_run(argv, **_kwargs):
        assert _kwargs["env"]["CLOZN_LOCAL_ONLY"] == "1"
        calls.append(list(argv))
        report = {
            "schema_version": "clozn.ci-report.v1",
            "receipt_index": {"privacy": "metadata_only", "entries": []},
        }
        _write_json(Path(state["paths"]["report"]), report)
        Path(state["paths"]["summary"]).write_text("# failing gate\n", encoding="utf-8")
        Path(state["paths"]["junit"]).write_text("<testsuites/>", encoding="utf-8")
        return types.SimpleNamespace(returncode=1)

    github_summary = tmp_path / "github-summary.md"
    result = action.execute_state(
        state, run=fake_run, github_summary=str(github_summary), install=False)
    assert result["exit_code"] == 1
    assert result["verify_exit_code"] == 1
    assert github_summary.read_text(encoding="utf-8").startswith("# failing gate")
    assert "Artifact checksums" in github_summary.read_text(encoding="utf-8")
    assert len(calls) == 1

    # The final command reads the stored result rather than recomputing or
    # collapsing it to a generic action failure.
    assert action.main(["propagate", state["paths"]["state"]]) == 1


@pytest.mark.parametrize("event", ["pull_request", "pull_request_target", "workflow_call"])
def test_run_mode_refuses_untrusted_event_families(tmp_path, event):
    env = _run_env(tmp_path)
    env["GITHUB_EVENT_NAME"] = event
    with pytest.raises(action.ActionError, match="trusted push"):
        action.prepare_environment(env)


def test_run_mode_requires_acknowledgement_and_refuses_shell_argv(tmp_path):
    env = _run_env(tmp_path)
    env["INPUT_TRUSTED_RUN"] = "false"
    with pytest.raises(action.ActionError, match="trusted-run=true"):
        action.prepare_environment(env)

    env = _run_env(tmp_path)
    env["INPUT_PRODUCER_ARGV"] = '["bash", "-c", "anything"]'
    with pytest.raises(action.ActionError, match="shell interpreters"):
        action.prepare_environment(env)


def test_cache_identity_invalidates_model_adapter_engine_template_and_suite(tmp_path):
    env = _run_env(tmp_path)
    first = action.prepare_environment(env)["cache_identity"]["sha256"]

    adapter = Path(env["INPUT_ADAPTER_ARTIFACTS"])
    adapter.write_bytes(b"adapter-v2")
    adapter_changed = action.prepare_environment(env)["cache_identity"]["sha256"]
    assert adapter_changed != first

    adapter.write_bytes(b"adapter-v1")
    env["INPUT_ENGINE_VERSION"] = "0.1.1"
    engine_changed = action.prepare_environment(env)["cache_identity"]["sha256"]
    assert engine_changed != first

    env["INPUT_ENGINE_VERSION"] = "0.1.0"
    lock = json.loads(Path(env["INPUT_MODEL_LOCK"]).read_text(encoding="utf-8"))
    lock["models"]["candidate"]["sha256"] = "e" * 64
    _write_json(Path(env["INPUT_MODEL_LOCK"]), lock)
    model_changed = action.prepare_environment(env)["cache_identity"]["sha256"]
    assert model_changed != first

    lock["models"]["candidate"]["sha256"] = "c" * 64
    lock["models"]["candidate"]["chat_template_sha256"] = "f" * 64
    _write_json(Path(env["INPUT_MODEL_LOCK"]), lock)
    template_changed = action.prepare_environment(env)["cache_identity"]["sha256"]
    assert template_changed != first

    lock["models"]["candidate"]["chat_template_sha256"] = "d" * 64
    _write_json(Path(env["INPUT_MODEL_LOCK"]), lock)
    manifest = json.loads(Path(env["INPUT_MANIFEST"]).read_text(encoding="utf-8"))
    manifest["suites"]["target"]["cases"][0]["prompt"] = "different suite"
    _write_json(Path(env["INPUT_MANIFEST"]), manifest)
    suite_changed = action.prepare_environment(env)["cache_identity"]["sha256"]
    assert suite_changed != first


def test_run_mode_requests_cleanup_after_producer_failure(tmp_path, monkeypatch):
    env = _run_env(tmp_path)
    state = action.prepare_environment(env)
    calls = []
    monkeypatch.setattr(
        action,
        "_prepare_run_mode",
        lambda _state, run: (0, {"candidate": str(tmp_path / "model.gguf")}),
    )

    def fake_run(argv, **_kwargs):
        calls.append(list(argv))
        if argv[1:4] == ["-m", "clozn", "stop"]:
            return types.SimpleNamespace(returncode=0)
        return types.SimpleNamespace(returncode=37)

    result = action.execute_state(state, run=fake_run, install=False)
    assert result["exit_code"] == 37
    assert result["producer_exit_code"] == 37
    assert result["cleanup_requested"] is True
    assert any(call[1:5] == ["-m", "clozn", "stop", "all"] for call in calls)


class _Response:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(self.value).encode("utf-8")


def test_comment_rerun_updates_marker_and_read_only_degrades(tmp_path):
    env = _base_env(tmp_path)
    env["INPUT_COMMENT"] = "auto"
    state = action.prepare_environment(env)
    result = {"exit_code": 1, "mode": "verify"}
    event = _write_json(tmp_path / "event.json", {"number": 12, "pull_request": {"number": 12}})
    publish_env = {
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_EVENT_PATH": str(event),
        "GITHUB_REPOSITORY": "owner/repo",
        "GITHUB_TOKEN": "not-logged",
        "GITHUB_API_URL": "https://api.github.com",
    }
    requests = []

    def fake_open(request, timeout):
        requests.append((request.full_url, request.get_method(), timeout))
        if request.get_method() == "GET":
            return _Response([{"id": 99, "body": action.COMMENT_MARKER + "\nold"}])
        return _Response({})

    assert action.publish_comment(
        state, result, env=publish_env, urlopen=fake_open) == "updated"
    assert requests[-1][0].endswith("/issues/comments/99")
    assert requests[-1][1] == "PATCH"

    def forbidden(*_args, **_kwargs):
        raise OSError("read only")

    assert action.publish_comment(
        state, result, env=publish_env, urlopen=forbidden) == "read_only"


def test_action_uses_immutable_third_party_shas_and_defers_publication():
    action_yml = (ACTION_ROOT / "action.yml").read_text(encoding="utf-8")
    assert "actions/setup-python@ece7cb06caefa5fff74198d8649806c4678c61a1" in action_yml
    assert "actions/cache@0057852bfaa89a56745cba8c7296529d2fc39830" in action_yml
    assert "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02" in action_yml
    assert "actions/setup-python@v" not in action_yml
    release = (ACTION_ROOT / "RELEASE.md").read_text(encoding="utf-8")
    assert "performs none" in release
    security = (ACTION_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "Never execute run mode on `pull_request_target`" in security
