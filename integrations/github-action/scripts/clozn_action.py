#!/usr/bin/env python3
"""GitHub Action orchestration for the Clozn model gate.

This file is intentionally stdlib-only and is designed to live in the
dedicated ``bkawa-io/clozn-action`` repository.  It does not import Clozn until
after the exact requested release has been installed.

``verify`` consumes an existing experiment result and invokes only
``clozn ci check --experiment``.  It never parses a model lock, fetches a
model, installs an engine, or starts a worker.

``run`` is a separate trusted-event path.  It validates the suite and lock
before network work, installs a released engine, fetches SHA-pinned models
into a cache, invokes one direct argv producer command (never a shell string),
then uses the same verify operation and always requests worker cleanup.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

ACTION_RESULT_SCHEMA = "clozn.action-result.v1"
COMMENT_MARKER = "<!-- clozn-model-gate -->"
ALLOWED_RUN_EVENTS = frozenset({"push", "workflow_dispatch", "schedule"})
_VERSION_RE = re.compile(r"^[0-9]+[.][0-9]+[.][0-9]+(?:[-+._a-zA-Z0-9]*)$")
_URL_RE = re.compile(r"(?i)https?://[^\s'\"]+")
_BUDGET_INPUTS = {
    "max_execution_errors": "--max-execution-errors",
    "max_target_regressions": "--max-target-regressions",
    "max_guard_regressions": "--max-guard-regressions",
    "min_target_gains": "--min-target-gains",
}


class ActionError(ValueError):
    """A configuration or orchestration error with a privacy-safe message."""


def _safe_error(exc: Exception) -> str:
    return _URL_RE.sub("<redacted-url>", f"{type(exc).__name__}: {exc}")


def _truthy(value: object) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "on"}


def _canonical_bytes(value) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _load_json(path: Path, label: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ActionError(f"could not read {label}") from exc
    except json.JSONDecodeError as exc:
        raise ActionError(f"{label} is not valid JSON") from exc


def _within_workspace(raw: str, *, workspace: Path, label: str, must_exist: bool) -> Path:
    if not isinstance(raw, str) or not raw.strip():
        raise ActionError(f"{label} is required")
    value = Path(raw).expanduser()
    if not value.is_absolute():
        value = workspace / value
    value = value.resolve()
    try:
        value.relative_to(workspace)
    except ValueError:
        raise ActionError(f"{label} must stay within GITHUB_WORKSPACE") from None
    if must_exist and not value.is_file():
        raise ActionError(f"{label} does not name an existing file")
    return value


def _parse_nonnegative_int(value: object, label: str) -> int:
    try:
        parsed = int(str(value))
    except (TypeError, ValueError):
        raise ActionError(f"{label} must be a non-negative integer") from None
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ActionError(f"{label} must be a non-negative integer")
    return parsed


def _parse_argv(value: object) -> list[str]:
    try:
        parsed = json.loads(str(value or ""))
    except json.JSONDecodeError:
        raise ActionError("producer-argv must be a JSON array of strings") from None
    if (
        not isinstance(parsed, list)
        or not parsed
        or any(not isinstance(item, str) or not item for item in parsed)
    ):
        raise ActionError("producer-argv must be a non-empty JSON array of non-empty strings")
    executable = Path(parsed[0]).name.casefold()
    if executable in {
        "bash",
        "bash.exe",
        "cmd",
        "cmd.exe",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
    }:
        raise ActionError(
            "producer-argv must execute a program directly; shell interpreters are refused")
    return parsed


def _write_github_output(values: dict, path: str | None = None) -> None:
    target = path or os.environ.get("GITHUB_OUTPUT")
    if not target:
        return
    with open(target, "a", encoding="utf-8") as handle:
        for key, value in values.items():
            text = str(value)
            if "\n" in text or "\r" in text:
                raise ActionError(f"GitHub output {key!r} is not single-line")
            handle.write(f"{key}={text}\n")


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(_canonical_bytes(value))
    os.replace(temporary, path)


def _state_paths(temp_root: Path) -> dict:
    root = temp_root / "clozn-action"
    root.mkdir(parents=True, exist_ok=True)
    return {
        "root": str(root),
        "state": str(root / "state.json"),
        "result": str(root / "action-result.json"),
        "report": str(root / "clozn-ci-report.json"),
        "summary": str(root / "clozn-summary.md"),
        "junit": str(root / "clozn-junit.xml"),
        "receipts": str(root / "clozn-receipts.zip"),
        "model_cache": str(root / "models"),
    }


def _cache_identity(
    *, manifest: Path, lockfile: Path, adapters: list[Path],
    clozn_version: str, engine_version: str,
) -> dict:
    """Compute a redacted cache/run identity.

    The lock's artifact URL is deliberately excluded.  SHA, size, quantization,
    and template identity define the artifact; moving identical bytes between
    mirrors must not create a different identity.
    """
    lock = _load_json(lockfile, "model lock")
    manifest_document = _load_json(manifest, "experiment manifest")
    _validate_run_documents(manifest_document, lock)
    models = {}
    for role, pinned in sorted((lock.get("models") or {}).items()):
        if not isinstance(pinned, dict):
            continue
        models[str(role)] = {
            key: pinned.get(key)
            for key in ("sha256", "size_bytes", "quantization", "chat_template_sha256")
            if pinned.get(key) is not None
        }
    material = {
        "identity_version": 1,
        "clozn_version": clozn_version,
        "engine_version": engine_version,
        "models": models,
        "suite_manifest_sha256": hashlib.sha256(
            _canonical_bytes(manifest_document)).hexdigest(),
        "adapters": [
            {"name": path.name, "sha256": _sha256_file(path)}
            for path in sorted(adapters, key=lambda item: str(item))
        ],
    }
    digest = hashlib.sha256(_canonical_bytes(material)).hexdigest()
    return {"sha256": digest, "material": material}


def _validate_run_documents(manifest: dict, lock: dict) -> None:
    """Fail obvious suite/lock defects before an Action cache is restored.

    Clozn's released validators run again before fetch/generation and remain
    authoritative.  This standalone check exists because the cache step occurs
    before the package is installed.
    """
    if not isinstance(manifest, dict) or manifest.get("schema_version") != "clozn.experiment.v0":
        raise ActionError("manifest must be a clozn.experiment.v0 document")
    variants = manifest.get("variants")
    names = [
        item.get("name") for item in variants or []
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    if not isinstance(variants, list) or len(variants) < 2 or len(names) != len(variants):
        raise ActionError("manifest must contain at least two named variants")
    if len(set(names)) != len(names) or manifest.get("baseline_variant") not in set(names):
        raise ActionError("manifest variant names/baseline are invalid")
    seeds = manifest.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        raise ActionError("manifest seeds must be unique integers")
    suites = manifest.get("suites")
    for role in ("target", "guard"):
        cases = _dict(_dict(suites).get(role)).get("cases")
        if not isinstance(cases, list) or not cases:
            raise ActionError(f"manifest suites.{role}.cases must be non-empty")

    if not isinstance(lock, dict) or lock.get("schema_version") != "clozn.model-lock.v1":
        raise ActionError("model-lock must be a clozn.model-lock.v1 document")
    models = lock.get("models")
    if not isinstance(models, dict) or not models:
        raise ActionError("model-lock must pin at least one model role")
    for role, pinned in models.items():
        if not isinstance(role, str) or not role or not isinstance(pinned, dict):
            raise ActionError("model-lock contains an invalid role entry")
        url, sha = pinned.get("url"), pinned.get("sha256")
        if not isinstance(url, str) or not url.casefold().startswith("https://"):
            raise ActionError(f"model-lock role {role!r} must use HTTPS")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise ActionError(f"model-lock role {role!r} must pin lowercase SHA-256")
        size = pinned.get("size_bytes")
        if size is not None and (
            not isinstance(size, int) or isinstance(size, bool) or size < 0
        ):
            raise ActionError(f"model-lock role {role!r} has an invalid size")


def prepare_environment(env: dict[str, str] | None = None) -> dict:
    """Validate inputs before installation/network work and write action state."""
    env = dict(os.environ if env is None else env)
    mode = env.get("INPUT_MODE", "verify").strip().casefold()
    if mode not in {"verify", "run"}:
        raise ActionError("mode must be 'verify' or 'run'")

    version = env.get("INPUT_CLOZN_VERSION", "").strip()
    if not _VERSION_RE.fullmatch(version):
        raise ActionError("clozn-version must be one exact release version")
    workspace = Path(env.get("GITHUB_WORKSPACE") or os.getcwd()).resolve()
    temp_root = Path(env.get("RUNNER_TEMP") or workspace / ".clozn-action-tmp").resolve()
    paths = _state_paths(temp_root)
    evidence = _within_workspace(
        env.get("INPUT_EVIDENCE", ""), workspace=workspace, label="evidence",
        must_exist=(mode == "verify"),
    )
    budgets = {
        key: _parse_nonnegative_int(env.get(f"INPUT_{key.upper()}", "0"), key)
        for key in _BUDGET_INPUTS
    }
    config = {
        "schema_version": ACTION_RESULT_SCHEMA,
        "mode": mode,
        "clozn_version": version,
        "workspace": str(workspace),
        "evidence": str(evidence),
        "budgets": budgets,
        "receipt_bundle": _truthy(env.get("INPUT_RECEIPT_BUNDLE", "true")),
        "comment": env.get("INPUT_COMMENT", "off").strip().casefold(),
        "paths": paths,
    }
    if config["comment"] not in {"off", "auto"}:
        raise ActionError("comment must be 'off' or 'auto'")

    if mode == "verify":
        forbidden = {
            "manifest": env.get("INPUT_MANIFEST", ""),
            "model-lock": env.get("INPUT_MODEL_LOCK", ""),
            "producer-argv": env.get("INPUT_PRODUCER_ARGV", ""),
            "engine-version": env.get("INPUT_ENGINE_VERSION", ""),
            "adapter-artifacts": env.get("INPUT_ADAPTER_ARTIFACTS", ""),
        }
        supplied = [name for name, value in forbidden.items() if str(value).strip()]
        if supplied:
            raise ActionError(
                "verify mode refuses run-only inputs: " + ", ".join(sorted(supplied)))
        config["cache_identity"] = None
    else:
        if not _truthy(env.get("INPUT_TRUSTED_RUN")):
            raise ActionError("run mode requires trusted-run=true")
        event = env.get("GITHUB_EVENT_NAME", "")
        if event not in ALLOWED_RUN_EVENTS:
            raise ActionError(
                "run mode is limited to trusted push, workflow_dispatch, or schedule events")
        manifest = _within_workspace(
            env.get("INPUT_MANIFEST", ""), workspace=workspace,
            label="manifest", must_exist=True)
        lockfile = _within_workspace(
            env.get("INPUT_MODEL_LOCK", ""), workspace=workspace,
            label="model-lock", must_exist=True)
        engine_version = env.get("INPUT_ENGINE_VERSION", "").strip()
        if not _VERSION_RE.fullmatch(engine_version):
            raise ActionError("run mode requires one exact engine-version")
        adapter_paths = []
        for raw in env.get("INPUT_ADAPTER_ARTIFACTS", "").splitlines():
            if raw.strip():
                adapter_paths.append(_within_workspace(
                    raw.strip(), workspace=workspace,
                    label="adapter-artifacts entry", must_exist=True))
        producer_argv = _parse_argv(env.get("INPUT_PRODUCER_ARGV"))
        identity = _cache_identity(
            manifest=manifest,
            lockfile=lockfile,
            adapters=adapter_paths,
            clozn_version=version,
            engine_version=engine_version,
        )
        config.update({
            "manifest": str(manifest),
            "model_lock": str(lockfile),
            "engine_version": engine_version,
            "adapter_artifacts": [str(path) for path in adapter_paths],
            "producer_argv": producer_argv,
            "cache_identity": identity,
        })

    _atomic_json(Path(paths["state"]), config)
    outputs = {
        "mode": mode,
        "state_file": paths["state"],
        "result_path": paths["result"],
        "report_path": paths["report"],
        "summary_path": paths["summary"],
        "junit_path": paths["junit"],
        "receipt_bundle_path": paths["receipts"],
        "model_cache": paths["model_cache"],
        "cache_key": (
            f"clozn-run-{config['cache_identity']['sha256']}"
            if config.get("cache_identity") else "verify-no-model-cache"
        ),
    }
    _write_github_output(outputs, env.get("GITHUB_OUTPUT"))
    return config


def install_clozn(state: dict, *, run=subprocess.run) -> int:
    """Install exactly one released Clozn version without shell interpolation."""
    package = f"clozn=={state['clozn_version']}"
    completed = run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "--no-input",
        package,
    ], check=False)
    return int(completed.returncode)


def _verify_argv(state: dict) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "clozn",
        "ci",
        "check",
        "--experiment",
        state["evidence"],
        "--report",
        state["paths"]["report"],
        "--github-summary",
        state["paths"]["summary"],
        "--junit-report",
        state["paths"]["junit"],
    ]
    for key, flag in _BUDGET_INPUTS.items():
        argv.extend([flag, str(state["budgets"][key])])
    return argv


def _fallback_summary(state: dict, exit_code: int, reason: str) -> None:
    text = (
        "# clozn ci check -- ERROR\n\n"
        f"{reason}\n\n"
        f"- mode: `{state['mode']}`\n"
        f"- exit code: `{exit_code}`\n"
        "- report: unavailable; no evidence path was fabricated\n"
    )
    Path(state["paths"]["summary"]).write_text(text, encoding="utf-8")


def _fallback_junit(state: dict, reason: str) -> None:
    suites = ET.Element("testsuites")
    suite = ET.SubElement(suites, "testsuite", {
        "name": "clozn action",
        "tests": "1",
        "failures": "0",
        "errors": "1",
        "time": "0",
    })
    case = ET.SubElement(suite, "testcase", {
        "classname": "clozn.action",
        "name": "orchestration",
        "time": "0",
    })
    error = ET.SubElement(case, "error", {"message": reason})
    error.text = reason
    Path(state["paths"]["junit"]).write_bytes(
        ET.tostring(suites, encoding="utf-8", xml_declaration=True))


def _artifact_checksums(state: dict) -> dict:
    out = {}
    for key in ("evidence",):
        path = Path(state[key])
        if path.is_file():
            out[key] = {"path": str(path), "sha256": _sha256_file(path)}
    for key in ("report", "junit", "receipts"):
        path = Path(state["paths"][key])
        if path.is_file():
            out[key] = {"path": str(path), "sha256": _sha256_file(path)}
    return out


def _append_artifact_summary(state: dict, checksums: dict) -> None:
    path = Path(state["paths"]["summary"])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n## Artifact checksums\n")
        for name, detail in sorted(checksums.items()):
            handle.write(f"- `{name}`: `{detail['sha256']}`\n")
        if state.get("cache_identity"):
            handle.write(
                f"- `run cache identity`: `{state['cache_identity']['sha256']}`\n")


def _append_github_summary(summary_path: Path, github_summary: str | None) -> None:
    if not github_summary:
        return
    with open(github_summary, "a", encoding="utf-8") as destination:
        destination.write(summary_path.read_text(encoding="utf-8"))


def _expand_producer_argv(state: dict, fetched: dict[str, str]) -> list[str]:
    replacements = {
        "{evidence}": state["evidence"],
        "{manifest}": state["manifest"],
        "{model_cache}": state["paths"]["model_cache"],
    }
    replacements.update({f"{{model:{role}}}": path for role, path in fetched.items()})
    out = []
    for item in state["producer_argv"]:
        for marker, value in replacements.items():
            item = item.replace(marker, value)
        if "{model:" in item:
            raise ActionError("producer-argv references a model role absent from model-lock")
        out.append(item)
    return out


def _prepare_run_mode(state: dict, *, run=subprocess.run) -> tuple[int, dict[str, str]]:
    """Validate first, then install/fetch. Returns (status, fetched role paths)."""
    # These imports occur only on the trusted run path, after the pinned package
    # install.  Verify mode cannot reach them.
    from clozn.experiments.suite import load_manifest
    from clozn.models.lockfile import load_lockfile, model_roles

    load_manifest(state["manifest"])
    lock = load_lockfile(state["model_lock"])

    fetched = {}
    for role in model_roles(lock):
        command = [
            sys.executable,
            "-m",
            "clozn",
            "model-lock",
            "fetch",
            state["model_lock"],
            "--role",
            role,
            "--out",
            state["paths"]["model_cache"],
            "--json",
        ]
        completed = run(command, check=False, capture_output=True, text=True)
        if completed.returncode:
            return int(completed.returncode), {}
        try:
            result = json.loads(completed.stdout)
            fetched[role] = str(result["path"])
        except (json.JSONDecodeError, KeyError, TypeError):
            raise ActionError("model-lock fetch returned an invalid machine-readable result") from None

    setup_argv = [
        sys.executable,
        "-m",
        "clozn",
        "setup",
        "--version",
        state["engine_version"],
        "--json",
    ]
    setup = run(setup_argv, check=False)
    if setup.returncode:
        return int(setup.returncode), {}
    return 0, fetched


def _cleanup_workers(*, run=subprocess.run) -> int:
    completed = run(
        [sys.executable, "-m", "clozn", "stop", "all"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return int(completed.returncode)


def execute_state(
    state: dict,
    *,
    run=subprocess.run,
    github_summary: str | None = None,
    install: bool = False,
) -> dict:
    """Execute one mode, preserving the first real non-zero command status."""
    install_exit = None
    producer_exit = None
    verify_exit = None
    orchestration_error = None
    cleanup_requested = state["mode"] == "run"
    try:
        if install:
            install_exit = install_clozn(state, run=run)
            if install_exit:
                orchestration_error = "the pinned Clozn release could not be installed"
        if install_exit in (None, 0) and state["mode"] == "run":
            preparation_exit, fetched = _prepare_run_mode(state, run=run)
            if preparation_exit:
                producer_exit = preparation_exit
            else:
                producer = run(
                    _expand_producer_argv(state, fetched),
                    check=False,
                    cwd=state["workspace"],
                )
                producer_exit = int(producer.returncode)

        evidence_exists = Path(state["evidence"]).is_file()
        if install_exit in (None, 0) and (state["mode"] == "verify" or evidence_exists):
            verify_env = dict(os.environ)
            verify_env["CLOZN_LOCAL_ONLY"] = "1"
            verified = run(_verify_argv(state), check=False, env=verify_env)
            verify_exit = int(verified.returncode)
        elif install_exit in (None, 0) and producer_exit in (None, 0):
            producer_exit = 2
            orchestration_error = "run mode produced no experiment evidence"
    except Exception as exc:
        orchestration_error = _safe_error(exc)
        if producer_exit in (None, 0):
            producer_exit = 2
    finally:
        if cleanup_requested:
            _cleanup_workers(run=run)

    if install_exit not in (None, 0):
        exit_code = 2
    elif producer_exit not in (None, 0):
        exit_code = producer_exit
    else:
        exit_code = verify_exit if verify_exit is not None else producer_exit or 0
    summary_path = Path(state["paths"]["summary"])
    junit_path = Path(state["paths"]["junit"])
    if not summary_path.is_file():
        _fallback_summary(
            state,
            int(exit_code),
            orchestration_error or "Clozn verification did not produce a report",
        )
    if not junit_path.is_file():
        _fallback_junit(
            state,
            orchestration_error or "Clozn verification did not produce JUnit output",
        )

    receipt_result = None
    report_path = Path(state["paths"]["report"])
    if (
        state.get("receipt_bundle")
        and report_path.is_file()
        and Path(state["evidence"]).is_file()
    ):
        try:
            from clozn.receipts.ci_bundle import build_indexed_bundle

            report = _load_json(report_path, "CI report")
            evidence = _load_json(Path(state["evidence"]), "experiment evidence")
            receipt_result = build_indexed_bundle(
                report, evidence, state["paths"]["receipts"])
        except Exception as exc:
            receipt_result = {
                "ok": False,
                "error": _safe_error(exc),
                "evidence_unavailable": True,
            }

    checksums = _artifact_checksums(state)
    _append_artifact_summary(state, checksums)
    _append_github_summary(summary_path, github_summary)
    result = {
        "schema_version": ACTION_RESULT_SCHEMA,
        "mode": state["mode"],
        "exit_code": int(exit_code),
        "install_exit_code": install_exit,
        "producer_exit_code": producer_exit,
        "verify_exit_code": verify_exit,
        "orchestration_error": orchestration_error,
        "cleanup_requested": cleanup_requested,
        "cache_identity": (
            state.get("cache_identity", {}).get("sha256")
            if state.get("cache_identity") else None
        ),
        "artifacts": checksums,
        "receipt_bundle": receipt_result,
    }
    _atomic_json(Path(state["paths"]["result"]), result)
    return result


def _comment_body(result: dict, report: dict | None, run_url: str | None) -> str:
    exit_code = int(result.get("exit_code", 2))
    status = "PASS" if exit_code == 0 else "FAIL"
    lines = [
        COMMENT_MARKER,
        f"### Clozn model gate: {status}",
        "",
        f"- mode: `{result.get('mode', '?')}`",
        f"- exit code: `{exit_code}`",
    ]
    if isinstance(report, dict):
        checks = report.get("checks") or {}
        failed = [name for name, check in checks.items() if not check.get("passed")]
        lines.append(f"- failed checks: `{', '.join(failed) if failed else 'none'}`")
        entries = (report.get("receipt_index") or {}).get("entries") or []
        available = sum(1 for entry in entries if entry.get("run_id"))
        unavailable = len(entries) - available
        lines.append(f"- receipt evidence: `{available} indexed, {unavailable} unavailable`")
    if run_url:
        lines.extend(["", f"[Open workflow run]({run_url})"])
    lines.extend([
        "",
        "_Prompts, responses, and raw receipt content are not included in this comment._",
    ])
    return "\n".join(lines)


def publish_comment(
    state: dict,
    result: dict,
    *,
    env: dict[str, str] | None = None,
    urlopen=urllib.request.urlopen,
) -> str:
    """Create/update one marker comment; permissions failure is a degradation."""
    env = dict(os.environ if env is None else env)
    if state.get("comment") != "auto":
        return "disabled"
    if env.get("GITHUB_EVENT_NAME") != "pull_request":
        return "not_pull_request"
    token = env.get("GITHUB_TOKEN")
    event_path = env.get("GITHUB_EVENT_PATH")
    repository = env.get("GITHUB_REPOSITORY")
    if not token or not event_path or not repository:
        return "read_only"
    try:
        event = _load_json(Path(event_path), "GitHub event")
        number = int(_dict(event.get("pull_request")).get("number") or event.get("number"))
    except Exception:
        return "read_only"

    api = env.get("GITHUB_API_URL", "https://api.github.com").rstrip("/")
    comments_url = f"{api}/repos/{repository}/issues/{number}/comments"
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "User-Agent": "clozn-action",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    report = None
    report_path = Path(state["paths"]["report"])
    if report_path.is_file():
        try:
            report = _load_json(report_path, "CI report")
        except ActionError:
            pass
    body = _comment_body(result, report, env.get("GITHUB_RUN_URL"))
    try:
        request = urllib.request.Request(comments_url, headers=headers)
        with urlopen(request, timeout=15) as response:
            comments = json.loads(response.read().decode("utf-8"))
        existing = next(
            (
                item for item in comments
                if isinstance(item, dict)
                and COMMENT_MARKER in str(item.get("body") or "")
            ),
            None,
        )
        payload = json.dumps({"body": body}).encode("utf-8")
        if existing and existing.get("id"):
            target = f"{api}/repos/{repository}/issues/comments/{existing['id']}"
            method = "PATCH"
            status = "updated"
        else:
            target = comments_url
            method = "POST"
            status = "created"
        request = urllib.request.Request(
            target,
            data=payload,
            headers={**headers, "Content-Type": "application/json"},
            method=method,
        )
        with urlopen(request, timeout=15):
            return status
    except (OSError, urllib.error.HTTPError, urllib.error.URLError, json.JSONDecodeError):
        return "read_only"


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _load_state(path: str) -> dict:
    state = _load_json(Path(path), "action state")
    if state.get("schema_version") != ACTION_RESULT_SCHEMA:
        raise ActionError("action state has an unsupported schema")
    return state


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: clozn_action.py prepare|install|execute|comment|propagate", file=sys.stderr)
        return 2
    command = argv.pop(0)
    try:
        if command == "prepare":
            prepare_environment()
            return 0
        if not argv:
            raise ActionError(f"{command} requires a state-file path")
        state = _load_state(argv[0])
        if command == "install":
            return install_clozn(state)
        if command == "execute":
            result = execute_state(
                state,
                github_summary=os.environ.get("GITHUB_STEP_SUMMARY"),
                install=True,
            )
            _write_github_output({
                "exit_code": result["exit_code"],
                "result_path": state["paths"]["result"],
                "report_path": state["paths"]["report"],
                "summary_path": state["paths"]["summary"],
                "junit_path": state["paths"]["junit"],
                "receipt_bundle_path": state["paths"]["receipts"],
            })
            # Artifact/summary steps must run before exact propagation.
            return 0
        if command == "comment":
            result = _load_json(Path(state["paths"]["result"]), "action result")
            status = publish_comment(state, result)
            _write_github_output({"comment_status": status})
            return 0
        if command == "propagate":
            result = _load_json(Path(state["paths"]["result"]), "action result")
            return int(result.get("exit_code", 2))
        raise ActionError(f"unknown command {command!r}")
    except ActionError as exc:
        print(f"clozn-action: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
