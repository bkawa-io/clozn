"""Versioned local contract and copy-only GitHub workflow preview for experiment CI."""
from __future__ import annotations

import copy
import json
from pathlib import PurePosixPath
import shlex
from typing import Any, Mapping

from clozn.experiments import suite


ACTION_INPUT_SCHEMA = "clozn.experiment-action-inputs.v1"
ACTION_INPUT_CONTRACT = {
    "schema_version": ACTION_INPUT_SCHEMA,
    "mode": {"type": "string", "enum": ["verify"], "required": True},
    "result_path": {"type": "repository-relative-path", "required": True},
    "suite_path": {"type": "repository-relative-path", "required": False},
    "lockfile_path": {"type": "repository-relative-path", "required": False},
    "budgets": {
        "type": "object",
        "required": True,
        "fields": {
            "max_execution_errors": {"type": "nonnegative-integer", "default": 0},
            "max_target_regressions": {"type": "nonnegative-integer", "default": 0},
            "max_guard_regressions": {"type": "nonnegative-integer", "default": 0},
            "min_target_gains": {"type": "nonnegative-integer", "default": 0},
        },
    },
}


class ActionContractError(ValueError):
    pass


def _relative_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ActionContractError(f"{label} must be a non-empty repository-relative path")
    value = value.strip()
    if "\\" in value:
        raise ActionContractError(f"{label} must use '/' separators")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.startswith("~"):
        raise ActionContractError(f"{label} must stay within the checked-out repository")
    return value


def validate_inputs(value: Mapping[str, Any]) -> dict:
    if not isinstance(value, Mapping):
        raise ActionContractError("Action inputs must be an object")
    allowed = {"schema_version", "mode", "result_path", "suite_path", "lockfile_path", "budgets"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ActionContractError(f"unsupported Action inputs: {unknown!r}")
    if value.get("schema_version") != ACTION_INPUT_SCHEMA:
        raise ActionContractError(f"schema_version must be {ACTION_INPUT_SCHEMA!r}")
    if value.get("mode") != "verify":
        raise ActionContractError("mode must be 'verify' in v1")
    out = {
        "schema_version": ACTION_INPUT_SCHEMA,
        "mode": "verify",
        "result_path": _relative_path(value.get("result_path"), "result_path"),
    }
    for field in ("suite_path", "lockfile_path"):
        if value.get(field) is not None:
            out[field] = _relative_path(value[field], field)
    raw_budgets = value.get("budgets", {})
    if not isinstance(raw_budgets, Mapping):
        raise ActionContractError("budgets must be an object")
    budget_fields = ACTION_INPUT_CONTRACT["budgets"]["fields"]
    unknown_budgets = sorted(set(raw_budgets) - set(budget_fields))
    if unknown_budgets:
        raise ActionContractError(f"unsupported budgets: {unknown_budgets!r}")
    budgets = {}
    for name, definition in budget_fields.items():
        item = raw_budgets.get(name, definition["default"])
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise ActionContractError(f"budgets.{name} must be a non-negative integer")
        budgets[name] = item
    out["budgets"] = budgets
    return out


def _yaml_string(value: str) -> str:
    # JSON strings are valid YAML double-quoted scalars and give deterministic escaping.
    return json.dumps(value, ensure_ascii=False)


def _command(inputs: dict) -> str:
    budgets = inputs["budgets"]
    args = [
        "clozn", "ci", "check", "--experiment", inputs["result_path"],
        "--max-execution-errors", str(budgets["max_execution_errors"]),
        "--max-target-regressions", str(budgets["max_target_regressions"]),
        "--max-guard-regressions", str(budgets["max_guard_regressions"]),
        "--min-target-gains", str(budgets["min_target_gains"]),
        "--report", "clozn-ci/clozn-ci-report.json",
        "--github-summary", "clozn-ci/summary.md",
        "--junit-report", "clozn-ci/junit.xml",
    ]
    return " ".join(shlex.quote(item) for item in args)


def render_workflow(inputs: Mapping[str, Any], fingerprint: Mapping[str, str]) -> str:
    validated = validate_inputs(inputs)
    if fingerprint != {
        "algorithm": suite.FINGERPRINT_ALGORITHM,
        "sha256": fingerprint.get("sha256") if isinstance(fingerprint, Mapping) else None,
    }:
        raise ActionContractError("suite_fingerprint must use canonical-json-v1")
    digest = fingerprint.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ActionContractError("suite_fingerprint.sha256 must be a lowercase SHA-256 digest")
    cache_key = f"clozn-model-gate-{suite.FINGERPRINT_ALGORITHM}-{digest}"
    lines = [
        "name: Clozn model gate",
        "",
        "on:",
        "  workflow_dispatch:",
        "",
        "permissions:",
        "  contents: read",
        "",
        "jobs:",
        "  model-gate:",
        "    runs-on: ubuntu-latest",
        "    env:",
        f"      CLOZN_SUITE_FINGERPRINT: {_yaml_string(digest)}",
        "    steps:",
        "      - uses: actions/checkout@v4",
        "      - uses: actions/setup-python@v5",
        "        with:",
        "          python-version: \"3.11\"",
        "      - name: Restore Clozn cache",
        "        uses: actions/cache@v4",
        "        with:",
        "          path: ~/.cache/clozn",
        f"          key: {_yaml_string(cache_key)}",
        "      - name: Install checked-out Clozn",
        "        run: python -m pip install .",
    ]
    if "suite_path" in validated:
        lines.extend([
            "      - name: Verify configured suite is present",
            f"        run: test -f {shlex.quote(validated['suite_path'])}",
        ])
    if "lockfile_path" in validated:
        lines.extend([
            "      - name: Verify model lock",
            f"        run: clozn model-lock verify {shlex.quote(validated['lockfile_path'])}",
        ])
    lines.extend([
        "      - name: Run experiment gate",
        "        run: |",
        "          mkdir -p clozn-ci",
        f"          {_command(validated)}",
        "          cat clozn-ci/summary.md >> \"$GITHUB_STEP_SUMMARY\"",
        "      - name: Upload gate artifacts",
        "        if: always()",
        "        uses: actions/upload-artifact@v4",
        "        with:",
        "          name: clozn-model-gate",
        "          path: clozn-ci/",
        "",
    ])
    return "\n".join(lines)


def ci_preview(result: dict, raw_inputs: Mapping[str, Any]) -> dict:
    inputs = {
        "schema_version": ACTION_INPUT_SCHEMA,
        "mode": raw_inputs.get("mode", "verify"),
        "result_path": raw_inputs.get("result_path"),
        "budgets": copy.deepcopy(raw_inputs.get("budgets", {})),
    }
    for field in ("suite_path", "lockfile_path"):
        if raw_inputs.get(field) is not None:
            inputs[field] = raw_inputs[field]
    inputs = validate_inputs(inputs)
    fingerprint = suite.result_fingerprint(result)
    return {
        "schema_version": "clozn.experiment-ci-preview.v1",
        "input_contract": copy.deepcopy(ACTION_INPUT_CONTRACT),
        "inputs": inputs,
        "suite_fingerprint": fingerprint,
        "cache_key": (
            f"clozn-model-gate-{fingerprint['algorithm']}-{fingerprint['sha256']}"
        ),
        "workflow_yaml": render_workflow(inputs, fingerprint),
    }


__all__ = [
    "ACTION_INPUT_CONTRACT", "ACTION_INPUT_SCHEMA", "ActionContractError", "ci_preview",
    "render_workflow", "validate_inputs",
]
