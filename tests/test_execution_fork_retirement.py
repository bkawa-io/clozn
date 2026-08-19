"""Source-level guards for retiring the child-creating execution-fork executor.

Two separate claims are asserted here:

1.  Canonical product code -- the experimental kernel, the product recipes, and Branch Fan --
    never imports the legacy executor or the legacy planner adapters that create child Runs.
    Exact execution is allowed to reach the worker's low-level ``engine.execution_fork`` RPC, but
    only underneath the neutral GenerateExecutionAdapter.

2.  The remaining legacy callers are an explicit, exact inventory.  ``clozn.replay.execution_fork_execute``
    cannot be deleted while these exist, so the inventory names them.  It is shrink-only: removing a
    caller must update this list, and adding one fails the test.
"""
from __future__ import annotations

import ast
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(REPO_ROOT, "clozn")

LEGACY_EXECUTOR_MODULE = "clozn.replay.execution_fork_execute"

# The legacy planner surface that plans or performs a child-creating exact fork.  Identity and
# runtime-projection helpers are deliberately NOT listed: those are immutable execution facts whose
# neutral owner is clozn.experiments.execution_facts.
LEGACY_CHILD_CREATING_NAMES = (
    "plan_execution_fork",
    "capture_exact_force_token_context",
    "plan_exact_force_token",
    "execute_exact_force_token",
    "execute_exact_fork",
)

# Canonical product code: the kernel, the recipes over it, and the features already converged onto it.
CANONICAL_MODULES = (
    "clozn/experiments",
    "clozn/recipes",
    "clozn/replay/branch_fan.py",
    "clozn/replay/sampler_sensitivity.py",
    "clozn/replay/test_this.py",
    "clozn/replay/execution_fork.py",
    "clozn/server/routes/branch_fan.py",
    "clozn/server/routes/sampler_sensitivity.py",
    "clozn/server/routes/time_travel_v1.py",
)

# Every production module that still imports the legacy executor, and why it is still here.
# Retiring clozn/replay/execution_fork_execute.py requires this to become empty.
LEGACY_EXECUTOR_CALLERS = {
    # Time Machine still creates its continuation child through the legacy executor.  Its lifecycle
    # redesign is deliberately a separate change.
    "clozn/server/routes/timetravel.py",
    # The canonical checkpoint capture seam proves its unchanged control with the executor's helper.
    "clozn/replay/checkpoint_capture.py",
}


def _python_files(relative: str) -> list[str]:
    target = os.path.join(REPO_ROOT, relative)
    if os.path.isfile(target):
        return [target]
    found = []
    for directory, _dirs, names in os.walk(target):
        for name in sorted(names):
            if name.endswith(".py"):
                found.append(os.path.join(directory, name))
    return found


def _all_package_files() -> list[str]:
    found = []
    for directory, _dirs, names in os.walk(PACKAGE_ROOT):
        for name in sorted(names):
            if name.endswith(".py"):
                found.append(os.path.join(directory, name))
    return found


def _imports(path: str) -> set[str]:
    """Every module path and imported name referenced by an import statement in `path`."""
    # utf-8-sig: a couple of vendored modules carry a byte-order mark, which is not a syntax error
    # to Python's own loader and must not be one here either.
    with open(path, encoding="utf-8-sig") as handle:
        tree = ast.parse(handle.read(), filename=path)
    referenced: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                referenced.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            referenced.add(module)
            for alias in node.names:
                referenced.add(f"{module}.{alias.name}")
                referenced.add(alias.name)
    return referenced


def test_canonical_product_code_does_not_import_the_legacy_executor():
    offenders = []
    for relative in CANONICAL_MODULES:
        for path in _python_files(relative):
            if LEGACY_EXECUTOR_MODULE in _imports(path):
                offenders.append(os.path.relpath(path, REPO_ROOT))
    assert offenders == [], (
        f"canonical product code must not import {LEGACY_EXECUTOR_MODULE}: {offenders}")


def test_canonical_product_code_does_not_use_the_legacy_child_creating_planner():
    offenders = []
    for relative in CANONICAL_MODULES:
        for path in _python_files(relative):
            referenced = _imports(path)
            for name in LEGACY_CHILD_CREATING_NAMES:
                if name in referenced:
                    offenders.append((os.path.relpath(path, REPO_ROOT), name))
    assert offenders == [], f"canonical product code still imports legacy fork entry points: {offenders}"


def test_branch_fan_reaches_exact_execution_only_through_the_neutral_adapter():
    """Branch Fan may not call the worker RPC itself; the adapter owns that boundary."""
    path = os.path.join(REPO_ROOT, "clozn/replay/branch_fan.py")
    with open(path, encoding="utf-8") as handle:
        source = handle.read()
    assert "execution_fork(" not in source
    assert "GenerateExecutionAdapter" in source


def test_remaining_legacy_executor_callers_are_an_exact_inventory():
    found = set()
    for path in _all_package_files():
        if LEGACY_EXECUTOR_MODULE in _imports(path):
            found.add(os.path.relpath(path, REPO_ROOT))
    assert found == LEGACY_EXECUTOR_CALLERS, (
        "the legacy executor caller inventory is stale -- update LEGACY_EXECUTOR_CALLERS when a "
        f"caller is migrated or added (found {sorted(found)})")


# Modules that evaluate counterfactuals must not write legacy terminal receipts.  Reading historical
# receipts stays allowed and is exercised by Rewind Fidelity and Run Diagnostics.
NON_WRITING_EVALUATION_MODULES = (
    "clozn/replay/branch_fan.py",
    "clozn/replay/sampler_sensitivity.py",
    "clozn/replay/test_this.py",
    "clozn/experiments",
    "clozn/recipes",
)


def test_evaluation_paths_never_write_legacy_execution_fork_receipts():
    offenders = []
    for relative in NON_WRITING_EVALUATION_MODULES:
        for path in _python_files(relative):
            if "clozn.replay.execution_fork_results" in _imports(path):
                offenders.append(os.path.relpath(path, REPO_ROOT))
    assert offenders == [], (
        f"evaluation paths must not reach the legacy receipt store at all: {offenders}")
