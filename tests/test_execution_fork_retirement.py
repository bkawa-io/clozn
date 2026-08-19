"""Source-level guards for retiring the child-creating execution-fork executor.

Two separate claims are asserted here:

1.  Canonical product code -- the experimental kernel, the product recipes, and Branch Fan --
    never imports the legacy executor or the legacy planner adapters that create child Runs.
    Exact execution is allowed to reach the worker's low-level ``engine.execution_fork`` RPC, but
    only underneath the neutral GenerateExecutionAdapter.

2.  Nothing imports ``clozn.replay.execution_fork_execute`` at all, because the module is gone.  The
    inventory that tracked its remaining callers is now empty and stays that way: reintroducing the
    old "execute a preflighted fork and persist terminal evidence" owner fails these tests.
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
    "clozn/replay/checkpoint_capture.py",
    "clozn/replay/sampler_sensitivity.py",
    "clozn/replay/test_this.py",
    "clozn/replay/execution_fork.py",
    "clozn/server/routes/branch_fan.py",
    "clozn/server/routes/sampler_sensitivity.py",
    "clozn/server/routes/time_travel_v1.py",
    "clozn/server/routes/timetravel.py",
)

# The deprecated executor has no callers left anywhere in the package, and the module itself is
# deleted.  Exact resume now runs through execution_facts.resolve_exact_resume_facts and
# experiments.exact_execution.prove_unchanged_control, straight onto the worker's own
# execution_fork RPC, with no product executor in between.
LEGACY_EXECUTOR_CALLERS: set[str] = set()


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
        "nothing may import the deprecated execution-fork executor again "
        f"(found {sorted(found)})")


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


def test_the_legacy_executor_module_is_gone():
    """The old "execute a preflighted fork and persist terminal evidence" owner no longer exists."""
    assert not os.path.exists(os.path.join(PACKAGE_ROOT, "replay", "execution_fork_execute.py"))


def test_public_execution_fork_routes_are_not_registered():
    from clozn.server import app as server

    route_modules = [
        getattr(item, "__name__", "")
        for item in (*server._POST_ROUTES, *server._GET_ROUTES)
    ]
    assert "execution_fork" not in route_modules


def test_no_production_code_writes_the_legacy_receipt_store():
    """Every active write to ExecutionForkResult is gone; only historical reads remain.

    Rewind Fidelity, Run Diagnostics, and the turn receipt still READ this store for historical
    proof, which is deliberate and is the subject of a separate cleanup.  What must never come back
    is a new write: nothing in the product produces terminal fork receipts any more.
    """
    writers = []
    for path in _all_package_files():
        relative = os.path.relpath(path, REPO_ROOT)
        if relative == "clozn/replay/execution_fork_results.py":
            continue                              # the store's own implementation
        with open(path, encoding="utf-8-sig") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                continue
            if node.func.attr not in {"save", "write_on_terminal"}:
                continue
            target = node.func.value
            if isinstance(target, ast.Name) and "execution_fork_results" in target.id:
                writers.append((relative, node.func.attr))
            elif isinstance(target, ast.Attribute) and target.attr == "execution_fork_results":
                writers.append((relative, node.func.attr))
    assert writers == [], f"production code must not write ExecutionForkResult receipts: {writers}"


def test_legacy_planner_module_is_not_dead_yet_and_is_not_silently_deleted():
    """A truthful record of what still imports the legacy planner, and for what.

    clozn/replay/execution_fork.py is NOT dead: four modules still import small helpers from it.
    Deleting it needs its own audit and its own change, so this test states the remaining surface
    rather than letting it rot unnoticed.
    """
    importers = set()
    for path in _all_package_files():
        relative = os.path.relpath(path, REPO_ROOT)
        if relative in {"clozn/replay/execution_fork.py", "clozn/replay/execution_fork_results.py"}:
            continue
        if "clozn.replay.execution_fork" in _imports(path):
            importers.add(relative)
    assert importers == {
        "clozn/replay/context_bisect.py",           # parent_execution_fingerprint
        "clozn/replay/influence_counterfactual.py",  # runtime projections + fingerprint
        "clozn/runs/selection_inspection.py",        # sampling_intervention_contract
        "clozn/runs/test_this.py",                   # normalize_intervention + fingerprint
    }, f"the legacy planner's remaining helper surface changed: {sorted(importers)}"
