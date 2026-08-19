"""Source-level guards for the retired Execution Fork product vertical.

"Execution fork" now means exactly one thing internally: the low-level worker resume RPC. It is no
longer a planner, a product concept, a result type, or a helper namespace.

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
LEGACY_PLANNER_MODULE = "clozn.replay.execution_fork"

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

# Canonical product code is now simply the whole package: there is no legacy fork vertical left for
# any module to depend on.
CANONICAL_MODULES = ("clozn",)

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


def test_the_legacy_receipt_store_is_gone():
    """ExecutionForkResult is retired outright: no module, no readers, no on-disk store.

    Historical exact proof is now canonical GeneratedObservation evidence, so there is nothing left
    for a compatibility adapter to convert and no reason to keep the old evidence model alive.
    """
    assert not os.path.exists(os.path.join(PACKAGE_ROOT, "replay", "execution_fork_results.py"))
    importers = [
        os.path.relpath(path, REPO_ROOT) for path in _all_package_files()
        if "clozn.replay.execution_fork_results" in _imports(path)
    ]
    assert importers == [], f"nothing may import the retired receipt store: {importers}"


def test_no_package_code_touches_the_legacy_results_database():
    """Not even by path: the ~/.clozn/execution-forks store is not read or written anywhere."""
    offenders = []
    for path in _all_package_files():
        with open(path, encoding="utf-8-sig") as handle:
            source = handle.read()
        if "execution-forks" in source or "execution_fork_results" in source:
            offenders.append(os.path.relpath(path, REPO_ROOT))
    assert offenders == [], f"the retired results database is still referenced: {offenders}"


def test_the_legacy_planner_module_is_gone():
    """The Execution Fork namespace is retired outright: no module, no importers."""
    assert not os.path.exists(os.path.join(PACKAGE_ROOT, "replay", "execution_fork.py"))
    importers = [
        os.path.relpath(path, REPO_ROOT) for path in _all_package_files()
        if LEGACY_PLANNER_MODULE in _imports(path)
    ]
    assert importers == [], f"nothing may import the retired planner namespace: {importers}"


def test_plan_execution_fork_is_defined_and_called_nowhere():
    """The old planner abstraction is gone, not renamed.

    State addressing belongs to StateRef/resolve_state and exact resume to
    execution_facts.resolve_exact_resume_facts. A module reintroducing this name -- under any
    namespace -- would be that duplicate planner coming back.
    """
    offenders = []
    for path in _all_package_files():
        relative = os.path.relpath(path, REPO_ROOT)
        with open(path, encoding="utf-8-sig") as handle:
            tree = ast.parse(handle.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "plan_execution_fork":
                offenders.append((relative, "definition"))
            elif isinstance(node, ast.Name) and node.id == "plan_execution_fork":
                offenders.append((relative, "reference"))
            elif isinstance(node, ast.Attribute) and node.attr == "plan_execution_fork":
                offenders.append((relative, "reference"))
    assert offenders == [], f"the retired fork planner is back: {offenders}"


def test_the_worker_resume_rpc_is_still_reachable_from_the_canonical_path():
    """What survives is the low-level primitive, exactly as intended.

    The canonical Generate adapter and unchanged-control proof both call engine.execution_fork(...)
    directly. Retiring the product vertical must never be mistaken for removing that RPC.
    """
    for relative in ("clozn/experiments/generation.py", "clozn/experiments/exact_execution.py"):
        with open(os.path.join(REPO_ROOT, relative), encoding="utf-8") as handle:
            assert "execution_fork(" in handle.read(), (
                f"{relative} must still reach the low-level worker resume RPC")
