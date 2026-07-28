"""The product must import with NOTHING but the standard library.

WHY THIS EXISTS
---------------
pyproject.toml declares `dependencies = []` and explains the rule: anything an optional command needs
beyond the stdlib is imported lazily inside that command's function body, never at module scope. That is
the promise `pip install clozn` makes.

Nothing enforced it. The `product-minimal` CI lane is named "Torch-free boundary" and, by design,
INSTALLS numpy before asserting torch is absent -- so a module-scope `import numpy` reachable from the
product import path sailed through it. Two such imports had accumulated:

  * clozn/receipts/swap_receipt.py imported clozn.behavior.steering.concept_dir (steering math, numpy)
    at module scope, and read two of its constants as DEFAULT ARGUMENT values -- reachable from
    clozn/cli/main.py via clozn.experiments. `clozn --help` therefore required numpy.
  * clozn/server/routes/readouts.py imported numpy at module scope for a single call in one route,
    making the whole gateway unimportable without it.

Both broke `scripts/release/clean_room_install_test.py` on a bare install, and neither was visible in a
dev checkout where numpy happens to be present. That is the failure mode this test closes: the invariant
is now checked in the ordinary suite, on every run, rather than by a lane whose environment hides it.

HOW
---
A subprocess with a meta_path hook that refuses every top-level module outside `sys.stdlib_module_names`
(plus the packages clozn itself ships). A subprocess rather than an in-process hook because by the time
this test runs, pytest and its plugins have already imported half the world -- clozn's own modules would
be served from a warm sys.modules and prove nothing.

ADDING A DEPENDENCY
-------------------
If clozn ever takes a real install-time dependency, add it to pyproject's `dependencies` AND to
`_ALLOWED` below, in the same commit. Editing only this list to make the test pass is how the promise
gets broken quietly.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# Top-level names the product may import despite not being stdlib: clozn itself and the asset packages
# setup.py remaps into the wheel (see setup.py's docstring).
_ALLOWED = {"clozn", "studio", "protocol"}

# Entry points a `pip install clozn` user reaches directly. Each must import stdlib-only.
_ENTRY_POINTS = [
    ("clozn.cli.main", "the CLI -- `clozn --help` and every subcommand's registration"),
    ("clozn.server.app", "the gateway -- `clozn serve`'s HTTP surface"),
]

_PROBE = textwrap.dedent("""
    import sys

    ALLOWED = {allowed!r}

    class StdlibOnly:
        def find_spec(self, name, path=None, target=None):
            top = name.split(".")[0]
            if top in sys.stdlib_module_names or top in ALLOWED or top.startswith("_"):
                return None
            raise ImportError(
                "clozn imported the non-stdlib module %r at module scope; "
                "pyproject declares dependencies = []" % name)

    sys.meta_path.insert(0, StdlibOnly())
    import {module}
    print("OK")
""")


@pytest.mark.parametrize("module,why", _ENTRY_POINTS, ids=[m for m, _ in _ENTRY_POINTS])
def test_entry_point_imports_with_stdlib_only(module, why):
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(allowed=sorted(_ALLOWED), module=module)],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode == 0, (
        f"{module} ({why}) does not import on a stdlib-only install.\n"
        f"A `pip install clozn` user hits this immediately. Move the offending import INSIDE the "
        f"function that needs it -- and if it is being read as a default argument value, replace the "
        f"default with None and resolve it in the body, since defaults evaluate at definition time.\n\n"
        f"{result.stderr.strip()[-2000:]}"
    )


def test_the_probe_actually_blocks_something():
    """A guard against the guard: if the blocker silently stopped working, the tests above would pass
    vacuously and this whole file would be decoration."""
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(allowed=sorted(_ALLOWED), module="numpy")],
        capture_output=True, text=True, timeout=180,
    )
    assert result.returncode != 0, "the stdlib-only probe failed to block numpy; it is not testing anything"
    assert "dependencies = []" in result.stderr


def test_pyproject_still_declares_no_dependencies():
    """If a real dependency is ever added, this test should be updated deliberately in the same commit
    that adds it -- not discovered later by a user whose install broke."""
    import os
    import re
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "pyproject.toml"), encoding="utf-8") as handle:
        content = handle.read()
    match = re.search(r"^dependencies\s*=\s*\[(.*?)\]", content, re.MULTILINE | re.DOTALL)
    assert match, "pyproject.toml has no [project] dependencies field"
    assert not match.group(1).strip(), (
        f"pyproject declares dependencies {match.group(1).strip()!r}. That may be correct -- but this "
        f"suite's stdlib-only boundary assumes none, so update _ALLOWED in this file too.")
