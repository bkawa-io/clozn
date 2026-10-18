"""Explicit registration manifest for CLI command modules.

Only modules named in :data:`COMMAND_MODULES` can extend the public command tree. Keeping the
existing ``CLOZN_AUTOLOAD`` marker as a second, module-local assertion makes an accidental manifest
entry fail visibly instead of silently exposing a command. Adding a Python file to this package is
therefore not enough to make it public.

Load failures are recorded and reported without taking down the rest of the CLI. The test suite
asserts that the production manifest loads cleanly, turning those runtime diagnostics into a hard CI
failure.
"""
from __future__ import annotations

from collections.abc import Iterable
import importlib
import sys

MARKER = "CLOZN_AUTOLOAD"
_PACKAGE = "clozn.cli.commands"

# This is intentionally explicit. Stepwise v1 cuts remove entries here before deleting their
# implementations, so an old file copied back into the package cannot restore a public command.
COMMAND_MODULES: tuple[str, ...] = ()

# (module_name, exception) for every manifested module that could not be loaded or registered. Read
# by tests/test_cli_autoload.py and by `clozn doctor`; never silently discarded.
LOAD_FAILURES: list[tuple[str, BaseException]] = []


def _candidates(module_names: Iterable[str] | None = None) -> list[str]:
    """Return a validated, deterministic registration list.

    ``module_names`` exists for contract tests and downstream packaging checks. Production callers
    omit it and receive the checked-in manifest.
    """
    names = list(COMMAND_MODULES if module_names is None else module_names)
    invalid = [name for name in names if not isinstance(name, str) or not name.isidentifier()]
    if invalid:
        raise ValueError(f"invalid command module names: {invalid!r}")
    if len(names) != len(set(names)):
        raise ValueError("duplicate command module names in registration manifest")
    return sorted(names)


def register_all(sub, module_names: Iterable[str] | None = None) -> int:
    """Register every command module in the explicit manifest.

    Returns the number of successful registrations. A broken manifested module is recorded in
    :data:`LOAD_FAILURES` and announced on stderr, leaving diagnostic commands reachable.
    """
    registered = 0
    for name in _candidates(module_names):
        dotted = f"{_PACKAGE}.{name}"
        try:
            module = importlib.import_module(dotted)
            if getattr(module, MARKER, False) is not True:
                raise AttributeError(f"is manifested but does not set {MARKER} = True")
            add_subparser = getattr(module, "add_subparser", None)
            if add_subparser is None:
                raise AttributeError(
                    f"sets {MARKER} = True but defines no add_subparser(sub) function")
            add_subparser(sub)
            registered += 1
        except BaseException as exc:  # noqa: BLE001 -- diagnostics must survive broken extensions
            LOAD_FAILURES.append((dotted, exc))
            print(
                f"clozn: command module {dotted} failed to load: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
    return registered


__all__ = ["COMMAND_MODULES", "LOAD_FAILURES", "MARKER", "register_all"]
