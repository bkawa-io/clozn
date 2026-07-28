"""Opt-in discovery for NEW subcommands, so adding one is a pure file addition.

THE PROBLEM THIS SOLVES
-----------------------
clozn/cli/main.py wires the command tree by hand: one `from clozn.cli.commands.X import add_subparser`
line near the top, one `_add_X(sub)` call at the bottom. That is perfectly good for a single author and
actively hostile to several working at once -- every new command touches the same two blocks, so N
parallel branches produce N conflicts in the same handful of lines, every time.

A module that opts in here is registered without main.py changing at all:

    # clozn/cli/commands/setup_engine.py
    CLOZN_AUTOLOAD = True

    def add_subparser(sub):
        p = sub.add_parser("setup", help="install and verify a matching native engine")
        p.add_argument("--dry-run", action="store_true")
        p.set_defaults(fn=cmd_setup)

WHY THE EXISTING COMMANDS ARE NOT MIGRATED
------------------------------------------
main.py's import block is order-sensitive in ways its docstring spells out (commands.serve/run/explain
read names directly off commands.models, so models must load first) and several entries exist purely as
stable re-exports for tests written against the pre-split flat module. Rewriting that by hand-migrating
25 working commands is a refactor with real regression risk and no user-visible payoff. This runs AFTER
all of it instead: by the time register_all() is called at the end of build_parser(), every hand-wired
import has completed, so an autoloaded module reaching back into clozn.cli.main finds it fully defined.
Existing modules do not set the flag and are therefore never double-registered.

WHY A TEXT SCAN RATHER THAN IMPORTING EVERYTHING
------------------------------------------------
Deciding whether a module opts in by importing it would mean importing every module in the package on
every `clozn --help`, paying the cost of commands the user did not invoke -- and dragging modules into
the process in an order nobody designed. Reading each file's source and looking for the marker is a
handful of small file reads and imports nothing that did not ask to be imported.

FAILURE BEHAVIOR
----------------
A module that opts in and then fails to import is recorded in LOAD_FAILURES and reported on stderr --
never swallowed (roadmap rule 3: no silent fallback). It does NOT take down the rest of the CLI, so a
half-installed or locally-broken command still leaves `clozn doctor` reachable to diagnose it.
tests/test_cli_autoload.py asserts LOAD_FAILURES is empty, which is what turns a broken command into a
hard CI failure rather than a warning a user learns to scroll past.
"""
from __future__ import annotations

import importlib
import os
import sys

MARKER = "CLOZN_AUTOLOAD"

_PACKAGE = "clozn.cli.commands"
_DIR = os.path.dirname(os.path.abspath(__file__))

# (module_name, exception) for every module that opted in but could not be loaded or registered. Read by
# tests/test_cli_autoload.py and by `clozn doctor`; never cleared, so a failure stays visible.
LOAD_FAILURES: list[tuple[str, BaseException]] = []


def _candidates() -> list[str]:
    """Module names in this package whose SOURCE mentions the marker, sorted for a stable --help order.

    Private modules (leading underscore -- including this one) and test modules are skipped outright.
    The scan is textual and therefore approximate in one harmless direction: a module that merely
    mentions the marker in a comment is imported and then found not to opt in, costing one import. The
    reverse -- a module that opts in but is skipped -- cannot happen, because setting the flag requires
    writing the marker into the file.
    """
    names = []
    try:
        entries = sorted(os.listdir(_DIR))
    except OSError:
        return []
    for entry in entries:
        if not entry.endswith(".py") or entry.startswith("_") or entry.startswith("test_"):
            continue
        try:
            with open(os.path.join(_DIR, entry), encoding="utf-8") as handle:
                if MARKER not in handle.read():
                    continue
        except OSError:
            continue
        names.append(entry[:-3])
    return names


def register_all(sub) -> int:
    """Call `add_subparser(sub)` on every command module that sets `CLOZN_AUTOLOAD = True`.

    Returns the number of modules successfully registered. Never raises: a broken command module is
    recorded in LOAD_FAILURES and announced on stderr, leaving the rest of the CLI usable.
    """
    registered = 0
    for name in _candidates():
        dotted = f"{_PACKAGE}.{name}"
        try:
            module = importlib.import_module(dotted)
            if getattr(module, MARKER, False) is not True:
                continue
            add_subparser = getattr(module, "add_subparser", None)
            if add_subparser is None:
                raise AttributeError(
                    f"sets {MARKER} = True but defines no add_subparser(sub) function")
            add_subparser(sub)
            registered += 1
        except BaseException as exc:            # noqa: BLE001 -- deliberately broad; see module docstring
            LOAD_FAILURES.append((dotted, exc))
            print(f"clozn: command module {dotted} failed to load: "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr)
    return registered
