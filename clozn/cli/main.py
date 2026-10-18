#!/usr/bin/env python3
"""clozn -- a boring, reliable front door to the local model engine.

The fast runtime is the C++ engine (clozn-server.exe). This wraps it so the daily path is one command:

    clozn run   <model> "<prompt>"     one-shot, streams tokens to the terminal
    clozn serve <model> [--port 8080]  bring up the OpenAI-compatible endpoint, print the base URL
    clozn models                       discover local GGUFs and the backend that would run them

The CLI/supervisor path uses no Torch. It finds the engine build (GPU preferred), puts the right DLLs on
PATH, picks per-model flags (diffusion
mask tokens, etc.), reports honestly what it's running on, and fails with one actionable line instead of a
stack trace. Model dirs: $CLOZN_MODELS, ~/.clozn/models, <repo>/models, ~/.clozn/config.json["model_dirs"].

This module is the argparse root: the full command tree (build_parser), dispatch (main), and the shared
constants (HOME, CloznError) every other clozn.cli.* module reads at call time via
`from clozn.cli import main as ctx` (never `from clozn.cli.main import HOME`, which would bind a stale
copy immune to a test's monkeypatch or a later change). Each actual `cmd_X` implementation lives in
clozn/cli/commands/*.py, grouped by family; this file only imports and wires them.
"""
from __future__ import annotations

import argparse
import os
import sys

from clozn.paths import data_root

# Supervisor-owned v1 state (worker registry, logs, and install metadata) is
# isolated alongside the v1 run journal. Model discovery still retains its
# existing catalogue locations, including ~/.clozn/models.
HOME = data_root()


class CloznError(Exception):
    """A clean, user-facing failure -- printed as one line, no traceback."""


def _positive_context(value: str) -> int:
    """argparse type for a real, non-zero worker context window."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("context size must be a positive integer") from None
    if parsed < 1:
        raise argparse.ArgumentTypeError("context size must be a positive integer")
    return parsed


def _nonnegative_int(value: str) -> int:
    """argparse type for compatibility flags whose zero value is meaningful."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError("value must be a non-negative integer") from None
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


# Imported after HOME/CloznError are defined: every module below reaches back into this one (`from
# clozn.cli import main as ctx` for HOME, `from clozn.cli.main import CloznError`), so this file must
# finish defining both before triggering those imports -- see engine_process.py's module docstring for the
# full circular-import trace this depends on. Order also matters *between* these: commands.serve/run import
# names directly off commands.models, so models must load first.
#
from clozn.cli import formatting as fmt                                                       # noqa: E402
from clozn.cli.engine_process import _free_port                                               # noqa: E402
from clozn.cli.commands.models import cmd_models, cmd_pull                                    # noqa: E402
from clozn.cli.commands.run import cmd_run                                                    # noqa: E402
from clozn.cli.commands.serve import cmd_serve, cmd_ps, cmd_stop                              # noqa: E402
from clozn.cli.commands.studio import cmd_studio                                              # noqa: E402
from clozn.cli.commands.version import cmd_version                                              # noqa: E402
from clozn.cli.commands.doctor import cmd_doctor                                                # noqa: E402
from clozn.cli.commands import _autoload                                                          # noqa: E402

# Exact public command boundary. Hand-wired parsers and the extension manifest must together produce
# this sequence; a command cannot become public through an incidental registration.
PUBLIC_COMMANDS = (
    "run",
    "serve",
    "models",
    "pull",
    "studio",
    "ps",
    "stop",
    "version",
    "doctor",
)


def build_parser():
    """The full argparse tree, factored out of main() so tests can introspect flags without dispatching."""
    p = argparse.ArgumentParser(prog="clozn", description="a reliable front door to the local model engine")
    sub = p.add_subparsers(dest="cmd")

    pr = sub.add_parser("run", help="one-shot: stream a completion to the terminal")
    pr.add_argument("model"); pr.add_argument("prompt", nargs="?", default=None)
    pr.add_argument("--max", type=int, default=256, help="max new tokens (default 256)")
    pr.add_argument("--ctx", type=_positive_context, default=None,
                    help="worker context window in tokens (default 4096; reduce on tight unified memory)")
    pr.add_argument("--batch", type=_positive_context, default=None,
                    help="maximum logical rows submitted per decode (default: --ctx)")
    pr.add_argument("--ubatch", type=_positive_context, default=None,
                    help="accelerator microbatch size (default: min(--batch, 512))")
    pr.add_argument("--cpu", action="store_true", help="force the CPU build")
    pr.add_argument("--port", type=int, default=0); pr.add_argument("--mask", type=int, default=None)
    pr.add_argument("--eos", type=int, default=None)
    pr.set_defaults(fn=cmd_run)

    ps = sub.add_parser("serve", help="bring up the OpenAI-compatible endpoint")
    ps.add_argument(
        "model", nargs="?",
        help="single GGUF name/path (or use -m/--model)",
    )
    ps.add_argument(
        "-m", "--model", dest="model_flag", metavar="PATH",
        help="single GGUF name/path (llama-server-compatible alias for MODEL)",
    )
    ps.add_argument("--port", type=int, default=0)
    ps.add_argument("-c", "--ctx", "--ctx-size", dest="ctx", type=_positive_context, default=None,
                    help="worker context window in tokens (default 4096; reduce on tight unified memory)")
    ps.add_argument("--batch", type=_positive_context, default=None,
                    help="maximum logical rows submitted per decode (default: --ctx)")
    ps.add_argument("--ubatch", type=_positive_context, default=None,
                    help="accelerator microbatch size (default: min(--batch, 512))")
    ps.add_argument(
        "-ngl", "--gpu-layers", "--n-gpu-layers", dest="gpu_layers",
        type=_nonnegative_int, default=None, metavar="N",
        help="explicitly offload N layers (integer subset; omitted keeps Clozn's current default)",
    )
    ps.add_argument(
        "--host", default="127.0.0.1", metavar="HOST",
        help="public gateway bind host (private model worker remains loopback-only)",
    )
    ps.add_argument("-a", "--alias", dest="alias", default=None, metavar="NAME",
                    help="single-model public model presentation name")
    ps.add_argument("-np", "--parallel", dest="parallel", type=_positive_context, default=1, metavar="N",
                    help="generation slots (only 1 is supported by Clozn)")
    ps.add_argument("--cpu", action="store_true"); ps.add_argument("--mask", type=int, default=None)
    ps.add_argument("--eos", type=int, default=None)
    ps.set_defaults(fn=cmd_serve)

    sub.add_parser("models", help="list local models + the engine backend").set_defaults(fn=cmd_models)
    pp = sub.add_parser("pull", help="download a model GGUF (by name, or owner/repo/file.gguf)")
    pp.add_argument("model"); pp.set_defaults(fn=cmd_pull)
    pst = sub.add_parser("studio", help="attach to the Studio served by a running Clozn runtime")
    pst.add_argument("--port", type=int, default=0); pst.add_argument("--open", action="store_true", help="open the UI in your browser")
    pst.set_defaults(fn=cmd_studio)
    sub.add_parser("ps", help="list running product runtimes").set_defaults(fn=cmd_ps)
    pstop = sub.add_parser("stop", help="stop a product runtime (by model name, port, or 'all')")
    pstop.add_argument("which"); pstop.set_defaults(fn=cmd_stop)
    # Register the checked-in command manifest LAST so a manifested module reaching back into this
    # one finds it fully defined and the hand-wired order above is already settled. A source file or
    # marker alone cannot extend the public CLI. See commands/_autoload.py.
    _autoload.register_all(sub)
    sub.add_parser("version", help="print the installed clozn version (+ git commit if available)"
                   ).set_defaults(fn=cmd_version)
    pdoc = sub.add_parser("doctor", help="diagnose this install; --verify-offline adds a strict "
                          "local-only enforcement gate")
    pdoc.add_argument("--json", action="store_true", help="print a machine-readable report")
    pdoc.add_argument("--verify-offline", action="store_true",
                      help="fail unless local-only enforcement is active and its ledger window is clean")
    pdoc.add_argument("--deep", action="store_true",
                      help="also validate the discovered engine's embedded build identity "
                           "(one extra subprocess launch, still no model/network)")
    pdoc.set_defaults(fn=cmd_doctor)
    registered = tuple(sub.choices)
    if registered != PUBLIC_COMMANDS:
        raise RuntimeError(
            "public CLI does not match PUBLIC_COMMANDS: "
            f"registered={registered!r} expected={PUBLIC_COMMANDS!r}")
    return p


def main(argv=None):
    fmt._setup_console()
    p = build_parser()
    args = p.parse_args(argv)
    if not getattr(args, "fn", None):
        p.print_help(); return 2
    try:
        rc = args.fn(args)
        return rc if isinstance(rc, int) else 0
    except CloznError as e:
        print(f"{fmt.BOLD}clozn:{fmt.RST} {e}", file=sys.stderr); return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
