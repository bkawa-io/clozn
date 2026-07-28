"""commands.model_lock -- `clozn model-lock verify <FILE>` (feature 02, "GitHub Action for model-change
gating"): validate a checked-in `clozn.model-lock.v1` lockfile with zero network access.

This is deliberately the entire surface for now. Resolving a pinned entry into a downloaded, SHA-256-
verified local model file is separate, deferred work (see clozn/models/lockfile.py's module docstring) --
this command answers "is this lockfile well-formed" only, which is exactly what a verify-mode CI job on a
free runner needs to check before a run-mode job (elsewhere, on a runner that actually downloads models)
trusts it.

Registered via CLOZN_AUTOLOAD (docs/SEAMS.md Seam 1) -- no edit to clozn/cli/main.py.
"""
from __future__ import annotations

import json

CLOZN_AUTOLOAD = True


def add_subparser(sub):
    parser = sub.add_parser(
        "model-lock",
        help="inspect/validate a clozn.model-lock.v1 lockfile (no network access, no download)")
    commands = parser.add_subparsers(dest="model_lock_cmd")
    parser.set_defaults(fn=_no_command)

    verify = commands.add_parser(
        "verify", help="validate a lockfile's shape, SHA-256 fields, and HTTPS-only URLs")
    verify.add_argument("lockfile", help="path to a clozn.model-lock.v1 JSON file")
    verify.add_argument("--json", action="store_true", help="print a machine-readable result")
    verify.set_defaults(fn=cmd_model_lock_verify)
    return parser


def _no_command(_args):
    print("clozn model-lock: use `clozn model-lock verify <FILE>`")
    return 2


def cmd_model_lock_verify(args):
    """Exit 0 if `args.lockfile` conforms to clozn.model-lock.v1 (schema, mandatory SHA-256 per pinned
    model, HTTPS-only URLs); exit 1 with a specific reason otherwise. Never exits 2/3 -- there is no
    execution-error/identity-refusal distinction here, just well-formed or not."""
    from clozn.models.lockfile import LockfileError, load_lockfile, model_roles

    try:
        document = load_lockfile(args.lockfile)
    except LockfileError as e:
        if args.json:
            print(json.dumps({"ok": False, "path": args.lockfile, "error": str(e)}, indent=2))
        else:
            print(f"clozn model-lock verify: {e}")
        return 1

    roles = model_roles(document)
    if args.json:
        print(json.dumps({"ok": True, "path": args.lockfile, "roles": roles}, indent=2))
    else:
        print(f"clozn model-lock verify: {args.lockfile} OK -- {len(roles)} pinned model(s): "
              f"{', '.join(roles) or '(none)'}")
    return 0
