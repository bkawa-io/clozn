"""``clozn model-lock``: offline verification and explicit secure artifact fetch.

``verify`` remains parser-only and never imports the fetcher or opens a socket. ``fetch`` is the separate
networked run-mode seam and resolves exactly one named role into a SHA-keyed destination.

Registered via CLOZN_AUTOLOAD (docs/SEAMS.md Seam 1) -- no edit to clozn/cli/main.py.
"""
from __future__ import annotations

import json
import re

CLOZN_AUTOLOAD = True


def add_subparser(sub):
    parser = sub.add_parser(
        "model-lock",
        help="verify a model lockfile offline or securely fetch one pinned role")
    commands = parser.add_subparsers(dest="model_lock_cmd")
    parser.set_defaults(fn=_no_command)

    verify = commands.add_parser(
        "verify", help="validate a lockfile's shape, SHA-256 fields, and HTTPS-only URLs")
    verify.add_argument("lockfile", help="path to a clozn.model-lock.v1 JSON file")
    verify.add_argument("--json", action="store_true", help="print a machine-readable result")
    verify.set_defaults(fn=cmd_model_lock_verify)

    fetch = commands.add_parser(
        "fetch", help="download and verify one pinned role into a SHA-keyed model cache")
    fetch.add_argument("lockfile", help="path to a clozn.model-lock.v1 JSON file")
    fetch.add_argument("--role", required=True, help="model role to fetch (for example: candidate)")
    fetch.add_argument(
        "--out", required=True, metavar="DIR",
        help="destination directory; the verified file is stored as DIR/<sha256>.gguf")
    fetch.add_argument("--json", action="store_true", help="print a machine-readable result")
    fetch.set_defaults(fn=cmd_model_lock_fetch)
    return parser


def _no_command(_args):
    print("clozn model-lock: use `clozn model-lock verify <FILE>` or "
          "`clozn model-lock fetch <FILE> --role ROLE --out DIR`")
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


_URL_IN_ERROR = re.compile(r"(?i)https?://[^\s'\"]+")


def _redacted_error(error: Exception) -> str:
    """Do not let credentials, signed query values, or URL paths reach CI output."""
    return _URL_IN_ERROR.sub("<redacted-url>", str(error))


def cmd_model_lock_fetch(args):
    """Resolve one explicitly selected lockfile role into ``--out/<sha256>.gguf``."""
    from clozn.models.fetch import ModelFetchError, fetch_locked_model
    from clozn.models.lockfile import LockfileError

    try:
        result = fetch_locked_model(args.lockfile, args.role, args.out)
    except (LockfileError, ModelFetchError) as error:
        message = _redacted_error(error)
        if args.json:
            print(json.dumps({
                "ok": False,
                "role": args.role,
                "out": args.out,
                "error": message,
            }, indent=2, sort_keys=True))
        else:
            print(f"clozn model-lock fetch: {message}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"clozn model-lock fetch: {result['role']} {result['cache']}")
        print(f"  path:       {result['path']}")
        print(f"  sha256:     {result['sha256']}")
        print(f"  size_bytes: {result['size_bytes']}")
    return 0
