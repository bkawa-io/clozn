"""`clozn retry`: compare a bounded prompt-first correction and optionally scope it."""
from __future__ import annotations

import json
import urllib.error
import urllib.request


_FLAG_TO_PRESET = {
    "less_verbose": "less-verbose",
    "more_concrete": "more-concrete",
    "use_context": "use-context",
    "ask_before_guessing": "ask-before-guessing",
    "preserve_formatting": "preserve-formatting",
    "stop_repeating": "stop-repeating",
}


def add_subparser(sub):
    parser = sub.add_parser("retry", help="compare and apply a prompt-first correction to the last run")
    parser.add_argument("action", choices=("last", "undo"))
    parser.add_argument("undo_id", nargs="?", help="repair id for `retry undo`")
    repairs = parser.add_mutually_exclusive_group()
    repairs.add_argument("--less-verbose", action="store_true")
    repairs.add_argument("--more-concrete", action="store_true")
    repairs.add_argument("--use-context", action="store_true")
    repairs.add_argument("--ask-before-guessing", action="store_true")
    repairs.add_argument("--preserve-formatting", action="store_true")
    repairs.add_argument("--stop-repeating", action="store_true")
    parser.add_argument("--scope", choices=("once", "session", "profile"), default="once",
                        help="how long the correction stays active (default once)")
    parser.add_argument("--port", type=int, default=0, help="Clozn gateway port (default 8080)")
    parser.add_argument("--json", action="store_true", help="print the machine-readable comparison")
    parser.set_defaults(fn=cmd_retry)
    return parser


def _post(port: int, path: str, body: dict) -> dict:
    from clozn.cli import main as ctx
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(body).encode("utf-8"),
        method="POST", headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as exc:
        try:
            payload = json.loads(exc.read())
            message = payload.get("error") or payload
        except Exception:
            message = str(exc)
        raise ctx.CloznError(f"retry failed ({exc.code}): {message}") from None
    except urllib.error.URLError as exc:
        raise ctx.CloznError(
            f"couldn't reach the Clozn gateway on port {port} ({getattr(exc, 'reason', exc)}). "
            "Start it first:  clozn serve <model>"
        ) from None


def _last_organic_id() -> str:
    from clozn.cli import main as ctx
    import clozn.runs.store as runlog
    rows = runlog.list_runs(limit=1, include_replays=False)
    if not rows:
        raise ctx.CloznError("no recorded run to retry")
    return str(rows[0]["id"])


def _selected_preset(args) -> str | None:
    return next((preset for flag, preset in _FLAG_TO_PRESET.items()
                 if getattr(args, flag, False)), None)


def _print_comparison(result: dict) -> None:
    print("stored original (context only):")
    print(result.get("stored_original_reply") or "(empty)")
    print("\nmatched greedy baseline:")
    print(result.get("baseline_reply") or "(empty)")
    print("\ncorrected candidate:")
    print(result.get("corrected_reply") or "(empty)")
    delta = result.get("delta") or {}
    print(f"\nchanged: {str(bool(result.get('changed'))).lower()}"
          f"  |  word change: {delta.get('changed', '?')}%"
          f"  |  scope: {(result.get('policy') or {}).get('scope', result.get('scope', 'once'))}")
    undo = result.get("undo") or {}
    if undo.get("available") and undo.get("id"):
        print(f"undo: clozn retry undo {undo['id']}")
    elif undo.get("status") == "automatic_restored":
        print("undo: automatic — this correction affected only the candidate replay")
    elif (result.get("policy") or {}).get("reason"):
        print("not activated: " + str(result["policy"]["reason"]))


def cmd_retry(args):
    from clozn.cli import main as ctx
    port = args.port or 8080
    preset = _selected_preset(args)
    if args.action == "undo":
        if preset or args.scope != "once":
            raise ctx.CloznError("`retry undo` accepts only a repair id (and optional --port/--json)")
        if not args.undo_id:
            raise ctx.CloznError("`retry undo` needs the repair id printed by the scoped retry")
        result = _post(port, f"/corrective-retries/{args.undo_id}/undo", {})
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        else:
            print(f"undone: {result.get('scope')} correction for {result.get('target')}")
        return 0

    if args.undo_id:
        raise ctx.CloznError("unexpected value after `retry last`")
    if not preset:
        raise ctx.CloznError(
            "choose one correction: --less-verbose, --more-concrete, --use-context, "
            "--ask-before-guessing, --preserve-formatting, or --stop-repeating"
        )
    result = _post(port, f"/runs/{_last_organic_id()}/retry",
                   {"preset": preset, "scope": args.scope})
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_comparison(result)
    return 0
