"""`clozn retry`: compare a request-local, prompt-first correction against the last run."""
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
    parser = sub.add_parser("retry", help="compare a prompt-first correction against the last run")
    repairs = parser.add_mutually_exclusive_group()
    repairs.add_argument("--less-verbose", action="store_true")
    repairs.add_argument("--more-concrete", action="store_true")
    repairs.add_argument("--use-context", action="store_true")
    repairs.add_argument("--ask-before-guessing", action="store_true")
    repairs.add_argument("--preserve-formatting", action="store_true")
    repairs.add_argument("--stop-repeating", action="store_true")
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
          f"  |  word change: {delta.get('changed', '?')}%")
    print("this correction applied only to the candidate replay above; it leaves no "
          "behavior change for future requests.")


def cmd_retry(args):
    from clozn.cli import main as ctx
    port = args.port or 8080
    preset = _selected_preset(args)
    if not preset:
        raise ctx.CloznError(
            "choose one correction: --less-verbose, --more-concrete, --use-context, "
            "--ask-before-guessing, --preserve-formatting, or --stop-repeating"
        )
    result = _post(port, f"/runs/{_last_organic_id()}/retry", {"preset": preset})
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        _print_comparison(result)
    return 0
