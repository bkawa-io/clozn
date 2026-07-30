"""`clozn snapshot`: FORK-PIN-01 -- durable checkpoint pin/unpin/list.

`pin` needs a live gateway (it materializes the run as a checkpoint on whichever worker currently
serves its model, via POST /runs/<id>/snapshot/pin -- see clozn/server/routes/snapshot.py) and always
shows the real byte cost FIRST (a `preview` call, no bytes written anywhere) before a second,
explicitly `--yes`-confirmed call actually persists it. `unpin`/`list` touch only the local
content-addressed blob store + SQLite metadata (clozn.replay.checkpoint_pin_store) and need no
running server at all.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.main import CloznError

CLOZN_AUTOLOAD = True


def _fmt_bytes(n) -> str:
    if not isinstance(n, (int, float)) or isinstance(n, bool) or n < 0:
        return "?"
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _post(port: int, path: str, body: dict) -> dict:
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
            payload, message = {}, str(exc)
        if isinstance(payload, dict) and payload.get("code") == "snapshot_pin_checkpoint_unavailable":
            capture = payload.get("capture") or {}
            reasons = capture.get("reasons") or []
            reason = reasons[0].get("message") if reasons else None
            raise CloznError(
                f"snapshot pin refused: {reason or message}") from None
        raise CloznError(f"snapshot pin failed ({exc.code}): {message}") from None
    except urllib.error.URLError as exc:
        raise CloznError(
            f"couldn't reach the Clozn gateway on port {port} ({getattr(exc, 'reason', exc)}). "
            "Start it first:  clozn serve <model>"
        ) from None


def cmd_pin(args) -> int:
    port = args.port or 8080
    path = f"/runs/{args.run_id}/snapshot/pin"

    preview = _post(port, path, {"note": args.note, "preview": True})
    size_bytes = preview.get("size_bytes")
    envelope_bytes = preview.get("envelope_bytes")

    if not args.yes:
        if args.json:
            print(json.dumps({
                "ok": False, "confirmed": False, "run_id": args.run_id,
                "size_bytes": size_bytes, "envelope_bytes": envelope_bytes,
                "message": "re-run with --yes to persist this pin",
            }, indent=2, ensure_ascii=False))
        else:
            print(f"snapshot pin - {args.run_id}: would write {_fmt_bytes(envelope_bytes)} "
                  f"(KV cache {_fmt_bytes(size_bytes)}, base64+JSON overhead included)")
            print(f"{fmt.DIM}  re-run with --yes to persist it{fmt.RST}")
        return 1

    result = _post(port, path, {"note": args.note, "preview": False})
    manifest = result.get("manifest") or {}
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        blob = manifest.get("blob") or {}
        print(f"snapshot pin - {manifest.get('run_id', args.run_id)} "
              f"({_fmt_bytes(blob.get('kv_bytes'))} KV, {_fmt_bytes(blob.get('envelope_bytes'))} pinned)")
        print(f"  pin_id: {manifest.get('pin_id')}")
        print(f"  sha256: {blob.get('sha256')}")
        state = manifest.get("state") or {}
        print(f"  n_past={state.get('n_past')} prompt_tokens={state.get('prompt_tokens')}")
    return 0


def cmd_unpin(args) -> int:
    from clozn.replay import checkpoint_pin_store as pins
    try:
        receipt = pins.unpin_checkpoint(args.run_id, cascade=bool(args.cascade))
    except pins.PinHasDependentsError as exc:
        raise CloznError(f"snapshot unpin refused: {exc}; re-run with --cascade to unpin anyway") \
            from None
    except pins.PinStoreError as exc:
        raise CloznError(f"snapshot unpin failed: {exc}") from None
    if args.json:
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    else:
        cleanup = receipt.get("blob_cleanup") or {}
        print(f"snapshot unpin - {args.run_id} ({cleanup.get('status', 'unknown')})")
    return 0


def cmd_list(args) -> int:
    from clozn.replay import checkpoint_pin_store as pins
    manifests = pins.list_pins()
    if args.json:
        print(json.dumps(manifests, indent=2, ensure_ascii=False))
        return 0
    if not manifests:
        print("no pinned checkpoints")
        return 0
    for manifest in manifests:
        blob = manifest.get("blob") or {}
        state = manifest.get("state") or {}
        note = manifest.get("note")
        line = (f"{manifest.get('run_id'):<24} {manifest.get('pinned_at', ''):<21} "
                f"{_fmt_bytes(blob.get('kv_bytes')):>9}  n_past={state.get('n_past')}")
        if note:
            line += f"  {fmt.DIM}# {note}{fmt.RST}"
        print(line)
    return 0


def _no_command(_args) -> int:
    print("clozn snapshot: use `clozn snapshot pin <run_id>`, `unpin <run_id>`, or `list`")
    return 2


def add_subparser(sub):
    parser = sub.add_parser("snapshot", help="durable checkpoint pin/unpin/list (FORK-PIN-01)")
    commands = parser.add_subparsers(dest="snapshot_cmd")
    parser.set_defaults(fn=_no_command)

    pin = commands.add_parser(
        "pin", help="durably pin a run's execution-fork checkpoint (survives worker restart)")
    pin.add_argument("run_id")
    pin.add_argument("--note", default=None, help="a short label to remember why this was pinned")
    pin.add_argument("--port", type=int, default=0, help="Clozn gateway port (default 8080)")
    pin.add_argument("--yes", action="store_true",
                     help="confirm persisting the pin (its size is always shown first)")
    pin.add_argument("--json", action="store_true")
    pin.set_defaults(fn=cmd_pin)

    unpin = commands.add_parser("unpin", help="remove a run's durable pin")
    unpin.add_argument("run_id")
    unpin.add_argument("--cascade", action="store_true",
                       help="unpin even if child runs depend on it (default: refuse)")
    unpin.add_argument("--json", action="store_true")
    unpin.set_defaults(fn=cmd_unpin)

    listing = commands.add_parser("list", help="list every durably pinned checkpoint")
    listing.add_argument("--json", action="store_true")
    listing.set_defaults(fn=cmd_list)

    return parser


__all__ = ["add_subparser", "cmd_list", "cmd_pin", "cmd_unpin"]
