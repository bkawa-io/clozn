"""Readable context-delivery receipts from the local run journal.

The compact view leads with full message and prompt text from the run record; ``--detailed`` adds the
receipt's segment, omission, transformation, and termination evidence.
"""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

from clozn.cli import main as ctx
from clozn.runs.context_receipt import _apply_privacy, read_receipt
import clozn.runs.store as runlog


def _receipt(run: dict) -> dict:
    view = read_receipt(run)
    if view["shape"] == "recorded":
        return view["receipt"]
    return {}


def _message_lines(messages) -> list[str]:
    if not messages:
        return ["  (none captured)"]
    lines: list[str] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "unknown")
        lines.append(f"  [{index}] {role}")
        content = str(message.get("content") or "")
        lines.extend("    " + line for line in content.splitlines() or [""])
    return lines or ["  (none captured)"]


def _body(run: dict, receipt: dict, *, detailed: bool) -> list[str]:
    lines = ["", "DELIVERED", "messages accepted by the gateway and handed to prompt assembly"]
    lines.extend(_message_lines(run.get("messages")))
    lines.extend(["", "ASSEMBLED", "messages passed to the model's prompt renderer"])
    assembled = run.get("assembled_messages")
    if isinstance(assembled, list):
        lines.extend(_message_lines(assembled))
    else:
        lines.append("  (not captured)")
    lines.extend(["", "RENDERED PROMPT"])
    final_prompt = run.get("final_prompt")
    if isinstance(final_prompt, str):
        lines.extend("    " + line for line in final_prompt.splitlines() or [""])
    else:
        lines.append("  (not captured)")

    termination = receipt.get("termination")
    if termination:
        raw = termination.get("reason_raw")
        raw_note = f" (raw: {raw})" if raw and raw != termination.get("reason") else ""
        source_note = f" via {termination['source']}" if termination.get("source") else ""
        lines.extend([
            "",
            "termination · " + str(termination.get("reason")) + raw_note + source_note,
        ])

    if detailed:
        lines.extend(["", f"SEGMENTS -- delivered ({len(receipt.get('delivered') or [])})"])
        for seg in receipt.get("delivered") or []:
            included = seg.get("included")
            mark = "omitted" if included is False else "included"
            reason = f" reason={seg['reason']}" if seg.get("reason") else ""
            label = seg.get("source_label", "?")
            source_id = (
                f" source={seg['client_source_id']}"
                if seg.get("client_source_id") else ""
            )
            lines.append(
                f"  [{seg.get('original_order')}] {seg.get('segment_id')} "
                f"{label}{source_id} - {mark}{reason}"
            )
        omissions = receipt.get("omissions") or []
        lines.append(f"OMISSIONS ({len(omissions)})")
        for omission in omissions:
            lines.append(f"  {omission.get('segment_id')}: {omission.get('reason')} "
                         f"- {omission.get('detail', '')}")
        transformations = receipt.get("transformations") or []
        lines.append(f"TRANSFORMATIONS ({len(transformations)})")
        for transformation in transformations:
            lines.append(f"  {transformation.get('reason')}: {transformation.get('detail', '')}")
        rendered = receipt.get("rendered") or {}
        if rendered:
            lines.append("RENDERED")
            if rendered.get("sha256"):
                lines.append(f"  sha256 {rendered['sha256'][:16]}...")
            if "bytes" in rendered:
                lines.append(f"  bytes {rendered['bytes']}")
            if "tokens" in rendered:
                estimated = " (estimated)" if rendered.get("estimated") else ""
                lines.append(f"  tokens {rendered['tokens']}{estimated}")
        privacy = receipt.get("privacy")
        if privacy:
            lines.append(f"receipt privacy · {privacy}")
    return lines


def format_context(run: dict, *, detailed: bool = False) -> str:
    receipt = _receipt(run)
    if not receipt:
        return f"context receipt · {run.get('id', '?')}\nstatus · no readable Context Receipt recorded"
    rendered = receipt.get("rendered") if isinstance(receipt.get("rendered"), dict) else {}
    termination = receipt.get("termination") if isinstance(receipt.get("termination"), dict) else {}
    warnings = run.get("warnings") if isinstance(run.get("warnings"), list) else []

    lines = [f"context receipt · {run.get('id', '?')}"]
    cutoff = (
        termination.get("reason") in {"max_tokens", "context_limit"}
        or run.get("finish_reason") == "length"
    )
    if cutoff:
        message = next(
            (str(item.get("message")) for item in warnings
             if isinstance(item, dict) and item.get("message")),
            "generation stopped at a token budget; the reply may be incomplete",
        )
        lines.append("WARNING · " + message)
    else:
        lines.append("status · no token-budget cutoff recorded")

    values = [
        f"{label} {value} tok"
        for label, value in (
            ("prompt", rendered.get("tokens")),
            ("context window", receipt.get("context_window_tokens")),
            ("requested output", receipt.get("reserved_output_tokens")),
            ("generated", termination.get("generated_tokens")),
        )
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    if values:
        lines.append("tokens · " + " · ".join(values))

    lines.extend(_body(run, receipt, detailed=detailed))
    return "\n".join(lines)


def _print(run: dict, args) -> None:
    if args.json:
        print(json.dumps({"run_id": run["id"], "context_receipt": _receipt(run)},
                         indent=2, ensure_ascii=False))
    else:
        print(format_context(run, detailed=bool(getattr(args, "detailed", False))))


def cmd_context_last(args):
    rows = runlog.list_runs(limit=1, include_replays=False)
    if not rows:
        raise ctx.CloznError("no recorded run found")
    run = runlog.get_run(rows[0]["id"])
    if not run:
        raise ctx.CloznError("the latest run could not be read")
    _print(run, args)
    return 0


def cmd_context_show(args):
    run = runlog.get_run(args.run_id)
    if not run:
        raise ctx.CloznError(f"no such run: {args.run_id}")
    _print(run, args)
    return 0


_PRIVACY_ORDER = {"full": 0, "metadata_only": 1, "hashes_only": 2, "off": 3}
_OMITTED = {
    "full": [],
    "metadata_only": [],
    "hashes_only": [
        "segments.source_label",
        "segments.client_source_id",
        "segments.delivered_bytes",
    ],
    "off": ["all receipt fields except schema_version, run_id, and privacy"],
}


def export_context_receipt(run: dict, *, privacy: str) -> dict:
    """Derive a lower- or equal-disclosure export without mutating the stored run."""
    view = read_receipt(run)
    if view["shape"] != "recorded":
        raise ValueError(
            f"privacy-safe export requires a clozn.context-receipt.v1 source; "
            f"this run's receipt shape is {view['shape']}"
        )
    receipt = deepcopy(view["receipt"])
    stored_privacy = str(receipt.get("privacy") or "full")
    if stored_privacy not in _PRIVACY_ORDER:
        stored_privacy = "full"
    if _PRIVACY_ORDER[privacy] < _PRIVACY_ORDER[stored_privacy]:
        raise ValueError(
            f"cannot export at privacy {privacy!r}: the stored receipt is already "
            f"{stored_privacy!r} and omitted content cannot be recovered"
        )
    canonical = json.dumps(
        receipt, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    derived = _apply_privacy(receipt, privacy)
    from clozn import schemas
    schemas.validate(derived, "clozn.context-receipt.v1")
    document = {
        "schema_version": "clozn.context-receipt-export.v1",
        "run_id": str(run.get("id") or derived.get("run_id") or ""),
        "source_receipt_sha256": hashlib.sha256(canonical).hexdigest(),
        "privacy": privacy,
        "omitted_fields": list(_OMITTED[privacy]),
        "context_receipt": derived,
    }
    schemas.validate(document)
    return document


def cmd_context_export(args):
    run = runlog.get_run(args.run_id)
    if not run:
        raise ctx.CloznError(f"no such run: {args.run_id}")
    try:
        document = export_context_receipt(run, privacy=args.privacy)
    except ValueError as exc:
        raise ctx.CloznError(str(exc)) from None
    target = Path(args.out).expanduser()
    if target.exists():
        raise ctx.CloznError(f"refusing to overwrite existing export: {target}")
    from clozn._io import atomic_write_json
    atomic_write_json(
        str(target), document, indent=2, ensure_ascii=False, sort_keys=True
    )
    print(str(target))
    return 0


def add_subparser(subparsers):
    parser = subparsers.add_parser(
        "context", help="inspect what a run delivered, assembled, and rendered")
    commands = parser.add_subparsers(dest="context_cmd")

    last = commands.add_parser("last", help="show the latest organic run's context receipt")
    last.add_argument("--json", action="store_true", help="print the structured receipt")
    last.add_argument("--detailed", action="store_true",
                      help="also show segments, omissions, transformations, and rendered detail")
    last.set_defaults(fn=cmd_context_last)

    show = commands.add_parser("show", help="show one exact run's context receipt")
    show.add_argument("run_id")
    show.add_argument("--json", action="store_true", help="print the structured receipt")
    show.add_argument("--detailed", action="store_true",
                      help="also show segments, omissions, transformations, and rendered detail")
    show.set_defaults(fn=cmd_context_show)

    export = commands.add_parser(
        "export", help="write an immutable privacy-scoped context receipt export")
    export.add_argument("run_id")
    export.add_argument("--out", required=True, help="new JSON file to create (never overwritten)")
    export.add_argument(
        "--privacy", choices=tuple(_PRIVACY_ORDER), default="metadata_only",
        help="maximum disclosure tier for the derived export (default: metadata_only)",
    )
    export.set_defaults(fn=cmd_context_export)
