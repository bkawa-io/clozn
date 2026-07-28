"""Readable context-delivery receipts from the local run journal.

Renders both receipt shapes a run can carry (clozn.runs.context_receipt.read_receipt): the pre-2026-07-27
legacy delivered/survived object, unchanged, and the new clozn.context-receipt.v1 segment-array shape,
whose compact view still leads with full message/prompt text (pulled from the run's own fields, not the
receipt's metadata-only segments -- see clozn.runs.context_receipt's module docstring) so the everyday
"what did the model see" question reads the same regardless of which shape produced it. --detailed adds
the segment/omission/transformation/termination detail the new shape carries that the old one did not.
"""
from __future__ import annotations

import json
import hashlib
from copy import deepcopy
from pathlib import Path

from clozn.cli import main as ctx
from clozn.runs.context_receipt import _apply_privacy, build_context_receipt, read_receipt
import clozn.runs.store as runlog


def _receipt(run: dict) -> dict:
    view = read_receipt(run)
    if view["shape"] in ("new", "legacy"):
        return view["receipt"]
    # absent (older-than-Phase-2.4 run, or context-receipt privacy was "off") or unrecognized (a future
    # schema bump this build predates) -- reconstruct best-effort from the run's own raw fields rather
    # than show nothing.
    return build_context_receipt(
        messages=run.get("messages"),
        assembled_messages=run.get("assembled_messages"),
        final_prompt=run.get("final_prompt"),
        finish_reason=run.get("finish_reason"),
        meta=run.get("meta"),
        trace=run.get("trace"),
        run_id=run.get("id"),
        identity=run.get("identity"),
        error=run.get("error"),
    )


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


def _legacy_body(receipt: dict) -> list[str]:
    """The pre-2026-07-27 shape's own rendering -- unchanged from before this feature's rewrite."""
    delivered = receipt.get("delivered") or {}
    survived = receipt.get("survived") or {}
    lines = ["", "DELIVERED", str(delivered.get("meaning") or "")]
    lines.extend(_message_lines(delivered.get("messages")))
    lines.extend(["", "SURVIVED", str(survived.get("meaning") or "")])
    assembled = survived.get("assembled_messages")
    if isinstance(assembled, list):
        lines.append("  assembled messages")
        lines.extend(_message_lines(assembled))
    else:
        lines.append("  assembled messages: (not captured)")
    final_prompt = survived.get("final_prompt")
    if isinstance(final_prompt, str):
        lines.append("  exact rendered prompt")
        lines.extend("    " + line for line in final_prompt.splitlines() or [""])
    else:
        lines.append("  exact rendered prompt: (not captured)")
    lines.extend(["", "input policy · " + str(receipt.get("input_policy") or "unknown")])
    return lines


def _new_body(run: dict, receipt: dict, *, detailed: bool) -> list[str]:
    survived = receipt.get("survived") or {}
    lines = ["", "DELIVERED", "messages accepted by the gateway and handed to prompt assembly"]
    lines.extend(_message_lines(run.get("messages")))
    lines.extend(["", "SURVIVED", "post-assembly input retained as evidence of what reached generation"])
    assembled = survived.get("assembled_messages")
    if isinstance(assembled, list):
        lines.append("  assembled messages")
        lines.extend(_message_lines(assembled))
    elif survived.get("content_withheld_by_privacy_tier"):
        lines.append(f"  assembled messages: withheld by receipt privacy tier "
                     f"{survived['content_withheld_by_privacy_tier']!r}")
    else:
        lines.append("  assembled messages: (not captured)")
    final_prompt = survived.get("final_prompt")
    if isinstance(final_prompt, str):
        lines.append("  exact rendered prompt")
        lines.extend("    " + line for line in final_prompt.splitlines() or [""])
    elif survived.get("content_withheld_by_privacy_tier"):
        lines.append("  exact rendered prompt: withheld by receipt privacy tier")
    else:
        lines.append("  exact rendered prompt: (not captured)")

    termination = receipt.get("termination")
    if termination:
        raw = termination.get("reason_raw")
        raw_note = f" (raw: {raw})" if raw and raw != termination.get("reason") else ""
        source_note = f" via {termination['source']}" if termination.get("source") else ""
        lines.extend([
            "",
            "termination · " + str(termination.get("reason")) + raw_note + source_note,
        ])

    lines.extend(["", "input policy · " + str(receipt.get("input_policy") or "unknown")])

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
            if "tokens" in rendered or "token_count" in rendered:
                estimated = " (estimated)" if rendered.get("estimated") else ""
                lines.append(
                    f"  tokens {rendered.get('tokens', rendered.get('token_count'))}{estimated}"
                )
            if "content_available" in rendered:
                lines.append(f"  content_available {str(rendered['content_available']).lower()}")
        privacy = receipt.get("privacy")
        if privacy:
            lines.append(f"receipt privacy · {privacy}")
    return lines


def format_context(run: dict, *, detailed: bool = False) -> str:
    receipt = _receipt(run)
    limits = receipt.get("limits") or {}
    warnings = receipt.get("warnings") or []

    lines = [f"context receipt · {run.get('id', '?')}"]
    if warnings:
        lines.append("WARNING · " + str(warnings[0].get("message") or "reply was cut off"))
    else:
        lines.append("status · no recorded input truncation or output cutoff")

    values = []
    for key, label in (("prompt_tokens", "prompt"),
                       ("context_window_tokens", "context window"),
                       ("requested_max_tokens", "requested output"),
                       ("generated_tokens", "generated")):
        if isinstance(limits.get(key), int):
            values.append(f"{label} {limits[key]} tok")
    if values:
        lines.append("limits · " + " · ".join(values))

    is_new = isinstance(receipt.get("schema_version"), str)
    lines.extend(_new_body(run, receipt, detailed=detailed) if is_new else _legacy_body(receipt))
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
    "metadata_only": ["survived.final_prompt", "survived.assembled_messages"],
    "hashes_only": [
        "survived.final_prompt",
        "survived.assembled_messages",
        "segments.source_label",
        "segments.client_source_id",
        "segments.delivered_bytes",
    ],
    "off": ["all receipt fields except schema_version, run_id, and privacy"],
}


def export_context_receipt(run: dict, *, privacy: str) -> dict:
    """Derive a lower- or equal-disclosure export without mutating the stored run."""
    view = read_receipt(run)
    if view["shape"] != "new":
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
        "context", help="inspect what a run delivered and what survived into generation")
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
