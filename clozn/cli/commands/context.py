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

from clozn.cli import main as ctx
from clozn.runs.context_receipt import build_context_receipt, read_receipt
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
        lines.extend(["", "termination · " + str(termination.get("reason")) + raw_note])

    lines.extend(["", "input policy · " + str(receipt.get("input_policy") or "unknown")])

    if detailed:
        lines.extend(["", f"SEGMENTS -- delivered ({len(receipt.get('delivered') or [])})"])
        for seg in receipt.get("delivered") or []:
            included = seg.get("included")
            mark = "omitted" if included is False else "included"
            reason = f" reason={seg['reason']}" if seg.get("reason") else ""
            label = seg.get("source_label", "?")
            lines.append(f"  [{seg.get('original_order')}] {seg.get('segment_id')} {label} - {mark}{reason}")
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
            if "token_count" in rendered:
                estimated = " (estimated)" if rendered.get("estimated") else ""
                lines.append(f"  token_count {rendered['token_count']}{estimated}")
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
