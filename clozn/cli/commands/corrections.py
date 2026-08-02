"""`clozn corrections ...` -- F5's CLI exposure for the scoped correction store ("Teach Once"), plus F6's
`verify` subcommand for the verify-before-save teaching loop (`clozn/runs/teaching_loop.py`).

The HTTP adapter lives at `/corrections` and the CLI remains a useful local/scriptable surface. Every
subcommand is a thin argument-parsing wrapper over clozn.runs.corrections (or, for `verify`,
clozn.runs.teaching_loop) -- no selection/precedence/conflict/verification logic lives here.
"""
from __future__ import annotations

import json

CLOZN_AUTOLOAD = True


def _print(doc, *, as_json: bool = True) -> None:
    # Every subcommand except `list` prints the same structured document either way today -- `--json` is
    # accepted on all of them for a stable, scriptable interface, but there is no separate prose rendering
    # to fall back to yet. `list` is the one command with a real human-readable form; see cmd_list.
    print(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True))


def add_subparser(sub) -> None:
    parser = sub.add_parser(
        "corrections",
        help="F5: scoped correction store (\"Teach Once\") -- draft/confirm/disable/enable/delete/"
             "undo/export/resolve; F6: verify (verify-before-save teaching loop)")
    actions = parser.add_subparsers(dest="corrections_command", required=True)

    draft = actions.add_parser("draft", help="create an inert, unconfirmed correction")
    draft.add_argument("--scope-kind", required=True,
                       choices=["session", "client", "model", "project", "global_local"])
    draft.add_argument("--scope-value", default=None,
                       help="required for every scope kind except global_local")
    draft.add_argument("--type", dest="correction_type", required=True,
                       choices=["output_format", "source_requirement", "style", "forbidden_behavior"])
    draft.add_argument("--content", required=True, help="the correction text")
    draft.add_argument("--json", action="store_true")
    draft.set_defaults(fn=cmd_draft)

    confirm = actions.add_parser("confirm", help="explicitly confirm a drafted correction (the ONLY "
                                                  "way it becomes selectable)")
    confirm.add_argument("correction_id")
    confirm.add_argument("--json", action="store_true")
    confirm.set_defaults(fn=cmd_confirm)

    disable = actions.add_parser("disable", help="reversible: stop selecting this correction")
    disable.add_argument("correction_id")
    disable.add_argument("--json", action="store_true")
    disable.set_defaults(fn=cmd_disable)

    enable = actions.add_parser("enable", help="re-enable a disabled correction")
    enable.add_argument("correction_id")
    enable.add_argument("--json", action="store_true")
    enable.set_defaults(fn=cmd_enable)

    delete = actions.add_parser("delete", help="PERMANENT: scrub content, keep the hash + event ledger")
    delete.add_argument("correction_id")
    delete.add_argument("--reason", default=None)
    delete.add_argument("--yes", action="store_true", help="required to confirm the permanent scrub")
    delete.add_argument("--json", action="store_true")
    delete.set_defaults(fn=cmd_delete)

    undo = actions.add_parser("undo", help="revert the most recent confirm/disable/enable transition "
                                            "(never a deletion)")
    undo.add_argument("correction_id")
    undo.add_argument("--json", action="store_true")
    undo.set_defaults(fn=cmd_undo)

    show = actions.add_parser("show", help="print one correction")
    show.add_argument("correction_id")
    show.add_argument("--json", action="store_true")
    show.set_defaults(fn=cmd_show)

    listp = actions.add_parser("list", help="list corrections")
    listp.add_argument("--scope-kind", default=None,
                       choices=["session", "client", "model", "project", "global_local"])
    listp.add_argument("--type", dest="correction_type", default=None,
                       choices=["output_format", "source_requirement", "style", "forbidden_behavior"])
    listp.add_argument("--exclude-disabled", action="store_true")
    listp.add_argument("--include-deleted", action="store_true")
    listp.add_argument("--json", action="store_true")
    listp.set_defaults(fn=cmd_list)

    export = actions.add_parser("export", help="print a portable bundle: the correction + its full "
                                                "event ledger")
    export.add_argument("correction_id")
    export.set_defaults(fn=cmd_export)

    resolve = actions.add_parser(
        "resolve",
        help="show what would apply for an explicit scope context, without recording anything -- the "
             "same content-blind computation a run creation path would use")
    resolve.add_argument("--session", default=None, dest="session_id")
    resolve.add_argument("--client", default=None, dest="client_id")
    resolve.add_argument("--project", default=None, dest="project_id")
    resolve.add_argument("--model-sha256", default=None)
    resolve.add_argument("--no-global-local", action="store_true",
                         help="exclude global_local-scoped corrections from this resolution")
    resolve.add_argument("--json", action="store_true")
    resolve.set_defaults(fn=cmd_resolve)

    verify = actions.add_parser(
        "verify",
        help="F6: verify a drafted correction against a target-failure/child-retry run pair "
             "(clozn.replay.controlled's exact comparison) and promote it ONLY if the child run did not "
             "reproduce the failure; a failed verification stays a draft and is recorded as evidence")
    verify.add_argument("correction_id")
    verify.add_argument("--target", required=True, dest="target_run_id",
                        help="the recorded failure run id this attempt tried to fix")
    verify.add_argument("--child", required=True, dest="child_run_id",
                        help="the already-persisted retry run id to compare against --target")
    verify.add_argument("--match", default="exact_output", dest="match_criterion",
                        choices=["exact_output", "tool_parse", "finish_reason", "token_budget"],
                        help="exact outcome criterion; no semantic similarity (default exact_output)")
    verify.add_argument("--json", action="store_true")
    verify.set_defaults(fn=cmd_verify)


def _fail(exc: Exception):
    from clozn.cli.main import CloznError
    raise CloznError(f"corrections: {exc}") from None


def cmd_draft(args) -> int:
    from clozn.runs import corrections
    try:
        doc = corrections.draft_correction(
            scope_kind=args.scope_kind, correction_type=args.correction_type,
            content=args.content, scope_value=args.scope_value)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(doc, as_json=args.json)
    return 0


def cmd_confirm(args) -> int:
    from clozn.runs import corrections
    try:
        result = corrections.confirm_correction(args.correction_id)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(result, as_json=args.json)
    if not args.json and result.get("potential_conflicts"):
        print(f"\nnote: {len(result['potential_conflicts'])} other confirmed correction(s) of the same "
              f"type could conflict with this one at resolution time -- surfaced, not blocked.")
    return 0


def cmd_disable(args) -> int:
    from clozn.runs import corrections
    try:
        doc = corrections.disable_correction(args.correction_id)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(doc, as_json=args.json)
    return 0


def cmd_enable(args) -> int:
    from clozn.runs import corrections
    try:
        doc = corrections.enable_correction(args.correction_id)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(doc, as_json=args.json)
    return 0


def cmd_delete(args) -> int:
    if not args.yes:
        from clozn.cli.main import CloznError
        raise CloznError("corrections delete is permanent (content is scrubbed); re-run with --yes")
    from clozn.runs import corrections
    try:
        doc = corrections.delete_correction(args.correction_id, reason=args.reason)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(doc, as_json=args.json)
    return 0


def cmd_undo(args) -> int:
    from clozn.runs import corrections
    try:
        doc = corrections.undo_last_change(args.correction_id)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(doc, as_json=args.json)
    return 0


def cmd_show(args) -> int:
    from clozn.runs import corrections
    doc = corrections.get_correction(args.correction_id)
    if doc is None:
        from clozn.cli.main import CloznError
        raise CloznError(f"no correction {args.correction_id!r}")
    _print(doc, as_json=args.json)
    return 0


def cmd_list(args) -> int:
    from clozn.runs import corrections
    docs = corrections.list_corrections(
        scope_kind=args.scope_kind, correction_type=args.correction_type,
        include_disabled=not args.exclude_disabled, include_deleted=args.include_deleted)
    if args.json:
        print(json.dumps(docs, indent=2, ensure_ascii=False, sort_keys=True))
        return 0
    if not docs:
        print("no corrections")
        return 0
    for doc in docs:
        scope = doc.get("scope") or {}
        scope_text = scope.get("kind", "?") + (f"={scope['value']}" if scope.get("value") else "")
        state = "deleted" if doc.get("deleted_ts") else ("enabled" if doc.get("enabled") else
                ("confirmed(disabled)" if doc.get("confirmed_ts") else "drafted"))
        print(f"{doc['id']}  {doc.get('type')}  {scope_text}  {state}")
    return 0


def cmd_export(args) -> int:
    from clozn.runs import corrections
    doc = corrections.export_correction(args.correction_id)
    if doc is None:
        from clozn.cli.main import CloznError
        raise CloznError(f"no correction {args.correction_id!r}")
    print(json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


def cmd_resolve(args) -> int:
    from clozn.runs import corrections
    try:
        resolution = corrections.resolve_corrections(
            session_id=args.session_id, client_id=args.client_id, project_id=args.project_id,
            model_sha256=args.model_sha256, include_global_local=not args.no_global_local)
    except corrections.CorrectionError as exc:
        _fail(exc)
    _print(resolution, as_json=args.json)
    return 0


def cmd_verify(args) -> int:
    from clozn.runs import corrections, teaching_loop
    try:
        result = teaching_loop.verify_and_promote(
            args.correction_id, target_run_id=args.target_run_id, child_run_id=args.child_run_id,
            match_criterion=args.match_criterion)
    except (teaching_loop.TeachingLoopError, corrections.CorrectionError) as exc:
        _fail(exc)
    _print(result, as_json=args.json)
    if not args.json:
        print(f"\n{result['verification']}: {result['reason']}")
    return 0
