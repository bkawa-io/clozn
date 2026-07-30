"""`clozn investigate-experiment RUN_ID ...` -- C3's CLI exposure (core slice, item 4).

"Point at any passage or setting and ask whether it mattered." This command loads one recorded run from
the local journal, builds the requested `clozn.investigation-experiment.v1` `intervention` from its flags,
and runs `clozn.runs.investigation_experiment.plan_experiment()` -- the SAME eligibility planner a future
HTTP surface will call. It prints the resulting document (refused, with a specific typed reason, or
planned, with the exact arm order and the bridged span/sampler spec a live executor would run) and exits 0
in EITHER case: a typed refusal is a successful ANSWER to "can this be run", not a command failure. Only a
malformed invocation (a bad flag combination, a run that does not exist) exits non-zero.

WHY THIS COMMAND NEVER EXECUTES (plan-only, deliberately)
-----------------------------------------------------------
`clozn.receipts.investigation_experiment.run_experiment()` -- the four-arm controlled executor -- exists
and is fully unit-tested against a fake substrate, but it needs a LIVE `sub.chat()`-capable substrate, and
this slice has no in-scope way to hand it one: the studio's live substrate lives in-process inside
`clozn/server` (off limits this slice -- another agent holds every file under `clozn/server/routes/`, and
the owner's brief says explicitly "the HTTP surface for C3 is a later slice"), and spawning a SECOND local
engine process the way `clozn diff-adapter`/`clozn diff-model` do is a materially bigger, riskier
integration for a single CLI command whose whole point here is the planner + bridge, not a new live-model-
boot path. `clozn.cli.commands.compare_runs`'s own `--plan`/`--dry-run` flags established exactly this
precedent already ("plan ... swaps and cost WITHOUT model runs") for a structurally similar
run-vs-run change test; this command is the same shape of tool for C3's single-run "did this matter"
question. Wiring live execution (through a route, once one exists, or a spawned engine) is a named,
disclosed follow-up, not something this command fakes by printing numbers it never measured.
"""
from __future__ import annotations

import json

CLOZN_AUTOLOAD = True

_SAMPLER_INT_FIELDS = {"top_k", "seed"}
_SAMPLER_FLOAT_FIELDS = {"temperature", "top_p", "rep_penalty"}
_SAMPLER_FIELDS = _SAMPLER_INT_FIELDS | _SAMPLER_FLOAT_FIELDS


def add_subparser(sub) -> None:
    parser = sub.add_parser(
        "investigate-experiment",
        help="plan a controlled 'did this matter' experiment over one recorded run (C3) -- "
             "eligibility + the arbitrary-span bridge, never executes")
    parser.add_argument("run_id", help="a run id from the local journal")

    kind = parser.add_mutually_exclusive_group(required=True)
    kind.add_argument("--remove-span", metavar="ADDRESS_ID",
                      help="remove the text-span-address's span entirely")
    kind.add_argument("--replace-span-neutral", metavar="ADDRESS_ID",
                      help="replace the text-span-address's span with register-matched neutral filler "
                           "of the same length")
    kind.add_argument("--omit-source", metavar="SOURCE_ID",
                      help="omit every resolvable span attached to this client_source_id")
    kind.add_argument("--sampler", metavar="KEY=VALUE[,KEY=VALUE...]",
                      help="change one or more sampler settings: temperature,top_k,top_p,seed,rep_penalty")
    kind.add_argument("--adapter-scale", type=float, metavar="F",
                      help="scale the currently-loaded adapter by F (0 = detach) -- always refused in "
                           "this slice; see this module's own docstring for why")

    parser.add_argument("--json", action="store_true", help="print the raw document as JSON")
    parser.set_defaults(fn=cmd_investigate_experiment)


def _parse_sampler(raw: str):
    from clozn.cli.main import CloznError

    overrides: dict = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise CloznError(f"--sampler entries must be KEY=VALUE (got {item!r})")
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in _SAMPLER_FIELDS:
            raise CloznError(
                f"unknown sampler field {key!r}; choose one of {sorted(_SAMPLER_FIELDS)}")
        try:
            overrides[key] = int(value) if key in _SAMPLER_INT_FIELDS else float(value)
        except ValueError:
            raise CloznError(f"--sampler {key} must be a number (got {value!r})") from None
    if not overrides:
        raise CloznError("--sampler needs at least one KEY=VALUE pair")
    return overrides


def _build_intervention(args) -> dict:
    if args.remove_span is not None:
        return {"kind": "remove_span", "span_address_id": args.remove_span}
    if args.replace_span_neutral is not None:
        return {"kind": "replace_span_neutral", "span_address_id": args.replace_span_neutral}
    if args.omit_source is not None:
        return {"kind": "omit_source", "source_id": args.omit_source}
    if args.sampler is not None:
        return {"kind": "sampler_change", "overrides": _parse_sampler(args.sampler)}
    return {"kind": "adapter_scale", "scale": args.adapter_scale}


def format_plan(document: dict) -> str:
    """A short, human-readable rendering that never adds a conclusion the document itself does not carry."""
    lines = [f"investigation-experiment - {document.get('run_id') or '?'}",
             f"  intervention: {document.get('intervention', {}).get('kind')}",
             f"  phase:        {document.get('phase')}"]
    eligibility = document.get("eligibility") or {}
    lines.append(f"  eligibility:  {eligibility.get('state')}")
    reason = eligibility.get("reason")
    if isinstance(reason, dict):
        lines.append(f"    reason:     {reason.get('code')} -- {reason.get('message')}")
    plan = document.get("plan")
    if isinstance(plan, dict):
        lines.append(f"  arm order:    {', '.join(plan.get('arm_order') or [])}")
        resolved = plan.get("resolved") or {}
        for span in resolved.get("spans") or []:
            lines.append(f"  span:         message {span.get('message_index')} "
                         f"[{span.get('start')}, {span.get('end')})")
        note = resolved.get("random_control_note")
        if note:
            lines.append(f"  random control: unavailable -- {note}")
        elif resolved.get("random_control_spans"):
            ctrl = resolved["random_control_spans"][0]
            lines.append(f"  random control: message {ctrl.get('message_index')} "
                         f"[{ctrl.get('start')}, {ctrl.get('end')})")
        overrides = resolved.get("sampler_overrides")
        if overrides:
            lines.append(f"  sampler overrides: {overrides}")
    return "\n".join(lines)


def cmd_investigate_experiment(args) -> int:
    from clozn.cli.main import CloznError
    import clozn.runs.store as runlog
    from clozn.runs.investigation_experiment import plan_experiment

    run = runlog.get_run(args.run_id)
    if run is None:
        raise CloznError(f"run not found: {args.run_id}")

    intervention = _build_intervention(args)
    document = plan_experiment(run, intervention)

    if args.json:
        print(json.dumps(document, indent=2, ensure_ascii=False))
    else:
        print(format_plan(document))
    return 0
