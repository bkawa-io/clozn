"""Evidence-only diagnosis of recorded request latency and output cutoff."""
from __future__ import annotations

import json

from clozn.runs.diagnosis import diagnose


def add_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "diagnose", help="explain recorded latency and output cutoff without generating")
    parser.add_argument(
        "target", nargs="?", default=None,
        help="'last' for the latest matching run, or an exact run id. Omit and pass --last instead.")
    parser.add_argument("--last", action="store_true",
                        help="alias for target 'last' (clozn diagnose --last --performance)")
    parser.add_argument("--session", default=None,
                        help="for 'last', exact caller-known X-Clozn-Session-Id")
    parser.add_argument("--client-id", default=None,
                        help="for 'last', exact caller-known X-Clozn-Client-Id")
    parser.add_argument("--client", default=None,
                        help="for 'last', coarse recorded client label")
    parser.add_argument("--model", default=None, help="for 'last', model filter")
    parser.add_argument("--include-derived", action="store_true",
                        help="allow replay/branch/fork runs when selecting 'last'")
    parser.add_argument("--performance", action="store_true",
                        help="also show the clozn.performance-trace.v1 phase/metric/rule-engine report "
                             "(clozn.runs.perf_diagnosis) instead of the why-slow/why-cut-off report")
    parser.add_argument("--json", action="store_true", help="print the structured diagnosis")
    parser.set_defaults(fn=cmd_diagnose)


def _resolve_target(args) -> str:
    from clozn.cli.main import CloznError

    target = args.target
    if args.last:
        if target not in (None, "last"):
            raise CloznError("--last cannot be combined with an explicit run id; drop one")
        target = "last"
    if target is None:
        raise CloznError("give a run id, 'last', or --last")
    return target


def _select_run(args, runlog):
    from clozn.cli.main import CloznError

    filters = {
        "client": args.client,
        "client_id": args.client_id,
        "session_id": args.session,
        "model": args.model,
        "include_derived": bool(args.include_derived),
    }
    if args.target == "last":
        summary = runlog.latest_run(**filters)
        if summary is None:
            raise CloznError("no matching recorded run found")
        run = runlog.get_run(summary.get("id", ""))
        if run is None:
            raise CloznError("the latest matching run could not be read")
        return run

    if any(value is not None for value in (
        args.session, args.client_id, args.client, args.model
    )) or args.include_derived:
        raise CloznError("selection filters apply only when the target is 'last'")
    run = runlog.get_run(args.target)
    if run is None:
        raise CloznError(f"run not found: {args.target}")
    return run


def _finding_line(finding: dict) -> str:
    status = str(finding.get("status") or "unknown").replace("_", " ")
    return f"  {str(finding.get('id') or 'unknown').replace('_', ' '):22} {status:13} {finding.get('text', '')}"


def format_diagnosis(report: dict) -> str:
    """Render the diagnosis without adding conclusions not present in its findings."""
    slow = report.get("why_slow") or {}
    cutoff = report.get("why_cut_off") or {}
    auxiliary = report.get("client_auxiliary_calls") or {}
    lines = [f"diagnosis - {report.get('run_id') or '?'}", "", "WHY SLOW"]
    findings = slow.get("findings") if isinstance(slow.get("findings"), list) else []
    if findings:
        lines.extend(_finding_line(item) for item in findings if isinstance(item, dict))
    else:
        lines.append("  no timing findings recorded")
    lines.extend(["", "WHY CUT OFF"])
    finding = cutoff.get("finding")
    lines.append(_finding_line(finding) if isinstance(finding, dict)
                 else "  output cutoff          unavailable   No cutoff finding recorded.")
    lines.extend(["", "CLIENT AUXILIARY CALLS", _finding_line(auxiliary)])
    return "\n".join(lines)


def _phase_line(phase: dict) -> str:
    name = str(phase.get("name") or "unknown").replace("_", " ")
    seconds = (phase.get("duration_ns") or 0) / 1e9
    owner = phase.get("owner")
    measurement = str(phase.get("measurement") or "measured").upper()
    aggregation = str(phase.get("aggregation") or "exclusive").replace("_", " ").upper()
    suffix = f"  ({owner})" if isinstance(owner, str) and owner else ""
    return f"  {name:20} {seconds:8.3f}s  {measurement:9} {aggregation:12}{suffix}"


def format_performance(report: dict) -> str:
    """Render the clozn.performance-trace.v1 report without adding a cause its `diagnoses` didn't fire.

    Fired rules are shown as LIKELY CAUSE/POSSIBLE FIX pairs; rules this particular run's evidence could
    not support are listed by name so their absence is never mistaken for 'nothing wrong here'.
    """
    lines = [f"performance - {report.get('run_id') or '?'}", "", "PHASES (monotonic)"]
    phases = report.get("phases") if isinstance(report.get("phases"), list) else []
    if phases:
        lines.extend(_phase_line(p) for p in phases if isinstance(p, dict))
    else:
        lines.append("  no measured phase was recorded for this run")

    metrics = report.get("metrics") if isinstance(report.get("metrics"), dict) else {}
    if metrics:
        lines.extend(["", "METRICS"])
        lines.extend(f"  {key:28} {value}" for key, value in metrics.items())

    aggregation = report.get("aggregation") if isinstance(report.get("aggregation"), dict) else {}
    if aggregation:
        known_s = (aggregation.get("known_duration_ns") or 0) / 1e9
        lines.extend(["", "ACCOUNTED TIME", f"  known measured phases       {known_s:.3f}s"])
        if isinstance(aggregation.get("unaccounted_duration_ns"), int):
            lines.append(
                f"  unaccounted                 {aggregation['unaccounted_duration_ns'] / 1e9:.3f}s"
            )
        if isinstance(aggregation.get("measurement_coverage"), (int, float)):
            lines.append(f"  coverage                    {aggregation['measurement_coverage'] * 100:.1f}%")

    diagnoses = report.get("diagnoses") if isinstance(report.get("diagnoses"), list) else []
    fired = [d for d in diagnoses if isinstance(d, dict) and d.get("status") == "fired"]
    unavailable = [d.get("rule") for d in diagnoses
                   if isinstance(d, dict) and d.get("status") == "unavailable"]

    lines.extend(["", "LIKELY CAUSE"])
    if fired:
        for item in fired:
            lines.append(f"  {item.get('likely_cause', '')}")
            lines.extend(["", "POSSIBLE FIX", f"  {item.get('possible_fix', '')}"])
    else:
        lines.append("  No cause could be identified from currently recorded evidence.")

    if unavailable:
        lines.extend(["", "EVIDENCE NOT YET AVAILABLE",
                      f"  {', '.join(str(r) for r in unavailable)}"])
    return "\n".join(lines)


def cmd_diagnose(args) -> int:
    import clozn.runs.store as runlog

    args.target = _resolve_target(args)
    run = _select_run(args, runlog)
    related = runlog.iter_runs(limit=200)

    if args.performance:
        from clozn.cli.main import CloznError
        from clozn.runs.perf_diagnosis import build_performance_report
        from clozn.runs.perf_trace import PerfTraceError
        try:
            report = build_performance_report(run, related_runs=related)
        except PerfTraceError as exc:
            raise CloznError(str(exc))
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            print(format_performance(report))
        return 0

    report = diagnose(run, related_runs=related)
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print(format_diagnosis(report))
    return 0
