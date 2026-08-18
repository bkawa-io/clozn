"""commands.test_model -- `clozn test-model` (FRONTIER_BETS §9.3, "the model's own CI"): run the curated
probe sets with GREEDY decoding against a live engine, and diff the outputs against a stored golden
fixture -- so a quant swap, a raw-steering change, or an engine upgrade that silently changes what the
model SAYS shows up as a failed regression check instead of a vibe.

Zero research risk, pure wiring: reuses `clozn.eval.probes.run_probes` (already greedy -- temperature=0.0,
see that function's body) and `clozn.eval.outcome.grade` completely unmodified via `clozn.eval.golden`,
which also owns the fixture's on-disk shape and the pairwise diff. This module is only the CLI shell
around it (mirrors commands.eval's split against eval.bench).

    clozn test-model                 run + diff against the saved fixture
    clozn test-model --save          run + OVERWRITE the fixture with the CURRENT outputs
    clozn test-model --set hard      just one probe set (easy|hard|arith|both|all; default all)
    clozn test-model --json          machine-readable report instead of text

Needs a running Clozn gateway (default http://127.0.0.1:8080) -- same live precondition as `clozn eval`.

Exit codes:
    0 -- --save (always), or a diff run with zero regressions
    1 -- a diff run found at least one regression
    2 -- a diff run was requested but no fixture has ever been saved (`clozn test-model --save` first)

Registration in clozn/cli/main.py mirrors commands.eval exactly: import `cmd_test_model, add_subparser as
_add_test_model` alongside the other commands.* imports, and call `_add_test_model(sub)` in
build_parser() before `return p`.
"""
from __future__ import annotations

import json

from clozn.cli import formatting as fmt


def add_subparser(sub):
    """Register `clozn test-model` on an argparse subparsers object (own function so its wiring is
    testable without dispatching; mirrors commands.eval.add_subparser / commands.quant_check.add_subparser)."""
    pt = sub.add_parser("test-model", help="the model's own CI: run pinned probes GREEDILY against a live "
                        "engine and diff against a saved golden fixture (a regression exits 1)")
    pt.add_argument("--url", default="http://127.0.0.1:8080", help="Clozn gateway base URL (default :8080)")
    pt.add_argument("--set", dest="which", default="all",
                    choices=["easy", "hard", "arith", "extended", "both", "all"],
                    help="which built-in probe set (default: all)")
    pt.add_argument("--save", action="store_true",
                    help="overwrite the golden fixture with THIS run's outputs, instead of diffing against it")
    pt.add_argument("--json", action="store_true", help="print the machine-readable report instead of text")
    pt.set_defaults(fn=cmd_test_model)
    return pt


def _format_report(report: dict) -> str:
    """Pure JSON(golden.diff's result, + a little provenance) -> text render -- no I/O, testable on a
    canned dict exactly like commands.test.format_test_report."""
    lines = [f"clozn test-model -- set={report.get('which')}  fixture set={report.get('fixture_which')}"]
    fmodel, cmodel = report.get("fixture_model"), report.get("current_model")
    if fmodel and cmodel and fmodel != cmodel:
        lines.append(f"  {fmt.DIM}note: fixture model {fmodel!r} != current model {cmodel!r} -- a diff "
                     f"here may reflect the model swap, not a regression{fmt.RST}")
    lines.append(f"  regressions={report['n_regressions']}  new_passes={report['n_new_passes']}  "
                 f"changed={report['n_changed']}  unchanged={report['n_unchanged']}  "
                 f"new={report['n_new']}  missing={report['n_missing']}")
    if report["regressions"]:
        lines.append("")
        lines.append(f"{fmt.BOLD}regressions (was correct, now wrong):{fmt.RST}")
        for r in report["regressions"]:
            lines.append(f"  - {r['q'][:60]}")
            lines.append(f"      was: {r['was_reply']!r}   now: {r['now_reply']!r}   (gold: {r['gold']!r})")
    if report["new_passes"]:
        lines.append("")
        lines.append(f"{fmt.BOLD}new passes (was wrong, now correct):{fmt.RST}")
        for r in report["new_passes"]:
            lines.append(f"  + {r['q'][:60]}   {r['was_reply']!r} -> {r['now_reply']!r}")
    if report["changed"]:
        lines.append("")
        lines.append("changed outputs (same correctness, different text):")
        for r in report["changed"]:
            lines.append(f"  ~ {r['q'][:60]}: {r['was_reply']!r} -> {r['now_reply']!r}")
    if report["missing"]:
        lines.append("")
        lines.append(f"{fmt.DIM}in the fixture but not this run's probe set: {report['n_missing']}{fmt.RST}")
    lines.append("")
    lines.append("PASS: no regressions" if not report["n_regressions"]
                 else f"FAIL: {report['n_regressions']} regression(s)")
    return "\n".join(lines)


def cmd_test_model(args):
    """`clozn test-model [--set ...] [--save] [--json]` -- LIVE: needs a running Clozn gateway (see module
    docstring). --save persists the current run as the golden fixture. Otherwise runs + diffs against the
    saved fixture and returns the exit code documented at the top of this module."""
    from clozn.eval import golden

    rows = golden.run_and_grade(args.url, args.which)
    health = golden.engine_health(args.url)

    if args.save:
        path = golden.save(rows, which=args.which, health=health)
        if args.json:
            print(json.dumps({"saved": path, "which": args.which, "n": len(rows), "model": health.get("model")},
                             indent=2, default=str))
        else:
            print(f"clozn test-model: saved {len(rows)} probe(s) -> {path}")
        return 0

    fixture = golden.load()
    if fixture is None:
        msg = "clozn test-model: no golden fixture saved yet -- run `clozn test-model --save` first"
        if args.json:
            print(json.dumps({"error": msg}, indent=2))
        else:
            print(msg)
        return 2

    report = golden.diff(fixture.get("rows") or [], rows)
    report["which"] = args.which
    report["fixture_which"] = fixture.get("which")
    report["fixture_model"] = fixture.get("model")
    report["fixture_saved_ts"] = fixture.get("saved_ts")
    report["current_model"] = health.get("model")

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print(_format_report(report))
    return 1 if report["n_regressions"] else 0
