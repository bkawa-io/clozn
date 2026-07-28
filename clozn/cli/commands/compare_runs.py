"""commands.compare_runs -- `clozn compare-runs RUN_A RUN_B [--json] [--replay]` (agent roadmap feature
10, "What changed"): the consumer-facing "two runs differed -- what changed?" explainer, over
`clozn.analysis.run_diff` (the pure comparison engine -- all real logic lives there; see that module's
docstring for what is compared and how identity["ext"] is diffed forward-compatibly).

Local-journal-only, zero generation, no model/GPU/network -- both runs are looked up via
`clozn.runs.store.get_run` (the same lookup `clozn/server/routes/diff.py`'s `POST /diff/runs` already
uses), matching `clozn inspect`'s own "local journal first" discipline
(`clozn/cli/commands/explain.py`'s `cmd_inspect`).

`--replay` prints the MODEL-FREE replay planner's proposal (`run_diff.plan_replay`) -- which of the
spec's three candidate swaps (context/template/sampling) are available and how many runs executing all
of them would cost. It never executes anything: that needs a live substrate/GPU
(`clozn.replay.replay.replay()`) and is a separately-scoped, deferred slice (see run_diff.py's own
docstring). Both `--json` and human output are stable enough for automation, per the roadmap's shared
definition of done.

Registered via `CLOZN_AUTOLOAD` (docs/SEAMS.md Seam 1) -- no edit to clozn/cli/main.py.
"""
from __future__ import annotations

import json

from clozn.analysis import run_diff
from clozn.cli import formatting as fmt

CLOZN_AUTOLOAD = True


# ------------------------------------------------------------------------------------------ plain labels

# The spec's own fixed vocabulary (notes/agent_roadmap/10-run-change-explainer.md's UI section): "Use
# plain labels: Model, Instructions, Documents, History, Settings, Tools, Output." clozn.runs.context_
# receipt does not yet carry segment-level typing (that is feature 06's un-shipped segment-ID work), so
# every context.* dimension is honestly labeled "History" rather than pretending to a finer-grained split
# this differ cannot actually make -- flagged once in the rendered output, not silently overclaimed.
def _label_for(dimension: str) -> str:
    if dimension.startswith("identity."):
        return "Model"
    if dimension.startswith("generation."):
        return "Settings"
    if dimension == "output.tool_call_status":
        return "Tools"
    if dimension.startswith("output."):
        return "Output"
    if dimension.startswith("context."):
        return "History"
    return "Other"


def _short(value) -> str:
    if value is None:
        return "-"
    text = str(value)
    if len(text) > 40:
        return text[:18] + "…" + text[-8:]
    return text


def _format_difference(d: dict) -> str:
    label = _label_for(d["dimension"])
    kind = d.get("kind", "?")
    if d["dimension"] == "output.text":
        return f"  {label:<12} {d['dimension']:<38} {kind:<11} (see clozn compare-runs --json for the token diff)"
    if kind == "unavailable":
        return f"  {label:<12} {d['dimension']:<38} {kind:<11} {d.get('note', '')}"
    if kind == "diff_failed":
        return f"  {label:<12} {d['dimension']:<38} {kind:<11} {d.get('note', '')}"
    a, b = _short(d.get("value_a")), _short(d.get("value_b"))
    return f"  {label:<12} {d['dimension']:<38} {kind:<11} {a} -> {b}"


def format_compare_runs(result: dict, *, replay_plan: dict | None = None) -> str:
    """Pure JSON(compare_runs result) -> text render. Never raises: a malformed section degrades to a
    one-line notice instead of losing the rest -- same discipline as `clozn/cli/commands/explain.py`'s
    `format_explain`."""
    result = result if isinstance(result, dict) else {}
    lines = [f"{fmt.BOLD}compare-runs{fmt.RST}  {result.get('run_a', '?')} vs {result.get('run_b', '?')}",
             "-" * 66]

    findings = [f for f in (result.get("findings") or []) if isinstance(f, dict)]
    lines.append(f"{fmt.BOLD}primary findings{fmt.RST}  "
                 f"{fmt.DIM}never stronger than the evidence -- see each finding's status{fmt.RST}")
    if not findings:
        lines.append(f"  {fmt.DIM}no classified findings -- see raw differences below{fmt.RST}")
    else:
        for f in findings:
            lines.append(f"  [{f.get('status', '?')}] {f.get('summary', '')}")
    lines.append("")

    differences = [d for d in (result.get("differences") or []) if isinstance(d, dict)]
    lines.append(f"{fmt.BOLD}differences{fmt.RST}  "
                 f"{fmt.DIM}ranked for presentation only -- rank is not evidence{fmt.RST}")
    if not differences:
        lines.append(f"  {fmt.DIM}none -- these two runs match on every dimension this compared{fmt.RST}")
    else:
        if any(d.get("dimension", "").startswith("context.") for d in differences):
            lines.append(f"  {fmt.DIM}History below covers everything clozn.runs.context_receipt captures "
                         f"today -- it cannot yet separate Instructions/Documents from History.{fmt.RST}")
        for d in differences:
            lines.append(_format_difference(d))

    if result.get("privacy_limited"):
        lines.append("")
        lines.append(f"  {fmt.DIM}note: at least one dimension above is 'unavailable' -- content wasn't "
                     f"captured on one side, so it was compared by hash/count only, never invented{fmt.RST}")

    if replay_plan is not None:
        lines.append("")
        lines.append(f"{fmt.BOLD}replay (planned, not executed){fmt.RST}")
        for c in replay_plan.get("candidates", []):
            mark = "available" if c.get("available") else "skipped"
            lines.append(f"  {c.get('order')}. {c.get('description')}  [{mark}]"
                         + (f" -- {c['note']}" if c.get("note") else ""))
        lines.append(f"  {fmt.DIM}{replay_plan.get('runs_required', 0)} run(s) would be required; "
                     f"{replay_plan.get('note', '')}{fmt.RST}")

    lines.append("-" * 66)
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the CLI

def add_subparser(sub):
    p = sub.add_parser("compare-runs", help="what changed between two recorded runs -- identity, "
                       "context, sampling settings, and output, ranked for presentation and capped to "
                       "the evidence clozn actually has (agent roadmap feature 10)")
    p.add_argument("run_a", help="the earlier/reference run id (clozn_run_id from `clozn trace --list` "
                   "or the Studio Runs list)")
    p.add_argument("run_b", help="the later/candidate run id")
    p.add_argument("--json", action="store_true", help="print the raw clozn.run-diff.v1 document as "
                   "JSON instead of the human table")
    p.add_argument("--replay", action="store_true", help="also print the model-free replay planner's "
                   "proposal (which of context/template/sampling could be swapped and re-run) -- never "
                   "executes anything")
    p.set_defaults(fn=cmd_compare_runs)
    return p


def cmd_compare_runs(args):
    from clozn.cli import main as ctx
    import clozn.runs.store as runlog

    run_a = runlog.get_run(args.run_a)
    run_b = runlog.get_run(args.run_b)
    missing = [rid for rid, run in ((args.run_a, run_a), (args.run_b, run_b)) if run is None]
    if missing:
        raise ctx.CloznError("run(s) not found in the local journal: " + ", ".join(missing) +
                             " (see ids in `clozn trace --list` or the Studio Runs list)")

    result = run_diff.compare_runs(run_a, run_b)
    if not result.get("ok"):
        raise ctx.CloznError(result.get("error") or "comparison failed for an unknown reason")

    replay_plan = run_diff.plan_replay(run_a, run_b, result) if args.replay else None

    if args.json:
        out = dict(result)
        if replay_plan is not None:
            out["replay_plan"] = replay_plan
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(format_compare_runs(result, replay_plan=replay_plan))
    return 0
