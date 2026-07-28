"""commands.compare_runs -- `clozn compare-runs RUN_A RUN_B [--json] [--replay]` (agent roadmap feature
10, "What changed"): the consumer-facing "two runs differed -- what changed?" explainer, over
`clozn.analysis.run_diff` (the pure comparison engine -- all real logic lives there; see that module's
docstring for what is compared and how identity["ext"] is diffed forward-compatibly).

Local-journal-only, zero generation, no model/GPU/network -- both runs are looked up via
`clozn.runs.store.get_run` (the same lookup `clozn/server/routes/diff.py`'s `POST /diff/runs` already
uses), matching `clozn inspect`'s own "local journal first" discipline
(`clozn/cli/commands/explain.py`'s `cmd_inspect`).

`--replay` retains the cheap planner-only view. `--test` executes the selected two-arm controlled swaps
through a running gateway; `--plan`/`--dry-run` emits the same versioned test artifact without starting a
model run. Both `--json` and human output expose the selected comparison, budget, stop state and child run
ids.

Registered via `CLOZN_AUTOLOAD` (docs/SEAMS.md Seam 1) -- no edit to clozn/cli/main.py.
"""
from __future__ import annotations

import json
import math
import os
import urllib.error
import urllib.request

from clozn.analysis import run_diff
from clozn.cli import formatting as fmt

CLOZN_AUTOLOAD = True


# ------------------------------------------------------------------------------------------ plain labels

# The spec's own fixed vocabulary (notes/agent_roadmap/10-run-change-explainer.md's UI section): "Use
# plain labels: Model, Instructions, Documents, History, Settings, Tools, Output." clozn.runs.context_
# receipt v1 carries stable message-segment IDs and hashes but no document/attachment type, so context.*
# remains "History" instead of inventing an Instructions/Documents distinction the captured source_type
# does not support.
def _label_for(dimension: str) -> str:
    if dimension.startswith("identity.ext.adapter"):
        return "Adapter"
    if dimension.startswith("identity.ext.engine_artifact") or dimension == "identity.engine_build":
        return "Engine"
    if dimension.startswith("identity.ext.machine"):
        return "Machine"
    if dimension.startswith("identity.ext.behavior") or dimension.startswith("identity.ext.intervention"):
        return "Behavior"
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


def format_compare_runs(result: dict, *, replay_plan: dict | None = None,
                        controlled_tests: dict | None = None) -> str:
    """Pure JSON(compare_runs result) -> text render. Never raises: a malformed section degrades to a
    one-line notice instead of losing the rest -- same discipline as `clozn/cli/commands/explain.py`'s
    `format_explain`."""
    result = result if isinstance(result, dict) else {}
    lines = [f"{fmt.BOLD}compare-runs{fmt.RST}  {result.get('run_a', '?')} vs {result.get('run_b', '?')}",
             "-" * 66]
    selection = result.get("comparison_selection") or {}
    lines.append(
        f"{fmt.BOLD}selection{fmt.RST}  {selection.get('mode', 'explicit')}: "
        f"{selection.get('reference_run_id', result.get('run_a', '?'))} -> "
        f"{selection.get('candidate_run_id', result.get('run_b', '?'))}"
    )
    if selection.get("reason"):
        lines.append(f"  {fmt.DIM}{selection['reason']}{fmt.RST}")
    lines.append("")

    axes = result.get("summary_axes") or {}
    lines.append(f"{fmt.BOLD}change summary{fmt.RST}")
    for key in ("model", "adapter", "template", "context", "sampling", "engine", "tool_parse", "output"):
        value = axes.get(key) or {"status": "unavailable"}
        lines.append(f"  {key.replace('_', ' '):<12} {value.get('status', 'unavailable')}")
    lines.append("")

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

    if controlled_tests is not None:
        lines.append("")
        lines.append(
            f"{fmt.BOLD}controlled tests{fmt.RST}  status={controlled_tests.get('status')} "
            f"runs={((controlled_tests.get('budget') or {}).get('runs_used', 0))}/"
            f"{((controlled_tests.get('budget') or {}).get('max_runs', 0))}"
        )
        for test in controlled_tests.get("tests") or []:
            lines.append(
                f"  [{test.get('status', '?')}] {test.get('kind', '?')} "
                f"runs={test.get('runs_used', 0)} -- {test.get('reason', '')}"
            )
            for evidence in test.get("evidence") or []:
                lines.append(f"    {evidence.get('arm')}: {evidence.get('run_id')}")
        summary = controlled_tests.get("summary") or {}
        if summary.get("entangled"):
            lines.append("  entangled: multiple swaps recovered the reference; no single cause is named")

    lines.append(
        f"reproduce: clozn compare-runs {result.get('run_a', '?')} {result.get('run_b', '?')} --json"
    )
    lines.append("-" * 66)
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the CLI

def add_subparser(sub):
    p = sub.add_parser("compare-runs", help="what changed between two recorded runs -- identity, "
                       "context, sampling settings, and output, ranked for presentation and capped to "
                       "the evidence clozn actually has (agent roadmap feature 10)")
    p.add_argument("run_a", help="the earlier/reference run id, or the candidate id when using --against",
                   nargs="?")
    p.add_argument("run_b", help="the later/candidate run id for an explicit pair", nargs="?")
    p.add_argument(
        "--against", choices=("previous_compatible", "same_session", "same_client", "same_task"),
        help="with one positional candidate id, automatically select its earlier reference")
    p.add_argument("--reference", help="with one positional candidate id, use this pinned reference run")
    p.add_argument("--include-child-runs", action="store_true",
                   help="allow replay/branch/fork children in automatic selection (excluded by default)")
    p.add_argument("--json", action="store_true", help="print the raw clozn.run-diff.v1 document as "
                   "JSON instead of the human table")
    p.add_argument("--replay", action="store_true", help="also print the model-free replay planner's "
                   "proposal (which of context/template/sampling could be swapped and re-run) -- never "
                   "executes anything")
    p.add_argument("--test", default=None, metavar="SWAP[,SWAP...]",
                   help="execute context/template/sampling controlled swaps through a running gateway")
    p.add_argument("--plan", action="store_true", help="plan --test swaps and cost without model runs")
    p.add_argument("--dry-run", action="store_true", help="alias for --plan")
    p.add_argument("--max-runs", type=int, default=4, help="hard controlled-arm budget (default 4)")
    p.add_argument("--max-seconds", type=float, default=120.0,
                   help="controlled-test wall deadline (default 120)")
    p.add_argument("--match", choices=("exact_output", "tool_parse", "finish_reason", "token_budget"),
                   default="exact_output", help="exact outcome criterion; no semantic similarity")
    p.add_argument("--port", type=int, default=0,
                   help="running gateway port for live --test (default CLOZN_PORT or 8080)")
    p.add_argument("--test-out", default=None,
                   help="write the separate clozn.run-change-test.v1 artifact to this path")
    p.add_argument("--force", action="store_true", help="overwrite --test-out if it exists")
    p.set_defaults(fn=cmd_compare_runs)
    return p


def _resolve_pair(args, runlog, ctx):
    positional_a, positional_b = getattr(args, "run_a", None), getattr(args, "run_b", None)
    against, pinned = getattr(args, "against", None), getattr(args, "reference", None)
    if positional_a and positional_b:
        if against or pinned:
            raise ctx.CloznError("--against/--reference require a single positional candidate run id")
        run_a, run_b = runlog.get_run(positional_a), runlog.get_run(positional_b)
        missing = [rid for rid, run in ((positional_a, run_a), (positional_b, run_b)) if run is None]
        if missing:
            raise ctx.CloznError("run(s) not found in the local journal: " + ", ".join(missing))
        return run_a, run_b, None
    if positional_a and not positional_b and (against or pinned):
        run_b = runlog.get_run(positional_a)
        if run_b is None:
            raise ctx.CloznError(f"candidate run not found in the local journal: {positional_a}")
        selected = run_diff.select_reference_run(
            run_b, runlog.iter_runs(), mode="pinned" if pinned else against,
            reference_run_id=pinned, include_child_runs=bool(getattr(args, "include_child_runs", False)),
        )
        if not selected.get("ok"):
            raise ctx.CloznError(selected.get("error") or "comparison selection failed")
        return selected["run"], run_b, selected["selection"]
    raise ctx.CloznError(
        "pass two run ids, or one candidate run id with --against/--reference"
    )


def _parse_tests(value, *, plan_only: bool) -> list[str] | None:
    if value:
        return [item.strip() for item in value.split(",") if item.strip()]
    return list(("context", "template", "sampling")) if plan_only else None


def _request_tests(run_a: str, run_b: str, *, tests, max_runs, max_seconds,
                   match_criterion, port) -> dict:
    from clozn.cli import main as ctx
    body = json.dumps({
        "a": run_a, "b": run_b, "tests": tests,
        "max_runs": max_runs, "max_seconds": max_seconds,
        "match_criterion": match_criterion,
    }).encode("utf-8")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/runs/compare/test", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=max(5.0, float(max_seconds) + 5.0)) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        try:
            detail = json.loads(detail).get("error") or detail
        except Exception:
            pass
        raise ctx.CloznError(f"controlled comparison failed: HTTP {exc.code}: {detail}") from None
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise ctx.CloznError(
            f"couldn't reach the Clozn gateway on port {port} for controlled tests: {exc}"
        ) from None


def _write_test_artifact(path: str, document: dict, *, force: bool):
    from pathlib import Path
    from clozn.cli import main as ctx
    from clozn._io import atomic_write_json
    target = Path(path).expanduser().resolve()
    if target.exists() and not force:
        raise ctx.CloznError(f"refusing to overwrite {target}; pass --force")
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(str(target), document, indent=2, ensure_ascii=False)


def cmd_compare_runs(args):
    from clozn.cli import main as ctx
    import clozn.runs.store as runlog

    budget_runs = getattr(args, "max_runs", 4)
    budget_seconds = getattr(args, "max_seconds", 120.0)
    if (isinstance(budget_runs, bool) or budget_runs < 0):
        raise ctx.CloznError("--max-runs must be a non-negative integer")
    if budget_seconds < 0 or not math.isfinite(float(budget_seconds)):
        raise ctx.CloznError("--max-seconds must be a finite non-negative number")

    run_a, run_b, selection = _resolve_pair(args, runlog, ctx)

    result = run_diff.compare_runs(run_a, run_b, selection=selection)
    if not result.get("ok"):
        raise ctx.CloznError(result.get("error") or "comparison failed for an unknown reason")

    replay_plan = run_diff.plan_replay(run_a, run_b, result) if args.replay else None
    plan_only = bool(getattr(args, "plan", False) or getattr(args, "dry_run", False))
    tests = _parse_tests(getattr(args, "test", None), plan_only=plan_only)
    controlled_tests = None
    if tests is not None:
        from clozn.replay import controlled
        try:
            if plan_only:
                controlled_tests = controlled.plan_change_tests(
                    run_a, run_b, tests=tests, max_runs=budget_runs,
                    max_seconds=budget_seconds, match_criterion=args.match,
                )
            else:
                port = int(args.port or os.environ.get("CLOZN_PORT", "8080"))
                controlled_tests = _request_tests(
                    run_a["id"], run_b["id"], tests=tests, max_runs=budget_runs,
                    max_seconds=budget_seconds, match_criterion=args.match, port=port,
                )
        except ValueError as exc:
            raise ctx.CloznError(str(exc)) from None
    if getattr(args, "test_out", None):
        if controlled_tests is None:
            raise ctx.CloznError("--test-out requires --test, --plan, or --dry-run")
        _write_test_artifact(args.test_out, controlled_tests, force=bool(args.force))

    if args.json:
        out = dict(result)
        if replay_plan is not None:
            out["replay_plan"] = replay_plan
        if controlled_tests is not None:
            out["controlled_tests"] = controlled_tests
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    else:
        print(format_compare_runs(
            result, replay_plan=replay_plan, controlled_tests=controlled_tests))
    return 0
