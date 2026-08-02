"""`clozn provenance` -- did an answer come from the CONTEXT or from the model's own weights?

CLI front door for clozn.analysis.provenance.trace_provenance (attention-edge knockout; see that
module's docstring for the measurement itself -- this file only parses args, resolves 'last' against
the run journal the same way `diagnose last`/`context last` do, and renders the receipt).

Requires a clozn-server started with --no-flash-attn. When it isn't, trace_provenance never raises --
it returns {"ok": False, "blocked": "..."} -- and this command prints that message verbatim (a clean
typed refusal) and exits 1, never a crash.
"""
from __future__ import annotations

import json

from clozn.cli import main as ctx
from clozn.analysis.provenance import DEFAULT_ENGINE, ProvenanceBudget, SCOPE_NOTE, verdict_label


def add_subparser(subparsers) -> None:
    parser = subparsers.add_parser(
        "provenance",
        help="did the answer come from the context or the model's weights? attention-knockout "
             "receipt (needs a clozn-server started with --no-flash-attn)")
    parser.add_argument(
        "prompt", help="prompt text to trace, or 'last' to use the latest journal run's own "
                       "rendered prompt + recorded answer")
    parser.add_argument(
        "--continuation", default=None,
        help="the answer to score (default: 'last' run's recorded answer; otherwise the model's own "
             "greedy completion)")
    parser.add_argument(
        "--focus", nargs=2, type=int, metavar=("START", "END"), default=None,
        help="restrict the question to one span of prompt TOKEN positions [START, END) -- 'did the "
             "answer use THIS region?' (the RAG document-level question; experimental, see the focus "
             "caveats in the JSON 'focus_trim'/'focus_null' fields)")
    parser.add_argument("--engine", default=DEFAULT_ENGINE, help="clozn-server base URL")
    parser.add_argument("--seed", type=int, default=0, help="rng seed for the matched-random-control arms")
    parser.add_argument("--json", action="store_true", help="print the raw receipt JSON")
    parser.set_defaults(fn=cmd_provenance)


def _resolve_last(continuation):
    """Latest organic (non-derived) run's exact rendered prompt + recorded answer, mirroring
    `context last`'s selection (runlog.list_runs(limit=1, include_replays=False))."""
    import clozn.runs.store as runlog

    rows = runlog.list_runs(limit=1, include_replays=False)
    if not rows:
        raise ctx.CloznError("no recorded run found")
    run = runlog.get_run(rows[0]["id"])
    if not run:
        raise ctx.CloznError("the latest run could not be read")
    prompt = run.get("final_prompt")
    if not isinstance(prompt, str) or not prompt:
        raise ctx.CloznError(
            "the latest run has no recorded final_prompt (the exact rendered prompt) to trace")
    if continuation is None:
        continuation = run.get("response")
        if not isinstance(continuation, str) or not continuation:
            continuation = None   # honestly fall through to trace_provenance's own greedy default
    return prompt, continuation


def format_provenance(receipt: dict) -> str:
    """Render a provenance receipt: the scope note once, then the verdict, dependence, best control
    ratio, carrying span tokens, and (when present) the focus_null p-value. Never claims more than the
    module's own verdict strings -- CONTEXT_CARRIED/MIXED/PARAMETRIC/INCONCLUSIVE, verbatim."""
    lines = [SCOPE_NOTE, ""]
    if not receipt.get("ok"):
        lines.append(f"provenance blocked: {receipt.get('blocked', 'unknown reason')}")
        return "\n".join(lines)

    lines.append(f"answer: {receipt.get('answer')!r}")
    if receipt.get("focus") is not None:
        lines.append(f"focus: prompt token span {tuple(receipt['focus'])}")

    dependence = receipt.get("dependence")
    ratio = receipt.get("best_control_ratio")
    dep_s = f"{dependence:.2f}" if isinstance(dependence, (int, float)) else "?"
    ratio_s = f"{ratio:.1f}x" if isinstance(ratio, (int, float)) else "n/a"
    raw_verdict = receipt.get("verdict", "?")
    lines.append(f"verdict: {raw_verdict} ({verdict_label(raw_verdict)})   "
                 f"(dependence {dep_s}, best control ratio {ratio_s})")

    span_tokens = receipt.get("span_tokens")
    if span_tokens:
        lines.append("carrying span: " + " ".join(repr(t) for t in span_tokens))
    else:
        note = receipt.get("note")
        lines.append("carrying span: (none)" + (f" -- {note}" if note else ""))

    focus_null = receipt.get("focus_null")
    if focus_null:
        lines.append(f"focus_null p-value: {focus_null.get('p_value')} "
                     f"(n_draws {focus_null.get('n_draws')}; evidence, not yet folded into the verdict)")
    return "\n".join(lines)


def cmd_provenance(args) -> int:
    from clozn.analysis import provenance as prov

    prompt, continuation = args.prompt, args.continuation
    if args.prompt == "last":
        prompt, continuation = _resolve_last(continuation)

    focus = tuple(args.focus) if args.focus else None
    budget = ProvenanceBudget()
    receipt = prov.trace_provenance(
        prompt, continuation, engine_url=args.engine, budget=budget, seed=args.seed, focus=focus)

    if args.json:
        print(json.dumps({**receipt, "scope": SCOPE_NOTE}, indent=2, ensure_ascii=False))
    else:
        print(format_provenance(receipt))
    return 0 if receipt.get("ok") else 1
