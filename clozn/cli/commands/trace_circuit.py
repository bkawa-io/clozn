"""commands.trace_circuit -- `clozn causal-trace`: intervention-validated CAUSAL tracing
(S0-S4; notes/CIRCUIT_TRACER_DESIGN.md). Drives clozn.analysis.tracer against a live clozn-server:
screen -> solo ablations + controls -> joint -> path patching -> generation arms, then a receipt
JSON + a terminal rendering with the honesty accounting front and center (verdict, noise floor,
interaction gap, per-node control ratio, dead candidates).

NAMING (deliberate, see notes/CIRCUIT_TRACER_DESIGN.md section 5e). This is **causal tracing** --
ROME-style (layer, position) activation patching -- NOT circuit discovery in the
features-and-components sense. Its nodes are LOCATIONS in the residual stream, not features with
functional roles, and a trace does not generalize beyond its prompt. The measured legibility
(~24% median) and the SAE study in section 5e are what forced the honest name: at these sites no
individual dictionary feature is load-bearing, so there is no sparse circuit to report. The
original `trace-circuit` spelling was retired once the rename to `causal-trace` shipped (no
compat alias -- nothing else in the product called it by the old name).

Registration in clozn/cli/main.py mirrors quant-check: import `cmd_trace_circuit,
add_subparser as _add_causal_trace` alongside the other commands.* imports, and call
`_add_causal_trace(sub)` in build_parser() before `return p`.
"""
from __future__ import annotations

import json
import sys


def add_subparser(sub):
    """Register `clozn causal-trace` (own function so wiring is testable without dispatching;
    mirrors commands.quant_check.add_subparser)."""
    pt = _build("causal-trace", sub,
                help="causal trace: which (layer, position) sites causally support continuation "
                     "token N? measured by ablation, not attention (needs a running clozn-server "
                     "with a J-lens sidecar)")
    return pt


def _build(name, sub, help):
    kw = {"help": help} if help else {}
    pt = sub.add_parser(name, **kw)
    pt.add_argument("--prompt", default=None, help="the prompt text (teacher-forced context)")
    pt.add_argument("--continuation", default=None,
                    help="the continuation text whose token --pos is being traced")
    pt.add_argument("--from-run", default=None, metavar="ID|last",
                    help="trace a stored run instead of --prompt/--continuation: 'last' (latest "
                         "non-derived journal run) or a run id. Uses the run's recorded final_prompt "
                         "+ answer, the same journal selection as `provenance last`/`context last`.")
    pt.add_argument("--pos", type=int, default=0,
                    help="0-based index of the target token within the continuation (default 0)")
    pt.add_argument("--contrast", default=None, metavar="FOIL|auto",
                    help="score CONTRASTIVELY against a foil token (answer-selectivity, not "
                         "answer-category): a token/word, or 'auto' = the baseline runner-up. See the "
                         "screen-null finding (scripts/tracer/screen_null.py) for why this matters.")
    pt.add_argument("--concepts", default="",
                    help="comma-separated extra concept words to screen with (beyond the target token)")
    pt.add_argument("--engine", default="http://127.0.0.1:8080", help="clozn-server base URL")
    pt.add_argument("--jlens-dir", default=None,
                    help="J-lens sidecar dir (default: CLOZN_JLENS_DIR or ~/.clozn/jlens)")
    pt.add_argument("--screen-mode", default="auto", choices=("auto", "jlens", "ablate"),
                    help="S0 nomination: 'auto' (jlens sidecar if it qualifies, else ablate), "
                         "'jlens' (require the sidecar), 'ablate' (any-GGUF mean-ablation grid, no "
                         "sidecar -- use this on the --no-flash-attn engine). Default auto.")
    pt.add_argument("--candidates", type=int, default=24, help="max screened candidate sites")
    pt.add_argument("--seed", type=int, default=0, help="rng seed for the control arms")
    pt.add_argument("--out", default=None, help="write the receipt JSON here (default: print-only)")
    pt.set_defaults(fn=cmd_trace_circuit)
    return pt


def _resolve_from_run(ref):
    """Populate (prompt, continuation) from the run journal -- 'last' (latest non-derived run) or a
    run id -- mirroring provenance/context `last`. Returns (prompt, continuation) or raises."""
    import clozn.runs.store as runlog
    if ref == "last":
        rows = runlog.list_runs(limit=1, include_replays=False)
        if not rows:
            raise SystemExit("no runs in the journal to trace")
        run = runlog.get_run(rows[0]["id"])
    else:
        run = runlog.get_run(ref)
        if run is None:
            raise SystemExit(f"no run with id {ref!r} in the journal")
    prompt = run.get("final_prompt")
    if not prompt:
        raise SystemExit("that run has no recorded final_prompt (the exact rendered prompt) to trace")
    cont = run.get("response") or run.get("answer") or ""
    if not cont:
        raise SystemExit("that run has no recorded answer to trace")
    return prompt, cont


def cmd_trace_circuit(args):
    from clozn.analysis import tracer

    prompt, continuation = args.prompt, args.continuation
    if args.from_run:
        prompt, continuation = _resolve_from_run(args.from_run)
    elif prompt is None or continuation is None:
        print("provide --prompt and --continuation, or --from-run ID|last", file=sys.stderr)
        return 1

    budget = tracer.TraceBudget(max_candidates=args.candidates,
                                extra_concepts=[w.strip() for w in args.concepts.split(",") if w.strip()])
    r = tracer.trace(prompt, continuation, args.pos,
                     engine_url=args.engine, jlens_dir=args.jlens_dir,
                     budget=budget, seed=args.seed, contrast=args.contrast,
                     screen_mode=args.screen_mode)
    if not r.get("ok"):
        print(f"trace blocked: {r.get('blocked')}", file=sys.stderr)
        return 1

    t = r["target"]
    ctl = r["controls"]
    acct = r["accounting"]
    margin = t["margin"]
    print(f"target: {t['piece']!r} (id {t['id']}) at continuation pos {t['pos']} "
          f"| baseline logprob {t['baseline_logprob']:.3f}"
          + (f" | margin {margin:+.3f}" if margin is not None else " | margin n/a"))
    cfg = r.get("config", {})
    if cfg.get("scoring") == "contrastive":
        foil = cfg.get("contrast") or {}
        print(f"scoring: CONTRASTIVE vs foil {foil.get('piece')!r} (id {foil.get('id')}) -- deltas are "
              f"the change in the y-vs-foil logit gap; a node counts only if it SUPPORTS y (delta>0).")
    print(f"verdict: {ctl['verdict']}   (noise floor {ctl['noise_floor']:.4f} = 3x median |control|; "
          f"control max {ctl['max_abs']:.4f})")
    if ctl["verdict"] == "FAILED_CONTROLS":
        print("  !! random interventions moved the target as much as the real ones -- DO NOT trust "
              "this trace (the design's STOP check).", file=sys.stderr)
    print(f"screened {acct['screened_sites']} sites -> {acct['candidates']} candidates -> "
          f"{acct['survivors']} survivors")
    if r["nodes"]:
        print(f"\n  {'layer':>5} {'pos':>4}  {'d_full':>8} {'d_dir':>8}  {'legible':>7}  "
              f"{'vs ctl':>7}  {'flip?':>5}  name")
        for n in sorted(r["nodes"], key=lambda x: -abs(x["delta_full"])):
            leg = f"{n['legibility']:.0%}" if n["legibility"] is not None else "n/a"
            flip = "YES" if n["margin_flip_predicted"] else "no"
            ratio = f"{n['control_ratio']:.1f}x" if n.get("control_ratio") is not None else "n/a"
            tier = "" if n.get("strength") == "strong" else f"  [{n.get('strength')}]"
            print(f"  {n['layer']:>5} {n['pos']:>4}  {n['delta_full']:>+8.4f} {n['delta_dir']:>+8.4f}"
                  f"  {leg:>7}  {ratio:>7}  {flip:>5}  {n['name']}{tier}")
        print("  (vs ctl = |delta| / strongest control arm; strong >= 3x, weak 1-3x, marginal <= 1x)")
        if acct["delta_total"] is not None:
            print(f"\n  joint (all survivors ablated at once): delta_total {acct['delta_total']:+.4f}"
                  f" | sum of solos {acct['sum_solo']:+.4f}"
                  f" | interaction gap {acct['interaction_gap']:+.4f}")
        if r.get("edges"):
            print("\n  edges (path patching: A's effect routed through B alone):")
            any_cross = False
            for e in r["edges"]:
                frac = f"{e['routed_fraction']:.0%}" if e["routed_fraction"] is not None else "n/a"
                shuf = f"{e['delta_shuffled']:+.4f}" if e["delta_shuffled"] is not None else "n/a"
                tag = "CLAIMED" if e["claimed"] else "not claimed"
                kind = "same-col" if e.get("same_column") else "CROSS-POS"
                if not e.get("same_column"):
                    any_cross = True
                print(f"    L{e['from'][0]}@{e['from'][1]} -> L{e['to'][0]}@{e['to'][1]} [{kind}]: "
                      f"delta {e['delta_edge']:+.4f} (routed {frac}) | shuffled-ctl {shuf} | {tag}")
            if any_cross:
                print("    NOTE: a CROSS-POS routed fraction is a LOWER BOUND, not a measurement --")
                print("    single-site patching cannot hold a cross-position path (the source keeps")
                print("    re-supplying it downstream, and the last layer is unpatchable). Same-col")
                print("    fractions are close to structural. See CIRCUIT_TRACER_DESIGN.md 5f.")
        sc = r["prediction_scorecard"]
        gen = sc.get("generation_tier") or {}
        if gen.get("ran"):
            if gen.get("baseline_greedy") is False:
                print("\n  prediction scorecard: baseline reply is not greedy-reproducible -- "
                      "behavioral tier inapplicable (reported, not fudged)")
            else:
                print(f"\n  prediction scorecard (patch + greedy decode, observed vs predicted): "
                      f"{sc['correct_predictions']} correct, {sc['wrong_predictions']} wrong, "
                      f"{sc['diverged_early']} diverged early | flips predicted "
                      f"{sc['predicted_flips']}, observed {sc['observed_flips']}")
    else:
        print("\n  no nodes beat the noise floor -- either a distributed circuit (many small "
              "contributions) or nothing screenable at these layers. That is the finding.")
    dead = [c for c in r["all_candidates"] if not c["survived"]]
    print(f"\n  dead candidates (screen over-nomination, published): {len(dead)}")
    if r["config"]["concept_notes"]:
        for cn in r["config"]["concept_notes"]:
            print(f"  note: concept {cn['concept']!r} skipped: {cn['skipped']}")

    if args.out:
        path = tracer.save_receipt(r, args.out)
        print(f"\nreceipt -> {path}")
    return 0
