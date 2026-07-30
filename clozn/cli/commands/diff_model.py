"""commands.diff_model -- `clozn diff-model <reference> <candidate>` (Phase-1 §4.1 wedge feature): "did
the fine-tune change what the model says, or is it a silent no-op?" as one command. Generalizes
`clozn quant-check`'s machinery (clozn/cli/commands/quant_check.py, clozn/receipts/quant_receipts.py --
both reused, NOT copied) from "two quants of one model" to "any two GGUFs a person wants to compare
per-token": base vs fine-tune, fine-tune vs merge, checkpoint N vs checkpoint N+1.

THE TRAP this command exists to avoid: `quant_receipts.diff_quant_scores`' per-position id check verifies
that BOTH engines RETURNED the forced continuation ids it asked for -- it does NOT verify the two
engines' TOKENIZERS agree on what those ids MEAN. Two quants of the same model share a tokenizer by
construction (same GGUF vocab, just different weight precision), so quant-check never had to think about
this. A base-vs-fine-tune pair usually shares a tokenizer too -- but a MERGE may not (a frankenmerge can
staple together checkpoints from different tokenizer vintages), and nothing about the wire shapes here
would catch it: if id 5 means "the" in A's vocab and "cat" in B's vocab, diff_quant_scores would happily
report "id 5 preserved at every position" while the two models said completely different things. Per-token
teacher-forced diffing across mismatched tokenizers isn't wrong, it's MEANINGLESS -- there is no shared
unit to diff. So diff-model runs a MANDATORY tokenizer preflight (`check_tokenizer_compat`) before any
diffing, and REFUSES outright (a typed `ctx.CloznError`, never a silently-degraded or approximate diff) on
incompatibility. This is the repo's Gate-0 no-silent-degradation discipline applied to a new failure mode.

Two more honesty layers on top of quant-check's (both riding through UNCHANGED: the argmax-flip vs
dependence-shift distinction and the topk caveat -- see quant_receipts.py's module docstring):

  * TEMPLATE POLICY -- base models and fine-tunes often ship different chat templates. `diff-model`
    compares `apply_template` output on a canonical 2-message conversation under both engines. If they
    match, nothing changes. If they differ, the DEFAULT policy renders every prompt under the REFERENCE
    engine's template and scores BOTH arms on that byte-identical string -- so the diff isolates WEIGHTS,
    at the honestly-labeled cost of evaluating the candidate slightly off its own deployed format.
    `--own-templates` flips this: each model renders with its own template, which measures the
    candidate's actual DEPLOYED behavior (weights AND template both in play) instead of an isolated
    weights diff. Implemented via one plumbing change to quant_check.py's `_EngineScoreSub`: an optional
    `template_engine` param (default None -> self.engine) used for `apply_template` while `self.engine`
    still does the actual `/score` call -- so "B's weights, A's template" is just
    `_EngineScoreSub(eng_b, template_engine=eng_a)`.
  * VERDICT LAYER -- the wedge feature. After `aggregate_receipts`, classify the whole ladder into one of
    three honestly-labeled buckets (see `classify_verdict`): NO_DETECTABLE_DIFF (flags a likely silent
    no-op fine-tune/LoRA), CHANGED (with per-category flip counts), or INSUFFICIENT_SAMPLE (too few
    verified tokens to say either). Explicitly labeled a heuristic SCREEN against two fixed thresholds and
    a token-count floor -- never "proof" the candidate is or isn't different.

DIRECTIONS: the default ladder is REFERENCE-ANCHORED (generate under the reference, teacher-force under
both) -- "what would the candidate change about the reference's behavior" (the forgetting/no-op view: the
question a fine-tune author usually cares about first). `--both` additionally runs the reverse,
CANDIDATE-ANCHORED ladder (generate under the candidate, teacher-force under both) -- "what would the
reference do differently on the candidate's own answers" (the target-gain view: did the fine-tune actually
learn the thing it was trained to do).

A measured caveat on choosing the anchor (live smoke, Qwen2.5-0.5B base vs instruct, 2026-07-20): an
anchored ladder is only as informative as the anchor's own generations. A BASE model driven through a
chat template often emits degenerate text (base GGUFs ship the chat_template metadata without having
been trained on it), and a diff over degenerate continuations is mechanically valid but behaviorally
noisy. For base-vs-instruct pairs, the CANDIDATE-anchored direction is the informative one (measured:
its flips read as exactly what instruct-tuning changed -- refusal openers, markdown fences, list
formatting); reference-anchored earns its keep on tuned-vs-tuned / checkpoint-vs-checkpoint pairs where
the reference generates competently. When in doubt, run --both and read the direction whose anchor
produced sane text. Both directions call the SAME gather/build/aggregate pipeline
(`run_direction`, wrapping quant_check's `gather_fresh_runs`/`gather_from_log_runs`/`build_receipts`/
`aggregate_receipts` by import, unmodified) with the generation role swapped.

This module owns ONLY the CLI shell + the three pieces of NEW logic above (tokenizer preflight, template
policy, verdict classification) -- all rendering below the ladder line reuses `quant_check.format_ladder`
verbatim (relabeled headline only), and every caveat string `quant_receipts.py` attaches to a receipt rides
through unchanged.

SLICE 3.1: the tokenizer preflight (`check_tokenizer_compat`/`_tokenizer_refusal_message`) and the
template-policy probe (`check_template_match`) now live in `clozn.analysis.pair_compatibility` -- the
SHARED, versioned model-pair compatibility contract several features consume (diff-model, Experiments,
mechanistic diff, Studio Compare, causal bisect) instead of each reimplementing this preflight. This
module imports them under their original names (see the import block below) so its own observable
behavior -- the exact refusal wording, the exact caveat text, `--own-templates`' exact semantics -- is
byte-for-byte unchanged; `tests/test_diff_model.py` is the regression net that proves it. What stays HERE
(diff-model-specific, not shared): the verdict classification (`classify_verdict`), the ladder
orchestration (`run_direction`/`run_diff_model`), and all rendering.

Model-free / unit-tested (no engine, no GPU -- tests/test_diff_model.py): `check_tokenizer_compat` and
`check_template_match` against fake engines exposing only `.score`/`.apply_template`; `classify_verdict`
against FIXTURE receipts built with `quant_receipts.diff_quant_scores` (mirrors test_quant_check.py's own
discipline -- no new fixture shape invented); `run_direction`/`run_diff_model` against a fake engine
exposing `.apply_template`/`.score`/`.complete` (mirrors quant_check's own `FakeEngine`); `add_subparser`'s
argparse wiring.

DEFERRED (by design, same as quant_check.py): `cmd_diff_model`'s real two-engine boot -- needs a free GPU
and two engine processes, so it is never invoked by this module's own tests. Once a GPU is free:

    clozn diff-model <base.gguf> <finetune.gguf> --runs 8
    clozn diff-model <base.gguf> <merge.gguf> --from-log --runs 20 --both

`add_subparser` builds its OWN `diff-model` subparser and calls `.set_defaults(fn=cmd_diff_model)` itself;
registered in clozn/cli/main.py alongside the other subcommands (registration only -- no other change to
main.py's dispatch).
"""
from __future__ import annotations

import json
import os
import sys

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import resolve_model, _flags_for
from clozn.cli.engine_process import _free_port, spawn_engine
import clozn.cli.commands.quant_check as qc

# ------------------------------------------------------------------------------------ tokenizer preflight
# Delegated to clozn.analysis.pair_compatibility (slice 3.1's shared model-pair compatibility contract --
# see this module's docstring). Imported under the original names so every existing caller/test in this
# file (`check_tokenizer_compat`, `_TOKENIZER_PROBES`, `_tokenizer_refusal_message`) keeps working
# unmodified; `pair_compatibility` is the single implementation now.
from clozn.analysis.pair_compatibility import (           # noqa: E402
    TOKENIZER_PROBES as _TOKENIZER_PROBES,
    check_tokenizer_compat,
    tokenizer_refusal_message as _tokenizer_refusal_message,
    CANONICAL_TEMPLATE_MESSAGES as _CANONICAL_TEMPLATE_MESSAGES,
    TEMPLATE_DIFFER_REFERENCE_CAVEAT as _TEMPLATE_DIFFER_REFERENCE_CAVEAT,
    TEMPLATE_OWN_CAVEAT as _TEMPLATE_OWN_CAVEAT,
    check_template_match,
)

# -------------------------------------------------------------------------------------------- the ladder

def run_direction(sub_gen, sub_a, sub_b, *, label_a: str, label_b: str, args) -> tuple[list, dict]:
    """One direction of the ladder: gather runs (fresh greedy generations under `sub_gen`, or the N most
    recent run-journal entries with `args.from_log` -- that path is generation-agnostic, so `sub_gen` is
    unused there, meaning `--both --from-log` re-diffs the SAME historical runs twice; a documented no-op
    in that combination, not a bug), diff each run under `sub_a`/`sub_b` via `quant_check.build_receipts`
    (unmodified), and aggregate. `sub_gen` is what makes the reference-anchored default and the `--both`
    reverse ladder different calls to this SAME function: the default passes `sub_a` (generate under the
    reference), `--both`'s reverse ladder passes `sub_b` (generate under the candidate) -- `sub_a`/`sub_b`
    themselves (who gets teacher-forced) never change. Model-free wiring; delegates entirely to
    quant_check's already-tested gather/build/aggregate functions."""
    if getattr(args, "from_log", False):
        runs = qc.gather_from_log_runs(args.runs)
    else:
        runs = qc.gather_fresh_runs(sub_gen, args.runs, max_tokens=args.max_tokens, topk=args.topk)
    receipts = qc.build_receipts(runs, sub_a, sub_b, label_a=label_a, label_b=label_b, topk=args.topk)
    agg = qc.aggregate_receipts(receipts, label_a=label_a, label_b=label_b)
    return receipts, agg


# -------------------------------------------------------------------------------------------- verdict layer

_VERDICT_MIN_TOKENS = 100
_VERDICT_MAX_MEAN_ABS_DELTA_NATS = 0.02

_NO_DETECTABLE_DIFF_MESSAGE = (
    "the candidate is behaviorally indistinguishable from the reference on this sample -- if the "
    "candidate is supposed to be a fine-tune of the reference, the adapter may not have been applied "
    "(silent no-op)."
)


def classify_verdict(receipts: list, agg: dict) -> dict:
    """The wedge feature's honesty layer: classify one ladder's receipts into one of three buckets. NEVER
    a claim of proof -- a heuristic SCREEN against two fixed thresholds and a token-count floor
    (`_VERDICT_MIN_TOKENS`, `_VERDICT_MAX_MEAN_ABS_DELTA_NATS`), stated in the returned dict so the
    caller can print them alongside the verdict rather than let a bare label imply more than it does.

      * INSUFFICIENT_SAMPLE -- fewer than `_VERDICT_MIN_TOKENS` verified tokens were diffed (checked
        FIRST, before either of the other two): too small a sample to say anything either way.
      * NO_DETECTABLE_DIFF -- zero argmax flips AND the MEAN, across verified runs, of each run's own
        `summary.mean_abs_delta_nats_all` is below the threshold AND the token floor is met. Flags the
        classic silent no-op fine-tune/LoRA failure mode.
      * CHANGED -- everything else that cleared the token floor (including "zero flips but confidence
        shifted enough to fail the mean-delta threshold" -- a real, if flip-free, behavioral change).
        Carries per-category argmax-flip counts (category comes from the run, stamped onto the receipt
        by `quant_check.build_receipts`) alongside the existing top-flips detail already in `agg`.

    `receipts` is the per-run list from `build_receipts`/`run_direction` (unverified entries are skipped);
    `agg` is that same list's `aggregate_receipts` rollup (supplies `total_tokens`/`total_flipped`). Pure,
    never raises."""
    total_tokens = agg.get("total_tokens", 0) or 0
    total_flipped = agg.get("total_flipped", 0) or 0
    thresholds = {"min_total_tokens": _VERDICT_MIN_TOKENS,
                 "max_mean_abs_delta_nats_all": _VERDICT_MAX_MEAN_ABS_DELTA_NATS}

    verified_means = []
    per_category_flips: dict = {}
    for r in receipts:
        if not isinstance(r, dict) or not r.get("causal_verified"):
            continue
        summary = r.get("summary") or {}
        mean_all = summary.get("mean_abs_delta_nats_all")
        if isinstance(mean_all, (int, float)):
            verified_means.append(float(mean_all))
        category = r.get("category") or "uncategorized"
        per_category_flips[category] = per_category_flips.get(category, 0) + int(summary.get("n_flipped") or 0)
    mean_of_means = round(sum(verified_means) / len(verified_means), 6) if verified_means else None

    base = {"thresholds": thresholds, "total_tokens": total_tokens, "total_flipped": total_flipped,
           "mean_abs_delta_nats_all_mean": mean_of_means, "is_heuristic": True}

    if total_tokens < _VERDICT_MIN_TOKENS:
        base.update(verdict="INSUFFICIENT_SAMPLE",
                    message=(f"only {total_tokens} verified token(s) diffed (need >= {_VERDICT_MIN_TOKENS}) "
                             "-- too small a sample to render a verdict either way; gather more runs "
                             "(--runs N, or --from-log against a bigger journal)."))
        return base

    if total_flipped == 0 and mean_of_means is not None and mean_of_means < _VERDICT_MAX_MEAN_ABS_DELTA_NATS:
        base.update(verdict="NO_DETECTABLE_DIFF", message=_NO_DETECTABLE_DIFF_MESSAGE)
        return base

    base.update(verdict="CHANGED",
                message=(f"{total_flipped} argmax flip(s) detected across {total_tokens} verified token(s)."),
                per_category_flips=per_category_flips)
    return base


def format_verdict(v: dict) -> str:
    """Pure JSON(`classify_verdict` result) -> text render -- prints the verdict, the thresholds it was
    screened against (never just the label), the observed numbers, the honesty message, and (CHANGED
    only) the per-category flip breakdown."""
    th = v.get("thresholds") or {}
    lines = [f"verdict: {v.get('verdict', '?')}  (heuristic screen on this sample -- not proof)",
            f"  thresholds: total_flipped == 0 AND mean(per-run mean_abs_delta_nats_all) < "
            f"{th.get('max_mean_abs_delta_nats_all')} AND total_tokens >= {th.get('min_total_tokens')}",
            f"  observed: total_tokens={v.get('total_tokens')}, total_flipped={v.get('total_flipped')}, "
            f"mean_abs_delta_nats_all_mean={v.get('mean_abs_delta_nats_all_mean')}"]
    if v.get("message"):
        lines.append(f"  {v['message']}")
    per_cat = v.get("per_category_flips")
    if v.get("verdict") == "CHANGED" and per_cat:
        lines.append("  per-category argmax flips:")
        for cat, n in sorted(per_cat.items(), key=lambda kv: -kv[1]):
            lines.append(f"    {cat}: {n}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------- full orchestration

def run_diff_model(eng_a, eng_b, args, *, label_a: str, label_b: str) -> dict:
    """Model-free ORCHESTRATION of the whole `diff-model` pipeline, given two engine-like objects (a real
    `EngineClient` in production -- one per port/quant/checkpoint file -- a fake exposing
    `.apply_template`/`.score`/`.complete` in tests) and a parsed-args-shaped namespace (reads
    `.runs`/`.from_log`/`.topk`/`.max_tokens`/`.both`/`.own_templates`). Every engine call in this
    function's whole call tree is `eng.apply_template`/`.score`/`.complete` (via `_EngineScoreSub` and
    quant_check's gather/build helpers), so this is fully exercised in tests/test_diff_model.py without a
    process or GPU.

    Order of operations: (1) the mandatory tokenizer preflight -- raises `ctx.CloznError` (never a silent
    degraded diff) if `check_tokenizer_compat` finds a mismatch; (2) the template-policy check and
    resolution (`--own-templates` vs the default reference-rendering policy) via `check_template_match`
    plus `quant_check._EngineScoreSub`'s `template_engine` param; (3) the reference-anchored ladder
    (always); (4) with `args.both`, the candidate-anchored reverse ladder too. Returns the JSON-shaped
    result dict that both the `--json` path and `format_diff_model_report` consume."""
    from clozn.cli import main as ctx

    sub_a0 = qc._EngineScoreSub(eng_a)
    sub_b0 = qc._EngineScoreSub(eng_b)

    compat = check_tokenizer_compat(sub_a0, sub_b0)
    if not compat["compatible"]:
        raise ctx.CloznError(_tokenizer_refusal_message(compat))

    tmpl = check_template_match(sub_a0, sub_b0)
    own_templates = bool(getattr(args, "own_templates", False))
    if own_templates:
        sub_a, sub_b = sub_a0, sub_b0
        template_policy = "own"
        template_caveat = None if tmpl["match"] else _TEMPLATE_OWN_CAVEAT
    elif tmpl["match"]:
        sub_a, sub_b = sub_a0, sub_b0
        template_policy = "reference"
        template_caveat = None
    else:
        sub_a = sub_a0
        sub_b = qc._EngineScoreSub(eng_b, template_engine=eng_a)
        template_policy = "reference"
        template_caveat = _TEMPLATE_DIFFER_REFERENCE_CAVEAT

    receipts_ref, agg_ref = run_direction(sub_a, sub_a, sub_b, label_a=label_a, label_b=label_b, args=args)
    verdict_ref = classify_verdict(receipts_ref, agg_ref)

    result = {
        "label_a": label_a, "label_b": label_b,
        "tokenizer_compat": compat,
        "template_match": tmpl["match"],
        "template_policy": template_policy,
        "template_caveat": template_caveat,
        "reference_anchored": {"agg": agg_ref, "verdict": verdict_ref},
    }

    if bool(getattr(args, "both", False)):
        receipts_cand, agg_cand = run_direction(sub_b, sub_a, sub_b, label_a=label_a, label_b=label_b, args=args)
        verdict_cand = classify_verdict(receipts_cand, agg_cand)
        result["candidate_anchored"] = {"agg": agg_cand, "verdict": verdict_cand}

    return result


def _relabel_ladder(ladder_text: str, headline: str) -> str:
    """`quant_check.format_ladder`'s render, first line swapped for diff-model's own headline -- reuses
    ALL of that function's rendering (per-run rows, top flips, caveat/topk_note verbatim) rather than
    forking it; only the "quant-check: A vs B" headline differs."""
    lines = ladder_text.split("\n")
    if lines:
        lines[0] = headline
    return "\n".join(lines)


def format_diff_model_report(result: dict) -> str:
    """Pure JSON(`run_diff_model` result) -> text render. Headline is "diff-model: <ref> vs <candidate>"
    (never quant-check's own headline -- see `_relabel_ladder`). Prints the tokenizer-preflight summary,
    the template-policy caveat (if any), the reference-anchored ladder + verdict always, and -- with
    `--both` -- the candidate-anchored ladder + verdict too, each under its own explicit direction label
    (see module docstring's DIRECTIONS)."""
    label_a, label_b = result.get("label_a", "A"), result.get("label_b", "B")
    headline = f"diff-model: {label_a} vs {label_b}"
    lines = [headline]

    compat = result.get("tokenizer_compat") or {}
    lines.append(f"tokenizer preflight: {'compatible' if compat.get('compatible') else 'INCOMPATIBLE'} "
                f"across {len(compat.get('probes') or [])} probe(s)")

    if result.get("template_match"):
        lines.append("chat templates: identical under both engines -- no rendering override needed.")
    elif result.get("template_caveat"):
        lines.append(result["template_caveat"])
    lines.append("")

    ref = result.get("reference_anchored") or {}
    lines.append("=== reference-anchored: what would the candidate change about the reference's behavior "
                "(forgetting/no-op view) ===")
    lines.append(_relabel_ladder(qc.format_ladder(ref.get("agg", {})), headline))
    lines.append("")
    lines.append(format_verdict(ref.get("verdict", {})))

    cand = result.get("candidate_anchored")
    if cand:
        lines.append("")
        lines.append("=== candidate-anchored: what the reference would do differently on the candidate's "
                    "own answers (target-gain view) ===")
        lines.append(_relabel_ladder(qc.format_ladder(cand.get("agg", {})), headline))
        lines.append("")
        lines.append(format_verdict(cand.get("verdict", {})))

    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the CLI

def add_subparser(sub):
    """Registers `clozn diff-model` on an argparse subparsers object -- same pattern as
    `quant_check.add_subparser` (see that module's docstring for the two reasons this is its own
    function): this module's own tests can build a throwaway parser and exercise --help/defaults/flag-
    parsing without touching main.py, and it documents the exact main.py registration edit as real,
    testable code. NOT called automatically -- clozn/cli/main.py owns build_parser()/dispatch."""
    pd = sub.add_parser("diff-model", help="base vs fine-tune/merge per-token behavior receipts -- did "
                        "the candidate change what the model says, or is it a silent no-op? (generalizes "
                        "clozn quant-check; see clozn/cli/commands/diff_model.py's module docstring for "
                        "the tokenizer-compatibility trap this command exists to refuse on)")
    pd.add_argument("reference", help="the REFERENCE model (e.g. the base checkpoint) -- a known short "
                    "name, local GGUF path, or fuzzy filename fragment, resolved like `clozn run`'s model arg")
    pd.add_argument("candidate", help="the CANDIDATE model under test (e.g. a fine-tune, LoRA merge, or "
                    "a later checkpoint of the SAME model family -- must share a tokenizer with the "
                    "reference, or this command refuses)")
    pd.add_argument("--runs", type=int, default=8, help="how many runs to diff: fresh greedy prompts "
                    "generated under the anchor model (default, capped at the built-in prompt table's "
                    "size), or the N most recent run-journal entries with --from-log (default 8)")
    pd.add_argument("--from-log", action="store_true", help="diff the N most recent runs from your own "
                    "run journal (clozn trace/explain's history) instead of generating fresh prompts")
    pd.add_argument("--topk", type=int, default=8, help="topk requested on every /score call -- rank 0 of "
                    "topk IS that arm's argmax, needed for flip detection (default 8)")
    pd.add_argument("--max-tokens", type=int, default=200, help="max tokens for a FRESH generation under "
                    "the anchor model (ignored with --from-log, which reuses each run's own recorded answer)")
    pd.add_argument("--port-a", type=int, default=0, help="port for the reference engine (default: a free port)")
    pd.add_argument("--port-b", type=int, default=0, help="port for the candidate engine (default: a free port)")
    pd.add_argument("--cpu", action="store_true", help="force the CPU build for both engines")
    pd.add_argument("--json", action="store_true",
                    help="print the raw result (tokenizer preflight, template policy, ladder(s), verdict(s)) "
                         "as JSON instead of the text report")
    pd.add_argument("--both", action="store_true", help="also run the candidate-anchored reverse ladder "
                    "(generate under the candidate, teacher-force under both) -- the target-gain view, in "
                    "addition to the default reference-anchored forgetting/no-op view")
    pd.add_argument("--own-templates", action="store_true", help="render each model's prompts with its OWN "
                    "chat template instead of the default policy (both arms rendered under the reference's "
                    "template when the two differ) -- measures the candidate's DEPLOYED behavior including "
                    "template differences, not an isolated weights diff")
    pd.add_argument("--mechanistic", action="store_true", help="resolve a clozn.mechanistic-target.v1 "
                    "artifact (MECH-CASE-00) for a failed --case cell against these two model files, "
                    "instead of running the ordinary per-token ladder -- needs no engine/GPU boot; see "
                    "clozn/cli/commands/mechanistic.py")
    pd.add_argument("--case", default=None, metavar="RESULT.json:SUITE/CASE", help="with --mechanistic: "
                    "an experiment result file and the suite/case whose failing cell anchors the target "
                    "(e.g. result.json:target/refusal_case)")
    pd.add_argument("--variant", default="candidate", help="with --mechanistic: which variant's cell is "
                    "the candidate under test (default: candidate)")
    pd.add_argument("--reference-variant", default=None, help="with --mechanistic: override the reference "
                    "variant (default: the manifest's own baseline_variant)")
    pd.add_argument("--seed", type=int, default=0, help="with --mechanistic: which seed's cell to use "
                    "(default 0) -- unrelated to --runs/--from-log's own sampling")
    pd.add_argument("--out", default=None, help="with --mechanistic: explicit output path for the target "
                    "artifact (default: ~/.clozn/mechanistic-targets/<target_id>.json)")
    pd.set_defaults(fn=cmd_diff_model)
    return pd


def cmd_diff_model(args):
    """`clozn diff-model <reference> <candidate> [--runs N | --from-log] [--both] [--own-templates]` --
    THE LIVE PATH. Written and reachable, but deferred: needs a free GPU and two engine processes, so it
    is never invoked by this module's own test suite (mirrors quant_check.cmd_quant_check's own deferral).
    Once a GPU is free:

        clozn diff-model <base.gguf> <finetune.gguf> --runs 8
        clozn diff-model <base.gguf> <merge.gguf> --from-log --runs 20 --both

    Boots the reference then the candidate via `spawn_engine` (the SAME boot path `clozn run`/`clozn
    serve`/`clozn quant-check` use) on --port-a/--port-b (or two free ports), then delegates ALL of the
    new logic to `run_diff_model` (model-free, unit-tested against fakes). Always tears down both engines
    it spawned, even on error -- including the tokenizer-preflight refusal, which raises through the
    `finally` below exactly like `cmd_quant_check`'s own `ctx.CloznError` does.

    `--mechanistic` (MECH-CLI-01) is a DIFFERENT path entirely -- resolving a `clozn.mechanistic-target.v1`
    artifact for a failed `--case` cell needs no engine/GPU at all (GGUF-header + JSON-file work only), so
    it is dispatched here, first, before any engine is ever spawned; see
    `clozn.cli.commands.mechanistic.cmd_diff_model_mechanistic`."""
    if getattr(args, "mechanistic", False):
        from clozn.cli.commands import mechanistic
        return mechanistic.cmd_diff_model_mechanistic(args)

    EngineClient = qc._import_engine_client()

    model_a = resolve_model(args.reference)
    model_b = resolve_model(args.candidate)
    label_a = os.path.splitext(os.path.basename(model_a))[0]
    label_b = os.path.splitext(os.path.basename(model_b))[0]
    port_a = args.port_a or _free_port()
    port_b = args.port_b or _free_port()
    prefer_gpu = not args.cpu

    proc_a = proc_b = None
    try:
        print(f"{fmt.DIM}- booting {label_a} (reference) on port {port_a}...{fmt.RST}", file=sys.stderr)
        proc_a, _health_a, _gpu_a = spawn_engine(model_a, port_a, _flags_for(model_a), prefer_gpu=prefer_gpu)
        print(f"{fmt.DIM}- booting {label_b} (candidate) on port {port_b}...{fmt.RST}", file=sys.stderr)
        proc_b, _health_b, _gpu_b = spawn_engine(model_b, port_b, _flags_for(model_b), prefer_gpu=prefer_gpu)

        eng_a = EngineClient(port=port_a)
        eng_b = EngineClient(port=port_b)
        result = run_diff_model(eng_a, eng_b, args, label_a=label_a, label_b=label_b)
    finally:
        for proc in (proc_a, proc_b):
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(format_diff_model_report(result))
    return 0
