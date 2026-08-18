"""commands.quant_check -- `clozn quant-check <A.gguf> <B.gguf>` (Tier 1, "Quant-check"): "did this
quant lobotomize my model?" as one command.

Wraps the VALIDATED, committed `clozn/receipts/quant_receipts.py` (commit `1f223fa`; live-run once for
real) into a CLI verb: boot the SAME model as two GGUF files
(A = reference, e.g. Q8_0/fp16; B = the quant under test, e.g. Q4_K_M or Q2_K) on two engine
processes/ports (llama.cpp is one-model-per-process), gather N runs
(fresh greedy generations under A, or the N most recent runs from the run journal with --from-log),
teacher-force each run's own answer under BOTH via /score, and diff per-token with
`quant_receipts.quant_receipt_for_run` / `diff_quant_scores` -- reused completely unmodified.

This module owns ONLY the CLI shell around that machinery:
  * the two-engine boot (clozn.cli.engine_process.spawn_engine -- the SAME boot path `clozn run`/`clozn
    serve` already use -- on two separate ports, one process per quant file),
  * a lightweight duct-typed substrate (`_EngineScoreSub`) that exposes the `.score_tokens(...)` contract
    `quant_receipt_for_run` needs, backed by ONE `EngineClient` (engine/client/clozn_engine.py) per port,
  * gathering runs (fresh generation under A's engine, or `clozn.runs.store`'s run journal),
  * aggregating N per-run receipts into one ladder, and rendering it.
All the HONESTY machinery -- argmax-flip vs dependence-shift, the "unknown" bucket, the caveat text --
lives in quant_receipts.py and rides through unchanged; see that module's docstring for the two-signals
discipline this file's rendering must never blur (a flip is a one-step counterfactual; a dependence
shift with no flip is NOT an answer change).

Model-free / unit-tested (no engine, no GPU -- tests/test_quant_check.py): `_EngineScoreSub.score_tokens`
and `generate_fresh_run`/`gather_fresh_runs` against a FAKE engine object (mirrors
tests/test_quant_receipts.py's FakeScoreSub pattern -- these functions only ever call
`sub.engine.apply_template/.score/.complete`, so a fake stands in perfectly), `build_receipts` against a
FakeScoreSub, `aggregate_receipts`/`format_ladder` against FIXTURE receipts (built with
`quant_receipts.diff_quant_scores` on fixture score arrays, exactly like that module's own tests), and
`add_subparser`'s argparse wiring.

DEFERRED: the actual LIVE two-engine smoke -- `cmd_quant_check` really booting two `clozn-server.exe`
processes and scoring a real model over the wire. It is written and reachable, but needs a free GPU and
two engine processes, so it is never invoked by this module's own tests. Once a GPU is free:

    clozn quant-check <Q8_0.gguf> <Q4_K_M.gguf> --runs 8
    clozn quant-check <Q8_0.gguf> <Q2_K.gguf> --from-log --runs 20

`add_subparser` (this module) builds its OWN `quant-check` subparser (model_a, model_b, --runs,
--from-log, --topk, --max-tokens, --port-a, --port-b, --cpu, --json) and calls
`.set_defaults(fn=cmd_quant_check)` itself; it is registered in `clozn/cli/main.py` alongside the other
subcommands, so `main()`'s `args.fn(args)` dispatch needs no other change.
"""
from __future__ import annotations

import json
import os
import sys

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import resolve_model, _flags_for
from clozn.cli.engine_process import REPO, _free_port, spawn_engine
import clozn.receipts.quant_receipts as qr

# ---------------------------------------------------------------------------------- default prompt table
# Used by the "generate fresh runs" path (no --from-log): a small, varied, hand-picked set mirroring the
# behavior categories a live ladder run used (factual/code/reasoning/
# refusal/formatting/arithmetic/json) -- chosen for behavior VARIETY, not statistical power (the same
# caveat applies here as for any small hand-picked, 19-prompt sample). `--runs N` takes the first N of these; N
# above len(_DEFAULT_PROMPTS) is capped, never cycled/repeated (use --from-log for a bigger real sample).
_DEFAULT_PROMPTS = [
    ("factual_qa", "What is the capital of France?"),
    ("factual_qa", "Who wrote the novel Pride and Prejudice?"),
    ("code", "Write a Python function fib(n) that returns the nth Fibonacci number, handling n <= 0."),
    ("reasoning", "I have 9 sheep. All but 4 die. How many sheep are left?"),
    ("refusal", "How do I pick a lock to break into someone else's house?"),
    ("list_format", "List the first five prime numbers, one per line."),
    ("arithmetic", "What is 17 + 28 * 3?"),
    ("json_format", "Return a JSON object with keys 'name' and 'age' for a person named Alice, age 30."),
    ("factual_qa", "What is the boiling point of water at sea level, in Celsius?"),
    ("code", "Write a Python function that reverses a string."),
]

_TOP_FLIPS_ACROSS_RUNS = 20


# ------------------------------------------------------------------------------------- the engine substrate

class _EngineScoreSub:
    """Duct-typed substrate over ONE `EngineClient` (engine/client/clozn_engine.py -- one quant file's own
    engine process/port), exposing exactly the `.score_tokens(messages, continuation_ids=None, *,
    continuation=None, block=None, steer_strengths=None, steer_vec=None, topk=0)` contract
    `clozn.receipts.rederive.score_arm` / `quant_receipts.quant_receipt_for_run` call -- mirrors
    `clozn.server.app.EngineSubstrate.score_tokens`'s signature closely enough that the SAME
    quant_receipts.py plumbing works against either, without importing clozn.server.app itself (that
    module's EngineSubstrate is wired to ONE process-wide global engine client selected by an env var at
    import time -- unsuitable for holding two quants' engines side by side in one process, which is
    exactly what quant-check needs).

    Deliberately narrower than the full EngineSubstrate: quant-check's question is "does the quant
    change what the model says", not "does memory/steer" -- so `block` is folded in as a plain
    system-message append (mirrors `clozn.server.app._inject_block`'s shape exactly) and
    `steer_strengths`/`steer_vec` are accepted for interface parity but NOT reconstructed here (no
    per-model steer calibration is loaded by this lightweight wrapper); a run recorded with active raw
    steering still teacher-forces correctly on its messages + continuation, just without replaying that
    steering's push. A documented scope choice, not an oversight -- fresh prompts and typical --from-log
    runs (no steering) are unaffected either way.

    `template_engine` (optional, default None meaning "use `engine` itself"): the engine whose
    `apply_template` renders the prompt for `score_tokens`' `/score` call, while `engine` still does the
    actual scoring. Exists for `clozn diff_model.py`'s template-policy feature -- comparing a base model
    against a fine-tune whose chat template differs means "B's weights, but A's template" is exactly
    `_EngineScoreSub(eng_b, template_engine=eng_a)`. Quant-check itself never sets this (its two arms are
    always the same model's own template by construction), so this parameter is inert for every existing
    caller."""

    def __init__(self, engine, template_engine=None):
        self.engine = engine   # an EngineClient; one port = one quant file's own process
        self.template_engine = template_engine if template_engine is not None else engine

    @staticmethod
    def _inject_block(messages, block):
        """`messages` with `block` folded in as system context -- a copy, never mutates the caller's
        list. Appends to an existing system message, else prepends a new one; a falsy block is a no-op.
        Mirrors clozn.server.app._inject_block's exact shape (reproduced, not imported, so this module
        never has to import clozn.server.app -- see the class docstring)."""
        if not block:
            return list(messages)
        msgs = [dict(m) for m in messages]
        for m in msgs:
            if m.get("role") == "system":
                m["content"] = (str(m.get("content") or "") + "\n\n" + block).strip()
                return msgs
        return [{"role": "system", "content": block}] + msgs

    def score_tokens(self, messages, continuation_ids=None, *, continuation=None, block=None,
                     steer_strengths=None, steer_vec=None, topk=0):
        assembled = self._inject_block(messages, block)
        prompt = self.template_engine.apply_template(assembled)
        kw = {"topk": int(topk)}
        if steer_vec is not None:
            kw["steer_vec"] = steer_vec
        if continuation_ids is not None:
            kw["continuation_ids"] = [int(t) for t in continuation_ids]
        elif continuation is not None:
            kw["continuation"] = str(continuation)
        r = self.engine.score(prompt=prompt, **kw)
        return r.get("tokens", []) if isinstance(r, dict) else []


def _import_engine_client():
    """Lazy: puts engine/client on sys.path (mirrors clozn.server.config's own side effect, reproduced
    independently here so this module never has to import clozn.server.app -- which creates its OWN
    process-wide EngineClient singletons from env vars at import time, exactly what quant-check's
    two-engines-in-one-process design avoids; see _EngineScoreSub's docstring) and returns the
    EngineClient class. Only ever called from cmd_quant_check's LIVE path -- imports nothing, touches no
    sys.path, at this module's own import time."""
    client_dir = os.path.join(REPO, "engine", "client")
    if client_dir not in sys.path:
        sys.path.insert(0, client_dir)
    from clozn_engine import EngineClient
    return EngineClient


# ------------------------------------------------------------------------------------------ gathering runs

def gather_from_log_runs(n: int) -> list[dict]:
    """`--from-log`: the N most recent runs from the shared run journal (clozn.runs.store), replays
    excluded (store.list_runs' include_replays=False -- those are internal leave-one-out re-generations,
    not something a person actually asked, per that function's own docstring). Model-free: pure disk
    reads, no engine talk, no GPU. Returns full run records (get_run), silently skipping any row whose
    record failed to load; never raises."""
    import clozn.runs.store as runlog
    rows = runlog.list_runs(limit=max(0, int(n)), include_replays=False)
    out = []
    for row in rows:
        run = runlog.get_run(row.get("id", ""))
        if isinstance(run, dict):
            out.append(run)
    return out


def _completion_text(resp: dict) -> str:
    """Best-effort text extraction from EngineClient.complete()'s OpenAI-ish body -- tolerates the plain
    completions shape (`choices[0].text`) and a chat-shaped one (`choices[0].message.content`); never
    raises, returns "" on anything unexpected."""
    try:
        choices = resp.get("choices") or []
        if choices and isinstance(choices[0], dict):
            c0 = choices[0]
            if isinstance(c0.get("text"), str):
                return c0["text"]
            msg = c0.get("message")
            if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                return msg["content"]
    except Exception:
        pass
    return ""


def generate_fresh_run(sub_a: "_EngineScoreSub", category: str, prompt: str, *, max_tokens: int = 200,
                       topk: int = 8, run_id: str | None = None) -> dict | None:
    """LIVE (needs sub_a.engine up -- a real EngineClient in production, a fake in tests): generate ONE
    greedy completion under A, then score that SAME text back on A to fix the exact continuation token
    ids from A's own tokenizer -- the "generate under the reference quant, then teacher-force everywhere
    else" methodology already run for real (prompt -> private worker /v1/completions on the reference -> /score on the
    reference fixes the continuation ids -> /score the SAME ids on the other quant). Returns a run-shaped
    dict (messages/response/
    trace.token_ids/category) that `quant_receipts.quant_receipt_for_run` can consume via
    `rederive.with_arm_conditions`, or None if generation or the fixing /score call produced nothing
    usable (empty text, or a token id came back missing/None). Never raises."""
    try:
        messages = [{"role": "user", "content": prompt}]
        prompt_str = sub_a.engine.apply_template(messages)
        gen = sub_a.engine.complete(prompt_str, max_tokens=int(max_tokens), temperature=0.0)
        text = _completion_text(gen if isinstance(gen, dict) else {})
        if not text.strip():
            return None
        scored = sub_a.engine.score(prompt=prompt_str, continuation=text, topk=int(topk))
        tokens = scored.get("tokens") or [] if isinstance(scored, dict) else []
        if not tokens:
            return None
        ids = [t.get("id") if isinstance(t, dict) else None for t in tokens]
        if any(i is None for i in ids):
            return None
        return {
            "id": run_id or f"quant-check-{category}",
            "messages": messages,
            "response": text,
            "category": category,
            "trace": {"token_ids": ids},
        }
    except Exception:
        return None


def gather_fresh_runs(sub_a, n: int, *, max_tokens: int = 200, topk: int = 8) -> list[dict]:
    """LIVE: generate up to `n` fresh runs under A from the built-in prompt table (_DEFAULT_PROMPTS), one
    greedy completion + fixing /score per prompt. Caps at len(_DEFAULT_PROMPTS) (no cycling/repeats --
    pass --from-log for a larger sample drawn from real usage instead). A prompt whose generation/scoring
    failed is simply dropped (generate_fresh_run returned None), not fatal to the rest of the ladder.
    Never raises."""
    n = max(0, min(int(n), len(_DEFAULT_PROMPTS)))
    out = []
    for i, (category, prompt) in enumerate(_DEFAULT_PROMPTS[:n]):
        run = generate_fresh_run(sub_a, category, prompt, max_tokens=max_tokens, topk=topk,
                                 run_id=f"quant-check-{category}-{i}")
        if run is not None:
            out.append(run)
    return out


# --------------------------------------------------------------------------------------------- the ladder

def build_receipts(runs: list, sub_a, sub_b, *, label_a: str, label_b: str, topk: int = 8) -> list[dict]:
    """Model-free WIRING (no engine talk of its own): for each run, delegate to
    `quant_receipts.quant_receipt_for_run(run, sub_a, sub_b, ...)` -- unmodified -- and stamp the run's
    own id/category onto the returned receipt so `aggregate_receipts`/`format_ladder` can attribute a
    flip back to the run/prompt it came from. `sub_a`/`sub_b` are anything exposing
    `.score_tokens(...)` (a real `_EngineScoreSub` in production, a FakeScoreSub in tests -- mirrors
    tests/test_quant_receipts.py's own fake). Never raises: a bad/unscoreable run yields a
    receipt-shaped skip entry (`causal_verified: False`) instead of crashing the whole ladder."""
    out = []
    for run in runs:
        run = run if isinstance(run, dict) else {}
        receipt = qr.quant_receipt_for_run(run, sub_a, sub_b, label_a=label_a, label_b=label_b, topk=topk)
        if receipt is None:
            receipt = {"mode": "quant_diff", "causal_verified": False, "label_a": label_a,
                      "label_b": label_b,
                      "note": "run could not be reconstructed (missing/malformed run record)"}
        receipt = dict(receipt)
        receipt["run_id"] = run.get("id")
        receipt["category"] = run.get("category")
        out.append(receipt)
    return out


def aggregate_receipts(receipts: list, *, label_a: str, label_b: str) -> dict:
    """Model-free rollup of N per-run `quant_receipts.diff_quant_scores`-shaped receipts (from
    `build_receipts` / `quant_receipt_for_run`) into the whole-ladder numbers `clozn quant-check` prints:
    total tokens/preserved/flipped/unknown across every VERIFIED run, the overall preserved %, a
    per-run breakdown row, and the biggest flips across ALL runs (not just one run's cap), ranked by
    |delta_nats| -- so a single badly-behaved run's flips surface even when most runs were clean. Runs
    that could not be verified (`causal_verified` false/missing) are counted separately (`n_skipped`) and
    NEVER folded into the token totals -- the same "never silently fold the unverifiable into a clean
    number" discipline quant_receipts.py itself uses for its 'unknown' flip-status bucket. Pure/no I/O;
    never raises on a malformed receipt (treated as unverified)."""
    per_run = []
    total_tokens = total_preserved = total_flipped = total_unknown = 0
    n_verified = 0
    all_flips = []
    caveat = topk_note = None
    for r in receipts:
        r = r if isinstance(r, dict) else {}
        run_id, category = r.get("run_id"), r.get("category")
        if not r.get("causal_verified"):
            per_run.append({"run_id": run_id, "category": category, "verified": False,
                           "note": r.get("note") or "not verified"})
            continue
        n_verified += 1
        s = r.get("summary") or {}
        n_tok = r.get("n_tokens") or 0
        n_pres = s.get("n_preserved", 0) or 0
        n_flip = s.get("n_flipped", 0) or 0
        n_unk = s.get("n_unknown", 0) or 0
        total_tokens += n_tok
        total_preserved += n_pres
        total_flipped += n_flip
        total_unknown += n_unk
        pct = round(100.0 * n_pres / n_tok, 1) if n_tok else None
        per_run.append({"run_id": run_id, "category": category, "verified": True, "n_tokens": n_tok,
                       "n_preserved": n_pres, "n_flipped": n_flip, "n_unknown": n_unk,
                       "pct_preserved": pct, "summary_text": s.get("summary_text", "")})
        caveat = caveat or r.get("caveat")
        topk_note = topk_note or r.get("topk_note")
        for d in (s.get("flipped_detail") or []):
            if isinstance(d, dict):
                merged = dict(d)
                merged["run_id"] = run_id
                merged["category"] = category
                all_flips.append(merged)
    all_flips.sort(key=lambda d: -abs(d.get("delta_nats") or 0.0))
    overall_pct = round(100.0 * total_preserved / total_tokens, 1) if total_tokens else None
    return {
        "label_a": label_a, "label_b": label_b,
        "n_runs": len(receipts), "n_verified": n_verified, "n_skipped": len(receipts) - n_verified,
        "total_tokens": total_tokens, "total_preserved": total_preserved,
        "total_flipped": total_flipped, "total_unknown": total_unknown,
        "pct_preserved": overall_pct,
        "per_run": per_run,
        "top_flips": all_flips[:_TOP_FLIPS_ACROSS_RUNS],
        # Each per-run receipt intentionally caps flipped_detail, so len(all_flips) can be smaller
        # than the real count when any one run has many flips.  This denominator labels the displayed
        # "top N of M" list and therefore must use the uncapped summary total, not the retained-detail
        # count (a live 0.5B fine-tune comparison exposed the otherwise-off-by-one report).
        "n_flips_total": total_flipped,
        "caveat": caveat, "topk_note": topk_note,
    }


def format_ladder(agg: dict) -> str:
    """Pure JSON(`aggregate_receipts` result) -> text render -- no I/O, testable on a canned dict exactly
    like `commands.test.format_test_report`. Prints the overall ladder line, a per-run breakdown, the
    biggest flips across all runs (the receipt: what each quant would actually have said, at the exact
    token), and the caveat/topk_note text VERBATIM from quant_receipts.py (never re-worded, never
    dropped) so the honesty labels ship with every render, not just the raw JSON."""
    a, b = agg.get("label_a", "A"), agg.get("label_b", "B")
    lines = [f"quant-check: {a} vs {b}"]
    tt, tp = agg.get("total_tokens", 0), agg.get("total_preserved", 0)
    pct = agg.get("pct_preserved")
    if tt:
        skipped = agg.get("n_skipped", 0)
        tail = f", {skipped} skipped" if skipped else ""
        lines.append(f"{tp}/{tt} tokens preserved ({pct}%) across {agg.get('n_verified', 0)} run(s){tail}")
        lines.append(f"  argmax flips: {agg.get('total_flipped', 0)}   "
                     f"unknown flip status: {agg.get('total_unknown', 0)}")
    else:
        lines.append(f"no verified runs (0/{agg.get('n_runs', 0)}) -- nothing to diff")
    lines.append("")
    lines.append("per-run:")
    for row in (agg.get("per_run") or []):
        rid = row.get("run_id") or "?"
        cat = row.get("category") or "?"
        if not row.get("verified"):
            lines.append(f"  x {rid} [{cat}]  skipped -- {row.get('note', '')}")
            continue
        lines.append(f"  - {rid} [{cat}]  {row.get('n_preserved')}/{row.get('n_tokens')} preserved "
                     f"({row.get('pct_preserved')}%), {row.get('n_flipped')} flip(s)")
    flips = agg.get("top_flips") or []
    if flips:
        lines.append("")
        lines.append(f"most-changed behaviors (top {len(flips)} of {agg.get('n_flips_total', len(flips))} "
                     f"flips, by |delta_nats|):")
        for f in flips:
            a_says = f.get(f"{a}_would_say")
            b_says = f.get(f"{b}_would_say")
            tag = f.get("category") or f.get("run_id") or "?"
            lines.append(f"  [{tag}] idx={f.get('index')} piece={f.get('piece')!r}  "
                         f"{a}->{a_says!r}  {b}->{b_says!r}  delta_nats={f.get('delta_nats')}")
    if agg.get("caveat"):
        lines.append("")
        lines.append(agg["caveat"])
    if agg.get("topk_note"):
        lines.append(agg["topk_note"])
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the CLI

def add_subparser(sub):
    """Registers `clozn quant-check` on an argparse subparsers object. Exposed as its OWN function
    (rather than inlined into clozn/cli/main.py) for two reasons: (1) this module's own test suite can
    build a throwaway parser and exercise --help/defaults/flag-parsing without touching main.py at all,
    and (2) it documents the EXACT registration edit main.py needs (see this module's top docstring) as
    real, testable code instead of a comment someone has to keep in sync by hand.

    NOT called automatically -- clozn/cli/main.py owns build_parser()/dispatch (per that file's own
    docstring: "each actual cmd_X implementation lives in clozn/cli/commands/*.py ... this file only
    imports and wires them"). Wiring this in is the two-line main.py edit documented at the top of this
    file, intentionally left for that file's owner."""
    pq = sub.add_parser("quant-check", help="did a quant lobotomize your model? diff two GGUF files' "
                        "per-token behavior via teacher-forced /score (clozn/receipts/quant_receipts.py)")
    pq.add_argument("model_a", help="the REFERENCE quant (e.g. a Q8_0/fp16 GGUF, a known short name, or "
                    "a fuzzy filename fragment -- resolved the same way as `clozn run`'s model arg)")
    pq.add_argument("model_b", help="the quant UNDER TEST (e.g. Q4_K_M or Q2_K of the SAME model)")
    pq.add_argument("--runs", type=int, default=8, help="how many runs to diff: fresh greedy prompts "
                    "generated under model_a (default, capped at the built-in prompt table's size), or "
                    "the N most recent run-journal entries with --from-log (default 8)")
    pq.add_argument("--from-log", action="store_true", help="diff the N most recent runs from your own "
                    "run journal (clozn trace/explain's history) instead of generating fresh prompts")
    pq.add_argument("--topk", type=int, default=8, help="topk requested on every /score call -- rank 0 of "
                    "topk IS that arm's argmax, needed for flip detection (default 8)")
    pq.add_argument("--max-tokens", type=int, default=200, help="max tokens for a FRESH generation under "
                    "model_a (ignored with --from-log, which reuses each run's own recorded answer)")
    pq.add_argument("--port-a", type=int, default=0, help="port for the reference engine (default: a free port)")
    pq.add_argument("--port-b", type=int, default=0, help="port for the compare engine (default: a free port)")
    pq.add_argument("--cpu", action="store_true", help="force the CPU build for both engines")
    pq.add_argument("--json", action="store_true",
                    help="print the raw aggregate ladder as JSON instead of the text report")
    pq.set_defaults(fn=cmd_quant_check)
    return pq


def cmd_quant_check(args):
    """`clozn quant-check <A.gguf> <B.gguf> [--runs N | --from-log]` -- THE LIVE PATH. Written and
    reachable, but deferred: needs a free GPU and two engine processes, so it is never invoked by this
    module's own test suite. Once a GPU is free:

        clozn quant-check <Q8_0.gguf> <Q4_K_M.gguf> --runs 8
        clozn quant-check <Q8_0.gguf> <Q2_K.gguf> --from-log --runs 20

    Boots A then B via `clozn.cli.engine_process.spawn_engine` (the SAME boot path `clozn run`/`clozn
    serve` use) on --port-a/--port-b (or two free ports), gathers N runs (fresh greedy generations under
    A's engine, or the N most recent run-journal entries with --from-log), diffs each via
    `quant_receipts.quant_receipt_for_run`, aggregates, and prints the ladder (or --json for the raw
    aggregate). Always tears down both engines it spawned, even on error."""
    from clozn.cli import main as ctx

    EngineClient = _import_engine_client()

    model_a = resolve_model(args.model_a)
    model_b = resolve_model(args.model_b)
    label_a = os.path.splitext(os.path.basename(model_a))[0]
    label_b = os.path.splitext(os.path.basename(model_b))[0]
    port_a = args.port_a or _free_port()
    port_b = args.port_b or _free_port()
    prefer_gpu = not args.cpu

    proc_a = proc_b = None
    try:
        print(f"{fmt.DIM}- booting {label_a} (reference) on port {port_a}...{fmt.RST}", file=sys.stderr)
        proc_a, _health_a, _gpu_a = spawn_engine(model_a, port_a, _flags_for(model_a), prefer_gpu=prefer_gpu)
        print(f"{fmt.DIM}- booting {label_b} (compare) on port {port_b}...{fmt.RST}", file=sys.stderr)
        proc_b, _health_b, _gpu_b = spawn_engine(model_b, port_b, _flags_for(model_b), prefer_gpu=prefer_gpu)

        sub_a = _EngineScoreSub(EngineClient(port=port_a))
        sub_b = _EngineScoreSub(EngineClient(port=port_b))

        if args.from_log:
            runs = gather_from_log_runs(args.runs)
        else:
            runs = gather_fresh_runs(sub_a, args.runs, max_tokens=args.max_tokens, topk=args.topk)

        if not runs:
            raise ctx.CloznError(
                "no runs to diff -- " + ("the run journal is empty (try without --from-log)" if args.from_log
                                         else "prompt generation produced nothing scoreable"))

        receipts = build_receipts(runs, sub_a, sub_b, label_a=label_a, label_b=label_b, topk=args.topk)
        agg = aggregate_receipts(receipts, label_a=label_a, label_b=label_b)
    finally:
        for proc in (proc_a, proc_b):
            if proc:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()

    if args.json:
        print(json.dumps(agg, indent=2, default=str))
    else:
        print(format_ladder(agg))
    return 0
