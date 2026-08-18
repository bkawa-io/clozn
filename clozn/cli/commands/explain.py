"""commands.explain -- the run-inspection family: `clozn trace` (the last run's confidence timeline),
`clozn inspect <run_id>` (local-journal-first, zero-generation explanation), `clozn branch` (re-run from
an uncertain point), and `clozn explain` (+ its opt-in `--why` accountable-self narration). These read a recorded run rather than generating new
text by default -- `explain` is DISPLAY ONLY unless `--why` is given, and `branch` is the one command here
that does generate (a continuation past the forked token).

format_explain()/format_narrate() are pure functions (JSON in, text out) factored out specifically so
they're testable with a canned dict -- no server, no model, no GPU -- mirroring `clozn trace`'s confidence-
bar language (trace_io._render_trace).

HOME/CloznError live on `clozn.cli.main`; imported INSIDE the functions that need them (not at module
level) for the same circular-import reason documented in engine_process.py's module docstring.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import _flags_for, _friendly, resolve_model
from clozn.cli.commands.run import _chat_wrap, complete_once
from clozn.cli.engine_process import _find_warm, _free_port
from clozn.cli.runtime_process import RuntimeConfig, spawn_runtime
from clozn.cli.trace_io import (_import_runlog, _list_runlog_traces, _render_trace,
                                _runlog_trace_meta, _runlog_trace_steps)


# ----------------------------------------------------------------------------- trace / branch

def cmd_trace(args):
    from clozn.cli import main as ctx
    try:
        runlog = _import_runlog()
        # include_replays=False: don't let an internal leave-one-out/redundancy re-generation from a
        # `/runs/<id>/receipts` prove-all (clozn.replay.replay, source="replay") masquerade as "the last
        # run" or clutter --list -- it's real data (still fully readable by id), just not something the
        # user actually did.
        rows = runlog.list_runs(limit=12 if args.list else 1, include_replays=False)
    except Exception as e:
        raise ctx.CloznError(f"could not read the SQLite run journal: {e}")
    if not rows:
        print('no runs yet -- run something first:  clozn run qwen "..."')
        return
    if args.list:
        _list_runlog_traces(runlog, limit=12)
        return
    run = runlog.get_run(rows[0].get("id", "")) or {}
    if not run:
        raise ctx.CloznError("latest run disappeared from the run journal")
    steps = _runlog_trace_steps(run)
    _render_trace(_runlog_trace_meta(run, steps), steps)


def cmd_branch(args):
    """Take the road not taken: re-run from an uncertain point with the alternative the model nearly chose.

    Text-level (re-runs prompt + kept tokens + the alt through /api/clozn/generate), so token boundaries can
    shift a hair -- but it shows, concretely, 'what if it had said X instead'. The seed of branch-a-bad-answer."""
    from clozn.cli import main as ctx
    try:
        runlog = _import_runlog()
        rows = runlog.list_runs(limit=1, include_replays=False)
        run = runlog.get_run(rows[0]["id"]) if rows else None
    except Exception as exc:
        raise ctx.CloznError(f"could not read the run journal: {exc}")
    if not run:
        raise ctx.CloznError('no run yet -- run something first:  clozn run qwen "..."')
    prompt = next((m.get("content", "") for m in reversed(run.get("messages") or [])
                   if isinstance(m, dict) and m.get("role") == "user"), "")
    meta = {"id": run.get("id"), "model": run.get("model"), "prompt": prompt}
    steps = [s for s in _runlog_trace_steps(run)
             if (s.get("piece", s.get("text", "")) or "").strip()]   # branch on real tokens
    if not steps:
        raise ctx.CloznError("the last trace has no branchable tokens.")
    idx = (max(0, min(args.at, len(steps) - 1)) if args.at is not None
           else min(range(len(steps)),
                    key=lambda i: fmt._num(steps[i].get("prob", steps[i].get("conf", steps[i].get("confidence"))), 1.0)))
    step = steps[idx]; alts = step.get("alts", [])
    if not alts:
        raise ctx.CloznError(f"no recorded alternative at '{step.get('piece', step.get('text', '')).strip()}' "
                             f"(conf {fmt._num(step.get('prob', step.get('conf', step.get('confidence')))):.2f}).")
    alt = alts[max(0, min(args.pick, len(alts) - 1))]
    model = resolve_model(meta.get("model", "")); flags = _flags_for(model)
    head = _chat_wrap(meta.get("prompt", "")) if flags.get("chat") else meta.get("prompt", "")
    kept = "".join(s.get("piece", s.get("text", "")) for s in steps[:idx])
    alt_piece = alt.get("piece", alt.get("text", ""))
    prefix = head + kept + alt_piece
    print(f"{fmt.BOLD}branch{fmt.RST} of \"{meta.get('prompt', '')[:54]}\"")
    print(f"  fork at token {idx}: it chose {fmt.BOLD}{step.get('piece', step.get('text', '')).strip()!r}{fmt.RST} "
          f"({fmt._num(step.get('prob', step.get('conf', step.get('confidence')))):.2f})"
          f"  ->  branch on {fmt.BOLD}{alt_piece.strip()!r}{fmt.RST} ({fmt._num(alt.get('prob')):.2f})")
    original = fmt._oneline("".join(s.get("piece", s.get("text", "")) for s in steps).strip())
    print(f"  {fmt.DIM}original:{fmt.RST} {original[:130]}")
    warm = _find_warm(model); stack = worker_log = None
    if warm:
        port = warm[0]
    else:
        os.makedirs(ctx.HOME, exist_ok=True)
        worker_log = open(os.path.join(ctx.HOME, "worker-run.log"), "w", encoding="utf-8")
        port = _free_port()
        print(f"{fmt.DIM}  - loading {_friendly(model)} …{fmt.RST}", file=sys.stderr, flush=True)
        stack = spawn_runtime(
            RuntimeConfig(model=model, public_port=port, flags=flags, prefer_gpu=not args.cpu),
            worker_log=worker_log,
        )
    try:
        # rstrip only: a leading space here is the separator between the alt piece and the continuation
        # (word-piece tokenization keeps it as part of `cont`) -- an eager .strip() used to eat it, running
        # the two together ("...suddenlyand it lived...") in the line printed below.
        cont = complete_once(port, prefix, args.max).rstrip()
    finally:
        if stack:
            stack.stop()
        if worker_log:
            worker_log.close()
    branch_text = fmt._oneline((kept + alt_piece + cont).strip())
    print(f"  {fmt.BOLD}branch:{fmt.RST}   {branch_text[:160]}")


# ----------------------------------------------------------------------------- explain (M5 display; M1 assembles)
# `clozn explain <run_id>` (EXPLAIN_THIS_ANSWER_SPEC.md) renders the Studio's already-shipped, zero-generation
# /runs/<id>/explain (clozn.receipts.explain) as a terminal view: the confidence hesitations, the influences
# that were active, and the concepts note. DISPLAY ONLY by default -- this command generates nothing; it POSTs
# to the endpoint (the same "any client" bridge the Run Inspector's own Explain tab uses) and renders whatever
# comes back. The one opt-in exception is `--why` (below): it additionally POSTs to /runs/<id>/narrate, which
# DOES generate (M4's accountable-self narration, two model calls) -- opt-in for exactly that reason.
#
# format_explain() is factored out as a pure function (JSON in, text out) specifically so it's testable with a
# canned /explain dict -- no server, no model, no GPU -- mirroring cmd_trace's confidence-bar language.

def _verified_tag(v) -> str:
    """causal_verified -> a label that never overclaims. M1 (this command's only data source) tags every
    influence None ("active, not proven" -- the spec's own wording); True/False can only ever come from
    M2's on-demand ablation receipt, once it exists -- handled here so the CLI needs no change that day."""
    return "proven" if v is True else "ruled out" if v is False else "was active"


def _format_confidence(conf: dict) -> list[str]:
    out = [f"{fmt.BOLD}confidence{fmt.RST}  {fmt.DIM}measured per token -- never an overall score{fmt.RST}"]
    if not conf.get("available"):
        out.append(f"  {fmt.DIM}not available -- {conf.get('note', 'no trace on this run')}{fmt.RST}")
        return out
    moments = [m for m in fmt._as_list(conf.get("uncertain_moments")) if isinstance(m, dict)]
    spark = fmt._paint_sparkline(conf.get("n_tokens"), moments)
    if spark:
        out.append(f"  {spark}")
        if fmt.COLOR:
            out.append(f"  {fmt._conf_legend()}")
    out.append(f"  {fmt.DIM}{conf.get('summary', '')} of {conf.get('n_tokens', 0)} tokens"
               f" (threshold {conf.get('threshold')}){fmt.RST}")
    for m in moments:
        piece = str(m.get("token") or "").replace("\n", "\\n").replace("\t", "\\t")
        c = fmt._num(m.get("confidence"))
        shown = piece[:16]
        cell = fmt._paint(shown, c) + " " * max(0, 16 - len(shown))
        line = f"   ? {cell} {fmt._confbar(c)} {c:.2f}"
        alts = [a for a in fmt._as_list(m.get("alternatives")) if isinstance(a, dict)]
        if alts:
            altxt = "  ".join(f"{(a.get('piece') or '').strip() or '_'} {fmt._num(a.get('prob')):.2f}" for a in alts[:3])
            line += f"   {fmt.DIM}almost: {altxt}{fmt.RST}"
        out.append(line)
    return out


def _format_influences(inf: dict) -> list[str]:
    out = [f"{fmt.BOLD}influences active{fmt.RST}  {fmt.DIM}active this turn -- not yet proven causal{fmt.RST}"]
    gate, mode = inf.get("gate"), inf.get("mode")
    if gate is not None or mode:
        gate_s = f"{gate:.2f}" if isinstance(gate, (int, float)) else str(gate)
        out.append(f"  {fmt.DIM}gate {gate_s}{(' · ' + str(mode)) if mode else ''}{fmt.RST}")
    cards = [c for c in fmt._as_list(inf.get("cards")) if isinstance(c, dict)]
    if cards:
        for c in cards:
            out.append(f"  [{_verified_tag(c.get('causal_verified'))}] {c.get('text', '')}")
            quote = c.get("quoted_span")
            if quote:
                out.append(f"      {fmt.DIM}“{quote}”{fmt.RST}")
            elif c.get("note"):
                out.append(f"      {fmt.DIM}{c['note']}{fmt.RST}")
    else:
        out.append(f"  {fmt.DIM}{inf.get('note', 'no memory applied')}{fmt.RST}")
    return out


def _format_concepts(conc: dict) -> list[str]:
    out = [f"{fmt.BOLD}concepts{fmt.RST}"]
    if not conc.get("available"):
        out.append(f"  {fmt.DIM}not available -- "
                   f"{conc.get('note', 'no named concept readout was captured for this run')}{fmt.RST}")
        return out
    spans = [s for s in fmt._as_list(conc.get("spans")) if isinstance(s, dict)]
    if not spans:
        out.append(f"  {fmt.DIM}(no spans recorded){fmt.RST}")
        return out
    for span in spans:
        piece = span.get("piece")
        head = f"  {piece!r} " if piece is not None else "  "
        feats = [f for f in fmt._as_list(span.get("features")) if isinstance(f, dict)]
        feat_s = ", ".join(f"{f.get('label') or f.get('id') or '?'} {fmt._num(f.get('score')):.2f}" for f in feats)
        out.append(f"{head}{feat_s}")
    return out


def format_explain(expl: dict) -> str:
    """The M1 explanation object (POST /runs/<id>/explain's JSON body) -> the terminal render. Pure: no
    I/O, no server, no model -- a canned fixture dict renders identically to a live response, which is
    exactly what makes this testable without either. Mirrors cmd_trace's confidence-bar language.

    Honesty is enforced HERE too, not just trusted from the server: never synthesizes an aggregate
    confidence number (only the per-hesitation values/bars explain.py already measured, or plain counts);
    an {"available": false, "note": ...} panel always prints its note (never silently skipped); every
    influence is labeled "was active", never "caused" (see _verified_tag). Never raises: a malformed panel
    degrades to a one-line notice instead of losing the ones that DID render, same discipline as
    clozn.receipts.explain's own explain()."""
    expl = expl if isinstance(expl, dict) else {}
    lines = [f"{fmt.BOLD}explain{fmt.RST}  run {expl.get('run_id') or '?'}", "-" * 62]
    try:
        lines += _format_confidence(fmt._as_dict(expl.get("confidence")))
    except Exception:
        lines += [f"{fmt.BOLD}confidence{fmt.RST}", f"  {fmt.DIM}couldn't render this panel{fmt.RST}"]
    lines.append("")
    try:
        lines += _format_influences(fmt._as_dict(expl.get("influences_active")))
    except Exception:
        lines += [f"{fmt.BOLD}influences active{fmt.RST}", f"  {fmt.DIM}couldn't render this panel{fmt.RST}"]
    lines.append("")
    try:
        lines += _format_concepts(fmt._as_dict(expl.get("concepts")))
    except Exception:
        lines += [f"{fmt.BOLD}concepts{fmt.RST}", f"  {fmt.DIM}couldn't render this panel{fmt.RST}"]
    lines.append("-" * 62)
    return "\n".join(lines)


def _last_run_id():
    """The most recent run in the shared Studio run log (~/.clozn/runs), read directly -- mirrors
    _log_run_cli's own direct `import runlog` (clozn.runs is a stdlib-only sibling). `clozn run`/`serve`
    write every turn straight into this log whether or not the Studio HTTP server is up, so --last
    resolves even while it's down; only the actual /explain fetch below needs the server.

    include_replays=False: an internal leave-one-out/redundancy re-generation from a `/runs/<id>/receipts`
    prove-all (clozn.replay.replay, source="replay") must never outrank the user's own last turn here --
    same reasoning as cmd_trace's "last run" pick above."""
    try:
        import clozn.runs.store as runlog
        runs = runlog.list_runs(limit=1, include_replays=False)
        return runs[0]["id"] if runs else None
    except Exception:
        return None


def _fetch_explain(port: int, run_id: str) -> dict:
    """POST /runs/<id>/explain on the Studio backend -- M1's assembly (clozn.receipts.explain), zero
    generation. A clean CloznError (one line, no traceback) when the Studio isn't up or the run doesn't
    resolve, matching the rest of this CLI's error style."""
    from clozn.cli import main as ctx
    url = f"http://127.0.0.1:{port}/runs/{run_id}/explain"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ctx.CloznError(f"explain failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise ctx.CloznError(
            f"couldn't reach the Clozn gateway on port {port} ({e.reason}). "
            "Start it first:  clozn serve <model>"
        )
    except Exception as e:
        raise ctx.CloznError(f"explain failed: {e}")


def _local_explain(run_id: str) -> dict | None:
    """Assemble M1 directly from the authoritative local SQLite journal.

    This is the default ``clozn inspect`` path: no HTTP server, model, or GPU is needed. ``None`` means the
    id is not in this journal (the caller may then fall back to a gateway); store/assembly failures are
    actionable errors rather than being mistaken for a missing id.
    """
    from clozn.cli import main as ctx
    try:
        import clozn.runs.store as runlog
        run = runlog.get_run(run_id)
        if not run:
            return None
        from clozn.receipts import explain as explain_receipt
        return explain_receipt.explain(run)
    except Exception as exc:
        raise ctx.CloznError(f"could not inspect the local run journal: {exc}")


def cmd_inspect(args):
    """Inspect a returned run id without generating or requiring a running model.

    Prefer the local journal so a gateway can be stopped after a client call and the receipt remains
    inspectable. If the id is not local, fall back to ``--port`` (8080 by default), which covers a caller
    attached to another local Clozn gateway. ``--json`` emits the exact assembled object for scripting.
    """
    from clozn.cli import main as ctx
    rid = _last_run_id() if args.last else args.run_id
    if not rid:
        raise ctx.CloznError("give the clozn_run_id returned by the API, or pass --last")
    obj = _local_explain(rid)
    if obj is None:
        obj = _fetch_explain(args.port or 8080, rid)
    if args.json:
        print(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_explain(obj))


# --------------------------------------------------------------------- narrate (M4 display; narrate.py assembles)
# `clozn explain --why` additionally renders the Studio's POST /runs/<id>/narrate (clozn.receipts.narrate): the
# accountable-self narration -- a receipt-CONSTRAINED "why" diffed against an independent judge, with every
# unsupported claim it catches shown as a warning. Opt-in (--why), unlike the rest of `explain`, because this
# one GENERATES: two model calls (the constrained narration, and the unconstrained confabulation sample it is
# diffed against -- the latter is never returned by the endpoint at all, per narrate.py's trap guard, so there
# is nothing here that could render it even by accident).
#
# format_narrate() is factored out as a pure function (JSON in, text out), exactly like format_explain(), so
# it is testable with a canned /narrate dict -- no server, no model, no GPU.

def _fetch_narrate(port: int, run_id: str) -> dict:
    """POST /runs/<id>/narrate on the Studio backend -- M4's accountable-self narration (clozn.receipts.narrate).
    Unlike _fetch_explain (M1, free), this generates -- two model calls -- so it gets a longer timeout. A
    clean CloznError (one line, no traceback) when the gateway isn't up, the run doesn't resolve, or its
    model worker isn't ready (503), matching _fetch_explain's error style exactly."""
    from clozn.cli import main as ctx
    url = f"http://127.0.0.1:{port}/runs/{run_id}/narrate"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ctx.CloznError(f"narrate failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise ctx.CloznError(
            f"couldn't reach the Clozn gateway on port {port} ({e.reason}). "
            "Start it first:  clozn serve <model>"
        )
    except Exception as e:
        raise ctx.CloznError(f"narrate failed: {e}")


def _format_narration(cn: dict) -> list[str]:
    out = [f"{fmt.BOLD}why did it say this?{fmt.RST}  "
           f"{fmt.DIM}receipt-constrained -- never the raw self-report (M4){fmt.RST}"]
    narration = cn.get("narration")
    narration = narration.strip() if isinstance(narration, str) else ""
    if narration:
        out.append(f"  {narration}")
    else:
        out.append(f"  {fmt.DIM}no receipt-backed narration was produced for this reply -- with a thin or "
                   f"empty record, that's a complete and honest answer, not a failure.{fmt.RST}")
    return out


def _format_flags(flags: list) -> list[str]:
    flags = [f for f in flags if isinstance(f, str) and f]
    out = [f"{fmt.BOLD}caught in the diff{fmt.RST}  {fmt.DIM}claimed with no receipt to back it{fmt.RST}"]
    if not flags:
        out.append(f"  {fmt.DIM}no unsupported claims flagged this time.{fmt.RST}")
        return out
    for f in flags:
        # reuse the file's own warm/confidence palette (the "wavered" end of the denoise ramp) for each
        # flag -- a flagged claim IS the same kind of thing a low-confidence token is: a place the honest
        # record does not back up. No-op (plain text) when COLOR is off, exactly like every other _paint call.
        out.append(f"  {fmt._paint('⚠ ' + f, 0.0)}")
    return out


def format_narrate(obj: dict) -> str:
    """The M4 /narrate response object (POST /runs/<id>/narrate's JSON body -- exactly narrate.narrate()'s
    four keys: constrained_narration, flags, unsupported_claims, note) -> the terminal render. Pure: no I/O,
    no server, no model -- mirrors format_explain()'s contract exactly, so a canned dict renders identically
    to a live response and is testable without either.

    Renders ONLY what the endpoint can return: the constrained narration prose (the "why"), each flag as a
    visible warning line (the caught confabulations -- a claim the model was about to make that no receipt
    backs), and the note (which matcher ran + its honesty caveat), always shown so the reader knows the
    honesty level. `unsupported_claims` is not re-rendered separately -- every one of its entries is already
    represented, verbatim, inside `flags`. An empty/thin narration is rendered as an honest first-class
    result ("no receipt-backed narration..."), never as an error. Never raises: a malformed section degrades
    to a one-line notice instead of losing the rest, same discipline as format_explain."""
    obj = obj if isinstance(obj, dict) else {}
    lines = [f"{fmt.BOLD}narrate{fmt.RST}  {fmt.DIM}the accountable-self narration -- opt-in, generates (--why){fmt.RST}",
             "-" * 62]
    try:
        lines += _format_narration(fmt._as_dict(obj.get("constrained_narration")))
    except Exception:
        lines += [f"{fmt.BOLD}why did it say this?{fmt.RST}", f"  {fmt.DIM}couldn't render the narration{fmt.RST}"]
    lines.append("")
    try:
        lines += _format_flags(fmt._as_list(obj.get("flags")))
    except Exception:
        lines += [f"{fmt.BOLD}caught in the diff{fmt.RST}", f"  {fmt.DIM}couldn't render the flags{fmt.RST}"]
    lines.append("")
    try:
        note = obj.get("note")
        if isinstance(note, str) and note:
            lines.append(f"{fmt.DIM}{note}{fmt.RST}")
    except Exception:
        pass
    lines.append("-" * 62)
    return "\n".join(lines)


# --------------------------------------------------------------------- prove (M2 display; core.py assembles)
# `clozn prove <run_id>` renders the Studio's already-shipped POST /runs/<id>/receipts (clozn.receipts.core.
# prove_all: leave-one-out receipts + the pairwise redundancy guard) as a terminal view -- this route
# previously had NO CLI front door at all (only the Studio UI / a raw curl could reach it). The opt-in
# `--coalitions` flag additionally requests the Phase-8-tail coalition/Shapley credit report
# (clozn.receipts.coalition) alongside it; never changes the default receipts shape when omitted.

def _fetch_prove(port: int, run_id: str, *, mode: str, coalitions: bool, coalitions_batch: str) -> dict:
    """POST /runs/<id>/receipts on the Studio backend -- regenerates both-arms-greedy for every fired
    influence (and, opted in, the coalition/Shapley arms), so it gets the same longer timeout as
    _fetch_narrate. A clean CloznError (one line, no traceback) when the gateway isn't up, the run doesn't
    resolve, or its model worker isn't ready (503), matching this file's other _fetch_* helpers."""
    from clozn.cli import main as ctx
    url = f"http://127.0.0.1:{port}/runs/{run_id}/receipts"
    body = json.dumps({"mode": mode, "coalitions": bool(coalitions),
                       "coalitions_batch": coalitions_batch}).encode("utf-8")
    req = urllib.request.Request(url, data=body, method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise ctx.CloznError(f"prove failed ({e.code}): {msg}")
    except urllib.error.URLError as e:
        raise ctx.CloznError(
            f"couldn't reach the Clozn gateway on port {port} ({e.reason}). "
            "Start it first:  clozn serve <model>"
        )
    except Exception as e:
        raise ctx.CloznError(f"prove failed: {e}")


def _format_receipt_line(rec: dict) -> str:
    inf = rec.get("influence") or {}
    label = f"card {inf.get('card_id')}" if inf.get("card_id") else (
        f"section {inf.get('section')}" if inf.get("section") else "influence")
    tag = "changed" if rec.get("has_effect") else "no effect"
    verified = "" if rec.get("causal_verified") else "  (not verified -- see ablation_note)"
    return f"  [{tag}] {label}{verified}"


def format_prove(out: dict) -> str:
    """The M2 prove-all response object -> the terminal render. Pure: no I/O, no server, no model --
    mirrors format_explain/format_narrate's contract exactly, so a canned dict renders identically to a
    live response. Never raises: a malformed section degrades to a one-line notice instead of losing the
    rest, same discipline as this file's other format_* functions."""
    out = out if isinstance(out, dict) else {}
    lines = [f"{fmt.BOLD}prove{fmt.RST}  run {out.get('run_id') or '?'}  {fmt.DIM}leave-one-out + redundancy "
            f"guard{fmt.RST}", "-" * 62]
    try:
        receipts = [r for r in fmt._as_list(out.get("receipts")) if isinstance(r, dict)]
        if not receipts:
            lines.append(f"  {fmt.DIM}no fired influences on this run to prove{fmt.RST}")
        else:
            lines.extend(_format_receipt_line(r) for r in receipts)
        for pair in fmt._as_list(out.get("redundant_pairs")):
            if isinstance(pair, dict):
                lines.append(f"  {fmt.DIM}redundant pair: {pair.get('redundant')} -- {pair.get('note', '')}{fmt.RST}")
    except Exception:
        lines.append(f"  {fmt.DIM}couldn't render the receipts{fmt.RST}")
    coalitions = out.get("coalitions")
    if isinstance(coalitions, dict):
        lines.append("")
        try:
            from clozn.receipts.coalition import format_report
            lines.append(format_report(coalitions))
        except Exception:
            lines.append(f"  {fmt.DIM}couldn't render the coalition/Shapley report{fmt.RST}")
    lines.append("-" * 62)
    return "\n".join(lines)


def cmd_prove(args):
    from clozn.cli import main as ctx
    rid = _last_run_id() if args.last else args.run_id
    if not rid:
        raise ctx.CloznError("give a run id, or pass --last for the most recent one "
                             "(see ids in the Studio's Runs list, or run something first:  clozn run qwen \"...\")")
    port = args.port or 8080
    out = _fetch_prove(port, rid, mode=args.mode, coalitions=args.coalitions,
                       coalitions_batch=args.coalitions_batch)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(format_prove(out))


def cmd_explain(args):
    from clozn.cli import main as ctx
    rid = _last_run_id() if args.last else args.run_id
    if not rid:
        raise ctx.CloznError("give a run id, or pass --last for the most recent one "
                             "(see ids in the Studio's Runs list, or run something first:  clozn run qwen \"...\")")
    port = args.port or 8080
    print(format_explain(_fetch_explain(port, rid)))
    if args.why:
        print()
        print(format_narrate(_fetch_narrate(port, rid)))
