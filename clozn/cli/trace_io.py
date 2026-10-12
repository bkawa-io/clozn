"""Terminal rendering bridge for traces stored in the canonical SQLite run journal."""
from __future__ import annotations

from clozn.cli import formatting as fmt


def _render_trace(meta: dict, steps: list):
    m = meta or {}
    steps = [s for s in (steps or []) if isinstance(s, dict)]
    print(f"{fmt.BOLD}{m.get('model', '?')}{fmt.RST}  \"{m.get('prompt', '')[:64]}\"")
    print(f"{fmt.DIM}{m.get('n', len(steps))} tokens - {m.get('backend', '?')} - short bar = less sure; "
          f"'almost' = what it nearly said{fmt.RST}")
    if not steps:
        print("-" * 62)
        print(f"{fmt.DIM}no per-token trace recorded on this run{fmt.RST}")
        return
    # the reply reconstructed from the trace, each token painted by its confidence -- the denoise board, in
    # the terminal. Then the per-token detail below (piece also painted, so the two views share one palette).
    hm = fmt._heatmap_lines([(s.get("piece", s.get("text", "")),
                             fmt._num(s.get("prob", s.get("conf", s.get("confidence"))))) for s in steps])
    if hm:
        print("-" * 62)
        for ln in hm:
            print(ln)
        print(fmt._conf_legend())
    print("-" * 62)
    for i, s in enumerate(steps):
        piece = (s.get("piece", s.get("text", "")) or "").replace("\n", "\\n").replace("\t", "\\t")
        conf = fmt._num(s.get("prob", s.get("conf", s.get("confidence"))))
        mark = " " if conf >= 0.5 else "?"
        shown = piece[:16]
        cell = fmt._paint(shown, conf) + " " * max(0, 16 - len(shown))
        idx = s.get("index", s.get("pos", i))
        meta_bits = []
        if s.get("token_id") is not None:
            meta_bits.append(f"id {s.get('token_id')}")
        if s.get("logprob") is not None:                    # derived: log(confidence), never a separate signal
            meta_bits.append(f"logp {fmt._num(s.get('logprob')):.3f}")
        if s.get("entropy") is not None:                    # true full-distribution entropy (HF/Qwen path only)
            meta_bits.append(f"H {fmt._num(s.get('entropy')):.3f}")
        if s.get("topk_entropy") is not None:                # TOP-K APPROXIMATION only (engine path) -- say so
            meta_bits.append(f"H@k(approx) {fmt._num(s.get('topk_entropy')):.3f}")
        line = f" {mark} {str(idx):>3} {cell} {fmt._confbar(conf)} {conf:.2f}"
        if meta_bits:
            line += f"   {fmt.DIM}{' '.join(meta_bits)}{fmt.RST}"
        if conf < 0.5 and s.get("alts"):
            alts = "  ".join(f"{(a.get('piece', a.get('text', '')) or '').strip() or '_'} "
                              f"{fmt._num(a.get('prob')):.2f}" +
                              (f" id {a.get('token_id')}" if a.get("token_id") is not None else "")
                              for a in s["alts"][:3] if isinstance(a, dict))
            if alts:
                line += f"   {fmt.DIM}almost: {alts}{fmt.RST}"
        print(line)
    lows = [s for s in steps if fmt._num(s.get("prob", s.get("conf", s.get("confidence"))), 1.0) < 0.5]
    print("-" * 62)
    tail = " -> " + ", ".join((s.get("piece", s.get("text", "")) or "").strip() for s in lows[:6]) if lows else ""
    print(f"{fmt.DIM}{len(lows)} uncertain moment(s){tail}{fmt.RST}")


def _import_runlog():
    import clozn.runs.store as runlog
    return runlog


def _run_prompt(run: dict) -> str:
    for msg in reversed(run.get("messages") or []):
        if isinstance(msg, dict) and msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(run.get("prompt_summary", ""))


def _runlog_trace_steps(run: dict) -> list[dict]:
    trace = run.get("trace") if isinstance(run, dict) else {}
    trace = trace if isinstance(trace, dict) else {}
    rich = trace.get("steps") if isinstance(trace.get("steps"), list) else []
    out = []
    for i, s in enumerate(rich):
        if not isinstance(s, dict):
            continue
        piece = s.get("piece", s.get("text", ""))
        conf = s.get("prob", s.get("conf", s.get("confidence")))
        alts = s.get("alts", s.get("alternatives", []))
        step = {"index": s.get("index", s.get("pos", i)), "piece": str(piece),
                "conf": fmt._num(conf, 1.0), "alts": alts if isinstance(alts, list) else []}
        for k in ("token_id", "logprob", "entropy", "topk_entropy", "wall_ms", "dt_ms"):
            if s.get(k) is not None:
                step[k] = s.get(k)
        out.append(step)
    return out


def _runlog_trace_meta(run: dict, steps: list[dict]) -> dict:
    backend = run.get("substrate") or run.get("source") or run.get("client") or "runlog"
    return {"id": run.get("id", ""), "model": run.get("model", ""), "prompt": _run_prompt(run),
            "backend": backend, "n": len(steps), "t": run.get("created_ts")}


def _list_runlog_traces(runlog, limit=12):
    # include_replays=False: skip the internal leave-one-out/redundancy-guard re-generations a
    # `/runs/<id>/receipts` prove-all persists (clozn.replay.replay, source="replay") -- real runs, still
    # fully readable by id, just not something a person actually did, so they shouldn't spam this list.
    rows = runlog.list_runs(limit=limit, include_replays=False)
    print(f"{'WHEN':<19} {'MODEL':<11} {'TOK':>4}  PROMPT")
    for row in reversed(rows):
        run = runlog.get_run(row.get("id", "")) or {}
        trace = run.get("trace") if isinstance(run.get("trace"), dict) else {}
        toks = trace.get("tokens") if isinstance(trace.get("tokens"), list) else []
        when = str(row.get("created_at") or row.get("id", ""))[:19]
        print(f"{when:<19} {str(row.get('model', ''))[:11]:<11} {len(toks):>4}  "
              f"{str(row.get('prompt_summary', ''))[:46]}")
