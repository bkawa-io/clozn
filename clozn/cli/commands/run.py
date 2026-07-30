"""commands.run -- `clozn run`: one-shot or interactive (Ollama-style REPL) generation against a model,
reusing a warm `clozn serve` daemon when one is up. Owns the prompting/streaming plumbing (chat templates,
the AR SSE stream, the diffusion non-stream path) and the trace-capture side effect of every turn: each
reply is paired into a replayable per-token trace and written to the SQLite run journal
(clozn.runs.store), which Studio and `clozn trace`/`clozn explain` read.

HOME lives on `clozn.cli.main`; imported INSIDE cmd_run (not at module level) for the same circular-import
reason documented in engine_process.py's module docstring.
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
from clozn.cli.engine_process import _find_warm, _free_port
from clozn.cli.runtime_process import RuntimeConfig, spawn_runtime

SYS = "You are a helpful assistant."


def _chat_session(history, new_user: str, family: str = "qwen") -> str:
    """Build the whole conversation in the family's chat template. BOS is left to the engine (add_bos).
    history: list of (user, assistant) pairs already exchanged; new_user: the current turn."""
    if family == "mistral":
        s = "".join(f"[INST] {u} [/INST] {a}</s>" for u, a in history)
        return s + f"[INST] {new_user} [/INST]"
    if family == "llama3":
        s = f"<|start_header_id|>system<|end_header_id|>\n\n{SYS}<|eot_id|>"
        for u, a in history:
            s += (f"<|start_header_id|>user<|end_header_id|>\n\n{u}<|eot_id|>"
                  f"<|start_header_id|>assistant<|end_header_id|>\n\n{a}<|eot_id|>")
        return s + (f"<|start_header_id|>user<|end_header_id|>\n\n{new_user}<|eot_id|>"
                    f"<|start_header_id|>assistant<|end_header_id|>\n\n")
    if family == "gemma":
        s = "".join(f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n{a}<end_of_turn>\n"
                    for u, a in history)
        return s + f"<start_of_turn>user\n{new_user}<end_of_turn>\n<start_of_turn>model\n"
    s = f"<|im_start|>system\n{SYS}<|im_end|>\n"
    for u, a in history:
        s += f"<|im_start|>user\n{u}<|im_end|>\n<|im_start|>assistant\n{a}<|im_end|>\n"
    return s + f"<|im_start|>user\n{new_user}<|im_end|>\n<|im_start|>assistant\n"


def _chat_wrap(prompt: str, family: str = "qwen") -> str:
    return _chat_session([], prompt, family)


def stream_ar(port: int, prompt: str, max_tokens: int, heat: bool = False):
    """POST /api/clozn/generate (stream); print committed tokens and retain native events.

    heat=True paints each token as it lands by its confidence (the denoise heatmap, live); False (default)
    is the plain, byte-for-byte-unchanged stream. Painting also no-ops when color is off (piped/NO_COLOR),
    so `--heat | cat` is still clean text.

    The engine emits, per token, a `tokens_committed` frame (the chosen piece + its confidence) and a
    `step_lens` frame (the top-k it weighed), plus a `gen_finished` frame (the real stop cause: eos vs.
    length) once done. We pair the per-token frames by position into a replayable trace -- the raw
    material for the timeline: where was it uncertain, and what did it almost say -- and also pluck the
    stop cause so CLI runs get a real `finish_reason` in the run journal, same as Studio's chat path.
    -> (token count, steps, finish_reason, prompt_tokens, public_text, worker_timing)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens, "stream": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/clozn/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    n = 0
    frames = []                                             # every parsed SSE frame, for the shared accumulator
    from clozn.runs.think_tags import ThinkTagStream, prompt_opens_think
    think_stream = ThinkTagStream(implicit_open=prompt_opens_think(prompt))
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except Exception:
                continue
            frames.append(obj)
            if obj.get("type") == "tokens_committed":       # print live as tokens land
                for it in obj.get("items", []):
                    public_piece = think_stream.feed(it.get("piece", ""))
                    if public_piece:
                        sys.stdout.write(fmt._stream_token(public_piece, it.get("conf"), heat))
                        sys.stdout.flush()
                    n += 1
    tail, think_result = think_stream.finish()
    if tail:
        sys.stdout.write(fmt._stream_token(tail, None, heat))
        sys.stdout.flush()
    # Accumulation (pair tokens_committed with step_lens by position) lives in runlog so the CLI and the
    # engine-chat capture share ONE tested implementation. Fall back to a local pairing if the import fails
    # -- the stdlib CLI must never break on a missing sibling.
    try:
        import clozn.runs.store as runlog
        steps = runlog.accumulate_ar_events(frames)
        finish = runlog.finish_reason_from_frames(frames)
        generation_timing = runlog.generation_timing_from_frames(frames)
    except Exception:
        by_pos: dict = {}
        for obj in frames:
            if obj.get("type") == "tokens_committed":
                for it in obj.get("items", []):
                    by_pos[it.get("pos")] = {"pos": it.get("pos"), "index": it.get("pos"),
                                             "token_id": it.get("id"), "piece": it.get("piece", ""),
                                             "prob": round(float(it.get("conf", 0.0)), 3),
                                             "conf": round(float(it.get("conf", 0.0)), 3), "alts": []}
            elif obj.get("type") == "step_lens":
                step = by_pos.get((obj.get("positions") or [None])[0])
                if step:
                    pieces, probs = obj.get("pieces", []), obj.get("probs", [])
                    ids = obj.get("ids") or [None] * len(pieces)
                    chosen_id = step.get("token_id")
                    step["alts"] = [{"token_id": tid, "piece": p, "prob": round(float(pr), 3)}
                                    for tid, p, pr in zip(ids, pieces, probs)
                                    if (chosen_id is None or tid != chosen_id) and p != step["piece"]][:3]
        steps = [by_pos[p] for p in sorted(by_pos, key=lambda x: (x is None, x))]
        finish = None
        for obj in frames:
            if obj.get("type") == "gen_finished" and isinstance(obj.get("reason"), str):
                finish = "stop" if obj["reason"] == "eos" else "length"
        generation_timing = {}
    prompt_tokens = next((obj.get("prompt_tokens") for obj in frames
                          if obj.get("type") == "gen_started"
                          and isinstance(obj.get("prompt_tokens"), int)), None)
    routing = next((obj.get("clozn_model_routing") for obj in frames
                    if obj.get("type") == "model_routing"
                    and isinstance(obj.get("clozn_model_routing"), dict)), None)
    if routing is not None:
        generation_timing = dict(generation_timing or {})
        generation_timing["model_routing"] = routing
    return n, steps, finish, prompt_tokens, think_result.public_text, generation_timing


def _complete_once_raw(port: int, prompt: str, max_tokens: int) -> dict:
    """Non-streaming /api/clozn/generate -- the raw completion response (choices[0] carries text + the
    engine's real finish_reason)."""
    body = json.dumps({"prompt": prompt, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/api/clozn/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        return json.loads(resp.read())


def complete_once(port: int, prompt: str, max_tokens: int) -> str:
    """Non-streaming native generation (used for diffusion, which commits out of reading order)."""
    r = _complete_once_raw(port, prompt, max_tokens)
    return (r.get("choices") or [{}])[0].get("text", "")


def _run_turn(port, mode, text, max_tokens, gpu, model_name, prompt_for_trace, heat=False):
    """One generation: stream (AR, auto-saving the trace) or denoise (diffusion). Prints stats. -> response."""
    g0 = time.time()
    steps = []
    finish = None
    prompt_tokens = None
    generation_timing = {}
    native_routing = None
    if mode == "autoregressive":
        streamed = stream_ar(port, text, max_tokens, heat=heat)
        # Keep the helper seam friendly to older/custom stream implementations that return the original
        # (count, steps, finish) tuple. The built-in path returns the richer prompt-count/public-answer
        # values needed for context receipts and think-tag hygiene.
        if len(streamed) == 3:
            n, steps, finish = streamed
            public_resp = "".join(str(step.get("piece") or "") for step in steps)
        elif len(streamed) == 5:
            n, steps, finish, prompt_tokens, public_resp = streamed
        else:
            n, steps, finish, prompt_tokens, public_resp, generation_timing = streamed
        sys.stdout.write("\n")
        if heat and fmt.COLOR:                             # a legend + how many tokens wavered, after the reply
            lows = sum(1 for s in steps if fmt._num(s.get("conf")) < 0.5)
            print(f"{fmt._conf_legend()}   {fmt.DIM}{lows} wavered{fmt.RST}", file=sys.stderr)
        raw_resp = "".join(s["piece"] for s in steps)
        resp = public_resp.strip()
    else:
        r = _complete_once_raw(port, text, max_tokens)
        raw_resp = str((r.get("choices") or [{}])[0].get("text", ""))
        from clozn.runs.think_tags import prompt_opens_think, sanitize_reply
        resp = sanitize_reply(raw_resp, implicit_open=prompt_opens_think(text)).public_text.strip()
        finish = (r.get("choices") or [{}])[0].get("finish_reason")
        prompt_tokens = (r.get("usage") or {}).get("prompt_tokens")
        native_routing = r.get("clozn_model_routing")
        print(resp); n = len(resp.split())
    dt = time.time() - g0
    rate = f", ~{n/dt:.0f} tok/s" if dt > 0 and mode == "autoregressive" else ""
    # A "length" finish means the engine stopped because it hit --max, not because the model was done:
    # say so, or the reply just appears to end mid-sentence for no visible reason.
    cut = f" - cut off at --max {max_tokens}; rerun with --max {max_tokens * 4}" if finish == "length" else ""
    print(f"{fmt.DIM}- {n} tok in {dt:.1f}s{rate} - {'GPU' if gpu else 'CPU'}{cut}{fmt.RST}", file=sys.stderr)
    # every CLI turn becomes an inspectable run -- finish_reason mirrors Studio's chat path (the engine's
    # real stop cause: "stop" on eos, "length" on truncation), not left null like before this fix.
    # Keep the user-authored message readable in run.messages, but ALSO persist `text`: this is the
    # exact, already-rendered string sent to /api/clozn/generate (including the family chat template and,
    # in REPL mode, prior turns).  Gate 0 requires the journal to record what the model actually saw;
    # replacing messages with template syntax would satisfy that mechanically while making the ordinary
    # run view worse, so runlog's purpose-built final_prompt field carries the exact wire input instead.
    run_meta = {"max_tokens": int(max_tokens), "prompt_tokens": prompt_tokens}
    run_meta.update(dict(generation_timing or {}))
    if isinstance(native_routing, dict):
        run_meta["model_routing"] = native_routing
    _log_run_cli(model_name, prompt_for_trace, raw_resp, steps, g0, finish_reason=finish, port=port,
                 final_prompt=text, meta=run_meta)
    return resp


# identity block per engine port, fetched once per process (the /health hit is cheap but there is no
# reason to repeat it every REPL turn; the engine's model_sha256 is fixed for the process lifetime).
_IDENTITY_BY_PORT: dict = {}


def _identity_for_port(port):
    """The run-record `identity` block for the engine on `port` (clozn.runs.identity), from the engine's
    own /health (which carries model path + the boot-time model_sha256) + a canonical /apply_template
    rendering for the template fingerprint. Mirrors the gateway wiring (server/substrates.identity_meta)
    so CLI runs stop being the audit's 'narrower journal record'. Never raises; {} when anything is
    unavailable -- absence stays visible, never faked."""
    if port in _IDENTITY_BY_PORT:
        return _IDENTITY_BY_PORT[port]
    ident: dict = {}
    try:
        import clozn.runs.identity as rid
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            health = json.loads(r.read())

        def _apply(messages):
            body = json.dumps({"messages": list(messages), "add_assistant": True}).encode()
            req = urllib.request.Request(f"http://127.0.0.1:{port}/apply_template", data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as rr:
                return json.loads(rr.read()).get("prompt", "")

        ident = rid.runtime_identity(model_path=health.get("model"),
                                     model_sha256_hint=health.get("model_sha256"),
                                     apply_template_fn=_apply, engine_health=health)
    except Exception:
        ident = {}
    _IDENTITY_BY_PORT[port] = ident
    return ident


def _log_run_cli(model_name, prompt, resp, steps, started, finish_reason=None, port=None,
                 final_prompt=None, meta=None):
    """Write this CLI turn to the Run Log so `clozn run`/REPL turns show up in the Studio alongside chats.
    runlog.py lives in clozn.runs (a sibling of this stdlib-only CLI) and is itself stdlib-only, so we
    import it directly. Logging must NEVER break a run -- swallow everything.

    PROMPT-SECTION INFLUENCE (foundation) -- an honest divergence worth stating plainly: the natural home
    for this would be `/api/clozn/generate` itself, but that route (generation_gateway.native_completion)
    is a transparent proxy straight to the C++ engine's own `/v1/completions` -- it never calls
    runlog.record() at all. THIS function is the only place a native/CLI generation actually gets
    persisted, so it's where the section manifest has to be built instead. There's no per-request JSON
    body here (this is a Python call, not an HTTP handler) and so no way for a caller to pass explicit char
    ranges -- every CLI/native run gets the deterministic auto-chunker only, over `prompt` (the exact
    string sent to the engine; also recorded verbatim as `final_prompt`, since a raw-prompt run's section
    offsets are into that field, not a chat `messages` list -- see clozn.runs.sections's docstring)."""
    try:
        import clozn.runs.store as runlog
        from clozn.runs import sections as clozn_sections
        # stream_ar hands us per-token steps ({piece, conf, alts}); runlog owns the steps->trace mapping so
        # the on-disk trace schema stays one contract shared with the engine-chat capture (issue B3).
        trace = runlog.steps_to_trace(steps)
        identity = _identity_for_port(port) if port else {}
        messages = [{"role": "user", "content": prompt}]
        try:
            manifest = clozn_sections.auto_chunk_prompt(prompt)
        except Exception:
            manifest = []
        runlog.record(source="cli", client="cli", model=model_name, substrate="engine",
                      messages=messages, response=resp, trace=trace, started=started,
                      finish_reason=finish_reason, identity=identity,
                      assembled_messages=messages,
                      final_prompt=final_prompt if final_prompt is not None else prompt,
                      meta={k: v for k, v in (meta or {}).items() if v is not None},
                      sections=manifest)
    except Exception:
        pass


# --------------------------------------------------------------------- --show-influence / REPL /influence
# Quick, opt-in read of the prompt-section influence fast path (clozn.server.routes.section_influence) for
# the LAST run this process just wrote -- approximate, teacher-forced, never causal proof (the route's own
# `note` field says so; this CLI just relays it). Both call sites below share the same two helpers so a
# one-shot `--show-influence` and the REPL's `/influence` render byte-identically.

def _last_run_id():
    """The most recent run in the shared Studio run log, read directly -- mirrors clozn.cli.commands.
    explain's own private `_last_run_id` (duplicated here, not imported: explain.py imports FROM this
    module already, so importing back would be circular -- see this module's docstring on why HOME/
    CloznError are imported inside functions for the same reason). include_replays=False so an internal
    receipts/replay re-generation (source="replay") never outranks the turn the user actually just ran."""
    try:
        import clozn.runs.store as runlog
        runs = runlog.list_runs(limit=1, include_replays=False)
        return runs[0]["id"] if runs else None
    except Exception:
        return None


def _fetch_section_influence(port: int, run_id: str) -> dict:
    """POST /runs/<id>/section-influence on the Studio gateway (mirrors commands.explain._fetch_explain's
    request shape exactly: same bare `{}` body, same URL pattern). Unlike _fetch_explain/_fetch_narrate --
    which raise a CloznError that aborts the command with a clean one-line error -- this is an opt-in
    ADD-ON to an ALREADY-successful `clozn run` turn: a gateway that's down, one too old to have this
    route yet, or a run with no section manifest must never turn a successful generation into a failed
    command. So every failure degrades to a plain {"error": "..."} dict here instead of raising, and
    _format_influence_line renders that as one honest line, never a stack trace."""
    url = f"http://127.0.0.1:{port}/runs/{run_id}/section-influence"
    req = urllib.request.Request(url, data=b"{}", method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {"error": "section-influence isn't available on this gateway (old build?)"}
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        return {"error": f"section-influence failed ({e.code}): {msg}"}
    except urllib.error.URLError as e:
        return {"error": f"couldn't reach the Clozn gateway on port {port} ({e.reason})"}
    except Exception as e:
        return {"error": f"section-influence failed: {e}"}


def _format_influence_line(payload: dict) -> str:
    """The section-influence response -> one compact terminal line:
        - sections: rag_context (42% influence), few_shot (8%), auto_3 (3%)  [approximate, teacher-forced]
    Pure function (dict in, text out) -- testable with a canned payload, no server, no model, no I/O --
    mirrors format_explain's contract. Ranked by |influence_share| descending (the biggest measured effect
    leads); only the FIRST entry spells out "influence" (matches the fixed contract's own worked example),
    the rest just show the percentage. Degrades to one honest line on any missing/error/empty shape -- a
    down gateway, an old gateway with no route, a run with no sections manifest -- never a stack trace,
    never silence."""
    payload = payload if isinstance(payload, dict) else {}
    if payload.get("error"):
        return f"- sections: {payload['error']}"
    sections = [s for s in (payload.get("sections") or []) if isinstance(s, dict)]
    if not sections:
        return f"- sections: {payload.get('note') or 'no sections on this run'}"
    ranked = sorted(sections, key=lambda s: -abs(fmt._num(s.get("influence_share"))))
    bits = []
    for i, s in enumerate(ranked):
        name = s.get("name") or s.get("id") or "?"
        pct = round(abs(fmt._num(s.get("influence_share"))) * 100)
        bits.append(f"{name} ({pct}% influence)" if i == 0 else f"{name} ({pct}%)")
    return f"- sections: {', '.join(bits)}  [approximate, teacher-forced]"


def _print_influence(port: int):
    """The shared render step for --show-influence and /influence: resolve the last run, fetch, print to
    stderr (same channel as this file's other post-turn diagnostics, e.g. the '- N tok in Xs' line) --
    never raises, since a nice-to-have inspection line must never take down the turn that already
    succeeded."""
    rid = _last_run_id()
    if not rid:
        print(f"{fmt.DIM}- sections: no run recorded yet to inspect{fmt.RST}", file=sys.stderr)
        return
    print(f"{fmt.DIM}{_format_influence_line(_fetch_section_influence(port, rid))}{fmt.RST}", file=sys.stderr)


def _repl(port, mode, flags, fam, gpu, model, max_tokens, heat=False):
    """Interactive chat loop on a warm engine (Ollama-style). /reset clears, /bye quits."""
    name = _friendly(model)
    is_chat = flags.get("chat") and mode == "autoregressive"
    tty = sys.stdin.isatty()
    print(f"{fmt.DIM}chat with {name}  -  /reset clears history, /bye quits, "
          f"/influence shows the last turn's per-section influence{fmt.RST}", file=sys.stderr)
    history = []
    while True:
        if tty:
            try:
                msg = input("\nyou> ").strip()
            except EOFError:
                break
        else:
            line = sys.stdin.readline()
            if not line:
                break
            msg = line.strip()
        if not msg:
            continue
        if msg in ("/bye", "/exit", "/quit"):
            break
        if msg == "/reset":
            history = []; print(f"{fmt.DIM}(history cleared){fmt.RST}", file=sys.stderr); continue
        if msg == "/influence":
            _print_influence(port); continue
        text = (_chat_session(history, msg, fam) if is_chat
                else "".join(f"{u}\n{a}\n" for u, a in history) + msg)
        sys.stdout.write(f"{fmt.BOLD}{name}>{fmt.RST} ")
        resp = _run_turn(port, mode, text, max_tokens, gpu, name, msg, heat=heat)
        history.append((msg, resp))
    print(f"{fmt.DIM}bye{fmt.RST}", file=sys.stderr)


def cmd_run(args):
    from clozn.cli import main as ctx
    model = resolve_model(args.model)
    print(f"{fmt.DIM}- model: {model}{fmt.RST}", file=sys.stderr, flush=True)
    flags = _flags_for(model)
    if args.mask is not None:
        flags["mask"] = args.mask
    if args.eos is not None:
        flags["eos"] = args.eos
    if args.ctx is not None:
        flags["ctx"] = args.ctx
    fam = flags.get("tmpl", "qwen")
    # An explicit context window is part of the requested runtime shape. Reuse a warm server only when
    # its live /readyz worker reports the same n_ctx; silently borrowing a different context would make
    # --ctx look accepted while ignoring it.
    warm = None if args.cpu else _find_warm(model, args.ctx)
    stack = worker_log = gateway_log = None
    if warm:
        port, gpu, mode = warm
        print(f"{fmt.DIM}- {_friendly(model)} warm on port {port} ({'GPU' if gpu else 'CPU'}, {mode}){fmt.RST}",
              file=sys.stderr, flush=True)
    else:
        os.makedirs(ctx.HOME, exist_ok=True)
        worker_log = open(os.path.join(ctx.HOME, "worker-run.log"), "w", encoding="utf-8")
        gateway_log = open(os.path.join(ctx.HOME, "gateway-run.log"), "w", encoding="utf-8")
        port = args.port or _free_port()
        print(f"{fmt.DIM}- loading {_friendly(model)} …{fmt.RST}", file=sys.stderr, flush=True)
        t0 = time.time()
        stack = spawn_runtime(
            RuntimeConfig(model=model, public_port=port, flags=flags, prefer_gpu=not args.cpu),
            worker_log=worker_log,
            gateway_log=gateway_log,
        )
        health, gpu = stack.worker_health, stack.gpu
        mode = health.get("mode", "?")
        print(f"{fmt.DIM}- {_friendly(model)} on {'GPU' if gpu else 'CPU'} build ({mode}), "
              f"ready in {time.time()-t0:.1f}s{fmt.RST}", file=sys.stderr, flush=True)
    try:
        if args.prompt is not None:                            # one-shot
            is_chat = flags.get("chat") and mode == "autoregressive"
            text = _chat_wrap(args.prompt, fam) if is_chat else args.prompt
            _run_turn(port, mode, text, args.max, gpu, _friendly(model), args.prompt, heat=args.heat)
            if mode == "autoregressive":
                print(f"{fmt.DIM}  next: run `clozn trace` to inspect where it was uncertain + "
                      f"what it almost said{fmt.RST}", file=sys.stderr)
            if args.show_influence:
                _print_influence(port)
        else:                                                  # interactive REPL (Ollama-style)
            _repl(port, mode, flags, fam, gpu, model, args.max, heat=args.heat)
    finally:
        if stack:                                      # leave a warm `serve` stack up; stop only our temporary one
            stack.stop()
        if worker_log:
            worker_log.close()
        if gateway_log:
            gateway_log.close()
