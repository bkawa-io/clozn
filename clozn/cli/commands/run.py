"""The retained ``clozn run`` text client.

The command uses the same OpenAI Chat Completions route as Studio and other API
clients. The gateway owns prompt templating, trace capture, and durable recording;
the CLI prints the response and reports the returned run id.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

from clozn.cli import formatting as fmt
from clozn.cli.commands.models import _flags_for, _friendly, resolve_model
from clozn.cli.engine_process import _find_warm, _free_port
from clozn.cli.runtime_process import RuntimeConfig, spawn_runtime


SYS = "You are a helpful assistant."


def _messages(history: list[tuple[str, str]], new_user: str) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYS}]
    for user, assistant in history:
        messages.extend((
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ))
    messages.append({"role": "user", "content": new_user})
    return messages


def _chat_request(port: int, body: dict):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(request, timeout=600)


def stream_chat(port: int, messages: list[dict[str, str]], max_tokens: int) -> dict:
    """Stream one recorded Chat Completions request and print its public text."""
    body = {
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    pieces: list[str] = []
    finish_reason = None
    run_id = None
    usage = None
    with _chat_request(port, body) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                frame = json.loads(payload)
            except json.JSONDecodeError:
                continue
            error = frame.get("error") if isinstance(frame, dict) else None
            if error:
                if isinstance(error, dict):
                    error = error.get("message") or error
                raise RuntimeError(str(error))
            if isinstance(frame.get("usage"), dict):
                usage = dict(frame["usage"])
            choices = frame.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            choice = choices[0] if isinstance(choices[0], dict) else {}
            delta = choice.get("delta") if isinstance(choice.get("delta"), dict) else {}
            piece = delta.get("content")
            if isinstance(piece, str) and piece:
                pieces.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
            if isinstance(choice.get("finish_reason"), str):
                finish_reason = choice["finish_reason"]
                candidate = frame.get("clozn_run_id")
                if isinstance(candidate, str) and candidate:
                    run_id = candidate
    if not run_id:
        raise RuntimeError("generation finished without a durable Clozn run id")
    return {
        "text": "".join(pieces),
        "finish_reason": finish_reason,
        "run_id": run_id,
        "usage": usage or {},
    }


def complete_once(port: int, prompt: str, max_tokens: int) -> str:
    """Small non-streaming helper retained for internal callers during migration."""
    with _chat_request(port, {
        "messages": _messages([], prompt),
        "max_tokens": int(max_tokens),
        "temperature": 0,
    }) as response:
        payload = json.loads(response.read())
    return str(((payload.get("choices") or [{}])[0].get("message") or {}).get("content") or "")


def _run_turn(port: int, messages: list[dict[str, str]], max_tokens: int,
              gpu: bool, model_name: str) -> str:
    started = time.time()
    result = stream_chat(port, messages, max_tokens)
    sys.stdout.write("\n")
    usage = result["usage"]
    count = usage.get("completion_tokens")
    count_text = f"{count} tok" if isinstance(count, int) else "completed"
    elapsed = time.time() - started
    cutoff = (
        f" - cut off at --max {max_tokens}; rerun with --max {max_tokens * 4}"
        if result["finish_reason"] == "length" else ""
    )
    print(
        f"{fmt.DIM}- {count_text} in {elapsed:.1f}s - {'GPU' if gpu else 'CPU'}"
        f" - run {result['run_id']}{cutoff}{fmt.RST}",
        file=sys.stderr,
    )
    return result["text"].strip()


def _repl(port: int, gpu: bool, model: str, max_tokens: int):
    """Interactive chat loop. ``/reset`` clears history and ``/bye`` exits."""
    name = _friendly(model)
    tty = sys.stdin.isatty()
    print(
        f"{fmt.DIM}chat with {name} - /reset clears history, /bye quits{fmt.RST}",
        file=sys.stderr,
    )
    history: list[tuple[str, str]] = []
    while True:
        if tty:
            try:
                message = input("\nyou> ").strip()
            except EOFError:
                break
        else:
            line = sys.stdin.readline()
            if not line:
                break
            message = line.strip()
        if not message:
            continue
        if message in ("/bye", "/exit", "/quit"):
            break
        if message == "/reset":
            history = []
            print(f"{fmt.DIM}(history cleared){fmt.RST}", file=sys.stderr)
            continue
        sys.stdout.write(f"{fmt.BOLD}{name}>{fmt.RST} ")
        response = _run_turn(port, _messages(history, message), max_tokens, gpu, name)
        history.append((message, response))
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
    if getattr(args, "batch", None) is not None:
        flags["batch"] = args.batch
    if getattr(args, "ubatch", None) is not None:
        flags["ubatch"] = args.ubatch

    warm = None if args.cpu else _find_warm(
        model, args.ctx, getattr(args, "batch", None), getattr(args, "ubatch", None)
    )
    stack = worker_log = None
    if warm:
        port, gpu, mode = warm
        print(
            f"{fmt.DIM}- {_friendly(model)} warm on port {port} "
            f"({'GPU' if gpu else 'CPU'}, {mode}){fmt.RST}",
            file=sys.stderr,
            flush=True,
        )
    else:
        os.makedirs(ctx.HOME, exist_ok=True)
        worker_log = open(os.path.join(ctx.HOME, "worker-run.log"), "w", encoding="utf-8")
        port = args.port or _free_port()
        print(f"{fmt.DIM}- loading {_friendly(model)} …{fmt.RST}", file=sys.stderr, flush=True)
        started = time.time()
        stack = spawn_runtime(
            RuntimeConfig(model=model, public_port=port, flags=flags, prefer_gpu=not args.cpu),
            worker_log=worker_log,
        )
        gpu = stack.gpu
        mode = stack.worker_health.get("mode", "?")
        print(
            f"{fmt.DIM}- {_friendly(model)} on {'GPU' if gpu else 'CPU'} build ({mode}), "
            f"ready in {time.time() - started:.1f}s{fmt.RST}",
            file=sys.stderr,
            flush=True,
        )
    try:
        if args.prompt is not None:
            _run_turn(
                port,
                _messages([], args.prompt),
                args.max,
                gpu,
                _friendly(model),
            )
        else:
            _repl(port, gpu, model, args.max)
    finally:
        if stack:
            stack.stop()
        if worker_log:
            worker_log.close()
