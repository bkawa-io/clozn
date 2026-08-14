#!/usr/bin/env python3
"""Measure worker context/batch sizing and llama.cpp allocation breakdowns.

This is an opt-in live-GGUF harness. It starts one private worker at a time, waits for
``/health``, records the effective ``n_ctx``/``n_batch``/``n_ubatch`` and the vendored
llama.cpp memory breakdown, then stops only the process it started. With
``--probe-prompt-file`` it also submits one non-streaming completion per context so the
reported shape is exercised after startup.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import time
import urllib.error
import urllib.request


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _get(url: str, timeout: float) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def _post(url: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _wait_health(process: subprocess.Popen, port: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last_error = "worker did not answer /health"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"worker exited with code {process.returncode}: {last_error}")
        try:
            health = _get(f"http://127.0.0.1:{port}/health", 2.0)
            if health.get("status") == "ok":
                return health
            last_error = f"unexpected health response: {health}"
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as error:
            last_error = str(error)
        time.sleep(0.25)
    raise TimeoutError(last_error)


def _run_one(args: argparse.Namespace, context: int, prompt: str | None) -> dict:
    port = _free_port()
    command = [
        args.executable,
        args.model,
        "--port", str(port),
        "--host", "127.0.0.1",
        "--ctx", str(context),
        "--ar",
    ]
    if args.gpu_layers is not None:
        command += ["--gpu-layers", str(args.gpu_layers)]
    if args.batch is not None:
        command += ["--batch", str(args.batch)]
    if args.ubatch is not None:
        command += ["--ubatch", str(args.ubatch)]

    log_path = Path(args.log_dir) / f"context-{context}.log" if args.log_dir else None
    log_handle = log_path.open("w", encoding="utf-8") if log_path else subprocess.DEVNULL
    process = subprocess.Popen(command, stdout=log_handle, stderr=log_handle, text=True)
    try:
        health = _wait_health(process, port, args.startup_timeout)
        row = {
            "context_requested": context,
            "port": port,
            "health": health,
            "n_ctx": health.get("n_ctx"),
            "n_batch": health.get("n_batch"),
            "n_ubatch": health.get("n_ubatch"),
            "memory": health.get("memory", {}),
        }
        memory = health.get("memory") or {}
        if isinstance(memory, dict) and memory.get("compute_bytes") is not None:
            row["compute_gib"] = memory["compute_bytes"] / (1024 ** 3)
        if prompt is not None:
            started = time.perf_counter()
            response = _post(
                f"http://127.0.0.1:{port}/v1/completions",
                {"prompt": prompt, "max_tokens": args.probe_max_tokens,
                 "temperature": 0.0, "stream": False},
                args.probe_timeout,
            )
            row["probe_seconds"] = time.perf_counter() - started
            row["probe_status"] = "ok" if isinstance(response, dict) else "invalid"
        return row
    finally:
        if not args.keep:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        if log_path:
            log_handle.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="real GGUF model path")
    parser.add_argument("--executable", default="engine/core/build-serve/clozn-server")
    parser.add_argument("--contexts", type=int, nargs="+", default=[4096, 12288, 16384])
    parser.add_argument("--gpu-layers", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--ubatch", type=int, default=None)
    parser.add_argument("--startup-timeout", type=float, default=180.0)
    parser.add_argument("--probe-timeout", type=float, default=600.0)
    parser.add_argument("--probe-max-tokens", type=int, default=1)
    parser.add_argument("--probe-prompt-file", default=None)
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--keep", action="store_true", help="leave workers running for manual inspection")
    args = parser.parse_args(argv)
    if args.log_dir:
        Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    prompt = Path(args.probe_prompt_file).read_text(encoding="utf-8") if args.probe_prompt_file else None
    rows = []
    for context in args.contexts:
        rows.append(_run_one(args, context, prompt))
    print(json.dumps({"status": "ok", "results": rows}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
