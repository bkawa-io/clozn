"""test_engine_ctx_overflow -- regression for the /v1/completions decode-500 bugs (both root causes the
routing experiment surfaced). A long generation must NEVER 500: it stops gracefully at the context window
with finish_reason "length" (a clean 200), and a prompt that itself exceeds n_ctx is a clean 400 -- never
an uncaught throw that cpp-httplib turns into an empty-body 500.

Two bugs this locks down:
  1. n_ctx OVERFLOW during decode -- GgmlAdapter::ar_forward threw "exceeds n_ctx" when a generation
     reached the context window; the non-streaming handler didn't catch it -> empty 500. Fixed in
     generate_ar (graceful "length" stop) + the handler try/catch. Reproduced DETERMINISTICALLY here on
     the small Llama-1B with a tiny --ctx (fast; no 7B needed).
  2. strict-UTF-8 serialization -- the non-stream response used resp.dump() (strict), which threw on a
     byte-fallback token whose piece is a partial multi-byte UTF-8 sequence (the deterministic
     `nitrogen_cycle` 500). Fixed by dump_json() (replace handler). Not asserted here (needs a specific
     token on the 7B); see the manual repro in the task report. This test focuses on the fast, deterministic
     overflow case, which is the one that reproduces cheaply.

Gated behind -m model: it launches the REAL clozn-server.exe on the GPU (needs the build + the Llama-1B
GGUF). Skips cleanly when either is missing. Mirrors test_timetravel_determinism.py's -m model gating.

    python -m pytest tests/test_engine_ctx_overflow.py -m model -q
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "clozn"))

MODEL = os.path.expanduser("~/.clozn/models/Llama-3.2-1B-Instruct-Q4_K_M.gguf")
CTX = 128


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _post(base, path, body, timeout=60):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(payload)
        except json.JSONDecodeError:
            return e.code, payload


@pytest.fixture(scope="module")
def engine():
    """Launch the real clozn-server on Llama-1B at --ctx 128; skip cleanly if the build/model is missing."""
    try:
        from clozn.cli import find_engine, _env_with_dlls
    except Exception as e:                                    # pragma: no cover
        pytest.skip(f"clozn.cli unavailable: {e}")
    if not os.path.isfile(MODEL):
        pytest.skip(f"no Llama-1B GGUF at {MODEL}")
    try:
        exe, dll_dirs, gpu = find_engine(prefer_gpu=True)
    except Exception as e:
        pytest.skip(f"no engine build: {e}")

    port = _free_port()
    args = [exe, MODEL, "--port", str(port), "--host", "127.0.0.1", "--ctx", str(CTX)]
    if gpu:
        args += ["--gpu-layers", "99"]
    proc = subprocess.Popen(args, env=_env_with_dlls(dll_dirs, gpu),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if proc.poll() is not None:
                pytest.skip(f"engine exited early (code {proc.returncode})")
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1.0)
        else:
            pytest.skip("engine did not become healthy in 120s")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except Exception:
            proc.kill()


@pytest.mark.model
def test_overflow_during_decode_stops_at_length_not_500(engine):
    """prompt (short) + max_tokens far exceeding n_ctx: the generation must STOP at the context window with
    finish_reason "length" (200), NOT throw "ar_forward: exceeds n_ctx" -> empty 500."""
    code, resp = _post(engine, "/v1/completions",
                       {"prompt": "Count and keep going listing many numbers one two three four:",
                        "max_tokens": 4 * CTX})   # 512 >> 128: guaranteed to reach the context window
    assert code == 200, f"expected 200, got {code}: {resp}"
    ch = resp["choices"][0]
    assert ch["finish_reason"] == "length", ch
    # it actually generated tokens (didn't 0-length), and stopped within the window.
    n = resp["usage"]["completion_tokens"]
    assert 0 < n <= CTX, n


@pytest.mark.model
def test_prompt_longer_than_ctx_is_a_clean_400_not_500(engine):
    """A prompt that itself exceeds n_ctx is a client error -> a clean 400 JSON, never an uncaught 500."""
    long_prompt = " ".join(f"number {i}" for i in range(4 * CTX))   # ~1000 tokens >> 128
    code, resp = _post(engine, "/v1/completions", {"prompt": long_prompt, "max_tokens": 8})
    assert code == 400, f"expected 400, got {code}: {resp}"
    assert isinstance(resp, dict) and "error" in resp
    assert "context" in resp["error"].lower()


@pytest.mark.model
def test_normal_generation_still_works(engine):
    """A generation that fits the window is unaffected -- a normal 200 with finish_reason stop/length."""
    code, resp = _post(engine, "/v1/completions",
                       {"prompt": "The capital of France is", "max_tokens": 5})
    assert code == 200, resp
    assert resp["choices"][0]["finish_reason"] in ("stop", "length")
    assert isinstance(resp["choices"][0]["text"], str)
