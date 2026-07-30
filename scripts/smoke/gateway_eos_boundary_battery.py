"""gateway_eos_boundary_battery.py -- deterministic reproducer + regression battery for the gateway
unchanged-control EOS-boundary divergence (characterized 2026-07-30).

THE BUG THIS REPRODUCES
-------------------------
`POST /runs/<id>/execution-fork/checkpoint` (recorded-parent checkpoint capture,
clozn.replay.checkpoint_capture.capture_parent_checkpoint) runs a mandatory "unchanged control": it
reconstructs the parent's checkpoint and replays it with intervention:none, then requires the replay
to reproduce the parent's own recorded response TOKEN FOR TOKEN
(clozn.replay.execution_fork_execute.prove_unchanged_control). Before the fix this comparison was
UNCONDITIONALLY EXACT on paper but only ACTUALLY exact when the parent's generation was cut off by
max_tokens (finish_reason == "length"). Whenever the parent instead finished NATURALLY via EOS/stop
(finish_reason == "stop" -- the ordinary, common case for a short chat reply), the comparison
diverged 100% of the time, always by exactly one trailing token: the chat turn's terminal stop/EOS-
class token id (e.g. Qwen2.5's <|im_end|> = 151645), which the recorded trace (native chat-io's
transcript) includes as an explicit entry but the raw engine's OWN generation loop (generate_ar, what
this route replays through) never returns as part of `tokens` -- sampling that token TERMINATES the
loop there; it is a stop signal, not committed output. Two independently-correct conventions for
"what counts as generated" that disagree at exactly one position. Decoded TEXT was identical in every
observed case (the extra token decodes to an empty string); only the raw token-id list length and
content at that one boundary position differed.

Discriminator, established empirically against qwen2.5-0.5b-instruct-q4_k_m.gguf, 8/8 deterministic
reproductions (4 finish_reason=="stop" cases: 4/4 diverged; 4 finish_reason=="length" cases: 4/4
matched), independent of prompt content, response length, or max_tokens value:

    finish_reason == "stop"    (natural EOS/chat-turn-end)  -> ALWAYS diverged (pre-fix)
    finish_reason == "length"  (max_tokens truncation)       -> ALWAYS matched

The fix (clozn.replay.execution_fork_execute._boundary_stop_token_exempt, used by
prove_unchanged_control) narrowly exempts EXACTLY this shape: parent AND replay both independently
report finish_reason=="stop", the parent's suffix is exactly one token longer, every token before
that final one matches exactly, and the decoded text matches exactly. Model-free unit coverage for
every edge of that exemption (including that it must NOT mask a real divergence hiding behind the
same length delta) lives in tests/test_checkpoint_capture.py's "Boundary stop-token exemption"
section; THIS script is the live, real-GGUF proof the fix actually holds end to end through the real
gateway + a real chat completion, not just a model-free stub.

WHAT THIS SCRIPT PROVES, LIVE
-------------------------------
1. An ordinary short chat completion that finishes via EOS (finish_reason=="stop") is eligible for
   recorded-parent checkpoint capture -- status=="available", exact_match==True. THIS IS THE
   REGRESSION CHECK: on the pre-fix code, this case reproduces the divergence deterministically (see
   the characterization above) -- a HARD FAIL here means the fix has regressed.
2. A chat completion forced to truncate at max_tokens (finish_reason=="length") is STILL eligible,
   unaffected by the fix (this path never needed the exemption -- it must keep working unchanged).
3. On any divergence, prints the exact positions and both token-id sequences by replaying the SAME
   execution_fork call directly against the raw worker (the gateway's own HTTP response only ever
   carries hashes, by design) -- so a genuine regression is diagnosable from this script's output
   alone, never just "it failed."

HONEST DEGRADE
---------------
Needs a real engine + a real GGUF; missing either -> SKIPPED, exit 0.

Usage:
    python scripts/smoke/gateway_eos_boundary_battery.py
    python scripts/smoke/gateway_eos_boundary_battery.py --model C:\\...\\qwen2.5-0.5b...gguf --port 8193
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ENGINE_CLIENT_DIR = os.path.join(REPO, "engine", "client")
if ENGINE_CLIENT_DIR not in sys.path:
    sys.path.insert(0, ENGINE_CLIENT_DIR)

sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
DEFAULT_PORT = 8193
MODELS_DIR = os.path.expanduser("~/.clozn/models")
_PREFERRED = [
    os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    os.path.join(MODELS_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
]


def _find_model(explicit) -> "str | None":
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for path in _PREFERRED:
        if os.path.isfile(path):
            return path
    return None


def _http(port, method, path, body=None, timeout=180):
    encoded = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=encoded, method=method,
        headers={"Content-Type": "application/json"} if encoded is not None else {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            return exc.code, json.loads(raw.decode("utf-8", "replace") or "{}")
        except Exception:
            return exc.code, {"_raw": raw.decode("utf-8", "replace")}


def _worker_url(gw_port):
    _status, doc = _http(gw_port, "GET", "/engine/health")
    return ((doc or {}).get("engine") or {}).get("worker_url")


def _row(name, status, detail) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _deep_dive(gw_port, run_id, cap) -> str:
    """Replay prove_unchanged_control's own execution_fork call directly against the raw worker so
    a divergence is diagnosable from raw token ids, not just a hash mismatch."""
    checkpoint_reference = cap.get("checkpoint_reference") or {}
    worker_receipt = (cap.get("proof") or {}).get("worker_receipt") or {}
    ckpt_id = checkpoint_reference.get("checkpoint_id")
    ckpt_gen = checkpoint_reference.get("worker_generation_id")
    n_past_restored = worker_receipt.get("n_past_restored")
    if not (ckpt_id and ckpt_gen and isinstance(n_past_restored, int)):
        return "cannot deep-dive: incomplete checkpoint_reference/worker_receipt"

    status, parent = _http(gw_port, "GET", f"/runs/{run_id}")
    if status != 200:
        return f"cannot deep-dive: GET /runs/{run_id} -> {status}"
    trace = parent.get("trace") or {}
    expected_tokens = [int(t) for t in (trace.get("token_ids") or [])]
    expected_text = "".join(trace.get("tokens") or [])

    wurl = _worker_url(gw_port)
    parsed = urlparse(wurl or "")
    from clozn_engine import EngineClient
    eng = EngineClient(host=parsed.hostname, port=parsed.port)
    reply = eng.execution_fork(
        checkpoint_id=ckpt_id, worker_generation_id=ckpt_gen, truncate_to=n_past_restored,
        max_tokens=len(expected_tokens), intervention={"type": "none"})
    actual_tokens = [int(t) for t in reply.get("tokens", [])]
    actual_text = reply.get("text")

    n = min(len(expected_tokens), len(actual_tokens))
    first_diff = next((i for i in range(n) if expected_tokens[i] != actual_tokens[i]), None)
    if first_diff is None and len(expected_tokens) != len(actual_tokens):
        first_diff = n
    lines = [
        f"parent finish_reason={parent.get('finish_reason')!r} replay finish_reason={reply.get('finish_reason')!r}",
        f"expected ({len(expected_tokens)} tokens): {expected_tokens}",
        f"actual   ({len(actual_tokens)} tokens): {actual_tokens}",
        f"expected text: {expected_text!r}",
        f"actual   text: {actual_text!r}",
    ]
    if first_diff is not None:
        exp_tok = expected_tokens[first_diff] if first_diff < len(expected_tokens) else None
        act_tok = actual_tokens[first_diff] if first_diff < len(actual_tokens) else None
        lines.append(f"FIRST DIVERGENCE at output position {first_diff}: "
                     f"expected token_id={exp_tok} got token_id={act_tok}")
    return " | ".join(lines)


def run_case(gw_port, rows, *, label, messages, max_tokens, expect_finish_reason):
    status, doc = _http(gw_port, "POST", "/v1/chat/completions", {
        "messages": messages, "max_tokens": max_tokens, "temperature": 0,
    })
    run_id = doc.get("clozn_run_id")
    finish_reason = ((doc.get("choices") or [{}])[0].get("finish_reason"))
    if status != 200 or not run_id:
        rows.append(_row(label, FAIL, f"could not create parent run: HTTP {status} doc={doc}"))
        return
    if finish_reason != expect_finish_reason:
        rows.append(_row(
            label, SKIP,
            f"parent finished with finish_reason={finish_reason!r}, expected {expect_finish_reason!r} "
            f"for this case (the model's exact output length is not something this script controls "
            f"precisely -- adjust max_tokens if this SKIPs repeatedly)"))
        return

    status, cap = _http(gw_port, "POST", f"/runs/{run_id}/execution-fork/checkpoint", {})
    cap_status = cap.get("status")
    if cap_status == "available":
        exact = cap.get("proof", {}).get("control_result", {}).get("exact_match")
        rows.append(_row(label, PASS if exact is True else FAIL,
                         f"finish_reason={finish_reason!r} status=available exact_match={exact!r}"))
        return
    detail = f"finish_reason={finish_reason!r} HTTP={status} status={cap_status!r} " \
             f"reasons={cap.get('reasons')}"
    try:
        detail += " || " + _deep_dive(gw_port, run_id, cap)
    except Exception as exc:  # noqa: BLE001 -- the deep-dive is best-effort diagnostic, not the check
        detail += f" || deep-dive failed: {type(exc).__name__}: {exc}"
    rows.append(_row(label, FAIL, detail))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args(argv)

    model = _find_model(args.model)
    if model is None:
        print("SKIPPED: no AR GGUF found. Pass --model, or place one under ~/.clozn/models/*.gguf.")
        return 0

    from clozn.cli import main as ctx
    from clozn.cli.engine_process import find_engine_ex
    try:
        find_engine_ex(prefer_gpu=not args.cpu)
    except ctx.CloznError as exc:
        print(f"SKIPPED: no engine build available -- {exc}")
        return 0

    from clozn.cli.runtime_process import gateway_health
    tmp = tempfile.TemporaryDirectory(prefix="clozn-eos-boundary-")
    log_path = os.path.join(tmp.name, "serve.log")
    log = open(log_path, "w", encoding="utf-8")
    command = [sys.executable, "-m", "clozn", "serve", model, "--port", str(args.port)]
    if args.cpu:
        command.append("--cpu")
    proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                            start_new_session=(os.name != "nt"))
    log.close()
    rows: list = []
    try:
        deadline = time.monotonic() + 180
        ready = None
        while time.monotonic() < deadline:
            ready = gateway_health(args.port, timeout=1.0)
            if ready:
                break
            if proc.poll() is not None:
                print(f"SERVE EXITED EARLY: {open(log_path, encoding='utf-8', errors='replace').read()[-3000:]}")
                return 1
            time.sleep(0.3)
        if not ready:
            print("SKIPPED: gateway did not become ready")
            return 0
        print(f"gateway ready on port {args.port}")

        run_case(
            args.port, rows,
            label="EOS-terminated chat reply (finish_reason='stop') is eligible for capture",
            messages=[{"role": "user", "content": "Count from one to ten, writing each number as "
                                                   "a word, separated by commas."}],
            max_tokens=32, expect_finish_reason="stop",
        )
        run_case(
            args.port, rows,
            label="max_tokens-truncated chat reply (finish_reason='length') is eligible for capture",
            messages=[{"role": "user", "content": "Count from one to twenty, writing each number "
                                                   "as a word, separated by commas."}],
            max_tokens=8, expect_finish_reason="length",
        )
        run_case(
            args.port, rows,
            label="a second, different EOS-terminated reply (finish_reason='stop') -- not a fluke",
            messages=[{"role": "user", "content": "Say hello in one short sentence."}],
            max_tokens=16, expect_finish_reason="stop",
        )

        print("\n" + "=" * 108)
        for r in rows:
            print(f"[{r['status']}] {r['name']}")
            print(f"    -> {r['detail']}")
        n_pass = sum(1 for r in rows if r["status"] == PASS)
        n_fail = sum(1 for r in rows if r["status"] == FAIL)
        n_skip = sum(1 for r in rows if r["status"] == SKIP)
        print("-" * 108)
        print(f"{n_pass}/{len(rows)} PASS, {n_fail} FAIL, {n_skip} SKIP")
        return 1 if n_fail else 0
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=15)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
        tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
