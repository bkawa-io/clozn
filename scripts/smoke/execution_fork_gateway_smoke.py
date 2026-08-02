"""execution_fork_gateway_smoke.py -- the first LIVE acceptance battery for the exact-execution-fork
GATEWAY path: POST /runs/<id>/execution-fork/checkpoint, /plan, /execution-fork, and
GET /execution-forks/<id> (docs/EXECUTION_FORK_CONTRACT.md).

The engine-level primitive (POST /v1/execution-fork on the raw private worker) is already proven live
by scripts/smoke/execution_fork_battery.py (20/20). This script proves the layer ABOVE it that landed
model-free-only on origin/main@8ec835f: the public gateway routes a real Studio/CLI/API client actually
calls, recorded-parent checkpoint capture reconstructed from an ORDINARY persisted chat run (not one
generated with checkpoint_on_finish), the plan/execute split, and the mandatory unchanged-control gate.
docs/HANDOFF_2026-07-29.md section 3: "No live ... managed exact-fork smoke was completed."

WHAT THIS PROVES
-----------------
1. A real instrumented run through POST /v1/chat/completions (not a raw engine call) is eligible for
   recorded-parent checkpoint capture -- prompt IDs reconstructed via the worker's own /score, not a
   second, potentially drifting retokenization.
2. POST .../execution-fork/plan returns exactly one honest classification -- exact_execution_fork,
   reconstructed_replay, or unavailable -- with a reason.
3. The mandatory unchanged control (intervention: {"type": "none"}) that the implementation runs before
   ANY intervention actually fires and actually passes: the child reproduces the parent's own response
   text token-for-token when nothing was asked to change.
4. A real intervention (force_token, using a token id/piece the model itself already produced elsewhere
   in the SAME run -- so it is a genuinely valid (id, piece) pair for this exact tokenizer, discovered
   without a second engine round trip) produces a child run with immutable lineage and a terminal
   receipt retrievable at GET /execution-forks/<id>.
5. Two distinct honest-failure paths, proven live rather than assumed:
     (a) a checkpoint reference that was never actually issued -> plan returns unavailable, not a
         guess.
     (b) Protocol 1.1's whole reason for generation-scoped checkpoint IDs: a checkpoint captured
         before a worker RESTART, resubmitted unchanged to /plan AFTER the restart, must return
         unavailable -- a stale ckpt-N must never resolve against the new process generation.
6. An explicit durable-pin path: a run-scoped checkpoint pin survives that worker restart, and
   `POST .../execution-fork/checkpoint {"pinned": true}` hydrates it into the new generation before
   proving the unchanged control. The public receipt remains ephemeral; the durable bytes never leave
   the local pin store.

HONEST DEGRADE
---------------
Needs the real engine build and one small AR GGUF; missing either -> SKIPPED with an explicit reason,
exit 0. A 404 from these routes (build predates the feature) is also a clean SKIP. Any genuine gap
between the contract above and what the live gateway actually does -- checkpoint capture unexpectedly
unavailable on a plain greedy chat run, a wrong plan classification, a control that does not reproduce,
a stale checkpoint that silently resolves -- is a HARD FAIL naming the exact observed response. Nothing
here is tuned to make the battery pass.

Usage:
    python scripts/smoke/execution_fork_gateway_smoke.py
    python scripts/smoke/execution_fork_gateway_smoke.py --model C:\\...\\qwen2.5-0.5b...gguf --port 8199
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

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
MODELS_DIR = os.path.expanduser("~/.clozn/models")
_PREFERRED = [
    os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf"),
    os.path.join(MODELS_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
]
PROMPT = "Count from one to twenty, writing each number as a word, separated by commas."


def _row(name: str, status: str, detail: str) -> dict:
    assert status in (PASS, FAIL, SKIP)
    return {"name": name, "status": status, "detail": detail}


class Client:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = float(timeout)

    def request(self, method: str, path: str, body=None):
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=encoded, method=method,
            headers={"Content-Type": "application/json"} if encoded is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return int(resp.status), resp.read(), ""
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
            except Exception:
                raw = b""
            return int(exc.code), raw, str(exc)
        except Exception as exc:
            return 0, b"", str(exc)

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, body=None):
        return self.request("POST", path, {} if body is None else body)


def _shallow_diff(a: dict, b: dict) -> dict:
    """Top-level (and one nested level) keys that differ between two dicts, for a precise FAIL
    detail instead of a bare True/False -- names exactly which field broke immutability."""
    out = {}
    for key in sorted(set(a) | set(b)):
        av, bv = a.get(key, "<missing>"), b.get(key, "<missing>")
        if av == bv:
            continue
        if isinstance(av, dict) and isinstance(bv, dict):
            nested = {k: (av.get(k, "<missing>"), bv.get(k, "<missing>"))
                     for k in sorted(set(av) | set(bv)) if av.get(k, "<missing>") != bv.get(k, "<missing>")}
            out[key] = nested
        else:
            out[key] = (av, bv)
    return out


def _j(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8", "replace") or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _find_model(explicit: "str | None") -> "str | None":
    if explicit:
        return explicit if os.path.isfile(explicit) else None
    for path in _PREFERRED:
        if os.path.isfile(path):
            return path
    return None


def _wait_ready(port: int, proc, timeout: float):
    from clozn.cli.runtime_process import gateway_health
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = gateway_health(port, timeout=1.0)
        if state:
            return state, ""
        if proc.poll() is not None:
            return None, f"serve exited with code {proc.returncode}"
        time.sleep(0.2)
    return None, f"gateway did not become ready within {timeout:g}s"


def _tail(path, limit=4000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()[-limit:].strip().replace("\n", " | ")
    except Exception:
        return ""


# =================================================================================== battery

def run_battery(args) -> list:
    rows: list = []
    model = _find_model(args.model)
    if model is None:
        rows.append(_row("model available", SKIP,
                         f"no AR GGUF found. Tried {args.model or _PREFERRED}"))
        return rows

    from clozn.cli import main as ctx
    from clozn.cli.engine_process import find_engine_ex
    try:
        find_engine_ex(prefer_gpu=not args.cpu)
    except ctx.CloznError as exc:
        rows.append(_row("engine build available", SKIP, str(exc)))
        return rows

    from clozn.cli.engine_process import _free_port
    tmp = tempfile.TemporaryDirectory(prefix="clozn-fork-gw-smoke-")
    port = args.port or _free_port()
    base = f"http://127.0.0.1:{port}"
    log_path = os.path.join(tmp.name, "serve.log")
    log = open(log_path, "w", encoding="utf-8")
    command = [sys.executable, "-m", "clozn", "serve", model, "--port", str(port)]
    if args.cpu:
        command.append("--cpu")
    started = time.monotonic()
    proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=(os.name != "nt"),
                            creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
    log.close()
    try:
        ready, problem = _wait_ready(port, proc, args.startup_timeout)
        if not ready:
            rows.append(_row("clozn serve MODEL boots", FAIL, f"{problem}; log: {_tail(log_path)}"))
            return rows
        rows.append(_row("clozn serve MODEL boots", PASS,
                         f"port={port} in {time.monotonic()-started:.1f}s"))
        client = Client(base, args.timeout)
        _exercise(rows, client, port, args)
    finally:
        _teardown(port, proc)
        tmp.cleanup()
    return rows


class RouteNotBuilt(Exception):
    pass


def _post_fork_route(client: Client, path: str, body: dict, *, confirmed: list):
    """POST one of the execution-fork gateway routes. A 404 BEFORE any such route has ever answered
    on this worker means the route is not built on this engine yet (a clean SKIP, mirroring
    execution_fork_battery.py's own ForkProbe contract); a 404 after is an ordinary miss."""
    status, raw, err = client.post(path, body)
    if status == 404 and not confirmed:
        raise RouteNotBuilt(f"POST {path} -> 404: {err}")
    if status != 404:
        confirmed.append(True)
    return status, _j(raw), err


def _exercise(rows: list, client: Client, port: int, args) -> None:
    confirmed: list = []

    # --- step 1: a real instrumented parent run through the gateway -------------------------
    status, raw, err = client.post("/v1/chat/completions", {
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 48, "temperature": 0,
    })
    doc = _j(raw)
    run_id = doc.get("clozn_run_id")
    reply = ((doc.get("choices") or [{}])[0].get("message") or {}).get("content")
    if status != 200 or not run_id or not reply:
        rows.append(_row("real instrumented parent run via /v1/chat/completions", FAIL,
                         f"HTTP {status} run_id={run_id!r} reply={reply!r} err={err}"))
        return
    rows.append(_row("real instrumented parent run via /v1/chat/completions", PASS,
                     f"run_id={run_id} reply={reply!r}"))

    status, raw, err = client.get(f"/runs/{run_id}")
    parent = _j(raw)
    trace = parent.get("trace") if isinstance(parent.get("trace"), dict) else {}
    token_ids = trace.get("token_ids") if isinstance(trace.get("token_ids"), list) else None
    pieces = trace.get("tokens") if isinstance(trace.get("tokens"), list) else None
    if status != 200 or not token_ids or not pieces or len(token_ids) < 4:
        rows.append(_row("parent run has a usable token trace", FAIL,
                         f"HTTP {status} token_ids={token_ids!r} tokens_len={len(pieces or [])}"))
        return
    rows.append(_row("parent run has a usable token trace", PASS,
                     f"{len(token_ids)} generated tokens recorded"))

    # --- step 2: recorded-parent checkpoint capture ------------------------------------------
    try:
        status, cap, err = _post_fork_route(
            client, f"/runs/{run_id}/execution-fork/checkpoint", {}, confirmed=confirmed)
    except RouteNotBuilt as exc:
        rows.append(_row("execution-fork gateway routes are built on this worker", SKIP, str(exc)))
        return
    cap_status = cap.get("status")
    rows.append(_row("POST .../execution-fork/checkpoint on a plain greedy chat run",
                     PASS if cap_status == "available" else FAIL,
                     f"HTTP {status} status={cap_status!r} reasons={cap.get('reasons')}"))
    if cap_status != "available":
        rows.append(_row("(cascade) plan/execute/unchanged-control require an available checkpoint",
                         SKIP, f"checkpoint capture did not reach 'available' (status={cap_status!r}); "
                         "see the row above for the exact reason returned"))
        _run_bogus_reference_check(rows, client, run_id, confirmed)
        return
    checkpoint_reference = cap.get("checkpoint_reference") or {}
    original_worker_generation = checkpoint_reference.get("worker_generation_id")

    # --- step 3: plan an unchanged-control fork at an interior position ----------------------
    position_none = 1
    status, plan_none, err = _post_fork_route(
        client, f"/runs/{run_id}/execution-fork/plan",
        {"request": {"position": position_none, "change": {"type": "none"}},
         "checkpoint_reference": checkpoint_reference},
        confirmed=confirmed)
    classification = plan_none.get("classification")
    valid_outcome = classification in ("exact_execution_fork", "reconstructed_replay", "unavailable")
    has_reason = bool(plan_none.get("reasons"))
    rows.append(_row("POST .../execution-fork/plan returns exactly one honest outcome + a reason",
                     PASS if valid_outcome and (classification != "unavailable" or has_reason) else FAIL,
                     f"HTTP {status} classification={classification!r} reasons={plan_none.get('reasons')}"))
    if classification != "exact_execution_fork":
        rows.append(_row("a fresh available checkpoint + supported 'none' intervention plans exact",
                         FAIL, f"expected classification='exact_execution_fork' on this straightforward "
                         f"happy-path case, got {classification!r}: reasons={plan_none.get('reasons')}"))
        _run_bogus_reference_check(rows, client, run_id, confirmed)
        return
    rows.append(_row("a fresh available checkpoint + supported 'none' intervention plans exact",
                     PASS, f"plan_id={plan_none.get('plan_id')}"))

    # --- step 4: execute it -- the MANDATORY unchanged control must fire and pass ------------
    status, exec_none, err = _post_fork_route(
        client, f"/runs/{run_id}/execution-fork", {"plan": plan_none}, confirmed=confirmed)
    receipt = exec_none.get("receipt") or {}
    child = exec_none.get("child") or {}
    control = (receipt.get("unchanged_control") or {}).get("result") or {}
    control_exact = control.get("exact_match") is True
    phase_completed = receipt.get("phase") == "completed"
    rows.append(_row("the mandatory unchanged control actually fires and actually passes",
                     PASS if control_exact and phase_completed else FAIL,
                     f"HTTP {status} phase={receipt.get('phase')!r} "
                     f"unchanged_control.result={control!r}"))
    reproduces = child.get("response") == parent.get("response")
    rows.append(_row("child reproduces the parent's response TOKEN-FOR-TOKEN under change=none",
                     PASS if reproduces else FAIL,
                     f"parent.response={parent.get('response')!r} child.response={child.get('response')!r}"))

    # --- step 5: one real intervention (force_token), using a token this run already proved ---
    forced_pos = min(3, len(token_ids) - 1)
    forced_id, forced_piece = token_ids[0], pieces[0]
    if token_ids[forced_pos] == forced_id:   # pick a genuinely different natural token if they coincide
        forced_id, forced_piece = token_ids[-1], pieces[-1]
    status, plan_force, err = _post_fork_route(
        client, f"/runs/{run_id}/execution-fork/plan",
        {"request": {"position": forced_pos,
                     "change": {"type": "force_token", "token_id": forced_id, "token_piece": forced_piece}},
         "checkpoint_reference": checkpoint_reference},
        confirmed=confirmed)
    force_classification = plan_force.get("classification")
    if force_classification != "exact_execution_fork":
        rows.append(_row("force_token intervention plans exact", FAIL,
                         f"classification={force_classification!r} reasons={plan_force.get('reasons')}"))
    else:
        status, exec_force, err = _post_fork_route(
            client, f"/runs/{run_id}/execution-fork", {"plan": plan_force}, confirmed=confirmed)
        f_receipt = exec_force.get("receipt") or {}
        f_child = exec_force.get("child") or {}
        f_control = (f_receipt.get("unchanged_control") or {}).get("result") or {}
        ok = (f_receipt.get("phase") == "completed" and f_control.get("exact_match") is True
             and f_child.get("source") == "fork" and f_child.get("parent_run_id") == run_id
             and isinstance(f_receipt.get("execution_id"), str) and f_receipt.get("execution_id"))
        rows.append(_row("force_token intervention: child run has lineage + immutable terminal receipt",
                         PASS if ok else FAIL,
                         f"HTTP {status} phase={f_receipt.get('phase')!r} "
                         f"child.parent_run_id={f_child.get('parent_run_id')!r} "
                         f"child.source={f_child.get('source')!r} execution_id={f_receipt.get('execution_id')!r} "
                         f"forced=({forced_id}, {forced_piece!r}) at position={forced_pos}"))
        if ok:
            execution_id = f_receipt["execution_id"]
            status, raw, err = client.get(f"/execution-forks/{execution_id}")
            fetched = _j(raw)
            matches = status == 200 and fetched == f_receipt
            diff = "" if matches else _shallow_diff(f_receipt, fetched)
            rows.append(_row("GET /execution-forks/<id> returns the SAME immutable terminal receipt",
                             PASS if matches else FAIL,
                             f"HTTP {status} identical={matches}" + (f" diff={diff}" if diff else "")))

    # --- step 6a: a checkpoint reference that was never actually issued ---------------------
    _run_bogus_reference_check(rows, client, run_id, confirmed)

    # --- step 6b: durable pin creation (preview, then explicit write) -----------------------
    # Keep this late in the battery so an earlier happy-path refusal cannot leave a durable test
    # artifact behind. The restart check below always removes a successfully-created pin.
    pin_status, pin_preview_raw, pin_err = client.post(
        f"/runs/{run_id}/snapshot/pin", {"preview": True})
    pin_preview = _j(pin_preview_raw)
    preview_ok = pin_status == 200 and pin_preview.get("preview") is True
    rows.append(_row("durable snapshot pin preview reports its byte cost",
                     PASS if preview_ok else FAIL,
                     f"HTTP {pin_status} preview={pin_preview.get('preview')!r} "
                     f"envelope_bytes={pin_preview.get('envelope_bytes')!r}"))
    pin_manifest = {}
    if preview_ok:
        pin_status, pin_receipt_raw, pin_err = client.post(
            f"/runs/{run_id}/snapshot/pin", {"preview": False, "note": "gateway smoke"})
        pin_receipt = _j(pin_receipt_raw)
        pin_manifest = pin_receipt.get("manifest") if isinstance(pin_receipt.get("manifest"), dict) else {}
        pin_ok = pin_status == 201 and pin_manifest.get("run_id") == run_id
        rows.append(_row("durable snapshot pin persists only after explicit confirmation",
                         PASS if pin_ok else FAIL,
                         f"HTTP {pin_status} run_id={pin_manifest.get('run_id')!r} err={pin_err}"))
    else:
        rows.append(_row("durable snapshot pin persists only after explicit confirmation",
                         SKIP, "preview did not complete; see the row above"))

    # --- step 6c: THE guard -- a checkpoint captured before a worker restart must not resolve
    #     after the restart (generation-scoped checkpoint IDs, Protocol 1.1) ------------------
    _run_restart_invalidation_check(rows, client, port, run_id, checkpoint_reference,
                                    original_worker_generation, confirmed, args,
                                    pin_manifest=pin_manifest)


def _run_bogus_reference_check(rows: list, client: Client, run_id: str, confirmed: list) -> None:
    bogus = {
        "checkpoint_id": "ckpt-never-issued-0",
        "worker_generation_id": "generation-that-was-never-issued",
        "state": "available",
        "parent_run_id": run_id,
        "prompt_tokens": 1,
        "n_past": 2,
    }
    try:
        status, plan, err = _post_fork_route(
            client, f"/runs/{run_id}/execution-fork/plan",
            {"request": {"position": 1, "change": {"type": "none"}}, "checkpoint_reference": bogus},
            confirmed=confirmed)
    except RouteNotBuilt as exc:
        rows.append(_row("a checkpoint reference that was never issued fails honestly", SKIP, str(exc)))
        return
    classification = plan.get("classification")
    ok = classification == "unavailable" and bool(plan.get("reasons"))
    rows.append(_row("a checkpoint reference that was never issued -> plan returns 'unavailable', not a guess",
                     PASS if ok else FAIL,
                     f"HTTP {status} classification={classification!r} reasons={plan.get('reasons')}"))


def _run_restart_invalidation_check(rows: list, client: Client, port: int, run_id: str,
                                    checkpoint_reference: dict, original_generation, confirmed: list,
                                    args, *, pin_manifest: dict | None = None) -> None:
    if not checkpoint_reference or not original_generation:
        rows.append(_row("a stale checkpoint from BEFORE a worker restart fails honestly after it",
                         SKIP, "no successful earlier checkpoint capture to make stale"))
        return
    from clozn.cli.engine_process import _kill, _reg_read
    entry = _reg_read().get(str(port)) or {}
    worker_pid = entry.get("worker_pid")
    if not worker_pid:
        rows.append(_row("a stale checkpoint from BEFORE a worker restart fails honestly after it",
                         SKIP, "could not read the worker PID from the local runtime registry"))
        return

    _kill(int(worker_pid))
    from clozn.cli.runtime_process import gateway_health
    deadline = time.monotonic() + args.startup_timeout
    new_generation = None
    while time.monotonic() < deadline:
        state = gateway_health(port, timeout=1.0)
        gen = ((state or {}).get("worker") or {}).get("worker_generation_id")
        if state and gen and gen != original_generation:
            new_generation = gen
            break
        time.sleep(0.3)
    if not new_generation:
        rows.append(_row("worker independently recovers after being killed", FAIL,
                         f"gateway never reported a NEW worker_generation_id "
                         f"(original={original_generation!r}) within {args.startup_timeout:g}s"))
        return
    rows.append(_row("worker independently recovers after being killed", PASS,
                     f"{original_generation!r} -> {new_generation!r}"))

    try:
        status, plan, err = _post_fork_route(
            client, f"/runs/{run_id}/execution-fork/plan",
            {"request": {"position": 1, "change": {"type": "none"}},
             "checkpoint_reference": checkpoint_reference},
            confirmed=confirmed)
    except RouteNotBuilt as exc:
        rows.append(_row("a stale checkpoint from BEFORE a worker restart fails honestly after it",
                         SKIP, str(exc)))
        return
    classification = plan.get("classification")
    ok = classification == "unavailable" and bool(plan.get("reasons"))
    rows.append(_row(
        "PROTOCOL 1.1 GUARD: a checkpoint captured before a worker restart must NOT resolve after it "
        "(generation-scoped checkpoint IDs)",
        PASS if ok else FAIL,
        f"HTTP {status} old_generation={original_generation!r} new_generation={new_generation!r} "
        f"classification={classification!r} reasons={plan.get('reasons')}"))

    # The durable pin is intentionally a separate, explicit path: the old in-memory reference above
    # must fail, while importing the verified pin into this new worker generation must succeed.
    if pin_manifest:
        status, hydrated_raw, err = client.post(
            f"/runs/{run_id}/execution-fork/checkpoint", {"pinned": True})
        hydrated = _j(hydrated_raw)
        hydrated_reference = hydrated.get("checkpoint_reference") if isinstance(
            hydrated.get("checkpoint_reference"), dict) else {}
        hydrated_ok = (
            status == 201
            and hydrated.get("status") == "available"
            and hydrated_reference.get("worker_generation_id") == new_generation
            and hydrated_reference.get("worker_generation_id") != original_generation
            and hydrated.get("proof", {}).get("status") == "matched"
        )
        rows.append(_row(
            "durable snapshot hydrates and proves after worker restart",
            PASS if hydrated_ok else FAIL,
            f"HTTP {status} status={hydrated.get('status')!r} "
            f"generation={hydrated_reference.get('worker_generation_id')!r} "
            f"reasons={hydrated.get('reasons')!r} err={err}"))
        cleanup_status, _cleanup_raw, cleanup_err = client.post(
            f"/snapshots/{run_id}/unpin", {"cascade": True})
        rows.append(_row(
            "gateway smoke removes its durable pin explicitly",
            PASS if cleanup_status == 200 else FAIL,
            f"HTTP {cleanup_status} err={cleanup_err}"))
    else:
        rows.append(_row(
            "durable snapshot hydrates and proves after worker restart",
            SKIP, "pin was not created; see the pin rows above"))


def _teardown(port: int, proc) -> None:
    try:
        subprocess.run([sys.executable, "-m", "clozn", "stop", str(port)], cwd=REPO,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            if os.name == "nt":
                from clozn.cli.engine_process import _kill
                _kill(proc.pid)
            else:
                proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass


def _print_table(rows: list) -> None:
    print("\n" + "=" * 116)
    print(f"{'name':<80} status")
    print("-" * 116)
    for r in rows:
        print(f"{r['name'][:78]:<80} {r['status']}")
        print(f"    -> {r['detail'][:500]}")
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    n_skip = sum(1 for r in rows if r["status"] == SKIP)
    print("=" * 116)
    print(f"{n_pass} PASS, {n_fail} FAIL, {n_skip} SKIP")


def main(argv=None) -> int:
    # Windows consoles default stdout to the system codepage (cp1252 etc.), which cannot encode
    # arbitrary model output (token pieces can be any Unicode codepoint) -- reconfigure to UTF-8 so a
    # PRINT never crashes and silently discards an already-collected live result.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--startup-timeout", type=float, default=120.0)
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tag", default="fork_gateway")
    args = ap.parse_args(argv)

    rows = run_battery(args)

    # Write the JSON summary BEFORE printing: a console-encoding crash in the human-readable table
    # must never discard an already-collected live result.
    out_dir = os.path.join(REPO, "runs", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = os.path.join(out_dir, f"execution_fork_gateway_smoke_{args.tag}_{ts}.json")
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    n_skip = sum(1 for r in rows if r["status"] == SKIP)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"schema": "clozn.execution_fork_gateway_smoke_report.v1", "generated_at": ts,
                   "pass": n_pass, "fail": n_fail, "skip": n_skip, "rows": rows}, f, indent=2, default=str)

    _print_table(rows)
    print(f"\nJSON summary -> {out_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
