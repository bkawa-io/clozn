"""Live acceptance battery for ADR 010 exact appended-turn Time Machine continuation.

This exercises the product gateway and a real private GGUF worker.  It creates two organic runs in
one session, proves that continuation fails closed before pinning, pins both exact source boundaries,
restarts the private worker, and continues both the latest and historical turns from durable imports.
It also probes the private wire's stale-generation and cooperative-cancellation terminal results.

The battery never treats a missing model/build as a pass.  Those prerequisites produce an explicit
SKIP; any contradiction after the gateway boots is a FAIL.  Created source pins are explicitly
removed in ``finally``.  Immutable run records remain in the ordinary local journal as live evidence.

Usage:
    python scripts/smoke/time_machine_continuation_gateway_smoke.py
    python scripts/smoke/time_machine_continuation_gateway_smoke.py --model /path/model.gguf
    python scripts/smoke/time_machine_continuation_gateway_smoke.py \
        --engine engine/core/build-serve/clozn-server --gpu
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import glob
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ENGINE_CLIENT_DIR = os.path.join(REPO, "engine", "client")
if ENGINE_CLIENT_DIR not in sys.path:
    sys.path.insert(0, ENGINE_CLIENT_DIR)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
_MODEL_FRAGMENTS = (
    "qwen2.5-0.5b-instruct", "llama-3.2-1b-instruct", "reasoning-0.5b",
)


def _row(name: str, status: str, detail: str) -> dict:
    return {"name": name, "status": status, "detail": detail}


def _json(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8", "replace") or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


class Client:
    def __init__(self, port: int, *, session_id: str | None = None, timeout: float = 90.0):
        self.base = f"http://127.0.0.1:{port}"
        self.session_id = session_id
        self.timeout = timeout

    def request(self, method: str, path: str, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"} if data is not None else {}
        if self.session_id:
            headers["X-Clozn-Session-Id"] = self.session_id
        request = urllib.request.Request(
            self.base + path, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return int(response.status), _json(response.read()), ""
        except urllib.error.HTTPError as exc:
            return int(exc.code), _json(exc.read()), str(exc)
        except Exception as exc:
            return 0, {}, f"{type(exc).__name__}: {exc}"

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, body=None):
        return self.request("POST", path, {} if body is None else body)


def _find_model(explicit: str | None) -> str | None:
    if explicit:
        return os.path.abspath(explicit) if os.path.isfile(explicit) else None
    paths = glob.glob(os.path.expanduser("~/.clozn/models/*.gguf"))
    for fragment in _MODEL_FRAGMENTS:
        for path in paths:
            if fragment in os.path.basename(path).lower():
                return path
    return None


def _tail(path: str, limit: int = 5000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()[-limit:].strip().replace("\n", " | ")
    except Exception:
        return ""


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


def _teardown(port: int, proc) -> None:
    try:
        subprocess.run(
            [sys.executable, "-m", "clozn", "stop", str(port)], cwd=REPO,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=20,
        )
    except Exception:
        pass
    try:
        proc.wait(timeout=10)
    except Exception:
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:
            pass


def _append_row(rows: list[dict], name: str, ok: bool, detail: str) -> bool:
    rows.append(_row(name, PASS if ok else FAIL, detail))
    return ok


def _create_session_runs(client: Client, rows: list[dict]):
    messages_1 = [{"role": "user", "content": "Name one primary color in a short sentence."}]
    status, first, error = client.post("/v1/chat/completions", {
        "messages": messages_1, "max_tokens": 16, "temperature": 0,
    })
    answer_1 = (((first.get("choices") or [{}])[0].get("message") or {}).get("content"))
    run_1 = first.get("clozn_run_id")
    ok = status == 200 and isinstance(run_1, str) and isinstance(answer_1, str) and bool(answer_1)
    _append_row(rows, "first organic session run is recorded", ok,
                f"HTTP {status} run_id={run_1!r} answer={answer_1!r} error={error}")
    if not ok:
        return None

    messages_2 = messages_1 + [
        {"role": "assistant", "content": answer_1},
        {"role": "user", "content": "Name a different primary color in a short sentence."},
    ]
    status, second, error = client.post("/v1/chat/completions", {
        "messages": messages_2, "max_tokens": 16, "temperature": 0,
    })
    answer_2 = (((second.get("choices") or [{}])[0].get("message") or {}).get("content"))
    run_2 = second.get("clozn_run_id")
    ok = status == 200 and isinstance(run_2, str) and isinstance(answer_2, str) and bool(answer_2)
    _append_row(rows, "second organic run preserves the completed first-turn prefix", ok,
                f"HTTP {status} run_id={run_2!r} answer={answer_2!r} error={error}")
    if not ok:
        return None

    _, parent_before, _ = client.get(f"/runs/{run_2}")
    return run_1, run_2, parent_before


def _pin_sources(client: Client, run_ids: tuple[str, str], rows: list[dict]) -> dict[str, dict]:
    manifests: dict[str, dict] = {}
    for index, run_id in enumerate(run_ids):
        status, reply, error = client.post(
            f"/runs/{run_id}/snapshot/pin",
            {"preview": False, "note": "ADR-010 continuation live smoke"},
        )
        manifest = reply.get("manifest") if isinstance(reply.get("manifest"), dict) else {}
        ok = status == 201 and reply.get("ok") is True and manifest.get("run_id") == run_id
        _append_row(rows, f"turn {index} exact source is durably pinned", ok,
                    f"HTTP {status} pin_id={manifest.get('pin_id')!r} error={error or reply.get('error')}")
        if ok:
            manifests[run_id] = manifest
    return manifests


def _restart_worker(port: int, old_generation: str, rows: list[dict], timeout: float) -> str | None:
    from clozn.cli.engine_process import _kill, _reg_read
    from clozn.cli.runtime_process import gateway_health

    pid = (_reg_read().get(str(port)) or {}).get("worker_pid")
    if not pid:
        rows.append(_row("private worker can be restarted independently", FAIL,
                         "runtime registry did not contain a worker_pid"))
        return None
    _kill(int(pid))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = gateway_health(port, timeout=1.0)
        generation = ((state or {}).get("worker") or {}).get("worker_generation_id")
        if generation and generation != old_generation:
            rows.append(_row("private worker recovers with a new generation", PASS,
                             f"{old_generation} -> {generation}"))
            return generation
        time.sleep(0.25)
    rows.append(_row("private worker recovers with a new generation", FAIL,
                     f"no new generation within {timeout:g}s"))
    return None


def _continue(client: Client, requested_run_id: str, turn: int, rows: list[dict], label: str):
    status, receipt, error = client.post(
        f"/runs/{requested_run_id}/time-machine/continue",
        {"turn": turn,
         "user": {"content": "Now name one secondary color in a short sentence."},
         "max_tokens": 20},
    )
    lineage = receipt.get("child_lineage") if isinstance(receipt.get("child_lineage"), dict) else {}
    checkpoint = receipt.get("source_checkpoint") if isinstance(
        receipt.get("source_checkpoint"), dict) else {}
    exactness = receipt.get("exactness") if isinstance(receipt.get("exactness"), dict) else {}
    child_id = lineage.get("child_run_id")
    ok = (
        status == 201 and receipt.get("status") == "completed"
        and checkpoint.get("provenance") == "durable_pin_import"
        and checkpoint.get("source_worker_generation_id") != checkpoint.get("executing_worker_generation_id")
        and exactness.get("append_only_execution") is True
        and exactness.get("historical_prefix_recomputed") is False
        and exactness.get("structural_fallback_used") is False
        and isinstance(child_id, str)
    )
    _append_row(rows, label, ok,
                f"HTTP {status} child={child_id!r} source={checkpoint.get('source_run_id')!r} "
                f"source_generation={checkpoint.get('source_worker_generation_id')!r} "
                f"executing_generation={checkpoint.get('executing_worker_generation_id')!r} "
                f"error={error or (receipt.get('failure') or {}).get('message')}")
    if not ok:
        return receipt, None

    child_status, child, child_error = client.get(f"/runs/{child_id}")
    persisted = child.get("time_machine_continuation")
    child_ok = (
        child_status == 200 and child.get("id") == child_id
        and child.get("parent_run_id") == requested_run_id
        and persisted == receipt
        and child.get("response")
    )
    _append_row(rows, f"{label}: immutable child embeds the identical terminal receipt", child_ok,
                f"HTTP {child_status} child={child_id!r} response={child.get('response')!r} "
                f"error={child_error}")
    return receipt, child if child_ok else None


def _stale_generation_probe(port: int, source_run_id: str, rows: list[dict]) -> None:
    from clozn.cli.engine_process import _reg_read
    from clozn.replay.checkpoint_pin_store import resolve_pin
    from clozn.replay.time_machine_continuation import token_ids_sha256
    from clozn_engine import EngineClient

    pin = resolve_pin(source_run_id)
    manifest = pin.get("manifest") or {}
    envelope = pin.get("envelope") or {}
    state = envelope.get("state") or {}
    try:
        reply = EngineClient(port=int(_reg_read()[str(port)]["worker_port"])).time_machine_continue(
            checkpoint_id=manifest["source"]["checkpoint_id"],
            worker_generation_id=manifest["source"]["worker_generation_id"],
            expected_n_past=state["n_past"],
            expected_token_history_sha256=token_ids_sha256(state["tokens"]),
            expected_checkpoint_payload_sha256=envelope["payload_sha256"],
            append_token_ids=[198], append_token_ids_sha256=token_ids_sha256([198]),
            max_tokens=4, request_id="tmc-live-stale-generation",
        )
    except Exception as exc:
        reply = {"error": f"{type(exc).__name__}: {exc}"}
    ok = reply.get("status") == "unavailable" and reply.get("code") == "worker_generation_stale"
    _append_row(rows, "stale process-local checkpoint generation is rejected precisely", ok,
                json.dumps(reply, sort_keys=True))


def _cancellation_probe(port: int, source_run_id: str, rows: list[dict]) -> None:
    from clozn.cli.engine_process import _reg_read
    from clozn.replay.checkpoint_pin_store import resolve_pin
    from clozn.replay.time_machine_continuation import token_ids_sha256
    from clozn.replay.timetravel import _completed_messages
    from clozn.runs.store import get_run
    from clozn_engine import EngineClient

    private_port = int(_reg_read()[str(port)]["worker_port"])
    engine = EngineClient(port=private_port, timeout=120)
    pin = resolve_pin(source_run_id)
    envelope = pin["envelope"]
    historical_ids = envelope["state"]["tokens"]
    try:
        imported = engine.import_checkpoint(envelope)
        messages = _completed_messages(get_run(source_run_id))
        messages.append({
            "role": "user",
            "content": "Write the integers from 1 through 500, one per line, without commentary.",
        })
        prompt = engine.apply_template_info(messages, add_assistant=True)["prompt"]
        full_ids = engine.score(
            prompt=prompt, continuation_ids=[historical_ids[-1]], topk=0)["prompt_ids"]
        if full_ids[:len(historical_ids)] != historical_ids:
            raise RuntimeError("validation render did not preserve the source token prefix")
        append_ids = full_ids[len(historical_ids):]
    except Exception as exc:
        rows.append(_row("in-flight exact continuation cooperatively cancels", FAIL,
                         f"setup failed: {type(exc).__name__}: {exc}"))
        return

    request_id = "tmc-live-cancellation"
    result: dict = {}

    def generate():
        try:
            result["reply"] = engine.time_machine_continue(
                checkpoint_id=imported["checkpoint_id"],
                worker_generation_id=imported["worker_generation_id"],
                expected_n_past=envelope["state"]["n_past"],
                expected_token_history_sha256=token_ids_sha256(historical_ids),
                expected_checkpoint_payload_sha256=envelope["payload_sha256"],
                append_token_ids=append_ids,
                append_token_ids_sha256=token_ids_sha256(append_ids),
                max_tokens=700,
                request_id=request_id,
            )
        except Exception as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"

    thread = threading.Thread(target=generate, daemon=True)
    thread.start()
    cancel_ack = False
    attempts = 0
    while thread.is_alive() and attempts < 200:
        attempts += 1
        try:
            if EngineClient(port=private_port, timeout=5).cancel(request_id).get("cancelled") is True:
                cancel_ack = True
                break
        except Exception:
            pass
        time.sleep(0.005)
    thread.join(timeout=30)
    reply = result.get("reply") or {}
    ok = (
        cancel_ack and not thread.is_alive()
        and reply.get("status") == "cancelled"
        and reply.get("code") == "request_cancelled"
        and reply.get("cancelled") is True
    )
    _append_row(rows, "in-flight exact continuation cooperatively cancels", ok,
                f"ack={cancel_ack} attempts={attempts} reply={reply} error={result.get('error')}")


def _exercise(
    rows: list[dict], client: Client, port: int, ready: dict, args, cleanup_runs: list[str],
) -> None:
    worker = ready.get("worker") if isinstance(ready.get("worker"), dict) else {}
    capabilities = worker.get("capabilities") if isinstance(worker.get("capabilities"), dict) else {}
    generation = worker.get("worker_generation_id")
    capable = (
        capabilities.get("time_machine_continuation") is True
        and capabilities.get("checkpoint_pin") is True
        and isinstance(worker.get("tokenizer_sha256"), str)
        and isinstance(generation, str)
    )
    if not _append_row(rows, "worker advertises continuation, pin, tokenizer, and generation identity",
                       capable, f"generation={generation!r} capabilities={capabilities}"):
        return

    created = _create_session_runs(client, rows)
    if created is None:
        return
    run_1, run_2, parent_before = created

    # Public exact continuation is restart-safe by contract, so a source without a durable pin must
    # not quietly use the worker's current process-local cache or a fresh full-prefix prefill.
    status, missing_pin, error = client.post(
        f"/runs/{run_2}/time-machine/continue",
        {"turn": 1, "user": {"content": "Continue without a pin."}, "max_tokens": 8},
    )
    no_child = ((missing_pin.get("child_lineage") or {}).get("status") == "not_created")
    missing_ok = (
        status == 422 and missing_pin.get("status") == "unavailable"
        and (missing_pin.get("failure") or {}).get("code") == "checkpoint_unavailable"
        and no_child
    )
    _append_row(rows, "missing durable pin fails closed without creating a child", missing_ok,
                f"HTTP {status} failure={missing_pin.get('failure')} error={error}")

    manifests = _pin_sources(client, (run_1, run_2), rows)
    cleanup_runs.extend(manifests)
    if len(manifests) != 2:
        return
    recovered = _restart_worker(port, generation, rows, args.startup_timeout)
    if recovered is None:
        return

    _stale_generation_probe(port, run_2, rows)
    latest_receipt, _latest_child = _continue(
        client, run_2, 1, rows,
        "latest turn restores after restart and appends without historical recompute",
    )
    historical_receipt, _historical_child = _continue(
        client, run_2, 0, rows,
        "earlier organic turn resolves its own pin and continues after restart",
    )
    historical_source = (historical_receipt.get("source") or {}).get("source_run_id")
    _append_row(rows, "historical receipt distinguishes requested/source run provenance",
                historical_source == run_1 and historical_source != run_2,
                f"requested={run_2!r} source={historical_source!r}")

    _, parent_after, _ = client.get(f"/runs/{run_2}")
    _append_row(rows, "requested parent remains byte-for-byte immutable", parent_after == parent_before,
                f"parent_run_id={run_2}")
    _cancellation_probe(port, run_2, rows)


def run_battery(args) -> tuple[list[dict], dict]:
    rows: list[dict] = []
    model = _find_model(args.model)
    if model is None:
        rows.append(_row("AR GGUF model is available", SKIP,
                         f"no model found for --model={args.model!r}"))
        return rows, {}

    from clozn.cli import main as cli
    from clozn.cli.engine_process import _free_port, find_engine_ex

    environment = os.environ.copy()
    if args.engine:
        engine = os.path.abspath(args.engine)
        if not os.path.isfile(engine):
            rows.append(_row("worker build is available", SKIP, f"engine not found: {engine}"))
            return rows, {}
        environment["CLOZN_ENGINE_BIN"] = engine
        environment["CLOZN_ENGINE_GPU"] = "1" if args.gpu else "0"
    if args.engine:
        discovery = args.engine
    else:
        try:
            discovery = find_engine_ex(prefer_gpu=args.gpu)
        except cli.CloznError as exc:
            rows.append(_row("worker build is available", SKIP, str(exc)))
            return rows, {}

    port = args.port or _free_port()
    temporary = tempfile.TemporaryDirectory(prefix="clozn-tmc-gateway-smoke-")
    log_path = os.path.join(temporary.name, "serve.log")
    log = open(log_path, "w", encoding="utf-8")
    command = [sys.executable, "-m", "clozn", "serve", model, "--port", str(port),
               "--ctx", str(args.ctx)]
    if not args.gpu:
        command.append("--cpu")
    process = subprocess.Popen(
        command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT,
        start_new_session=(os.name != "nt"),
    )
    log.close()
    cleanup_runs: list[str] = []
    ready = None
    try:
        ready, problem = _wait_ready(port, process, args.startup_timeout)
        if ready is None:
            rows.append(_row("clozn serve boots", FAIL, f"{problem}; log={_tail(log_path)}"))
            return rows, {}
        rows.append(_row("clozn serve boots", PASS,
                         f"port={port} model={model} engine={args.engine or discovery}"))
        session_id = "tmc-smoke-" + time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        client = Client(port, session_id=session_id, timeout=args.timeout)
        _exercise(rows, client, port, ready, args, cleanup_runs)
    finally:
        client = Client(port, timeout=10)
        for run_id in cleanup_runs:
            status, _reply, error = client.post(f"/snapshots/{run_id}/unpin", {"cascade": True})
            rows.append(_row(f"source pin cleanup for {run_id}", PASS if status == 200 else FAIL,
                             f"HTTP {status} error={error}"))
        _teardown(port, process)
        temporary.cleanup()
    return rows, ready or {}


def _print(rows: list[dict]) -> None:
    print("\n" + "=" * 112)
    for row in rows:
        print(f"{row['status']:<4}  {row['name']}")
        print(f"      {row['detail'][:800]}")
    counts = {name: sum(row["status"] == name for row in rows) for name in (PASS, FAIL, SKIP)}
    print("=" * 112)
    print(f"{counts[PASS]} PASS, {counts[FAIL]} FAIL, {counts[SKIP]} SKIP")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--engine")
    parser.add_argument("--port", type=int)
    parser.add_argument("--ctx", type=int, default=1024)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--startup-timeout", type=float, default=120.0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--tag", default="live")
    args = parser.parse_args(argv)

    rows, ready = run_battery(args)
    counts = {name: sum(row["status"] == name for row in rows) for name in (PASS, FAIL, SKIP)}
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = os.path.join(REPO, "runs", "experiments")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"time_machine_continuation_gateway_smoke_{args.tag}_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema": "clozn.time-machine-continuation-gateway-smoke.v1",
            "generated_at": timestamp,
            "pass": counts[PASS], "fail": counts[FAIL], "skip": counts[SKIP],
            "worker": deepcopy(ready.get("worker") or {}),
            "rows": rows,
        }, handle, indent=2, ensure_ascii=False)

    _print(rows)
    print(f"\nJSON summary -> {output_path}")
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    raise SystemExit(main())
