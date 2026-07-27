"""Managed end-to-end acceptance test for the product runtime.

This exercises the same external process boundary a user gets from ``clozn serve``.
It intentionally does not know how to launch the private worker by itself: managed mode
starts the public CLI, discovers the child PIDs through the runtime registry, verifies the
public protocols and SQLite artifacts, restarts the worker, and stops the whole stack.
"""
from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse

from clozn.cli.commands.models import resolve_model
from clozn.cli.engine_process import REPO, _free_port, _kill, _reg_read, find_engine
from clozn.cli.runtime_process import gateway_health, port_is_open


@dataclass
class Check:
    name: str
    status: str
    detail: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    metrics: dict[str, object] = field(default_factory=dict)

    def add(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, "pass" if ok else "fail", detail))
        return ok

    def skip(self, name: str, detail: str) -> None:
        self.checks.append(Check(name, "skip", detail))

    @property
    def ok(self) -> bool:
        return not any(check.status == "fail" for check in self.checks)

    def document(self) -> dict:
        return {
            "ok": self.ok,
            "checks": [asdict(check) for check in self.checks],
            "metrics": self.metrics,
        }

    def render(self) -> str:
        marks = {"pass": "PASS", "fail": "FAIL", "skip": "SKIP"}
        lines = []
        for check in self.checks:
            suffix = f" -- {check.detail}" if check.detail else ""
            lines.append(f"  [{marks[check.status]}] {check.name}{suffix}")
        if self.metrics:
            lines.append("")
            lines.append("  metrics")
            for key, value in self.metrics.items():
                lines.append(f"    {key}: {value}")
        passed = sum(check.status == "pass" for check in self.checks)
        failed = sum(check.status == "fail" for check in self.checks)
        skipped = sum(check.status == "skip" for check in self.checks)
        lines.append("")
        lines.append(f"  {'PASS' if self.ok else 'FAIL'}: {passed} passed, {failed} failed, {skipped} skipped")
        return "\n".join(lines)


@dataclass
class HTTPResult:
    status: int
    body: bytes = b""
    headers: dict[str, str] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str = ""

    def json(self) -> dict:
        try:
            value = json.loads(self.body.decode("utf-8", "replace") or "{}")
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}


@dataclass
class SSEResult:
    status: int
    frames: list[object] = field(default_factory=list)
    done: bool = False
    first_data_s: float | None = None
    elapsed_s: float = 0.0
    error: str = ""


class Client:
    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = float(timeout)

    def request(self, method: str, path: str, body: dict | None = None) -> HTTPResult:
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=encoded,
            method=method,
            headers={"Content-Type": "application/json"} if encoded is not None else {},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read()
                headers = {key.lower(): value for key, value in response.headers.items()}
                return HTTPResult(int(response.status), raw, headers, time.monotonic() - started)
        except urllib.error.HTTPError as exc:
            try:
                raw = exc.read()
            except Exception:
                raw = b""
            headers = {key.lower(): value for key, value in (exc.headers.items() if exc.headers else [])}
            return HTTPResult(int(exc.code), raw, headers, time.monotonic() - started, str(exc))
        except Exception as exc:
            return HTTPResult(0, elapsed_s=time.monotonic() - started, error=str(exc))

    def get(self, path: str) -> HTTPResult:
        return self.request("GET", path)

    def post(self, path: str, body: dict | None = None) -> HTTPResult:
        return self.request("POST", path, body or {})

    def sse(self, path: str, body: dict) -> SSEResult:
        encoded = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.base + path,
            data=encoded,
            method="POST",
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        )
        started = time.monotonic()
        frames: list[object] = []
        first_data = None
        done = False
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                for raw in response:
                    line = raw.decode("utf-8", "replace").strip()
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        done = True
                        break
                    if first_data is None:
                        first_data = time.monotonic() - started
                    try:
                        frames.append(json.loads(payload))
                    except Exception:
                        frames.append(payload)
                    if len(frames) >= 4096:
                        return SSEResult(
                            int(response.status), frames, False, first_data,
                            time.monotonic() - started, "stream exceeded 4096 frames",
                        )
                return SSEResult(
                    int(response.status), frames, done, first_data, time.monotonic() - started
                )
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except Exception:
                detail = str(exc)
            return SSEResult(int(exc.code), frames, done, first_data,
                             time.monotonic() - started, detail)
        except Exception as exc:
            return SSEResult(0, frames, done, first_data, time.monotonic() - started, str(exc))


def _chat_content(payload: dict) -> str:
    try:
        return str(payload["choices"][0]["message"]["content"])
    except Exception:
        return ""


def _completion_stream_text(result: SSEResult) -> tuple[bool, str, str]:
    if result.status != 200:
        return False, "", f"HTTP {result.status}: {result.error}"
    if not result.done:
        return False, "", result.error or "missing [DONE]"
    text = []
    for frame in result.frames:
        if not isinstance(frame, dict):
            return False, "", "non-JSON frame on /v1/completions"
        if "error" in frame:
            return False, "", str(frame["error"])
        if frame.get("object") != "text_completion" or "type" in frame:
            return False, "", f"native frame leaked into /v1/completions: {frame}"
        choices = frame.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False, "", "completion chunk has no choices[0]"
        text.append(str(choices[0].get("text") or ""))
    joined = "".join(text)
    return bool(joined), joined, "standard chunks + [DONE]" if joined else "stream produced no text"


def _chat_stream_text(result: SSEResult) -> tuple[bool, str, str]:
    if result.status != 200:
        return False, "", f"HTTP {result.status}: {result.error}"
    if not result.done:
        return False, "", result.error or "missing [DONE]"
    pieces = []
    finish_seen = False
    stream_id = None
    created = None
    for frame in result.frames:
        if not isinstance(frame, dict):
            return False, "", "non-JSON frame on /v1/chat/completions"
        if "error" in frame:
            return False, "", str(frame["error"])
        if frame.get("object") != "chat.completion.chunk" or "type" in frame:
            return False, "", f"foreign frame leaked into /v1/chat/completions: {frame}"
        if not isinstance(frame.get("created"), int):
            return False, "", "chat chunk has no integer created timestamp"
        if stream_id is None:
            stream_id, created = frame.get("id"), frame.get("created")
        elif frame.get("id") != stream_id or frame.get("created") != created:
            return False, "", "chat chunk identity changed during the stream"
        choices = frame.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            return False, "", "chat chunk has no choices[0]"
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            return False, "", "chat chunk has no delta object"
        if delta.get("content") is not None:
            pieces.append(str(delta["content"]))
        if choice.get("finish_reason"):
            finish_seen = True
    joined = "".join(pieces)
    ok = bool(joined) and finish_seen
    detail = "standard chunks + finish reason + [DONE]" if ok else "stream produced no text or finish reason"
    return ok, joined, detail


def _native_stream_text(result: SSEResult) -> tuple[bool, str, str]:
    if result.status != 200:
        return False, "", f"HTTP {result.status}: {result.error}"
    if not result.done:
        return False, "", result.error or "missing [DONE]"
    custom = False
    pieces = []
    final_text = ""
    for frame in result.frames:
        if not isinstance(frame, dict):
            continue
        if frame.get("type"):
            custom = True
        if frame.get("type") == "tokens_committed":
            pieces.extend(str(item.get("piece") or "") for item in (frame.get("items") or [])
                          if isinstance(item, dict))
        choices = frame.get("choices")
        if isinstance(choices, list) and choices and isinstance(choices[0], dict):
            final_text = str(choices[0].get("text") or final_text)
    text = "".join(pieces) or final_text
    ok = custom and bool(text)
    detail = "native typed events + text + [DONE]" if ok else "missing typed events or generated text"
    return ok, text, detail


def _tail(path: str, limit: int = 3000) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            return handle.read()[-limit:].strip().replace("\n", " ")
    except Exception:
        return ""


def _runtime_entry(port: int, timeout: float = 0.0) -> dict | None:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        value = _reg_read().get(str(port))
        if isinstance(value, dict) and value.get("kind") == "runtime":
            return value
        if time.monotonic() >= deadline:
            return None
        time.sleep(0.1)


def _process_rss_mb(pid) -> float | None:
    try:
        with open(f"/proc/{int(pid)}/status", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except Exception:
        pass
    return None


def _preflight(model: str, cpu: bool, report: Report) -> str | None:
    resolved = None
    try:
        resolved = resolve_model(model)
        report.add("model resolves", True, resolved)
    except Exception as exc:
        report.add("model resolves", False, str(exc))

    engine = None
    try:
        engine, _, gpu = find_engine(prefer_gpu=not cpu)
        report.add("clozn-server build exists", True, f"{engine} ({'GPU' if gpu else 'CPU'})")
    except Exception as exc:
        report.add("clozn-server build exists", False, str(exc))

    studio_root = os.path.abspath(os.path.expanduser(
        os.environ.get("CLOZN_STUDIO_DIR", os.path.join(REPO, "studio"))
    ))
    # The app the gateway redirects "/" to (clozn.server.static.APP_INDEX) -- must match, or the next
    # check ("Studio loads from the product gateway") 404s while this one reports present.
    studio_index = os.path.join(studio_root, "app", "index.html")
    report.add("Studio assets are present", os.path.isfile(studio_index), studio_index)

    if engine is None:
        vendor = os.path.join(REPO, "engine", "core", "third_party", "llama.cpp")
        report.add("pinned llama.cpp source is present", os.path.isdir(vendor),
                   "run: python engine/core/third_party/bootstrap_llama.py")
        cmake = shutil.which("cmake")
        report.add("CMake is available", bool(cmake), cmake or "install CMake >= 3.18")
        compiler = shutil.which("g++") or shutil.which("clang++") or shutil.which("cl")
        report.add("C++ compiler is available", bool(compiler), compiler or "install a C++17 compiler")
    return resolved


def _wait_for_ready(port: int, process: subprocess.Popen, timeout: float) -> tuple[dict | None, str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = gateway_health(port, timeout=1.0)
        if state:
            return state, ""
        if process.poll() is not None:
            return None, f"serve exited with code {process.returncode}"
        time.sleep(0.2)
    return None, f"gateway did not become ready within {timeout:g}s"


def _persistence_checks(report: Report, rid: str, expected_text: str) -> None:
    from clozn.runs import store

    run = store.get_run(rid)
    report.add("chat run persisted in SQLite", bool(run), rid)
    if not run:
        report.add("trace stored as a content-addressed blob", False, "run row is missing")
        return
    response_matches = str(run.get("response") or "") == expected_text
    report.add("persisted response matches API response", response_matches,
               f"source={run.get('source')} substrate={run.get('substrate')}")
    try:
        with closing(sqlite3.connect(store._db_path())) as db:
            row = db.execute("SELECT payload_json FROM runs WHERE id = ?", (rid,)).fetchone()
        payload = json.loads(row[0]) if row else {}
        digest = str((payload.get("trace_ref") or {}).get("sha256") or "")
        blob = store._blob_path(digest) if digest else ""
        actual = ""
        if blob and os.path.isfile(blob):
            hasher = hashlib.sha256()
            with open(blob, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    hasher.update(chunk)
            actual = hasher.hexdigest()
        report.add(
            "trace blob exists and matches recorded SHA-256",
            bool(digest and actual == digest),
            digest if actual == digest else f"recorded={digest or '?'} actual={actual or 'missing'}",
        )
    except Exception as exc:
        report.add("trace blob exists and matches recorded SHA-256", False, str(exc))


def _exercise(base: str, timeout: float, report: Report, *, deep: bool = False) -> str | None:
    client = Client(base, timeout)

    live = client.get("/healthz")
    report.add("gateway liveness", live.status == 200 and live.json().get("status") == "ok",
               f"HTTP {live.status}")

    ready = client.get("/readyz")
    ready_doc = ready.json()
    report.add("gateway readiness includes one worker",
               ready.status == 200 and ready_doc.get("active") == "engine" and bool(ready_doc.get("worker")),
               f"HTTP {ready.status} mode={ready_doc.get('mode')} model={ready_doc.get('model')}")
    queue = ready_doc.get("queue")
    report.add("gateway exposes its bounded request queue",
               isinstance(queue, dict) and int(queue.get("capacity") or 0) > 0
               and int(queue.get("active") or 0) >= 0 and int(queue.get("waiting") or 0) >= 0,
               str(queue or "queue state missing"))

    studio = client.get("/")
    studio_type = studio.headers.get("content-type", "")
    report.add("Studio loads from the product gateway",
               studio.status == 200 and "text/html" in studio_type.lower(),
               f"HTTP {studio.status} content-type={studio_type or '?'}")

    models = client.get("/v1/models")
    model_rows = models.json().get("data") or []
    model_id = str(model_rows[0].get("id")) if model_rows and isinstance(model_rows[0], dict) else ""
    report.add("OpenAI model discovery", models.status == 200 and bool(model_id),
               f"HTTP {models.status} model={model_id or '?'}")

    chat_body = {
        "model": model_id or "clozn-local",
        "messages": [{"role": "user", "content": "Reply with the single word ready."}],
        "max_tokens": 12,
        "temperature": 0,
        "clozn_receipt": False,
    }
    chat = client.post("/v1/chat/completions", chat_body)
    chat_doc = chat.json()
    reply = _chat_content(chat_doc)
    rid = str(chat_doc.get("clozn_run_id") or "")
    report.metrics["chat_seconds"] = round(chat.elapsed_s, 3)
    report.add("OpenAI chat completion", chat.status == 200 and bool(reply),
               f"HTTP {chat.status} text={reply[:60]!r}")
    report.add("chat response exposes its run id", bool(rid), rid or "clozn_run_id missing")
    if rid:
        stored = client.get(f"/runs/{rid}")
        report.add("run resolves over the public gateway", stored.status == 200 and stored.json().get("id") == rid,
                   f"HTTP {stored.status}")
        explained = client.post(f"/runs/{rid}/explain")
        report.add("run explanation remains available",
                   explained.status == 200 and explained.json().get("run_id") == rid,
                   f"HTTP {explained.status}")
        _persistence_checks(report, rid, reply)

    chat_stream = client.sse("/v1/chat/completions", {
        **chat_body,
        "stream": True,
    })
    chat_stream_ok, chat_stream_text, chat_stream_detail = _chat_stream_text(chat_stream)
    report.add("OpenAI chat stream contains only standard chunk envelopes",
               chat_stream_ok, chat_stream_detail)
    report.metrics["chat_stream_ttft_seconds"] = (
        round(chat_stream.first_data_s, 3) if chat_stream.first_data_s is not None else None
    )
    report.metrics["chat_stream_seconds"] = round(chat_stream.elapsed_s, 3)

    completion_stream = client.sse("/v1/completions", {
        "model": model_id or "clozn-local",
        "prompt": "Reply with the single word ready.",
        "max_tokens": 12,
        "temperature": 0,
        "stream": True,
    })
    standard_ok, standard_text, standard_detail = _completion_stream_text(completion_stream)
    report.add("OpenAI completion stream contains only standard chunks", standard_ok, standard_detail)
    report.metrics["openai_stream_ttft_seconds"] = (
        round(completion_stream.first_data_s, 3) if completion_stream.first_data_s is not None else None
    )
    report.metrics["openai_stream_seconds"] = round(completion_stream.elapsed_s, 3)

    native_stream = client.sse("/api/clozn/generate", {
        "prompt": "Reply with the single word ready.",
        "max_tokens": 12,
        "temperature": 0,
        "stream": True,
    })
    native_ok, native_text, native_detail = _native_stream_text(native_stream)
    report.add("native stream preserves typed Clozn events", native_ok, native_detail)
    report.add("all generation protocols produced text",
               bool(chat_stream_text and standard_text and native_text),
               f"chat={chat_stream_text[:30]!r} completion={standard_text[:30]!r} "
               f"native={native_text[:30]!r}")

    if deep and rid:
        forced = client.post(f"/runs/{rid}/receipts", {"mode": "forced"})
        forced_doc = forced.json()
        forced_rows = forced_doc.get("forced_receipts")
        skipped_rows = forced_doc.get("skipped")
        forced_ok = (
            forced.status == 200
            and forced_doc.get("mode") == "forced"
            and isinstance(forced_rows, list)
            and isinstance(skipped_rows, list)
        )
        report.add(
            "forced causal receipts",
            forced_ok,
            f"HTTP {forced.status} mode={forced_doc.get('mode') or '?'} "
            f"receipts={len(forced_rows) if isinstance(forced_rows, list) else '?'} "
            f"skipped={len(skipped_rows) if isinstance(skipped_rows, list) else '?'}",
        )
        replay = client.post(f"/runs/{rid}/replay", {"changes": {"greedy": True}})
        replay_doc = replay.json()
        report.add("run replay produces a child", replay.status == 200 and bool(replay_doc.get("id")),
                   f"HTTP {replay.status} child={replay_doc.get('id') or '?'}")

    return rid or None


def _restart_worker(port: int, client: Client, timeout: float, report: Report) -> None:
    entry = _runtime_entry(port, timeout=2.0)
    if not entry:
        report.add("worker restart recovery", False, "runtime registry entry is missing")
        return
    try:
        old_worker = int(entry["worker_pid"])
        old_gateway = int(entry["gateway_pid"])
    except Exception:
        report.add("worker restart recovery", False, "registry has no worker/gateway PID")
        return

    started = time.monotonic()
    _kill(old_worker)
    saw_not_ready = False
    replacement = None
    deadline = started + timeout
    while time.monotonic() < deadline:
        state = gateway_health(port, timeout=0.5)
        if state is None:
            saw_not_ready = True
        current = _runtime_entry(port)
        if current:
            try:
                new_worker = int(current.get("worker_pid"))
                new_gateway = int(current.get("gateway_pid"))
            except Exception:
                new_worker = new_gateway = 0
            if new_worker and new_worker != old_worker and new_gateway == old_gateway and state:
                replacement = new_worker
                break
        time.sleep(0.1)

    elapsed = time.monotonic() - started
    report.metrics["worker_restart_seconds"] = round(elapsed, 3)
    report.add("worker restart recovery", replacement is not None,
               f"worker {old_worker} -> {replacement or '?'}; gateway {old_gateway} unchanged")
    if replacement is not None:
        post = client.post("/v1/chat/completions", {
            "messages": [{"role": "user", "content": "Reply with ready."}],
            "max_tokens": 8,
            "temperature": 0,
        })
        report.add("generation succeeds after worker restart", post.status == 200 and bool(_chat_content(post.json())),
                   f"HTTP {post.status}; readiness transition observed={saw_not_ready}")


def _pid_alive(pid) -> bool:
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.OpenProcess.argtypes = (ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32)
            kernel32.OpenProcess.restype = ctypes.c_void_p
            kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
            kernel32.GetExitCodeProcess.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
            handle = kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return False
            try:
                exit_code = ctypes.c_uint32()
                if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                    return False
                return exit_code.value == 259  # STILL_ACTIVE
            finally:
                kernel32.CloseHandle(handle)
        except Exception:
            return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _cleanup_state(port: int, process: subprocess.Popen, entry: dict) -> tuple[bool, str]:
    pids = {
        int(value)
        for key in ("pid", "gateway_pid", "worker_pid")
        if (value := entry.get(key)) is not None and str(value).isdigit()
    }
    live_pids = sorted(pid for pid in pids if _pid_alive(pid))
    worker_port = int(entry.get("worker_port") or 0)
    open_ports = [candidate for candidate in (int(port), worker_port)
                  if candidate and port_is_open(candidate)]
    registered = _runtime_entry(port, timeout=0.2) is not None
    supervisor_alive = process.poll() is None
    ok = not registered and not supervisor_alive and not live_pids and not open_ports
    detail = (
        f"registry={'present' if registered else 'clear'} "
        f"supervisor={'alive' if supervisor_alive else 'stopped'} "
        f"live_pids={live_pids or 'none'} open_ports={open_ports or 'none'}"
    )
    return ok, detail


def _wait_for_cleanup(port: int, process: subprocess.Popen, entry: dict,
                      timeout: float = 10.0) -> tuple[bool, str]:
    deadline = time.monotonic() + timeout
    state = (False, "cleanup not checked")
    while time.monotonic() < deadline:
        state = _cleanup_state(port, process, entry)
        if state[0]:
            return state
        time.sleep(0.1)
    return _cleanup_state(port, process, entry)


def _stop_managed(port: int, process: subprocess.Popen) -> dict:
    entry = _runtime_entry(port, timeout=2.0)
    try:
        subprocess.run(
            [sys.executable, "-m", "clozn", "stop", str(port)],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
    except Exception:
        pass
    try:
        process.wait(timeout=8)
        return entry or {}
    except Exception:
        pass
    for key in ("gateway_pid", "worker_pid", "pid"):
        try:
            _kill(int((entry or {}).get(key)))
        except Exception:
            pass
    try:
        if os.name == "nt":
            _kill(process.pid)  # taskkill /T terminates any unregistered descendants too
        else:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        process.wait(timeout=5)
    except Exception:
        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        except Exception:
            pass
    return entry or {}


def _parse_base(url: str) -> tuple[str, int]:
    parsed = urlparse(url if "://" in url else "http://" + url)
    if parsed.scheme != "http" or not parsed.hostname or parsed.path not in ("", "/"):
        raise ValueError("--url must be an HTTP origin such as http://127.0.0.1:8080")
    port = parsed.port or 80
    return f"http://{parsed.hostname}:{port}", port


def cmd_smoke(args):
    """Run preflight or the managed/attach acceptance suite."""
    from clozn.cli import main as ctx

    report = Report()
    if args.url:
        if args.model:
            raise ctx.CloznError("give either MODEL for managed smoke or --url to attach, not both")
        try:
            base, port = _parse_base(args.url)
        except ValueError as exc:
            raise ctx.CloznError(str(exc))
        if args.preflight:
            raise ctx.CloznError("--preflight needs a MODEL; an attached gateway is already built")
        managed = False
        resolved = None
    else:
        if not args.model:
            raise ctx.CloznError("give a MODEL, or attach to an existing gateway with --url")
        resolved = _preflight(args.model, args.cpu, report)
        if args.preflight or not report.ok:
            if args.json:
                print(json.dumps(report.document(), indent=2))
            else:
                print("Clozn product-runtime preflight\n")
                print(report.render())
            return 0 if report.ok else 1
        port = args.port or _free_port()
        base = f"http://127.0.0.1:{port}"
        managed = True

    restart_worker = managed if args.restart_worker is None else bool(args.restart_worker)
    process = None
    log_path = None
    temp = None
    try:
        if managed:
            if gateway_health(port):
                raise ctx.CloznError(f"a Clozn gateway is already ready on port {port}; choose another port")
            temp = tempfile.TemporaryDirectory(prefix="clozn-product-smoke-")
            log_path = os.path.join(temp.name, "serve.log")
            log = open(log_path, "w", encoding="utf-8")
            command = [sys.executable, "-m", "clozn", "serve", resolved, "--port", str(port)]
            if args.cpu:
                command.append("--cpu")
            started = time.monotonic()
            try:
                process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=(os.name != "nt"),
                    creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0),
                )
            finally:
                log.close()  # the child owns its duplicated descriptor from here
            ready, problem = _wait_for_ready(port, process, args.startup_timeout)
            report.metrics["startup_seconds"] = round(time.monotonic() - started, 3)
            if not ready:
                report.add("managed clozn serve startup", False,
                           f"{problem}; {_tail(log_path) or 'no serve output'}")
            else:
                report.add("managed clozn serve startup", True,
                           f"port={port} mode={ready.get('mode')} model={ready.get('model')}")
                entry = _runtime_entry(port, timeout=5.0)
                report.add("serve registered the supervised process tree", bool(entry),
                           f"gateway={entry.get('gateway_pid') if entry else '?'} "
                           f"worker={entry.get('worker_pid') if entry else '?'}")
                if entry:
                    gateway_rss = _process_rss_mb(entry.get("gateway_pid"))
                    worker_rss = _process_rss_mb(entry.get("worker_pid"))
                    if gateway_rss is not None:
                        report.metrics["gateway_rss_mb"] = gateway_rss
                    if worker_rss is not None:
                        report.metrics["worker_rss_mb"] = worker_rss
        else:
            report.add("attached gateway is ready", bool(gateway_health(port)), base)

        if report.ok:
            _exercise(base, args.timeout, report, deep=args.deep)
        if report.ok and restart_worker:
            _restart_worker(port, Client(base, args.timeout), args.startup_timeout, report)
        elif not restart_worker:
            report.skip("worker restart recovery", "disabled (automatic only for managed smoke)")
    finally:
        if managed and process is not None:
            cleanup_entry = _stop_managed(port, process)
            cleanup_ok, cleanup_detail = _wait_for_cleanup(port, process, cleanup_entry)
            report.add("managed runtime cleanup", cleanup_ok, cleanup_detail)
        if temp is not None:
            temp.cleanup()

    if args.json:
        print(json.dumps(report.document(), indent=2))
    else:
        print(f"Clozn product-runtime smoke at {base}\n")
        print(report.render())
    return 0 if report.ok else 1
