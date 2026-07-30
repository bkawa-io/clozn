"""managed_runtime_smoke.py -- the first LIVE acceptance battery for the managed preloaded
multi-model runtime (RT-BOOT-01 / ADR 004, docs/design/004-multi-model-routing-contract.md).

Two large waves landed on origin/main@8ec835f with model-free coverage only. This script is the
gap-closer docs/HANDOFF_2026-07-29.md section 3 names explicitly: "No live two-GGUF managed smoke
... was completed." It drives the REAL `clozn serve --models-config` / `clozn ps` / `clozn stop`
CLI boundary -- the same external process boundary a user gets -- against two real small GGUFs and
the real GPU engine build, not a fake/mocked worker.

WHAT THIS PROVES
-----------------
1. A hand-qualified `clozn.managed-models.v1` manifest (built live from two real GGUFs' own SHA-256,
   the real engine executable's SHA-256, and each model's own live template-fingerprint probe -- the
   exact same qualification facts `clozn.cli.runtime_process` computes) boots TWO independent private
   workers under ONE public gateway.
2. Generation through the NON-DEFAULT model, on all three protocol surfaces (native, OpenAI, Ollama),
   is served by the worker actually keyed to that model -- not silently by the default.
3. The persisted run record (GET /runs/<id>) carries the EXACT resolved model id, GGUF SHA-256, and
   runtime-key SHA-256 that served it -- never the default's, never a friendly label that merely
   happens to look right. This is the specific claim ADR 004 exists to make impossible to break:
   "a run unable to claim a model it did not execute on."
4. Killing one worker's OS process triggers independent supervisor-side recovery (a NEW worker
   generation, same canonical model id) while the OTHER worker keeps serving throughout, unaffected.
5. `clozn ps` reports the managed gateway sensibly and `clozn stop` tears down every nested PID
   (gateway + both workers), leaving no live process, no open port, and no registry row behind.

HONEST DEGRADE
---------------
Needs the real GPU engine build and both small GGUFs on disk; neither is guaranteed present. Every
such gap is reported as SKIPPED with an explicit reason and a 0 exit code -- never a FAIL and never
silently. A genuine divergence from the claims above (wrong identity served, a run claiming the
wrong model, a worker that doesn't recover, a leaked PID/port after stop) is a HARD FAIL naming the
exact observed value. Nothing here is tuned to make the battery pass.

Usage:
    python scripts/smoke/managed_runtime_smoke.py
    python scripts/smoke/managed_runtime_smoke.py --port 8181 --tag managed
"""
from __future__ import annotations

import argparse
import json
import os
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
LLAMA_1B = os.path.join(MODELS_DIR, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
QWEN_05B = os.path.join(MODELS_DIR, "qwen2.5-0.5b-instruct-q4_k_m.gguf")
DEFAULT_ID = "smoke-llama1b"
OTHER_ID = "smoke-qwen05"


def _row(battery: str, name: str, status: str, detail: str) -> dict:
    assert status in (PASS, FAIL, SKIP)
    return {"battery": battery, "name": name, "status": status, "detail": detail}


# =================================================================================== HTTP client

class Client:
    """Minimal urllib client mirroring clozn.cli.commands.smoke.Client's own shape, kept
    independent (this script owns no product code) but deliberately unsurprising."""

    def __init__(self, base: str, timeout: float):
        self.base = base.rstrip("/")
        self.timeout = float(timeout)

    def request(self, method: str, path: str, body: dict | None = None):
        encoded = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self.base + path, data=encoded, method=method,
            headers={"Content-Type": "application/json"} if encoded is not None else {},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read()
                return int(resp.status), raw, ""
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

    def post(self, path: str, body: dict | None = None):
        return self.request("POST", path, body or {})


def _j(raw: bytes) -> dict:
    try:
        value = json.loads(raw.decode("utf-8", "replace") or "{}")
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


# =================================================================================== qualification

def _qualify(model_path: str, port: int, engine_build_sha: str, backend: str):
    """Boot a throwaway probe worker for `model_path` to learn its EXACT live runtime facts (n_ctx,
    device, canonical template fingerprint), then build the immutable RuntimeKey/WorkerDefinition
    ADR 004 needs. Reuses runtime_process._worker_template_fingerprint -- the SAME probe the real
    supervisor runs -- rather than reimplementing it slightly differently and risking a fingerprint
    that fails to qualify on the real managed boot."""
    from clozn.cli.commands.models import _flags_for
    from clozn.cli.engine_process import spawn_engine, _terminate_process
    from clozn.cli.runtime_process import _worker_template_fingerprint
    from clozn.cli.worker_registry import AdapterRuntimeIdentity, RuntimeKey, WorkerDefinition
    from clozn.runs.identity import model_sha256

    flags = dict(_flags_for(model_path))
    flags["_disable_auto_jlens"] = True   # keep the probe fast/deterministic; managed boot always sets this too
    proc, health, gpu = spawn_engine(model_path, port, flags, prefer_gpu=True)
    try:
        fingerprint = _worker_template_fingerprint(port)
        n_ctx = int(health["n_ctx"])
        device = str(health.get("device") or "")
    finally:
        _terminate_process(proc)

    flags.pop("_disable_auto_jlens", None)
    flags["ctx"] = n_ctx
    runtime_key = RuntimeKey(
        gguf_artifact_sha256=model_sha256(model_path),
        context_size=n_ctx,
        backend=backend,
        adapter=AdapterRuntimeIdentity.absent(),
        template_fingerprint=fingerprint,
        engine_build=f"sha256:{engine_build_sha}",
        white_box_flags={"sae": False, "jlens": False, "attn_knockout": False},
    )
    return flags, runtime_key, gpu, device


def build_manifest(rows: list, tmp_dir: str, free_port) -> "tuple[str, object, object] | None":
    """Qualify both small models and write a real clozn.managed-models.v1 manifest. Returns
    (manifest_path, default_definition, other_definition) or None on an environmental SKIP (rows
    records the reason either way)."""
    if not (os.path.isfile(LLAMA_1B) and os.path.isfile(QWEN_05B)):
        missing = [p for p in (LLAMA_1B, QWEN_05B) if not os.path.isfile(p)]
        rows.append(_row("setup", "both small GGUFs present", SKIP,
                         f"missing: {missing}. Battery 1 needs both {os.path.basename(LLAMA_1B)} "
                         f"and {os.path.basename(QWEN_05B)} under {MODELS_DIR}"))
        return None

    from clozn.cli import main as ctx
    from clozn.artifacts.contracts import sha256_file
    from clozn.cli.engine_process import find_engine_ex

    try:
        discovery = find_engine_ex(prefer_gpu=True)
    except ctx.CloznError as exc:
        rows.append(_row("setup", "engine build discovered", SKIP, str(exc)))
        return None
    engine_sha = sha256_file(os.path.abspath(discovery.exe))
    backend = discovery.backend or ("gpu" if discovery.gpu else "cpu")
    rows.append(_row("setup", "engine build discovered", PASS,
                     f"{discovery.exe} (backend={backend}, source={discovery.discovery_source})"))

    try:
        default_flags, default_key, default_gpu, default_device = _qualify(
            LLAMA_1B, free_port(), engine_sha, backend)
        rows.append(_row("setup", "qualify default model (llama-1b)", PASS,
                         f"n_ctx={default_key.context_size} device={default_device} gpu={default_gpu} "
                         f"template_fingerprint={default_key.template_fingerprint} "
                         f"gguf_sha256={default_key.gguf_artifact_sha256}"))
        other_flags, other_key, other_gpu, other_device = _qualify(
            QWEN_05B, free_port(), engine_sha, backend)
        rows.append(_row("setup", "qualify non-default model (qwen-0.5b)", PASS,
                         f"n_ctx={other_key.context_size} device={other_device} gpu={other_gpu} "
                         f"template_fingerprint={other_key.template_fingerprint} "
                         f"gguf_sha256={other_key.gguf_artifact_sha256}"))
    except Exception as exc:
        rows.append(_row("setup", "qualify both models", SKIP,
                         f"could not qualify test fixtures (probe boot failed, not a product claim): "
                         f"{type(exc).__name__}: {exc}"))
        return None

    if default_key.key_sha256 == other_key.key_sha256:
        rows.append(_row("setup", "the two runtime keys are distinct", FAIL,
                         f"both models qualified to the SAME runtime_key_sha256 "
                         f"({default_key.key_sha256}) -- the manifest cannot distinguish them"))
        return None
    rows.append(_row("setup", "the two runtime keys are distinct", PASS,
                     f"default={default_key.key_sha256} other={other_key.key_sha256}"))

    from clozn.cli.worker_registry import WorkerDefinition
    default_def = WorkerDefinition(model_id=DEFAULT_ID, model=LLAMA_1B, runtime_key=default_key,
                                   flags=default_flags, prefer_gpu=True)
    other_def = WorkerDefinition(model_id=OTHER_ID, model=QWEN_05B, runtime_key=other_key,
                                 flags=other_flags, prefer_gpu=True)

    manifest = {
        "schema_version": "clozn.managed-models.v1",
        "default_model_id": DEFAULT_ID,
        "preload_model_ids": [DEFAULT_ID, OTHER_ID],
        "max_loaded_models": 2,
        "models": [
            {"model_id": d.model_id, "model": d.model, "runtime_key": d.runtime_key.as_dict(),
             "flags": dict(d.flags), "prefer_gpu": d.prefer_gpu}
            for d in (default_def, other_def)
        ],
    }
    manifest_path = os.path.join(tmp_dir, "managed-models.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    from clozn.cli.managed_models import ManagedModelsConfigError, load_managed_models
    try:
        loaded = load_managed_models(manifest_path)
    except ManagedModelsConfigError as exc:
        rows.append(_row("setup", "manifest round-trips through load_managed_models", FAIL, str(exc)))
        return None
    rows.append(_row("setup", "manifest round-trips through load_managed_models", PASS,
                     f"default={loaded.default_model_id} preload={list(loaded.preload_model_ids)} "
                     f"max_loaded={loaded.max_loaded_models}"))
    return manifest_path, default_def, other_def


# =================================================================================== boot / teardown

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


# =================================================================================== identity assertion

def _assert_identity(rows: list, surface: str, run: dict, *, expected_id: str, forbidden_id: str,
                     expected_sha: str, forbidden_sha: str) -> "str | None":
    """The load-bearing check: the PERSISTED run claims the model that actually ran it, never the
    default and never merely a plausible label. Returns a failure detail string, or None on success."""
    problems = []
    if run.get("model") != expected_id:
        problems.append(f"run['model']={run.get('model')!r}, expected {expected_id!r}")
    if run.get("model") == forbidden_id:
        problems.append(f"run['model'] equals the DEFAULT model id {forbidden_id!r}")
    identity = run.get("identity") if isinstance(run.get("identity"), dict) else {}
    sha = identity.get("model_sha256")
    if sha != expected_sha:
        problems.append(f"identity.model_sha256={sha!r}, expected {expected_sha!r}")
    if sha == forbidden_sha:
        problems.append(f"identity.model_sha256 equals the DEFAULT model's GGUF SHA-256 {forbidden_sha!r}")
    meta = run.get("meta") if isinstance(run.get("meta"), dict) else {}
    routing = meta.get("model_routing") if isinstance(meta.get("model_routing"), dict) else {}
    receipt = (((routing.get("result") or {}).get("receipt")) or {}) if routing else {}
    resolved_id = receipt.get("resolved_model_id")
    resolved_artifact = receipt.get("resolved_artifact") or {}
    resolved_sha = resolved_artifact.get("artifact_sha256")
    if resolved_id != expected_id:
        problems.append(f"meta.model_routing receipt resolved_model_id={resolved_id!r}, expected {expected_id!r}")
    if resolved_sha != expected_sha:
        problems.append(f"meta.model_routing receipt artifact_sha256={resolved_sha!r}, expected {expected_sha!r}")
    if problems:
        return f"[{surface}] " + "; ".join(problems)
    return None


# =================================================================================== main battery

def run_battery(args) -> tuple[list, dict]:
    rows: list = []
    metrics: dict = {}

    from clozn.cli.engine_process import _free_port
    tmp = tempfile.TemporaryDirectory(prefix="clozn-managed-smoke-")
    try:
        built = build_manifest(rows, tmp.name, _free_port)
        if built is None:
            return rows, metrics
        manifest_path, default_def, other_def = built

        port = args.port or _free_port()
        base = f"http://127.0.0.1:{port}"
        log_path = os.path.join(tmp.name, "serve.log")
        log = open(log_path, "w", encoding="utf-8")
        command = [sys.executable, "-m", "clozn", "serve", "--models-config", manifest_path,
                   "--port", str(port)]
        started = time.monotonic()
        import subprocess
        proc = subprocess.Popen(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=(os.name != "nt"),
                                creationflags=(subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0))
        log.close()
        try:
            ready, problem = _wait_ready(port, proc, args.startup_timeout)
            metrics["startup_seconds"] = round(time.monotonic() - started, 3)
            if not ready:
                rows.append(_row("1", "clozn serve --models-config boots", FAIL,
                                 f"{problem}; log: {_tail(log_path)}"))
                return rows, metrics
            rows.append(_row("1", "clozn serve --models-config boots", PASS,
                             f"port={port} in {metrics['startup_seconds']}s"))

            client = Client(base, args.timeout)
            _exercise(rows, metrics, client, port, default_def, other_def, args)
        finally:
            _teardown(rows, port, proc)
    finally:
        tmp.cleanup()
    return rows, metrics


def _exercise(rows, metrics, client: Client, port: int, default_def, other_def, args) -> None:
    # --- both workers up under one gateway --------------------------------------------------
    status, raw, err = client.get("/readyz")
    doc = _j(raw)
    managed = doc.get("models") or {}
    both_ready = (status == 200 and doc.get("status") == "ok"
                 and managed.get("resident_count") == 2
                 and managed.get("default_model_id") == default_def.model_id)
    rows.append(_row("1", "/readyz reports two resident managed workers", PASS if both_ready else FAIL,
                     f"HTTP {status} resident_count={managed.get('resident_count')} "
                     f"default={managed.get('default_model_id')} err={err}"))

    status, raw, err = client.get("/runtime/models")
    doc = _j(raw)
    by_id = {m.get("model_id"): m for m in (doc.get("models") or []) if isinstance(m, dict)}
    keys_ok = (
        status == 200 and doc.get("managed") is True
        and by_id.get(default_def.model_id, {}).get("state") == "ready"
        and by_id.get(other_def.model_id, {}).get("state") == "ready"
        and by_id.get(default_def.model_id, {}).get("runtime_key_sha256") == default_def.runtime_key.key_sha256
        and by_id.get(other_def.model_id, {}).get("runtime_key_sha256") == other_def.runtime_key.key_sha256
    )
    rows.append(_row("1", "/runtime/models: both workers ready with the exact configured runtime keys",
                     PASS if keys_ok else FAIL, f"HTTP {status} models={by_id}"))
    # /runtime/models must never disclose a private worker port (ADR 004 privacy invariant).
    wire = json.dumps(doc)
    port_leak = "worker_port" in wire
    rows.append(_row("1", "/runtime/models never discloses a private worker port",
                     FAIL if port_leak else PASS, "worker_port key present in response" if port_leak
                     else "no worker_port key in response body"))

    original_default_generation = by_id.get(default_def.model_id, {}).get("worker_generation")

    # --- generate through the NON-DEFAULT model on all three surfaces -----------------------
    prompt_text = "Reply with the single word ready."
    native_body = {"model": other_def.model_id, "prompt": prompt_text, "max_tokens": 12,
                  "temperature": 0, "stream": False}
    status, raw, err = client.post("/api/clozn/generate", native_body)
    doc = _j(raw)
    native_routing = doc.get("clozn_model_routing") or {}
    native_receipt = (((native_routing.get("result") or {}).get("receipt")) or {})
    native_resolved_ok = (status == 200
                          and native_receipt.get("resolved_model_id") == other_def.model_id
                          and (native_receipt.get("resolved_artifact") or {}).get("artifact_sha256")
                          == other_def.runtime_key.gguf_artifact_sha256)
    native_run_id = doc.get("clozn_run_id")
    rows.append(_row("1", "native /api/clozn/generate routes to the non-default model",
                     PASS if native_resolved_ok else FAIL,
                     f"HTTP {status} resolved_model_id={native_receipt.get('resolved_model_id')} "
                     f"run_id={native_run_id!r} err={err}"))
    if native_resolved_ok and not native_run_id:
        rows.append(_row("1", "native surface persists a run for its routed generation", FAIL,
                         "response carried a correct clozn_model_routing receipt but no clozn_run_id "
                         "-- native /api/clozn/generate proxies straight to the private worker's own "
                         "/v1/completions and does not call runlog.record; there is no persisted run "
                         "for this surface to make an identity claim about at all (a real, notable "
                         "finding, reported here rather than silently skipped)"))

    openai_body = {"model": other_def.model_id,
                  "messages": [{"role": "user", "content": prompt_text}],
                  "max_tokens": 12, "temperature": 0}
    status, raw, err = client.post("/v1/chat/completions", openai_body)
    doc = _j(raw)
    openai_run_id = doc.get("clozn_run_id")
    openai_reply = ((doc.get("choices") or [{}])[0].get("message") or {}).get("content")
    rows.append(_row("1", "OpenAI /v1/chat/completions generates through the non-default model",
                     PASS if status == 200 and openai_run_id and openai_reply else FAIL,
                     f"HTTP {status} run_id={openai_run_id!r} reply={openai_reply!r} err={err}"))
    if openai_run_id:
        _check_persisted_identity(rows, "1", client, "openai /v1/chat/completions", openai_run_id,
                                  default_def, other_def)

    ollama_body = {"model": other_def.model_id, "messages": [{"role": "user", "content": prompt_text}],
                  "stream": False, "options": {"temperature": 0}}
    status, raw, err = client.post("/api/chat", ollama_body)
    doc = _j(raw)
    ollama_run_id = doc.get("clozn_run_id")
    ollama_reply = (doc.get("message") or {}).get("content")
    rows.append(_row("1", "Ollama /api/chat generates through the non-default model",
                     PASS if status == 200 and ollama_run_id and ollama_reply else FAIL,
                     f"HTTP {status} run_id={ollama_run_id!r} reply={ollama_reply!r} err={err}"))
    if ollama_run_id:
        _check_persisted_identity(rows, "1", client, "ollama /api/chat", ollama_run_id,
                                  default_def, other_def)

    # --- kill one worker; confirm independent restart AND the other keeps serving ----------
    from clozn.cli.engine_process import _kill, _pid_alive, _reg_read
    entry = _reg_read().get(str(port)) or {}
    default_pid = None
    for m in entry.get("models") or []:
        if isinstance(m, dict) and m.get("model_id") == default_def.model_id:
            default_pid = m.get("worker_pid")
    if not default_pid:
        rows.append(_row("1", "killed-worker recovery", SKIP,
                         "could not read the default worker's PID from the local runtime registry"))
    else:
        killed_at = time.monotonic()
        _kill(int(default_pid))

        # The OTHER worker must keep serving RIGHT AWAY, unaffected by the sibling's death.
        status, raw, err = client.post("/v1/chat/completions",
                                       {"model": other_def.model_id,
                                        "messages": [{"role": "user", "content": prompt_text}],
                                        "max_tokens": 8, "temperature": 0})
        doc = _j(raw)
        survives = status == 200 and bool(((doc.get("choices") or [{}])[0].get("message") or {}).get("content"))
        rows.append(_row("1", "the OTHER worker keeps serving immediately after its sibling is killed",
                         PASS if survives else FAIL,
                         f"HTTP {status} within {time.monotonic()-killed_at:.2f}s of kill; err={err}"))

        recovered = None
        deadline = time.monotonic() + args.startup_timeout
        while time.monotonic() < deadline:
            status, raw, _ = client.get("/runtime/models")
            doc = _j(raw)
            by_id2 = {m.get("model_id"): m for m in (doc.get("models") or []) if isinstance(m, dict)}
            row = by_id2.get(default_def.model_id) or {}
            if (row.get("state") == "ready"
                    and row.get("worker_generation") != original_default_generation):
                recovered = row
                break
            time.sleep(0.5)
        metrics["worker_restart_seconds"] = round(time.monotonic() - killed_at, 3)
        rows.append(_row("1", "the killed worker independently recovers to a NEW process generation",
                         PASS if recovered else FAIL,
                         f"original_generation={original_default_generation} "
                         f"recovered={recovered} within {metrics['worker_restart_seconds']}s"))
        if recovered:
            status, raw, err = client.post("/v1/chat/completions",
                                           {"model": default_def.model_id,
                                            "messages": [{"role": "user", "content": prompt_text}],
                                            "max_tokens": 8, "temperature": 0})
            doc = _j(raw)
            works = status == 200 and bool(((doc.get("choices") or [{}])[0].get("message") or {}).get("content"))
            rows.append(_row("1", "the recovered default worker serves a real generation again",
                             PASS if works else FAIL, f"HTTP {status} err={err}"))
            still_default_alive = default_pid and _pid_alive(int(default_pid))
            rows.append(_row("1", "the OLD worker PID is actually gone after recovery",
                             FAIL if still_default_alive else PASS,
                             f"old pid={default_pid} alive={bool(still_default_alive)}"))


def _check_persisted_identity(rows, battery, client: Client, surface: str, run_id: str,
                              default_def, other_def) -> None:
    status, raw, err = client.get(f"/runs/{run_id}")
    doc = _j(raw)
    if status != 200 or doc.get("id") != run_id:
        rows.append(_row(battery, f"persisted run resolves over the gateway [{surface}]", FAIL,
                         f"HTTP {status} err={err}"))
        return
    problem = _assert_identity(
        rows, surface, doc,
        expected_id=other_def.model_id, forbidden_id=default_def.model_id,
        expected_sha=other_def.runtime_key.gguf_artifact_sha256,
        forbidden_sha=default_def.runtime_key.gguf_artifact_sha256,
    )
    rows.append(_row(battery, f"persisted run's identity is EXACTLY the non-default worker [{surface}]",
                     FAIL if problem else PASS,
                     problem or (f"run['model']={doc.get('model')!r} "
                                f"identity.model_sha256={(doc.get('identity') or {}).get('model_sha256')!r} "
                                "-- matches the non-default model, not the default")))


def _teardown(rows: list, port: int, proc) -> None:
    from clozn.cli.engine_process import _pid_alive, _reg_read
    entry = _reg_read().get(str(port)) or {}
    worker_pids = [m.get("worker_pid") for m in (entry.get("models") or []) if isinstance(m, dict)]

    import subprocess
    try:
        ps_result = subprocess.run([sys.executable, "-m", "clozn", "ps"], cwd=REPO,
                                   capture_output=True, text=True, timeout=15)
        ps_out = ps_result.stdout
    except Exception as exc:
        ps_out = f"<clozn ps raised: {exc}>"
    ps_ok = str(port) in ps_out
    rows.append(_row("1", "clozn ps reports the managed gateway", PASS if ps_ok else FAIL,
                     ps_out.strip()[:300]))

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

    from clozn.cli.engine_process import _await_dead
    all_pids = [p for p in ([entry.get("pid"), entry.get("gateway_pid")] + worker_pids) if p]
    _await_dead(all_pids, timeout=8.0)
    still_alive = [p for p in all_pids if _pid_alive(p)]
    registry_clear = str(port) not in _reg_read()
    supervisor_dead = proc.poll() is not None
    clean = not still_alive and registry_clear and supervisor_dead
    rows.append(_row("1", "clozn stop tears down every nested PID (gateway + both workers)",
                     PASS if clean else FAIL,
                     f"still_alive_pids={still_alive} registry_clear={registry_clear} "
                     f"supervisor_dead={supervisor_dead}"))


def _print_table(rows: list) -> None:
    print("\n" + "=" * 116)
    print(f"{'#':<3} {'name':<70} status")
    print("-" * 116)
    for r in rows:
        print(f"{r['battery']:<3} {r['name'][:68]:<70} {r['status']}")
        print(f"    -> {r['detail'][:400]}")
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
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=60.0, help="per-request timeout (s)")
    ap.add_argument("--startup-timeout", type=float, default=120.0, help="boot/restart timeout (s)")
    ap.add_argument("--tag", default="managed", help="output filename tag")
    args = ap.parse_args(argv)

    rows, metrics = run_battery(args)

    # Write the JSON summary BEFORE printing: a console-encoding crash in the human-readable table
    # must never discard an already-collected live result.
    out_dir = os.path.join(REPO, "runs", "experiments")
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_path = os.path.join(out_dir, f"managed_runtime_smoke_{args.tag}_{ts}.json")
    n_pass = sum(1 for r in rows if r["status"] == PASS)
    n_fail = sum(1 for r in rows if r["status"] == FAIL)
    n_skip = sum(1 for r in rows if r["status"] == SKIP)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"schema": "clozn.managed_runtime_smoke_report.v1", "generated_at": ts,
                   "pass": n_pass, "fail": n_fail, "skip": n_skip, "rows": rows, "metrics": metrics},
                  f, indent=2, default=str)

    _print_table(rows)
    if metrics:
        print("\nmetrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v}")
    print(f"\nJSON summary -> {out_path}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
