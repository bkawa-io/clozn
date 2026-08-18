"""Q3--Q8 qualification orchestration.

The product boundary stays deliberately boring: this module is stdlib-only and never imports
Torch or Transformers.  It can prove the core runtime identity and (when explicitly requested)
run the existing product smoke through the normal gateway.  Lab work is represented as typed
steps and may only be marked passed by an explicit external command or an artifact that passes
the existing fail-closed artifact contracts.

The returned document is a receipt, not a marketing claim.  ``qualification_status`` is
``not_qualified`` until every required capability has evidence; a successful core smoke does not
implicitly qualify J-lenses or white-box features.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence
import urllib.request

from clozn import schemas
from clozn.artifacts import contracts
from clozn._io import atomic_write_json

RUN_SCHEMA = "clozn.qualification-run.v1"
_STEP_STATUSES = {"passed", "failed", "blocked", "not_run", "unavailable", "cancelled"}


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_json(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _step(step_id: str, boundary: str, status: str, *, evidence: Any = None,
          reason: str | None = None, started_at: str | None = None,
          finished_at: str | None = None) -> dict[str, Any]:
    if status not in _STEP_STATUSES:
        raise ValueError(f"unknown qualification step status: {status}")
    value: dict[str, Any] = {"id": step_id, "boundary": boundary, "status": status}
    if evidence is not None:
        value["evidence"] = evidence
    if reason:
        value["reason"] = reason
    if started_at:
        value["started_at"] = started_at
    if finished_at:
        value["finished_at"] = finished_at
    return value


def _identity_evidence(identity: Mapping[str, Any] | None) -> tuple[str, dict[str, Any]]:
    if not isinstance(identity, Mapping):
        return "failed", {"reason": "GGUF identity could not be read"}
    required = ("sha256", "architecture", "hidden_size", "layer_count", "vocab_size",
                "tokenizer_sha256", "chat_template_sha256")
    missing = [field for field in required if identity.get(field) in (None, "")]
    if missing:
        return "failed", {"reason": "identity is incomplete", "missing": missing}
    return "passed", {
        "model_sha256": str(identity["sha256"]).lower(),
        "architecture": identity["architecture"],
        "hidden_size": identity["hidden_size"],
        "layer_count": identity["layer_count"],
        "vocab_size": identity["vocab_size"],
        "tokenizer_sha256": identity["tokenizer_sha256"],
        "chat_template_sha256": identity["chat_template_sha256"],
        "quantization": identity.get("quantization"),
        "file_size": identity.get("file_size"),
    }


def _load_identity(model: str, identity: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    if identity is not None:
        return dict(identity), None
    try:
        return contracts.gguf_identity(model), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def _run_live_smoke(model: str, *, timeout: float = 180.0) -> dict[str, Any]:
    """Run one deterministic core request through the product runtime.

    Imports are inside this opt-in path so importing the qualification package remains cheap and
    Torch-free.  The function intentionally records only the stable evidence needed by Q3; it does
    not add a second engine protocol or bypass the gateway.
    """
    from clozn.cli.engine_process import _free_port
    from clozn.cli.runtime_process import RuntimeConfig, spawn_runtime

    stack = None
    started = time.monotonic()
    try:
        port = _free_port()
        stack = spawn_runtime(RuntimeConfig(
            model=model, public_port=port, flags={"ctx": 1024}, prefer_gpu=False,
            gateway_boot_timeout=min(timeout, 60.0), worker_boot_timeout=timeout,
        ))
        body = json.dumps({
            "model": "clozn-qualification",
            "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
            "temperature": 0,
            "max_tokens": 8,
        }).encode("utf-8")
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            generated = json.loads(response.read().decode("utf-8"))
        run_id = generated.get("clozn_run_id")
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("core smoke response did not include a Clozn run ID")
        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/runs/{run_id}/context-receipt", timeout=20
        ) as response:
            receipt = json.loads(response.read().decode("utf-8"))
        shape = receipt.get("shape")
        if shape not in {"new", "legacy"}:
            raise ValueError("core smoke did not produce a readable Context Receipt")
        return {
            "status": "passed",
            "run_id": run_id,
            "receipt_shape": shape,
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
            "backend": (stack.worker_health or {}).get("backend"),
            "protocol_version": (stack.worker_health or {}).get("protocol_version"),
            "warnings": (["CPU smoke only; GPU remains unqualified"] if not stack.gpu else []),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
        }
    finally:
        if stack is not None:
            stack.stop()


def build_run(model: str, *, identity: Mapping[str, Any] | None = None,
              generated_at: str | None = None, live: bool = False,
              live_smoke: Callable[[str], Mapping[str, Any]] | None = None,
              smoke_timeout: float = 180.0,
              jlens: Mapping[str, Any] | None = None,
              batteries: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    """Build and validate one Q3--Q8 receipt.

    ``identity`` and ``live_smoke`` are injectable to keep model-free tests deterministic.  In
    normal use a local GGUF is hashed and ``live=True`` starts the real runtime.  Q5/Q6 inputs
    are already-produced step receipts; this function does not treat a missing lab result as a pass.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty local GGUF path")
    resolved_identity, identity_error = _load_identity(model, identity)
    identity_status, identity_evidence = _identity_evidence(resolved_identity)
    steps: list[dict[str, Any]] = [
        _step("core.identity", "product", identity_status, evidence=identity_evidence,
              reason=identity_error),
    ]
    if identity_status == "passed":
        steps.append(_step("core.template", "product", "passed", evidence={
            "tokenizer_sha256": resolved_identity["tokenizer_sha256"],
            "chat_template_sha256": resolved_identity["chat_template_sha256"],
            "method": "exact GGUF tokenizer metadata identity",
        }))
    else:
        steps.append(_step("core.template", "product", "blocked",
                           reason="model identity is unavailable"))

    if live:
        probe = live_smoke(model) if live_smoke is not None else _run_live_smoke(model, timeout=smoke_timeout)
        probe = dict(probe)
        probe_status = str(probe.pop("status", "failed"))
        steps.append(_step("core.smoke", "product", probe_status,
                           evidence=probe, reason=probe.get("error")))
        receipt_status = "passed" if probe_status == "passed" else "failed"
        steps.append(_step("core.context_receipt", "product", receipt_status,
                           evidence={"run_id": probe.get("run_id"),
                                     "shape": probe.get("receipt_shape")},
                           reason=probe.get("error")))
        steps.append(_step("core.performance", "product", "passed" if probe_status == "passed" else "failed",
                           evidence={"elapsed_ms": probe.get("elapsed_ms"),
                                     "method": "gateway smoke wall clock"},
                           reason=probe.get("error")))
    else:
        steps.extend([
            _step("core.smoke", "product", "not_run", reason="use --run to start the worker"),
            _step("core.context_receipt", "product", "not_run", reason="requires a live core smoke"),
            _step("core.performance", "product", "not_run", reason="requires a live core smoke"),
        ])
    steps.append(_step("structured_io", "product", "not_run",
                       reason="structured I/O requires an exact model/template qualification battery"))
    steps.append(_step("white_box", "product", "not_run",
                       reason="white-box capability requires targeted engine probes"))

    jlens_step = dict(jlens or {})
    if jlens_step:
        steps.append(_step("jlens", "lab", str(jlens_step.get("status", "failed")),
                           evidence=jlens_step.get("evidence"), reason=jlens_step.get("reason")))
    else:
        steps.append(_step("jlens", "lab", "not_run", reason="Q5 J-lens artifact has not been supplied"))
    battery_steps = [dict(item) for item in (batteries or [])]
    if battery_steps:
        battery_status = "passed" if all(item.get("status") == "passed" for item in battery_steps) else "failed"
        steps.append(_step("batteries", "lab", battery_status,
                           evidence={"steps": battery_steps},
                           reason=None if battery_status == "passed" else "one or more battery cells failed"))
    else:
        steps.append(_step("batteries", "lab", "not_run", reason="Q6 battery has not been supplied"))
    steps.append(_step("install", "product", "not_run", reason="Q7 installation is explicit and transactional"))

    core_ids = {"core.identity", "core.template", "core.smoke", "core.context_receipt", "core.performance"}
    core = [item for item in steps if item["id"] in core_ids]
    if any(item["status"] == "failed" for item in core):
        overall = "failed"
    elif all(item["status"] == "passed" for item in core):
        overall = "core_passed"
    else:
        overall = "not_qualified"
    document = {
        "schema_version": RUN_SCHEMA,
        "generated_at": generated_at or _now(),
        "model": {"input": model, "identity": resolved_identity},
        "steps": steps,
        "claims": {
            "qualification_status": overall,
            "generation_performed": bool(live),
            "artifacts_installed": False,
            "note": "Core runtime evidence is separate from lab and white-box qualification.",
        },
        "receipt_sha256": None,
    }
    unsigned = dict(document)
    unsigned["receipt_sha256"] = None
    document["receipt_sha256"] = _sha256_json(unsigned)
    schemas.validate(document, RUN_SCHEMA)
    return document


def run_core(model: str, *, output: str | None = None, live: bool = False,
             smoke_timeout: float = 180.0) -> dict[str, Any]:
    """Run Q3 core qualification and optionally write the immutable receipt."""
    document = build_run(model, live=live, smoke_timeout=smoke_timeout)
    if output:
        atomic_write_json(output, document, indent=2, ensure_ascii=False)
    return document


def default_run_path(model: str) -> str:
    """Return the user-data path used by the CLI when ``--out`` is omitted."""
    stem = Path(str(model).rstrip("/\\")).stem or "model"
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    from clozn.cli import main as ctx
    return os.path.join(ctx.HOME, "qualification-runs", f"{safe}.json")


def run_external_step(step_id: str, argv: Sequence[str], *, cwd: str | None = None,
                      timeout: float = 3600.0, output_dir: str | None = None) -> dict[str, Any]:
    """Execute one explicitly supplied lab command and return a bounded receipt.

    Commands are sequences, never shell strings.  The caller owns the lab environment and is
    responsible for reviewing the generated artifact before installation.
    """
    if not step_id or not isinstance(argv, Sequence) or isinstance(argv, (str, bytes)) or not argv:
        raise ValueError("step_id and a non-empty argv sequence are required")
    started = time.monotonic()
    try:
        completed = subprocess.run(list(argv), cwd=cwd, capture_output=True, text=True,
                                   timeout=timeout, check=False)
    except subprocess.TimeoutExpired as exc:
        return {"id": step_id, "status": "failed", "reason": "lab command timed out",
                "evidence": {"argv": list(argv), "timeout_seconds": timeout,
                             "stdout": str(exc.stdout or "")[-4000:], "stderr": str(exc.stderr or "")[-4000:]}}
    evidence = {
        "argv": list(argv), "returncode": completed.returncode,
        "stdout": completed.stdout[-4000:], "stderr": completed.stderr[-4000:],
        "elapsed_ms": round((time.monotonic() - started) * 1000, 2),
    }
    if output_dir:
        evidence["output_dir"] = os.path.abspath(output_dir)
    return {"id": step_id, "status": "passed" if completed.returncode == 0 else "failed",
            "evidence": evidence,
            "reason": None if completed.returncode == 0 else "lab command returned non-zero"}


def validate_jlens_artifact(model_identity: Mapping[str, Any], artifact_dir: str) -> dict[str, Any]:
    """Validate a Q5 J-lens directory without importing the lab runtime."""
    try:
        manifest_path = Path(artifact_dir) / "manifest.json"
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        checked = contracts.validate_artifact_manifest(manifest, model_identity, artifact_dir,
                                                       expected_type="jlens")
    except Exception as exc:
        return {"id": "jlens", "status": "failed", "reason": f"{type(exc).__name__}: {exc}"}
    return {"id": "jlens", "status": "passed", "evidence": checked}


__all__ = ["RUN_SCHEMA", "build_run", "run_core", "default_run_path", "run_external_step",
           "validate_jlens_artifact"]
