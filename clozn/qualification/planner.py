"""Q1/Q2 -- a model-free qualification manifest and planner.

``clozn qualify MODEL --plan`` is intentionally a *plan*, not a qualification claim.  It reads only
GGUF metadata (the existing ``fit_planner`` header reader), estimates resource needs, checks for
optional lab tools by name, and emits a versioned ``clozn.qualification-plan.v1`` document.  No model
weights are loaded, no worker is started, and no heavyweight dependency is imported.

The returned steps keep product-only checks separate from lab-only work.  A later Q3--Q8 runner may
consume this artifact, but this module never changes a qualification registry or installs artifacts.
"""
from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import os
from pathlib import Path
from typing import Any, Mapping

from clozn.cli import fit_planner

PLAN_SCHEMA = "clozn.qualification-plan.v1"

_LAB_DEPENDENCIES = (
    ("torch", "PyTorch", "white-box fitting"),
    ("transformers", "Transformers", "reference-model and calibration probes"),
)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _positive_float(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return fallback
    return parsed if parsed > 0 else fallback


def _dependency_report() -> list[dict[str, Any]]:
    """Return presence only; never import the optional lab packages."""
    result = []
    for module, label, purpose in _LAB_DEPENDENCIES:
        present = importlib.util.find_spec(module) is not None
        result.append({
            "module": module,
            "label": label,
            "present": present,
            "purpose": purpose,
        })
    return result


def _step(step_id: str, label: str, boundary: str, status: str, *, requires: list[str] | None = None,
          reason: str | None = None, estimated: bool = False) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": step_id,
        "label": label,
        "boundary": boundary,
        "status": status,
        "estimated": estimated,
    }
    if requires:
        value["requires"] = list(requires)
    if reason:
        value["reason"] = reason
    return value


def _resource_estimate(header: Mapping[str, Any], *, vram_gb: float, context: int) -> dict[str, Any]:
    size = _nonnegative_int(header.get("file_size_bytes")) or 0
    fit = fit_planner.fit_report(dict(header), size, vram_gb=vram_gb, ctx_for_estimate=context)
    # This is a planning bound, not a measurement: keep the label explicit and round up so the
    # estimate does not suggest that a machine with exactly the file size has no runtime overhead.
    ram_gb = round(size / 1e9 * 1.20, 2) if size else None
    return {
        "disk_bytes": size or None,
        "disk_gb": round(size / 1e9, 2) if size else None,
        "vram_budget_gb": round(vram_gb, 2),
        "estimated_vram_gb": fit.get("est_vram_gb"),
        "estimated_ram_gb": ram_gb,
        "vram_fits_estimate": fit.get("fits") if size else None,
        "context_tokens": context,
        "method": "clozn.fit_planner header math; approximate, not a runtime measurement",
        "note": fit.get("note"),
        "offload_hint": fit.get("offload_hint"),
    }


def _model_summary(header: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        ("name", "name"), ("arch", "architecture"), ("quant", "quantization"),
        ("quant_source", "quantization_source"), ("context_length", "context_length"),
        ("n_layers", "layer_count"), ("embedding_length", "hidden_size"),
        ("head_count", "head_count"), ("head_count_kv", "kv_head_count"),
    )
    return {dest: header[src] for src, dest in fields if header.get(src) is not None}


def build_plan(model: str, header: Mapping[str, Any], *, generated_at: str | None = None,
               vram_gb: float = 16.0, context: int = 8192,
               lab_dependencies: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build and schema-validate one qualification plan from an already-read header.

    Keeping the header as an argument makes the contract deterministic and lets callers/tests use a
    fabricated header without touching a real model.  ``plan_from_model`` owns path/URL resolution.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty path, URL, or model name")
    if not isinstance(header, Mapping):
        raise ValueError("GGUF header must be an object")
    context_value = _nonnegative_int(context)
    if context_value is None or context_value < 1:
        raise ValueError("context must be a positive integer")
    budget = _positive_float(vram_gb, 16.0)
    deps = lab_dependencies if lab_dependencies is not None else _dependency_report()
    if not isinstance(deps, list):
        raise ValueError("lab_dependencies must be a list")
    missing = [str(item.get("label") or item.get("module") or "dependency")
               for item in deps if isinstance(item, Mapping) and item.get("present") is not True]
    source = "url" if str(model).lower().startswith(("http://", "https://")) else "local"
    path = header.get("path")
    if isinstance(path, str) and path:
        source = "local"

    steps = [
        _step("core.identity", "Verify model identity and GGUF architecture", "product", "ready",
              reason="header metadata is available"),
        _step("core.template", "Validate chat template and tokenizer", "product", "ready",
              reason="requires a live worker during Q3; this plan only reserves the check"),
        _step("core.smoke", "Run basic/deep generation and replay smoke", "product", "ready",
              reason="requires a live worker during Q3; no generation was run by this plan"),
        _step("structured_io", "Run structured-output qualification battery", "product", "ready",
              reason="model-specific parser/renderer checks are planned, not executed"),
        _step("influence_provenance", "Check influence and provenance eligibility", "product", "ready",
              reason="white-box capability and attention materialization are verified by Q3"),
        _step("jlens", "Fit and transfer-check a model-scoped J-lens", "lab", "blocked" if missing else "planned",
              requires=["torch", "transformers"],
              reason=(f"missing lab dependencies: {', '.join(missing)}" if missing
                      else "lab dependencies are present; J-lens runner is a later Q5 step")),
        _step("batteries", "Run model-specific and cross-model batteries", "lab", "planned",
              reason="depends on core and optional white-box artifacts"),
        _step("install", "Review and install checksummed qualification artifacts", "product", "planned",
              reason="only Q7 may modify the local qualification registry"),
    ]
    plan = {
        "schema_version": PLAN_SCHEMA,
        "generated_at": generated_at or _now(),
        "model": {
            "input": model,
            "source": source,
            "path": str(path) if isinstance(path, str) and path else None,
            "summary": _model_summary(header),
            "file_size_bytes": _nonnegative_int(header.get("file_size_bytes")),
            "header_bytes_read": _nonnegative_int(header.get("bytes_read")),
        },
        "resources": _resource_estimate(header, vram_gb=budget, context=context_value),
        "lab_dependencies": deps,
        "steps": steps,
        "claims": {
            "qualification_status": "not_qualified",
            "generation_performed": False,
            "artifacts_installed": False,
            "note": "This plan is a model-free readiness report, not qualification evidence.",
        },
    }
    from clozn import schemas
    schemas.validate(plan, PLAN_SCHEMA)
    return plan


def plan_from_model(model: str, *, vram_gb: float = 16.0, context: int = 8192,
                    timeout: float = 30.0) -> dict[str, Any]:
    """Resolve a local path, URL, or known CLI model and read only its GGUF header."""
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty path, URL, or model name")
    value = model.strip()
    if value.lower().startswith(("http://", "https://")):
        header = fit_planner.gguf_header_from_url(value, timeout=timeout)
    else:
        path = value
        if not os.path.isfile(path):
            from clozn.cli.commands.models import resolve_model
            path = resolve_model(value)
        header = fit_planner.gguf_header_from_path(path)
    return build_plan(value, header, vram_gb=vram_gb, context=context)


def default_plan_path(plan: Mapping[str, Any]) -> str:
    model = plan.get("model") if isinstance(plan, Mapping) else {}
    raw = model.get("path") or model.get("input") or "model" if isinstance(model, Mapping) else "model"
    stem = Path(str(raw).rstrip("/\\")).stem or "model"
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in stem)
    from clozn.cli import main as ctx
    return os.path.join(ctx.HOME, "qualification-plans", f"{safe}.json")


__all__ = ["PLAN_SCHEMA", "build_plan", "default_plan_path", "plan_from_model"]
