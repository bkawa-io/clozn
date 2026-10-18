"""Selection and identity checks for Clozn's single private model worker."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import secrets
import time
from typing import Callable, Iterable, Mapping

from clozn import schemas
from clozn.protocol import check_worker_protocol
from clozn.server.request_gate import WorkerGateRegistry


SCHEMA_VERSION = "clozn.model-routing.v1"
_ERRORS = {
    "invalid_model_selection": (400, False, "selection"),
    "unknown_model": (404, False, "resolution"),
    "generation_queue_full": (429, True, "generation_queue"),
    "queue_timeout": (504, True, "generation_queue"),
    "request_cancelled": (499, False, "request"),
    "worker_failed": (502, True, "generation"),
    "worker_identity_mismatch": (502, False, "handshake"),
}
_GATE_OUTCOME_CODES = {
    "full": "generation_queue_full",
    "timeout": "queue_timeout",
    "cancelled": "request_cancelled",
}


class ModelRoutingConfigError(ValueError):
    """The configured single-worker binding is incomplete or contradictory."""


class ModelRoutingError(RuntimeError):
    """One schema-governed selection refusal."""

    def __init__(self, artifact: dict):
        error = artifact["result"]["error"]
        super().__init__(error["message"])
        self.artifact = artifact
        self.code = error["code"]
        self.http_status = error["http_status"]
        self.retryable = error["retryable"]
        self.phase = error["phase"]
        self.lifecycle_state = artifact["result"]["lifecycle_state"]


def _json_copy(value, field_name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ModelRoutingConfigError(f"{field_name} must be an object")
    return deepcopy(dict(value))


@dataclass
class PreloadedModelBinding:
    """The one canonical model and its request-local dispatch objects."""

    model_id: str
    resolved_artifact: dict
    runtime_key: dict
    state: str
    worker_identity: dict | None
    sub: object | None
    engine: object | None
    preloaded: bool = True
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ModelRoutingConfigError("model_id must be a non-empty string")
        self.resolved_artifact = _json_copy(self.resolved_artifact, "resolved_artifact")
        self.runtime_key = _json_copy(self.runtime_key, "runtime_key")
        if self.state != "ready":
            raise ModelRoutingConfigError("the configured v1 worker must be ready")
        if self.sub is None or self.engine is None:
            raise ModelRoutingConfigError("the ready model needs a substrate and engine")
        self.worker_identity = _json_copy(self.worker_identity, "worker_identity")
        if self.resolved_artifact.get("model_id") != self.model_id:
            raise ModelRoutingConfigError("resolved artifact uses another model ID")
        if self.resolved_artifact.get("format") != "gguf":
            raise ModelRoutingConfigError("resolved artifact is not GGUF")
        if (self.resolved_artifact.get("artifact_sha256")
                != self.runtime_key.get("gguf_artifact_sha256")):
            raise ModelRoutingConfigError("artifact digest does not match runtime key")
        if (self.worker_identity.get("runtime_key_sha256")
                != self.runtime_key.get("key_sha256")):
            raise ModelRoutingConfigError("worker identity uses another runtime key")

    def _identity_mismatch(self, message: str) -> None:
        self.failure_code = "worker_identity_mismatch"
        raise RuntimeError(message)

    def qualify_live_identity(self) -> None:
        """Refresh the cheap worker handshake before each model-bound operation."""
        health_fn = getattr(self.engine, "health", None)
        if not callable(health_fn):
            self._identity_mismatch("selected worker exposes no health handshake")
        try:
            health = health_fn()
        except Exception as exc:
            raise RuntimeError(f"selected worker health failed: {exc}") from exc
        if not isinstance(health, Mapping) or health.get("status") != "ok":
            self._identity_mismatch("selected worker did not report status=ok")
        compatible, reason = check_worker_protocol(health.get("protocol_version"))
        if not compatible:
            self._identity_mismatch(reason)
        opaque_generation = health.get("worker_generation_id")
        if not isinstance(opaque_generation, str) or not opaque_generation:
            self._identity_mismatch("selected worker omitted worker_generation_id")

        key = self.runtime_key
        if health.get("model_sha256") != key.get("gguf_artifact_sha256"):
            self._identity_mismatch("selected worker model digest does not match runtime key")
        if health.get("n_ctx") != key.get("context_size"):
            self._identity_mismatch("selected worker context size does not match runtime key")
        backend = key.get("backend")
        if backend in {"cpu", "cuda"} and health.get("device") != backend:
            self._identity_mismatch("selected worker device does not match runtime key")
        if ("engine_build" in health
                and health.get("engine_build") != key.get("engine_build")):
            self._identity_mismatch("selected worker engine build does not match runtime key")
        if ("template_fingerprint" in health
                and health.get("template_fingerprint") != key.get("template_fingerprint")):
            self._identity_mismatch("selected worker template fingerprint does not match runtime key")

        current = self.worker_identity or {}
        if current.get("worker_generation_id") != opaque_generation:
            generation = current.get("worker_generation")
            generation = (
                generation + 1
                if isinstance(generation, int) and not isinstance(generation, bool)
                else 1
            )
            current = {
                **current,
                "worker_id": opaque_generation,
                "worker_generation_id": opaque_generation,
                "worker_generation": generation,
            }
        current.update({
            "runtime_key_sha256": key["key_sha256"],
            "protocol_version": health["protocol_version"],
            "engine_build": key["engine_build"],
            "backend": key["backend"],
        })
        self.worker_identity = current


@dataclass(frozen=True)
class ModelSelection:
    model_id: str
    sub: object
    engine: object
    artifact: dict | None
    runtime_key: dict | None = None
    worker_identity: dict | None = None
    gate_release: Callable[[], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "runtime_key",
            deepcopy(self.runtime_key) if self.runtime_key is not None else None,
        )
        object.__setattr__(
            self, "worker_identity",
            deepcopy(self.worker_identity) if self.worker_identity is not None else None,
        )

    def apply(self, handler) -> None:
        handler._route_sub = self.sub
        handler._route_engine = self.engine
        handler._route_subname = "engine"
        handler._selected_model_id = self.model_id
        handler._model_routing_artifact = deepcopy(self.artifact) if self.artifact else None
        if self.gate_release is not None:
            handler._generation_gate_release = self.gate_release


class PreloadedModelRouter:
    """Strict selection for exactly one configured, already-loaded model."""

    def __init__(
        self,
        bindings: Iterable[PreloadedModelBinding],
        *,
        default_model_id: str,
        preload_model_ids: Iterable[str] = (),
        max_loaded_workers: int = 1,
        gate: WorkerGateRegistry | None = None,
        **unsupported,
    ) -> None:
        bindings = tuple(bindings)
        preload_model_ids = tuple(preload_model_ids)
        if len(bindings) != 1:
            raise ModelRoutingConfigError("v1 requires exactly one model binding")
        binding = bindings[0]
        if binding.model_id != default_model_id:
            raise ModelRoutingConfigError("the only binding must be the default model")
        if preload_model_ids and preload_model_ids != (default_model_id,):
            raise ModelRoutingConfigError("v1 preloads exactly the default model")
        if max_loaded_workers != 1:
            raise ModelRoutingConfigError("v1 supports exactly one loaded worker")
        if unsupported:
            raise ModelRoutingConfigError(
                "v1 does not support cold loading, eviction, or projection routing"
            )
        self.binding = binding
        self.default_model_id = default_model_id
        self.preload_model_ids = (default_model_id,)
        self.max_loaded_workers = 1
        self._gate = gate

    def model_ids(self) -> list[str]:
        return [self.default_model_id]

    def catalog(self) -> list[dict]:
        return [{
            "model_id": self.binding.model_id,
            "artifact_sha256": self.binding.resolved_artifact["artifact_sha256"],
            "state": "ready",
        }]

    def runtime_status(self) -> dict:
        identity = self.binding.worker_identity or {}
        return {
            "default_model_id": self.default_model_id,
            "preload_model_ids": [self.default_model_id],
            "max_loaded_models": 1,
            "configured_count": 1,
            "resident_count": 1,
            "models": [{
                "model_id": self.default_model_id,
                "state": "ready",
                "default": True,
                "preloaded": True,
                "runtime_key_sha256": self.binding.runtime_key["key_sha256"],
                "worker_generation": identity.get("worker_generation"),
                "worker_id": identity.get("worker_id"),
                "failure_code": self.binding.failure_code,
            }],
        }

    def control_pair(self) -> tuple[object, object]:
        return self.binding.sub, self.binding.engine

    def _base(self, requested_model, selection_source, surface, route) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": {"surface": surface, "route": route},
            "request": {
                "request_id": "route_" + secrets.token_hex(12),
                "requested_model": requested_model,
                "selection_source": selection_source,
            },
            "policy": {
                "configured_model_id": self.default_model_id,
                "omitted_model_policy": "configured_model",
                "unknown_model_policy": "error_no_fallback",
            },
        }

    def _receipt(self, requested_model, selection_source) -> dict:
        return {
            "requested_model": requested_model,
            "selection_source": selection_source,
            "resolved_model_id": self.binding.model_id,
            "resolved_artifact": deepcopy(self.binding.resolved_artifact),
            "runtime_key": deepcopy(self.binding.runtime_key),
            "worker_identity": deepcopy(self.binding.worker_identity),
        }

    def _error(self, base: dict, code: str, message: str) -> ModelRoutingError:
        status, retryable, phase = _ERRORS[code]
        artifact = {
            **base,
            "result": {
                "status": "error",
                "lifecycle_state": "failed" if code.startswith("worker_") else "ready",
                "error": {
                    "code": code,
                    "http_status": status,
                    "retryable": retryable,
                    "phase": phase,
                    "message": message,
                },
            },
        }
        schemas.validate(artifact, SCHEMA_VERSION)
        return ModelRoutingError(artifact)

    def _resolve(self, requested_model, *, field_present, surface, route):
        if field_present:
            literal = requested_model if isinstance(requested_model, str) else None
            base = self._base(literal, "explicit", surface, route)
            if (not isinstance(requested_model, str) or not requested_model
                    or requested_model != requested_model.strip()):
                raise self._error(
                    base, "invalid_model_selection",
                    "model must be a non-empty canonical model ID",
                )
            if requested_model != self.default_model_id:
                raise self._error(base, "unknown_model", f"unknown model {requested_model!r}")
            return requested_model, "explicit", base
        return None, "default", self._base(None, "default", surface, route)

    def _qualify(self, base: dict) -> None:
        try:
            self.binding.qualify_live_identity()
        except RuntimeError as exc:
            code = (
                "worker_identity_mismatch"
                if self.binding.failure_code == "worker_identity_mismatch"
                else "worker_failed"
            )
            raise self._error(base, code, str(exc)) from exc

    def select(
        self,
        requested_model,
        *,
        field_present: bool,
        surface: str,
        route: str,
        cancel_check: Callable[[], bool] | None = None,
    ) -> ModelSelection:
        requested_model, source, base = self._resolve(
            requested_model, field_present=field_present, surface=surface, route=route
        )
        self._qualify(base)
        gate_release = None
        if self._gate is not None:
            outcome, gate_release = self._gate.acquire_generation(
                self.default_model_id, cancel_check=cancel_check
            )
            if outcome is not None:
                code = _GATE_OUTCOME_CODES[outcome]
                message = (
                    "generation queue timed out waiting for its turn"
                    if outcome == "timeout"
                    else "generation queue is full"
                    if outcome == "full"
                    else "request was abandoned while queued"
                )
                raise self._error(base, code, message)
        artifact = {
            **base,
            "result": {
                "status": "routed",
                "lifecycle_state": "ready",
                "receipt": self._receipt(requested_model, source),
            },
        }
        schemas.validate(artifact, SCHEMA_VERSION)
        return ModelSelection(
            model_id=self.default_model_id,
            sub=self.binding.sub,
            engine=self.binding.engine,
            artifact=artifact,
            runtime_key=self.binding.runtime_key,
            worker_identity=self.binding.worker_identity,
            gate_release=gate_release,
        )

    def select_control_model(self, requested_model, *, route: str) -> ModelSelection:
        _requested, _source, base = self._resolve(
            requested_model, field_present=True, surface="native", route=route
        )
        self._qualify(base)
        return ModelSelection(
            model_id=self.default_model_id,
            sub=self.binding.sub,
            engine=self.binding.engine,
            artifact=None,
            runtime_key=self.binding.runtime_key,
            worker_identity=self.binding.worker_identity,
        )


def clear_handler_selection(handler) -> None:
    release = getattr(handler, "_generation_gate_release", None)
    if release is not None:
        release()
    for name in (
        "_route_sub", "_route_engine", "_route_subname", "_selected_model_id",
        "_model_routing_artifact", "_generation_gate_release",
    ):
        if hasattr(handler, name):
            delattr(handler, name)


def _emit_error(handler, error: ModelRoutingError, surface: str) -> None:
    artifact = deepcopy(error.artifact)
    common = artifact["result"]["error"]
    if surface == "openai":
        handler._json(error.http_status, {
            "error": {
                "message": common["message"],
                "type": "model_routing_error",
                "param": "model",
                "code": common["code"],
                "retryable": common["retryable"],
                "phase": common["phase"],
            },
            "clozn_model_routing": artifact,
        })
    else:
        handler._json(error.http_status, {
            "error": deepcopy(common),
            "clozn_model_routing": artifact,
        })


def select_for_handler(handler, body: Mapping, *, surface: str, route: str):
    """Select the configured worker and attach it to the request handler."""
    from clozn.server import app as ctx
    from clozn.server.http_policy import client_gone

    clear_handler_selection(handler)
    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None:
        handler._json(503, {"error": {
            "message": "model router unavailable",
            "type": "service_unavailable",
            "param": "model",
            "code": "model_router_unavailable",
        }})
        return None
    started_ns = time.monotonic_ns()
    try:
        selection = router.select(
            body.get("model"),
            field_present="model" in body,
            surface=surface,
            route=route,
            cancel_check=lambda: client_gone(handler),
        )
    except ModelRoutingError as error:
        recorder = getattr(handler, "_record_gateway_phase", None)
        if callable(recorder):
            recorder("gateway_queue", time.monotonic_ns() - started_ns,
                     aggregation="exclusive")
        _emit_error(handler, error, surface)
        return None
    recorder = getattr(handler, "_record_gateway_phase", None)
    if callable(recorder):
        recorder("gateway_queue", time.monotonic_ns() - started_ns,
                 aggregation="exclusive")
    selection.apply(handler)
    return selection


def select_control_model_for_run(handler, model, *, route: str):
    """Resolve the worker recorded on an immutable run."""
    from clozn.server import app as ctx

    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None:
        handler._json(503, {
            "error": "model router unavailable", "code": "model_router_unavailable",
        })
        return None
    try:
        return router.select_control_model(model, route=route)
    except ModelRoutingError as error:
        _emit_error(handler, error, "native")
        return None


def select_control_model_for_request(
    handler, requested_model, *, field_present: bool, route: str
):
    """Resolve cancellation and other non-generation control requests."""
    from clozn.server import app as ctx

    clear_handler_selection(handler)
    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None:
        handler._json(503, {
            "error": "model router unavailable", "code": "model_router_unavailable",
        })
        return None
    model = requested_model if field_present else getattr(router, "default_model_id", None)
    try:
        selection = router.select_control_model(model, route=route)
    except ModelRoutingError as error:
        _emit_error(handler, error, "native")
        return None
    selection.apply(handler)
    return selection


__all__ = [
    "ModelRoutingConfigError", "ModelRoutingError", "ModelSelection",
    "PreloadedModelBinding", "PreloadedModelRouter", "clear_handler_selection",
    "select_control_model_for_request", "select_control_model_for_run",
    "select_for_handler",
]
