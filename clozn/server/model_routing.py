"""Common preloaded-worker selection for native, OpenAI, and Ollama routes.

RT-03 stops at routing among already-configured, already-preloaded workers.
There is no alias expansion, loading, eviction, queueing, or concurrency policy
here.  Every successful selection produces ``clozn.model-routing.v1`` evidence
before generation starts; every refusal uses the same typed error and is shaped
only at the outer protocol adapter.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import os
import secrets
import threading
from typing import Callable, Iterable, Mapping

from clozn import schemas
from clozn.protocol import check_worker_protocol


SCHEMA_VERSION = "clozn.model-routing.v1"
_LIFECYCLE = frozenset({"unloaded", "loading", "ready", "evicting", "failed"})
_ERRORS = {
    "invalid_model_selection": (400, False, "selection"),
    "unknown_model": (404, False, "resolution"),
    "model_not_ready": (409, True, "resolution"),
    "model_load_failed": (503, True, "load"),
    "worker_failed": (502, True, "generation"),
    "worker_identity_mismatch": (502, False, "handshake"),
}
_WHITE_BOX_CAPABILITIES = ("sae", "jlens", "attn_knockout")


class ModelRoutingConfigError(ValueError):
    """The gateway's preloaded binding document is unsafe or ambiguous."""


class ModelRoutingError(RuntimeError):
    """One schema-governed selection refusal."""

    def __init__(self, artifact: dict):
        result = artifact["result"]
        error = result["error"]
        super().__init__(error["message"])
        self.artifact = artifact
        self.code = error["code"]
        self.http_status = error["http_status"]
        self.retryable = error["retryable"]
        self.phase = error["phase"]
        self.lifecycle_state = result["lifecycle_state"]


def _json_copy(value, field_name: str) -> dict:
    if not isinstance(value, Mapping):
        raise ModelRoutingConfigError(f"{field_name} must be an object")
    # Routing contract values are JSON-shaped.  deepcopy owns nested lists/maps
    # without accepting arbitrary serializer hooks or mutating caller state.
    return deepcopy(dict(value))


@dataclass
class PreloadedModelBinding:
    """One configured canonical model and its request-local dispatch objects."""

    model_id: str
    resolved_artifact: dict
    runtime_key: dict
    adapter: dict
    state: str
    worker_identity: dict | None
    sub: object | None
    engine: object | None
    preloaded: bool = True
    failure_code: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ModelRoutingConfigError("model_id must be a non-empty string")
        self.resolved_artifact = _json_copy(
            self.resolved_artifact, "resolved_artifact"
        )
        self.runtime_key = _json_copy(self.runtime_key, "runtime_key")
        self.adapter = _json_copy(self.adapter, "adapter")
        if self.state not in _LIFECYCLE:
            raise ModelRoutingConfigError(
                f"{self.model_id!r} has unknown lifecycle state {self.state!r}"
            )
        if type(self.preloaded) is not bool:
            raise ModelRoutingConfigError("preloaded must be a boolean")
        if self.worker_identity is not None:
            self.worker_identity = _json_copy(
                self.worker_identity, "worker_identity"
            )

        artifact = self.resolved_artifact
        key = self.runtime_key
        if artifact.get("model_id") != self.model_id:
            raise ModelRoutingConfigError(
                f"{self.model_id!r} resolved artifact uses another model ID"
            )
        if artifact.get("format") != "gguf":
            raise ModelRoutingConfigError(
                f"{self.model_id!r} resolved artifact is not GGUF"
            )
        if artifact.get("artifact_sha256") != key.get("gguf_artifact_sha256"):
            raise ModelRoutingConfigError(
                f"{self.model_id!r} artifact digest does not match runtime key"
            )
        if self.adapter != key.get("adapter"):
            raise ModelRoutingConfigError(
                f"{self.model_id!r} adapter receipt does not match runtime key"
            )
        if self.state == "ready":
            if self.sub is None or self.engine is None:
                raise ModelRoutingConfigError(
                    f"ready model {self.model_id!r} needs a substrate and engine"
                )
            if not isinstance(self.worker_identity, dict):
                raise ModelRoutingConfigError(
                    f"ready model {self.model_id!r} needs worker identity"
                )
            if (self.worker_identity.get("runtime_key_sha256")
                    != key.get("key_sha256")):
                raise ModelRoutingConfigError(
                    f"{self.model_id!r} worker identity uses another runtime key"
                )

    def _identity_mismatch(self, message: str) -> None:
        self.state = "failed"
        self.failure_code = "worker_identity_mismatch"
        raise RuntimeError(message)

    def qualify_live_identity(self) -> None:
        """Verify the selected ready process generation before dispatch.

        The supervisor projection is point-in-time.  A worker may restart on
        the same private port, so selection refreshes its cheap loopback health
        handshake and updates the numeric generation only when the opaque
        process identity changes.  No request can journal a stale checkpoint
        generation or evidence identity.
        """
        if self.state != "ready":
            return
        health_fn = getattr(self.engine, "health", None)
        if not callable(health_fn):
            self._identity_mismatch(
                "selected worker exposes no health handshake"
            )
        try:
            health = health_fn()
        except Exception as exc:
            # A point-in-time loopback failure is not identity evidence.  Keep
            # the supervisor-projected ready state retryable so the next
            # request can re-probe this same binding even when the projection
            # file itself has not changed.
            raise RuntimeError(f"selected worker health failed: {exc}") from exc
        if not isinstance(health, Mapping) or health.get("status") != "ok":
            self._identity_mismatch("selected worker did not report status=ok")
        compatible, reason = check_worker_protocol(health.get("protocol_version"))
        if not compatible:
            self._identity_mismatch(reason)
        opaque_generation = health.get("worker_generation_id")
        if not isinstance(opaque_generation, str) or not opaque_generation:
            self._identity_mismatch(
                "selected worker omitted worker_generation_id"
            )

        key = self.runtime_key
        if health.get("model_sha256") != key.get("gguf_artifact_sha256"):
            self._identity_mismatch(
                "selected worker model digest does not match runtime key"
            )
        if health.get("n_ctx") != key.get("context_size"):
            self._identity_mismatch(
                "selected worker context size does not match runtime key"
            )
        backend = key.get("backend")
        if backend in {"cpu", "cuda"} and health.get("device") != backend:
            self._identity_mismatch(
                "selected worker device does not match runtime key"
            )
        adapter = key.get("adapter")
        if not isinstance(adapter, Mapping):
            self._identity_mismatch("runtime key omitted adapter identity")
        lora = health.get("lora")
        if adapter.get("present") is True:
            if not isinstance(lora, Mapping):
                self._identity_mismatch(
                    "selected worker omitted configured adapter"
                )
            scale = lora.get("scale")
            expected_scale = adapter.get("scale")
            if (isinstance(scale, bool)
                    or not isinstance(scale, (int, float))
                    or float(scale) != expected_scale):
                self._identity_mismatch(
                    "selected worker adapter scale does not match runtime key"
                )
        elif adapter.get("present") is False:
            if lora is not None:
                self._identity_mismatch(
                    "selected worker attached an unconfigured adapter"
                )
        else:
            self._identity_mismatch("runtime key adapter presence is invalid")
        if ("engine_build" in health
                and health.get("engine_build") != key.get("engine_build")):
            self._identity_mismatch(
                "selected worker engine build does not match runtime key"
            )
        if ("template_fingerprint" in health
                and health.get("template_fingerprint")
                != key.get("template_fingerprint")):
            self._identity_mismatch(
                "selected worker template fingerprint does not match runtime key"
            )
        capabilities = health.get("capabilities")
        if not isinstance(capabilities, Mapping):
            self._identity_mismatch("selected worker omitted capabilities")
        white_box = key.get("white_box_flags")
        if not isinstance(white_box, Mapping):
            self._identity_mismatch("runtime key omitted white_box_flags")
        for name in _WHITE_BOX_CAPABILITIES:
            expected = white_box.get(name)
            observed = capabilities.get(name)
            if type(expected) is not bool or type(observed) is not bool or observed != expected:
                self._identity_mismatch(
                    f"selected worker capability {name!r} does not match runtime key"
                )

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

    def __post_init__(self) -> None:
        # A control-plane consumer needs these two exact facts even when no
        # generation-routing artifact is created.  Keep them on the selection,
        # not as mutable ad-hoc attributes on EngineSubstrate.
        object.__setattr__(
            self,
            "runtime_key",
            deepcopy(self.runtime_key) if self.runtime_key is not None else None,
        )
        object.__setattr__(
            self,
            "worker_identity",
            (
                deepcopy(self.worker_identity)
                if self.worker_identity is not None else None
            ),
        )

    def apply(self, handler) -> None:
        handler._route_sub = self.sub
        handler._route_engine = self.engine
        handler._route_subname = "engine"
        handler._selected_model_id = self.model_id
        handler._model_routing_artifact = (
            deepcopy(self.artifact) if self.artifact is not None else None
        )


class PreloadedModelRouter:
    """Exact canonical selection across a closed preloaded binding set."""

    def __init__(
        self,
        bindings: Iterable[PreloadedModelBinding],
        *,
        default_model_id: str,
        preload_model_ids: Iterable[str],
        max_loaded_workers: int,
        load_queue_limit: int = 1,
        generation_queue_limit: int = 1,
        load_timeout_ms: int = 180_000,
        queue_timeout_ms: int = 600_000,
    ) -> None:
        bindings = tuple(bindings)
        preload_model_ids = tuple(preload_model_ids)
        if not bindings:
            raise ModelRoutingConfigError("at least one binding is required")
        by_id = {}
        for binding in bindings:
            if not isinstance(binding, PreloadedModelBinding):
                raise ModelRoutingConfigError(
                    "bindings must contain PreloadedModelBinding values"
                )
            if binding.model_id in by_id:
                raise ModelRoutingConfigError(
                    f"duplicate canonical model ID {binding.model_id!r}"
                )
            by_id[binding.model_id] = binding
        if default_model_id not in by_id:
            raise ModelRoutingConfigError("default model is not configured")
        if len(set(preload_model_ids)) != len(preload_model_ids):
            raise ModelRoutingConfigError("preload model IDs contain a duplicate")
        if any(model_id not in by_id for model_id in preload_model_ids):
            raise ModelRoutingConfigError("preload model is not configured")
        if default_model_id not in preload_model_ids:
            raise ModelRoutingConfigError("default model must be preloaded")
        for name, value in (
            ("max_loaded_workers", max_loaded_workers),
            ("load_queue_limit", load_queue_limit),
            ("generation_queue_limit", generation_queue_limit),
            ("load_timeout_ms", load_timeout_ms),
            ("queue_timeout_ms", queue_timeout_ms),
        ):
            if (isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 1):
                raise ModelRoutingConfigError(f"{name} must be a positive integer")
        if len(preload_model_ids) > max_loaded_workers:
            raise ModelRoutingConfigError(
                "preloads exceed max_loaded_workers"
            )
        self._by_id = by_id
        self.default_model_id = default_model_id
        self.preload_model_ids = preload_model_ids
        self.max_loaded_workers = max_loaded_workers
        self._limits = {
            "load_queue_limit": load_queue_limit,
            "generation_queue_limit": generation_queue_limit,
            "load_timeout_ms": load_timeout_ms,
            "queue_timeout_ms": queue_timeout_ms,
        }

    @classmethod
    def from_projection(
        cls,
        projection: Mapping,
        *,
        engine_factory: Callable[[int], object],
        substrate_factory: Callable[[object], object],
    ) -> "PreloadedModelRouter":
        """Build gateway bindings from ``WorkerRegistry.routing_projection``."""
        if not isinstance(projection, Mapping):
            raise ModelRoutingConfigError("routing projection must be an object")
        bindings = []
        for raw in projection.get("models") or []:
            if not isinstance(raw, Mapping):
                raise ModelRoutingConfigError("routing model entry must be an object")
            state = raw.get("state")
            port = raw.get("worker_port")
            engine = sub = None
            if state == "ready":
                if (isinstance(port, bool)
                        or not isinstance(port, int)
                        or not 1 <= port <= 65535):
                    raise ModelRoutingConfigError(
                        f"ready model {raw.get('model_id')!r} has no private port"
                    )
                engine = engine_factory(port)
                sub = substrate_factory(engine)
            bindings.append(PreloadedModelBinding(
                model_id=raw.get("model_id"),
                resolved_artifact=raw.get("resolved_artifact"),
                runtime_key=raw.get("runtime_key"),
                adapter=raw.get("adapter"),
                state=state,
                worker_identity=raw.get("worker_identity"),
                sub=sub,
                engine=engine,
                preloaded=raw.get("preloaded"),
                failure_code=raw.get("failure_code"),
            ))
        return cls(
            bindings,
            default_model_id=projection.get("default_model_id"),
            preload_model_ids=projection.get("preload_model_ids") or [],
            max_loaded_workers=projection.get("max_loaded_workers"),
        )

    def model_ids(self) -> list[str]:
        return sorted(self._by_id)

    def catalog(self) -> list[dict]:
        return [{
            "model_id": model_id,
            "artifact_sha256": self._by_id[model_id].resolved_artifact[
                "artifact_sha256"
            ],
            "state": self._by_id[model_id].state,
        } for model_id in self.model_ids()]

    def runtime_status(self) -> dict:
        models = []
        for model_id in self.model_ids():
            binding = self._by_id[model_id]
            identity = binding.worker_identity or {}
            models.append({
                "model_id": model_id,
                "state": binding.state,
                "default": model_id == self.default_model_id,
                "preloaded": binding.preloaded,
                "runtime_key_sha256": binding.runtime_key["key_sha256"],
                "worker_generation": identity.get("worker_generation"),
                "worker_id": identity.get("worker_id"),
                "failure_code": binding.failure_code,
            })
        resident = sum(model["state"] == "ready" for model in models)
        return {
            "default_model_id": self.default_model_id,
            "preload_model_ids": list(self.preload_model_ids),
            "max_loaded_models": self.max_loaded_workers,
            "configured_count": len(models),
            "resident_count": resident,
            "models": models,
        }

    def control_pair(self) -> tuple[object | None, object | None]:
        """A private control-plane substrate/engine, never request routing.

        Prefer the configured default.  If its preload failed, another ready
        worker keeps read-only runtime inspection and explicitly-modelled
        requests usable; omitted-model generation still fails on the default.
        """
        ordered = [self.default_model_id] + [
            model_id for model_id in self.model_ids()
            if model_id != self.default_model_id
        ]
        for model_id in ordered:
            binding = self._by_id[model_id]
            if binding.state == "ready":
                return binding.sub, binding.engine
        return None, None

    def _policy(self) -> dict:
        return {
            "default_model_id": self.default_model_id,
            "max_loaded_workers": self.max_loaded_workers,
            "preload_model_ids": list(self.preload_model_ids),
            **self._limits,
            "eviction_policy": "lru_idle",
            "active_worker_eviction": "forbidden",
            "cold_load_coalescing": True,
            "cancellation": "request_scoped_release_permits",
            "omitted_model_policy": "configured_default",
            "unknown_model_policy": "error_no_fallback",
            "alias_policy": "mutable_config_immutable_receipt",
        }

    @staticmethod
    def _load_event(state: str, *, ready: bool) -> dict:
        return {
            "event_id": None,
            "kind": "not_required" if ready else "not_started",
            "outcome": "already_ready" if ready else "not_started",
            "state_before": state,
            "state_after": state,
            "coalesced": False,
            "wait_ms": 0,
        }

    def _base(
        self,
        *,
        requested_model: str | None,
        selection_source: str,
        surface: str,
        route: str,
    ) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "protocol": {"surface": surface, "route": route},
            "request": {
                "request_id": "route_" + secrets.token_hex(12),
                "requested_model": requested_model,
                "selection_source": selection_source,
                "load_policy": "wait",
            },
            "policy": self._policy(),
        }

    @staticmethod
    def _attempt_receipt(
        requested_model: str | None,
        selection_source: str,
        binding: PreloadedModelBinding | None,
        load_event: dict,
    ) -> dict:
        receipt = {
            "requested_model": requested_model,
            "selection_source": selection_source,
            "load_event": load_event,
        }
        if binding is not None:
            receipt.update({
                "resolved_model_id": binding.model_id,
                "resolved_artifact": deepcopy(binding.resolved_artifact),
                "runtime_key": deepcopy(binding.runtime_key),
                "adapter": deepcopy(binding.adapter),
            })
            if binding.worker_identity is not None:
                receipt["worker_identity"] = deepcopy(binding.worker_identity)
        return receipt

    def _error(
        self,
        base: dict,
        *,
        code: str,
        lifecycle_state: str,
        message: str,
        binding: PreloadedModelBinding | None = None,
    ) -> ModelRoutingError:
        status, retryable, phase = _ERRORS[code]
        request = base["request"]
        load_event = self._load_event(lifecycle_state, ready=False)
        artifact = {
            **base,
            "result": {
                "status": "error",
                "lifecycle_state": lifecycle_state,
                "receipt": self._attempt_receipt(
                    request["requested_model"],
                    request["selection_source"],
                    binding,
                    load_event,
                ),
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

    def select(
        self,
        requested_model,
        *,
        field_present: bool,
        surface: str,
        route: str,
    ) -> ModelSelection:
        if field_present:
            literal = requested_model if isinstance(requested_model, str) else None
            if (not isinstance(requested_model, str)
                    or not requested_model
                    or requested_model != requested_model.strip()):
                base = self._base(
                    requested_model=literal,
                    selection_source="explicit",
                    surface=surface,
                    route=route,
                )
                raise self._error(
                    base,
                    code="invalid_model_selection",
                    lifecycle_state="unloaded",
                    message="model must be a non-empty canonical model ID",
                )
            selection_source = "explicit"
            resolved_id = requested_model
        else:
            requested_model = None
            selection_source = "default"
            resolved_id = self.default_model_id
        base = self._base(
            requested_model=requested_model,
            selection_source=selection_source,
            surface=surface,
            route=route,
        )
        binding = self._by_id.get(resolved_id)
        if binding is None:
            raise self._error(
                base,
                code="unknown_model",
                lifecycle_state="unloaded",
                message=f"unknown model {resolved_id!r}",
            )
        if binding.state != "ready":
            code = (
                "worker_identity_mismatch"
                if binding.failure_code == "worker_identity_mismatch"
                else "model_load_failed"
                if binding.state == "failed"
                else "model_not_ready"
            )
            raise self._error(
                base,
                code=code,
                lifecycle_state=binding.state,
                message=(
                    f"model {resolved_id!r} is {binding.state}; "
                    "RT-03 dispatches ready preloaded workers only"
                ),
                binding=binding,
            )
        try:
            binding.qualify_live_identity()
        except RuntimeError as exc:
            code = (
                "worker_identity_mismatch"
                if binding.failure_code == "worker_identity_mismatch"
                else "worker_failed"
            )
            raise self._error(
                base,
                code=code,
                lifecycle_state="failed",
                message=str(exc),
                binding=binding,
            ) from exc

        load_event = self._load_event("ready", ready=True)
        receipt = self._attempt_receipt(
            requested_model, selection_source, binding, load_event
        )
        artifact = {
            **base,
            "result": {
                "status": "routed",
                "lifecycle_state": "ready",
                "receipt": receipt,
            },
        }
        schemas.validate(artifact, SCHEMA_VERSION)
        return ModelSelection(
            model_id=binding.model_id,
            sub=binding.sub,
            engine=binding.engine,
            artifact=artifact,
            runtime_key=binding.runtime_key,
            worker_identity=binding.worker_identity,
        )

    def select_control_model(
        self,
        requested_model,
        *,
        route: str,
    ) -> ModelSelection:
        """Select one exact ready worker for a model-bound control-plane operation.

        Unlike :meth:`select`, success does not create a generation-routing artifact: checkpoint
        preparation and exact-fork planning already have their own immutable identity receipts.
        Refusals still use ``ModelRoutingError`` and ``clozn.model-routing.v1`` so unknown, unready,
        failed, and drifted parent models fail with the same typed public contract as generation.
        ``route`` is the truthful normalized route template, never a borrowed compatibility route.
        """
        if (
            not isinstance(requested_model, str)
            or not requested_model
            or requested_model != requested_model.strip()
        ):
            literal = requested_model if isinstance(requested_model, str) else None
            base = self._base(
                requested_model=literal,
                selection_source="explicit",
                surface="native",
                route=route,
            )
            raise self._error(
                base,
                code="invalid_model_selection",
                lifecycle_state="unloaded",
                message="parent run model must be a non-empty canonical model ID",
            )
        base = self._base(
            requested_model=requested_model,
            selection_source="explicit",
            surface="native",
            route=route,
        )
        binding = self._by_id.get(requested_model)
        if binding is None:
            raise self._error(
                base,
                code="unknown_model",
                lifecycle_state="unloaded",
                message=f"unknown parent run model {requested_model!r}",
            )
        if binding.state != "ready":
            code = (
                "worker_identity_mismatch"
                if binding.failure_code == "worker_identity_mismatch"
                else "model_load_failed"
                if binding.state == "failed"
                else "model_not_ready"
            )
            raise self._error(
                base,
                code=code,
                lifecycle_state=binding.state,
                message=(
                    f"parent run model {requested_model!r} is {binding.state}; "
                    "exact-fork control operations require a ready preloaded worker"
                ),
                binding=binding,
            )
        try:
            binding.qualify_live_identity()
        except RuntimeError as exc:
            code = (
                "worker_identity_mismatch"
                if binding.failure_code == "worker_identity_mismatch"
                else "worker_failed"
            )
            raise self._error(
                base,
                code=code,
                lifecycle_state="failed",
                message=str(exc),
                binding=binding,
            ) from exc
        return ModelSelection(
            model_id=binding.model_id,
            sub=binding.sub,
            engine=binding.engine,
            artifact=None,
            runtime_key=binding.runtime_key,
            worker_identity=binding.worker_identity,
        )


class ProjectionFileRouter:
    """Atomically refreshed RT-03 router backed by a private supervisor file."""

    _MAX_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        path: str,
        *,
        engine_factory: Callable[[int], object],
        substrate_factory: Callable[[object], object],
    ) -> None:
        self.path = os.path.abspath(os.fspath(path))
        self._engine_factory = engine_factory
        self._substrate_factory = substrate_factory
        self._fingerprint: str | None = None
        self._router: PreloadedModelRouter | None = None
        self._lock = threading.Lock()
        self.refresh(force=True)

    def _read_projection(self) -> tuple[str, dict]:
        try:
            size = os.path.getsize(self.path)
            if size < 2 or size > self._MAX_BYTES:
                raise ModelRoutingConfigError(
                    "routing projection has an invalid byte size"
                )
            with open(self.path, "rb") as handle:
                raw = handle.read(self._MAX_BYTES + 1)
        except ModelRoutingConfigError:
            raise
        except Exception as error:
            raise ModelRoutingConfigError(
                f"routing projection is unavailable: {error}"
            ) from None
        if len(raw) > self._MAX_BYTES:
            raise ModelRoutingConfigError("routing projection exceeds size limit")
        try:
            projection = json.loads(raw)
        except Exception as error:
            raise ModelRoutingConfigError(
                f"routing projection is invalid JSON: {error}"
            ) from None
        if not isinstance(projection, Mapping):
            raise ModelRoutingConfigError(
                "routing projection must be an object"
            )
        return hashlib.sha256(raw).hexdigest(), dict(projection)

    def refresh(self, *, force: bool = False) -> bool:
        fingerprint, projection = self._read_projection()
        with self._lock:
            if not force and fingerprint == self._fingerprint:
                return False
            router = PreloadedModelRouter.from_projection(
                projection,
                engine_factory=self._engine_factory,
                substrate_factory=self._substrate_factory,
            )
            self._router = router
            self._fingerprint = fingerprint
            return True

    def _current(self) -> PreloadedModelRouter:
        self.refresh()
        with self._lock:
            if self._router is None:  # constructor/refresh invariant
                raise ModelRoutingConfigError(
                    "routing projection has not been initialized"
                )
            return self._router

    def select(self, *args, **kwargs) -> ModelSelection:
        return self._current().select(*args, **kwargs)

    def select_control_model(self, *args, **kwargs) -> ModelSelection:
        return self._current().select_control_model(*args, **kwargs)

    def model_ids(self) -> list[str]:
        return self._current().model_ids()

    def catalog(self) -> list[dict]:
        return self._current().catalog()

    def runtime_status(self) -> dict:
        return self._current().runtime_status()

    def control_pair(self) -> tuple[object | None, object | None]:
        return self._current().control_pair()


def clear_handler_selection(handler) -> None:
    for name in (
        "_route_sub", "_route_engine", "_route_subname",
        "_selected_model_id", "_model_routing_artifact",
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
    elif surface == "ollama":
        handler._json(error.http_status, {
            "error": common["message"],
            "code": common["code"],
            "retryable": common["retryable"],
            "phase": common["phase"],
            "clozn_model_routing": artifact,
        })
    else:
        handler._json(error.http_status, {
            "error": deepcopy(common),
            "clozn_model_routing": artifact,
        })


def select_for_handler(
    handler,
    body: Mapping,
    *,
    surface: str,
    route: str,
) -> ModelSelection | None:
    """Select and apply one binding, or serialize a protocol-specific refusal.

    When no router is configured this is a strict compatibility shim over the
    process's original single substrate.  That path deliberately creates no
    synthetic runtime receipt because the legacy supervisor did not provide an
    ADR 004 key; existing one-model clients keep their historical behavior.
    """
    from clozn.server import app as ctx

    clear_handler_selection(handler)
    router = getattr(ctx, "MODEL_ROUTER", None)
    if router is None:
        from clozn.server.generation_gateway import model_id
        requested = body.get("model") if "model" in body else None
        selected = str(requested or model_id())
        sub = ctx.active_sub(handler)
        engine = ctx.active_engine(handler)
        if sub is None or engine is None:
            return ModelSelection(
                model_id=selected, sub=sub, engine=engine, artifact=None
            )
        selection = ModelSelection(
            model_id=selected, sub=sub, engine=engine, artifact=None
        )
        selection.apply(handler)
        return selection
    try:
        selection = router.select(
            body.get("model"),
            field_present="model" in body,
            surface=surface,
            route=route,
        )
    except ModelRoutingError as error:
        _emit_error(handler, error, surface)
        return None
    selection.apply(handler)
    return selection
