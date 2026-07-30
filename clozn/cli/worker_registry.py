"""Model-free registry for configured, preloaded private workers.

This module is the process-lifecycle half of ADR 004.  It deliberately does
not select a worker for a request, discover models, cold-load on demand, evict,
or admit generations.  It gives the future router one exact, immutable key for
every configured worker and keeps failures isolated between preloaded handles.

The current ``clozn serve MODEL`` path remains unchanged.  It is the
one-definition/one-preload compatibility case that a later supervisor wiring
ticket can express through this registry once the gateway can consume more than
one private worker.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Callable, Iterable, Mapping

from clozn.cli.engine_process import _free_port, _health, spawn_engine
from clozn.cli.worker_handle import WorkerHandle, WorkerRestartLimitError
from clozn.protocol import check_worker_protocol


_HEX = frozenset("0123456789abcdef")
_LIFECYCLE_STATES = frozenset({"unloaded", "loading", "ready", "evicting", "failed"})
_WHITE_BOX_CAPABILITY_FLAGS = ("sae", "jlens", "attn_knockout")
_NON_WORKER_FLAGS = frozenset({"ctx", "adapter", "adapter_scale", "chat", "tmpl"})
_VALUE_BEARING_WORKER_FLAGS = frozenset({"mask", "eos", "sae", "sae_k", "jlens"})


class WorkerRegistryConfigError(ValueError):
    """A registry definition is ambiguous or cannot satisfy ADR 004."""


class UnknownWorkerModelError(KeyError):
    """A caller named no configured canonical model ID.

    There is intentionally no alias/default recovery here.  Request routing is
    a later layer and must resolve aliases exactly once before registry lookup.
    """


class WorkerIdentityMismatchError(RuntimeError):
    """A live worker's handshake does not match its immutable runtime key."""


def _require_string(value, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkerRegistryConfigError(f"{field_name} must be a non-empty string")
    return value


def _require_sha256(value, field_name: str) -> str:
    value = _require_string(value, field_name)
    if len(value) != 64 or any(character not in _HEX for character in value):
        raise WorkerRegistryConfigError(
            f"{field_name} must be a lowercase 64-character SHA-256"
        )
    return value


@dataclass(frozen=True)
class AdapterRuntimeIdentity:
    """The closed adapter facet from ``clozn.model-routing.v1``."""

    present: bool
    identity_sha256: str | None = None
    artifact_sha256: str | None = None
    scale: float | None = None

    def __post_init__(self) -> None:
        if type(self.present) is not bool:
            raise WorkerRegistryConfigError("adapter.present must be a boolean")
        if not self.present:
            if any(value is not None for value in (
                self.identity_sha256, self.artifact_sha256, self.scale
            )):
                raise WorkerRegistryConfigError(
                    "an absent adapter must have null identity, artifact, and scale"
                )
            return
        _require_sha256(self.identity_sha256, "adapter.identity_sha256")
        _require_sha256(self.artifact_sha256, "adapter.artifact_sha256")
        if (isinstance(self.scale, bool)
                or not isinstance(self.scale, (int, float))
                or not math.isfinite(float(self.scale))):
            raise WorkerRegistryConfigError(
                "a present adapter scale must be a finite number"
            )
        object.__setattr__(self, "scale", float(self.scale))

    @classmethod
    def absent(cls) -> "AdapterRuntimeIdentity":
        return cls(present=False)

    def as_dict(self) -> dict:
        return {
            "present": self.present,
            "identity_sha256": self.identity_sha256,
            "artifact_sha256": self.artifact_sha256,
            "scale": self.scale,
        }


@dataclass(frozen=True)
class RuntimeKey:
    """All behavior-bearing facets of one private worker, canonically hashed."""

    gguf_artifact_sha256: str
    context_size: int
    backend: str
    adapter: AdapterRuntimeIdentity
    template_fingerprint: str
    engine_build: str
    white_box_flags: Mapping[str, bool] = field(default_factory=dict)
    key_sha256: str = field(init=False)

    def __post_init__(self) -> None:
        _require_sha256(self.gguf_artifact_sha256, "gguf_artifact_sha256")
        if (isinstance(self.context_size, bool)
                or not isinstance(self.context_size, int)
                or self.context_size < 1):
            raise WorkerRegistryConfigError("context_size must be a positive integer")
        _require_string(self.backend, "backend")
        if not isinstance(self.adapter, AdapterRuntimeIdentity):
            raise WorkerRegistryConfigError(
                "adapter must be an AdapterRuntimeIdentity"
            )
        fingerprint = _require_string(
            self.template_fingerprint, "template_fingerprint"
        )
        if (not 16 <= len(fingerprint) <= 64
                or any(character not in _HEX for character in fingerprint)):
            raise WorkerRegistryConfigError(
                "template_fingerprint must be 16-64 lowercase hexadecimal characters"
            )
        _require_string(self.engine_build, "engine_build")
        flags = dict(self.white_box_flags)
        for name, enabled in flags.items():
            _require_string(name, "white_box_flags key")
            if type(enabled) is not bool:
                raise WorkerRegistryConfigError(
                    f"white_box_flags[{name!r}] must be a boolean"
                )
        object.__setattr__(
            self,
            "white_box_flags",
            MappingProxyType(dict(sorted(flags.items()))),
        )
        encoded = json.dumps(
            self.facets(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "key_sha256", hashlib.sha256(encoded).hexdigest())

    def facets(self) -> dict:
        """The canonical object whose digest identifies this runtime."""
        return {
            "gguf_artifact_sha256": self.gguf_artifact_sha256,
            "context_size": self.context_size,
            "backend": self.backend,
            "adapter": self.adapter.as_dict(),
            "template_fingerprint": self.template_fingerprint,
            "engine_build": self.engine_build,
            "white_box_flags": dict(self.white_box_flags),
        }

    def as_dict(self) -> dict:
        return {"key_sha256": self.key_sha256, **self.facets()}


@dataclass(frozen=True)
class WorkerDefinition:
    """One canonical configured model and the worker it must launch."""

    model_id: str
    model: str
    runtime_key: RuntimeKey
    flags: Mapping[str, object] = field(default_factory=dict)
    prefer_gpu: bool = True
    port: int | None = None
    boot_timeout: float = 180.0
    restart_limit: int = 3
    restart_window: float = 60.0
    log: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        _require_string(self.model_id, "model_id")
        _require_string(self.model, "model")
        if not isinstance(self.runtime_key, RuntimeKey):
            raise WorkerRegistryConfigError("runtime_key must be a RuntimeKey")
        if type(self.prefer_gpu) is not bool:
            raise WorkerRegistryConfigError("prefer_gpu must be a boolean")
        if self.port is not None and (
                isinstance(self.port, bool)
                or not isinstance(self.port, int)
                or not 1 <= self.port <= 65535):
            raise WorkerRegistryConfigError("port must be between 1 and 65535")
        if self.boot_timeout <= 0:
            raise WorkerRegistryConfigError("boot_timeout must be positive")
        if (isinstance(self.restart_limit, bool)
                or not isinstance(self.restart_limit, int)
                or self.restart_limit < 1):
            raise WorkerRegistryConfigError("restart_limit must be a positive integer")
        if self.restart_window <= 0:
            raise WorkerRegistryConfigError("restart_window must be positive")

        flags = dict(self.flags)
        launch_context = flags.get("ctx", 4096)
        if launch_context != self.runtime_key.context_size:
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} launch context {launch_context!r} does not "
                f"match runtime key context_size {self.runtime_key.context_size}"
            )
        launch_adapter_present = bool(flags.get("adapter"))
        if launch_adapter_present != self.runtime_key.adapter.present:
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} launch adapter presence does not match runtime key"
            )
        if launch_adapter_present:
            launch_scale = flags.get("adapter_scale", 1.0)
            if (isinstance(launch_scale, bool)
                    or not isinstance(launch_scale, (int, float))
                    or not math.isfinite(float(launch_scale))
                    or float(launch_scale) != self.runtime_key.adapter.scale):
                raise WorkerRegistryConfigError(
                    f"{self.model_id!r} launch adapter scale does not match runtime key"
                )

        # ADR 004 v1 has a boolean white-box map, not a place for an SAE/J-lens
        # artifact identity, diffusion token ID, or arbitrary argv.  Fail closed
        # instead of hashing a friendly boolean while launching a different
        # value-bearing behavior.  --no-flash-attn is the sole current worker
        # toggle representable exactly as a boolean.
        unsupported = sorted(set(flags) - _NON_WORKER_FLAGS - {"extra_args"})
        value_bearing = sorted(set(unsupported) & _VALUE_BEARING_WORKER_FLAGS)
        if value_bearing:
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} launch flag {value_bearing[0]!r} is "
                "value-bearing and has no exact ADR 004 v1 runtime-key facet"
            )
        if unsupported:
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} launch flag {unsupported[0]!r} is not "
                "covered by the exact ADR 004 v1 runtime key"
            )

        extra_args = flags.get("extra_args", ())
        if extra_args is None:
            extra_args = ()
        if (not isinstance(extra_args, (list, tuple))
                or any(not isinstance(value, str) for value in extra_args)):
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} extra_args must be a list of strings"
            )
        extra_args = tuple(extra_args)
        if extra_args not in ((), ("--no-flash-attn",)):
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} extra_args contains behavior not covered "
                "by the exact ADR 004 v1 runtime key"
            )
        if "extra_args" in flags:
            # MappingProxyType only freezes the outer mapping.  Own an
            # immutable copy so mutating the caller's list cannot change the
            # worker argv after its runtime key has been accepted.
            flags["extra_args"] = extra_args
        expected_white_box_flags = {
            "sae": False,
            "jlens": False,
            "attn_knockout": extra_args == ("--no-flash-attn",),
        }
        if dict(self.runtime_key.white_box_flags) != expected_white_box_flags:
            raise WorkerRegistryConfigError(
                f"{self.model_id!r} white_box_flags must exactly match the "
                f"current launch profile {expected_white_box_flags!r}"
            )
        object.__setattr__(self, "flags", MappingProxyType(flags))


@dataclass
class _WorkerEntry:
    definition: WorkerDefinition
    state: str = "unloaded"
    handle: WorkerHandle | None = None
    port: int | None = None
    worker_generation: int = 0
    worker_identity: dict | None = None
    failure_code: str | None = None
    last_error: str | None = None


SpawnWorker = Callable[..., tuple[object, dict, bool]]
HealthProbe = Callable[[int], dict | None]
TemplateProbe = Callable[[int], str]


class WorkerRegistry:
    """Own independent preloaded workers keyed by exact runtime identity."""

    def __init__(
        self,
        definitions: Iterable[WorkerDefinition],
        *,
        default_model_id: str,
        preload_model_ids: Iterable[str],
        max_loaded_workers: int | None = None,
        spawn: SpawnWorker = spawn_engine,
        health_probe: HealthProbe = _health,
        template_probe: TemplateProbe | None = None,
        port_factory: Callable[[], int] = _free_port,
    ) -> None:
        definitions = tuple(definitions)
        preload_model_ids = tuple(preload_model_ids)
        if not definitions:
            raise WorkerRegistryConfigError(
                "at least one worker definition is required"
            )
        if not isinstance(default_model_id, str) or not default_model_id:
            raise WorkerRegistryConfigError(
                "default_model_id must be a configured canonical model ID"
            )
        if len(set(preload_model_ids)) != len(preload_model_ids):
            raise WorkerRegistryConfigError("preload_model_ids contains a duplicate")
        if max_loaded_workers is None:
            max_loaded_workers = len(preload_model_ids)
        if (isinstance(max_loaded_workers, bool)
                or not isinstance(max_loaded_workers, int)
                or max_loaded_workers < 1):
            raise WorkerRegistryConfigError(
                "max_loaded_workers must be a positive integer"
            )
        if len(preload_model_ids) > max_loaded_workers:
            raise WorkerRegistryConfigError(
                "preload_model_ids exceeds max_loaded_workers"
            )

        by_id: dict[str, _WorkerEntry] = {}
        by_key: dict[str, _WorkerEntry] = {}
        explicit_ports: set[int] = set()
        for definition in definitions:
            if not isinstance(definition, WorkerDefinition):
                raise WorkerRegistryConfigError(
                    "definitions must contain WorkerDefinition values"
                )
            if definition.model_id in by_id:
                raise WorkerRegistryConfigError(
                    f"duplicate canonical model ID {definition.model_id!r}"
                )
            key = definition.runtime_key.key_sha256
            if key in by_key:
                raise WorkerRegistryConfigError(
                    f"duplicate runtime key {key}; use one canonical model ID "
                    "rather than alias-like duplicate definitions"
                )
            if definition.port is not None:
                if definition.port in explicit_ports:
                    raise WorkerRegistryConfigError(
                        f"duplicate private worker port {definition.port}"
                    )
                explicit_ports.add(definition.port)
            entry = _WorkerEntry(definition=definition, port=definition.port)
            by_id[definition.model_id] = entry
            by_key[key] = entry

        if default_model_id not in by_id:
            raise WorkerRegistryConfigError(
                f"default model {default_model_id!r} is not configured"
            )
        unknown_preloads = [
            model_id for model_id in preload_model_ids if model_id not in by_id
        ]
        if unknown_preloads:
            raise WorkerRegistryConfigError(
                f"preload model {unknown_preloads[0]!r} is not configured"
            )
        if default_model_id not in preload_model_ids:
            raise WorkerRegistryConfigError(
                "the default model must be preloaded until load-on-demand exists"
            )

        self.default_model_id = default_model_id
        self.preload_model_ids = preload_model_ids
        self.max_loaded_workers = max_loaded_workers
        self._by_id = by_id
        self._by_key = by_key
        self._spawn = spawn
        self._health_probe = health_probe
        self._template_probe = template_probe
        self._port_factory = port_factory

    def _entry(self, model_id: str) -> _WorkerEntry:
        try:
            return self._by_id[model_id]
        except (KeyError, TypeError):
            raise UnknownWorkerModelError(model_id) from None

    def by_runtime_key(self, key_sha256: str) -> WorkerDefinition:
        """Return the exact definition; never fall back to the default."""
        try:
            return self._by_key[key_sha256].definition
        except (KeyError, TypeError):
            raise UnknownWorkerModelError(key_sha256) from None

    def definition(self, model_id: str) -> WorkerDefinition:
        """Exact canonical-ID lookup; aliases are a later routing concern."""
        return self._entry(model_id).definition

    def worker_handle(self, model_id: str) -> WorkerHandle | None:
        """The owned live handle for supervisor integration, if resident."""
        return self._entry(model_id).handle

    def _allocate_port(self, entry: _WorkerEntry) -> int:
        if entry.port is not None:
            return entry.port
        occupied = {
            other.port for other in self._by_id.values()
            if other is not entry and other.port is not None
        }
        port = self._port_factory()
        if (isinstance(port, bool)
                or not isinstance(port, int)
                or not 1 <= port <= 65535
                or port in occupied):
            raise WorkerRegistryConfigError(
                f"port factory returned an invalid or duplicate private port {port!r}"
            )
        entry.port = port
        return port

    @staticmethod
    def _backend_expects_gpu(backend: str) -> bool | None:
        normalized = backend.strip().lower()
        if normalized == "cpu":
            return False
        if normalized in {"gpu", "cuda", "metal"}:
            return True
        return None

    def _qualify_handshake(
        self,
        entry: _WorkerEntry,
        handle: WorkerHandle,
        *,
        generation: int,
    ) -> dict:
        health = handle.health
        if not isinstance(health, dict) or health.get("status") != "ok":
            raise WorkerIdentityMismatchError(
                "worker handshake did not report status=ok"
            )
        protocol_version = health.get("protocol_version")
        compatible, reason = check_worker_protocol(protocol_version)
        if not compatible:
            raise WorkerIdentityMismatchError(reason)
        worker_id = health.get("worker_generation_id")
        if not isinstance(worker_id, str) or not worker_id:
            raise WorkerIdentityMismatchError(
                "worker handshake omitted worker_generation_id"
            )

        key = entry.definition.runtime_key
        capabilities = health.get("capabilities")
        if not isinstance(capabilities, dict):
            raise WorkerIdentityMismatchError(
                "worker handshake omitted capabilities"
            )
        for capability in _WHITE_BOX_CAPABILITY_FLAGS:
            observed = capabilities.get(capability)
            expected = key.white_box_flags[capability]
            if type(observed) is not bool or observed != expected:
                raise WorkerIdentityMismatchError(
                    f"worker capability {capability!r} does not match the "
                    "configured runtime key"
                )
        if health.get("model_sha256") != key.gguf_artifact_sha256:
            raise WorkerIdentityMismatchError(
                "worker model_sha256 does not match the configured runtime key"
            )
        if health.get("n_ctx") != key.context_size:
            raise WorkerIdentityMismatchError(
                "worker n_ctx does not match the configured runtime key"
            )

        expected_gpu = self._backend_expects_gpu(key.backend)
        if expected_gpu is not None and bool(handle.gpu) != expected_gpu:
            raise WorkerIdentityMismatchError(
                "worker CPU/GPU residency does not match the configured backend"
            )
        if key.backend in {"cpu", "cuda"} and health.get("device") != key.backend:
            raise WorkerIdentityMismatchError(
                "worker device does not match the configured backend"
            )

        lora = health.get("lora")
        if key.adapter.present:
            if not isinstance(lora, dict):
                raise WorkerIdentityMismatchError(
                    "worker handshake omitted the configured adapter"
                )
            scale = lora.get("scale")
            if (isinstance(scale, bool)
                    or not isinstance(scale, (int, float))
                    or float(scale) != key.adapter.scale):
                raise WorkerIdentityMismatchError(
                    "worker adapter scale does not match the configured runtime key"
                )
        elif lora is not None:
            raise WorkerIdentityMismatchError(
                "worker attached an adapter absent from the configured runtime key"
            )

        # Newer workers may announce these exact facets.  Current protocol 1.1
        # workers do not, so absence is honest-unavailable rather than guessed.
        if ("engine_build" in health
                and health.get("engine_build") != key.engine_build):
            raise WorkerIdentityMismatchError(
                "worker engine_build does not match the configured runtime key"
            )
        if ("template_fingerprint" in health
                and health.get("template_fingerprint") != key.template_fingerprint):
            raise WorkerIdentityMismatchError(
                "worker template_fingerprint does not match the configured runtime key"
            )
        if self._template_probe is not None:
            try:
                observed_template = self._template_probe(handle.port)
            except Exception as error:
                raise WorkerIdentityMismatchError(
                    f"worker template fingerprint probe failed: {error}"
                ) from None
            if observed_template != key.template_fingerprint:
                raise WorkerIdentityMismatchError(
                    "worker canonical template rendering does not match the "
                    "configured runtime key"
                )

        return {
            "worker_id": worker_id,
            # Protocol 1.1 checkpoint/fork references use this opaque value
            # verbatim.  Keep it explicit rather than asking consumers to
            # infer that ADR 004's worker_id currently has the same source.
            "worker_generation_id": worker_id,
            "worker_generation": generation,
            "runtime_key_sha256": key.key_sha256,
            "protocol_version": protocol_version,
            "engine_build": key.engine_build,
            "backend": key.backend,
        }

    @staticmethod
    def _failed(entry: _WorkerEntry, code: str, error: BaseException) -> None:
        entry.state = "failed"
        entry.failure_code = code
        entry.last_error = str(error)
        entry.worker_identity = None

    def _start_entry(self, entry: _WorkerEntry) -> bool:
        entry.state = "loading"
        entry.failure_code = None
        entry.last_error = None
        handle = None
        try:
            port = self._allocate_port(entry)
            definition = entry.definition
            handle = WorkerHandle.start(
                model=definition.model,
                port=port,
                flags=definition.flags,
                prefer_gpu=definition.prefer_gpu,
                log=definition.log,
                boot_timeout=definition.boot_timeout,
                restart_limit=definition.restart_limit,
                restart_window=definition.restart_window,
                spawn=lambda *args, **kwargs: self._spawn(
                    *args,
                    model_id=definition.model_id,
                    **kwargs,
                ),
            )
            next_generation = entry.worker_generation + 1
            # A process that handshakes with the wrong identity is still a
            # distinct process generation.  Consume the ordinal before
            # qualification so later recovery can never reuse it.
            entry.worker_generation = next_generation
            identity = self._qualify_handshake(
                entry, handle, generation=next_generation
            )
        except Exception as error:
            if handle is not None:
                handle.stop()
            entry.handle = None
            code = (
                "worker_identity_mismatch"
                if isinstance(error, WorkerIdentityMismatchError)
                else "model_load_failed"
            )
            self._failed(entry, code, error)
            return False
        entry.handle = handle
        entry.worker_identity = identity
        entry.state = "ready"
        return True

    def start_preloaded(self) -> dict:
        """Start every preload independently and return a status projection.

        One failed worker never tears down a ready sibling.  Callers decide
        whether their service policy can proceed with a partial preload set.
        """
        for model_id in self.preload_model_ids:
            entry = self._entry(model_id)
            if entry.state != "ready":
                self._start_entry(entry)
        return self.status()

    def recover_failed(self, model_id: str) -> bool:
        """Retry one failed preload; no other worker is touched."""
        entry = self._entry(model_id)
        if model_id not in self.preload_model_ids:
            raise WorkerRegistryConfigError(
                f"{model_id!r} is configured but not preloaded"
            )
        if entry.state != "failed":
            raise WorkerRegistryConfigError(
                f"{model_id!r} is {entry.state}, not failed"
            )
        return self._start_entry(entry)

    def _restart_entry(self, entry: _WorkerEntry) -> bool:
        handle = entry.handle
        if handle is None:
            return self._start_entry(entry)
        entry.state = "loading"
        try:
            handle.restart()
            next_generation = entry.worker_generation + 1
            entry.worker_generation = next_generation
            identity = self._qualify_handshake(
                entry, handle, generation=next_generation
            )
        except Exception as error:
            code = (
                "worker_identity_mismatch"
                if isinstance(error, WorkerIdentityMismatchError)
                else "worker_restart_limit"
                if isinstance(error, WorkerRestartLimitError)
                else "worker_restart_failed"
            )
            handle.stop()
            entry.handle = None
            self._failed(entry, code, error)
            return False
        entry.worker_identity = identity
        entry.state = "ready"
        entry.failure_code = None
        entry.last_error = None
        return True

    def maintain(self) -> dict:
        """Restart every unexpectedly-dead ready worker independently."""
        for entry in self._by_id.values():
            handle = entry.handle
            if (entry.state == "ready"
                    and handle is not None
                    and handle.process.poll() is not None
                    and not handle.stopping):
                self._restart_entry(entry)
        return self.status()

    def refresh_health(self, model_id: str) -> bool:
        """Probe and re-qualify one worker without touching its siblings."""
        entry = self._entry(model_id)
        handle = entry.handle
        if entry.state != "ready" or handle is None or entry.port is None:
            return False
        if handle.process.poll() is not None:
            entry.handle = None
            self._failed(
                entry,
                "worker_exited",
                RuntimeError("worker process exited before health probe"),
            )
            return False
        try:
            health = self._health_probe(entry.port)
            if health is None:
                raise RuntimeError("worker health probe returned no document")
            handle.health = health
            entry.worker_identity = self._qualify_handshake(
                entry, handle, generation=entry.worker_generation
            )
            return True
        except Exception as error:
            handle.stop()
            entry.handle = None
            code = (
                "worker_identity_mismatch"
                if isinstance(error, WorkerIdentityMismatchError)
                else "worker_health_failed"
            )
            self._failed(entry, code, error)
            return False

    def stop(self, model_id: str) -> None:
        """Stop one configured worker and return it to ``unloaded``."""
        entry = self._entry(model_id)
        entry.state = "evicting"
        if entry.handle is not None:
            entry.handle.stop()
        entry.handle = None
        entry.worker_identity = None
        entry.failure_code = None
        entry.last_error = None
        entry.state = "unloaded"

    def stop_all(self) -> None:
        for model_id in self._by_id:
            self.stop(model_id)

    def status(self) -> dict:
        """Privacy-safe, deterministic projection for a later runtime endpoint."""
        workers = []
        for model_id in sorted(self._by_id):
            entry = self._by_id[model_id]
            if entry.state not in _LIFECYCLE_STATES:  # internal invariant, fail closed
                raise RuntimeError(f"unknown internal worker state {entry.state!r}")
            handle = entry.handle
            workers.append({
                "model_id": model_id,
                "runtime_key_sha256": entry.definition.runtime_key.key_sha256,
                "state": entry.state,
                "default": model_id == self.default_model_id,
                "preloaded": model_id in self.preload_model_ids,
                "worker_port": entry.port,
                "worker_pid": handle.process.pid if handle is not None else None,
                "worker_alive": (
                    handle is not None and handle.process.poll() is None
                ),
                "worker_identity": (
                    dict(entry.worker_identity)
                    if entry.worker_identity is not None else None
                ),
                "failure_code": entry.failure_code,
            })
        return {
            "default_model_id": self.default_model_id,
            "max_loaded_workers": self.max_loaded_workers,
            "preload_model_ids": list(self.preload_model_ids),
            "workers": workers,
        }

    def routing_projection(self) -> dict:
        """Exact private binding document for the gateway routing process.

        Unlike :meth:`status`, this supervisor-to-gateway projection includes
        the complete immutable runtime key.  It still omits local artifact
        paths and failure text.  A future process transport may serialize this
        document unchanged; RT-03 also consumes it directly in model-free
        tests.  It is a point-in-time lifecycle snapshot, not a discovery or
        load command.
        """
        status_by_id = {
            worker["model_id"]: worker for worker in self.status()["workers"]
        }
        models = []
        for model_id in sorted(self._by_id):
            entry = self._by_id[model_id]
            definition = entry.definition
            status = status_by_id[model_id]
            key = definition.runtime_key
            models.append({
                "model_id": model_id,
                "resolved_artifact": {
                    "model_id": model_id,
                    "format": "gguf",
                    "artifact_sha256": key.gguf_artifact_sha256,
                },
                "runtime_key": key.as_dict(),
                "adapter": key.adapter.as_dict(),
                "state": status["state"],
                "worker_port": status["worker_port"],
                "worker_identity": status["worker_identity"],
                "failure_code": status["failure_code"],
                "preloaded": status["preloaded"],
            })
        return {
            "default_model_id": self.default_model_id,
            "max_loaded_workers": self.max_loaded_workers,
            "preload_model_ids": list(self.preload_model_ids),
            "models": models,
        }
